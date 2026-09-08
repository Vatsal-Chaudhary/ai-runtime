use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Instant,
};

use futures_core::{Stream, stream::BoxStream};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio_util::sync::CancellationToken;

use crate::{
    llm::LlmProvider,
    runtime::{
        AgentRuntime, ConversationContext, RunTurnOptions, RuntimeConfig, RuntimeError,
        events::{RequestState, RuntimeEvent},
    },
    tools::ToolRegistry,
};

pub type Result<T> = std::result::Result<T, TtsError>;
pub type SttResult<T> = std::result::Result<T, SttError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioChunk {
    pub bytes: Vec<u8>,
    pub elapsed_ms: u128,
}

impl AudioChunk {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
            elapsed_ms: 0,
        }
    }

    pub fn with_elapsed_ms(mut self, elapsed_ms: u128) -> Self {
        self.elapsed_ms = elapsed_ms;
        self
    }
}

#[derive(Debug, Error)]
pub enum SttError {
    #[error("STT backend failed: {0}")]
    Backend(String),
}

#[derive(Debug, Error)]
pub enum TtsError {
    #[error("TTS backend failed: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsEvent {
    FirstAudio { elapsed_ms: u128 },
    AudioChunk { bytes: Vec<u8>, elapsed_ms: u128 },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptEvent {
    Partial { text: String, elapsed_ms: u128 },
    Final { text: String, elapsed_ms: u128 },
    Done,
}

pub trait SpeechToText: Send + Sync {
    fn stream_audio(
        &self,
        audio: BoxStream<'static, AudioChunk>,
    ) -> BoxStream<'static, SttResult<TranscriptEvent>>;
}

pub trait TextToSpeech: Send + Sync {
    fn stream_text(&self, text: BoxStream<'static, String>)
    -> BoxStream<'static, Result<TtsEvent>>;
}

#[derive(Debug, Error)]
pub enum TtsAdapterError {
    #[error(transparent)]
    Tts(#[from] TtsError),

    #[error("TTS adapter was cancelled")]
    Cancelled,
}

#[derive(Debug, Error)]
pub enum VoiceTurnError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error(transparent)]
    Stt(#[from] SttError),

    #[error(transparent)]
    Tts(#[from] TtsError),

    #[error("voice turn was cancelled")]
    Cancelled,

    #[error("STT stream completed without a final transcript")]
    MissingFinalTranscript,
}

#[derive(Debug, Clone)]
pub struct VoiceTurnOptions {
    pub cancellation_token: CancellationToken,
    pub tts_events: Option<UnboundedSender<TtsEvent>>,
}

impl VoiceTurnOptions {
    pub fn new(cancellation_token: CancellationToken) -> Self {
        Self {
            cancellation_token,
            tts_events: None,
        }
    }

    pub fn with_tts_events(mut self, events: UnboundedSender<TtsEvent>) -> Self {
        self.tts_events = Some(events);
        self
    }
}

impl Default for VoiceTurnOptions {
    fn default() -> Self {
        Self::new(CancellationToken::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceTurnOutput {
    pub text: String,
}

pub struct VoiceTurnRunner<P, T> {
    runtime: AgentRuntime<P>,
    tts: RuntimeTtsAdapter<T>,
}

pub struct VoicePipelineRunner<P, S, T> {
    stt: S,
    turn: VoiceTurnRunner<P, T>,
}

impl<P, S, T> VoicePipelineRunner<P, S, T>
where
    P: LlmProvider,
    S: SpeechToText,
    T: TextToSpeech,
{
    pub fn new(provider: P, tools: ToolRegistry, config: RuntimeConfig, stt: S, tts: T) -> Self {
        Self {
            stt,
            turn: VoiceTurnRunner::new(provider, tools, config, tts),
        }
    }

    pub async fn run_audio_turn(
        &self,
        context: &mut ConversationContext,
        audio: BoxStream<'static, AudioChunk>,
    ) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
        self.run_audio_turn_with_options(context, audio, VoiceTurnOptions::default())
            .await
    }

    pub async fn run_audio_turn_with_options(
        &self,
        context: &mut ConversationContext,
        audio: BoxStream<'static, AudioChunk>,
        options: VoiceTurnOptions,
    ) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
        let cancellation_token = options.cancellation_token.clone();
        let mut transcripts = self.stt.stream_audio(audio);

        let transcript = loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => return Err(VoiceTurnError::Cancelled),
                event = transcripts.next() => {
                    match event {
                        Some(Ok(TranscriptEvent::Partial { .. })) => {}
                        Some(Ok(TranscriptEvent::Final { text, .. })) => break text,
                        Some(Ok(TranscriptEvent::Done)) | None => {
                            return Err(VoiceTurnError::MissingFinalTranscript);
                        }
                        Some(Err(err)) => return Err(VoiceTurnError::Stt(err)),
                    }
                }
            }
        };

        drop(transcripts);
        self.turn
            .run_transcript_with_options(context, transcript, options)
            .await
    }
}

impl<P, T> VoiceTurnRunner<P, T>
where
    P: LlmProvider,
    T: TextToSpeech,
{
    pub fn new(provider: P, tools: ToolRegistry, config: RuntimeConfig, tts: T) -> Self {
        Self {
            runtime: AgentRuntime::new(provider, tools, config),
            tts: RuntimeTtsAdapter::new(tts),
        }
    }

    pub async fn run_transcript(
        &self,
        context: &mut ConversationContext,
        transcript: impl Into<String>,
    ) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
        self.run_transcript_with_options(context, transcript, VoiceTurnOptions::default())
            .await
    }

    pub async fn run_transcript_with_options(
        &self,
        context: &mut ConversationContext,
        transcript: impl Into<String>,
        options: VoiceTurnOptions,
    ) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
        let (runtime_tx, runtime_rx) = unbounded_channel();
        let (fallback_tts_tx, _fallback_tts_rx) = unbounded_channel();
        let tts_events = options.tts_events.unwrap_or(fallback_tts_tx);
        let runtime_token = options.cancellation_token.clone();
        let tts_token = options.cancellation_token.clone();

        let runtime_future = self.runtime.run_turn_with_options(
            context,
            transcript,
            RunTurnOptions::new(runtime_token).with_events(runtime_tx),
        );
        let tts_future = self.tts.run(runtime_rx, tts_token, tts_events);
        tokio::pin!(runtime_future);
        tokio::pin!(tts_future);

        tokio::select! {
            runtime_result = &mut runtime_future => {
                if runtime_result.is_err() {
                    options.cancellation_token.cancel();
                }
                let tts_result = tts_future.await;
                voice_turn_result(runtime_result, tts_result)
            }
            tts_result = &mut tts_future => {
                if tts_result.is_err() {
                    options.cancellation_token.cancel();
                }
                let runtime_result = runtime_future.await;
                voice_turn_result(runtime_result, tts_result)
            }
        }
    }
}

fn voice_turn_result(
    runtime_result: crate::runtime::Result<String>,
    tts_result: std::result::Result<(), TtsAdapterError>,
) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
    match runtime_result {
        Ok(text) => match tts_result {
            Ok(()) => Ok(VoiceTurnOutput { text }),
            Err(TtsAdapterError::Cancelled) => Err(VoiceTurnError::Cancelled),
            Err(TtsAdapterError::Tts(err)) => Err(VoiceTurnError::Tts(err)),
        },
        Err(RuntimeError::Cancelled) => match tts_result {
            Err(TtsAdapterError::Tts(err)) => Err(VoiceTurnError::Tts(err)),
            _ => Err(VoiceTurnError::Cancelled),
        },
        Err(err) => Err(VoiceTurnError::Runtime(err)),
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeTtsAdapter<T> {
    tts: T,
}

impl<T> RuntimeTtsAdapter<T>
where
    T: TextToSpeech,
{
    pub fn new(tts: T) -> Self {
        Self { tts }
    }

    pub async fn run(
        &self,
        runtime_events: UnboundedReceiver<RuntimeEvent>,
        cancellation_token: CancellationToken,
        tts_events: UnboundedSender<TtsEvent>,
    ) -> std::result::Result<(), TtsAdapterError> {
        let (text_tx, text_rx) = unbounded_channel();
        let mut text_tx = Some(text_tx);
        let mut runtime_events = Some(runtime_events);
        let mut audio_events = self.tts.stream_text(receiver_stream(text_rx));
        let mut audio_done = false;

        loop {
            if audio_done {
                return Ok(());
            }

            tokio::select! {
                _ = cancellation_token.cancelled() => return Err(TtsAdapterError::Cancelled),
                runtime_event = recv_runtime_event(&mut runtime_events), if runtime_events.is_some() => {
                    match runtime_event {
                        Some(RuntimeEvent::AssistantToken { text, .. }) => {
                            if let Some(sender) = &text_tx {
                                let _ = sender.send(text);
                            }
                        }
                        Some(RuntimeEvent::Lifecycle { to, .. }) if to.is_terminal() => {
                            text_tx = None;
                            if matches!(to, RequestState::Cancelled) {
                                return Err(TtsAdapterError::Cancelled);
                            }
                        }
                        Some(_) => {}
                        None => {
                            runtime_events = None;
                            text_tx = None;
                        }
                    }
                }
                audio_event = audio_events.next(), if !audio_done => {
                    match audio_event {
                        Some(Ok(event)) => {
                            audio_done = matches!(event, TtsEvent::Done);
                            let _ = tts_events.send(event);
                        }
                        Some(Err(err)) => return Err(TtsAdapterError::from(err)),
                        None => audio_done = true,
                    }
                }
            }
        }
    }
}

async fn recv_runtime_event(
    runtime_events: &mut Option<UnboundedReceiver<RuntimeEvent>>,
) -> Option<RuntimeEvent> {
    match runtime_events {
        Some(events) => events.recv().await,
        None => None,
    }
}

fn receiver_stream<T: Send + 'static>(rx: UnboundedReceiver<T>) -> BoxStream<'static, T> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    }))
}

#[derive(Debug, Clone)]
pub struct FakeSpeechToText {
    active_streams: Arc<AtomicUsize>,
}

impl FakeSpeechToText {
    pub fn new() -> Self {
        Self {
            active_streams: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::SeqCst)
    }
}

impl Default for FakeSpeechToText {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechToText for FakeSpeechToText {
    fn stream_audio(
        &self,
        audio: BoxStream<'static, AudioChunk>,
    ) -> BoxStream<'static, SttResult<TranscriptEvent>> {
        self.active_streams.fetch_add(1, Ordering::SeqCst);
        Box::pin(FakeSttStream {
            audio,
            transcript: String::new(),
            last_elapsed_ms: 0,
            pending: VecDeque::new(),
            done_sent: false,
            active_streams: Arc::clone(&self.active_streams),
        })
    }
}

struct FakeSttStream {
    audio: BoxStream<'static, AudioChunk>,
    transcript: String,
    last_elapsed_ms: u128,
    pending: VecDeque<TranscriptEvent>,
    done_sent: bool,
    active_streams: Arc<AtomicUsize>,
}

impl Stream for FakeSttStream {
    type Item = SttResult<TranscriptEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }

        if this.done_sent {
            return Poll::Ready(None);
        }

        match this.audio.as_mut().poll_next(cx) {
            Poll::Ready(Some(chunk)) => {
                let fragment = String::from_utf8_lossy(&chunk.bytes);
                this.transcript.push_str(&fragment);
                this.last_elapsed_ms = chunk.elapsed_ms;
                Poll::Ready(Some(Ok(TranscriptEvent::Partial {
                    text: this.transcript.clone(),
                    elapsed_ms: chunk.elapsed_ms,
                })))
            }
            Poll::Ready(None) => {
                this.done_sent = true;
                if !this.transcript.is_empty() {
                    this.pending.push_back(TranscriptEvent::Final {
                        text: this.transcript.clone(),
                        elapsed_ms: this.last_elapsed_ms,
                    });
                }
                this.pending.push_back(TranscriptEvent::Done);
                Poll::Ready(this.pending.pop_front().map(Ok))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for FakeSttStream {
    fn drop(&mut self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone)]
pub struct FakeTextToSpeech {
    active_streams: Arc<AtomicUsize>,
}

impl FakeTextToSpeech {
    pub fn new() -> Self {
        Self {
            active_streams: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::SeqCst)
    }
}

impl Default for FakeTextToSpeech {
    fn default() -> Self {
        Self::new()
    }
}

impl TextToSpeech for FakeTextToSpeech {
    fn stream_text(
        &self,
        text: BoxStream<'static, String>,
    ) -> BoxStream<'static, Result<TtsEvent>> {
        self.active_streams.fetch_add(1, Ordering::SeqCst);
        Box::pin(FakeTtsStream {
            text,
            started_at: Instant::now(),
            first_audio_sent: false,
            next_chunk_index: 0,
            pending: VecDeque::new(),
            done_sent: false,
            active_streams: Arc::clone(&self.active_streams),
        })
    }
}

struct FakeTtsStream {
    text: BoxStream<'static, String>,
    started_at: Instant,
    first_audio_sent: bool,
    next_chunk_index: usize,
    pending: VecDeque<TtsEvent>,
    done_sent: bool,
    active_streams: Arc<AtomicUsize>,
}

impl Stream for FakeTtsStream {
    type Item = Result<TtsEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }

        if this.done_sent {
            return Poll::Ready(None);
        }

        match this.text.as_mut().poll_next(cx) {
            Poll::Ready(Some(fragment)) => {
                let elapsed_ms = this.started_at.elapsed().as_millis();

                if !this.first_audio_sent {
                    this.first_audio_sent = true;
                    this.pending.push_back(TtsEvent::FirstAudio { elapsed_ms });
                }

                let bytes = deterministic_chunk(this.next_chunk_index, &fragment);
                this.next_chunk_index += 1;
                this.pending
                    .push_back(TtsEvent::AudioChunk { bytes, elapsed_ms });

                Poll::Ready(this.pending.pop_front().map(Ok))
            }
            Poll::Ready(None) => {
                this.done_sent = true;
                Poll::Ready(Some(Ok(TtsEvent::Done)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for FakeTtsStream {
    fn drop(&mut self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }
}

fn deterministic_chunk(index: usize, fragment: &str) -> Vec<u8> {
    format!("fake-tts:{index}:{fragment}").into_bytes()
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use tokio::{
        sync::mpsc,
        time::{sleep, timeout},
    };

    use super::*;
    use crate::{
        llm::{ChatRequest, LlmProvider, Result as LlmResult, TokenEvent},
        runtime::{AgentRuntime, ConversationContext, RunTurnOptions, RuntimeConfig, RuntimeError},
        tools::ToolRegistry,
    };

    #[tokio::test]
    async fn fake_tts_converts_text_fragments_to_deterministic_audio_chunks() {
        let tts = FakeTextToSpeech::new();
        let adapter = RuntimeTtsAdapter::new(tts);
        let cancellation_token = CancellationToken::new();
        let (runtime_tx, runtime_rx) = mpsc::unbounded_channel();
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

        let task = tokio::spawn(async move {
            adapter
                .run(runtime_rx, cancellation_token, audio_tx)
                .await
                .expect("adapter should finish");
        });

        runtime_tx
            .send(RuntimeEvent::AssistantToken {
                text: "Hel".to_string(),
                elapsed_ms: 1,
            })
            .expect("runtime event receiver should be active");
        runtime_tx
            .send(RuntimeEvent::AssistantToken {
                text: "lo".to_string(),
                elapsed_ms: 2,
            })
            .expect("runtime event receiver should be active");
        drop(runtime_tx);

        let mut events = Vec::new();
        while let Some(event) = timeout(Duration::from_secs(1), audio_rx.recv())
            .await
            .expect("audio event should arrive")
        {
            let done = matches!(event, TtsEvent::Done);
            events.push(event);
            if done {
                break;
            }
        }

        task.await.expect("adapter task should not panic");

        assert!(matches!(events.first(), Some(TtsEvent::FirstAudio { .. })));
        assert_eq!(
            audio_chunks(&events),
            vec![b"fake-tts:0:Hel".to_vec(), b"fake-tts:1:lo".to_vec()]
        );
        assert!(matches!(events.last(), Some(TtsEvent::Done)));
    }

    #[tokio::test]
    async fn fake_stt_converts_audio_fragments_to_transcript_events() {
        let stt = FakeSpeechToText::new();
        let audio = Box::pin(futures_util::stream::iter(vec![
            AudioChunk::new("hel").with_elapsed_ms(7),
            AudioChunk::new("lo").with_elapsed_ms(11),
        ]));

        let events = stt
            .stream_audio(audio)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<SttResult<Vec<_>>>()
            .expect("fake STT stream should succeed");

        assert_eq!(
            events,
            vec![
                TranscriptEvent::Partial {
                    text: "hel".to_string(),
                    elapsed_ms: 7,
                },
                TranscriptEvent::Partial {
                    text: "hello".to_string(),
                    elapsed_ms: 11,
                },
                TranscriptEvent::Final {
                    text: "hello".to_string(),
                    elapsed_ms: 11,
                },
                TranscriptEvent::Done,
            ]
        );
        assert_eq!(stt.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn voice_pipeline_runner_uses_final_transcript_and_streams_tts_audio() {
        let provider = StaticTokenProvider::new(vec!["Hi", " there"]);
        let stt = FakeSpeechToText::new();
        let tts = FakeTextToSpeech::new();
        let pipeline = VoicePipelineRunner::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            stt,
            tts,
        );
        let mut context = ConversationContext::new();
        let (tts_tx, mut tts_rx) = mpsc::unbounded_channel();
        let audio = Box::pin(futures_util::stream::iter(vec![
            AudioChunk::new("hello").with_elapsed_ms(3),
            AudioChunk::new(" runtime").with_elapsed_ms(9),
        ]));

        let output = pipeline
            .run_audio_turn_with_options(
                &mut context,
                audio,
                VoiceTurnOptions::default().with_tts_events(tts_tx),
            )
            .await
            .expect("voice pipeline should succeed");

        assert_eq!(output.text, "Hi there");
        assert_eq!(context.messages()[0].role, "user");
        assert_eq!(context.messages()[0].content, "hello runtime");
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| message.content.as_str()),
            Some("Hi there")
        );

        let events = drain_tts_events(&mut tts_rx);
        assert_eq!(
            audio_chunks(&events),
            vec![b"fake-tts:0:Hi".to_vec(), b"fake-tts:1: there".to_vec()]
        );
        assert!(matches!(events.last(), Some(TtsEvent::Done)));
    }

    #[tokio::test]
    async fn cancelling_during_stt_streaming_drops_active_stt_stream() {
        let provider = StaticTokenProvider::new(vec!["unused"]);
        let stt = FakeSpeechToText::new();
        let stt_probe = stt.clone();
        let pipeline = VoicePipelineRunner::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            stt,
            FakeTextToSpeech::new(),
        );
        let cancellation_token = CancellationToken::new();
        let mut context = ConversationContext::new();
        let (_audio_tx, audio_rx) = mpsc::unbounded_channel();
        let audio = receiver_stream(audio_rx);

        let result = {
            let turn = pipeline.run_audio_turn_with_options(
                &mut context,
                audio,
                VoiceTurnOptions::new(cancellation_token.clone()),
            );
            tokio::pin!(turn);

            tokio::select! {
                _ = wait_for_count_value(&stt_probe, 1) => {}
                result = &mut turn => panic!("voice pipeline finished before cancellation: {result:?}"),
            }

            cancellation_token.cancel();

            timeout(Duration::from_secs(1), &mut turn)
                .await
                .expect("voice pipeline should stop after cancellation")
        };

        assert!(matches!(result, Err(VoiceTurnError::Cancelled)));
        assert!(context.messages().is_empty());
        assert_eq!(stt_probe.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn voice_turn_runner_streams_audio_and_commits_completed_answer() {
        let provider = StaticTokenProvider::new(vec!["Hel", "lo"]);
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let runner = VoiceTurnRunner::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            tts,
        );
        let mut context = ConversationContext::new();
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

        let output = runner
            .run_transcript_with_options(
                &mut context,
                "hello",
                VoiceTurnOptions::default().with_tts_events(audio_tx),
            )
            .await
            .expect("voice turn should succeed");

        assert_eq!(output.text, "Hello");
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| message.content.as_str()),
            Some("Hello")
        );

        let events = drain_tts_events(&mut audio_rx);
        assert!(matches!(events.first(), Some(TtsEvent::FirstAudio { .. })));
        assert_eq!(
            audio_chunks(&events),
            vec![b"fake-tts:0:Hel".to_vec(), b"fake-tts:1:lo".to_vec()]
        );
        assert!(matches!(events.last(), Some(TtsEvent::Done)));
        assert_eq!(tts_probe.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn voice_turn_runner_cancellation_drops_tts_and_leaves_partial_text_uncommitted() {
        let provider = TokenThenHangingProvider {
            token: "partial".to_string(),
        };
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let runner = VoiceTurnRunner::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            tts,
        );
        let cancellation_token = CancellationToken::new();
        let mut context = ConversationContext::new();
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

        let result = {
            let turn = runner.run_transcript_with_options(
                &mut context,
                "hello",
                VoiceTurnOptions::new(cancellation_token.clone()).with_tts_events(audio_tx),
            );
            tokio::pin!(turn);

            let first_chunk = tokio::select! {
                chunk = next_audio_chunk(&mut audio_rx) => chunk,
                result = &mut turn => panic!("voice turn finished before cancellation: {result:?}"),
            };
            assert_eq!(first_chunk, b"fake-tts:0:partial".to_vec());

            cancellation_token.cancel();

            timeout(Duration::from_secs(1), &mut turn)
                .await
                .expect("voice turn should stop after cancellation")
        };

        assert!(matches!(result, Err(VoiceTurnError::Cancelled)));
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.role != "assistant"),
            "partial assistant output should not be committed: {:#?}",
            context.messages()
        );
        assert_eq!(tts_probe.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn cancelling_during_tts_streaming_drops_active_tts_stream() {
        let tts = HangingTextToSpeech::new();
        let active_streams = tts.active_streams();
        let adapter = RuntimeTtsAdapter::new(tts);
        let cancellation_token = CancellationToken::new();
        let (runtime_tx, runtime_rx) = mpsc::unbounded_channel();
        let (audio_tx, _audio_rx) = mpsc::unbounded_channel();

        let task_token = cancellation_token.clone();
        let task = tokio::spawn(async move { adapter.run(runtime_rx, task_token, audio_tx).await });

        wait_for_count(&active_streams, 1).await;
        runtime_tx
            .send(RuntimeEvent::AssistantToken {
                text: "hello".to_string(),
                elapsed_ms: 1,
            })
            .expect("runtime event receiver should be active");
        cancellation_token.cancel();

        let result = timeout(Duration::from_secs(1), task)
            .await
            .expect("adapter should stop after cancellation")
            .expect("adapter task should not panic");

        assert!(matches!(result, Err(TtsAdapterError::Cancelled)));
        wait_for_count(&active_streams, 0).await;
    }

    #[tokio::test]
    async fn cancelling_during_llm_streaming_stops_token_forwarding_to_tts() {
        let provider = DelayedTokenProvider::new(vec!["first", "late"], Duration::from_millis(200));
        let runtime = AgentRuntime::new(provider, ToolRegistry::new(), RuntimeConfig::new("test"));
        let tts = FakeTextToSpeech::new();
        let adapter = RuntimeTtsAdapter::new(tts);
        let cancellation_token = CancellationToken::new();
        let (runtime_tx, runtime_rx) = mpsc::unbounded_channel();
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

        let adapter_token = cancellation_token.clone();
        let adapter_task =
            tokio::spawn(async move { adapter.run(runtime_rx, adapter_token, audio_tx).await });

        let runtime_token = cancellation_token.clone();
        let runtime_task = tokio::spawn(async move {
            let mut context = ConversationContext::new();
            runtime
                .run_turn_with_options(
                    &mut context,
                    "hello",
                    RunTurnOptions::new(runtime_token).with_events(runtime_tx),
                )
                .await
        });

        let first_chunk = next_audio_chunk(&mut audio_rx).await;
        assert_eq!(first_chunk, b"fake-tts:0:first".to_vec());

        cancellation_token.cancel();

        let runtime_result = timeout(Duration::from_secs(1), runtime_task)
            .await
            .expect("runtime should stop after cancellation")
            .expect("runtime task should not panic");
        assert!(matches!(runtime_result, Err(RuntimeError::Cancelled)));

        let adapter_result = timeout(Duration::from_secs(1), adapter_task)
            .await
            .expect("adapter should stop after cancellation")
            .expect("adapter task should not panic");
        assert!(matches!(adapter_result, Err(TtsAdapterError::Cancelled)));

        sleep(Duration::from_millis(250)).await;
        let late_chunks = drain_audio_chunks(&mut audio_rx);
        assert!(
            late_chunks
                .iter()
                .all(|chunk| chunk != &b"fake-tts:1:late".to_vec()),
            "late token was forwarded after cancellation: {late_chunks:?}"
        );
    }

    fn audio_chunks(events: &[TtsEvent]) -> Vec<Vec<u8>> {
        events
            .iter()
            .filter_map(|event| match event {
                TtsEvent::AudioChunk { bytes, .. } => Some(bytes.clone()),
                _ => None,
            })
            .collect()
    }

    async fn next_audio_chunk(rx: &mut UnboundedReceiver<TtsEvent>) -> Vec<u8> {
        loop {
            match timeout(Duration::from_secs(1), rx.recv())
                .await
                .expect("audio event should arrive")
                .expect("audio event stream should still be open")
            {
                TtsEvent::AudioChunk { bytes, .. } => return bytes,
                TtsEvent::FirstAudio { .. } => {}
                TtsEvent::Done => panic!("TTS stream completed before audio chunk"),
            }
        }
    }

    fn drain_audio_chunks(rx: &mut UnboundedReceiver<TtsEvent>) -> Vec<Vec<u8>> {
        let mut chunks = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let TtsEvent::AudioChunk { bytes, .. } = event {
                chunks.push(bytes);
            }
        }
        chunks
    }

    fn drain_tts_events(rx: &mut UnboundedReceiver<TtsEvent>) -> Vec<TtsEvent> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    async fn wait_for_count(count: &AtomicUsize, expected: usize) {
        timeout(Duration::from_secs(1), async {
            while count.load(Ordering::SeqCst) != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("counter should reach expected value");
    }

    async fn wait_for_count_value(stt: &FakeSpeechToText, expected: usize) {
        timeout(Duration::from_secs(1), async {
            while stt.active_stream_count() != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("counter should reach expected value");
    }

    #[derive(Clone)]
    struct HangingTextToSpeech {
        active_streams: Arc<AtomicUsize>,
    }

    impl HangingTextToSpeech {
        fn new() -> Self {
            Self {
                active_streams: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn active_streams(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.active_streams)
        }
    }

    impl TextToSpeech for HangingTextToSpeech {
        fn stream_text(
            &self,
            text: BoxStream<'static, String>,
        ) -> BoxStream<'static, Result<TtsEvent>> {
            self.active_streams.fetch_add(1, Ordering::SeqCst);
            Box::pin(HangingTtsStream {
                text,
                text_seen: Arc::new(AtomicBool::new(false)),
                recorded: Arc::new(Mutex::new(Vec::new())),
                active_streams: Arc::clone(&self.active_streams),
            })
        }
    }

    struct HangingTtsStream {
        text: BoxStream<'static, String>,
        text_seen: Arc<AtomicBool>,
        recorded: Arc<Mutex<Vec<String>>>,
        active_streams: Arc<AtomicUsize>,
    }

    impl Stream for HangingTtsStream {
        type Item = Result<TtsEvent>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            if !this.text_seen.load(Ordering::SeqCst) {
                match this.text.as_mut().poll_next(cx) {
                    Poll::Ready(Some(text)) => {
                        this.recorded.lock().expect("record mutex").push(text);
                        this.text_seen.store(true, Ordering::SeqCst);
                    }
                    Poll::Ready(None) => return Poll::Ready(Some(Ok(TtsEvent::Done))),
                    Poll::Pending => {}
                }
            }

            Poll::Pending
        }
    }

    impl Drop for HangingTtsStream {
        fn drop(&mut self) {
            self.active_streams.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct StaticTokenProvider {
        tokens: Vec<String>,
    }

    impl StaticTokenProvider {
        fn new(tokens: Vec<&str>) -> Self {
            Self {
                tokens: tokens.into_iter().map(str::to_string).collect(),
            }
        }
    }

    impl LlmProvider for StaticTokenProvider {
        fn stream_chat(&self, _req: ChatRequest) -> BoxStream<'static, LlmResult<TokenEvent>> {
            let mut events = self
                .tokens
                .iter()
                .cloned()
                .map(|text| Ok(TokenEvent::Token { text }))
                .collect::<Vec<_>>();
            events.push(Ok(TokenEvent::Done));
            Box::pin(futures_util::stream::iter(events))
        }
    }

    struct TokenThenHangingProvider {
        token: String,
    }

    impl LlmProvider for TokenThenHangingProvider {
        fn stream_chat(&self, _req: ChatRequest) -> BoxStream<'static, LlmResult<TokenEvent>> {
            Box::pin(TokenThenHangingStream {
                token: Some(self.token.clone()),
            })
        }
    }

    struct TokenThenHangingStream {
        token: Option<String>,
    }

    impl Stream for TokenThenHangingStream {
        type Item = LlmResult<TokenEvent>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if let Some(text) = self.token.take() {
                return Poll::Ready(Some(Ok(TokenEvent::Token { text })));
            }

            Poll::Pending
        }
    }

    struct DelayedTokenProvider {
        tokens: Vec<String>,
        delay: Duration,
    }

    impl DelayedTokenProvider {
        fn new(tokens: Vec<&str>, delay: Duration) -> Self {
            Self {
                tokens: tokens.into_iter().map(str::to_string).collect(),
                delay,
            }
        }
    }

    impl LlmProvider for DelayedTokenProvider {
        fn stream_chat(&self, _req: ChatRequest) -> BoxStream<'static, LlmResult<TokenEvent>> {
            let tokens = self.tokens.clone();
            let delay = self.delay;

            Box::pin(futures_util::stream::unfold(0, move |index| {
                let tokens = tokens.clone();
                async move {
                    if index < tokens.len() {
                        if index > 0 {
                            sleep(delay).await;
                        }
                        Some((
                            Ok(TokenEvent::Token {
                                text: tokens[index].clone(),
                            }),
                            index + 1,
                        ))
                    } else if index == tokens.len() {
                        Some((Ok(TokenEvent::Done), index + 1))
                    } else {
                        None
                    }
                }
            }))
        }
    }
}
