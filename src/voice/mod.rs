use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Instant,
};

use futures_core::{Stream, stream::BoxStream};
use futures_util::{StreamExt, future::BoxFuture};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::{
    llm::LlmProvider,
    runtime::{
        AgentRuntime, ConversationContext, RunTurnOptions, RuntimeConfig, RuntimeError,
        events::{RequestState, RuntimeEvent},
    },
    tools::ToolRegistry,
};

pub mod deepgram;
pub mod livekit;

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
    PlaybackReset,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptEvent {
    Partial { text: String, elapsed_ms: u128 },
    Final { text: String, elapsed_ms: u128 },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceEvent {
    SttPartial {
        elapsed_ms: u128,
        stt_elapsed_ms: u128,
        transcript_chars: usize,
    },
    SttFinal {
        elapsed_ms: u128,
        stt_elapsed_ms: u128,
        transcript_chars: usize,
    },
    LlmFirstToken {
        elapsed_ms: u128,
        runtime_elapsed_ms: u128,
    },
    TtsFirstAudio {
        elapsed_ms: u128,
        tts_elapsed_ms: u128,
    },
    VoiceTurnCompleted {
        elapsed_ms: u128,
        output_chars: usize,
    },
    VoiceTurnCancelled {
        elapsed_ms: u128,
    },
    VoiceTurnInterrupted {
        elapsed_ms: u128,
    },
    VoiceTurnFailed {
        elapsed_ms: u128,
    },
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

#[derive(Debug, Error)]
pub enum VoiceTransportError {
    #[error("voice transport is closed")]
    Closed,
}

#[derive(Debug, Error)]
pub enum VoiceSessionError {
    #[error(transparent)]
    VoiceTurn(#[from] VoiceTurnError),

    #[error("voice turn task failed: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
}

pub enum VoiceTransportEvent {
    SpeechStarted,
    AudioTurn(BoxStream<'static, AudioChunk>),
    Closed,
}

pub trait VoiceTransport: Send {
    fn next_event(&mut self) -> BoxFuture<'_, Option<VoiceTransportEvent>>;

    fn tts_events(&self) -> UnboundedSender<TtsEvent>;
}

#[derive(Debug, Clone)]
pub struct VoiceTurnOptions {
    pub cancellation_token: CancellationToken,
    pub tts_events: Option<UnboundedSender<TtsEvent>>,
    pub voice_events: Option<UnboundedSender<VoiceEvent>>,
}

impl VoiceTurnOptions {
    pub fn new(cancellation_token: CancellationToken) -> Self {
        Self {
            cancellation_token,
            tts_events: None,
            voice_events: None,
        }
    }

    pub fn with_tts_events(mut self, events: UnboundedSender<TtsEvent>) -> Self {
        self.tts_events = Some(events);
        self
    }

    pub fn with_voice_events(mut self, events: UnboundedSender<VoiceEvent>) -> Self {
        self.voice_events = Some(events);
        self
    }

    fn with_optional_voice_events(mut self, events: Option<UnboundedSender<VoiceEvent>>) -> Self {
        self.voice_events = events;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceLatencyMetric {
    SttFinalization,
    LlmFirstToken,
    TtsFirstAudio,
    VoiceTurnRoundTrip,
}

impl VoiceLatencyMetric {
    pub fn name(self) -> &'static str {
        match self {
            Self::SttFinalization => "stt_finalization",
            Self::LlmFirstToken => "llm_first_token",
            Self::TtsFirstAudio => "tts_first_audio",
            Self::VoiceTurnRoundTrip => "voice_turn_round_trip",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceLatencyStats {
    pub samples: usize,
    pub p50_ms: u128,
    pub p95_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceLatencySnapshot {
    pub stt_finalization: Option<VoiceLatencyStats>,
    pub llm_first_token: Option<VoiceLatencyStats>,
    pub tts_first_audio: Option<VoiceLatencyStats>,
    pub voice_turn_round_trip: Option<VoiceLatencyStats>,
}

#[derive(Debug, Clone, Default)]
pub struct VoiceLatencyRecorder {
    stt_finalization: Vec<u128>,
    llm_first_token: Vec<u128>,
    tts_first_audio: Vec<u128>,
    voice_turn_round_trip: Vec<u128>,
}

impl VoiceLatencyRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, metric: VoiceLatencyMetric, elapsed_ms: u128) {
        match metric {
            VoiceLatencyMetric::SttFinalization => self.stt_finalization.push(elapsed_ms),
            VoiceLatencyMetric::LlmFirstToken => self.llm_first_token.push(elapsed_ms),
            VoiceLatencyMetric::TtsFirstAudio => self.tts_first_audio.push(elapsed_ms),
            VoiceLatencyMetric::VoiceTurnRoundTrip => self.voice_turn_round_trip.push(elapsed_ms),
        }
    }

    pub fn record_event(&mut self, event: &VoiceEvent) {
        match event {
            VoiceEvent::SttFinal { stt_elapsed_ms, .. } => {
                self.record(VoiceLatencyMetric::SttFinalization, *stt_elapsed_ms);
            }
            VoiceEvent::LlmFirstToken {
                runtime_elapsed_ms, ..
            } => {
                self.record(VoiceLatencyMetric::LlmFirstToken, *runtime_elapsed_ms);
            }
            VoiceEvent::TtsFirstAudio { tts_elapsed_ms, .. } => {
                self.record(VoiceLatencyMetric::TtsFirstAudio, *tts_elapsed_ms);
            }
            VoiceEvent::VoiceTurnCompleted { elapsed_ms, .. } => {
                self.record(VoiceLatencyMetric::VoiceTurnRoundTrip, *elapsed_ms);
            }
            VoiceEvent::SttPartial { .. }
            | VoiceEvent::VoiceTurnCancelled { .. }
            | VoiceEvent::VoiceTurnInterrupted { .. }
            | VoiceEvent::VoiceTurnFailed { .. } => {}
        }
    }

    pub fn snapshot(&self) -> VoiceLatencySnapshot {
        VoiceLatencySnapshot {
            stt_finalization: latency_stats(&self.stt_finalization),
            llm_first_token: latency_stats(&self.llm_first_token),
            tts_first_audio: latency_stats(&self.tts_first_audio),
            voice_turn_round_trip: latency_stats(&self.voice_turn_round_trip),
        }
    }
}

fn latency_stats(samples: &[u128]) -> Option<VoiceLatencyStats> {
    if samples.is_empty() {
        return None;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    Some(VoiceLatencyStats {
        samples: sorted.len(),
        p50_ms: percentile_nearest_rank(&sorted, 50),
        p95_ms: percentile_nearest_rank(&sorted, 95),
    })
}

fn percentile_nearest_rank(sorted_samples: &[u128], percentile: usize) -> u128 {
    debug_assert!(!sorted_samples.is_empty());
    let len = sorted_samples.len();
    let rank = (percentile * len).div_ceil(100);
    sorted_samples[rank.saturating_sub(1)]
}

pub struct VoiceTurnRunner<P, T> {
    runtime: AgentRuntime<P>,
    tts: RuntimeTtsAdapter<T>,
}

pub struct VoicePipelineRunner<P, S, T> {
    stt: S,
    turn: VoiceTurnRunner<P, T>,
}

pub struct VoiceSession<P, S, T> {
    pipeline: Arc<VoicePipelineRunner<P, S, T>>,
    context: Arc<tokio::sync::Mutex<ConversationContext>>,
    voice_events: Option<UnboundedSender<VoiceEvent>>,
    active_turn: Option<ActiveVoiceTurn>,
}

struct ActiveVoiceTurn {
    cancellation_token: CancellationToken,
    tts_events: UnboundedSender<TtsEvent>,
    started_at: Instant,
    task: JoinHandle<std::result::Result<VoiceTurnOutput, VoiceTurnError>>,
}

impl<P, S, T> VoiceSession<P, S, T>
where
    P: LlmProvider + 'static,
    S: SpeechToText + 'static,
    T: TextToSpeech + 'static,
{
    pub fn new(
        provider: P,
        tools: ToolRegistry,
        config: RuntimeConfig,
        stt: S,
        tts: T,
        context: ConversationContext,
    ) -> Self {
        Self {
            pipeline: Arc::new(VoicePipelineRunner::new(provider, tools, config, stt, tts)),
            context: Arc::new(tokio::sync::Mutex::new(context)),
            voice_events: None,
            active_turn: None,
        }
    }

    pub fn with_voice_events(mut self, events: UnboundedSender<VoiceEvent>) -> Self {
        self.voice_events = Some(events);
        self
    }

    pub fn has_active_turn(&self) -> bool {
        self.active_turn.is_some()
    }

    pub async fn context_snapshot(&self) -> ConversationContext {
        self.context.lock().await.clone()
    }

    pub async fn start_audio_turn(
        &mut self,
        audio: BoxStream<'static, AudioChunk>,
        tts_events: UnboundedSender<TtsEvent>,
    ) -> std::result::Result<(), VoiceSessionError> {
        self.interrupt_active_turn().await?;
        self.spawn_audio_turn(audio, tts_events);
        Ok(())
    }

    fn spawn_audio_turn(
        &mut self,
        audio: BoxStream<'static, AudioChunk>,
        tts_events: UnboundedSender<TtsEvent>,
    ) {
        let pipeline = Arc::clone(&self.pipeline);
        let context = Arc::clone(&self.context);
        let cancellation_token = CancellationToken::new();
        let options = VoiceTurnOptions::new(cancellation_token.clone())
            .with_tts_events(tts_events.clone())
            .with_optional_voice_events(self.voice_events.clone());
        let task = tokio::spawn(async move {
            let mut context = context.lock().await;
            pipeline
                .run_audio_turn_with_options(&mut context, audio, options)
                .await
        });

        self.active_turn = Some(ActiveVoiceTurn {
            cancellation_token,
            tts_events,
            started_at: Instant::now(),
            task,
        });
    }

    pub async fn finish_active_turn(
        &mut self,
    ) -> std::result::Result<Option<VoiceTurnOutput>, VoiceSessionError> {
        let Some(active_turn) = self.active_turn.take() else {
            return Ok(None);
        };

        let output = active_turn.task.await??;
        Ok(Some(output))
    }

    pub async fn cancel_active_turn(&mut self) -> std::result::Result<(), VoiceSessionError> {
        let Some(active_turn) = self.active_turn.take() else {
            return Ok(());
        };

        let was_running = !active_turn.task.is_finished();
        if was_running {
            active_turn.cancellation_token.cancel();
            reset_playback(&active_turn.tts_events);
        }

        let result = active_turn.task.await?;
        if was_running {
            reset_playback(&active_turn.tts_events);
        }

        match result {
            Ok(_) => Ok(()),
            Err(VoiceTurnError::Cancelled) if was_running => Ok(()),
            Err(VoiceTurnError::MissingFinalTranscript) => Ok(()),
            Err(err) => Err(VoiceSessionError::VoiceTurn(err)),
        }
    }

    pub async fn run_transport<V>(
        &mut self,
        mut transport: V,
    ) -> std::result::Result<(), VoiceSessionError>
    where
        V: VoiceTransport,
    {
        loop {
            match transport.next_event().await {
                Some(VoiceTransportEvent::SpeechStarted) => {
                    self.interrupt_active_transport_turn().await?;
                }
                Some(VoiceTransportEvent::AudioTurn(audio)) => {
                    self.interrupt_active_transport_turn().await?;
                    self.spawn_audio_turn(audio, transport.tts_events());
                }
                Some(VoiceTransportEvent::Closed) => {
                    let _ = self.finish_active_turn().await?;
                    return Ok(());
                }
                None => {
                    self.cancel_active_turn().await?;
                    return Ok(());
                }
            }
        }
    }

    async fn interrupt_active_turn(&mut self) -> std::result::Result<(), VoiceSessionError> {
        let Some(active_turn) = self.active_turn.take() else {
            return Ok(());
        };

        let was_running = !active_turn.task.is_finished();
        if was_running {
            emit_voice_event(
                &self.voice_events,
                VoiceEvent::VoiceTurnInterrupted {
                    elapsed_ms: active_turn.started_at.elapsed().as_millis(),
                },
            );
            active_turn.cancellation_token.cancel();
            reset_playback(&active_turn.tts_events);
        }

        let result = active_turn.task.await?;
        if was_running {
            reset_playback(&active_turn.tts_events);
        }

        match result {
            Ok(_) => Ok(()),
            Err(VoiceTurnError::Cancelled) if was_running => Ok(()),
            Err(err) => Err(VoiceSessionError::VoiceTurn(err)),
        }
    }

    async fn interrupt_active_transport_turn(
        &mut self,
    ) -> std::result::Result<(), VoiceSessionError> {
        match self.interrupt_active_turn().await {
            Ok(()) => Ok(()),
            Err(VoiceSessionError::VoiceTurn(VoiceTurnError::MissingFinalTranscript)) => Ok(()),
            Err(err) => Err(err),
        }
    }
}

impl<P, S, T> Drop for VoiceSession<P, S, T> {
    fn drop(&mut self) {
        if let Some(active_turn) = &self.active_turn {
            active_turn.cancellation_token.cancel();
        }
    }
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
        let started_at = Instant::now();
        let cancellation_token = options.cancellation_token.clone();
        let voice_events = options.voice_events.clone();
        let mut transcripts = self.stt.stream_audio(audio);

        let transcript = loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    emit_voice_event(
                        &voice_events,
                        VoiceEvent::VoiceTurnCancelled {
                            elapsed_ms: started_at.elapsed().as_millis(),
                        },
                    );
                    return Err(VoiceTurnError::Cancelled);
                }
                event = transcripts.next() => {
                    match event {
                        Some(Ok(TranscriptEvent::Partial { text, elapsed_ms })) => {
                            emit_voice_event(
                                &voice_events,
                                VoiceEvent::SttPartial {
                                    elapsed_ms: started_at.elapsed().as_millis(),
                                    stt_elapsed_ms: elapsed_ms,
                                    transcript_chars: text.len(),
                                },
                            );
                        }
                        Some(Ok(TranscriptEvent::Final { text, elapsed_ms })) => {
                            emit_voice_event(
                                &voice_events,
                                VoiceEvent::SttFinal {
                                    elapsed_ms: started_at.elapsed().as_millis(),
                                    stt_elapsed_ms: elapsed_ms,
                                    transcript_chars: text.len(),
                                },
                            );
                            break text;
                        }
                        Some(Ok(TranscriptEvent::Done)) | None => {
                            emit_voice_event(
                                &voice_events,
                                VoiceEvent::VoiceTurnFailed {
                                    elapsed_ms: started_at.elapsed().as_millis(),
                                },
                            );
                            return Err(VoiceTurnError::MissingFinalTranscript);
                        }
                        Some(Err(err)) => {
                            emit_voice_event(
                                &voice_events,
                                VoiceEvent::VoiceTurnFailed {
                                    elapsed_ms: started_at.elapsed().as_millis(),
                                },
                            );
                            return Err(VoiceTurnError::Stt(err));
                        }
                    }
                }
            }
        };

        drop(transcripts);
        self.turn
            .run_transcript_with_started_at(context, transcript, options, started_at)
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
        self.run_transcript_with_started_at(context, transcript, options, Instant::now())
            .await
    }

    async fn run_transcript_with_started_at(
        &self,
        context: &mut ConversationContext,
        transcript: impl Into<String>,
        options: VoiceTurnOptions,
        started_at: Instant,
    ) -> std::result::Result<VoiceTurnOutput, VoiceTurnError> {
        let VoiceTurnOptions {
            cancellation_token,
            tts_events,
            voice_events,
        } = options;
        let (runtime_tx, runtime_rx) = unbounded_channel();
        let (fallback_tts_tx, _fallback_tts_rx) = unbounded_channel();
        let tts_events = tts_events.unwrap_or(fallback_tts_tx);
        let runtime_token = cancellation_token.clone();
        let tts_token = cancellation_token.clone();

        let runtime_future = self.runtime.run_turn_with_options(
            context,
            transcript,
            RunTurnOptions::new(runtime_token).with_events(runtime_tx),
        );
        let tts_future = self.tts.run_observed(
            runtime_rx,
            tts_token,
            tts_events,
            voice_events.clone(),
            started_at,
        );
        tokio::pin!(runtime_future);
        tokio::pin!(tts_future);

        let result = tokio::select! {
            runtime_result = &mut runtime_future => {
                if runtime_result.is_err() {
                    cancellation_token.cancel();
                }
                let tts_result = tts_future.await;
                voice_turn_result(runtime_result, tts_result)
            }
            tts_result = &mut tts_future => {
                if tts_result.is_err() {
                    cancellation_token.cancel();
                }
                let runtime_result = runtime_future.await;
                voice_turn_result(runtime_result, tts_result)
            }
        };

        emit_voice_turn_result(&voice_events, &result, started_at);
        result
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

fn emit_voice_turn_result(
    voice_events: &Option<UnboundedSender<VoiceEvent>>,
    result: &std::result::Result<VoiceTurnOutput, VoiceTurnError>,
    started_at: Instant,
) {
    let elapsed_ms = started_at.elapsed().as_millis();
    let event = match result {
        Ok(output) => VoiceEvent::VoiceTurnCompleted {
            elapsed_ms,
            output_chars: output.text.len(),
        },
        Err(VoiceTurnError::Cancelled) => VoiceEvent::VoiceTurnCancelled { elapsed_ms },
        Err(_) => VoiceEvent::VoiceTurnFailed { elapsed_ms },
    };
    emit_voice_event(voice_events, event);
}

fn emit_voice_event(events: &Option<UnboundedSender<VoiceEvent>>, event: VoiceEvent) {
    if let Some(events) = events {
        let _ = events.send(event);
    }
}

fn reset_playback(tts_events: &UnboundedSender<TtsEvent>) {
    let _ = tts_events.send(TtsEvent::PlaybackReset);
}

pub(crate) fn voice_stt_trace_enabled() -> bool {
    std::env::var("AI_RUNTIME_STT_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(crate) fn trace_voice_stt(message: &str) {
    if !voice_stt_trace_enabled() {
        return;
    }

    static TRACE_STARTED_AT: OnceLock<Instant> = OnceLock::new();
    let started_at = TRACE_STARTED_AT.get_or_init(Instant::now);
    println!(
        "voice-trace> trace_elapsed_ms={} {message}",
        started_at.elapsed().as_millis()
    );
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
        self.run_observed(
            runtime_events,
            cancellation_token,
            tts_events,
            None,
            Instant::now(),
        )
        .await
    }

    async fn run_observed(
        &self,
        runtime_events: UnboundedReceiver<RuntimeEvent>,
        cancellation_token: CancellationToken,
        tts_events: UnboundedSender<TtsEvent>,
        voice_events: Option<UnboundedSender<VoiceEvent>>,
        started_at: Instant,
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
                        Some(RuntimeEvent::LlmFirstToken { elapsed_ms }) => {
                            emit_voice_event(
                                &voice_events,
                                VoiceEvent::LlmFirstToken {
                                    elapsed_ms: started_at.elapsed().as_millis(),
                                    runtime_elapsed_ms: elapsed_ms,
                                },
                            );
                        }
                        Some(RuntimeEvent::AssistantToken { text, .. }) => {
                            if let Some(sender) = &text_tx {
                                let text = sanitize_tts_fragment(&text);
                                if !text.is_empty() {
                                    let _ = sender.send(text);
                                }
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
                            if let TtsEvent::FirstAudio { elapsed_ms } = event {
                                emit_voice_event(
                                    &voice_events,
                                    VoiceEvent::TtsFirstAudio {
                                        elapsed_ms: started_at.elapsed().as_millis(),
                                        tts_elapsed_ms: elapsed_ms,
                                    },
                                );
                                let _ = tts_events.send(TtsEvent::FirstAudio { elapsed_ms });
                            } else {
                            let _ = tts_events.send(event);
                            }
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

fn sanitize_tts_fragment(text: &str) -> String {
    text.chars()
        .filter(|ch| !matches!(ch, '*' | '`' | '#'))
        .collect()
}

fn receiver_stream<T: Send + 'static>(rx: UnboundedReceiver<T>) -> BoxStream<'static, T> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    }))
}

pub struct FakeVoiceTransport {
    input_rx: UnboundedReceiver<VoiceTransportEvent>,
    tts_tx: UnboundedSender<TtsEvent>,
}

pub struct FakeVoiceTransportHandle {
    input_tx: Option<UnboundedSender<VoiceTransportEvent>>,
    tts_rx: UnboundedReceiver<TtsEvent>,
}

impl FakeVoiceTransport {
    pub fn new() -> (Self, FakeVoiceTransportHandle) {
        let (input_tx, input_rx) = unbounded_channel();
        let (tts_tx, tts_rx) = unbounded_channel();

        (
            Self { input_rx, tts_tx },
            FakeVoiceTransportHandle {
                input_tx: Some(input_tx),
                tts_rx,
            },
        )
    }
}

impl VoiceTransport for FakeVoiceTransport {
    fn next_event(&mut self) -> BoxFuture<'_, Option<VoiceTransportEvent>> {
        Box::pin(async move { self.input_rx.recv().await })
    }

    fn tts_events(&self) -> UnboundedSender<TtsEvent> {
        self.tts_tx.clone()
    }
}

impl FakeVoiceTransportHandle {
    pub fn send_speech_started(&self) -> std::result::Result<(), VoiceTransportError> {
        self.input_tx
            .as_ref()
            .ok_or(VoiceTransportError::Closed)?
            .send(VoiceTransportEvent::SpeechStarted)
            .map_err(|_| VoiceTransportError::Closed)
    }

    pub fn send_audio_turn(
        &self,
        chunks: Vec<AudioChunk>,
    ) -> std::result::Result<(), VoiceTransportError> {
        self.send_audio_stream(Box::pin(futures_util::stream::iter(chunks)))
    }

    pub fn send_audio_stream(
        &self,
        audio: BoxStream<'static, AudioChunk>,
    ) -> std::result::Result<(), VoiceTransportError> {
        self.input_tx
            .as_ref()
            .ok_or(VoiceTransportError::Closed)?
            .send(VoiceTransportEvent::AudioTurn(audio))
            .map_err(|_| VoiceTransportError::Closed)
    }

    pub fn close(&mut self) -> std::result::Result<(), VoiceTransportError> {
        let Some(input_tx) = self.input_tx.take() else {
            return Err(VoiceTransportError::Closed);
        };
        input_tx
            .send(VoiceTransportEvent::Closed)
            .map_err(|_| VoiceTransportError::Closed)
    }

    pub fn disconnect(&mut self) {
        self.input_tx = None;
    }

    pub async fn recv_tts_event(&mut self) -> Option<TtsEvent> {
        self.tts_rx.recv().await
    }

    pub fn drain_tts_events(&mut self) -> Vec<TtsEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.tts_rx.try_recv() {
            events.push(event);
        }
        events
    }
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
        let (voice_tx, mut voice_rx) = mpsc::unbounded_channel();
        let audio = Box::pin(futures_util::stream::iter(vec![
            AudioChunk::new("hello").with_elapsed_ms(3),
            AudioChunk::new(" runtime").with_elapsed_ms(9),
        ]));

        let output = pipeline
            .run_audio_turn_with_options(
                &mut context,
                audio,
                VoiceTurnOptions::default()
                    .with_tts_events(tts_tx)
                    .with_voice_events(voice_tx),
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

        let voice_events = drain_voice_events(&mut voice_rx);
        assert!(voice_events.iter().any(|event| matches!(
            event,
            VoiceEvent::SttPartial {
                transcript_chars: 5,
                stt_elapsed_ms: 3,
                ..
            }
        )));
        assert!(voice_events.iter().any(|event| matches!(
            event,
            VoiceEvent::SttFinal {
                transcript_chars: 13,
                stt_elapsed_ms: 9,
                ..
            }
        )));
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::LlmFirstToken { .. }))
        );
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::TtsFirstAudio { .. }))
        );
        assert!(matches!(
            voice_events.last(),
            Some(VoiceEvent::VoiceTurnCompleted {
                output_chars: 8,
                ..
            })
        ));
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
        let (voice_tx, mut voice_rx) = mpsc::unbounded_channel();
        let audio = receiver_stream(audio_rx);

        let result = {
            let turn = pipeline.run_audio_turn_with_options(
                &mut context,
                audio,
                VoiceTurnOptions::new(cancellation_token.clone()).with_voice_events(voice_tx),
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
        assert!(matches!(
            drain_voice_events(&mut voice_rx).last(),
            Some(VoiceEvent::VoiceTurnCancelled { .. })
        ));
    }

    #[tokio::test]
    async fn voice_session_barge_in_cancels_active_turn_and_completes_next_turn() {
        let provider = InterruptibleProvider::new();
        let stt = FakeSpeechToText::new();
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let (voice_tx, mut voice_rx) = mpsc::unbounded_channel();
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            stt,
            tts,
            ConversationContext::new(),
        )
        .with_voice_events(voice_tx);
        let (first_tts_tx, mut first_tts_rx) = mpsc::unbounded_channel();

        session
            .start_audio_turn(audio_stream_from_text("first request"), first_tts_tx)
            .await
            .expect("first turn should start");
        assert!(session.has_active_turn());

        let first_chunk = next_audio_chunk(&mut first_tts_rx).await;
        assert_eq!(first_chunk, b"fake-tts:0:first partial".to_vec());

        let (second_tts_tx, mut second_tts_rx) = mpsc::unbounded_channel();
        session
            .start_audio_turn(audio_stream_from_text("second request"), second_tts_tx)
            .await
            .expect("second turn should interrupt first and start");

        let output = timeout(Duration::from_secs(1), session.finish_active_turn())
            .await
            .expect("second turn should finish")
            .expect("second turn should succeed")
            .expect("second turn should be active");

        assert_eq!(output.text, "second complete");
        assert_eq!(
            drain_audio_chunks(&mut second_tts_rx),
            vec![
                b"fake-tts:0:second ".to_vec(),
                b"fake-tts:1:complete".to_vec()
            ]
        );
        assert!(!session.has_active_turn());
        assert_eq!(tts_probe.active_stream_count(), 0);

        let context = session.context_snapshot().await;
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.content != "first partial"),
            "interrupted assistant text should not be committed: {:#?}",
            context.messages()
        );
        assert_eq!(
            context
                .messages()
                .iter()
                .filter(|message| message.role == "user")
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>(),
            vec!["first request", "second request"]
        );
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| (message.role.as_str(), message.content.as_str())),
            Some(("assistant", "second complete"))
        );

        let voice_events = drain_voice_events(&mut voice_rx);
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::VoiceTurnInterrupted { .. })),
            "barge-in should emit an interruption event: {voice_events:#?}"
        );
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::VoiceTurnCancelled { .. })),
            "cancelled interrupted turn should be observable: {voice_events:#?}"
        );
        assert!(matches!(
            voice_events.last(),
            Some(VoiceEvent::VoiceTurnCompleted {
                output_chars: 15,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn voice_session_runs_normal_turn_from_fake_transport() {
        let provider = StaticTokenProvider::new(vec!["transport ", "ok"]);
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            FakeSpeechToText::new(),
            FakeTextToSpeech::new(),
            ConversationContext::new(),
        );
        let (transport, mut handle) = FakeVoiceTransport::new();

        let result = {
            let run = session.run_transport(transport);
            tokio::pin!(run);

            handle
                .send_audio_turn(vec![AudioChunk::new("transport input").with_elapsed_ms(12)])
                .expect("transport input should send");
            handle.close().expect("transport should close gracefully");

            timeout(Duration::from_secs(1), &mut run)
                .await
                .expect("transport runner should finish")
        };

        result.expect("transport runner should succeed");
        assert_eq!(
            audio_chunks(&handle.drain_tts_events()),
            vec![b"fake-tts:0:transport ".to_vec(), b"fake-tts:1:ok".to_vec()]
        );

        let context = session.context_snapshot().await;
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| (message.role.as_str(), message.content.as_str())),
            Some(("assistant", "transport ok"))
        );
    }

    #[tokio::test]
    async fn fake_transport_barge_in_cancels_playback_and_runs_next_turn() {
        let provider = InterruptibleProvider::new();
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let (voice_tx, mut voice_rx) = mpsc::unbounded_channel();
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            FakeSpeechToText::new(),
            tts,
            ConversationContext::new(),
        )
        .with_voice_events(voice_tx);
        let (transport, mut handle) = FakeVoiceTransport::new();

        let result = {
            let run = session.run_transport(transport);
            tokio::pin!(run);

            handle
                .send_audio_turn(vec![AudioChunk::new("first request").with_elapsed_ms(1)])
                .expect("first input should send");
            let first_chunk = tokio::select! {
                chunk = next_transport_audio_chunk(&mut handle) => chunk,
                result = &mut run => panic!("transport runner finished before first playback: {result:?}"),
            };
            assert_eq!(first_chunk, b"fake-tts:0:first partial".to_vec());

            handle
                .send_audio_turn(vec![AudioChunk::new("second request").with_elapsed_ms(2)])
                .expect("second input should send");
            handle.close().expect("transport should close gracefully");

            timeout(Duration::from_secs(1), &mut run)
                .await
                .expect("transport runner should finish")
        };

        result.expect("transport runner should succeed");
        assert_eq!(tts_probe.active_stream_count(), 0);

        let context = session.context_snapshot().await;
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.content != "first partial"),
            "interrupted assistant text should not be committed: {:#?}",
            context.messages()
        );
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| (message.role.as_str(), message.content.as_str())),
            Some(("assistant", "second complete"))
        );

        let voice_events = drain_voice_events(&mut voice_rx);
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::VoiceTurnInterrupted { .. })),
            "transport barge-in should emit interruption: {voice_events:#?}"
        );
    }

    #[tokio::test]
    async fn fake_transport_speech_started_cancels_active_turn_before_audio_turn() {
        let provider = TokenThenHangingProvider {
            token: "partial".to_string(),
        };
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let (voice_tx, mut voice_rx) = mpsc::unbounded_channel();
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            FakeSpeechToText::new(),
            tts,
            ConversationContext::new(),
        )
        .with_voice_events(voice_tx);
        let (transport, mut handle) = FakeVoiceTransport::new();

        let result = {
            let run = session.run_transport(transport);
            tokio::pin!(run);

            handle
                .send_audio_turn(vec![AudioChunk::new("first request").with_elapsed_ms(1)])
                .expect("first input should send");
            let first_chunk = tokio::select! {
                chunk = next_transport_audio_chunk(&mut handle) => chunk,
                result = &mut run => panic!("transport runner finished before first playback: {result:?}"),
            };
            assert_eq!(first_chunk, b"fake-tts:0:partial".to_vec());

            handle
                .send_speech_started()
                .expect("speech start should send");
            handle.close().expect("transport should close gracefully");

            timeout(Duration::from_secs(1), &mut run)
                .await
                .expect("transport runner should finish")
        };

        result.expect("speech start should cancel active turn without surfacing an error");
        assert_eq!(tts_probe.active_stream_count(), 0);

        let context = session.context_snapshot().await;
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.content != "partial"),
            "interrupted assistant text should not be committed: {:#?}",
            context.messages()
        );

        let voice_events = drain_voice_events(&mut voice_rx);
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::VoiceTurnInterrupted { .. })),
            "early speech start should emit interruption: {voice_events:#?}"
        );
        assert!(
            voice_events
                .iter()
                .any(|event| matches!(event, VoiceEvent::VoiceTurnCancelled { .. })),
            "early speech start should cancel active turn: {voice_events:#?}"
        );
    }

    #[tokio::test]
    async fn fake_transport_skips_missing_transcript_segment_and_runs_next_turn() {
        let provider = StaticTokenProvider::new(vec!["valid ", "turn"]);
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            FakeSpeechToText::new(),
            FakeTextToSpeech::new(),
            ConversationContext::new(),
        );
        let (transport, mut handle) = FakeVoiceTransport::new();

        let result = {
            let run = session.run_transport(transport);
            tokio::pin!(run);

            handle
                .send_audio_turn(Vec::new())
                .expect("empty transport segment should send");
            handle
                .send_audio_turn(vec![AudioChunk::new("real input").with_elapsed_ms(10)])
                .expect("valid transport segment should send");
            handle.close().expect("transport should close gracefully");

            timeout(Duration::from_secs(1), &mut run)
                .await
                .expect("transport runner should finish")
        };

        result.expect("empty no-transcript segment should not kill transport");

        let context = session.context_snapshot().await;
        assert_eq!(
            context
                .messages()
                .last()
                .map(|message| (message.role.as_str(), message.content.as_str())),
            Some(("assistant", "valid turn"))
        );
    }

    #[tokio::test]
    async fn dropped_fake_transport_cancels_active_turn_cleanly() {
        let provider = TokenThenHangingProvider {
            token: "partial".to_string(),
        };
        let tts = FakeTextToSpeech::new();
        let tts_probe = tts.clone();
        let mut session = VoiceSession::new(
            provider,
            ToolRegistry::new(),
            RuntimeConfig::new("test"),
            FakeSpeechToText::new(),
            tts,
            ConversationContext::new(),
        );
        let (transport, mut handle) = FakeVoiceTransport::new();

        let result = {
            let run = session.run_transport(transport);
            tokio::pin!(run);

            handle
                .send_audio_turn(vec![AudioChunk::new("dropped input").with_elapsed_ms(1)])
                .expect("input should send");
            let first_chunk = tokio::select! {
                chunk = next_transport_audio_chunk(&mut handle) => chunk,
                result = &mut run => panic!("transport runner finished before active turn: {result:?}"),
            };
            assert_eq!(first_chunk, b"fake-tts:0:partial".to_vec());

            handle.disconnect();

            timeout(Duration::from_secs(1), &mut run)
                .await
                .expect("transport runner should stop after disconnect")
        };

        result.expect("disconnect should not surface as an error");
        assert!(!session.has_active_turn());
        assert_eq!(tts_probe.active_stream_count(), 0);

        let context = session.context_snapshot().await;
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.role != "assistant"),
            "partial assistant output should not be committed: {:#?}",
            context.messages()
        );
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

    #[test]
    fn voice_latency_recorder_summarizes_nearest_rank_percentiles() {
        let mut recorder = VoiceLatencyRecorder::new();

        for elapsed_ms in [30, 10, 40, 20] {
            recorder.record(VoiceLatencyMetric::LlmFirstToken, elapsed_ms);
        }

        assert_eq!(
            recorder.snapshot().llm_first_token,
            Some(VoiceLatencyStats {
                samples: 4,
                p50_ms: 20,
                p95_ms: 40,
            })
        );
    }

    #[test]
    fn voice_latency_recorder_ingests_voice_events() {
        let mut recorder = VoiceLatencyRecorder::new();

        for event in [
            VoiceEvent::SttPartial {
                elapsed_ms: 5,
                stt_elapsed_ms: 5,
                transcript_chars: 2,
            },
            VoiceEvent::SttFinal {
                elapsed_ms: 15,
                stt_elapsed_ms: 12,
                transcript_chars: 8,
            },
            VoiceEvent::LlmFirstToken {
                elapsed_ms: 30,
                runtime_elapsed_ms: 18,
            },
            VoiceEvent::TtsFirstAudio {
                elapsed_ms: 45,
                tts_elapsed_ms: 9,
            },
            VoiceEvent::VoiceTurnCompleted {
                elapsed_ms: 70,
                output_chars: 11,
            },
        ] {
            recorder.record_event(&event);
        }

        assert_eq!(
            recorder.snapshot(),
            VoiceLatencySnapshot {
                stt_finalization: Some(VoiceLatencyStats {
                    samples: 1,
                    p50_ms: 12,
                    p95_ms: 12,
                }),
                llm_first_token: Some(VoiceLatencyStats {
                    samples: 1,
                    p50_ms: 18,
                    p95_ms: 18,
                }),
                tts_first_audio: Some(VoiceLatencyStats {
                    samples: 1,
                    p50_ms: 9,
                    p95_ms: 9,
                }),
                voice_turn_round_trip: Some(VoiceLatencyStats {
                    samples: 1,
                    p50_ms: 70,
                    p95_ms: 70,
                }),
            }
        );
    }

    #[test]
    fn tts_sanitizer_drops_markdown_formatting_markers() {
        assert_eq!(
            sanitize_tts_fragment("***In a nutshell***: `async` lets Rust #tasks run."),
            "In a nutshell: async lets Rust tasks run."
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
                TtsEvent::PlaybackReset => {}
                TtsEvent::Done => panic!("TTS stream completed before audio chunk"),
            }
        }
    }

    async fn next_transport_audio_chunk(handle: &mut FakeVoiceTransportHandle) -> Vec<u8> {
        loop {
            match timeout(Duration::from_secs(1), handle.recv_tts_event())
                .await
                .expect("transport audio event should arrive")
                .expect("transport audio stream should still be open")
            {
                TtsEvent::AudioChunk { bytes, .. } => return bytes,
                TtsEvent::FirstAudio { .. } => {}
                TtsEvent::PlaybackReset => {}
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

    fn drain_voice_events(rx: &mut UnboundedReceiver<VoiceEvent>) -> Vec<VoiceEvent> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    fn audio_stream_from_text(text: &'static str) -> BoxStream<'static, AudioChunk> {
        Box::pin(futures_util::stream::iter(vec![
            AudioChunk::new(text).with_elapsed_ms(1),
        ]))
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

    struct InterruptibleProvider {
        request_count: Arc<AtomicUsize>,
    }

    impl InterruptibleProvider {
        fn new() -> Self {
            Self {
                request_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl LlmProvider for InterruptibleProvider {
        fn stream_chat(&self, _req: ChatRequest) -> BoxStream<'static, LlmResult<TokenEvent>> {
            match self.request_count.fetch_add(1, Ordering::SeqCst) {
                0 => Box::pin(TokenThenHangingStream {
                    token: Some("first partial".to_string()),
                }),
                _ => Box::pin(futures_util::stream::iter(vec![
                    Ok(TokenEvent::Token {
                        text: "second ".to_string(),
                    }),
                    Ok(TokenEvent::Token {
                        text: "complete".to_string(),
                    }),
                    Ok(TokenEvent::Done),
                ])),
            }
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
