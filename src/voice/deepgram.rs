use std::{
    env,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use futures_core::{Stream, stream::BoxStream};
use futures_util::{SinkExt, StreamExt};
use reqwest::{Client, StatusCode, header};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::mpsc,
    task::{JoinError, JoinHandle},
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{
        Message,
        client::IntoClientRequest,
        http::{HeaderValue, header::AUTHORIZATION},
    },
};

use super::{
    AudioChunk, SpeechToText, SttError, SttResult, SttTimingBreakdown, TextToSpeech,
    TranscriptEvent, TtsError, TtsEvent, trace_voice_stt, voice_stt_trace_enabled,
};

static ACTIVE_STT_STREAMS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_TTS_STREAMS: AtomicUsize = AtomicUsize::new(0);

const DEFAULT_LISTEN_URL: &str = "wss://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true&interim_results=true&encoding=linear16&sample_rate=16000&channels=1&endpointing=300";
const DEFAULT_SPEAK_URL: &str = "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&container=none&sample_rate=24000";
const DEFAULT_TTS_MAX_CHARS: usize = 1_800;

pub fn active_deepgram_stt_stream_count() -> usize {
    ACTIVE_STT_STREAMS.load(Ordering::SeqCst)
}

pub fn active_deepgram_tts_stream_count() -> usize {
    ACTIVE_TTS_STREAMS.load(Ordering::SeqCst)
}

#[derive(Clone)]
pub struct DeepgramConfig {
    pub api_key: String,
    pub listen_url: String,
    pub speak_url: String,
    pub timeout: Duration,
    pub stream_buffer_capacity: usize,
    pub tts_max_chars: usize,
}

impl DeepgramConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            listen_url: DEFAULT_LISTEN_URL.to_string(),
            speak_url: DEFAULT_SPEAK_URL.to_string(),
            timeout: Duration::from_secs(30),
            stream_buffer_capacity: 8,
            tts_max_chars: DEFAULT_TTS_MAX_CHARS,
        }
    }

    pub fn from_env() -> std::result::Result<Self, DeepgramConfigError> {
        let api_key =
            env::var("DEEPGRAM_API_KEY").map_err(|_| DeepgramConfigError::MissingApiKey)?;
        let mut config = Self::new(api_key);

        if let Ok(listen_url) = env::var("DEEPGRAM_LISTEN_URL") {
            config.listen_url = listen_url;
        }

        if let Ok(speak_url) = env::var("DEEPGRAM_SPEAK_URL") {
            config.speak_url = speak_url;
        }

        if let Ok(max_chars) = env::var("DEEPGRAM_TTS_MAX_CHARS") {
            config.tts_max_chars = max_chars
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| DeepgramConfigError::InvalidTtsMaxChars {
                    value: max_chars.clone(),
                })?;
        }

        Ok(config)
    }

    pub fn with_listen_url(mut self, listen_url: impl Into<String>) -> Self {
        self.listen_url = listen_url.into();
        self
    }

    pub fn with_speak_url(mut self, speak_url: impl Into<String>) -> Self {
        self.speak_url = speak_url.into();
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeepgramConfigError {
    #[error("DEEPGRAM_API_KEY must be set")]
    MissingApiKey,

    #[error("DEEPGRAM_TTS_MAX_CHARS must be a positive integer, got {value:?}")]
    InvalidTtsMaxChars { value: String },
}

#[derive(Clone)]
pub struct DeepgramSpeechToText {
    config: Arc<DeepgramConfig>,
}

impl DeepgramSpeechToText {
    pub fn new(config: DeepgramConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

impl SpeechToText for DeepgramSpeechToText {
    fn stream_audio(
        &self,
        audio: BoxStream<'static, AudioChunk>,
    ) -> BoxStream<'static, SttResult<TranscriptEvent>> {
        let capacity = self.config.stream_buffer_capacity.max(1);
        let (tx, rx) = mpsc::channel(capacity);
        let config = Arc::clone(&self.config);

        let lease = ActiveStreamLease::new(&ACTIVE_STT_STREAMS);
        let task_lease = lease.clone();
        let handle = tokio::spawn(async move {
            let _lease = task_lease;
            if let Err(err) = run_listen_stream(config, audio, tx.clone()).await {
                let _ = tx.send(Err(err)).await;
            }
        });

        Box::pin(DeepgramSttStream { rx, handle, lease })
    }
}

struct DeepgramSttStream {
    rx: mpsc::Receiver<SttResult<TranscriptEvent>>,
    handle: JoinHandle<()>,
    lease: ActiveStreamLease,
}

impl Stream for DeepgramSttStream {
    type Item = SttResult<TranscriptEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        Pin::new(&mut this.rx).poll_recv(cx)
    }
}

impl Drop for DeepgramSttStream {
    fn drop(&mut self) {
        self.handle.abort();
        self.lease.release();
    }
}

#[derive(Clone)]
pub struct DeepgramTextToSpeech {
    client: Client,
    config: Arc<DeepgramConfig>,
}

impl DeepgramTextToSpeech {
    pub fn new(config: DeepgramConfig) -> Self {
        Self {
            client: Client::new(),
            config: Arc::new(config),
        }
    }
}

impl TextToSpeech for DeepgramTextToSpeech {
    fn stream_text(
        &self,
        text: BoxStream<'static, String>,
    ) -> BoxStream<'static, super::Result<TtsEvent>> {
        let capacity = self.config.stream_buffer_capacity.max(1);
        let (tx, rx) = mpsc::channel(capacity);
        let client = self.client.clone();
        let config = Arc::clone(&self.config);

        let lease = ActiveStreamLease::new(&ACTIVE_TTS_STREAMS);
        let task_lease = lease.clone();
        let handle = tokio::spawn(async move {
            let _lease = task_lease;
            if let Err(err) = run_speak_stream(client, config, text, tx.clone()).await {
                let _ = tx.send(Err(err)).await;
            }
        });

        Box::pin(DeepgramTtsStream { rx, handle, lease })
    }
}

struct DeepgramTtsStream {
    rx: mpsc::Receiver<super::Result<TtsEvent>>,
    handle: JoinHandle<()>,
    lease: ActiveStreamLease,
}

impl Stream for DeepgramTtsStream {
    type Item = super::Result<TtsEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        Pin::new(&mut this.rx).poll_recv(cx)
    }
}

impl Drop for DeepgramTtsStream {
    fn drop(&mut self) {
        self.handle.abort();
        self.lease.release();
    }
}

#[derive(Clone)]
struct ActiveStreamLease {
    counter: &'static AtomicUsize,
    released: Arc<AtomicBool>,
}

impl ActiveStreamLease {
    fn new(counter: &'static AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self {
            counter,
            released: Arc::new(AtomicBool::new(false)),
        }
    }

    fn release(&self) {
        if !self.released.swap(true, Ordering::SeqCst) {
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl Drop for ActiveStreamLease {
    fn drop(&mut self) {
        self.release();
    }
}

async fn run_listen_stream(
    config: Arc<DeepgramConfig>,
    audio: BoxStream<'static, AudioChunk>,
    tx: mpsc::Sender<SttResult<TranscriptEvent>>,
) -> SttResult<()> {
    let started = Instant::now();
    let request = listen_request(&config)?;
    let (socket, _) = connect_async(request)
        .await
        .map_err(|err| SttError::Backend(format!("Deepgram WebSocket connect failed: {err}")))?;
    let (mut writer, mut reader) = socket.split();
    let trace = voice_stt_trace_enabled();
    let connect_ms = started.elapsed().as_millis();
    trace_stt(
        trace,
        &format!("deepgram_connected elapsed_ms={connect_ms}"),
    );
    let sent_timing = Arc::new(Mutex::new(SentAudioTiming::default()));
    record_deepgram_connected(&sent_timing, connect_ms);
    let sender_timing = Arc::clone(&sent_timing);

    let sender = tokio::spawn(async move {
        let mut audio = audio;
        while let Some(chunk) = audio.next().await {
            if chunk.bytes.is_empty() {
                continue;
            }

            let sent_at = Instant::now();
            let bytes = chunk.bytes;
            let audio_elapsed_ms = chunk.elapsed_ms;
            let is_speech = chunk.is_speech;
            writer
                .send(Message::Binary(bytes.into()))
                .await
                .map_err(|err| SttError::Backend(format!("Deepgram audio send failed: {err}")))?;
            record_sent_audio(
                &sender_timing,
                started,
                sent_at,
                audio_elapsed_ms,
                is_speech,
                trace,
            );
        }

        record_audio_input_ended(&sender_timing, started.elapsed().as_millis());
        trace_stt(
            trace,
            &format!(
                "deepgram_audio_input_ended elapsed_ms={}",
                started.elapsed().as_millis()
            ),
        );
        let finalize_sent_at = Instant::now();
        writer
            .send(Message::Text(r#"{"type":"Finalize"}"#.into()))
            .await
            .map_err(|err| SttError::Backend(format!("Deepgram finalize send failed: {err}")))?;
        record_finalize_sent(
            &sender_timing,
            started.elapsed().as_millis(),
            finalize_sent_at,
        );
        trace_stt(
            trace,
            &format!(
                "deepgram_finalize_sent elapsed_ms={}",
                started.elapsed().as_millis()
            ),
        );
        writer
            .send(Message::Text(r#"{"type":"CloseStream"}"#.into()))
            .await
            .map_err(|err| SttError::Backend(format!("Deepgram close send failed: {err}")))?;
        SttResult::Ok(())
    });

    let mut state = DeepgramTranscriptState::default();
    let mut first_interim_seen = false;
    while let Some(message) = reader.next().await {
        match message
            .map_err(|err| SttError::Backend(format!("Deepgram WebSocket read failed: {err}")))?
        {
            Message::Text(text) => {
                let elapsed_ms = stt_elapsed_ms(&sent_timing, started);
                trace_deepgram_message(
                    &sent_timing,
                    &text,
                    elapsed_ms,
                    &mut first_interim_seen,
                    trace,
                );

                for event in state.ingest(&text, elapsed_ms)? {
                    send_transcript_event(&tx, &sent_timing, event, trace, "deepgram_message")
                        .await?;
                }
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Binary(_) => {}
        }
    }

    await_stt_sender(sender).await?;

    if let Some(event) = state.finish(stt_elapsed_ms(&sent_timing, started)) {
        send_transcript_event(&tx, &sent_timing, event, trace, "stream_finish").await?;
    }

    let _ = tx.send(Ok(TranscriptEvent::Done)).await;
    Ok(())
}

#[derive(Default)]
struct SentAudioTiming {
    first: Option<SentAudioSample>,
    latest: Option<SentAudioSample>,
    deepgram_connect_ms: Option<u128>,
    first_interim_ms: Option<u128>,
    audio_input_ended_ms: Option<u128>,
    finalize_sent_at: Option<Instant>,
    finalize_sent_ms: Option<u128>,
    speech_audio_ms: u128,
    total_silence_ms: u128,
    trailing_silence_ms: u128,
}

#[derive(Clone, Copy)]
struct SentAudioSample {
    sent_at: Instant,
    audio_elapsed_ms: u128,
}

fn record_sent_audio(
    sent_timing: &Mutex<SentAudioTiming>,
    stream_started_at: Instant,
    sent_at: Instant,
    audio_elapsed_ms: u128,
    is_speech: Option<bool>,
    trace: bool,
) {
    let mut timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    let sample = SentAudioSample {
        sent_at,
        audio_elapsed_ms,
    };

    if timing.first.is_none() {
        timing.first = Some(sample);
        trace_stt(
            trace,
            &format!(
                "deepgram_first_audio_sent stream_elapsed_ms={} audio_elapsed_ms={audio_elapsed_ms}",
                stream_started_at.elapsed().as_millis()
            ),
        );
    }

    let previous_audio_elapsed_ms = timing
        .latest
        .map(|latest| latest.audio_elapsed_ms)
        .unwrap_or(0);
    let chunk_elapsed_ms = audio_elapsed_ms.saturating_sub(previous_audio_elapsed_ms);
    match is_speech {
        Some(true) => {
            timing.speech_audio_ms += chunk_elapsed_ms;
            timing.trailing_silence_ms = 0;
        }
        Some(false) => {
            timing.total_silence_ms += chunk_elapsed_ms;
            timing.trailing_silence_ms += chunk_elapsed_ms;
        }
        None => {}
    }

    timing.latest = Some(sample);
}

fn record_deepgram_connected(sent_timing: &Mutex<SentAudioTiming>, connect_ms: u128) {
    let mut timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    timing.deepgram_connect_ms = Some(connect_ms);
}

fn record_audio_input_ended(sent_timing: &Mutex<SentAudioTiming>, input_ended_ms: u128) {
    let mut timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    timing.audio_input_ended_ms = Some(input_ended_ms);
}

fn record_finalize_sent(
    sent_timing: &Mutex<SentAudioTiming>,
    finalize_sent_ms: u128,
    finalize_sent_at: Instant,
) {
    let mut timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    timing.finalize_sent_ms = Some(finalize_sent_ms);
    timing.finalize_sent_at = Some(finalize_sent_at);
}

fn record_first_interim(sent_timing: &Mutex<SentAudioTiming>, first_interim_ms: u128) {
    let mut timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    if timing.first_interim_ms.is_none() {
        timing.first_interim_ms = Some(first_interim_ms);
    }
}

fn stt_elapsed_ms(sent_timing: &Mutex<SentAudioTiming>, fallback_started: Instant) -> u128 {
    let timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    match timing.latest {
        Some(sample) => sample.audio_elapsed_ms + sample.sent_at.elapsed().as_millis(),
        None => fallback_started.elapsed().as_millis(),
    }
}

fn trace_stt(enabled: bool, message: &str) {
    if enabled {
        trace_voice_stt(message);
    }
}

fn trace_deepgram_message(
    sent_timing: &Mutex<SentAudioTiming>,
    data: &str,
    elapsed_ms: u128,
    first_interim_seen: &mut bool,
    trace: bool,
) {
    let Ok(message) = serde_json::from_str::<DeepgramListenMessage>(data) else {
        return;
    };
    if !message.is_results() {
        return;
    }
    let Some(transcript) = message.transcript() else {
        return;
    };
    if transcript.is_empty() {
        return;
    }

    let is_final = message.is_final.unwrap_or(false);
    let speech_final = message.speech_final.unwrap_or(false);
    if !is_final && !*first_interim_seen {
        *first_interim_seen = true;
        record_first_interim(sent_timing, elapsed_ms);
        trace_stt(
            trace,
            &format!(
                "deepgram_first_interim elapsed_ms={elapsed_ms} transcript_chars={}",
                transcript.len()
            ),
        );
    }
    if speech_final {
        trace_stt(
            trace,
            &format!(
                "deepgram_speech_final elapsed_ms={elapsed_ms} is_final={is_final} transcript_chars={}",
                transcript.len()
            ),
        );
    }
}

async fn send_transcript_event(
    tx: &mpsc::Sender<SttResult<TranscriptEvent>>,
    sent_timing: &Mutex<SentAudioTiming>,
    event: TranscriptEvent,
    trace: bool,
    source: &str,
) -> SttResult<()> {
    if let TranscriptEvent::Final { elapsed_ms, .. } = &event {
        let breakdown = stt_timing_breakdown(sent_timing, *elapsed_ms);
        trace_stt(
            trace,
            &format!(
                "stt_breakdown speech_audio_ms={} vad_silence_ms={} total_silence_ms={} audio_duration_ms={} deepgram_connect_ms={} first_interim_ms={} finalize_to_final_ms={} final_emitted_ms={}",
                breakdown.speech_audio_ms,
                breakdown.vad_silence_ms,
                breakdown.total_silence_ms,
                breakdown.audio_duration_ms,
                optional_ms(breakdown.deepgram_connect_ms),
                optional_ms(breakdown.first_interim_ms),
                optional_ms(breakdown.finalize_to_final_ms),
                breakdown.final_emitted_ms
            ),
        );
        if tx
            .send(Ok(TranscriptEvent::Breakdown(breakdown)))
            .await
            .is_err()
        {
            return Ok(());
        }
    }

    trace_transcript_event(trace, &event, source);
    let _ = tx.send(Ok(event)).await;
    Ok(())
}

fn stt_timing_breakdown(
    sent_timing: &Mutex<SentAudioTiming>,
    final_emitted_ms: u128,
) -> SttTimingBreakdown {
    let timing = sent_timing
        .lock()
        .expect("Deepgram STT sent-audio timing mutex poisoned");
    SttTimingBreakdown {
        speech_audio_ms: timing.speech_audio_ms,
        vad_silence_ms: timing.trailing_silence_ms,
        total_silence_ms: timing.total_silence_ms,
        audio_duration_ms: timing
            .latest
            .map(|latest| latest.audio_elapsed_ms)
            .unwrap_or(0),
        deepgram_connect_ms: timing.deepgram_connect_ms,
        first_interim_ms: timing.first_interim_ms,
        finalize_to_final_ms: timing
            .finalize_sent_at
            .map(|finalize_sent_at| finalize_sent_at.elapsed().as_millis()),
        final_emitted_ms,
    }
}

fn optional_ms(value: Option<u128>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn trace_transcript_event(enabled: bool, event: &TranscriptEvent, source: &str) {
    if !enabled {
        return;
    }
    if let TranscriptEvent::Final { text, elapsed_ms } = event {
        trace_stt(
            true,
            &format!(
                "stt_final_emitted elapsed_ms={elapsed_ms} source={source} transcript_chars={}",
                text.len()
            ),
        );
    }
}

fn listen_request(
    config: &DeepgramConfig,
) -> SttResult<tokio_tungstenite::tungstenite::http::Request<()>> {
    let mut request = config
        .listen_url
        .as_str()
        .into_client_request()
        .map_err(|err| SttError::Backend(format!("invalid Deepgram listen URL: {err}")))?;
    let auth = HeaderValue::from_str(&format!("Token {}", config.api_key))
        .map_err(|err| SttError::Backend(format!("invalid Deepgram auth header: {err}")))?;
    request.headers_mut().insert(AUTHORIZATION, auth);
    Ok(request)
}

async fn await_stt_sender(handle: JoinHandle<SttResult<()>>) -> SttResult<()> {
    match handle.await {
        Ok(result) => result,
        Err(err) => Err(join_error_to_stt(err)),
    }
}

fn join_error_to_stt(err: JoinError) -> SttError {
    if err.is_cancelled() {
        SttError::Backend("Deepgram audio sender task was cancelled".to_string())
    } else {
        SttError::Backend(format!("Deepgram audio sender task failed: {err}"))
    }
}

#[derive(Default)]
struct DeepgramTranscriptState {
    final_segments: Vec<String>,
    latest_partial: Option<String>,
    turn_final_emitted: bool,
}

impl DeepgramTranscriptState {
    fn ingest(&mut self, data: &str, elapsed_ms: u128) -> SttResult<Vec<TranscriptEvent>> {
        let message: DeepgramListenMessage = serde_json::from_str(data).map_err(|err| {
            SttError::Backend(format!(
                "invalid Deepgram transcript JSON: {err}; data={data:?}"
            ))
        })?;

        if message.from_finalize.unwrap_or(false) {
            return Ok(self
                .finish(elapsed_ms)
                .into_iter()
                .collect::<Vec<TranscriptEvent>>());
        }

        if !message.is_results() {
            return Ok(Vec::new());
        }

        let Some(transcript) = message.transcript() else {
            return Ok(Vec::new());
        };

        if transcript.is_empty() {
            return Ok(Vec::new());
        }

        if message.is_final.unwrap_or(false) {
            self.final_segments.push(transcript);
            let text = join_transcript_segments(&self.final_segments, "");
            if message.speech_final.unwrap_or(false) {
                self.turn_final_emitted = true;
                Ok(vec![TranscriptEvent::Final { text, elapsed_ms }])
            } else {
                self.latest_partial = Some(text.clone());
                Ok(vec![TranscriptEvent::Partial { text, elapsed_ms }])
            }
        } else {
            let text = join_transcript_segments(&self.final_segments, &transcript);
            self.latest_partial = Some(text.clone());
            Ok(vec![TranscriptEvent::Partial { text, elapsed_ms }])
        }
    }

    fn finish(&mut self, elapsed_ms: u128) -> Option<TranscriptEvent> {
        if self.turn_final_emitted {
            return None;
        }

        let text = if self.final_segments.is_empty() {
            self.latest_partial.clone()
        } else {
            Some(join_transcript_segments(&self.final_segments, ""))
        }?;

        self.turn_final_emitted = true;
        Some(TranscriptEvent::Final { text, elapsed_ms })
    }
}

fn join_transcript_segments(final_segments: &[String], interim: &str) -> String {
    let mut text = final_segments
        .iter()
        .filter(|segment| !segment.trim().is_empty())
        .map(|segment| segment.trim())
        .collect::<Vec<_>>()
        .join(" ");

    if !interim.trim().is_empty() {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(interim.trim());
    }

    text
}

#[derive(Debug, Deserialize)]
struct DeepgramListenMessage {
    #[serde(rename = "type")]
    kind: Option<String>,
    is_final: Option<bool>,
    speech_final: Option<bool>,
    from_finalize: Option<bool>,
    channel: Option<DeepgramChannel>,
}

impl DeepgramListenMessage {
    fn is_results(&self) -> bool {
        self.kind.as_deref().is_none_or(|kind| kind == "Results")
    }

    fn transcript(&self) -> Option<String> {
        self.channel
            .as_ref()
            .and_then(|channel| channel.alternatives.first())
            .map(|alternative| alternative.transcript.clone())
    }
}

#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    transcript: String,
}

async fn run_speak_stream(
    client: Client,
    config: Arc<DeepgramConfig>,
    mut text: BoxStream<'static, String>,
    tx: mpsc::Sender<super::Result<TtsEvent>>,
) -> super::Result<()> {
    let mut input = String::new();
    while let Some(fragment) = text.next().await {
        input.push_str(&fragment);
    }

    if input.trim().is_empty() {
        let _ = tx.send(Ok(TtsEvent::Done)).await;
        return Ok(());
    }
    input = truncate_tts_input(input, config.tts_max_chars);

    let started = Instant::now();
    let response = client
        .post(&config.speak_url)
        .timeout(config.timeout)
        .header(AUTHORIZATION.as_str(), format!("Token {}", config.api_key))
        .header(header::CONTENT_TYPE, "application/json")
        .json(&DeepgramSpeakRequest { text: input })
        .send()
        .await
        .map_err(map_reqwest_tts_error)?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|err| err.to_string());
        return Err(deepgram_tts_status_error(status, body));
    }

    let mut first_audio_sent = false;
    let mut bytes = response.bytes_stream();
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk.map_err(map_reqwest_tts_error)?;
        if chunk.is_empty() {
            continue;
        }

        let elapsed_ms = started.elapsed().as_millis();
        if !first_audio_sent {
            first_audio_sent = true;
            if tx
                .send(Ok(TtsEvent::FirstAudio { elapsed_ms }))
                .await
                .is_err()
            {
                return Ok(());
            }
        }

        if tx
            .send(Ok(TtsEvent::AudioChunk {
                bytes: chunk.to_vec(),
                elapsed_ms,
            }))
            .await
            .is_err()
        {
            return Ok(());
        }
    }

    let _ = tx.send(Ok(TtsEvent::Done)).await;
    Ok(())
}

fn map_reqwest_tts_error(err: reqwest::Error) -> TtsError {
    if err.is_timeout() {
        TtsError::Backend("Deepgram TTS request timed out".to_string())
    } else {
        TtsError::Backend(format!("Deepgram TTS request failed: {err}"))
    }
}

fn deepgram_tts_status_error(status: StatusCode, body: String) -> TtsError {
    TtsError::Backend(format!("Deepgram TTS returned HTTP {status}: {body}"))
}

#[derive(Debug, Serialize)]
struct DeepgramSpeakRequest {
    text: String,
}

fn truncate_tts_input(input: String, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input;
    }

    let kept = max_chars.saturating_sub(3);
    let mut output = input.chars().take(kept).collect::<String>();
    if max_chars >= 3 {
        output.push_str("...");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::StreamExt;
    use tokio::time::{sleep, timeout};

    use super::*;

    #[test]
    fn transcript_state_maps_interim_and_speech_final_events() {
        let mut state = DeepgramTranscriptState::default();

        let events = state
            .ingest(
                r#"{"type":"Results","is_final":false,"speech_final":false,"channel":{"alternatives":[{"transcript":"hello wor"}]}}"#,
                10,
            )
            .expect("interim event should parse");
        assert_eq!(
            events,
            vec![TranscriptEvent::Partial {
                text: "hello wor".to_string(),
                elapsed_ms: 10,
            }]
        );

        let events = state
            .ingest(
                r#"{"type":"Results","is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"hello world"}]}}"#,
                30,
            )
            .expect("final event should parse");
        assert_eq!(
            events,
            vec![TranscriptEvent::Final {
                text: "hello world".to_string(),
                elapsed_ms: 30,
            }]
        );
        assert_eq!(state.finish(40), None);
    }

    #[test]
    fn transcript_state_combines_final_segments_until_speech_final() {
        let mut state = DeepgramTranscriptState::default();

        let first = state
            .ingest(
                r#"{"type":"Results","is_final":true,"speech_final":false,"channel":{"alternatives":[{"transcript":"first segment"}]}}"#,
                10,
            )
            .expect("first segment should parse");
        assert_eq!(
            first,
            vec![TranscriptEvent::Partial {
                text: "first segment".to_string(),
                elapsed_ms: 10,
            }]
        );

        let second = state
            .ingest(
                r#"{"type":"Results","is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"second segment"}]}}"#,
                20,
            )
            .expect("second segment should parse");
        assert_eq!(
            second,
            vec![TranscriptEvent::Final {
                text: "first segment second segment".to_string(),
                elapsed_ms: 20,
            }]
        );
    }

    #[test]
    fn tts_input_truncates_to_max_chars() {
        let input = "abcdef".to_string();

        assert_eq!(truncate_tts_input(input, 5), "ab...");
    }

    #[test]
    fn transcript_state_finalizes_buffer_on_stream_finish() {
        let mut state = DeepgramTranscriptState::default();

        state
            .ingest(
                r#"{"type":"Results","is_final":true,"speech_final":false,"channel":{"alternatives":[{"transcript":"buffered final"}]}}"#,
                10,
            )
            .expect("buffered final should parse");

        assert_eq!(
            state.finish(50),
            Some(TranscriptEvent::Final {
                text: "buffered final".to_string(),
                elapsed_ms: 50,
            })
        );
    }

    #[test]
    fn transcript_state_promotes_latest_interim_on_stream_finish() {
        let mut state = DeepgramTranscriptState::default();

        state
            .ingest(
                r#"{"type":"Results","is_final":false,"speech_final":false,"channel":{"alternatives":[{"transcript":"latest interim"}]}}"#,
                10,
            )
            .expect("interim should parse");

        assert_eq!(
            state.finish(50),
            Some(TranscriptEvent::Final {
                text: "latest interim".to_string(),
                elapsed_ms: 50,
            })
        );
    }

    #[tokio::test]
    async fn dropping_stt_stream_aborts_active_task() {
        let stt = DeepgramSpeechToText::new(
            DeepgramConfig::new("test-key").with_listen_url("ws://127.0.0.1:1/v1/listen"),
        );
        let (_audio_tx, audio_rx) = mpsc::unbounded_channel();
        let stream = stt.stream_audio(super::super::receiver_stream(audio_rx));

        wait_for_active_stt_streams(1).await;
        drop(stream);
        wait_for_active_stt_streams(0).await;
    }

    #[tokio::test]
    async fn dropping_tts_stream_aborts_active_task() {
        let tts = DeepgramTextToSpeech::new(
            DeepgramConfig::new("test-key").with_speak_url("http://127.0.0.1:1/v1/speak"),
        );
        let (_text_tx, text_rx) = mpsc::unbounded_channel();
        let stream = tts.stream_text(super::super::receiver_stream(text_rx));

        wait_for_active_tts_streams(1).await;
        drop(stream);
        wait_for_active_tts_streams(0).await;
    }

    #[tokio::test]
    #[ignore = "requires DEEPGRAM_API_KEY and raw 16kHz mono linear16 PCM in DEEPGRAM_STT_RAW_PCM"]
    async fn real_deepgram_stt_stream_smoke_test() {
        let config = DeepgramConfig::from_env().expect("Deepgram config should load from env");
        let path = std::env::var("DEEPGRAM_STT_RAW_PCM")
            .expect("DEEPGRAM_STT_RAW_PCM must point to raw PCM audio");
        let bytes = tokio::fs::read(path)
            .await
            .expect("raw PCM fixture should be readable");
        let stt = DeepgramSpeechToText::new(config);
        let audio = Box::pin(futures_util::stream::iter(vec![AudioChunk::new(bytes)]));

        let events = stt
            .stream_audio(audio)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<SttResult<Vec<_>>>()
            .expect("Deepgram STT stream should succeed");

        assert!(
            events
                .iter()
                .any(|event| matches!(event, TranscriptEvent::Final { .. })),
            "real STT smoke test should produce a final transcript: {events:#?}"
        );
    }

    #[tokio::test]
    #[ignore = "requires DEEPGRAM_API_KEY"]
    async fn real_deepgram_tts_stream_smoke_test() {
        let config = DeepgramConfig::from_env().expect("Deepgram config should load from env");
        let tts = DeepgramTextToSpeech::new(config);
        let text = Box::pin(futures_util::stream::iter(vec![
            "Hello from the Deepgram text to speech smoke test.".to_string(),
        ]));

        let events = tts
            .stream_text(text)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<super::super::Result<Vec<_>>>()
            .expect("Deepgram TTS stream should succeed");

        assert!(matches!(events.first(), Some(TtsEvent::FirstAudio { .. })));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, TtsEvent::AudioChunk { .. }))
        );
        assert!(matches!(events.last(), Some(TtsEvent::Done)));
    }

    async fn wait_for_active_stt_streams(expected: usize) {
        timeout(Duration::from_secs(1), async {
            while active_deepgram_stt_stream_count() != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("active STT stream count should reach expected value");
    }

    async fn wait_for_active_tts_streams(expected: usize) {
        timeout(Duration::from_secs(1), async {
            while active_deepgram_tts_stream_count() != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("active TTS stream count should reach expected value");
    }
}
