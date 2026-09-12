use std::{env, time::Duration};

use livekit_api::access_token::{AccessToken, VideoGrants};
use thiserror::Error;

#[cfg(feature = "livekit-transport")]
use {
    super::{AudioChunk, TtsEvent, VoiceTransport, VoiceTransportEvent},
    futures_core::stream::BoxStream,
    futures_util::{StreamExt, future::BoxFuture},
    livekit::{
        options::TrackPublishOptions,
        prelude::{
            LocalAudioTrack, LocalTrack, RemoteAudioTrack, RemoteTrack, Room, RoomEvent,
            RoomOptions, TrackSource,
        },
        webrtc::{
            audio_source::native::NativeAudioSource,
            audio_stream::native::NativeAudioStream,
            prelude::{AudioFrame, AudioSourceOptions, RtcAudioSource},
        },
    },
    std::{collections::VecDeque, sync::Arc},
    tokio::{
        sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
        task::JoinHandle,
    },
    tracing::warn,
};

const DEFAULT_ROOM: &str = "ai-runtime";
const DEFAULT_IDENTITY: &str = "ai-runtime-agent";
const DEFAULT_NAME: &str = "ai-runtime agent";
const DEFAULT_INPUT_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_INPUT_CHANNELS: u32 = 1;
const DEFAULT_INPUT_SPEECH_THRESHOLD: u32 = 250;
const DEFAULT_INPUT_MIN_SPEECH_MS: u64 = 120;
const DEFAULT_INPUT_SILENCE_TIMEOUT_MS: u64 = 700;
const DEFAULT_OUTPUT_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_OUTPUT_CHANNELS: u32 = 1;
const DEFAULT_OUTPUT_BUFFER_MS: u32 = 100;
const DEFAULT_OUTPUT_FRAME_MS: u32 = 20;
const DEFAULT_TOKEN_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone)]
pub struct LiveKitConfig {
    pub url: String,
    pub api_key: String,
    pub api_secret: String,
    pub room: String,
    pub identity: String,
    pub name: String,
    pub input_sample_rate: u32,
    pub input_channels: u32,
    pub input_speech_threshold: u32,
    pub input_min_speech_duration: Duration,
    pub input_silence_timeout: Duration,
    pub output_sample_rate: u32,
    pub output_channels: u32,
    pub output_buffer_ms: u32,
    pub output_frame_ms: u32,
    pub token_ttl: Duration,
}

impl LiveKitConfig {
    pub fn new(
        url: impl Into<String>,
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            room: DEFAULT_ROOM.to_string(),
            identity: DEFAULT_IDENTITY.to_string(),
            name: DEFAULT_NAME.to_string(),
            input_sample_rate: DEFAULT_INPUT_SAMPLE_RATE,
            input_channels: DEFAULT_INPUT_CHANNELS,
            input_speech_threshold: DEFAULT_INPUT_SPEECH_THRESHOLD,
            input_min_speech_duration: Duration::from_millis(DEFAULT_INPUT_MIN_SPEECH_MS),
            input_silence_timeout: Duration::from_millis(DEFAULT_INPUT_SILENCE_TIMEOUT_MS),
            output_sample_rate: DEFAULT_OUTPUT_SAMPLE_RATE,
            output_channels: DEFAULT_OUTPUT_CHANNELS,
            output_buffer_ms: DEFAULT_OUTPUT_BUFFER_MS,
            output_frame_ms: DEFAULT_OUTPUT_FRAME_MS,
            token_ttl: DEFAULT_TOKEN_TTL,
        }
    }

    pub fn from_env() -> Result<Self, LiveKitConfigError> {
        let mut config = Self::new(
            required_env("LIVEKIT_URL")?,
            required_env("LIVEKIT_API_KEY")?,
            required_env("LIVEKIT_API_SECRET")?,
        );

        config.room = optional_env("LIVEKIT_ROOM").unwrap_or_else(|| DEFAULT_ROOM.to_string());
        config.identity =
            optional_env("LIVEKIT_IDENTITY").unwrap_or_else(|| DEFAULT_IDENTITY.to_string());
        config.name = optional_env("LIVEKIT_NAME").unwrap_or_else(|| DEFAULT_NAME.to_string());
        config.input_sample_rate =
            optional_u32_env("LIVEKIT_INPUT_SAMPLE_RATE")?.unwrap_or(DEFAULT_INPUT_SAMPLE_RATE);
        config.input_channels =
            optional_u32_env("LIVEKIT_INPUT_CHANNELS")?.unwrap_or(DEFAULT_INPUT_CHANNELS);
        config.input_speech_threshold = optional_u32_env("LIVEKIT_INPUT_SPEECH_THRESHOLD")?
            .unwrap_or(DEFAULT_INPUT_SPEECH_THRESHOLD);
        config.input_min_speech_duration = Duration::from_millis(
            optional_u64_env("LIVEKIT_INPUT_MIN_SPEECH_MS")?.unwrap_or(DEFAULT_INPUT_MIN_SPEECH_MS),
        );
        config.input_silence_timeout = Duration::from_millis(
            optional_u64_env("LIVEKIT_INPUT_SILENCE_TIMEOUT_MS")?
                .unwrap_or(DEFAULT_INPUT_SILENCE_TIMEOUT_MS),
        );
        config.output_sample_rate =
            optional_u32_env("LIVEKIT_OUTPUT_SAMPLE_RATE")?.unwrap_or(DEFAULT_OUTPUT_SAMPLE_RATE);
        config.output_channels =
            optional_u32_env("LIVEKIT_OUTPUT_CHANNELS")?.unwrap_or(DEFAULT_OUTPUT_CHANNELS);
        config.output_buffer_ms =
            optional_u32_env("LIVEKIT_OUTPUT_BUFFER_MS")?.unwrap_or(DEFAULT_OUTPUT_BUFFER_MS);
        config.output_frame_ms =
            optional_u32_env("LIVEKIT_OUTPUT_FRAME_MS")?.unwrap_or(DEFAULT_OUTPUT_FRAME_MS);

        Ok(config)
    }

    pub fn with_room(mut self, room: impl Into<String>) -> Self {
        self.room = room.into();
        self
    }

    pub fn with_identity(mut self, identity: impl Into<String>) -> Self {
        self.identity = identity.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_input_audio(mut self, sample_rate: u32, channels: u32) -> Self {
        self.input_sample_rate = sample_rate;
        self.input_channels = channels;
        self
    }

    pub fn with_input_vad(
        mut self,
        speech_threshold: u32,
        min_speech_duration: Duration,
        silence_timeout: Duration,
    ) -> Self {
        self.input_speech_threshold = speech_threshold;
        self.input_min_speech_duration = min_speech_duration;
        self.input_silence_timeout = silence_timeout;
        self
    }

    pub fn with_output_audio(mut self, sample_rate: u32, channels: u32) -> Self {
        self.output_sample_rate = sample_rate;
        self.output_channels = channels;
        self
    }

    pub fn with_output_buffering(mut self, buffer_ms: u32, frame_ms: u32) -> Self {
        self.output_buffer_ms = buffer_ms;
        self.output_frame_ms = frame_ms;
        self
    }

    pub fn with_token_ttl(mut self, token_ttl: Duration) -> Self {
        self.token_ttl = token_ttl;
        self
    }

    pub fn access_token(&self) -> Result<String, LiveKitConfigError> {
        AccessToken::with_api_key(&self.api_key, &self.api_secret)
            .with_ttl(self.token_ttl)
            .with_identity(&self.identity)
            .with_name(&self.name)
            .with_grants(VideoGrants {
                room_join: true,
                room: self.room.clone(),
                can_publish: true,
                can_subscribe: true,
                ..Default::default()
            })
            .to_jwt()
            .map_err(|err| LiveKitConfigError::Token(err.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum LiveKitConfigError {
    #[error("{0} must be set")]
    MissingEnv(&'static str),

    #[error("{name} must be a positive integer, got {value:?}")]
    InvalidU32 { name: &'static str, value: String },

    #[error("{name} must be a positive integer, got {value:?}")]
    InvalidU64 { name: &'static str, value: String },

    #[error("LiveKit token generation failed: {0}")]
    Token(String),
}

#[cfg(feature = "livekit-transport")]
#[derive(Debug, Error)]
pub enum LiveKitTransportError {
    #[error(transparent)]
    Config(#[from] LiveKitConfigError),

    #[error("LiveKit room connection failed: {0}")]
    Connect(String),

    #[error("LiveKit TTS output track publish failed: {0}")]
    Publish(String),
}

#[cfg(not(feature = "livekit-transport"))]
#[derive(Debug, Error)]
pub enum LiveKitTransportError {
    #[error(
        "LiveKit transport requires building with `--features livekit-transport` and clang++ 21 or newer"
    )]
    FeatureDisabled,
}

#[cfg(feature = "livekit-transport")]
pub struct LiveKitVoiceTransport {
    room: Arc<Room>,
    input_rx: UnboundedReceiver<VoiceTransportEvent>,
    tts_tx: UnboundedSender<TtsEvent>,
    event_task: JoinHandle<()>,
    playback_task: JoinHandle<()>,
    _output_track: LocalAudioTrack,
}

#[cfg(feature = "livekit-transport")]
impl LiveKitVoiceTransport {
    pub async fn connect(config: LiveKitConfig) -> Result<Self, LiveKitTransportError> {
        let token = config.access_token()?;
        let (room, room_events) = Room::connect(&config.url, &token, RoomOptions::default())
            .await
            .map_err(|err| LiveKitTransportError::Connect(err.to_string()))?;
        let room = Arc::new(room);
        let (input_tx, input_rx) = unbounded_channel();
        let (tts_tx, tts_rx) = unbounded_channel();

        let output = LiveKitAudioOutput::publish(room.clone(), &config).await?;
        let event_task = tokio::spawn(room_event_task(room_events, input_tx, config.clone()));
        let playback_task = tokio::spawn(tts_playback_task(tts_rx, output.source, config));

        Ok(Self {
            room,
            input_rx,
            tts_tx,
            event_task,
            playback_task,
            _output_track: output.track,
        })
    }
}

#[cfg(feature = "livekit-transport")]
impl VoiceTransport for LiveKitVoiceTransport {
    fn next_event(&mut self) -> BoxFuture<'_, Option<VoiceTransportEvent>> {
        Box::pin(async move { self.input_rx.recv().await })
    }

    fn tts_events(&self) -> UnboundedSender<TtsEvent> {
        self.tts_tx.clone()
    }
}

#[cfg(feature = "livekit-transport")]
impl Drop for LiveKitVoiceTransport {
    fn drop(&mut self) {
        self.event_task.abort();
        self.playback_task.abort();
        let room = self.room.clone();
        tokio::spawn(async move {
            let _ = room.close().await;
        });
    }
}

#[cfg(feature = "livekit-transport")]
struct LiveKitAudioOutput {
    source: NativeAudioSource,
    track: LocalAudioTrack,
}

#[cfg(feature = "livekit-transport")]
impl LiveKitAudioOutput {
    async fn publish(
        room: Arc<Room>,
        config: &LiveKitConfig,
    ) -> Result<Self, LiveKitTransportError> {
        let source = NativeAudioSource::new(
            AudioSourceOptions::default(),
            config.output_sample_rate,
            config.output_channels,
            config.output_buffer_ms,
        );
        let track = LocalAudioTrack::create_audio_track(
            "assistant-audio",
            RtcAudioSource::Native(source.clone()),
        );
        let options = TrackPublishOptions {
            source: TrackSource::Microphone,
            ..Default::default()
        };
        room.local_participant()
            .publish_track(LocalTrack::Audio(track.clone()), options)
            .await
            .map_err(|err| LiveKitTransportError::Publish(err.to_string()))?;

        Ok(Self { source, track })
    }
}

#[cfg(feature = "livekit-transport")]
async fn room_event_task(
    mut room_events: UnboundedReceiver<RoomEvent>,
    input_tx: UnboundedSender<VoiceTransportEvent>,
    config: LiveKitConfig,
) {
    while let Some(event) = room_events.recv().await {
        match event {
            RoomEvent::TrackSubscribed {
                track: RemoteTrack::Audio(track),
                ..
            } => {
                tokio::spawn(remote_audio_turn_task(
                    track,
                    input_tx.clone(),
                    config.clone(),
                ));
            }
            RoomEvent::Disconnected { reason } => {
                warn!("LiveKit room disconnected: {reason:?}");
                let _ = input_tx.send(VoiceTransportEvent::Closed);
                break;
            }
            _ => {}
        }
    }
}

#[cfg(feature = "livekit-transport")]
async fn remote_audio_turn_task(
    track: RemoteAudioTrack,
    input_tx: UnboundedSender<VoiceTransportEvent>,
    config: LiveKitConfig,
) {
    let mut stream = NativeAudioStream::new(
        track.rtc_track(),
        config.input_sample_rate as i32,
        config.input_channels as i32,
    );
    let mut active_turn = ActiveLiveKitAudioTurn::default();

    while let Some(frame) = stream.next().await {
        let frame_ms = audio_frame_elapsed_ms(frame.sample_rate, frame.samples_per_channel);
        let speech = frame_has_speech(frame.data.as_ref(), config.input_speech_threshold);
        let Some((chunks, close_after_send)) =
            active_turn.push_frame(frame, frame_ms, speech, &config, &input_tx)
        else {
            continue;
        };

        for chunk in chunks {
            if active_turn.send_chunk(chunk).is_err() {
                active_turn.close();
                break;
            }
        }

        if close_after_send {
            active_turn.close();
        }
    }
}

#[cfg(feature = "livekit-transport")]
#[derive(Default)]
struct ActiveLiveKitAudioTurn {
    tx: Option<UnboundedSender<AudioChunk>>,
    pending: VecDeque<PendingLiveKitFrame>,
    pending_speech_ms: u128,
    elapsed_ms: u128,
    silence_ms: u128,
}

#[cfg(feature = "livekit-transport")]
struct PendingLiveKitFrame {
    bytes: Vec<u8>,
    frame_ms: u128,
}

#[cfg(feature = "livekit-transport")]
impl ActiveLiveKitAudioTurn {
    fn push_frame(
        &mut self,
        frame: AudioFrame<'static>,
        frame_ms: u128,
        speech: bool,
        config: &LiveKitConfig,
        input_tx: &UnboundedSender<VoiceTransportEvent>,
    ) -> Option<(Vec<AudioChunk>, bool)> {
        let pending_frame = PendingLiveKitFrame {
            bytes: linear16_samples_to_bytes(frame.data.as_ref()),
            frame_ms,
        };

        if self.tx.is_none() {
            if !speech {
                self.pending.clear();
                self.pending_speech_ms = 0;
                return None;
            }

            self.pending_speech_ms += frame_ms;
            self.pending.push_back(pending_frame);
            if self.pending_speech_ms < config.input_min_speech_duration.as_millis() {
                return None;
            }

            let (turn_tx, turn_rx) = unbounded_channel();
            if input_tx
                .send(VoiceTransportEvent::AudioTurn(audio_turn_stream(turn_rx)))
                .is_err()
            {
                return None;
            }
            self.tx = Some(turn_tx);
            self.elapsed_ms = 0;
            self.silence_ms = 0;

            let chunks = self.drain_pending_chunks();
            return Some((chunks, false));
        }

        self.elapsed_ms += frame_ms;
        if speech {
            self.silence_ms = 0;
        } else {
            self.silence_ms += frame_ms;
        }

        let chunk = AudioChunk {
            bytes: pending_frame.bytes,
            elapsed_ms: self.elapsed_ms,
        };
        let close_after_send =
            !speech && self.silence_ms >= config.input_silence_timeout.as_millis();

        Some((vec![chunk], close_after_send))
    }

    fn drain_pending_chunks(&mut self) -> Vec<AudioChunk> {
        let mut chunks = Vec::with_capacity(self.pending.len());

        while let Some(frame) = self.pending.pop_front() {
            self.elapsed_ms += frame.frame_ms;
            chunks.push(AudioChunk {
                bytes: frame.bytes,
                elapsed_ms: self.elapsed_ms,
            });
        }
        self.pending_speech_ms = 0;

        chunks
    }

    fn send_chunk(&self, chunk: AudioChunk) -> std::result::Result<(), AudioChunk> {
        match &self.tx {
            Some(tx) => tx.send(chunk).map_err(|err| err.0),
            None => Err(chunk),
        }
    }

    fn close(&mut self) {
        self.tx = None;
        self.pending.clear();
        self.pending_speech_ms = 0;
        self.elapsed_ms = 0;
        self.silence_ms = 0;
    }
}

#[cfg(feature = "livekit-transport")]
fn audio_turn_stream(rx: UnboundedReceiver<AudioChunk>) -> BoxStream<'static, AudioChunk> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }))
}

#[cfg(feature = "livekit-transport")]
async fn tts_playback_task(
    mut tts_rx: UnboundedReceiver<TtsEvent>,
    source: NativeAudioSource,
    config: LiveKitConfig,
) {
    let mut pending_byte = None;

    while let Some(event) = tts_rx.recv().await {
        match event {
            TtsEvent::FirstAudio { .. } => {}
            TtsEvent::AudioChunk { bytes, .. } => {
                let mut samples = linear16_bytes_to_samples_with_pending(&mut pending_byte, &bytes);
                truncate_to_complete_channels(&mut samples, config.output_channels);
                if samples.is_empty() {
                    continue;
                }

                let frame_samples = output_frame_sample_count(
                    config.output_sample_rate,
                    config.output_channels,
                    config.output_frame_ms,
                );
                for frame_samples in samples.chunks(frame_samples) {
                    let frame = AudioFrame {
                        data: frame_samples.into(),
                        sample_rate: config.output_sample_rate,
                        num_channels: config.output_channels,
                        samples_per_channel: frame_samples.len() as u32 / config.output_channels,
                    };

                    if let Err(err) = source.capture_frame(&frame).await {
                        warn!("LiveKit audio frame capture failed: {err}");
                        break;
                    }
                }
            }
            TtsEvent::Done => {
                pending_byte = None;
            }
        }
    }
}

fn required_env(name: &'static str) -> Result<String, LiveKitConfigError> {
    env::var(name).map_err(|_| LiveKitConfigError::MissingEnv(name))
}

fn optional_env(name: &'static str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn optional_u32_env(name: &'static str) -> Result<Option<u32>, LiveKitConfigError> {
    let Some(value) = optional_env(name) else {
        return Ok(None);
    };

    value
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or(LiveKitConfigError::InvalidU32 { name, value })
}

fn optional_u64_env(name: &'static str) -> Result<Option<u64>, LiveKitConfigError> {
    let Some(value) = optional_env(name) else {
        return Ok(None);
    };

    value
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or(LiveKitConfigError::InvalidU64 { name, value })
}

pub fn linear16_samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend(sample.to_le_bytes());
    }
    bytes
}

pub fn linear16_bytes_to_samples(bytes: &[u8]) -> Vec<i16> {
    let mut pending_byte = None;
    linear16_bytes_to_samples_with_pending(&mut pending_byte, bytes)
}

fn linear16_bytes_to_samples_with_pending(pending_byte: &mut Option<u8>, bytes: &[u8]) -> Vec<i16> {
    let mut samples = Vec::with_capacity((bytes.len() + pending_byte.is_some() as usize) / 2);
    let mut chunks = bytes.chunks_exact(2);

    if let Some(first) = pending_byte.take() {
        if let Some(second) = bytes.first() {
            samples.push(i16::from_le_bytes([first, *second]));
            chunks = bytes[1..].chunks_exact(2);
        } else {
            *pending_byte = Some(first);
            return samples;
        }
    }

    for chunk in chunks.by_ref() {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    if let [last] = chunks.remainder() {
        *pending_byte = Some(*last);
    }

    samples
}

#[cfg(any(feature = "livekit-transport", test))]
fn truncate_to_complete_channels(samples: &mut Vec<i16>, channels: u32) {
    if channels == 0 {
        samples.clear();
        return;
    }

    let channels = channels as usize;
    let complete_samples = samples.len() / channels * channels;
    samples.truncate(complete_samples);
}

#[cfg(any(feature = "livekit-transport", test))]
fn frame_has_speech(samples: &[i16], speech_threshold: u32) -> bool {
    if samples.is_empty() {
        return false;
    }

    let average_abs = samples
        .iter()
        .map(|sample| sample.unsigned_abs() as u64)
        .sum::<u64>()
        / samples.len() as u64;
    average_abs >= speech_threshold as u64
}

#[cfg(any(feature = "livekit-transport", test))]
fn output_frame_sample_count(sample_rate: u32, channels: u32, frame_ms: u32) -> usize {
    let samples_per_channel = (sample_rate as usize * frame_ms.max(1) as usize / 1000).max(1);
    samples_per_channel * channels.max(1) as usize
}

#[cfg(feature = "livekit-transport")]
fn audio_frame_elapsed_ms(sample_rate: u32, samples_per_channel: u32) -> u128 {
    if sample_rate == 0 {
        return 0;
    }

    samples_per_channel as u128 * 1000 / sample_rate as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_api::access_token::Claims;

    #[test]
    fn config_generates_join_publish_subscribe_token() {
        let config = LiveKitConfig::new("wss://example.livekit.cloud", "devkey", "secret")
            .with_room("test-room")
            .with_identity("agent")
            .with_name("Agent")
            .with_token_ttl(Duration::from_secs(60));

        let token = config.access_token().expect("token should be generated");
        let claims = Claims::from_unverified(&token).expect("token claims should decode");

        assert_eq!(claims.iss, "devkey");
        assert_eq!(claims.sub, "agent");
        assert_eq!(claims.name, "Agent");
        assert_eq!(claims.video.room, "test-room");
        assert!(claims.video.room_join);
        assert!(claims.video.can_publish);
        assert!(claims.video.can_subscribe);
    }

    #[test]
    fn linear16_samples_round_trip_as_little_endian_bytes() {
        let samples = [-32_768, -2, -1, 0, 1, 2, 32_767];

        let bytes = linear16_samples_to_bytes(&samples);
        let decoded = linear16_bytes_to_samples(&bytes);

        assert_eq!(decoded, samples);
    }

    #[test]
    fn split_linear16_bytes_preserve_pending_odd_byte() {
        let mut pending = None;

        let first = linear16_bytes_to_samples_with_pending(&mut pending, &[0x34]);
        let second = linear16_bytes_to_samples_with_pending(&mut pending, &[0x12, 0x78, 0x56]);

        assert!(first.is_empty());
        assert_eq!(second, vec![0x1234, 0x5678]);
        assert_eq!(pending, None);
    }

    #[test]
    fn incomplete_channel_samples_are_dropped() {
        let mut samples = vec![1, 2, 3, 4, 5];

        truncate_to_complete_channels(&mut samples, 2);

        assert_eq!(samples, vec![1, 2, 3, 4]);
    }

    #[test]
    fn frame_has_speech_uses_average_absolute_sample_energy() {
        assert!(!frame_has_speech(&[], 250));
        assert!(!frame_has_speech(&[0, 10, -10, 20], 250));
        assert!(frame_has_speech(&[300, -400, 500, -600], 250));
        assert!(frame_has_speech(&[i16::MIN], 250));
    }

    #[test]
    fn output_frame_sample_count_uses_channels_and_frame_duration() {
        assert_eq!(output_frame_sample_count(24_000, 1, 20), 480);
        assert_eq!(output_frame_sample_count(48_000, 2, 10), 960);
        assert_eq!(output_frame_sample_count(16_000, 0, 0), 16);
    }

    #[cfg(feature = "livekit-transport")]
    #[test]
    fn active_turn_waits_for_minimum_speech_before_opening_stream() {
        let config = LiveKitConfig::new("wss://example.livekit.cloud", "devkey", "secret")
            .with_input_vad(250, Duration::from_millis(30), Duration::from_millis(700));
        let (input_tx, mut input_rx) = unbounded_channel();
        let mut turn = ActiveLiveKitAudioTurn::default();

        let first = AudioFrame {
            data: [500i16].as_slice().into(),
            sample_rate: 16_000,
            num_channels: 1,
            samples_per_channel: 160,
        };
        assert!(
            turn.push_frame(first, 10, true, &config, &input_tx)
                .is_none()
        );
        assert!(input_rx.try_recv().is_err());

        let second = AudioFrame {
            data: [600i16].as_slice().into(),
            sample_rate: 16_000,
            num_channels: 1,
            samples_per_channel: 160,
        };
        assert!(
            turn.push_frame(second, 10, true, &config, &input_tx)
                .is_none()
        );
        assert!(input_rx.try_recv().is_err());

        let third = AudioFrame {
            data: [700i16].as_slice().into(),
            sample_rate: 16_000,
            num_channels: 1,
            samples_per_channel: 160,
        };
        let (chunks, close_after_send) = turn
            .push_frame(third, 10, true, &config, &input_tx)
            .expect("minimum speech should open a turn");

        assert!(!close_after_send);
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.elapsed_ms)
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert!(matches!(
            input_rx.try_recv(),
            Ok(VoiceTransportEvent::AudioTurn(_))
        ));
    }
}
