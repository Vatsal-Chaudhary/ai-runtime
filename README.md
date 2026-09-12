A streaming LLM orchestration + tool-calling + real-time voice runtime, built directly against provider APIs in Rust. No LangChain, no Pipecat, no agent framework dependency.

Manual voice provider smoke test: [docs/voice-smoke-test.md](docs/voice-smoke-test.md)

Status: LiveKit + Deepgram voice bridge working, with best-effort barge-in cancellation.

## Architecture

`ai-runtime` owns the orchestration layer that frameworks usually hide:

- OpenAI-compatible streaming LLM client using `reqwest` + SSE parsing
- Trait-based tool registry with schema validation, timeout handling, and multi-turn context
- Runtime lifecycle events for first-token latency, tool dispatch, cancellation, and completion
- Voice pipeline that connects LiveKit audio transport to Deepgram streaming STT/TTS
- Cancellation propagation from user interruption through active STT, LLM, TTS, and context updates

Runtime shape:

```text
LiveKit mic audio
  -> VoiceTransport
  -> SpeechToText
  -> AgentRuntime
  -> RuntimeTtsAdapter
  -> TextToSpeech
  -> LiveKit assistant audio track
```

The core behavior is implemented in Rust. LiveKit, Deepgram, and the LLM provider are integrated as external providers rather than reimplemented.

## Setup

Install Rust, then configure provider credentials in `.env`:

```sh
cp .env.example .env
nvim .env
```

Required for LLM/chat:

```env
OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1
OPENAI_COMPAT_MODEL=gpt-4o-mini
```

Required for Deepgram STT/TTS:

```env
DEEPGRAM_API_KEY=
DEEPGRAM_LISTEN_URL=wss://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true&interim_results=true&encoding=linear16&sample_rate=16000&channels=1&endpointing=300
DEEPGRAM_SPEAK_URL=https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&container=none&sample_rate=24000
```

Required for LiveKit:

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_ROOM=ai-runtime
LIVEKIT_IDENTITY=ai-runtime-agent
LIVEKIT_NAME=ai-runtime agent
```

LiveKit audio tuning defaults:

```env
LIVEKIT_INPUT_SAMPLE_RATE=16000
LIVEKIT_INPUT_CHANNELS=1
LIVEKIT_INPUT_SPEECH_THRESHOLD=250
LIVEKIT_INPUT_MIN_SPEECH_MS=120
LIVEKIT_INPUT_SILENCE_TIMEOUT_MS=700
LIVEKIT_OUTPUT_SAMPLE_RATE=24000
LIVEKIT_OUTPUT_CHANNELS=1
LIVEKIT_OUTPUT_BUFFER_MS=100
LIVEKIT_OUTPUT_FRAME_MS=20
```

On this machine, LiveKit WebRTC builds need the system clang 22 instead of the Solang clang 16 that appears earlier in `PATH`:

```sh
env CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo test --features livekit-transport
```

## Commands

Run tests:

```sh
cargo test
env CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo test --features livekit-transport
```

Run LLM-only chat:

```sh
cargo run -- chat
```

Run chat with demo tools:

```sh
cargo run -- chat --tools
```

Run deterministic fake voice without provider credentials:

```sh
cargo run -- voice-fake
```

Smoke-test Deepgram TTS:

```sh
cargo run -- deepgram-tts-smoke "Hello from ai-runtime."
```

Smoke-test Deepgram TTS -> STT loopback:

```sh
cargo run -- deepgram-loopback-smoke "Please transcribe this sentence."
```

Smoke-test LiveKit token generation:

```sh
cargo run -- livekit-token-smoke
```

Run the real LiveKit voice bridge:

```sh
env CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo run --features livekit-transport -- voice-livekit
```

To test manually, join the same `LIVEKIT_ROOM` from another LiveKit participant, speak into the browser microphone, and listen for the `assistant-audio` track.

## Voice Latency Metrics

`VoiceLatencyRecorder` consumes `VoiceEvent`s and reports p50/p95 summaries for:

- `stt_finalization`: audio stream start to final transcript, using the STT backend elapsed timestamp
- `llm_first_token`: runtime turn start to first assistant token
- `tts_first_audio`: TTS text stream start to first audio event
- `voice_turn_round_trip`: audio stream start to completed voice turn

Manual LiveKit + Deepgram smoke run:

| Metric | Samples | p50 | p95 |
|---|---:|---:|---:|
| STT finalization | 5 | 3454 ms | 4087 ms |
| LLM first token | 5 | 542 ms | 909 ms |
| TTS first audio | 5 | 1033 ms | 1440 ms |
| Voice round trip | 5 | 10179 ms | 11469 ms |

Raw turn data from the same run:

| Turn | Prompt | STT final | LLM first token | TTS first audio | Round trip |
|---:|---|---:|---:|---:|---:|
| 1 | hello in one sentence | 2696 ms | 909 ms | 1440 ms | 5899 ms |
| 2 | what is rust in one sentence | 3338 ms | 542 ms | 1033 ms | 10632 ms |
| 3 | explain async programming in one sentence | 3541 ms | 474 ms | 1019 ms | 10179 ms |
| 4 | what is a websocket in one sentence | 4087 ms | 458 ms | 1030 ms | 11469 ms |
| 5 | give one benefit of rust for backend systems | 3454 ms | 645 ms | 1391 ms | 8846 ms |

## Failure Modes Covered

- Malformed and partial SSE chunks are tested.
- Dropping an LLM stream mid-response cancels the provider task.
- Tool schema validation and tool timeouts are tested.
- Cancelling during LLM streaming stops token forwarding into TTS.
- Cancelling during TTS streaming drops the active TTS stream.
- Cancelling during STT streaming drops the active STT stream.
- Interrupted voice turns do not commit partial assistant output.
- Dropped fake transport cancels the active turn without panicking.
- Empty/no-transcript LiveKit audio segments are skipped without killing the room runner.

## Limitations

- The implemented LLM provider is OpenAI-compatible only. The trait boundary is ready for additional providers, but they are not implemented yet.
- LiveKit and Deepgram are provider integrations, not custom WebRTC or speech-model implementations.
- Barge-in cancellation is runtime-correct: the active LLM/TTS turn is cancelled and partial assistant text is not committed incorrectly. Already-buffered LiveKit/browser audio may still play briefly after cancellation.
- This is a single-agent portfolio runtime, not a multi-tenant production service.
- RAG was intentionally cut from the MVP because cancellation, streaming, latency, and voice turn-taking provide the stronger systems signal.
