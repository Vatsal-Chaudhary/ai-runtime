A streaming LLM orchestration + tool-calling + real-time voice runtime, built directly against provider APIs in Rust - no LangChain, no Pipecat, no agent framework dependency.

Functional design document: [docs/fdd.md](docs/fdd.md)

Manual voice provider smoke test: [docs/voice-smoke-test.md](docs/voice-smoke-test.md)

Status: Voice runtime fake pipeline in progress

## CLI demo

Run an LLM-only chat loop:

```sh
OPENAI_COMPAT_API_KEY=... cargo run -- chat
```

Run with the demo calculator and mock CRM tools enabled:

```sh
OPENAI_COMPAT_API_KEY=... cargo run -- chat --tools
```

Run the deterministic fake voice pipeline without API keys:

```sh
cargo run -- voice-fake
```

Smoke-test Deepgram TTS without LiveKit:

```sh
DEEPGRAM_API_KEY=... cargo run -- deepgram-tts-smoke "Hello from ai-runtime."
```

Smoke-test Deepgram STT with raw 16kHz mono linear16 PCM:

```sh
DEEPGRAM_API_KEY=... cargo run -- deepgram-stt-file ./sample.raw
```

Smoke-test Deepgram TTS and STT together without a microphone:

```sh
DEEPGRAM_API_KEY=... cargo run -- deepgram-loopback-smoke "Please transcribe this sentence."
```

Smoke-test LiveKit config and token generation without connecting WebRTC:

```sh
LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... cargo run -- livekit-token-smoke
```

Run the real LiveKit voice bridge after installing/selecting `clang++` 21 or newer:

```sh
cargo run --features livekit-transport -- voice-livekit
```

Optional env vars:

- `OPENAI_COMPAT_MODEL` defaults to `gpt-4o-mini`
- `OPENAI_COMPAT_BASE_URL` defaults to `https://api.openai.com/v1`
- `DEEPGRAM_LISTEN_URL` defaults to Nova-3 with interim results, smart formatting, 16kHz mono linear16 input, and 300ms endpointing
- `DEEPGRAM_SPEAK_URL` defaults to Aura-2 Thalia with 24kHz linear16 output
- `deepgram-loopback-smoke` overrides TTS output to 16kHz linear16 so it can feed the STT smoke path
- `LIVEKIT_ROOM` defaults to `ai-runtime`
- `LIVEKIT_IDENTITY` defaults to `ai-runtime-agent`
- `LIVEKIT_INPUT_SAMPLE_RATE`/`LIVEKIT_INPUT_CHANNELS` default to 16kHz mono
- `LIVEKIT_INPUT_SPEECH_THRESHOLD` defaults to `250`; lower it if quiet speech does not start a turn, raise it if room noise starts turns
- `LIVEKIT_INPUT_MIN_SPEECH_MS` defaults to `120`; raise it if noise creates empty turns
- `LIVEKIT_INPUT_SILENCE_TIMEOUT_MS` defaults to `700`
- `LIVEKIT_OUTPUT_SAMPLE_RATE`/`LIVEKIT_OUTPUT_CHANNELS` default to 24kHz mono

The CLI prints structured runtime events for lifecycle state, LLM first-token latency, tool dispatch start/end, tool timeouts, and turn completion. Press Ctrl-C during a turn to cancel the in-flight LLM stream or tool call.

`voice-fake` treats typed text as fake audio chunks, runs it through `FakeSpeechToText`, a deterministic fake LLM provider, and `FakeTextToSpeech`, then prints voice latency events plus fake TTS audio chunks.

The voice library also exposes `VoiceSession`, which owns conversation state for a running voice interaction. Starting a new audio turn while another turn is active cancels the in-flight LLM/TTS work, emits `VoiceTurnInterrupted`, and prevents partial assistant output from being committed.

`VoiceTransport` is the boundary for real-time audio transport. The included `FakeVoiceTransport` can inject user audio turns, capture outgoing TTS events, and exercise disconnect behavior without LiveKit or provider credentials.

`LiveKitVoiceTransport` is available behind the `livekit-transport` feature. It joins a room with a generated LiveKit token, segments subscribed remote audio tracks into repeated runtime audio turns, publishes an `assistant-audio` track, and writes linear16 TTS chunks into that track.

## Voice latency metrics

`VoiceLatencyRecorder` consumes `VoiceEvent`s and reports p50/p95 summaries for the latency table called out in the FDD. The fake voice CLI prints these after each turn:

```text
voice-metrics> stt_finalization samples=1 p50_ms=40 p95_ms=40
voice-metrics> llm_first_token samples=1 p50_ms=0 p95_ms=0
voice-metrics> tts_first_audio samples=1 p50_ms=0 p95_ms=0
voice-metrics> voice_turn_round_trip samples=1 p50_ms=1 p95_ms=1
```

Measurement boundaries:

- `stt_finalization`: audio stream start to final transcript, using the STT backend elapsed timestamp
- `llm_first_token`: runtime turn start to first assistant token
- `tts_first_audio`: TTS text stream start to first audio event
- `voice_turn_round_trip`: audio stream start to completed voice turn

## Failure modes covered

- Cancelling during LLM streaming stops token forwarding into TTS.
- Cancelling during TTS streaming drops the active TTS stream.
- Cancelling during STT streaming drops the active STT stream.
- Interrupted voice turns do not commit partial assistant output.
- Dropped fake transport cancels the active turn without panicking.

## Remaining provider work

The core runtime and fake backends are runnable without credentials. Deepgram STT/TTS adapters and a feature-gated LiveKit transport are available for provider smoke tests. Real LiveKit audio requires `clang++` 21+ to build the WebRTC binding and still needs a manual mic/speaker smoke test.
