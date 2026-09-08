A streaming LLM orchestration + tool-calling + real-time voice runtime, built directly against provider APIs in Rust - no LangChain, no Pipecat, no agent framework dependency.

Functional design document: [docs/fdd.md](docs/fdd.md)

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

Optional env vars:

- `OPENAI_COMPAT_MODEL` defaults to `gpt-4o-mini`
- `OPENAI_COMPAT_BASE_URL` defaults to `https://api.openai.com/v1`

The CLI prints structured runtime events for lifecycle state, LLM first-token latency, tool dispatch start/end, tool timeouts, and turn completion. Press Ctrl-C during a turn to cancel the in-flight LLM stream or tool call.

`voice-fake` treats typed text as fake audio chunks, runs it through `FakeSpeechToText`, a deterministic fake LLM provider, and `FakeTextToSpeech`, then prints voice latency events plus fake TTS audio chunks.

The voice library also exposes `VoiceSession`, which owns conversation state for a running voice interaction. Starting a new audio turn while another turn is active cancels the in-flight LLM/TTS work, emits `VoiceTurnInterrupted`, and prevents partial assistant output from being committed.

`VoiceTransport` is the boundary for real-time audio transport. The included `FakeVoiceTransport` can inject user audio turns, capture outgoing TTS events, and exercise disconnect behavior without LiveKit or provider credentials.

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

The core runtime and fake backends are runnable without credentials. Real LiveKit transport and hosted STT/TTS provider adapters still need provider selection, API keys, and a manual mic/speaker smoke test.
