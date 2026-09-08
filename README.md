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
