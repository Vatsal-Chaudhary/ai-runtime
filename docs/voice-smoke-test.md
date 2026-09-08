# Voice Provider Smoke Test

Use this checklist when wiring a real LiveKit transport and hosted STT/TTS adapters. The fake pipeline already covers cancellation and state correctness in tests; this checklist validates device, network, and provider behavior.

## Prerequisites

- LiveKit URL, API key, and API secret are configured.
- STT provider API key is configured.
- TTS provider API key is configured.
- Microphone and speaker are available on the machine running the demo.
- `cargo test` passes before starting the manual smoke test.

## Happy Path

1. Start the voice demo with a fresh conversation context.
2. Speak a short request: `say hello in one sentence`.
3. Confirm STT emits partial transcript events while speech is active.
4. Confirm STT emits one final transcript for the turn.
5. Confirm the runtime emits one LLM first-token event.
6. Confirm TTS emits first-audio before any audio chunks.
7. Confirm playback is audible and completes.
8. Confirm the assistant response is committed only after the turn completes.
9. Record the printed p50/p95 metrics for:
   - STT finalization
   - LLM first token
   - TTS first audio
   - Voice turn round trip

## Interruption Path

1. Start a request that produces a multi-sentence answer.
2. Speak a new request while assistant audio is playing.
3. Confirm the active LLM/TTS work is cancelled.
4. Confirm playback for the interrupted response stops.
5. Confirm `VoiceTurnInterrupted` and `VoiceTurnCancelled` are emitted.
6. Confirm the partial assistant response is not committed to conversation context.
7. Confirm the new request runs to completion and is committed.

## Disconnect Path

1. Start a voice turn.
2. Disconnect the LiveKit room while STT, LLM, or TTS work is still active.
3. Confirm the active turn is cancelled without panic.
4. Confirm provider streams are dropped.
5. Confirm no partial assistant text is committed.

## Failure Notes To Capture

- Provider and model names used.
- Region or endpoint used for LiveKit/STT/TTS.
- Audio format and sample rate.
- Whether STT partials are stable or frequently revised.
- Any provider timeout or reconnect behavior.
- The exact command and environment variables used, excluding secret values.
