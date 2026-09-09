pub mod llm;
pub mod runtime;
pub mod tools;
pub mod voice;

pub use llm::{
    AssistantToolCall, AssistantToolCallFunction, ChatMessage, ChatRequest, LlmError, LlmProvider,
    Result, TokenEvent, ToolCall, ToolSpec,
    openai::{OpenAiCompatibleProvider, OpenAiConfig, active_stream_task_count},
};
pub use runtime::{
    AgentRuntime, ConversationContext, RunTurnOptions, RuntimeConfig, RuntimeError,
    events::{RequestLifecycle, RequestState, RuntimeEvent},
};
pub use tools::{
    CalculatorTool, MockCrmLookupTool, Tool, ToolDefinition, ToolError, ToolMetadata, ToolRegistry,
};
pub use voice::{
    AudioChunk, FakeSpeechToText, FakeTextToSpeech, FakeVoiceTransport, FakeVoiceTransportHandle,
    RuntimeTtsAdapter, SpeechToText, SttError, SttResult, TextToSpeech, TranscriptEvent,
    TtsAdapterError, TtsError, TtsEvent, VoiceEvent, VoiceLatencyMetric, VoiceLatencyRecorder,
    VoiceLatencySnapshot, VoiceLatencyStats, VoicePipelineRunner, VoiceSession, VoiceSessionError,
    VoiceTransport, VoiceTransportError, VoiceTransportEvent, VoiceTurnError, VoiceTurnOptions,
    VoiceTurnOutput, VoiceTurnRunner,
    deepgram::{
        DeepgramConfig, DeepgramConfigError, DeepgramSpeechToText, DeepgramTextToSpeech,
        active_deepgram_stt_stream_count, active_deepgram_tts_stream_count,
    },
    livekit::{
        LiveKitConfig, LiveKitConfigError, LiveKitTransportError, linear16_bytes_to_samples,
        linear16_samples_to_bytes,
    },
};

#[cfg(feature = "livekit-transport")]
pub use voice::livekit::LiveKitVoiceTransport;
