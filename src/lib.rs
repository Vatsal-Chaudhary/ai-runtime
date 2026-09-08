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
    AudioChunk, FakeSpeechToText, FakeTextToSpeech, RuntimeTtsAdapter, SpeechToText, SttError,
    SttResult, TextToSpeech, TranscriptEvent, TtsAdapterError, TtsError, TtsEvent,
    VoicePipelineRunner, VoiceTurnError, VoiceTurnOptions, VoiceTurnOutput, VoiceTurnRunner,
};
