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
    FakeTextToSpeech, RuntimeTtsAdapter, TextToSpeech, TtsAdapterError, TtsError, TtsEvent,
    VoiceTurnError, VoiceTurnOptions, VoiceTurnOutput, VoiceTurnRunner,
};
