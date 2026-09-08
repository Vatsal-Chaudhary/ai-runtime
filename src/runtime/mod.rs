pub mod events;

use std::time::Instant;

use futures_util::StreamExt;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::{
    llm::{ChatMessage, ChatRequest, LlmError, LlmProvider, TokenEvent, ToolCall},
    tools::{ToolError, ToolRegistry},
};

use self::events::{RequestLifecycle, RequestState, RuntimeEvent};

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error(transparent)]
    Tool(#[from] ToolError),

    #[error("turn was cancelled")]
    Cancelled,

    #[error("exceeded maximum tool rounds: {max_rounds}")]
    MaxToolRounds { max_rounds: usize },
}

impl RuntimeError {
    fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub model: String,
    pub max_tool_rounds: usize,
}

impl RuntimeConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tool_rounds: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunTurnOptions {
    pub cancellation_token: CancellationToken,
    pub events: Option<UnboundedSender<RuntimeEvent>>,
}

impl RunTurnOptions {
    pub fn new(cancellation_token: CancellationToken) -> Self {
        Self {
            cancellation_token,
            events: None,
        }
    }

    pub fn with_events(mut self, events: UnboundedSender<RuntimeEvent>) -> Self {
        self.events = Some(events);
        self
    }
}

impl Default for RunTurnOptions {
    fn default() -> Self {
        Self::new(CancellationToken::new())
    }
}

#[derive(Debug, Default, Clone)]
pub struct ConversationContext {
    messages: Vec<ChatMessage>,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::system(prompt)],
        }
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::assistant(content));
    }

    pub fn push_assistant_tool_calls(&mut self, content: impl Into<String>, calls: &[ToolCall]) {
        self.messages
            .push(ChatMessage::assistant_tool_calls(content, calls));
    }

    pub fn push_tool_result(&mut self, call: &ToolCall, output: &Value) {
        self.messages.push(ChatMessage::tool(
            call.id.clone(),
            call.name.clone(),
            output.to_string(),
        ));
    }
}

pub struct AgentRuntime<P> {
    provider: P,
    tools: ToolRegistry,
    config: RuntimeConfig,
}

impl<P> AgentRuntime<P>
where
    P: LlmProvider,
{
    pub fn new(provider: P, tools: ToolRegistry, config: RuntimeConfig) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    pub async fn run_turn(
        &self,
        context: &mut ConversationContext,
        user_message: impl Into<String>,
    ) -> Result<String> {
        self.run_turn_with_options(context, user_message, RunTurnOptions::default())
            .await
    }

    pub async fn run_turn_cancellable(
        &self,
        context: &mut ConversationContext,
        user_message: impl Into<String>,
        cancellation_token: CancellationToken,
    ) -> Result<String> {
        self.run_turn_with_options(
            context,
            user_message,
            RunTurnOptions::new(cancellation_token),
        )
        .await
    }

    pub async fn run_turn_with_options(
        &self,
        context: &mut ConversationContext,
        user_message: impl Into<String>,
        options: RunTurnOptions,
    ) -> Result<String> {
        let emitter = EventEmitter::new(options.events);
        let mut lifecycle = RequestLifecycle::new();
        emitter.emit(lifecycle.transition(RequestState::Started));

        context.push_user(user_message);
        let result = self
            .continue_until_answer(
                context,
                &options.cancellation_token,
                &emitter,
                &mut lifecycle,
            )
            .await;

        match &result {
            Ok(answer) => {
                emitter.emit(RuntimeEvent::TurnCompleted {
                    elapsed_ms: lifecycle.elapsed_ms(),
                    output_chars: answer.len(),
                });
                emitter.emit(lifecycle.transition(RequestState::Completed));
            }
            Err(err) if err.is_cancelled() => {
                emitter.emit(lifecycle.transition(RequestState::Cancelled));
            }
            Err(_) => {
                emitter.emit(lifecycle.transition(RequestState::Failed));
            }
        }

        result
    }

    async fn continue_until_answer(
        &self,
        context: &mut ConversationContext,
        cancellation_token: &CancellationToken,
        emitter: &EventEmitter,
        lifecycle: &mut RequestLifecycle,
    ) -> Result<String> {
        let mut first_token_recorded = false;

        for round in 0..=self.config.max_tool_rounds {
            if cancellation_token.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }

            emitter.emit(lifecycle.transition(RequestState::StreamingLlm));
            let request = ChatRequest::new(self.config.model.clone(), context.messages.clone())
                .with_tools(self.tools.specs());
            let stream = self.provider.stream_chat(request);
            let StreamOutcome { text, tool_calls } = collect_stream(
                stream,
                cancellation_token,
                emitter,
                lifecycle,
                &mut first_token_recorded,
            )
            .await?;

            if tool_calls.is_empty() && !text.is_empty() {
                context.push_assistant(text.clone());
            }

            if tool_calls.is_empty() {
                return Ok(text);
            }

            if round == self.config.max_tool_rounds {
                return Err(RuntimeError::MaxToolRounds {
                    max_rounds: self.config.max_tool_rounds,
                });
            }

            context.push_assistant_tool_calls(text, &tool_calls);

            for call in tool_calls {
                if cancellation_token.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }

                emitter.emit(lifecycle.transition(RequestState::DispatchingTool));
                let output = self
                    .dispatch_tool(&call, cancellation_token, emitter, lifecycle)
                    .await?;
                context.push_tool_result(&call, &output);
            }
        }

        Err(RuntimeError::MaxToolRounds {
            max_rounds: self.config.max_tool_rounds,
        })
    }

    async fn dispatch_tool(
        &self,
        call: &ToolCall,
        cancellation_token: &CancellationToken,
        emitter: &EventEmitter,
        lifecycle: &RequestLifecycle,
    ) -> Result<Value> {
        let started = Instant::now();
        emitter.emit(RuntimeEvent::ToolDispatchStart {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            elapsed_ms: lifecycle.elapsed_ms(),
        });

        let result = tokio::select! {
            _ = cancellation_token.cancelled() => Err(RuntimeError::Cancelled),
            result = self.tools.call(&call.name, call.arguments.clone()) => result.map_err(RuntimeError::from),
        };

        let duration_ms = started.elapsed().as_millis();
        match &result {
            Ok(_) => {
                emitter.emit(RuntimeEvent::ToolDispatchEnd {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    elapsed_ms: lifecycle.elapsed_ms(),
                    duration_ms,
                    success: true,
                });
            }
            Err(RuntimeError::Tool(ToolError::Timeout { .. })) => {
                let timeout_ms = self
                    .tools
                    .definition(&call.name)
                    .map(|definition| definition.metadata.timeout.as_millis())
                    .unwrap_or(duration_ms);
                emitter.emit(RuntimeEvent::ToolTimeout {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    timeout_ms,
                    elapsed_ms: lifecycle.elapsed_ms(),
                });
                emitter.emit(RuntimeEvent::ToolDispatchEnd {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    elapsed_ms: lifecycle.elapsed_ms(),
                    duration_ms,
                    success: false,
                });
            }
            Err(RuntimeError::Cancelled) => {}
            Err(_) => {
                emitter.emit(RuntimeEvent::ToolDispatchEnd {
                    call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    elapsed_ms: lifecycle.elapsed_ms(),
                    duration_ms,
                    success: false,
                });
            }
        }

        result
    }
}

struct StreamOutcome {
    text: String,
    tool_calls: Vec<ToolCall>,
}

async fn collect_stream(
    mut events: futures_core::stream::BoxStream<'static, crate::llm::Result<TokenEvent>>,
    cancellation_token: &CancellationToken,
    emitter: &EventEmitter,
    lifecycle: &RequestLifecycle,
    first_token_recorded: &mut bool,
) -> Result<StreamOutcome> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    loop {
        let event = tokio::select! {
            _ = cancellation_token.cancelled() => return Err(RuntimeError::Cancelled),
            event = events.next() => event,
        };

        let Some(event) = event else {
            break;
        };

        match event? {
            TokenEvent::Token { text: token } => {
                if !*first_token_recorded {
                    emitter.emit(RuntimeEvent::LlmFirstToken {
                        elapsed_ms: lifecycle.elapsed_ms(),
                    });
                    *first_token_recorded = true;
                }

                emitter.emit(RuntimeEvent::AssistantToken {
                    text: token.clone(),
                    elapsed_ms: lifecycle.elapsed_ms(),
                });
                text.push_str(&token);
            }
            TokenEvent::ToolCall(call) => tool_calls.push(call),
            TokenEvent::Done => break,
        }
    }

    Ok(StreamOutcome { text, tool_calls })
}

#[derive(Debug, Clone)]
struct EventEmitter {
    events: Option<UnboundedSender<RuntimeEvent>>,
}

impl EventEmitter {
    fn new(events: Option<UnboundedSender<RuntimeEvent>>) -> Self {
        Self { events }
    }

    fn emit(&self, event: RuntimeEvent) {
        if let Some(events) = &self.events {
            let _ = events.send(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        task::{Context, Poll},
        time::Duration,
    };

    use futures_core::Stream;
    use futures_util::{future::BoxFuture, stream};
    use serde_json::json;
    use tokio::{
        sync::mpsc,
        time::{sleep, timeout},
    };

    use super::*;
    use crate::{
        llm::{Result as LlmResult, ToolCall},
        tools::{CalculatorTool, Tool, ToolDefinition, ToolMetadata},
    };

    #[tokio::test]
    async fn sequential_tool_result_is_reinjected_before_final_answer() {
        let provider = FakeProvider::new(vec![
            vec![
                Ok(TokenEvent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "calculator".to_string(),
                    arguments: json!({
                        "operation": "add",
                        "a": 20,
                        "b": 22
                    }),
                })),
                Ok(TokenEvent::Done),
            ],
            vec![
                Ok(TokenEvent::Token {
                    text: "The answer is 42.".to_string(),
                }),
                Ok(TokenEvent::Done),
            ],
        ]);
        let requests = provider.requests();
        let mut tools = ToolRegistry::new();
        tools
            .register(CalculatorTool::default())
            .expect("tool should register");
        let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new("test-model"));
        let mut context = ConversationContext::new();

        let answer = runtime
            .run_turn(&mut context, "What is 20 + 22?")
            .await
            .expect("turn should succeed");

        assert_eq!(answer, "The answer is 42.");
        assert!(
            context
                .messages()
                .iter()
                .any(|message| message.role == "tool"
                    && message.tool_call_id.as_deref() == Some("call_1")
                    && message.content == json!({ "result": 42.0 }).to_string())
        );

        let requests = requests
            .lock()
            .expect("requests mutex should not be poisoned");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].tools.len(), 1);
        assert!(
            requests[1]
                .messages
                .iter()
                .any(|message| message.role == "tool")
        );
    }

    #[tokio::test]
    async fn context_preserves_tool_results_across_two_turns() {
        let provider = FakeProvider::new(vec![
            vec![
                Ok(TokenEvent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "calculator".to_string(),
                    arguments: json!({
                        "operation": "multiply",
                        "a": 6,
                        "b": 7
                    }),
                })),
                Ok(TokenEvent::Done),
            ],
            vec![
                Ok(TokenEvent::Token {
                    text: "It is 42.".to_string(),
                }),
                Ok(TokenEvent::Done),
            ],
            vec![
                Ok(TokenEvent::Token {
                    text: "The previous tool result is still in context.".to_string(),
                }),
                Ok(TokenEvent::Done),
            ],
        ]);
        let mut tools = ToolRegistry::new();
        tools
            .register(CalculatorTool::default())
            .expect("tool should register");
        let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new("test-model"));
        let mut context = ConversationContext::new();

        runtime
            .run_turn(&mut context, "What is 6 * 7?")
            .await
            .expect("first turn should succeed");
        runtime
            .run_turn(&mut context, "Can you use that result again?")
            .await
            .expect("second turn should succeed");

        let tool_result_count = context
            .messages()
            .iter()
            .filter(|message| message.role == "tool")
            .count();
        assert_eq!(tool_result_count, 1);
        assert_eq!(
            context.messages().last().expect("last message").role,
            "assistant"
        );
    }

    #[tokio::test]
    async fn emits_lifecycle_latency_and_tool_events() {
        let provider = FakeProvider::new(vec![
            vec![
                Ok(TokenEvent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "calculator".to_string(),
                    arguments: json!({
                        "operation": "add",
                        "a": 20,
                        "b": 22
                    }),
                })),
                Ok(TokenEvent::Done),
            ],
            vec![
                Ok(TokenEvent::Token {
                    text: "The answer is 42.".to_string(),
                }),
                Ok(TokenEvent::Done),
            ],
        ]);
        let mut tools = ToolRegistry::new();
        tools
            .register(CalculatorTool::default())
            .expect("tool should register");
        let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new("test-model"));
        let mut context = ConversationContext::new();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let answer = runtime
            .run_turn_with_options(
                &mut context,
                "What is 20 + 22?",
                RunTurnOptions::default().with_events(tx),
            )
            .await
            .expect("turn should succeed");

        assert_eq!(answer, "The answer is 42.");

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::Lifecycle {
                    from: None,
                    to: RequestState::Started,
                    ..
                }
            )
        });
        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::Lifecycle {
                    to: RequestState::StreamingLlm,
                    ..
                }
            )
        });
        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::ToolDispatchStart {
                    call_id,
                    tool_name,
                    ..
                } if call_id == "call_1" && tool_name == "calculator"
            )
        });
        assert_event(&events, |event| {
            matches!(event, RuntimeEvent::LlmFirstToken { .. })
        });
        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::AssistantToken {
                    text,
                    ..
                } if text == "The answer is 42."
            )
        });
        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::TurnCompleted {
                    output_chars,
                    ..
                } if *output_chars == "The answer is 42.".len()
            )
        });
        assert!(matches!(
            events.last(),
            Some(RuntimeEvent::Lifecycle {
                to: RequestState::Completed,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn emits_assistant_token_events_in_stream_order() {
        let provider = FakeProvider::new(vec![vec![
            Ok(TokenEvent::Token {
                text: "Hel".to_string(),
            }),
            Ok(TokenEvent::Token {
                text: "lo".to_string(),
            }),
            Ok(TokenEvent::Token {
                text: ", world".to_string(),
            }),
            Ok(TokenEvent::Done),
        ]]);
        let runtime = AgentRuntime::new(provider, ToolRegistry::new(), RuntimeConfig::new("test"));
        let mut context = ConversationContext::new();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let answer = runtime
            .run_turn_with_options(
                &mut context,
                "say hello",
                RunTurnOptions::default().with_events(tx),
            )
            .await
            .expect("turn should succeed");

        assert_eq!(answer, "Hello, world");

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        let tokens = events
            .iter()
            .filter_map(|event| match event {
                RuntimeEvent::AssistantToken { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(tokens, vec!["Hel", "lo", ", world"]);
        assert!(matches!(
            events.last(),
            Some(RuntimeEvent::Lifecycle {
                to: RequestState::Completed,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn cancellation_drops_active_llm_stream() {
        let active_streams = Arc::new(AtomicUsize::new(0));
        let provider = HangingProvider {
            active_streams: Arc::clone(&active_streams),
        };
        let runtime = AgentRuntime::new(provider, ToolRegistry::new(), RuntimeConfig::new("test"));
        let cancellation_token = CancellationToken::new();
        let task_token = cancellation_token.clone();

        let task = tokio::spawn(async move {
            let mut context = ConversationContext::new();
            runtime
                .run_turn_cancellable(&mut context, "hello", task_token)
                .await
        });

        wait_for_count(&active_streams, 1).await;
        cancellation_token.cancel();

        let result = timeout(Duration::from_secs(1), task)
            .await
            .expect("cancelled runtime should return")
            .expect("runtime task should not panic");
        assert!(matches!(result, Err(RuntimeError::Cancelled)));
        wait_for_count(&active_streams, 0).await;
    }

    #[tokio::test]
    async fn cancellation_drops_active_tool_future() {
        let active_tools = Arc::new(AtomicUsize::new(0));
        let provider = FakeProvider::new(vec![vec![
            Ok(TokenEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "hanging".to_string(),
                arguments: json!({}),
            })),
            Ok(TokenEvent::Done),
        ]]);
        let mut tools = ToolRegistry::new();
        tools
            .register(HangingTool {
                active_tools: Arc::clone(&active_tools),
            })
            .expect("tool should register");
        let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new("test"));
        let cancellation_token = CancellationToken::new();
        let task_token = cancellation_token.clone();

        let task = tokio::spawn(async move {
            let mut context = ConversationContext::new();
            runtime
                .run_turn_cancellable(&mut context, "call a tool", task_token)
                .await
        });

        wait_for_count(&active_tools, 1).await;
        cancellation_token.cancel();

        let result = timeout(Duration::from_secs(1), task)
            .await
            .expect("cancelled runtime should return")
            .expect("runtime task should not panic");
        assert!(matches!(result, Err(RuntimeError::Cancelled)));
        wait_for_count(&active_tools, 0).await;
    }

    #[tokio::test]
    async fn partial_assistant_text_is_not_committed_after_cancellation() {
        let provider = TokenThenHangingProvider {
            token: "partial answer".to_string(),
        };
        let runtime = AgentRuntime::new(provider, ToolRegistry::new(), RuntimeConfig::new("test"));
        let cancellation_token = CancellationToken::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut context = ConversationContext::new();

        let result = {
            let turn = runtime.run_turn_with_options(
                &mut context,
                "hello",
                RunTurnOptions::new(cancellation_token.clone()).with_events(tx),
            );
            tokio::pin!(turn);

            loop {
                tokio::select! {
                    event = rx.recv() => {
                        match event.expect("runtime should emit events before cancellation") {
                            RuntimeEvent::AssistantToken { text, .. } if text == "partial answer" => break,
                            _ => {}
                        }
                    }
                    result = &mut turn => panic!("turn finished before cancellation: {result:?}"),
                }
            }

            cancellation_token.cancel();

            timeout(Duration::from_secs(1), &mut turn)
                .await
                .expect("cancelled runtime should return")
        };

        assert!(matches!(result, Err(RuntimeError::Cancelled)));
        assert!(
            context
                .messages()
                .iter()
                .all(|message| message.role != "assistant"),
            "partial assistant output should not be committed: {:#?}",
            context.messages()
        );
    }

    #[tokio::test]
    async fn tool_timeout_emits_timeout_and_failed_lifecycle_events() {
        let provider = FakeProvider::new(vec![vec![
            Ok(TokenEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "slow_runtime".to_string(),
                arguments: json!({}),
            })),
            Ok(TokenEvent::Done),
        ]]);
        let mut tools = ToolRegistry::new();
        tools
            .register(SlowRuntimeTool)
            .expect("tool should register");
        let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new("test"));
        let mut context = ConversationContext::new();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let err = runtime
            .run_turn_with_options(
                &mut context,
                "call a slow tool",
                RunTurnOptions::default().with_events(tx),
            )
            .await
            .expect_err("slow tool should fail");

        assert!(matches!(err, RuntimeError::Tool(ToolError::Timeout { .. })));

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        assert_event(&events, |event| {
            matches!(
                event,
                RuntimeEvent::ToolTimeout {
                    call_id,
                    tool_name,
                    timeout_ms,
                    ..
                } if call_id == "call_1" && tool_name == "slow_runtime" && *timeout_ms == 10
            )
        });
        assert!(matches!(
            events.last(),
            Some(RuntimeEvent::Lifecycle {
                to: RequestState::Failed,
                ..
            })
        ));
    }

    fn assert_event<F>(events: &[RuntimeEvent], predicate: F)
    where
        F: Fn(&RuntimeEvent) -> bool,
    {
        assert!(
            events.iter().any(predicate),
            "expected event was missing from {events:#?}"
        );
    }

    async fn wait_for_count(count: &AtomicUsize, expected: usize) {
        timeout(Duration::from_secs(1), async {
            while count.load(Ordering::SeqCst) != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("counter should reach expected value");
    }

    #[derive(Clone)]
    struct FakeProvider {
        responses: Arc<Mutex<Vec<Vec<LlmResult<TokenEvent>>>>>,
        requests: Arc<Mutex<Vec<ChatRequest>>>,
    }

    impl FakeProvider {
        fn new(responses: Vec<Vec<LlmResult<TokenEvent>>>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into_iter().rev().collect())),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<ChatRequest>>> {
            Arc::clone(&self.requests)
        }
    }

    impl LlmProvider for FakeProvider {
        fn stream_chat(
            &self,
            req: ChatRequest,
        ) -> futures_core::stream::BoxStream<'static, LlmResult<TokenEvent>> {
            self.requests
                .lock()
                .expect("requests mutex should not be poisoned")
                .push(req);
            let response = self
                .responses
                .lock()
                .expect("responses mutex should not be poisoned")
                .pop()
                .expect("fake response should exist");
            Box::pin(stream::iter(response))
        }
    }

    struct HangingProvider {
        active_streams: Arc<AtomicUsize>,
    }

    impl LlmProvider for HangingProvider {
        fn stream_chat(
            &self,
            _req: ChatRequest,
        ) -> futures_core::stream::BoxStream<'static, LlmResult<TokenEvent>> {
            self.active_streams.fetch_add(1, Ordering::SeqCst);
            Box::pin(HangingStream {
                active_streams: Arc::clone(&self.active_streams),
            })
        }
    }

    struct HangingStream {
        active_streams: Arc<AtomicUsize>,
    }

    impl Stream for HangingStream {
        type Item = LlmResult<TokenEvent>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for HangingStream {
        fn drop(&mut self) {
            self.active_streams.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct TokenThenHangingProvider {
        token: String,
    }

    impl LlmProvider for TokenThenHangingProvider {
        fn stream_chat(
            &self,
            _req: ChatRequest,
        ) -> futures_core::stream::BoxStream<'static, LlmResult<TokenEvent>> {
            Box::pin(TokenThenHangingStream {
                token: Some(self.token.clone()),
            })
        }
    }

    struct TokenThenHangingStream {
        token: Option<String>,
    }

    impl Stream for TokenThenHangingStream {
        type Item = LlmResult<TokenEvent>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if let Some(text) = self.token.take() {
                return Poll::Ready(Some(Ok(TokenEvent::Token { text })));
            }

            Poll::Pending
        }
    }

    struct HangingTool {
        active_tools: Arc<AtomicUsize>,
    }

    impl Tool for HangingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "hanging".to_string(),
                description: "Never completes unless cancelled.".to_string(),
                parameters: json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                }),
                metadata: ToolMetadata::read_only(Duration::from_secs(30)),
            }
        }

        fn call(&self, _args: Value) -> BoxFuture<'static, crate::tools::Result<Value>> {
            self.active_tools.fetch_add(1, Ordering::SeqCst);
            Box::pin(HangingToolFuture {
                active_tools: Arc::clone(&self.active_tools),
            })
        }
    }

    struct HangingToolFuture {
        active_tools: Arc<AtomicUsize>,
    }

    impl std::future::Future for HangingToolFuture {
        type Output = crate::tools::Result<Value>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for HangingToolFuture {
        fn drop(&mut self) {
            self.active_tools.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct SlowRuntimeTool;

    impl Tool for SlowRuntimeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "slow_runtime".to_string(),
                description: "Times out quickly.".to_string(),
                parameters: json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                }),
                metadata: ToolMetadata::read_only(Duration::from_millis(10)),
            }
        }

        fn call(&self, _args: Value) -> BoxFuture<'static, crate::tools::Result<Value>> {
            Box::pin(async {
                sleep(Duration::from_secs(30)).await;
                Ok(json!({ "ok": true }))
            })
        }
    }
}
