use std::{
    collections::HashMap,
    env,
    error::Error,
    io::{self, Write},
};

use ai_runtime::{
    AgentRuntime, CalculatorTool, ConversationContext, MockCrmLookupTool, OpenAiCompatibleProvider,
    OpenAiConfig, RequestState, RunTurnOptions, RuntimeConfig, RuntimeError, RuntimeEvent,
    ToolRegistry,
};
use serde_json::json;
use tokio::{sync::mpsc, task};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "chat" => chat(args.collect()).await,
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown command: {other}").into()),
    }
}

async fn chat(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let tools_enabled = parse_chat_args(args)?;
    let api_key = env::var("OPENAI_COMPAT_API_KEY")
        .map_err(|_| "OPENAI_COMPAT_API_KEY must be set to run `ai-runtime chat`")?;
    let base_url =
        env::var("OPENAI_COMPAT_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let model = env::var("OPENAI_COMPAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let provider = OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, api_key))?;
    let tools = if tools_enabled {
        demo_tools()?
    } else {
        ToolRegistry::new()
    };
    let runtime = AgentRuntime::new(provider, tools, RuntimeConfig::new(model));
    let mut context = ConversationContext::with_system_prompt(
        "You are a concise assistant. Use available tools when they are useful.",
    );

    println!("ai-runtime chat. Type `exit` or `quit` to stop. Press Ctrl-C to cancel a turn.");
    println!(
        "tools: {}",
        if tools_enabled {
            "calculator, mock_crm_lookup"
        } else {
            "disabled"
        }
    );

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "exit" | "quit") {
            break;
        }

        let cancellation_token = CancellationToken::new();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let event_task = task::spawn(print_events(event_rx));
        let turn = runtime.run_turn_with_options(
            &mut context,
            input,
            RunTurnOptions::new(cancellation_token.clone()).with_events(event_tx),
        );

        tokio::pin!(turn);
        let result = tokio::select! {
            result = &mut turn => result,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                cancellation_token.cancel();
                (&mut turn).await
            }
        };

        let _ = event_task.await;

        match result {
            Ok(_) => {}
            Err(RuntimeError::Cancelled) => println!("assistant> turn cancelled"),
            Err(err) => println!("assistant> error: {err}"),
        }
    }

    Ok(())
}

fn parse_chat_args(args: Vec<String>) -> Result<bool, Box<dyn Error>> {
    let mut tools_enabled = false;

    for arg in args {
        match arg.as_str() {
            "--tools" => tools_enabled = true,
            "--no-tools" => tools_enabled = false,
            "-h" | "--help" => {
                print_chat_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown chat option: {other}").into()),
        }
    }

    Ok(tools_enabled)
}

fn demo_tools() -> Result<ToolRegistry, Box<dyn Error>> {
    let mut tools = ToolRegistry::new();
    tools.register(CalculatorTool::default())?;

    let mut customers = HashMap::new();
    customers.insert(
        "cust_123".to_string(),
        json!({
            "customer_id": "cust_123",
            "name": "Ada Lovelace",
            "plan": "Enterprise",
            "renewal_status": "green"
        }),
    );
    tools.register(MockCrmLookupTool::new(customers))?;

    Ok(tools)
}

async fn print_events(mut events: mpsc::UnboundedReceiver<RuntimeEvent>) {
    let mut assistant_line_active = false;

    while let Some(event) = events.recv().await {
        match event {
            RuntimeEvent::Lifecycle { to, .. } => {
                finish_assistant_line(&mut assistant_line_active);
                println!("event> state={}", state_name(to));
            }
            RuntimeEvent::LlmFirstToken { elapsed_ms } => {
                finish_assistant_line(&mut assistant_line_active);
                println!("event> llm_first_token elapsed_ms={elapsed_ms}");
            }
            RuntimeEvent::AssistantToken { text, .. } => {
                if !assistant_line_active {
                    print!("assistant> ");
                    assistant_line_active = true;
                }
                print!("{text}");
                let _ = io::stdout().flush();
            }
            RuntimeEvent::ToolDispatchStart {
                call_id,
                tool_name,
                elapsed_ms,
            } => {
                finish_assistant_line(&mut assistant_line_active);
                println!("event> tool_start id={call_id} name={tool_name} elapsed_ms={elapsed_ms}")
            }
            RuntimeEvent::ToolDispatchEnd {
                call_id,
                tool_name,
                duration_ms,
                success,
                ..
            } => {
                finish_assistant_line(&mut assistant_line_active);
                println!(
                    "event> tool_end id={call_id} name={tool_name} duration_ms={duration_ms} success={success}"
                )
            }
            RuntimeEvent::ToolTimeout {
                call_id,
                tool_name,
                timeout_ms,
                ..
            } => {
                finish_assistant_line(&mut assistant_line_active);
                println!(
                    "event> tool_timeout id={call_id} name={tool_name} timeout_ms={timeout_ms}"
                )
            }
            RuntimeEvent::TurnCompleted {
                elapsed_ms,
                output_chars,
            } => {
                finish_assistant_line(&mut assistant_line_active);
                println!(
                    "event> turn_completed elapsed_ms={elapsed_ms} output_chars={output_chars}"
                );
            }
        }
    }
}

fn finish_assistant_line(assistant_line_active: &mut bool) {
    if *assistant_line_active {
        println!();
        *assistant_line_active = false;
    }
}

fn state_name(state: RequestState) -> &'static str {
    match state {
        RequestState::Started => "started",
        RequestState::StreamingLlm => "streaming_llm",
        RequestState::DispatchingTool => "dispatching_tool",
        RequestState::Completed => "completed",
        RequestState::Cancelled => "cancelled",
        RequestState::Failed => "failed",
    }
}

fn print_usage() {
    println!("usage: ai-runtime chat [--tools|--no-tools]");
    println!("env: OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_MODEL, OPENAI_COMPAT_BASE_URL");
}

fn print_chat_usage() {
    println!("usage: ai-runtime chat [--tools|--no-tools]");
    println!("  --tools     register calculator and mock_crm_lookup demo tools");
    println!("  --no-tools  run LLM-only chat (default)");
}
