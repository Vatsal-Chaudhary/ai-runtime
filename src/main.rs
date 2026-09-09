use std::{
    collections::HashMap,
    env,
    error::Error,
    io::{self, Write},
    time::Duration,
};

use ai_runtime::{
    AgentRuntime, AudioChunk, CalculatorTool, ChatRequest, ConversationContext, DeepgramConfig,
    DeepgramSpeechToText, DeepgramTextToSpeech, FakeSpeechToText, FakeTextToSpeech, LiveKitConfig,
    LlmProvider, MockCrmLookupTool, OpenAiCompatibleProvider, OpenAiConfig, RequestState,
    RunTurnOptions, RuntimeConfig, RuntimeError, RuntimeEvent, SpeechToText, TextToSpeech,
    TokenEvent, ToolRegistry, TtsEvent, VoiceEvent, VoiceLatencyRecorder, VoiceLatencySnapshot,
    VoiceLatencyStats, VoicePipelineRunner, VoiceTurnError, VoiceTurnOptions,
};
#[cfg(feature = "livekit-transport")]
use ai_runtime::{LiveKitVoiceTransport, VoiceSession};
use futures_core::stream::BoxStream;
use futures_util::StreamExt;
use serde_json::json;
use tokio::{sync::mpsc, task};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();

    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "chat" => chat(args.collect()).await,
        "voice-fake" => voice_fake(args.collect()).await,
        "deepgram-tts-smoke" => deepgram_tts_smoke(args.collect()).await,
        "deepgram-stt-file" => deepgram_stt_file(args.collect()).await,
        "deepgram-loopback-smoke" => deepgram_loopback_smoke(args.collect()).await,
        "livekit-token-smoke" => livekit_token_smoke(args.collect()).await,
        "voice-livekit" => voice_livekit(args.collect()).await,
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

async fn voice_fake(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    parse_voice_fake_args(args)?;

    let pipeline = VoicePipelineRunner::new(
        FakeVoiceProvider,
        ToolRegistry::new(),
        RuntimeConfig::new("fake-voice-model"),
        FakeSpeechToText::new(),
        FakeTextToSpeech::new(),
    );
    let mut context = ConversationContext::with_system_prompt(
        "You are running inside the deterministic fake voice demo.",
    );

    println!(
        "ai-runtime voice-fake. Type `exit` or `quit` to stop. Press Ctrl-C to cancel a turn."
    );

    loop {
        print!("audio-text> ");
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
        let (tts_tx, tts_rx) = mpsc::unbounded_channel();
        let (voice_tx, voice_rx) = mpsc::unbounded_channel();
        let tts_task = task::spawn(print_tts_events(tts_rx));
        let voice_task = task::spawn(print_voice_events(voice_rx));
        let turn = pipeline.run_audio_turn_with_options(
            &mut context,
            fake_audio_stream(input.to_string()),
            VoiceTurnOptions::new(cancellation_token.clone())
                .with_tts_events(tts_tx)
                .with_voice_events(voice_tx),
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

        let _ = tts_task.await;
        let latency_snapshot = voice_task.await?;
        print_voice_latency_snapshot(&latency_snapshot);

        match result {
            Ok(output) => println!("assistant-final> {}", output.text),
            Err(VoiceTurnError::Cancelled) => println!("assistant-final> turn cancelled"),
            Err(err) => println!("assistant-final> error: {err}"),
        }
    }

    Ok(())
}

async fn deepgram_tts_smoke(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let text = parse_deepgram_tts_smoke_args(args)?;
    let config = DeepgramConfig::from_env()?;
    let tts = DeepgramTextToSpeech::new(config);
    let mut events = tts.stream_text(Box::pin(futures_util::stream::iter(vec![text])));
    let mut chunks = 0usize;
    let mut bytes = 0usize;

    while let Some(event) = events.next().await {
        match event? {
            TtsEvent::FirstAudio { elapsed_ms } => {
                println!("deepgram-tts> first_audio elapsed_ms={elapsed_ms}");
            }
            TtsEvent::AudioChunk {
                bytes: chunk,
                elapsed_ms,
            } => {
                chunks += 1;
                bytes += chunk.len();
                println!(
                    "deepgram-tts> audio_chunk elapsed_ms={elapsed_ms} bytes={}",
                    chunk.len()
                );
            }
            TtsEvent::Done => {
                println!("deepgram-tts> done chunks={chunks} bytes={bytes}");
                break;
            }
        }
    }

    Ok(())
}

async fn deepgram_stt_file(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let path = parse_deepgram_stt_file_args(args)?;
    let audio = std::fs::read(&path)?;
    let config = DeepgramConfig::from_env()?;
    let stt = DeepgramSpeechToText::new(config);
    let mut events = stt.stream_audio(paced_audio_stream(raw_pcm_audio_chunks(&audio)));

    print_deepgram_stt_events(&mut events).await
}

async fn deepgram_loopback_smoke(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let text = parse_deepgram_loopback_smoke_args(args)?;
    let config = DeepgramConfig::from_env()?;
    let tts = DeepgramTextToSpeech::new(config.clone().with_speak_url(
        "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&container=none&sample_rate=16000",
    ));
    let audio = collect_deepgram_tts_audio(&tts, text).await?;
    println!(
        "deepgram-loopback> synthesized_raw_pcm_bytes={}",
        audio.len()
    );

    let stt = DeepgramSpeechToText::new(config);
    let mut events = stt.stream_audio(paced_audio_stream(raw_pcm_audio_chunks(&audio)));
    print_deepgram_stt_events(&mut events).await
}

async fn livekit_token_smoke(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    parse_livekit_token_smoke_args(args)?;
    let config = LiveKitConfig::from_env()?;
    let token = config.access_token()?;

    println!(
        "livekit-token> url={} room={} identity={} token_bytes={}",
        config.url,
        config.room,
        config.identity,
        token.len()
    );

    Ok(())
}

#[cfg(feature = "livekit-transport")]
async fn voice_livekit(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let tools_enabled = parse_voice_livekit_args(args)?;
    let livekit = LiveKitConfig::from_env()?;
    let deepgram = DeepgramConfig::from_env()?;
    let api_key = env::var("OPENAI_COMPAT_API_KEY")
        .map_err(|_| "OPENAI_COMPAT_API_KEY must be set to run `ai-runtime voice-livekit`")?;
    let base_url =
        env::var("OPENAI_COMPAT_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let model = env::var("OPENAI_COMPAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let provider = OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, api_key))?;
    let tools = if tools_enabled {
        demo_tools()?
    } else {
        ToolRegistry::new()
    };
    let transport = LiveKitVoiceTransport::connect(livekit.clone()).await?;
    let (voice_tx, voice_rx) = mpsc::unbounded_channel();
    let voice_task = task::spawn(print_voice_events(voice_rx));
    let mut session = VoiceSession::new(
        provider,
        tools,
        RuntimeConfig::new(model),
        DeepgramSpeechToText::new(deepgram.clone()),
        DeepgramTextToSpeech::new(deepgram),
        ConversationContext::with_system_prompt("You are a concise real-time voice assistant."),
    )
    .with_voice_events(voice_tx);

    println!(
        "voice-livekit> joined room={} identity={}",
        livekit.room, livekit.identity
    );
    println!("voice-livekit> speak from another LiveKit participant; press Ctrl-C to stop");

    let result = {
        let run = session.run_transport(transport);
        tokio::pin!(run);
        tokio::select! {
            result = &mut run => result,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                Ok(())
            }
        }
    };

    session.cancel_active_turn().await?;
    drop(session);
    let latency_snapshot = voice_task.await?;
    print_voice_latency_snapshot(&latency_snapshot);

    result?;
    Ok(())
}

#[cfg(not(feature = "livekit-transport"))]
async fn voice_livekit(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    parse_voice_livekit_args(args)?;
    Err(
        "voice-livekit requires `cargo run --features livekit-transport -- voice-livekit`; this machine also needs clang++ 21+ for LiveKit WebRTC"
            .into(),
    )
}

async fn collect_deepgram_tts_audio(
    tts: &DeepgramTextToSpeech,
    text: String,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut events = tts.stream_text(Box::pin(futures_util::stream::iter(vec![text])));
    let mut audio = Vec::new();

    while let Some(event) = events.next().await {
        match event? {
            TtsEvent::FirstAudio { elapsed_ms } => {
                println!("deepgram-tts> first_audio elapsed_ms={elapsed_ms}");
            }
            TtsEvent::AudioChunk { bytes, .. } => audio.extend(bytes),
            TtsEvent::Done => break,
        }
    }

    Ok(audio)
}

async fn print_deepgram_stt_events(
    events: &mut BoxStream<'static, ai_runtime::SttResult<ai_runtime::TranscriptEvent>>,
) -> Result<(), Box<dyn Error>> {
    while let Some(event) = events.next().await {
        match event? {
            ai_runtime::TranscriptEvent::Partial { text, elapsed_ms } => {
                println!("deepgram-stt> partial elapsed_ms={elapsed_ms} text={text:?}");
            }
            ai_runtime::TranscriptEvent::Final { text, elapsed_ms } => {
                println!("deepgram-stt> final elapsed_ms={elapsed_ms} text={text:?}");
            }
            ai_runtime::TranscriptEvent::Done => {
                println!("deepgram-stt> done");
                break;
            }
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

fn parse_voice_fake_args(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => {
                print_voice_fake_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown voice-fake option: {other}").into()),
        }
    }

    Ok(())
}

fn parse_deepgram_tts_smoke_args(args: Vec<String>) -> Result<String, Box<dyn Error>> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_deepgram_tts_smoke_usage();
        std::process::exit(0);
    }

    if args.is_empty() {
        Ok("Hello from the Deepgram TTS smoke test.".to_string())
    } else {
        Ok(args.join(" "))
    }
}

fn parse_deepgram_stt_file_args(args: Vec<String>) -> Result<String, Box<dyn Error>> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_deepgram_stt_file_usage();
        std::process::exit(0);
    }

    match args.as_slice() {
        [path] => Ok(path.clone()),
        _ => Err("usage: ai-runtime deepgram-stt-file <raw-16khz-mono-linear16-pcm-file>".into()),
    }
}

fn parse_deepgram_loopback_smoke_args(args: Vec<String>) -> Result<String, Box<dyn Error>> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_deepgram_loopback_smoke_usage();
        std::process::exit(0);
    }

    if args.is_empty() {
        Ok("Hello from the Deepgram loopback smoke test.".to_string())
    } else {
        Ok(args.join(" "))
    }
}

fn parse_livekit_token_smoke_args(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => {
                print_livekit_token_smoke_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown livekit-token-smoke option: {other}").into()),
        }
    }

    Ok(())
}

fn parse_voice_livekit_args(args: Vec<String>) -> Result<bool, Box<dyn Error>> {
    let mut tools_enabled = false;

    for arg in args {
        match arg.as_str() {
            "--tools" => tools_enabled = true,
            "--no-tools" => tools_enabled = false,
            "-h" | "--help" => {
                print_voice_livekit_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown voice-livekit option: {other}").into()),
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

fn fake_audio_stream(text: String) -> BoxStream<'static, AudioChunk> {
    let chunks = fake_audio_chunks(&text);
    Box::pin(futures_util::stream::iter(chunks))
}

fn fake_audio_chunks(text: &str) -> Vec<AudioChunk> {
    let mut elapsed_ms = 0;

    text.split_inclusive(' ')
        .map(|fragment| {
            elapsed_ms += 40;
            AudioChunk::new(fragment.as_bytes().to_vec()).with_elapsed_ms(elapsed_ms)
        })
        .collect()
}

fn raw_pcm_audio_chunks(bytes: &[u8]) -> Vec<AudioChunk> {
    const CHUNK_BYTES: usize = 3_200;
    const CHUNK_MS: u128 = 100;

    bytes
        .chunks(CHUNK_BYTES)
        .enumerate()
        .map(|(index, chunk)| {
            AudioChunk::new(chunk.to_vec()).with_elapsed_ms((index as u128 + 1) * CHUNK_MS)
        })
        .collect()
}

fn paced_audio_stream(chunks: Vec<AudioChunk>) -> BoxStream<'static, AudioChunk> {
    Box::pin(futures_util::stream::unfold(
        chunks.into_iter(),
        |mut chunks| async move {
            let chunk = chunks.next()?;
            tokio::time::sleep(Duration::from_millis(100)).await;
            Some((chunk, chunks))
        },
    ))
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

async fn print_tts_events(mut events: mpsc::UnboundedReceiver<TtsEvent>) {
    while let Some(event) = events.recv().await {
        match event {
            TtsEvent::FirstAudio { elapsed_ms } => {
                println!("tts> first_audio elapsed_ms={elapsed_ms}");
            }
            TtsEvent::AudioChunk { bytes, elapsed_ms } => {
                println!(
                    "tts> audio_chunk elapsed_ms={elapsed_ms} bytes={}",
                    String::from_utf8_lossy(&bytes)
                );
            }
            TtsEvent::Done => println!("tts> done"),
        }
    }
}

async fn print_voice_events(
    mut events: mpsc::UnboundedReceiver<VoiceEvent>,
) -> VoiceLatencySnapshot {
    let mut latency = VoiceLatencyRecorder::new();

    while let Some(event) = events.recv().await {
        latency.record_event(&event);

        match event {
            VoiceEvent::SttPartial {
                elapsed_ms,
                stt_elapsed_ms,
                transcript_chars,
            } => {
                println!(
                    "voice> stt_partial elapsed_ms={elapsed_ms} stt_elapsed_ms={stt_elapsed_ms} transcript_chars={transcript_chars}"
                );
            }
            VoiceEvent::SttFinal {
                elapsed_ms,
                stt_elapsed_ms,
                transcript_chars,
            } => {
                println!(
                    "voice> stt_final elapsed_ms={elapsed_ms} stt_elapsed_ms={stt_elapsed_ms} transcript_chars={transcript_chars}"
                );
            }
            VoiceEvent::LlmFirstToken {
                elapsed_ms,
                runtime_elapsed_ms,
            } => {
                println!(
                    "voice> llm_first_token elapsed_ms={elapsed_ms} runtime_elapsed_ms={runtime_elapsed_ms}"
                );
            }
            VoiceEvent::TtsFirstAudio {
                elapsed_ms,
                tts_elapsed_ms,
            } => {
                println!(
                    "voice> tts_first_audio elapsed_ms={elapsed_ms} tts_elapsed_ms={tts_elapsed_ms}"
                );
            }
            VoiceEvent::VoiceTurnCompleted {
                elapsed_ms,
                output_chars,
            } => {
                println!(
                    "voice> turn_completed elapsed_ms={elapsed_ms} output_chars={output_chars}"
                );
            }
            VoiceEvent::VoiceTurnCancelled { elapsed_ms } => {
                println!("voice> turn_cancelled elapsed_ms={elapsed_ms}");
            }
            VoiceEvent::VoiceTurnInterrupted { elapsed_ms } => {
                println!("voice> turn_interrupted elapsed_ms={elapsed_ms}");
            }
            VoiceEvent::VoiceTurnFailed { elapsed_ms } => {
                println!("voice> turn_failed elapsed_ms={elapsed_ms}");
            }
        }
    }

    latency.snapshot()
}

fn print_voice_latency_snapshot(snapshot: &VoiceLatencySnapshot) {
    print_voice_latency_stat("stt_finalization", snapshot.stt_finalization);
    print_voice_latency_stat("llm_first_token", snapshot.llm_first_token);
    print_voice_latency_stat("tts_first_audio", snapshot.tts_first_audio);
    print_voice_latency_stat("voice_turn_round_trip", snapshot.voice_turn_round_trip);
}

fn print_voice_latency_stat(name: &str, stats: Option<VoiceLatencyStats>) {
    if let Some(stats) = stats {
        println!(
            "voice-metrics> {name} samples={} p50_ms={} p95_ms={}",
            stats.samples, stats.p50_ms, stats.p95_ms
        );
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
    println!("usage:");
    println!("  ai-runtime chat [--tools|--no-tools]");
    println!("  ai-runtime voice-fake");
    println!("  ai-runtime deepgram-tts-smoke [text]");
    println!("  ai-runtime deepgram-stt-file <raw-16khz-mono-linear16-pcm-file>");
    println!("  ai-runtime deepgram-loopback-smoke [text]");
    println!("  ai-runtime livekit-token-smoke");
    println!("  ai-runtime voice-livekit [--tools|--no-tools]");
    println!("env for chat: OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_MODEL, OPENAI_COMPAT_BASE_URL");
    println!("env for Deepgram: DEEPGRAM_API_KEY, DEEPGRAM_LISTEN_URL, DEEPGRAM_SPEAK_URL");
    println!("env for LiveKit: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET");
}

fn print_chat_usage() {
    println!("usage: ai-runtime chat [--tools|--no-tools]");
    println!("  --tools     register calculator and mock_crm_lookup demo tools");
    println!("  --no-tools  run LLM-only chat (default)");
}

fn print_voice_fake_usage() {
    println!("usage: ai-runtime voice-fake");
    println!(
        "  Runs typed text through fake audio, fake STT, deterministic LLM tokens, and fake TTS."
    );
}

fn print_deepgram_tts_smoke_usage() {
    println!("usage: ai-runtime deepgram-tts-smoke [text]");
    println!("  Streams Deepgram Aura-2 TTS audio and prints chunk sizes.");
}

fn print_deepgram_stt_file_usage() {
    println!("usage: ai-runtime deepgram-stt-file <raw-16khz-mono-linear16-pcm-file>");
    println!("  Streams raw PCM bytes into Deepgram STT and prints transcript events.");
}

fn print_deepgram_loopback_smoke_usage() {
    println!("usage: ai-runtime deepgram-loopback-smoke [text]");
    println!(
        "  Synthesizes 16kHz linear16 audio with Deepgram TTS, then transcribes it with Deepgram STT."
    );
}

fn print_livekit_token_smoke_usage() {
    println!("usage: ai-runtime livekit-token-smoke");
    println!(
        "  Loads LiveKit env vars and generates a room join token without printing the token."
    );
}

fn print_voice_livekit_usage() {
    println!("usage: ai-runtime voice-livekit [--tools|--no-tools]");
    println!(
        "  Runs LiveKit audio transport with Deepgram STT/TTS and the OpenAI-compatible LLM provider."
    );
    println!("  Build with: cargo run --features livekit-transport -- voice-livekit");
}

struct FakeVoiceProvider;

impl LlmProvider for FakeVoiceProvider {
    fn stream_chat(&self, req: ChatRequest) -> BoxStream<'static, ai_runtime::Result<TokenEvent>> {
        let user_text = req
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map(|message| message.content.as_str())
            .unwrap_or("");
        let response = format!("fake voice response: {user_text}");
        let mut events = response
            .split_inclusive(' ')
            .map(|text| {
                Ok(TokenEvent::Token {
                    text: text.to_string(),
                })
            })
            .collect::<Vec<_>>();
        events.push(Ok(TokenEvent::Done));
        Box::pin(futures_util::stream::iter(events))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_audio_chunks_preserve_text_and_assign_deterministic_elapsed_ms() {
        let chunks = fake_audio_chunks("hello fake runtime");

        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.bytes.clone())
                .collect::<Vec<_>>(),
            vec![b"hello ".to_vec(), b"fake ".to_vec(), b"runtime".to_vec()]
        );
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.elapsed_ms)
                .collect::<Vec<_>>(),
            vec![40, 80, 120]
        );
    }

    #[test]
    fn raw_pcm_audio_chunks_assign_100ms_boundaries() {
        let input = vec![7; 6_500];

        let chunks = raw_pcm_audio_chunks(&input);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].bytes.len(), 3_200);
        assert_eq!(chunks[0].elapsed_ms, 100);
        assert_eq!(chunks[1].bytes.len(), 3_200);
        assert_eq!(chunks[1].elapsed_ms, 200);
        assert_eq!(chunks[2].bytes.len(), 100);
        assert_eq!(chunks[2].elapsed_ms, 300);
    }
}
