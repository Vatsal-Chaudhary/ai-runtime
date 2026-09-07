use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use bytes::Bytes;
use futures_core::{Stream, stream::BoxStream};
use futures_util::StreamExt;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::info;

use super::{ChatMessage, ChatRequest, LlmError, LlmProvider, Result, TokenEvent};

static ACTIVE_STREAM_TASKS: AtomicUsize = AtomicUsize::new(0);

pub fn active_stream_task_count() -> usize {
    ACTIVE_STREAM_TASKS.load(Ordering::SeqCst)
}

#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
    pub stream_buffer_capacity: usize,
}

impl OpenAiConfig {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(30),
            stream_buffer_capacity: 8,
        }
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    client: Client,
    config: Arc<OpenAiConfig>,
    metrics: Arc<LatencyMetrics>,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(map_reqwest_error)?;

        Ok(Self {
            client,
            config: Arc::new(config),
            metrics: Arc::new(LatencyMetrics::default()),
        })
    }
}

impl LlmProvider for OpenAiCompatibleProvider {
    fn stream_chat(&self, req: ChatRequest) -> BoxStream<'static, Result<TokenEvent>> {
        let capacity = self.config.stream_buffer_capacity.max(1);
        let (tx, rx) = mpsc::channel(capacity);
        let client = self.client.clone();
        let config = Arc::clone(&self.config);
        let metrics = Arc::clone(&self.metrics);

        ACTIVE_STREAM_TASKS.fetch_add(1, Ordering::SeqCst);
        let handle = tokio::spawn(async move {
            let _guard = ActiveStreamTaskGuard;
            if let Err(err) = run_stream(client, config, metrics, req, tx.clone()).await {
                let _ = tx.send(Err(err)).await;
            }
        });

        Box::pin(ProviderStream { rx, handle })
    }
}

struct ProviderStream {
    rx: mpsc::Receiver<Result<TokenEvent>>,
    handle: JoinHandle<()>,
}

impl Stream for ProviderStream {
    type Item = Result<TokenEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        Pin::new(&mut this.rx).poll_recv(cx)
    }
}

impl Drop for ProviderStream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct ActiveStreamTaskGuard;

impl Drop for ActiveStreamTaskGuard {
    fn drop(&mut self) {
        ACTIVE_STREAM_TASKS.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Default)]
struct LatencyMetrics {
    first_token_latencies: Mutex<Vec<Duration>>,
}

impl LatencyMetrics {
    fn record_first_token_latency(&self, latency: Duration) {
        let (samples, p50) = {
            let mut latencies = self
                .first_token_latencies
                .lock()
                .expect("latency metrics mutex poisoned");
            latencies.push(latency);

            let mut sorted = latencies.clone();
            sorted.sort_unstable();
            let mid = (sorted.len() - 1) / 2;
            (sorted.len(), sorted[mid])
        };

        info!(
            target: "ai_runtime::llm",
            first_token_latency_ms = latency.as_millis() as u64,
            p50_first_token_latency_ms = p50.as_millis() as u64,
            samples,
            "llm first token latency"
        );
    }
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

async fn run_stream(
    client: Client,
    config: Arc<OpenAiConfig>,
    metrics: Arc<LatencyMetrics>,
    req: ChatRequest,
    tx: mpsc::Sender<Result<TokenEvent>>,
) -> Result<()> {
    let url = config.chat_completions_url();
    let request_timeout = req.timeout.unwrap_or(config.timeout);
    let body = OpenAiChatRequest {
        model: req.model,
        messages: req.messages,
        stream: true,
    };

    let started = Instant::now();
    let response = client
        .post(url)
        .bearer_auth(&config.api_key)
        .header(header::ACCEPT, "text/event-stream")
        .timeout(request_timeout)
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest_error)?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|err| err.to_string());
        return Err(LlmError::HttpStatus { status, body });
    }

    let mut decoder = SseDecoder::default();
    let mut byte_stream = response.bytes_stream();
    let mut first_token_recorded = false;

    loop {
        while let Some(data) = decoder.next_data()? {
            match parse_openai_event(&data)? {
                ParsedOpenAiEvent::Token(text) => {
                    if !first_token_recorded {
                        metrics.record_first_token_latency(started.elapsed());
                        first_token_recorded = true;
                    }

                    if tx.send(Ok(TokenEvent::Token { text })).await.is_err() {
                        return Ok(());
                    }
                }
                ParsedOpenAiEvent::Done => {
                    let _ = tx.send(Ok(TokenEvent::Done)).await;
                    return Ok(());
                }
                ParsedOpenAiEvent::Ignored => {}
            }
        }

        match byte_stream.next().await {
            Some(Ok(bytes)) => decoder.push(bytes),
            Some(Err(err)) => return Err(map_reqwest_error(err)),
            None if decoder.has_incomplete_event() => return Err(LlmError::IncompleteSseEvent),
            None => return Ok(()),
        }
    }
}

fn map_reqwest_error(err: reqwest::Error) -> LlmError {
    if err.is_timeout() {
        LlmError::ProviderTimeout
    } else {
        LlmError::ProviderRequest(err)
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    pending: VecDeque<String>,
}

impl SseDecoder {
    fn push(&mut self, bytes: Bytes) {
        self.buffer.extend_from_slice(&bytes);
        self.collect_complete_events();
    }

    fn next_data(&mut self) -> Result<Option<String>> {
        loop {
            let Some(raw) = self.pending.pop_front() else {
                return Ok(None);
            };
            let data = event_data(&raw)?;
            if let Some(data) = data {
                return Ok(Some(data));
            }
        }
    }

    fn has_incomplete_event(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn collect_complete_events(&mut self) {
        while let Some((event_end, delimiter_len)) = find_event_boundary(&self.buffer) {
            let event = self.buffer.drain(..event_end).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            self.pending
                .push_back(String::from_utf8_lossy(&event).to_string());
        }
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len().saturating_sub(1) {
        if buffer[index] == b'\n' && buffer[index + 1] == b'\n' {
            return Some((index, 2));
        }

        if index + 3 < buffer.len() && &buffer[index..index + 4] == b"\r\n\r\n" {
            return Some((index, 4));
        }
    }

    None
}

fn event_data(raw: &str) -> Result<Option<String>> {
    let mut data_lines = Vec::new();

    for line in raw.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
    }

    if data_lines.is_empty() {
        Ok(None)
    } else {
        Ok(Some(data_lines.join("\n")))
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    delta: OpenAiDelta,
    finish_reason: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

enum ParsedOpenAiEvent {
    Token(String),
    Done,
    Ignored,
}

fn parse_openai_event(data: &str) -> Result<ParsedOpenAiEvent> {
    if data == "[DONE]" {
        return Ok(ParsedOpenAiEvent::Done);
    }

    let chunk =
        serde_json::from_str::<OpenAiStreamChunk>(data).map_err(|err| LlmError::MalformedSse {
            message: format!("invalid OpenAI-compatible JSON event: {err}; data={data:?}"),
        })?;

    let Some(choice) = chunk.choices.into_iter().next() else {
        return Ok(ParsedOpenAiEvent::Ignored);
    };

    if let Some(content) = choice.delta.content
        && !content.is_empty()
    {
        return Ok(ParsedOpenAiEvent::Token(content));
    }

    if choice.finish_reason.is_some() {
        Ok(ParsedOpenAiEvent::Done)
    } else {
        Ok(ParsedOpenAiEvent::Ignored)
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, time::Duration};

    use futures_util::StreamExt;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::Mutex as AsyncMutex,
        time::{sleep, timeout},
    };
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    use super::*;
    use crate::llm::{ChatMessage, ChatRequest};

    static TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_chunks_stream_tokens_and_done() {
        let _guard = TEST_LOCK.lock().await;
        init_tracing();

        let body = [
            sse_chunk("Hello"),
            sse_chunk(" world"),
            "data: [DONE]\n\n".to_string(),
        ]
        .join("");
        let server = MockSseServer::start([ServerWrite::bytes(body)]).await;
        let provider = provider_for(&server, Duration::from_secs(5));

        let events = collect_stream(provider.stream_chat(test_request())).await;

        assert_eq!(
            events.expect("stream should succeed"),
            vec![
                TokenEvent::Token {
                    text: "Hello".to_string()
                },
                TokenEvent::Token {
                    text: " world".to_string()
                },
                TokenEvent::Done,
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn partial_chunks_across_tcp_reads_are_reassembled() {
        let _guard = TEST_LOCK.lock().await;
        init_tracing();

        let chunk = sse_chunk("split-token");
        let split_at = chunk.find("split").expect("fixture should contain split");
        let server = MockSseServer::start([
            ServerWrite::bytes(chunk[..split_at].to_string()),
            ServerWrite::delay(Duration::from_millis(10)),
            ServerWrite::bytes(chunk[split_at..].to_string()),
            ServerWrite::bytes("data: [DONE]\n\n"),
        ])
        .await;
        let provider = provider_for(&server, Duration::from_secs(5));

        let events = collect_stream(provider.stream_chat(test_request())).await;

        assert_eq!(
            events.expect("stream should succeed"),
            vec![
                TokenEvent::Token {
                    text: "split-token".to_string()
                },
                TokenEvent::Done,
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn malformed_chunks_return_typed_error_without_panic() {
        let _guard = TEST_LOCK.lock().await;
        init_tracing();

        let server = MockSseServer::start([ServerWrite::bytes("data: {not-json}\n\n")]).await;
        let provider = provider_for(&server, Duration::from_secs(5));

        let mut stream = provider.stream_chat(test_request());
        let err = stream
            .next()
            .await
            .expect("stream should produce an error")
            .expect_err("malformed chunk should be an error");

        assert!(matches!(err, LlmError::MalformedSse { .. }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provider_timeout_returns_typed_error() {
        let _guard = TEST_LOCK.lock().await;
        init_tracing();

        let server = MockSseServer::start([
            ServerWrite::delay(Duration::from_millis(250)),
            ServerWrite::bytes(sse_chunk("late")),
        ])
        .await;
        let provider = provider_for(&server, Duration::from_millis(50));

        let mut stream = provider.stream_chat(test_request());
        let err = stream
            .next()
            .await
            .expect("stream should produce an error")
            .expect_err("timeout should be an error");

        assert!(matches!(err, LlmError::ProviderTimeout));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_stream_mid_response_cancels_provider_task() {
        let _guard = TEST_LOCK.lock().await;
        init_tracing();

        assert_eq!(active_stream_task_count(), 0);
        let server = MockSseServer::start([
            ServerWrite::bytes(sse_chunk("first")),
            ServerWrite::delay(Duration::from_secs(30)),
            ServerWrite::bytes(sse_chunk("never-read")),
        ])
        .await;
        let provider = provider_for(&server, Duration::from_secs(60));

        let mut stream = provider.stream_chat(test_request());
        assert_eq!(
            stream.next().await.expect("first event").expect("token"),
            TokenEvent::Token {
                text: "first".to_string()
            }
        );
        assert_eq!(active_stream_task_count(), 1);

        drop(stream);
        wait_for_no_active_stream_tasks().await;
        assert_eq!(active_stream_task_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires OPENAI_COMPAT_API_KEY and optionally OPENAI_COMPAT_BASE_URL/OPENAI_COMPAT_MODEL"]
    async fn real_openai_compatible_call_logs_p50_first_token_latency() {
        init_tracing();

        let api_key = std::env::var("OPENAI_COMPAT_API_KEY")
            .expect("OPENAI_COMPAT_API_KEY must be set for real provider smoke test");
        let base_url = std::env::var("OPENAI_COMPAT_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model =
            std::env::var("OPENAI_COMPAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let provider = OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, api_key))
            .expect("provider should initialize");
        let req = ChatRequest::new(
            model,
            vec![ChatMessage::user(
                "Reply with exactly one short sentence for a streaming smoke test.",
            )],
        );

        let events = collect_stream(provider.stream_chat(req))
            .await
            .expect("real stream should succeed");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, TokenEvent::Token { .. }))
        );
    }

    async fn collect_stream(
        mut stream: BoxStream<'static, Result<TokenEvent>>,
    ) -> Result<Vec<TokenEvent>> {
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event?);
        }
        Ok(events)
    }

    async fn wait_for_no_active_stream_tasks() {
        timeout(Duration::from_secs(1), async {
            while active_stream_task_count() != 0 {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("provider stream task should stop after stream drop");
    }

    fn provider_for(server: &MockSseServer, timeout: Duration) -> OpenAiCompatibleProvider {
        let mut config = OpenAiConfig::new(server.base_url(), "test-key");
        config.timeout = timeout;
        config.stream_buffer_capacity = 1;
        OpenAiCompatibleProvider::new(config).expect("provider should initialize")
    }

    fn test_request() -> ChatRequest {
        ChatRequest::new("test-model", vec![ChatMessage::user("hello")])
    }

    fn sse_chunk(content: &str) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{
                    "delta": { "content": content },
                    "finish_reason": null
                }]
            })
        )
    }

    fn init_tracing() {
        let _ = tracing_subscriber::registry()
            .with(fmt::layer().with_test_writer())
            .with(EnvFilter::new("ai_runtime=trace"))
            .try_init();
    }

    enum ServerWrite {
        Bytes(Vec<u8>),
        Delay(Duration),
    }

    impl ServerWrite {
        fn bytes(bytes: impl Into<Vec<u8>>) -> Self {
            Self::Bytes(bytes.into())
        }

        fn delay(duration: Duration) -> Self {
            Self::Delay(duration)
        }
    }

    struct MockSseServer {
        addr: std::net::SocketAddr,
        handle: JoinHandle<()>,
    }

    impl MockSseServer {
        async fn start<const N: usize>(writes: [ServerWrite; N]) -> Self {
            Self::start_with(writes, |_| async {}).await
        }

        async fn start_with<const N: usize, F, Fut>(
            writes: [ServerWrite; N],
            before_body: F,
        ) -> Self
        where
            F: FnOnce(Vec<u8>) -> Fut + Send + 'static,
            Fut: Future<Output = ()> + Send + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("mock server should bind");
            let addr = listener.local_addr().expect("mock server should have addr");

            let handle = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.expect("mock server should accept");
                let mut request = Vec::new();
                let mut buf = [0_u8; 1024];
                loop {
                    let n = socket
                        .read(&mut buf)
                        .await
                        .expect("mock server should read request");
                    if n == 0 {
                        return;
                    }
                    request.extend_from_slice(&buf[..n]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }

                before_body(request).await;

                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .expect("mock server should write response headers");

                for write in writes {
                    match write {
                        ServerWrite::Bytes(bytes) => {
                            if socket.write_all(&bytes).await.is_err() {
                                return;
                            }
                            if socket.flush().await.is_err() {
                                return;
                            }
                        }
                        ServerWrite::Delay(duration) => sleep(duration).await,
                    }
                }
            });

            Self { addr, handle }
        }

        fn base_url(&self) -> String {
            format!("http://{}/v1", self.addr)
        }
    }

    impl Drop for MockSseServer {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }
}
