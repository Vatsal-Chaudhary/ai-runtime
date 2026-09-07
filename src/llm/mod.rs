use std::time::Duration;

use futures_core::stream::BoxStream;
use serde::Serialize;
use thiserror::Error;

pub mod openai;

pub type Result<T> = std::result::Result<T, LlmError>;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("provider request timed out")]
    ProviderTimeout,

    #[error("provider request failed: {0}")]
    ProviderRequest(#[from] reqwest::Error),

    #[error("provider returned HTTP {status}: {body}")]
    HttpStatus {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("malformed SSE chunk: {message}")]
    MalformedSse { message: String },

    #[error("stream ended with an incomplete SSE event")]
    IncompleteSseEvent,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub timeout: Option<Duration>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            timeout: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenEvent {
    Token { text: String },
    Done,
}

pub trait LlmProvider: Send + Sync {
    fn stream_chat(&self, req: ChatRequest) -> BoxStream<'static, Result<TokenEvent>>;
}
