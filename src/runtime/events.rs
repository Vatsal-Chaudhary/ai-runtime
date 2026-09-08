use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    Started,
    StreamingLlm,
    DispatchingTool,
    Completed,
    Cancelled,
    Failed,
}

impl RequestState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeEvent {
    Lifecycle {
        from: Option<RequestState>,
        to: RequestState,
        elapsed_ms: u128,
    },
    LlmFirstToken {
        elapsed_ms: u128,
    },
    AssistantToken {
        text: String,
        elapsed_ms: u128,
    },
    ToolDispatchStart {
        call_id: String,
        tool_name: String,
        elapsed_ms: u128,
    },
    ToolDispatchEnd {
        call_id: String,
        tool_name: String,
        elapsed_ms: u128,
        duration_ms: u128,
        success: bool,
    },
    ToolTimeout {
        call_id: String,
        tool_name: String,
        timeout_ms: u128,
        elapsed_ms: u128,
    },
    TurnCompleted {
        elapsed_ms: u128,
        output_chars: usize,
    },
}

#[derive(Debug)]
pub struct RequestLifecycle {
    state: Option<RequestState>,
    started_at: Instant,
}

impl RequestLifecycle {
    pub fn new() -> Self {
        Self {
            state: None,
            started_at: Instant::now(),
        }
    }

    pub fn state(&self) -> Option<RequestState> {
        self.state
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.started_at.elapsed().as_millis()
    }

    pub fn transition(&mut self, next: RequestState) -> RuntimeEvent {
        let from = self.state;
        debug_assert!(
            from.is_none_or(|current| valid_transition(current, next)),
            "invalid runtime lifecycle transition from {from:?} to {next:?}"
        );
        self.state = Some(next);
        RuntimeEvent::Lifecycle {
            from,
            to: next,
            elapsed_ms: self.elapsed_ms(),
        }
    }
}

impl Default for RequestLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

fn valid_transition(from: RequestState, to: RequestState) -> bool {
    if from.is_terminal() {
        return false;
    }

    if from == to {
        return true;
    }

    matches!(
        (from, to),
        (RequestState::Started, RequestState::StreamingLlm)
            | (RequestState::Started, RequestState::Cancelled)
            | (RequestState::Started, RequestState::Failed)
            | (RequestState::StreamingLlm, RequestState::DispatchingTool)
            | (RequestState::StreamingLlm, RequestState::Completed)
            | (RequestState::StreamingLlm, RequestState::Cancelled)
            | (RequestState::StreamingLlm, RequestState::Failed)
            | (RequestState::DispatchingTool, RequestState::StreamingLlm)
            | (RequestState::DispatchingTool, RequestState::Completed)
            | (RequestState::DispatchingTool, RequestState::Cancelled)
            | (RequestState::DispatchingTool, RequestState::Failed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_tracks_state_transitions() {
        let mut lifecycle = RequestLifecycle::new();

        assert_eq!(lifecycle.state(), None);
        let event = lifecycle.transition(RequestState::Started);
        assert!(matches!(
            event,
            RuntimeEvent::Lifecycle {
                from: None,
                to: RequestState::Started,
                ..
            }
        ));

        lifecycle.transition(RequestState::StreamingLlm);
        lifecycle.transition(RequestState::Completed);

        assert_eq!(lifecycle.state(), Some(RequestState::Completed));
    }
}
