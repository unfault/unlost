//! OpenCode stdio JSON-RPC shim.
//!
//! Runs as a child process of the OpenCode plugin. Reads JSON requests from stdin,
//! writes JSON responses to stdout. No HTTP server, no port, no baseURL changes.
//!
//! Protocol:
//!   # Before LLM call - check for friction
//!   Request:  {"method": "check", "params": {"text": "user message", "directory": "/path/to/workspace"}}
//!   Response: {"note": "warning to inject"} or {"note": null}
//!
//!   # After LLM call - record the exchange
//!   Request:  {"method": "record", "params": {"user_text": "...", "assistant_text": "...", "directory": "..."}}
//!   Response: {"ok": true}
//!
//! The plugin spawns `unlost shim opencode` on init and communicates over stdio.
//!
//! IMPORTANT: `record` returns immediately after enqueueing; heavy work (LLM extraction,
//! embedding, LanceDB insert) happens in a background task so we never block the agent.

use crate::companion::flow::{
    AgentKind, CheckEvent, CheckResult, Flow, FlowConfig, RecordResult, RecordTurnEvent, UsageEvent,
};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

// ============================================================================
// Wire protocol types (JSON-RPC style, but simplified)
// ============================================================================

#[derive(Debug, Deserialize)]
struct Request {
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Deserialize, Default)]
struct CheckParams {
    /// User's last message text
    #[serde(default)]
    text: String,
    /// Workspace directory (absolute path)
    #[serde(default)]
    directory: String,
}

#[derive(Debug, Deserialize, Default)]
struct RecordParams {
    /// User's message text
    #[serde(default)]
    user_text: String,
    /// Assistant's response text
    #[serde(default)]
    assistant_text: String,
    /// Workspace directory (absolute path)
    #[serde(default)]
    directory: String,
    /// Best-effort list of touched paths (workspace-relative). Optional.
    #[serde(default)]
    touched_paths: Vec<String>,
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    #[serde(default)]
    agent_session_id: Option<String>,
    /// Best-effort assistant usage metrics (tokens/cost). Optional.
    #[serde(default)]
    usage: Option<UsageParams>,
}

#[derive(Debug, Deserialize, Default)]
struct UsageTokensCacheParams {
    #[serde(default)]
    read: Option<i64>,
    #[serde(default)]
    write: Option<i64>,
}

#[derive(Debug, Deserialize, Default)]
struct UsageTokensParams {
    #[serde(default)]
    input: Option<i64>,
    #[serde(default)]
    output: Option<i64>,
    #[serde(default)]
    reasoning: Option<i64>,
    #[serde(default)]
    cache: Option<UsageTokensCacheParams>,
}

#[derive(Debug, Deserialize, Default)]
struct UsageParams {
    #[serde(default)]
    provider_id: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    tokens: Option<UsageTokensParams>,
}

#[derive(Debug, Serialize)]
struct CheckResponse {
    /// Warning note to inject, or null if no friction detected
    note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecordResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Check(CheckResponse),
    Record(RecordResponse),
    Error { error: String },
}

// ============================================================================
// Conversions from wire types to flow events
// ============================================================================

impl From<CheckParams> for CheckEvent {
    fn from(p: CheckParams) -> Self {
        CheckEvent {
            directory: p.directory,
            text: p.text,
            agent_kind: AgentKind::OpenCode,
            agent_session_id: None,
        }
    }
}

impl From<RecordParams> for RecordTurnEvent {
    fn from(p: RecordParams) -> Self {
        let usage = p.usage.map(|u| UsageEvent {
            provider_id: u.provider_id,
            model_id: u.model_id,
            cost: u.cost,
            tokens_input: u.tokens.as_ref().and_then(|t| t.input),
            tokens_output: u.tokens.as_ref().and_then(|t| t.output),
            tokens_reasoning: u.tokens.as_ref().and_then(|t| t.reasoning),
            tokens_cache_read: u
                .tokens
                .as_ref()
                .and_then(|t| t.cache.as_ref())
                .and_then(|c| c.read),
            tokens_cache_write: u
                .tokens
                .as_ref()
                .and_then(|t| t.cache.as_ref())
                .and_then(|c| c.write),
        });

        RecordTurnEvent {
            directory: p.directory,
            user_text: p.user_text,
            assistant_text: p.assistant_text,
            touched_paths: p.touched_paths,
            agent_kind: AgentKind::OpenCode,
            agent_session_id: p.agent_session_id,
            usage,
        }
    }
}

impl From<CheckResult> for Response {
    fn from(r: CheckResult) -> Self {
        Response::Check(CheckResponse {
            note: r.note,
            error: r.error,
        })
    }
}

impl From<RecordResult> for Response {
    fn from(r: RecordResult) -> Self {
        Response::Record(RecordResponse {
            ok: r.ok,
            error: r.error,
        })
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Run the OpenCode stdio shim.
///
/// Reads JSON requests from stdin, processes them via the flow, and writes
/// JSON responses to stdout. Signals readiness with `{"ready": true}` on startup.
pub(crate) async fn run(
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let config = FlowConfig {
        embed_model,
        embed_cache_dir,
    };
    let mut flow = Flow::new(config);

    // Signal readiness
    let ready = serde_json::json!({"ready": true});
    writeln!(stdout, "{}", serde_json::to_string(&ready)?)?;
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::debug!("stdin read error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error {
                    error: format!("invalid JSON: {e}"),
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
                stdout.flush()?;
                continue;
            }
        };

        let resp = match req.method.as_str() {
            "check" => {
                let params: CheckParams = serde_json::from_value(req.params).unwrap_or_default();
                let event: CheckEvent = params.into();
                let result = flow.check(event).await;
                result.into()
            }
            "record" => {
                let params: RecordParams = serde_json::from_value(req.params).unwrap_or_default();
                let event: RecordTurnEvent = params.into();
                let result = flow.record_turn(event).await;
                result.into()
            }
            _ => Response::Error {
                error: format!("unknown method: {}", req.method),
            },
        };

        writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        stdout.flush()?;
    }

    Ok(())
}
