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
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

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
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    #[serde(default)]
    agent_session_id: Option<String>,
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
    /// Normalized outcomes from significant tool calls (build, test, publish, git, etc.).
    /// Each entry is a short fact: "succeeded" or "failed: <snippet>". Optional.
    #[serde(default)]
    tool_calls: Vec<crate::types::ToolCall>,
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    #[serde(default)]
    agent_session_id: Option<String>,
    /// Best-effort assistant usage metrics (tokens/cost). Optional.
    #[serde(default)]
    usage: Option<UsageParams>,
    /// OpenCode turn key (user_msg_id:assistant_msg_id) for deduplication
    #[serde(default)]
    turn_key: Option<String>,
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
            agent_session_id: p.agent_session_id,
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

        let source_pointer = match (p.agent_session_id.as_deref(), p.turn_key.as_deref()) {
            (Some(session), Some(turn_key))
                if !session.trim().is_empty() && !turn_key.trim().is_empty() =>
            {
                crate::companion::shims::opencode::build_opencode_source_pointer(
                    session, turn_key,
                )
            }
            (Some(session), _) if !session.trim().is_empty() => Some(format!(
                "opencode+message://{}",
                session.trim()
            )),
            _ => None,
        };

        RecordTurnEvent {
            directory: p.directory,
            user_text: p.user_text,
            assistant_text: p.assistant_text,
            touched_paths: p.touched_paths,
            tool_calls: p.tool_calls,
            agent_kind: AgentKind::OpenCode,
            agent_session_id: p.agent_session_id,
            usage,
            grounding_note: None,
            source_ts_ms: None,
            source_pointer,
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
pub async fn run(embed_model: String, embed_cache_dir: Option<String>, no_extraction: bool) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let extraction_mode = if no_extraction {
        crate::types::ExtractionMode::None
    } else {
        crate::types::ExtractionMode::Hybrid // stdio shim is for live, so use hybrid by default
    };
    let config = FlowConfig {
        embed_model: embed_model.clone(),
        embed_cache_dir: embed_cache_dir.clone(),
        extraction_mode,
    };
    let mut flow = Flow::new(config);

    // Signal readiness
    let ready = serde_json::json!({"ready": true});
    writeln!(stdout, "{}", serde_json::to_string(&ready)?)?;
    stdout.flush()?;

    // Track the last known workspace directory across record calls so we can
    // trigger incremental ingest when the session ends (stdin closes).
    let mut last_directory: Option<String> = None;
    // Accumulate all touched paths seen this session for changelog detection.
    let mut all_touched: Vec<String> = Vec::new();

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
                let result = flow.check_turn(event).await;
                result.into()
            }
            "record" => {
                let params: RecordParams = serde_json::from_value(req.params).unwrap_or_default();
                let dir_path = Path::new(&params.directory);

                // Track directory and touched paths for end-of-session ingest.
                if !params.directory.is_empty() {
                    last_directory = Some(params.directory.clone());
                }
                all_touched.extend(params.touched_paths.iter().cloned());

                // Stealth PR comment: detect when the agent created a GitHub PR and
                // spawn `unlost pr-comment` in the background without blocking the agent.
                if let Some(pr_url) =
                    detect_pr_creation_in_tool_calls(&params.tool_calls)
                {
                    let session_id = params.agent_session_id.clone();
                    let directory = params.directory.clone();
                    let em = embed_model.clone();
                    let ec = embed_cache_dir.clone();
                    tokio::spawn(async move {
                        tracing::info!(pr_url, "detected PR creation — posting unlost comment");
                        // Small delay so gh has time to fully register the PR.
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        if let Err(e) = spawn_pr_comment(
                            &pr_url,
                            session_id.as_deref(),
                            &directory,
                            &em,
                            ec.as_deref(),
                        )
                        .await
                        {
                            tracing::warn!(error = ?e, pr_url, "pr-comment spawn failed");
                        }
                    });
                }

                // Try deduplication if turn_key is provided
                let should_record = if let Some(ref tk) = params.turn_key {
                    if let Ok(ws) = get_or_create_workspace_paths(dir_path) {
                        let seen = crate::companion::shims::opencode::load_replayed(&ws.id);
                        if seen.contains(tk) {
                            false
                        } else {
                            crate::companion::shims::opencode::append_replayed(
                                &ws.id,
                                &[tk.clone()],
                            );
                            true
                        }
                    } else {
                        true
                    }
                } else {
                    true
                };

                if should_record {
                    let event: RecordTurnEvent = params.into();
                    let result = flow.record_turn(event).await;
                    result.into()
                } else {
                    Response::Record(RecordResponse {
                        ok: true,
                        error: None,
                    })
                }
            }
            _ => Response::Error {
                error: format!("unknown method: {}", req.method),
            },
        };

        writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        stdout.flush()?;
    }

    // ── Session end: drain + incremental ingest ───────────────────────────────
    // stdin closed = OpenCode session ended. Drain buffered capsules first, then
    // run the same incremental ingest as the Claude Stop hook: git tags (always)
    // and changelog (when touched or when un-ingested versions exist).
    flow.drain().await;

    if let Some(ref dir) = last_directory {
        let dir_path = Path::new(dir.as_str());
        if let Some(repo_root) = crate::workspace::git_toplevel(dir_path) {
            if let Ok(ws) = get_or_create_workspace_paths(dir_path) {
                let embedder = crate::embed::load_embedder(
                    &embed_model,
                    embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                    false,
                )
                .await;
                if let Ok(ref embedder) = embedder {
                    let use_color = std::io::IsTerminal::is_terminal(&std::io::stdout())
                        && std::env::var_os("NO_COLOR").is_none();
                    let _ =
                        crate::git::ingest_git_tags(&ws, &repo_root, embedder, use_color).await;
                    let changelog_path = repo_root.join("CHANGELOG.md");
                    let changelog_touched = all_touched
                        .iter()
                        .any(|p| p.ends_with("CHANGELOG.md") || p == "CHANGELOG.md");
                    if changelog_touched || changelog_path.exists() {
                        let _ = crate::changelog::ingest_changelog(
                            &ws,
                            &changelog_path,
                            embedder,
                            use_color,
                        )
                        .await;
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Stealth PR comment helpers
// ============================================================================

/// Scan tool_calls for evidence that the agent just created a GitHub PR.
/// Returns the PR URL if found, None otherwise.
fn detect_pr_creation_in_tool_calls(tool_calls: &[crate::types::ToolCall]) -> Option<String> {
    for tc in tool_calls {
        // Only inspect bash tool outputs — that's where `gh pr create` runs.
        if tc.name != "bash" {
            continue;
        }
        // Look for a GitHub PR URL in the output text.
        if let Some(url) = extract_github_pr_url(&tc.output) {
            return Some(url);
        }
    }
    None
}

/// Extract the first GitHub PR URL from a string, or None.
/// Public alias so sibling shims (e.g. claude) can reuse without duplication.
pub(crate) fn extract_github_pr_url_pub(text: &str) -> Option<String> {
    extract_github_pr_url(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_github_pr_url_full_url() {
        let text = "PR created: https://github.com/owner/repo/pull/42\nDone.";
        assert_eq!(
            extract_github_pr_url(text),
            Some("https://github.com/owner/repo/pull/42".to_string())
        );
    }

    #[test]
    fn test_extract_github_pr_url_trailing_punctuation_stripped() {
        let text = "See https://github.com/owner/repo/pull/42.";
        assert_eq!(
            extract_github_pr_url(text),
            Some("https://github.com/owner/repo/pull/42".to_string())
        );
    }

    #[test]
    fn test_extract_github_pr_url_no_pr() {
        let text = "Pushed to https://github.com/owner/repo/tree/main";
        assert_eq!(extract_github_pr_url(text), None);
    }

    #[test]
    fn test_extract_github_pr_url_empty() {
        assert_eq!(extract_github_pr_url(""), None);
    }

    #[test]
    fn test_detect_pr_creation_bash_tool_call() {
        let tool_calls = vec![crate::types::ToolCall {
            name: "bash".to_string(),
            output: "https://github.com/owner/repo/pull/7 created".to_string(),
        }];
        assert_eq!(
            detect_pr_creation_in_tool_calls(&tool_calls),
            Some("https://github.com/owner/repo/pull/7".to_string())
        );
    }

    #[test]
    fn test_detect_pr_creation_non_bash_ignored() {
        let tool_calls = vec![crate::types::ToolCall {
            name: "read_file".to_string(),
            output: "https://github.com/owner/repo/pull/7 created".to_string(),
        }];
        assert_eq!(detect_pr_creation_in_tool_calls(&tool_calls), None);
    }

    #[test]
    fn test_detect_pr_creation_empty() {
        assert_eq!(detect_pr_creation_in_tool_calls(&[]), None);
    }
}

fn extract_github_pr_url(text: &str) -> Option<String> {
    // Match: https://github.com/<owner>/<repo>/pull/<number>
    // Simple linear scan — no regex dependency.
    let marker = "github.com/";
    let mut pos = 0;
    while let Some(idx) = text[pos..].find(marker) {
        let start = pos + idx;
        // Backtrack to include the scheme.
        let url_start = text[..start]
            .rfind("https://")
            .map(|i| i)
            .unwrap_or(start);
        let candidate = &text[url_start..];
        // Find the end of the URL (whitespace or end of string).
        let end = candidate
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .unwrap_or(candidate.len());
        let url = &candidate[..end];
        if url.contains("/pull/") {
            return Some(url.trim_end_matches(|c| c == '.' || c == ',').to_string());
        }
        pos = start + marker.len();
    }
    None
}

/// Spawn `unlost pr-comment` as a subprocess in the given workspace directory.
/// Public alias so sibling shims can reuse.
pub(crate) async fn spawn_pr_comment_pub(
    pr_url: &str,
    session_id: Option<&str>,
    directory: &str,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    spawn_pr_comment(pr_url, session_id, directory, embed_model, embed_cache_dir).await
}

async fn spawn_pr_comment(
    pr_url: &str,
    session_id: Option<&str>,
    directory: &str,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    let unlost_bin = std::env::current_exe()
        .unwrap_or_else(|_| std::path::PathBuf::from("unlost"));

    let mut cmd = tokio::process::Command::new(unlost_bin);
    cmd.arg("pr-comment").arg(pr_url);

    if let Some(sid) = session_id {
        cmd.arg("--session-id").arg(sid);
    }

    cmd.arg("--embed-model").arg(embed_model);

    if let Some(ec) = embed_cache_dir {
        cmd.arg("--embed-cache-dir").arg(ec);
    }

    // Run in the workspace directory so unlost resolves the correct workspace.
    if !directory.is_empty() {
        cmd.current_dir(directory);
    }

    // Fire and forget — we don't care about the exit status here; failures
    // are logged inside the spawned process.
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let _ = cmd.spawn()?;
    Ok(())
}
