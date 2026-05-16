//! GitHub Copilot CLI hooks shim.
//!
//! Invoked by Copilot CLI hooks (sessionStart, userPromptSubmitted, sessionEnd)
//! via stdin JSON. Dispatches to Flow::check_turn() or events.jsonl ingestion.
//!
//! # Hook payload format (actual, from observation)
//!
//! Copilot CLI does NOT send a `hook_event_name` field. Event type is inferred
//! from which fields are present:
//!   - `initialPrompt` present → sessionStart
//!   - `prompt` present (no `initialPrompt`) → userPromptSubmitted
//!   - `reason` present → sessionEnd
//!
//! All payloads include `sessionId` directly — no discovery heuristics needed.
//!
//! Example payloads (from /tmp/z session capture):
//!   sessionStart:          {"sessionId":"...","timestamp":...,"cwd":"...","source":"new","initialPrompt":"hi"}
//!   userPromptSubmitted:   {"sessionId":"...","timestamp":...,"cwd":"...","prompt":"hi"}
//!   sessionEnd:            {"sessionId":"...","timestamp":...,"cwd":"...","reason":"complete"}
//!
//! Note: userPromptSubmitted fires *before* sessionStart for the initial prompt.
//! This is fine because both use the same sessionId.
//!
//! # Transcript format
//!
//! Copilot writes `~/.copilot/session-state/<uuid>/events.jsonl` with NDJSON
//! events. Relevant types: `user.message`, `assistant.message`,
//! `assistant.turn_start`, `assistant.turn_end`, `tool.execution_complete`.
//! The `session.shutdown` event exists in the schema but is NOT flushed to
//! disk — the file simply ends after the last assistant turn.

use crate::companion::flow::{AgentKind, CheckEvent, Flow, FlowConfig, RecordTurnEvent};
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

// ============================================================================
// Hook input types (from Copilot CLI)
// ============================================================================

/// Raw hook payload. Event type is inferred from which fields are present.
#[derive(Debug, Deserialize)]
struct HookInput {
    /// Present in all events — the Copilot session UUID.
    #[serde(rename = "sessionId")]
    session_id: String,
    /// Unix timestamp in milliseconds (reserved for future use).
    #[allow(dead_code)]
    timestamp: i64,
    cwd: String,
    /// sessionStart only — "new" | "resume" | "startup"
    #[serde(default)]
    #[allow(dead_code)]
    source: Option<String>,
    /// sessionStart only
    #[serde(default, rename = "initialPrompt")]
    initial_prompt: Option<String>,
    /// userPromptSubmitted only
    #[serde(default)]
    prompt: Option<String>,
    /// sessionEnd only
    #[serde(default)]
    reason: Option<String>,
}

enum HookEvent<'a> {
    SessionStart,
    UserPromptSubmitted { prompt: &'a str },
    SessionEnd,
    Unknown,
}

impl HookInput {
    fn event(&self) -> HookEvent<'_> {
        if self.initial_prompt.is_some() {
            HookEvent::SessionStart
        } else if let Some(ref p) = self.prompt {
            HookEvent::UserPromptSubmitted { prompt: p.as_str() }
        } else if self.reason.is_some() {
            HookEvent::SessionEnd
        } else {
            HookEvent::Unknown
        }
    }
}

// ============================================================================
// Cursor state (persisted per session UUID)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Default)]
struct CursorState {
    byte_offset: u64,
    last_event_id: Option<String>,
}

fn cursor_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("copilot")
        .join(format!("{}.cursor", session_id))
}

fn load_cursor(workspace_id: &str, session_id: &str) -> CursorState {
    let path = cursor_path(workspace_id, session_id);
    if let Ok(data) = std::fs::read_to_string(&path) {
        return serde_json::from_str(&data).unwrap_or_default();
    }
    CursorState::default()
}

fn save_cursor(workspace_id: &str, session_id: &str, cursor: &CursorState) {
    let path = cursor_path(workspace_id, session_id);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(data) = serde_json::to_string(cursor) {
        let _ = std::fs::write(&path, data);
    }
}

fn events_jsonl_path(session_id: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    Some(
        PathBuf::from(home)
            .join(".copilot")
            .join("session-state")
            .join(session_id)
            .join("events.jsonl"),
    )
}

// ============================================================================
// events.jsonl parsing
// ============================================================================

/// Parsed turn from events.jsonl
struct ParsedTurn {
    user_text: String,
    assistant_text: String,
    touched_paths: Vec<String>,
    timestamp_ms: i64,
    /// Byte offset where this turn's `user.message` event begins in `events.jsonl`.
    /// Used to build the `copilot+events://...#offset=N` source pointer.
    byte_offset: u64,
}

fn parse_iso8601_ms(s: &str) -> Option<i64> {
    // Hand-rolled parser for "2026-02-26T20:41:48.448Z" format. No extra deps.
    let s = s.trim().trim_end_matches('Z');
    let (date_part, time_part) = s.split_once('T')?;
    let dp: Vec<&str> = date_part.split('-').collect();
    if dp.len() != 3 {
        return None;
    }
    let year: i64 = dp[0].parse().ok()?;
    let month: i64 = dp[1].parse().ok()?;
    let day: i64 = dp[2].parse().ok()?;
    let (hms, frac) = time_part.split_once('.').unwrap_or((time_part, "0"));
    let tp: Vec<&str> = hms.split(':').collect();
    if tp.len() != 3 {
        return None;
    }
    let hour: i64 = tp[0].parse().ok()?;
    let minute: i64 = tp[1].parse().ok()?;
    let second: i64 = tp[2].parse().ok()?;
    let frac_str = &frac[..frac.len().min(3)];
    let millis: i64 = format!("{:0<3}", frac_str).parse().unwrap_or(0);
    let days = days_since_epoch(year, month, day)?;
    Some(days * 86_400_000 + hour * 3_600_000 + minute * 60_000 + second * 1_000 + millis)
}

fn days_since_epoch(year: i64, month: i64, day: i64) -> Option<i64> {
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 12 } else { month };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

fn looks_like_path(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.len() > 260 {
        return false;
    }
    if s.contains('\n') || s.contains('\r') || s.contains(' ') {
        return false;
    }
    s.contains('/') || s.contains('\\')
}

fn normalize_touched_path(p: &str, cwd: &Path) -> Option<String> {
    let mut s = p.trim();
    if s.is_empty() {
        return None;
    }
    let owned;
    if s.contains('\\') {
        owned = s.replace('\\', "/");
        s = &owned;
    }
    let s = s.strip_prefix("./").unwrap_or(s);
    if let Ok(abs) = std::path::Path::new(s).canonicalize() {
        if let Ok(cwd_abs) = cwd.canonicalize() {
            if let Ok(rel) = abs.strip_prefix(&cwd_abs) {
                let rel = rel.to_string_lossy().replace('\\', "/");
                let rel = rel.trim_start_matches('/').to_string();
                if !rel.is_empty() {
                    return Some(rel);
                }
            }
        }
    }
    Some(s.trim_start_matches('/').to_string())
}

fn collect_paths(v: &serde_json::Value, cwd: &Path, out: &mut HashSet<String>) {
    match v {
        serde_json::Value::String(s) => {
            if looks_like_path(s) {
                if let Some(p) = normalize_touched_path(s, cwd) {
                    out.insert(p);
                }
            }
        }
        serde_json::Value::Array(a) => {
            for x in a {
                collect_paths(x, cwd, out);
            }
        }
        serde_json::Value::Object(m) => {
            for k in ["path", "file", "file_path", "filepath", "filename", "target", "target_file"] {
                if let Some(val) = m.get(k) {
                    collect_paths(val, cwd, out);
                }
            }
            for val in m.values() {
                collect_paths(val, cwd, out);
            }
        }
        _ => {}
    }
}

fn truncate_text(mut s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    s.truncate(max_bytes);
    s.push_str("\n...(truncated)");
    s
}

/// Parse events.jsonl from cursor position, return new turns and updated cursor.
fn parse_events_from_cursor(
    events_path: &Path,
    cursor: &CursorState,
    cwd: &Path,
) -> anyhow::Result<(Vec<ParsedTurn>, CursorState)> {
    let file = std::fs::File::open(events_path)?;
    let file_size = file.metadata()?.len();

    let start_offset = if cursor.byte_offset > file_size {
        0
    } else {
        cursor.byte_offset
    };

    let mut reader = std::io::BufReader::new(file);
    reader.seek_relative(start_offset as i64)?;

    let mut turns: Vec<ParsedTurn> = Vec::new();

    // State machine over the event stream.
    // Turn structure in events.jsonl:
    //   user.message
    //   assistant.turn_start
    //   (assistant.message | tool.execution_start | tool.execution_complete)*
    //   assistant.turn_end
    //   [more turn_start/end pairs for the same user message, e.g. tool calls]
    //   [next user.message starts the next turn]
    let mut current_user_text: Option<String> = None;
    let mut current_assistant_chunks: Vec<String> = Vec::new();
    let mut current_touched: HashSet<String> = HashSet::new();
    let mut current_timestamp_ms: i64 = 0;
    let mut current_user_offset: u64 = 0;

    let mut last_event_id: Option<String> = cursor.last_event_id.clone();
    let mut new_offset = start_offset;
    let mut seen_last_id = start_offset > 0 || cursor.last_event_id.is_none();

    for line in reader.lines() {
        let line = line?;
        // `line_start_offset` is the byte offset of the start of this event's line
        // in the file (before we advance `new_offset` past it). Used to anchor a
        // source pointer at the user.message event that opens a turn.
        let line_start_offset = new_offset;
        new_offset += line.len() as u64 + 1; // +1 for newline

        if line.trim().is_empty() {
            continue;
        }

        let event: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let event_type = match event.get("type").and_then(|t| t.as_str()) {
            Some(t) => t.to_string(),
            None => continue,
        };

        let event_id = event.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());

        // Skip until we've passed the last processed event (resync after rotation)
        if let Some(ref last_id) = cursor.last_event_id {
            if !seen_last_id {
                if event_id.as_deref() == Some(last_id.as_str()) {
                    seen_last_id = true;
                }
                continue;
            }
        }

        if let Some(ref id) = event_id {
            last_event_id = Some(id.clone());
        }

        // Parse timestamp from event
        if let Some(ts_str) = event.get("timestamp").and_then(|t| t.as_str()) {
            if let Some(ms) = parse_iso8601_ms(ts_str) {
                current_timestamp_ms = ms;
            }
        }

        let data = event.get("data");

        match event_type.as_str() {
            "user.message" => {
                // Flush any pending turn before starting the next
                if let Some(user_text) = current_user_text.take() {
                    let assistant_text = truncate_text(
                        current_assistant_chunks.join("\n"),
                        64 * 1024,
                    );
                    turns.push(ParsedTurn {
                        user_text,
                        assistant_text,
                        touched_paths: current_touched.drain().collect(),
                        timestamp_ms: current_timestamp_ms,
                        byte_offset: current_user_offset,
                    });
                }
                current_assistant_chunks.clear();

                // Use `content` (raw user text), not `transformedContent` (system-augmented)
                if let Some(content) = data.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    let text = content.trim().to_string();
                    if !text.is_empty() {
                        current_user_text = Some(text);
                        current_user_offset = line_start_offset;
                    }
                }
            }

            "assistant.message" => {
                // content is a string in this format
                if let Some(content) = data.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    let text = content.trim().to_string();
                    if !text.is_empty() {
                        current_assistant_chunks.push(text);
                    }
                }
            }

            "tool.execution_complete" => {
                if let Some(data) = data {
                    collect_paths(data, cwd, &mut current_touched);
                }
            }

            // assistant.turn_start / assistant.turn_end / tool.execution_start / others — no-op
            _ => {}
        }
    }

    // Flush the final pending turn
    if let Some(user_text) = current_user_text {
        let assistant_text = truncate_text(current_assistant_chunks.join("\n"), 64 * 1024);
        turns.push(ParsedTurn {
            user_text,
            assistant_text,
            touched_paths: current_touched.drain().collect(),
            timestamp_ms: current_timestamp_ms,
            byte_offset: current_user_offset,
        });
    }

    let new_cursor = CursorState {
        byte_offset: new_offset,
        last_event_id,
    };

    Ok((turns, new_cursor))
}

// ============================================================================
// Hook handlers
// ============================================================================

async fn handle_session_start(flow: &mut Flow, input: &HookInput) -> anyhow::Result<()> {
    let cwd_path = Path::new(&input.cwd);
    let ws = match get_or_create_workspace_paths(cwd_path) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(error = %e, "copilot sessionStart: failed to get workspace");
            return Ok(());
        }
    };

    tracing::info!(
        session_id = %input.session_id,
        cwd = %input.cwd,
        "copilot sessionStart"
    );

    // Write an empty cursor to mark the session as known
    let cursor = load_cursor(&ws.id, &input.session_id);
    save_cursor(&ws.id, &input.session_id, &cursor);

    // Spawn a background checkpoint for previous sessions in this workspace
    {
        let ws_clone = ws.clone();
        tokio::spawn(async move {
            match crate::storage_checkpoint::maybe_create_checkpoint(
                &ws_clone,
                None,
                "new_session_copilot",
                None,
            )
            .await
            {
                Err(e) => tracing::debug!("checkpoint generation failed (non-fatal): {e}"),
                Ok(Err(reason)) => tracing::debug!("checkpoint skipped: {reason:?}"),
                Ok(Ok(_)) => {}
            }
        });
    }

    // Friction check on initial prompt (output ignored by Copilot, useful for metrics)
    if let Some(ref prompt) = input.initial_prompt {
        if !prompt.trim().is_empty() {
            let event = CheckEvent {
                directory: input.cwd.clone(),
                text: prompt.clone(),
                agent_kind: AgentKind::Copilot,
                agent_session_id: Some(input.session_id.clone()),
            };
            flow.check_turn(event).await;
        }
    }

    Ok(())
}

async fn handle_user_prompt_submitted(flow: &mut Flow, input: &HookInput, prompt: &str) -> anyhow::Result<()> {
    if prompt.trim().is_empty() {
        return Ok(());
    }

    tracing::info!(
        session_id = %input.session_id,
        "copilot userPromptSubmitted"
    );

    let event = CheckEvent {
        directory: input.cwd.clone(),
        text: prompt.to_string(),
        agent_kind: AgentKind::Copilot,
        agent_session_id: Some(input.session_id.clone()),
    };

    // Note: userPromptSubmitted hook output is currently ignored by Copilot CLI.
    // We run the check for metrics/logging only.
    flow.check_turn(event).await;

    Ok(())
}

async fn handle_session_end(
    flow: &mut Flow,
    input: &HookInput,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    let cwd_path = Path::new(&input.cwd);
    let ws = match get_or_create_workspace_paths(cwd_path) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(error = %e, "copilot sessionEnd: failed to get workspace");
            return Ok(());
        }
    };

    tracing::info!(
        session_id = %input.session_id,
        reason = ?input.reason,
        "copilot sessionEnd: reading events.jsonl"
    );

    let events_path = match events_jsonl_path(&input.session_id) {
        Some(p) => p,
        None => {
            tracing::warn!("copilot sessionEnd: could not locate events.jsonl");
            return Ok(());
        }
    };

    if !events_path.exists() {
        tracing::warn!(
            path = %events_path.display(),
            "copilot sessionEnd: events.jsonl not found"
        );
        return Ok(());
    }

    let cursor = load_cursor(&ws.id, &input.session_id);
    let (turns, new_cursor) = parse_events_from_cursor(&events_path, &cursor, cwd_path)?;

    tracing::info!(
        turns_count = turns.len(),
        old_offset = cursor.byte_offset,
        new_offset = new_cursor.byte_offset,
        "copilot: parsed events.jsonl"
    );

    let mut all_touched: Vec<String> = Vec::new();
    let mut all_assistant_texts: Vec<String> = Vec::new();

    for turn in turns {
        all_touched.extend(turn.touched_paths.iter().cloned());
        all_assistant_texts.push(turn.assistant_text.clone());

        let source_pointer = events_path
            .to_str()
            .filter(|s| !s.is_empty())
            .map(|p| format!("copilot+events://{p}#offset={}", turn.byte_offset));
        let event = RecordTurnEvent {
            directory: input.cwd.clone(),
            user_text: turn.user_text,
            assistant_text: turn.assistant_text,
            touched_paths: turn.touched_paths,
            tool_calls: vec![],
            agent_kind: AgentKind::Copilot,
            agent_session_id: Some(input.session_id.clone()),
            usage: None,
            grounding_note: None,
            source_ts_ms: if turn.timestamp_ms > 0 {
                Some(turn.timestamp_ms)
            } else {
                None
            },
            source_pointer,
        };

        let result = flow.record_turn(event).await;
        if let Some(error) = result.error {
            tracing::warn!(error = %error, "copilot: record_turn failed");
        }
    }

    save_cursor(&ws.id, &input.session_id, &new_cursor);

    // Stealth PR comment detection
    {
        let combined = all_assistant_texts.join("\n");
        if let Some(pr_url) =
            crate::companion::shims::opencode_stdio::extract_github_pr_url_pub(&combined)
        {
            let session_id = input.session_id.clone();
            let directory = input.cwd.clone();
            let em = embed_model.to_string();
            let ec = embed_cache_dir.map(|s| s.to_string());
            tokio::spawn(async move {
                tracing::info!(pr_url, "copilot: detected PR creation — posting unlost comment");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                if let Err(e) = crate::companion::shims::opencode_stdio::spawn_pr_comment_pub(
                    &pr_url,
                    Some(session_id.as_str()),
                    &directory,
                    &em,
                    ec.as_deref(),
                )
                .await
                {
                    tracing::warn!(error = ?e, pr_url, "copilot: pr-comment spawn failed");
                }
            });
        }
    }

    // Incremental changelog + git tag ingest
    if let Some(repo_root) = crate::workspace::git_toplevel(cwd_path) {
        let changelog_path = repo_root.join("CHANGELOG.md");
        let changelog_touched = all_touched
            .iter()
            .any(|p| p.ends_with("CHANGELOG.md") || p == "CHANGELOG.md");
        let embedder = crate::embed::load_embedder(
            embed_model,
            embed_cache_dir.map(std::path::PathBuf::from),
            false,
        )
        .await;
        if let Ok(ref embedder) = embedder {
            let use_color = std::io::IsTerminal::is_terminal(&std::io::stdout())
                && std::env::var_os("NO_COLOR").is_none();
            let _ = crate::git::ingest_git_tags(&ws, &repo_root, embedder, use_color).await;
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

    Ok(())
}

// ============================================================================
// Main entry point
// ============================================================================

pub async fn run(embed_model: String, embed_cache_dir: Option<String>) -> anyhow::Result<()> {
    let mut input_str = String::new();
    std::io::stdin().read_to_string(&mut input_str)?;

    let hook_input: HookInput = match serde_json::from_str(&input_str) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, raw = %input_str, "copilot: failed to parse hook input");
            return Ok(());
        }
    };

    let config = FlowConfig {
        embed_model: embed_model.clone(),
        embed_cache_dir: embed_cache_dir.clone(),
        extraction_mode: crate::types::ExtractionMode::Hybrid,
    };
    let mut flow = Flow::new(config);

    let event = hook_input.event();
    let event_label = match &event {
        HookEvent::SessionStart => "sessionStart",
        HookEvent::UserPromptSubmitted { .. } => "userPromptSubmitted",
        HookEvent::SessionEnd => "sessionEnd",
        HookEvent::Unknown => "unknown",
    };

    tracing::info!(
        hook_event = event_label,
        session_id = %hook_input.session_id,
        cwd = %hook_input.cwd,
        "copilot hook invoked"
    );

    let needs_drain = matches!(event, HookEvent::SessionEnd);

    match event {
        HookEvent::SessionStart => {
            handle_session_start(&mut flow, &hook_input).await?;
        }
        HookEvent::UserPromptSubmitted { prompt } => {
            handle_user_prompt_submitted(&mut flow, &hook_input, prompt).await?;
        }
        HookEvent::SessionEnd => {
            handle_session_end(
                &mut flow,
                &hook_input,
                &embed_model,
                embed_cache_dir.as_deref(),
            )
            .await?;
        }
        HookEvent::Unknown => {
            tracing::debug!(raw = %input_str, "copilot: could not determine hook event type");
        }
    }

    if needs_drain {
        flow.drain().await;
    }

    Ok(())
}
