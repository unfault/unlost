//! GitHub Copilot CLI hooks shim.
//!
//! Invoked by Copilot CLI hooks (sessionStart, userPromptSubmitted, sessionEnd)
//! via stdin JSON. Dispatches to Flow.check_friction() or events.jsonl ingestion.
//!
//! # Session discovery
//!
//! Copilot CLI does not pass a session ID in hook payloads. Instead, Copilot
//! maintains per-session state in `~/.copilot/session-state/<uuid>/`. At
//! `sessionStart` we discover the UUID by scanning `workspace.yaml` files for a
//! matching `cwd` whose `created_at` is within 5 seconds of the hook timestamp,
//! cross-checked against the `summary` field (which Copilot sets to the initial
//! prompt). We then write a cursor file keyed by UUID.
//!
//! At `sessionEnd` we find the right cursor file by matching `cwd` and picking
//! the session-state directory whose `updated_at` is closest to the hook
//! timestamp (most-recently-active session for that cwd wins). This is a
//! heuristic that fails only when two sessions in the same directory end
//! simultaneously — an accepted limitation documented in the README.
//!
//! # Transcript format
//!
//! Copilot writes `~/.copilot/session-state/<uuid>/events.jsonl` with NDJSON
//! events. Relevant types: `user.message`, `assistant.message`,
//! `assistant.turn_start`, `assistant.turn_end`, `tool.execution_complete`.
//! The `session.shutdown` event exists in the schema but is NOT flushed to
//! disk — the file simply ends at the last `assistant.turn_end`.

use crate::companion::flow::{AgentKind, CheckEvent, Flow, FlowConfig, RecordTurnEvent};
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

// ============================================================================
// Hook input types (from Copilot CLI)
// ============================================================================

/// Raw hook payload — all hooks share the same outer envelope.
#[derive(Debug, Deserialize)]
struct HookInput {
    /// One of: sessionStart, userPromptSubmitted, sessionEnd
    #[serde(default)]
    hook_event_name: Option<String>,
    /// Unix timestamp in milliseconds
    timestamp: i64,
    cwd: String,
    /// sessionStart only — "new" | "resume" | "startup" (not used, kept for completeness)
    #[serde(default)]
    #[allow(dead_code)]
    source: Option<String>,
    /// sessionStart only
    #[serde(default)]
    initial_prompt: Option<String>,
    /// userPromptSubmitted only
    #[serde(default)]
    prompt: Option<String>,
    /// sessionEnd only
    #[serde(default)]
    reason: Option<String>,
}

// ============================================================================
// Copilot session-state types
// ============================================================================

/// Parsed from ~/.copilot/session-state/<uuid>/workspace.yaml
#[derive(Debug, Deserialize)]
struct WorkspaceYaml {
    #[allow(dead_code)]
    id: String,
    cwd: String,
    #[serde(default)]
    summary: Option<String>,
    created_at: String,
    updated_at: String,
}

/// Cursor state persisted per Copilot session UUID.
#[derive(Debug, Serialize, Deserialize, Default)]
struct CursorState {
    byte_offset: u64,
    /// Last event ID processed (for resync after rotation)
    last_event_id: Option<String>,
}

// ============================================================================
// events.jsonl event types
// ============================================================================

/// Minimal envelope for every event in events.jsonl
#[derive(Debug, Deserialize)]
struct CopilotEvent {
    #[serde(rename = "type")]
    event_type: String,
    id: Option<String>,
    timestamp: Option<String>,
    data: Option<serde_json::Value>,
}

// ============================================================================
// Parsed turn
// ============================================================================

struct ParsedTurn {
    user_text: String,
    assistant_text: String,
    touched_paths: Vec<String>,
    timestamp_ms: i64,
}

// ============================================================================
// Path utilities (reused from claude shim logic)
// ============================================================================

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

fn collect_touched_paths_from_value(v: &serde_json::Value, cwd: &Path, out: &mut HashSet<String>) {
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
                collect_touched_paths_from_value(x, cwd, out);
            }
        }
        serde_json::Value::Object(m) => {
            for k in ["path", "file", "file_path", "filepath", "filename", "target", "target_file"] {
                if let Some(val) = m.get(k) {
                    collect_touched_paths_from_value(val, cwd, out);
                }
            }
            for (_k, val) in m.iter() {
                collect_touched_paths_from_value(val, cwd, out);
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

// ============================================================================
// Copilot session-state directory helpers
// ============================================================================

fn copilot_session_state_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    Some(PathBuf::from(home).join(".copilot").join("session-state"))
}

fn parse_iso8601_ms(s: &str) -> Option<i64> {
    // Parse RFC3339/ISO8601 timestamp to milliseconds since epoch.
    // Hand-rolled parser — no extra deps needed.
    // Quick hand-rolled parser for "2026-02-26T20:41:48.448Z" format
    let s = s.trim().trim_end_matches('Z');
    // Split on T
    let (date_part, time_part) = s.split_once('T')?;
    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return None;
    }
    let year: i64 = date_parts[0].parse().ok()?;
    let month: i64 = date_parts[1].parse().ok()?;
    let day: i64 = date_parts[2].parse().ok()?;

    let (hms, frac) = time_part.split_once('.').unwrap_or((time_part, "0"));
    let time_parts: Vec<&str> = hms.split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    let hour: i64 = time_parts[0].parse().ok()?;
    let minute: i64 = time_parts[1].parse().ok()?;
    let second: i64 = time_parts[2].parse().ok()?;
    let millis: i64 = {
        let frac = &frac[..frac.len().min(3)];
        let padded = format!("{:0<3}", frac);
        padded.parse().unwrap_or(0)
    };

    // Compute days since epoch (simplified: not leap-second aware)
    // Days from 1970-01-01 to year-month-day
    let days = days_since_epoch(year, month, day)?;
    let total_ms =
        days * 86_400_000 + hour * 3_600_000 + minute * 60_000 + second * 1_000 + millis;
    Some(total_ms)
}

fn days_since_epoch(year: i64, month: i64, day: i64) -> Option<i64> {
    // Days since 1970-01-01 using the proleptic Gregorian calendar.
    // Based on the algorithm from https://www.researchgate.net/publication/316558298
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 12 } else { month };
    let d = day;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (m - 3) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days_since_epoch_of_era = era * 146097 + doe - 719468;
    Some(days_since_epoch_of_era)
}

fn load_workspace_yaml(uuid: &str) -> Option<WorkspaceYaml> {
    let dir = copilot_session_state_dir()?;
    let path = dir.join(uuid).join("workspace.yaml");
    let content = std::fs::read_to_string(&path).ok()?;
    parse_workspace_yaml(&content)
}

/// Minimal hand-rolled YAML parser for workspace.yaml.
/// The file only uses simple `key: value` pairs — no nesting, no arrays.
fn parse_workspace_yaml(content: &str) -> Option<WorkspaceYaml> {
    fn extract(content: &str, key: &str) -> Option<String> {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix(key) {
                if let Some(rest) = rest.strip_prefix(':') {
                    return Some(rest.trim().to_string());
                }
            }
        }
        None
    }

    let id = extract(content, "id")?;
    let cwd = extract(content, "cwd")?;
    let created_at = extract(content, "created_at")?;
    let updated_at = extract(content, "updated_at")?;
    let summary = extract(content, "summary");

    Some(WorkspaceYaml {
        id,
        cwd,
        summary,
        created_at,
        updated_at,
    })
}

/// Scan all session-state dirs and return all parseable WorkspaceYaml entries.
fn scan_all_sessions() -> Vec<(String, WorkspaceYaml)> {
    let Some(dir) = copilot_session_state_dir() else {
        return vec![];
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let mut results = Vec::new();
    for entry in entries.flatten() {
        let uuid = entry.file_name().to_string_lossy().to_string();
        if let Some(ws) = load_workspace_yaml(&uuid) {
            results.push((uuid, ws));
        }
    }
    results
}

/// Find the Copilot session UUID for a new sessionStart hook.
/// Matches on cwd + created_at within 5s of hook timestamp, cross-checked
/// against initial_prompt == workspace.yaml summary.
fn discover_session_uuid_at_start(
    cwd: &str,
    hook_timestamp_ms: i64,
    initial_prompt: Option<&str>,
) -> Option<String> {
    let sessions = scan_all_sessions();
    let mut candidates: Vec<(String, i64)> = sessions
        .into_iter()
        .filter(|(_, ws)| ws.cwd == cwd)
        .filter_map(|(uuid, ws)| {
            let created_ms = parse_iso8601_ms(&ws.created_at)?;
            let diff = (created_ms - hook_timestamp_ms).abs();
            if diff > 5_000 {
                return None;
            }
            // Cross-check summary vs initial_prompt if both present
            if let (Some(prompt), Some(summary)) = (initial_prompt, &ws.summary) {
                let prompt_trimmed = prompt.trim();
                let summary_trimmed = summary.trim();
                // Allow partial match: summary starts with prompt (Copilot may truncate)
                if !summary_trimmed.starts_with(prompt_trimmed)
                    && !prompt_trimmed.starts_with(summary_trimmed)
                    && prompt_trimmed != summary_trimmed
                {
                    tracing::debug!(
                        uuid,
                        summary = summary_trimmed,
                        prompt = prompt_trimmed,
                        "sessionStart cross-check mismatch, skipping candidate"
                    );
                    return None;
                }
            }
            Some((uuid, diff))
        })
        .collect();

    // Pick closest created_at to hook timestamp
    candidates.sort_by_key(|(_, diff)| *diff);
    candidates.into_iter().next().map(|(uuid, _)| uuid)
}

/// Find the most likely Copilot session UUID at sessionEnd.
/// Among cursor files for this cwd, picks the one whose workspace.yaml
/// updated_at is closest to (and before or equal to) the hook timestamp.
fn discover_session_uuid_at_end(
    workspace_id: &str,
    cwd: &str,
    hook_timestamp_ms: i64,
) -> Option<String> {
    let cursor_dir = crate::workspace::unlost_workspace_dir(workspace_id).join("copilot");
    let entries = std::fs::read_dir(&cursor_dir).ok()?;

    let mut candidates: Vec<(String, i64)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let uuid = name.strip_suffix(".cursor")?.to_string();
            // Verify this cursor's session has matching cwd
            let ws = load_workspace_yaml(&uuid)?;
            if ws.cwd != cwd {
                return None;
            }
            let updated_ms = parse_iso8601_ms(&ws.updated_at)?;
            // updated_at should be <= hook_timestamp_ms (session updated before end hook fired)
            // Allow up to 10s slack for clock skew / hook latency
            if updated_ms > hook_timestamp_ms + 10_000 {
                return None;
            }
            let diff = (hook_timestamp_ms - updated_ms).abs();
            Some((uuid, diff))
        })
        .collect();

    candidates.sort_by_key(|(_, diff)| *diff);
    candidates.into_iter().next().map(|(uuid, _)| uuid)
}

// ============================================================================
// Cursor helpers
// ============================================================================

fn cursor_path(workspace_id: &str, session_uuid: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("copilot")
        .join(format!("{}.cursor", session_uuid))
}

fn load_cursor(workspace_id: &str, session_uuid: &str) -> CursorState {
    let path = cursor_path(workspace_id, session_uuid);
    if let Ok(data) = std::fs::read_to_string(&path) {
        return serde_json::from_str(&data).unwrap_or_default();
    }
    CursorState::default()
}

fn save_cursor(workspace_id: &str, session_uuid: &str, cursor: &CursorState) {
    let path = cursor_path(workspace_id, session_uuid);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(data) = serde_json::to_string(cursor) {
        let _ = std::fs::write(&path, data);
    }
}

fn events_jsonl_path(session_uuid: &str) -> Option<PathBuf> {
    let dir = copilot_session_state_dir()?;
    Some(dir.join(session_uuid).join("events.jsonl"))
}

// ============================================================================
// events.jsonl parsing
// ============================================================================

/// Parse events.jsonl from cursor, returning new turns and updated cursor.
fn parse_events_from_cursor(
    events_path: &Path,
    cursor: &CursorState,
    cwd: &Path,
) -> anyhow::Result<(Vec<ParsedTurn>, CursorState)> {
    let file = std::fs::File::open(events_path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    let start_offset = if cursor.byte_offset > file_size {
        0
    } else {
        cursor.byte_offset
    };

    let mut reader = std::io::BufReader::new(file);
    reader.seek_relative(start_offset as i64)?;

    let mut turns: Vec<ParsedTurn> = Vec::new();

    // State machine: accumulate within a turn bounded by assistant.turn_start / assistant.turn_end
    let mut current_user_text: Option<String> = None;
    let mut current_assistant_text = String::new();
    let mut current_touched: HashSet<String> = HashSet::new();
    let mut current_timestamp_ms: i64 = 0;
    let mut in_turn = false;

    let mut last_event_id: Option<String> = cursor.last_event_id.clone();
    let mut new_offset = start_offset;
    let mut seen_last_id = start_offset > 0 || cursor.last_event_id.is_none();

    for line in reader.lines() {
        let line = line?;
        new_offset += line.len() as u64 + 1;

        if line.trim().is_empty() {
            continue;
        }

        let event: CopilotEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip until we pass the last processed event ID (resync after rotation)
        if let Some(ref last_id) = cursor.last_event_id {
            if !seen_last_id {
                if event.id.as_deref() == Some(last_id.as_str()) {
                    seen_last_id = true;
                }
                continue;
            }
        }

        if let Some(ref id) = event.id {
            last_event_id = Some(id.clone());
        }

        // Parse timestamp from event if present
        if let Some(ref ts_str) = event.timestamp {
            if let Some(ms) = parse_iso8601_ms(ts_str) {
                current_timestamp_ms = ms;
            }
        }

        let data = event.data.as_ref();

        match event.event_type.as_str() {
            "user.message" => {
                // Flush any pending turn before starting a new one
                if let Some(user_text) = current_user_text.take() {
                    if !user_text.trim().is_empty() {
                        turns.push(ParsedTurn {
                            user_text,
                            assistant_text: truncate_text(
                                std::mem::take(&mut current_assistant_text),
                                64 * 1024,
                            ),
                            touched_paths: current_touched.drain().collect(),
                            timestamp_ms: current_timestamp_ms,
                        });
                    }
                }
                current_assistant_text.clear();
                in_turn = false;

                // Extract user message content (use `content`, not `transformedContent`)
                if let Some(content) = data.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    let text = content.trim().to_string();
                    if !text.is_empty() {
                        current_user_text = Some(text);
                    }
                }
            }

            "assistant.turn_start" => {
                in_turn = true;
            }

            "assistant.message" => {
                if !in_turn {
                    continue;
                }
                if let Some(content) = data.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    let text = content.trim();
                    if !text.is_empty() {
                        if !current_assistant_text.is_empty() {
                            current_assistant_text.push('\n');
                        }
                        current_assistant_text.push_str(text);
                    }
                }
            }

            "tool.execution_complete" => {
                // Extract touched paths from tool args and result content
                if let Some(data) = data {
                    collect_touched_paths_from_value(data, cwd, &mut current_touched);
                }
            }

            "assistant.turn_end" => {
                in_turn = false;
                // Don't flush here — wait for next user.message or end of file,
                // because a single user message may span multiple turn_start/end pairs
                // (e.g. tool calls interleaved with assistant messages).
            }

            _ => {}
        }
    }

    // Flush last pending turn
    if let Some(user_text) = current_user_text {
        if !user_text.trim().is_empty() {
            turns.push(ParsedTurn {
                user_text,
                assistant_text: truncate_text(current_assistant_text, 64 * 1024),
                touched_paths: current_touched.drain().collect(),
                timestamp_ms: current_timestamp_ms,
            });
        }
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

async fn handle_session_start(
    flow: &mut Flow,
    input: &HookInput,
    _embed_model: &str,
    _embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    let cwd_path = Path::new(&input.cwd);
    let ws = match get_or_create_workspace_paths(cwd_path) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(error = %e, "copilot sessionStart: failed to get workspace");
            return Ok(());
        }
    };

    // Discover session UUID
    let uuid = discover_session_uuid_at_start(
        &input.cwd,
        input.timestamp,
        input.initial_prompt.as_deref(),
    );

    let Some(uuid) = uuid else {
        tracing::warn!(
            cwd = %input.cwd,
            timestamp_ms = input.timestamp,
            "copilot sessionStart: could not discover session UUID"
        );
        return Ok(());
    };

    tracing::info!(uuid, cwd = %input.cwd, "copilot sessionStart: discovered session");

    // Create cursor file to mark session discovery
    let cursor = CursorState::default();
    save_cursor(&ws.id, &uuid, &cursor);

    // Spawn a background checkpoint for any previous sessions in this workspace
    // (same pattern as Claude shim on first Stop).
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

    // Run friction check on initial prompt if provided (output ignored by Copilot,
    // but useful for metrics and logging).
    if let Some(ref prompt) = input.initial_prompt {
        if !prompt.trim().is_empty() {
            let event = CheckEvent {
                directory: input.cwd.clone(),
                text: prompt.clone(),
                agent_kind: AgentKind::Copilot,
                agent_session_id: Some(uuid),
            };
            flow.check_friction(event).await;
        }
    }

    Ok(())
}

async fn handle_user_prompt_submitted(flow: &mut Flow, input: &HookInput) -> anyhow::Result<()> {
    let prompt = match &input.prompt {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            tracing::warn!("copilot userPromptSubmitted: missing prompt field");
            return Ok(());
        }
    };

    // Find the session UUID for this cwd so we can tag the friction event.
    let cwd_path = Path::new(&input.cwd);
    let session_uuid = get_or_create_workspace_paths(cwd_path)
        .ok()
        .and_then(|ws| discover_session_uuid_at_end(&ws.id, &input.cwd, input.timestamp));

    let event = CheckEvent {
        directory: input.cwd.clone(),
        text: prompt.clone(),
        agent_kind: AgentKind::Copilot,
        agent_session_id: session_uuid,
    };

    // Note: userPromptSubmitted output is currently ignored by Copilot CLI.
    // We run the check for metrics/logging only.
    flow.check_friction(event).await;

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

    let uuid = discover_session_uuid_at_end(&ws.id, &input.cwd, input.timestamp);
    let Some(uuid) = uuid else {
        tracing::warn!(
            cwd = %input.cwd,
            "copilot sessionEnd: could not find cursor file for this cwd — sessionStart may not have run"
        );
        return Ok(());
    };

    tracing::info!(uuid, cwd = %input.cwd, reason = ?input.reason, "copilot sessionEnd: recording");

    let events_path = match events_jsonl_path(&uuid) {
        Some(p) => p,
        None => {
            tracing::warn!(uuid, "copilot sessionEnd: could not locate events.jsonl");
            return Ok(());
        }
    };

    if !events_path.exists() {
        tracing::warn!(path = %events_path.display(), "copilot sessionEnd: events.jsonl not found");
        return Ok(());
    }

    let cursor = load_cursor(&ws.id, &uuid);
    let (turns, new_cursor) =
        parse_events_from_cursor(&events_path, &cursor, cwd_path)?;

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

        let event = RecordTurnEvent {
            directory: input.cwd.clone(),
            user_text: turn.user_text,
            assistant_text: turn.assistant_text,
            touched_paths: turn.touched_paths,
            tool_calls: vec![],
            agent_kind: AgentKind::Copilot,
            agent_session_id: Some(uuid.clone()),
            usage: None,
            grounding_note: None,
            source_ts_ms: if turn.timestamp_ms > 0 {
                Some(turn.timestamp_ms)
            } else {
                None
            },
        };

        let result = flow.record_turn(event).await;
        if let Some(error) = result.error {
            tracing::warn!(error = %error, "copilot: record_turn failed");
        }
    }

    save_cursor(&ws.id, &uuid, &new_cursor);

    // Stealth PR comment detection
    {
        let combined = all_assistant_texts.join("\n");
        if let Some(pr_url) =
            crate::companion::shims::opencode_stdio::extract_github_pr_url_pub(&combined)
        {
            let session_id = uuid.clone();
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

    // Incremental changelog + git tag ingest (same as Claude shim)
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
                let _ =
                    crate::changelog::ingest_changelog(&ws, &changelog_path, embedder, use_color)
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

    // Copilot CLI puts the event name in "hook_event_name"
    let event_name = hook_input
        .hook_event_name
        .as_deref()
        .unwrap_or("")
        .to_string();

    tracing::info!(
        hook_event = %event_name,
        cwd = %hook_input.cwd,
        timestamp_ms = hook_input.timestamp,
        "copilot hook invoked"
    );

    let config = FlowConfig {
        embed_model: embed_model.clone(),
        embed_cache_dir: embed_cache_dir.clone(),
        extraction_mode: crate::types::ExtractionMode::Hybrid,
    };
    let mut flow = Flow::new(config);

    let needs_drain = event_name == "sessionEnd";

    match event_name.as_str() {
        "sessionStart" => {
            handle_session_start(&mut flow, &hook_input, &embed_model, embed_cache_dir.as_deref())
                .await?;
        }
        "userPromptSubmitted" => {
            handle_user_prompt_submitted(&mut flow, &hook_input).await?;
        }
        "sessionEnd" => {
            handle_session_end(&mut flow, &hook_input, &embed_model, embed_cache_dir.as_deref())
                .await?;
        }
        _ => {
            tracing::debug!(event = %event_name, "copilot: ignoring unhandled hook event");
        }
    }

    if needs_drain {
        flow.drain().await;
    }

    Ok(())
}
