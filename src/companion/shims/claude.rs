//! Claude hooks shim.
//!
//! Invoked by Claude hooks (UserPromptSubmit, Stop) via stdin JSON.
//! Reads hook event, dispatches to Flow.check() or transcript ingestion.
//!
//! Cursor state is persisted per-session to avoid re-processing transcript lines.

use crate::companion::flow::{
    AgentKind, CheckEvent, Flow, FlowConfig, RecordTurnEvent, UsageEvent,
};
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::collections::HashSet;

// ============================================================================
// Hook input types (from Claude Code)
// ============================================================================

#[derive(Debug, Deserialize)]
struct HookInput {
    hook_event_name: String,
    session_id: String,
    cwd: String,
    #[serde(default)]
    transcript_path: Option<String>,
    /// For UserPromptSubmit
    #[serde(default)]
    prompt: Option<String>,
}

// ============================================================================
// Hook output types (to Claude Code)
// ============================================================================

#[derive(Debug, Serialize)]
struct HookOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: Option<HookSpecificOutput>,
}

#[derive(Debug, Serialize)]
struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    hook_event_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "additionalContext")]
    additional_context: Option<String>,
}

// ============================================================================
// Transcript parsing types
// ============================================================================

#[derive(Debug, Deserialize)]
struct TranscriptLine {
    #[serde(rename = "type")]
    line_type: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    uuid: Option<String>,
    message: Option<TranscriptMessage>,
    #[serde(flatten)]
    extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
    model: Option<String>,
    usage: Option<TranscriptUsage>,
}

#[derive(Debug, Deserialize)]
struct TranscriptUsage {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cache_read_input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
}

// ============================================================================
// Cursor state (persisted per session)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Default)]
struct CursorState {
    byte_offset: u64,
    last_uuid: Option<String>,
}

fn cursor_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("claude")
        .join(format!("{}.cursor", session_id))
}

fn legacy_cursor_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("claudecode")
        .join(format!("{}.cursor", session_id))
}

fn load_cursor(workspace_id: &str, session_id: &str) -> CursorState {
    let path = cursor_path(workspace_id, session_id);
    if let Ok(data) = std::fs::read_to_string(&path) {
        return serde_json::from_str(&data).unwrap_or_default();
    }
    // Legacy location (pre-claude rename).
    let legacy = legacy_cursor_path(workspace_id, session_id);
    if let Ok(data) = std::fs::read_to_string(&legacy) {
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

fn turnkeys_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("claude")
        .join(format!("{}.turnkeys", session_id))
}

fn legacy_turnkeys_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("claudecode")
        .join(format!("{}.turnkeys", session_id))
}

fn load_turnkeys(workspace_id: &str, session_id: &str) -> HashSet<String> {
    let path = turnkeys_path(workspace_id, session_id);
    let data = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => {
            // Legacy location (pre-claude rename).
            let legacy = legacy_turnkeys_path(workspace_id, session_id);
            match std::fs::read_to_string(&legacy) {
                Ok(s) => s,
                Err(_) => return HashSet::new(),
            }
        }
    };
    data.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn append_turnkeys(workspace_id: &str, session_id: &str, keys: &[String]) {
    if keys.is_empty() {
        return;
    }
    let path = turnkeys_path(workspace_id, session_id);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    for k in keys {
        if k.trim().is_empty() {
            continue;
        }
        let _ = writeln!(f, "{}", k.trim());
    }
}

// ============================================================================
// Transcript parsing
// ============================================================================

/// A parsed turn from the transcript (user prompt + assistant response)
struct ParsedTurn {
    user_text: String,
    assistant_text: String,
    usage: Option<UsageEvent>,
    touched_paths: Vec<String>,
    user_uuid: Option<String>,
    assistant_uuid: Option<String>,
    timestamp_ms: i64,
}

fn turn_key(turn: &ParsedTurn) -> Option<String> {
    let u = turn.user_uuid.as_deref()?.trim();
    if u.is_empty() {
        return None;
    }
    let a = turn.assistant_uuid.as_deref().unwrap_or("").trim();
    Some(format!("{u}:{a}"))
}

fn truncate_for_recording(mut s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    // Best-effort truncation without worrying about UTF-8 boundaries too much;
    // Claude transcripts are typically UTF-8 and mostly ASCII.
    s.truncate(max_bytes);
    s.push_str("\n...(truncated)");
    s
}

fn value_to_compact_text(v: &serde_json::Value, max_bytes: usize) -> String {
    match v {
        serde_json::Value::String(s) => truncate_for_recording(s.clone(), max_bytes),
        serde_json::Value::Array(a) => {
            // Some tool results are arrays of text blocks.
            let mut out = String::new();
            for item in a {
                if let Some(obj) = item.as_object() {
                    if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(t);
                        }
                    }
                }
            }
            if out.trim().is_empty() {
                truncate_for_recording(v.to_string(), max_bytes)
            } else {
                truncate_for_recording(out, max_bytes)
            }
        }
        _ => truncate_for_recording(v.to_string(), max_bytes),
    }
}

fn extract_tool_result_blocks_text(content: &serde_json::Value) -> Option<String> {
    // Claude logs represent tool results as a `user` message whose content includes
    // blocks of type `tool_result`.
    let serde_json::Value::Array(blocks) = content else {
        return None;
    };

    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        let Some(obj) = block.as_object() else {
            continue;
        };
        if obj.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
            continue;
        }

        let is_error = obj.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
        let tool_use_id = obj
            .get("tool_use_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let raw_content = obj.get("content").unwrap_or(&serde_json::Value::Null);
        let text = value_to_compact_text(raw_content, 12 * 1024);
        if text.trim().is_empty() {
            continue;
        }

        let mut s = String::new();
        if is_error {
            s.push_str("Tool result (error)");
        } else {
            s.push_str("Tool result");
        }
        if !tool_use_id.is_empty() {
            s.push_str(" ");
            s.push_str(tool_use_id);
        }
        s.push_str(":\n");
        s.push_str(text.trim());
        parts.push(s);
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

fn normalize_touched_path(p: &str, cwd: &Path) -> Option<String> {
    let mut s = p.trim();
    if s.is_empty() {
        return None;
    }

    // Normalize slashes.
    let owned;
    if s.contains('\\') {
        owned = s.replace('\\', "/");
        s = &owned;
    }

    // Strip leading ./
    let s = s.strip_prefix("./").unwrap_or(s);

    // If absolute and under cwd, strip cwd prefix.
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

fn looks_like_path(s: &str) -> bool {
    // Heuristic: avoid grabbing arbitrary prose.
    // Accept common repo-ish patterns.
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    if s.len() > 260 {
        return false;
    }
    if s.contains('\n') || s.contains('\r') {
        return false;
    }
    if s.contains(' ') {
        return false;
    }
    if !(s.contains('/') || s.contains('\\')) {
        return false;
    }
    true
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
            // Prefer keys commonly used for file paths.
            for k in [
                "path",
                "file",
                "file_path",
                "filepath",
                "filename",
                "target",
                "target_file",
            ] {
                if let Some(val) = m.get(k) {
                    collect_touched_paths_from_value(val, cwd, out);
                }
            }
            // Also walk everything; snapshots vary.
            for (_k, val) in m.iter() {
                collect_touched_paths_from_value(val, cwd, out);
            }
        }
        _ => {}
    }
}

fn extract_touched_paths_from_content(content: &serde_json::Value, cwd: &Path) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();

    // Content may be an array of blocks.
    if let serde_json::Value::Array(blocks) = content {
        for block in blocks {
            if let Some(obj) = block.as_object() {
                let ty = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if ty == "tool_use" || ty == "tool_result" {
                    if let Some(v) = obj.get("input") {
                        collect_touched_paths_from_value(v, cwd, &mut out);
                    }
                    if let Some(v) = obj.get("result") {
                        collect_touched_paths_from_value(v, cwd, &mut out);
                    }
                }
            }
        }
    }

    let mut v = out.into_iter().collect::<Vec<_>>();
    v.sort();
    v.truncate(64);
    v
}

/// Extract text content from a message's content field
fn extract_text_content(content: &serde_json::Value) -> String {
    match content {
        // Simple string content
        serde_json::Value::String(s) => s.clone(),
        // Array of content blocks
        serde_json::Value::Array(blocks) => {
            let mut texts = Vec::new();
            for block in blocks {
                if let Some(obj) = block.as_object() {
                    // Only extract "text" type blocks, ignore tool_use, tool_result, thinking
                    if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                            texts.push(text.to_string());
                        }
                    }
                }
            }
            texts.join("\n")
        }
        _ => String::new(),
    }
}

/// Check if a user message is a tool_result (which we skip)
fn is_tool_result_message(content: &serde_json::Value) -> bool {
    if let serde_json::Value::Array(blocks) = content {
        for block in blocks {
            if let Some(obj) = block.as_object() {
                if obj.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse transcript from cursor position, extract new turns
fn parse_transcript_from_cursor(
    transcript_path: &Path,
    cursor: &CursorState,
    cwd: &Path,
) -> anyhow::Result<(Vec<ParsedTurn>, CursorState)> {
    let file = std::fs::File::open(transcript_path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // If file is smaller than cursor, it was truncated/rotated - start from beginning
    let start_offset = if cursor.byte_offset > file_size {
        0
    } else {
        cursor.byte_offset
    };

    let mut reader = std::io::BufReader::new(file);
    reader.seek_relative(start_offset as i64)?;

    let mut turns = Vec::new();
    let mut current_user_text: Option<String> = None;
    let mut current_user_uuid: Option<String> = None;
    let mut current_assistant_text = String::new();
    let mut current_last_assistant_uuid: Option<String> = None;
    let mut current_timestamp_ms = 0i64;

    // Best-effort: collect touched paths during a turn.
    let mut pending_touched: HashSet<String> = HashSet::new();

    let mut current_usage: Option<UsageEvent> = None;
    let mut last_uuid: Option<String> = cursor.last_uuid.clone();
    let mut new_offset = start_offset;
    // Cursor uses byte_offset as the primary mechanism. Only use last_uuid as a
    // resync aid when we start from the beginning (e.g. file truncation/rotation).
    let mut seen_last_uuid = start_offset > 0 || cursor.last_uuid.is_none();

    for line in reader.lines() {
        let line = line?;
        new_offset += line.len() as u64 + 1; // +1 for newline

        if line.trim().is_empty() {
            continue;
        }

        let parsed: TranscriptLine = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Extract timestamp if available
        if let Some(ts) = parsed.extra.get("time").and_then(|v| v.as_i64()) {
            current_timestamp_ms = ts;
        }

        // Best-effort: some tool outputs include structured fields outside `message.content`
        // (e.g. `toolUseResult.filePath`). Collect any path-like strings.
        for (_k, v) in parsed.extra.iter() {
            collect_touched_paths_from_value(v, cwd, &mut pending_touched);
        }

        // Skip non-message lines (file-history-snapshot, etc.)
        let line_type = match &parsed.line_type {
            Some(t) => t.as_str(),
            None => continue,
        };

        if line_type == "file-history-snapshot" {
            for (_k, v) in parsed.extra.iter() {
                collect_touched_paths_from_value(v, cwd, &mut pending_touched);
            }
            if let Some(uuid) = &parsed.uuid {
                last_uuid = Some(uuid.clone());
            }
            continue;
        }

        // Skip sidechains (prompt suggestions, background agents)
        if parsed.is_sidechain == Some(true) {
            continue;
        }

        // Skip until we pass the last processed UUID
        if let Some(ref last) = cursor.last_uuid {
            if !seen_last_uuid {
                if parsed.uuid.as_ref() == Some(last) {
                    seen_last_uuid = true;
                }
                continue;
            }
        }

        let message = match &parsed.message {
            Some(m) => m,
            None => continue,
        };

        let role = match &message.role {
            Some(r) => r.as_str(),
            None => continue,
        };

        let content = match &message.content {
            Some(c) => c,
            None => continue,
        };

        for p in extract_touched_paths_from_content(content, cwd) {
            pending_touched.insert(p);
        }

        match (line_type, role) {
            ("user", "user") => {
                // Tool results are logged as `user` messages; keep them as part of the
                // assistant-side context so we don't lose error details.
                if is_tool_result_message(content) {
                    if current_user_text.is_some() {
                        if let Some(tool_text) = extract_tool_result_blocks_text(content) {
                            if !current_assistant_text.is_empty() {
                                current_assistant_text.push('\n');
                                current_assistant_text.push('\n');
                            }
                            current_assistant_text.push_str(&tool_text);
                        }
                    }
                    if let Some(uuid) = &parsed.uuid {
                        last_uuid = Some(uuid.clone());
                    }
                    continue;
                }

                let text = extract_text_content(content);
                if text.trim().is_empty() {
                    continue;
                }

                // If we have a pending turn, save it
                if let Some(user_text) = current_user_text.take() {
                    // Record even if assistant text is empty; user-only turns are
                    // valuable for friction detection and future context.
                    turns.push(ParsedTurn {
                        user_text,
                        assistant_text: std::mem::take(&mut current_assistant_text),
                        usage: current_usage.take(),
                        touched_paths: pending_touched.drain().collect(),
                        user_uuid: current_user_uuid.take(),
                        assistant_uuid: current_last_assistant_uuid.take(),
                        timestamp_ms: current_timestamp_ms,
                    });
                }

                current_user_text = Some(text);
                current_user_uuid = parsed.uuid.clone();
                current_assistant_text.clear();
                current_last_assistant_uuid = None;
                current_usage = None;
                // Don't reset current_timestamp_ms here, keep the last seen one for the turn
            }
            ("assistant", "assistant") => {
                let text = extract_text_content(content);
                if !text.trim().is_empty() {
                    if !current_assistant_text.is_empty() {
                        current_assistant_text.push('\n');
                    }
                    current_assistant_text.push_str(&text);
                }

                if parsed.uuid.is_some() {
                    current_last_assistant_uuid = parsed.uuid.clone();
                }

                // Capture usage from assistant messages
                if let Some(usage) = &message.usage {
                    current_usage = Some(UsageEvent {
                        provider_id: Some("anthropic".to_string()),
                        model_id: message.model.clone(),
                        cost: None,
                        tokens_input: usage.input_tokens,
                        tokens_output: usage.output_tokens,
                        tokens_reasoning: None,
                        tokens_cache_read: usage.cache_read_input_tokens,
                        tokens_cache_write: usage.cache_creation_input_tokens,
                    });
                }
            }
            _ => {}
        }

        if let Some(uuid) = &parsed.uuid {
            last_uuid = Some(uuid.clone());
        }
    }

    // Don't forget the last pending turn
    if let Some(user_text) = current_user_text {
        turns.push(ParsedTurn {
            user_text,
            assistant_text: current_assistant_text,
            usage: current_usage,
            touched_paths: pending_touched.drain().collect(),
            user_uuid: current_user_uuid,
            assistant_uuid: current_last_assistant_uuid,
            timestamp_ms: current_timestamp_ms,
        });
    }

    let new_cursor = CursorState {
        byte_offset: new_offset,
        last_uuid,
    };

    Ok((turns, new_cursor))
}

// ============================================================================
// Main entry point
// ============================================================================

pub async fn run(
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    // Read hook input from stdin
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let hook_input: HookInput = serde_json::from_str(&input)?;

    tracing::info!(
        hook_event = %hook_input.hook_event_name,
        session_id = %hook_input.session_id,
        cwd = %hook_input.cwd,
        "claude hook invoked"
    );

    let config = FlowConfig {
        embed_model,
        embed_cache_dir,
        no_llm: false, // hooks are for live, so no_llm defaults to false
    };
    let mut flow = Flow::new(config);

    // Track whether we need to drain (only for Stop hook which records turns)
    let needs_drain = hook_input.hook_event_name == "Stop";

    match hook_input.hook_event_name.as_str() {
        "UserPromptSubmit" => {
            handle_user_prompt_submit(&mut flow, &hook_input).await?;
        }
        "Stop" => {
            handle_stop(&mut flow, &hook_input).await?;
        }
        _ => {
            tracing::debug!(event = %hook_input.hook_event_name, "ignoring unhandled hook event");
        }
    }

    // For Stop hook, drain the background worker to ensure capsules are persisted
    // before the process exits. This is safe because Stop hook runs with async:true.
    if needs_drain {
        flow.drain().await;
    }

    Ok(())
}

async fn handle_user_prompt_submit(flow: &mut Flow, input: &HookInput) -> anyhow::Result<()> {
    let prompt = match &input.prompt {
        Some(p) => p,
        None => {
            tracing::warn!("UserPromptSubmit missing prompt field");
            return Ok(());
        }
    };

        let event = CheckEvent {
            directory: input.cwd.clone(),
            text: prompt.clone(),
            agent_kind: AgentKind::Claude,
            agent_session_id: Some(input.session_id.clone()),
        };

    let result = flow.check_friction(event).await;

    // Output hook response
    let output = if let Some(note) = result.note {
        HookOutput {
            hook_specific_output: Some(HookSpecificOutput {
                hook_event_name: "UserPromptSubmit".to_string(),
                additional_context: Some(note),
            }),
        }
    } else {
        HookOutput {
            hook_specific_output: None,
        }
    };

    let json = serde_json::to_string(&output)?;
    println!("{}", json);
    std::io::stdout().flush()?;

    Ok(())
}

async fn handle_stop(flow: &mut Flow, input: &HookInput) -> anyhow::Result<()> {
    let transcript_path = match &input.transcript_path {
        Some(p) => PathBuf::from(p),
        None => {
            tracing::warn!("Stop hook missing transcript_path");
            return Ok(());
        }
    };

    if !transcript_path.exists() {
        tracing::warn!(path = %transcript_path.display(), "transcript file not found");
        return Ok(());
    }

    // Get workspace for this cwd
    let cwd_path = std::path::Path::new(&input.cwd);
    let ws = match get_or_create_workspace_paths(cwd_path) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(error = %e, "failed to get workspace paths");
            return Ok(());
        }
    };

    // Load cursor and parse new turns
    let cursor = load_cursor(&ws.id, &input.session_id);
    let cwd_path = std::path::Path::new(&input.cwd);
    let (turns, new_cursor) = parse_transcript_from_cursor(&transcript_path, &cursor, cwd_path)?;

    tracing::info!(
        turns_count = turns.len(),
        old_offset = cursor.byte_offset,
        new_offset = new_cursor.byte_offset,
        "parsed transcript"
    );

    // Record each turn (best-effort de-dupe by transcript UUID keys)
    let mut seen = load_turnkeys(&ws.id, &input.session_id);
    let mut new_keys: Vec<String> = Vec::new();

    for turn in turns {
        if let Some(k) = turn_key(&turn) {
            if seen.contains(&k) {
                continue;
            }
            seen.insert(k.clone());
            new_keys.push(k);
        }
        let event = RecordTurnEvent {
            directory: input.cwd.clone(),
            user_text: turn.user_text,
            assistant_text: turn.assistant_text,
            touched_paths: turn.touched_paths,
            agent_kind: AgentKind::Claude,
            agent_session_id: Some(input.session_id.clone()),
            usage: turn.usage,
            grounding_note: None,
        };

        let result = flow.record_turn(event).await;
        if let Some(error) = result.error {
            tracing::warn!(error = %error, "record_turn failed");
        }
    }

    append_turnkeys(&ws.id, &input.session_id, &new_keys);

    // Save cursor
    save_cursor(&ws.id, &input.session_id, &new_cursor);

    Ok(())
}

// ============================================================================
// Cost warning helper
// ============================================================================

/// Models known to be expensive (reasoning models, large frontier models).
fn is_expensive_model(model: &str) -> bool {
    let m = model.to_lowercase();
    // OpenAI expensive models
    if m.contains("o1") || m.contains("o3") || m.contains("gpt-4o") && !m.contains("mini") {
        return true;
    }
    // Anthropic expensive models  
    if m.contains("opus") || (m.contains("sonnet") && !m.contains("3-5") && !m.contains("3.5")) {
        return true;
    }
    false
}

/// Suggest a cheaper alternative model for the same provider.
fn suggest_cheaper_model(provider: &str, current_model: &str) -> Option<(&'static str, &'static str)> {
    let m = current_model.to_lowercase();
    
    match provider {
        "openai" => {
            if m.contains("o1") || m.contains("o3") || (m.contains("gpt-4") && !m.contains("mini")) {
                return Some(("gpt-4o-mini", "unlost config llm openai --model gpt-4o-mini"));
            }
        }
        "anthropic" => {
            if m.contains("opus") || (m.contains("sonnet") && !m.contains("haiku")) {
                return Some(("claude-3-5-haiku-20241022", "unlost config llm anthropic --model claude-3-5-haiku-20241022"));
            }
        }
        _ => {}
    }
    None
}

/// Print a cost warning before replay starts.
fn print_cost_warning(turn_count: usize, use_color: bool) {
    use crate::config::LlmConfig;
    use crate::workspace::load_workspace_config;
    
    let cfg = load_workspace_config();
    
    let (provider, model) = match &cfg.llm {
        Some(LlmConfig::Openai { model, .. }) => ("openai", model.as_str()),
        Some(LlmConfig::Anthropic { model, .. }) => ("anthropic", model.as_str()),
        Some(LlmConfig::Ollama { model, .. }) => ("ollama", model.as_str()),
        Some(LlmConfig::Custom { model, .. }) => ("custom", model.as_str()),
        None => {
            if use_color {
                println!("\x1b[33m!\x1b[0m No LLM configured. Run: unlost config llm --help");
            } else {
                println!("! No LLM configured. Run: unlost config llm --help");
            }
            return;
        }
    };
    
    // Always show what we're about to do
    if use_color {
        println!(
            "\x1b[36mi\x1b[0m Replay will process ~{} turns using \x1b[1m{}/{}\x1b[0m",
            turn_count, provider, model
        );
    } else {
        println!("Replay will process ~{} turns using {}/{}", turn_count, provider, model);
    }
    
    // Warn about expensive models and suggest alternatives
    if is_expensive_model(model) {
        if let Some((suggested, cmd)) = suggest_cheaper_model(provider, model) {
            if use_color {
                println!(
                    "\x1b[33m!\x1b[0m This model may be expensive for bulk replay. Consider using \x1b[1m{}\x1b[0m:",
                    suggested
                );
                println!("  {}", cmd);
            } else {
                println!("! This model may be expensive for bulk replay. Consider using {}:", suggested);
                println!("  {}", cmd);
            }
        }
    }
    
    println!();
}

/// Quick turn count from a transcript file (without full parsing).
fn count_turns_in_transcript(path: &Path) -> usize {
    use std::io::BufRead;
    
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return 0,
    };
    
    let reader = std::io::BufReader::new(file);
    let mut user_count = 0;
    
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        // Quick heuristic: count lines with "type":"user" and "role":"user"
        if line.contains("\"type\":\"user\"") && line.contains("\"role\":\"user\"") {
            user_count += 1;
        }
    }
    
    user_count
}

pub async fn replay(
    path: String,
    transcript_path: String,
    session_id: Option<String>,
    from_start: bool,
    dedupe: bool,
    no_llm: bool,
    embed_model: String,
    embed_cache_dir: Option<String>,
    git_grounding: bool,
) -> anyhow::Result<()> {
    let transcript_path = PathBuf::from(transcript_path);
    if !transcript_path.exists() {
        anyhow::bail!("transcript path not found at {}", transcript_path.display());
    }

    let dir_path = Path::new(&path);
    let ws = get_or_create_workspace_paths(dir_path)?;

    let repo_root = crate::workspace::git_toplevel(dir_path);

    // Collect transcript files to process
    let transcript_files: Vec<PathBuf> = if transcript_path.is_file() {
        vec![transcript_path.clone()]
    } else {
        // Directory: collect all .jsonl files
        let mut files: Vec<PathBuf> = std::fs::read_dir(&transcript_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension()
                    .map(|ext| ext == "jsonl")
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect();
        files.sort();
        if files.is_empty() {
            anyhow::bail!("No .jsonl files found in directory {}", transcript_path.display());
        }
        files
    };

    // If directory provided without explicit session_id, we need to process each file with its own session
    let multiple_sessions = transcript_path.is_dir() && session_id.is_none();

    let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();

    // Count total turns across all files for cost warning
    let total_turns: usize = transcript_files.iter().map(|f| count_turns_in_transcript(f)).sum();
    if total_turns > 0 {
        print_cost_warning(total_turns, use_color);
    }

    // Spinner + progress
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    let files_done = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));

    let spinner = if use_color {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        Some(pb)
    } else {
        None
    };

    let total_files = transcript_files.len() as u64;
    let spinner_task = if let Some(ref pb) = spinner {
        let pb = pb.clone();
        let files_done = files_done.clone();
        let done = done.clone();
        Some(tokio::spawn(async move {
            while !done.load(Ordering::Relaxed) {
                tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
                let done_count = files_done.load(Ordering::Relaxed);
                pb.set_message(format!("Replaying {} sessions ({} done)", total_files, done_count));
                pb.tick();
            }
        }))
    } else {
        None
    };

    // Process files in parallel (bounded concurrency)
    use futures_util::future::join_all;

    let max_concurrency = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));

    let mut handles = Vec::new();
    
    for file_path in transcript_files.clone() {
        let ws_id = ws.id.clone();
        let dir_path = dir_path.to_path_buf();
        let path = path.clone();
        let session_id = session_id.clone();
        let transcript_path = transcript_path.clone();
        let embed_model = embed_model.clone();
        let embed_cache_dir = embed_cache_dir.clone();
        let files_done = files_done.clone();
        let semaphore = semaphore.clone();
        let repo_root_clone = repo_root.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow::anyhow!("replay semaphore closed"))?;

            let config = FlowConfig {
                embed_model,
                embed_cache_dir,
                no_llm,
            };
            let mut flow = Flow::new(config);

            let sid = if multiple_sessions {
                // Derive session_id from filename for directory mode
                file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("could not infer session_id from filename: {}", file_path.display()))?
            } else if let Some(ref s) = session_id {
                s.clone()
            } else {
                transcript_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("missing --session-id and could not infer from transcript filename"))?
            };

            let cursor = if from_start {
                CursorState {
                    byte_offset: 0,
                    last_uuid: None,
                }
            } else {
                load_cursor(&ws_id, &sid)
            };

            let (turns, new_cursor) = parse_transcript_from_cursor(&file_path, &cursor, &dir_path)?;
            
            // If git grounding is enabled, fetch commits for the session range
            let commits = if git_grounding {
                if let Some(ref root) = repo_root_clone {
                    let min_ts = turns.iter().filter(|t| t.timestamp_ms > 0).map(|t| t.timestamp_ms).min().unwrap_or(0);
                    let max_ts = turns.iter().filter(|t| t.timestamp_ms > 0).map(|t| t.timestamp_ms).max().unwrap_or(0);
                    
                    if min_ts > 0 {
                        // Look up to 15 minutes after the last turn
                        crate::git::get_commits_for_range(root, min_ts, max_ts + 15 * 60 * 1000).ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let mut recorded = 0usize;

            let mut seen = if dedupe {
                load_turnkeys(&ws_id, &sid)
            } else {
                HashSet::new()
            };
            let mut new_keys: Vec<String> = Vec::new();

            for turn in turns {
                if dedupe {
                    if let Some(k) = turn_key(&turn) {
                        if seen.contains(&k) {
                            continue;
                        }
                        seen.insert(k.clone());
                        new_keys.push(k);
                    }
                }

                // Grounding from git logs
                let grounding_note = if let Some(ref available) = commits {
                    if turn.timestamp_ms > 0 {
                        let matches = crate::git::find_corresponding_commits(
                            turn.timestamp_ms,
                            &turn.touched_paths,
                            available,
                            5 * 60 * 1000, // 5 minute window
                        );
                        if !matches.is_empty() {
                            let hashes: Vec<_> = matches.iter().map(|c| &c.hash[..7]).collect();
                            Some(format!("Verified via git: {}", hashes.join(", ")))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let event = RecordTurnEvent {
                    directory: path.clone(),
                    user_text: turn.user_text,
                    assistant_text: turn.assistant_text,
                    touched_paths: turn.touched_paths,
                    agent_kind: AgentKind::Claude,
                    agent_session_id: Some(sid.clone()),
                    usage: turn.usage,
                    grounding_note,
                };
                let result = flow.record_turn(event).await;
                if result.error.is_none() {
                    recorded += 1;
                }
            }
            
            // Wait for background flush jobs before marking session done
            flow.drain().await;

            if dedupe {
                append_turnkeys(&ws_id, &sid, &new_keys);
            }
            save_cursor(&ws_id, &sid, &new_cursor);

            files_done.fetch_add(1, Ordering::Relaxed);

            Ok::<_, anyhow::Error>((file_path, recorded))
        });
        
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let results = join_all(handles).await;
    
    done.store(true, Ordering::Relaxed);

    if let Some(task) = spinner_task {
        let _ = task.await;
    }
    if let Some(pb) = spinner {
        pb.finish_and_clear();
    }

    // Aggregate results
    let mut grand_recorded = 0usize;
    let mut files_processed = 0usize;

    for result in results {
        match result {
            Ok(Ok((_file_path, recorded))) => {
                grand_recorded += recorded;
                files_processed += 1;
            }
            Ok(Err(e)) => {
                eprintln!("✗ Error processing file: {}", e);
            }
            Err(e) => {
                eprintln!("✗ Task panicked: {}", e);
            }
        }
    }

    println!();
    if use_color {
        println!(
            "\x1b[1;32m✓\x1b[0m Replay complete: \x1b[1;36m{}\x1b[0m sessions, \x1b[1;32m{}\x1b[0m capsules recorded",
            files_processed, grand_recorded
        );
    } else {
        println!(
            "Replay complete: {} sessions, {} capsules recorded",
            files_processed, grand_recorded
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn mk_user(uuid: &str, text: &str) -> String {
        format!(
            r#"{{"type":"user","isSidechain":false,"uuid":"{uuid}","message":{{"role":"user","content":[{{"type":"text","text":{text_json}}}]}}}}"#,
            uuid = uuid,
            text_json = serde_json::to_string(text).unwrap()
        )
    }

    fn mk_assistant(uuid: &str, text: &str) -> String {
        format!(
            r#"{{"type":"assistant","isSidechain":false,"uuid":"{uuid}","message":{{"role":"assistant","model":"claude","content":[{{"type":"text","text":{text_json}}}]}}}}"#,
            uuid = uuid,
            text_json = serde_json::to_string(text).unwrap()
        )
    }

    fn mk_tool_result(uuid: &str, tool_use_id: &str, content: &str, is_error: bool) -> String {
        format!(
            r#"{{"type":"user","isSidechain":false,"uuid":"{uuid}","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":{tool_use_id_json},"content":{content_json},"is_error":{is_error}}}]}}}}"#,
            uuid = uuid,
            tool_use_id_json = serde_json::to_string(tool_use_id).unwrap(),
            content_json = serde_json::to_string(content).unwrap(),
            is_error = if is_error { "true" } else { "false" }
        )
    }

    #[test]
    fn test_parse_resumes_from_byte_offset_without_uuid_gate() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("t.jsonl");

        {
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "{}", mk_user("u1", "hello")).unwrap();
            writeln!(f, "{}", mk_assistant("a1", "hi")).unwrap();
        }

        let cwd = td.path();
        let (t1, c1) = parse_transcript_from_cursor(&p, &CursorState::default(), cwd).unwrap();
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].user_text, "hello");
        assert_eq!(t1[0].assistant_text, "hi");

        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
            writeln!(f, "{}", mk_user("u2", "next")).unwrap();
            writeln!(f, "{}", mk_assistant("a2", "ok")).unwrap();
        }

        let (t2, _c2) = parse_transcript_from_cursor(&p, &c1, cwd).unwrap();
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].user_text, "next");
        assert_eq!(t2[0].assistant_text, "ok");
    }

    #[test]
    fn test_tool_result_is_preserved_in_assistant_context() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("t.jsonl");

        {
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "{}", mk_user("u1", "run tests")).unwrap();
            writeln!(
                f,
                "{}",
                mk_tool_result("tr1", "toolu_1", "Exit code 127\\npython: not found", true)
            )
            .unwrap();
            writeln!(f, "{}", mk_assistant("a1", "switch to python3")).unwrap();
        }

        let cwd = td.path();
        let (t, _c) = parse_transcript_from_cursor(&p, &CursorState::default(), cwd).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].user_text, "run tests");
        assert!(t[0].assistant_text.contains("Tool result (error)"));
        assert!(t[0].assistant_text.contains("Exit code 127"));
        assert!(t[0].assistant_text.contains("switch to python3"));
    }

    #[test]
    fn test_user_only_turn_is_recorded() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("t.jsonl");

        {
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "{}", mk_user("u1", "i'm not happy")).unwrap();
        }

        let cwd = td.path();
        let (t, _c) = parse_transcript_from_cursor(&p, &CursorState::default(), cwd).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].user_text, "i'm not happy");
        assert!(t[0].assistant_text.trim().is_empty());
    }
}
