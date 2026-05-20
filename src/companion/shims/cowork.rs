//! Claude Cowork hooks shim.
//!
//! Invoked by Cowork hooks (UserPromptSubmit, Stop) via stdin JSON.
//! The wire protocol is identical to Claude Code hooks — same field names,
//! same transcript JSONL format — so this file is a thin re-skin of
//! `claude.rs` with Cowork-specific cursor paths and source pointers.
//!
//! Hook events handled:
//!   UserPromptSubmit — friction check; returns additionalContext if needed.
//!   Stop             — parse transcript, record new turns. Run with async:true.

use crate::companion::flow::{
    AgentKind, CheckEvent, Flow, FlowConfig, RecordTurnEvent, UsageEvent,
};
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// Wire protocol (identical to Claude Code)
// ============================================================================

#[derive(Debug, Deserialize)]
struct HookInput {
    hook_event_name: String,
    session_id: String,
    cwd: String,
    #[serde(default)]
    transcript_path: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

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
// Transcript types (same JSONL schema as Claude Code)
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
// Cursor state
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Default)]
struct CursorState {
    byte_offset: u64,
    last_uuid: Option<String>,
}

fn cursor_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("cowork")
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

fn turnkeys_path(workspace_id: &str, session_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("cowork")
        .join(format!("{}.turnkeys", session_id))
}

fn load_turnkeys(workspace_id: &str, session_id: &str) -> HashSet<String> {
    let path = turnkeys_path(workspace_id, session_id);
    match std::fs::read_to_string(&path) {
        Ok(s) => s
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
        Err(_) => HashSet::new(),
    }
}

fn append_turnkeys(workspace_id: &str, session_id: &str, keys: &[String]) {
    if keys.is_empty() {
        return;
    }
    let path = turnkeys_path(workspace_id, session_id);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
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
// Transcript parsing (shared helpers, mirrored from claude.rs)
// ============================================================================

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

fn build_source_pointer(transcript_path: &Path, turn: &ParsedTurn) -> Option<String> {
    let path_str = transcript_path.to_str()?;
    if path_str.is_empty() {
        return None;
    }
    let uuid = turn
        .user_uuid
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    match uuid {
        Some(u) => Some(format!("cowork+jsonl://{path_str}#turn={u}")),
        None => Some(format!("cowork+jsonl://{path_str}")),
    }
}

fn truncate(mut s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    s.truncate(max_bytes);
    s.push_str("\n...(truncated)");
    s
}

fn value_to_compact_text(v: &serde_json::Value, max_bytes: usize) -> String {
    match v {
        serde_json::Value::String(s) => truncate(s.clone(), max_bytes),
        serde_json::Value::Array(a) => {
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
                truncate(v.to_string(), max_bytes)
            } else {
                truncate(out, max_bytes)
            }
        }
        _ => truncate(v.to_string(), max_bytes),
    }
}

fn extract_tool_result_text(content: &serde_json::Value) -> Option<String> {
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
        let is_error = obj
            .get("is_error")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let tool_use_id = obj
            .get("tool_use_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let text = value_to_compact_text(
            obj.get("content").unwrap_or(&serde_json::Value::Null),
            12 * 1024,
        );
        if text.trim().is_empty() {
            continue;
        }
        let mut s = if is_error {
            "Tool result (error)".to_string()
        } else {
            "Tool result".to_string()
        };
        if !tool_use_id.is_empty() {
            s.push(' ');
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

fn looks_like_path(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty()
        && s.len() <= 260
        && !s.contains('\n')
        && !s.contains('\r')
        && !s.contains(' ')
        && (s.contains('/') || s.contains('\\'))
}

fn normalize_path(p: &str, cwd: &Path) -> Option<String> {
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
                if let Some(p) = normalize_path(s, cwd) {
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

fn paths_from_content(content: &serde_json::Value, cwd: &Path) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();
    if let serde_json::Value::Array(blocks) = content {
        for block in blocks {
            if let Some(obj) = block.as_object() {
                let ty = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if ty == "tool_use" || ty == "tool_result" {
                    for key in ["input", "result"] {
                        if let Some(v) = obj.get(key) {
                            collect_paths(v, cwd, &mut out);
                        }
                    }
                }
            }
        }
    }
    let mut v: Vec<_> = out.into_iter().collect();
    v.sort();
    v.truncate(64);
    v
}

fn extract_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => {
            let mut texts = Vec::new();
            for block in blocks {
                if let Some(obj) = block.as_object() {
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

fn is_tool_result(content: &serde_json::Value) -> bool {
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

fn parse_transcript(
    transcript_path: &Path,
    cursor: &CursorState,
    cwd: &Path,
) -> anyhow::Result<(Vec<ParsedTurn>, CursorState)> {
    let file = std::fs::File::open(transcript_path)?;
    let file_size = file.metadata()?.len();
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
    let mut pending_touched: HashSet<String> = HashSet::new();
    let mut current_usage: Option<UsageEvent> = None;
    let mut last_uuid: Option<String> = cursor.last_uuid.clone();
    let mut new_offset = start_offset;
    let mut seen_last_uuid = start_offset > 0 || cursor.last_uuid.is_none();

    for line in reader.lines() {
        let line = line?;
        new_offset += line.len() as u64 + 1;

        if line.trim().is_empty() {
            continue;
        }

        let parsed: TranscriptLine = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(_) => continue,
        };

        if let Some(ts) = parsed.extra.get("time").and_then(|v| v.as_i64()) {
            current_timestamp_ms = ts;
        }
        for val in parsed.extra.values() {
            collect_paths(val, cwd, &mut pending_touched);
        }

        let line_type = match &parsed.line_type {
            Some(t) => t.as_str(),
            None => continue,
        };

        if line_type == "file-history-snapshot" {
            for val in parsed.extra.values() {
                collect_paths(val, cwd, &mut pending_touched);
            }
            if let Some(uuid) = &parsed.uuid {
                last_uuid = Some(uuid.clone());
            }
            continue;
        }

        if parsed.is_sidechain == Some(true) {
            continue;
        }

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

        for p in paths_from_content(content, cwd) {
            pending_touched.insert(p);
        }

        match (line_type, role) {
            ("user", "user") => {
                if is_tool_result(content) {
                    if current_user_text.is_some() {
                        if let Some(tool_text) = extract_tool_result_text(content) {
                            if !current_assistant_text.is_empty() {
                                current_assistant_text.push_str("\n\n");
                            }
                            current_assistant_text.push_str(&tool_text);
                        }
                    }
                    if let Some(uuid) = &parsed.uuid {
                        last_uuid = Some(uuid.clone());
                    }
                    continue;
                }

                let text = extract_text(content);
                if text.trim().is_empty() {
                    continue;
                }

                if let Some(user_text) = current_user_text.take() {
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
            }
            ("assistant", "assistant") => {
                let text = extract_text(content);
                if !text.trim().is_empty() {
                    if !current_assistant_text.is_empty() {
                        current_assistant_text.push('\n');
                    }
                    current_assistant_text.push_str(&text);
                }
                if parsed.uuid.is_some() {
                    current_last_assistant_uuid = parsed.uuid.clone();
                }
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

    Ok((
        turns,
        CursorState {
            byte_offset: new_offset,
            last_uuid,
        },
    ))
}

// ============================================================================
// Hook handlers
// ============================================================================

async fn handle_user_prompt_submit(flow: &mut Flow, input: &HookInput) -> anyhow::Result<()> {
    let prompt = match &input.prompt {
        Some(p) => p,
        None => {
            tracing::warn!("cowork UserPromptSubmit missing prompt field");
            return Ok(());
        }
    };

    let result = flow
        .check_turn(CheckEvent {
            directory: input.cwd.clone(),
            text: prompt.clone(),
            agent_kind: AgentKind::Cowork,
            agent_session_id: Some(input.session_id.clone()),
        })
        .await;

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

    println!("{}", serde_json::to_string(&output)?);
    std::io::stdout().flush()?;
    Ok(())
}

async fn handle_stop(
    flow: &mut Flow,
    input: &HookInput,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    let transcript_path = match &input.transcript_path {
        Some(p) => PathBuf::from(p),
        None => {
            tracing::warn!("cowork Stop hook missing transcript_path");
            return Ok(());
        }
    };

    if !transcript_path.exists() {
        tracing::warn!(path = %transcript_path.display(), "cowork: transcript file not found");
        return Ok(());
    }

    let cwd_path = Path::new(&input.cwd);
    let ws = match get_or_create_workspace_paths(cwd_path) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(error = %e, "cowork: failed to get workspace paths");
            return Ok(());
        }
    };

    let cursor = load_cursor(&ws.id, &input.session_id);

    // First Stop hook for this session — trigger a background checkpoint for prior sessions.
    if cursor.byte_offset == 0 && cursor.last_uuid.is_none() {
        let ws_clone = ws.clone();
        tokio::spawn(async move {
            match crate::storage_checkpoint::maybe_create_checkpoint(
                &ws_clone,
                None,
                "new_session_cowork",
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

    let (turns, new_cursor) = parse_transcript(&transcript_path, &cursor, cwd_path)?;

    tracing::info!(
        turns_count = turns.len(),
        old_offset = cursor.byte_offset,
        new_offset = new_cursor.byte_offset,
        "cowork: parsed transcript"
    );

    let mut seen = load_turnkeys(&ws.id, &input.session_id);
    let mut new_keys: Vec<String> = Vec::new();
    let mut all_touched: Vec<String> = Vec::new();
    let mut all_assistant_texts: Vec<String> = Vec::new();

    for turn in turns {
        all_touched.extend(turn.touched_paths.iter().cloned());
        all_assistant_texts.push(turn.assistant_text.clone());

        if let Some(k) = turn_key(&turn) {
            if seen.contains(&k) {
                continue;
            }
            seen.insert(k.clone());
            new_keys.push(k);
        }

        let source_pointer = build_source_pointer(&transcript_path, &turn);
        let result = flow
            .record_turn(RecordTurnEvent {
                directory: input.cwd.clone(),
                user_text: turn.user_text,
                assistant_text: turn.assistant_text,
                touched_paths: turn.touched_paths,
                tool_calls: vec![],
                agent_kind: AgentKind::Cowork,
                agent_session_id: Some(input.session_id.clone()),
                usage: turn.usage,
                grounding_note: None,
                source_ts_ms: None,
                source_pointer,
            })
            .await;
        if let Some(error) = result.error {
            tracing::warn!(error = %error, "cowork: record_turn failed");
        }
    }

    append_turnkeys(&ws.id, &input.session_id, &new_keys);
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
                tracing::info!(pr_url, "cowork: detected PR creation — posting unlost comment");
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
                    tracing::warn!(error = ?e, pr_url, "cowork: pr-comment spawn failed");
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
                let _ =
                    crate::changelog::ingest_changelog(&ws, &changelog_path, embedder, use_color)
                        .await;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Entry points
// ============================================================================

pub async fn run(embed_model: String, embed_cache_dir: Option<String>) -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let hook_input: HookInput = serde_json::from_str(&input)?;

    tracing::info!(
        hook_event = %hook_input.hook_event_name,
        session_id = %hook_input.session_id,
        cwd = %hook_input.cwd,
        "cowork hook invoked"
    );

    let config = FlowConfig {
        embed_model: embed_model.clone(),
        embed_cache_dir: embed_cache_dir.clone(),
        extraction_mode: crate::types::ExtractionMode::Hybrid,
    };
    let mut flow = Flow::new(config);

    let needs_drain = hook_input.hook_event_name == "Stop";

    match hook_input.hook_event_name.as_str() {
        "UserPromptSubmit" => handle_user_prompt_submit(&mut flow, &hook_input).await?,
        "Stop" => {
            handle_stop(
                &mut flow,
                &hook_input,
                &embed_model,
                embed_cache_dir.as_deref(),
            )
            .await?
        }
        _ => {
            tracing::debug!(event = %hook_input.hook_event_name, "cowork: ignoring unhandled hook event");
        }
    }

    if needs_drain {
        flow.drain().await;
    }

    Ok(())
}

pub async fn replay(
    path: String,
    transcript_path: String,
    session_id: Option<String>,
    from_start: bool,
    dedupe: bool,
    clear: bool,
    extraction_mode: crate::types::ExtractionMode,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let dir_path = Path::new(&path);
    let transcript_path_buf = PathBuf::from(&transcript_path);

    let ws = get_or_create_workspace_paths(dir_path)?;

    if clear {
        clear_replay_data(&ws, session_id.as_deref())?;
    }

    let transcript_files: Vec<PathBuf> = if transcript_path_buf.is_file() {
        vec![transcript_path_buf.clone()]
    } else {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&transcript_path_buf)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "jsonl")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        files.sort();
        if files.is_empty() {
            anyhow::bail!(
                "No .jsonl files found in directory {}",
                transcript_path_buf.display()
            );
        }
        files
    };

    let multiple_sessions = transcript_path_buf.is_dir() && session_id.is_none();
    let use_color =
        std::io::IsTerminal::is_terminal(&std::io::stdout()) && std::env::var_os("NO_COLOR").is_none();

    use futures_util::future::join_all;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    let max_concurrency = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    let files_done = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for file_path in transcript_files {
        let ws_id = ws.id.clone();
        let dir_path = dir_path.to_path_buf();
        let path = path.clone();
        let session_id = session_id.clone();
        let transcript_path_buf = transcript_path_buf.clone();
        let embed_model = embed_model.clone();
        let embed_cache_dir = embed_cache_dir.clone();
        let files_done = files_done.clone();
        let semaphore = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow::anyhow!("semaphore closed"))?;

            let config = FlowConfig {
                embed_model,
                embed_cache_dir,
                extraction_mode,
            };
            let mut flow = Flow::new(config);

            let sid = if multiple_sessions {
                file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("could not infer session_id from filename"))?
            } else if let Some(ref s) = session_id {
                s.clone()
            } else {
                transcript_path_buf
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("missing --session-id"))?
            };

            let cursor = if from_start {
                CursorState::default()
            } else {
                load_cursor(&ws_id, &sid)
            };

            let (turns, new_cursor) = parse_transcript(&file_path, &cursor, &dir_path)?;

            let mut seen = if dedupe {
                load_turnkeys(&ws_id, &sid)
            } else {
                HashSet::new()
            };
            let mut new_keys: Vec<String> = Vec::new();
            let mut recorded = 0usize;

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

                let source_pointer = build_source_pointer(&file_path, &turn);
                let event = RecordTurnEvent {
                    directory: path.clone(),
                    user_text: turn.user_text,
                    assistant_text: turn.assistant_text,
                    touched_paths: turn.touched_paths,
                    tool_calls: vec![],
                    agent_kind: AgentKind::Cowork,
                    agent_session_id: Some(sid.clone()),
                    usage: turn.usage,
                    grounding_note: None,
                    source_ts_ms: if turn.timestamp_ms > 0 {
                        Some(turn.timestamp_ms)
                    } else {
                        None
                    },
                    source_pointer,
                };
                let result = flow.record_turn(event).await;
                if result.error.is_none() {
                    recorded += 1;
                }
            }

            flow.drain().await;

            if dedupe {
                append_turnkeys(&ws_id, &sid, &new_keys);
            }
            save_cursor(&ws_id, &sid, &new_cursor);
            files_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            Ok::<_, anyhow::Error>((file_path, recorded, flow.llm_calls().await))
        });

        handles.push(handle);
    }

    let results = join_all(handles).await;

    let mut grand_recorded = 0usize;
    for result in results {
        match result {
            Ok(Ok((_file, recorded, _llm))) => grand_recorded += recorded,
            Ok(Err(e)) => eprintln!("Error: {}", e),
            Err(e) => eprintln!("Task panicked: {}", e),
        }
    }

    if use_color {
        println!(
            "\x1b[1;32m✓\x1b[0m Replay complete: \x1b[1;36m{}\x1b[0m turns indexed",
            grand_recorded
        );
    } else {
        println!("Replay complete: {} turns indexed", grand_recorded);
    }

    Ok(())
}

fn clear_replay_data(ws: &crate::workspace::WorkspacePaths, session_id: Option<&str>) -> anyhow::Result<()> {
    let base = crate::workspace::unlost_workspace_dir(&ws.id).join("cowork");
    if !base.exists() {
        return Ok(());
    }
    if let Some(sid) = session_id {
        let _ = std::fs::remove_file(base.join(format!("{}.cursor", sid)));
        let _ = std::fs::remove_file(base.join(format!("{}.turnkeys", sid)));
    } else {
        let _ = std::fs::remove_dir_all(&base);
    }
    Ok(())
}
