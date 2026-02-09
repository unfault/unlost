//! Claude Code hooks shim.
//!
//! Invoked by Claude Code hooks (UserPromptSubmit, Stop) via stdin JSON.
//! Reads hook event, dispatches to Flow.check() or transcript ingestion.
//!
//! Cursor state is persisted per-session to avoid re-processing transcript lines.

use crate::companion::flow::{
    AgentKind, CheckEvent, Flow, FlowConfig, RecordTurnEvent, UsageEvent,
};
use crate::workspace::get_or_create_workspace_paths;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

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
        .join("claudecode")
        .join(format!("{}.cursor", session_id))
}

fn load_cursor(workspace_id: &str, session_id: &str) -> CursorState {
    let path = cursor_path(workspace_id, session_id);
    if let Ok(data) = std::fs::read_to_string(&path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        CursorState::default()
    }
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

// ============================================================================
// Transcript parsing
// ============================================================================

/// A parsed turn from the transcript (user prompt + assistant response)
struct ParsedTurn {
    user_text: String,
    assistant_text: String,
    usage: Option<UsageEvent>,
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
    let mut current_assistant_text = String::new();

    let mut current_usage: Option<UsageEvent> = None;
    let mut last_uuid: Option<String> = cursor.last_uuid.clone();
    let mut new_offset = start_offset;
    let mut seen_last_uuid = cursor.last_uuid.is_none();

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

        // Skip non-message lines (file-history-snapshot, etc.)
        let line_type = match &parsed.line_type {
            Some(t) => t.as_str(),
            None => continue,
        };

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

        match (line_type, role) {
            ("user", "user") => {
                // Skip tool_result messages
                if is_tool_result_message(content) {
                    continue;
                }

                let text = extract_text_content(content);
                if text.trim().is_empty() {
                    continue;
                }

                // If we have a pending turn, save it
                if let Some(user_text) = current_user_text.take() {
                    if !current_assistant_text.trim().is_empty() {
                        turns.push(ParsedTurn {
                            user_text,
                            assistant_text: std::mem::take(&mut current_assistant_text),
                            usage: current_usage.take(),
                        });
                    }
                }

                current_user_text = Some(text);
                current_assistant_text.clear();
                current_usage = None;
            }
            ("assistant", "assistant") => {
                let text = extract_text_content(content);
                if !text.trim().is_empty() {
                    if !current_assistant_text.is_empty() {
                        current_assistant_text.push('\n');
                    }
                    current_assistant_text.push_str(&text);
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
        if !current_assistant_text.trim().is_empty() {
            turns.push(ParsedTurn {
                user_text,
                assistant_text: current_assistant_text,
                usage: current_usage,
            });
        }
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

pub(crate) async fn run(embed_model: String, embed_cache_dir: Option<String>) -> anyhow::Result<()> {
    // Read hook input from stdin
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let hook_input: HookInput = serde_json::from_str(&input)?;

    tracing::info!(
        hook_event = %hook_input.hook_event_name,
        session_id = %hook_input.session_id,
        cwd = %hook_input.cwd,
        "claudecode hook invoked"
    );

    let config = FlowConfig {
        embed_model,
        embed_cache_dir,
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
        agent_kind: AgentKind::ClaudeCode,
        agent_session_id: Some(input.session_id.clone()),
    };

    let result = flow.check(event).await;

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
    let (turns, new_cursor) = parse_transcript_from_cursor(&transcript_path, &cursor)?;

    tracing::info!(
        turns_count = turns.len(),
        old_offset = cursor.byte_offset,
        new_offset = new_cursor.byte_offset,
        "parsed transcript"
    );

    // Record each turn
    for turn in turns {
        let event = RecordTurnEvent {
            directory: input.cwd.clone(),
            user_text: turn.user_text,
            assistant_text: turn.assistant_text,
            agent_kind: AgentKind::ClaudeCode,
            agent_session_id: Some(input.session_id.clone()),
            usage: turn.usage,
        };

        let result = flow.record_turn(event).await;
        if let Some(error) = result.error {
            tracing::warn!(error = %error, "record_turn failed");
        }
    }

    // Save cursor
    save_cursor(&ws.id, &input.session_id, &new_cursor);

    Ok(())
}
