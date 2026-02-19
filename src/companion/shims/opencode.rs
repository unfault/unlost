//! OpenCode replay shim.
//!
//! Replays OpenCode messages from disk storage into unlost capsules.
//! 
//! OpenCode stores data in `~/.local/share/opencode/storage/`:
//! - `session/<project-hash>/ses_*.json` - Session metadata with `directory` field
//! - `message/ses_*/msg_*.json` - Messages (mixed across all projects)
//!
//! Session files contain:
//! - `id`: Session ID (matches message's `sessionID`)
//! - `projectID`: Hash of the project
//! - `directory`: The workspace path
//!
//! Message files contain:
//! - `id`, `sessionID`, `role` (user/assistant)
//! - `time.created` (epoch ms)
//! - `summary.title` - A summary/title of the message (user text)
//! - `summary.diffs` - File diffs with before/after content
//! - `model.providerID`, `model.modelID`
//! - `tokens`, `cost` (for assistant messages)
//! - `parentID` (links assistant to user message)

use crate::companion::flow::{AgentKind, Flow, FlowConfig, RecordTurnEvent, UsageEvent};
use crate::workspace::get_or_create_workspace_paths;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// OpenCode storage types
// ============================================================================

#[derive(Debug, Deserialize)]
struct SessionFile {
    id: String,
    #[serde(rename = "projectID")]
    project_id: String,
    directory: String,
    #[allow(dead_code)]
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageFile {
    id: String,
    #[serde(rename = "sessionID")]
    #[allow(dead_code)]
    session_id: String,
    role: String,
    time: MessageTime,
    summary: Option<MessageSummary>,
    #[serde(rename = "parentID")]
    parent_id: Option<String>,
    #[serde(rename = "providerID")]
    provider_id: Option<String>,
    #[serde(rename = "modelID")]
    model_id: Option<String>,
    tokens: Option<MessageTokens>,
    cost: Option<f64>,
    #[allow(dead_code)]
    path: Option<MessagePath>,
}

#[derive(Debug, Deserialize)]
struct MessageTime {
    created: i64,
    #[allow(dead_code)]
    completed: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct MessageSummary {
    title: Option<String>,
    diffs: Option<Vec<MessageDiff>>,
}

#[derive(Debug, Deserialize)]
struct MessageDiff {
    file: Option<String>,
    #[allow(dead_code)]
    before: Option<String>,
    #[allow(dead_code)]
    after: Option<String>,
    #[allow(dead_code)]
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageTokens {
    input: Option<i64>,
    output: Option<i64>,
    reasoning: Option<i64>,
    cache: Option<MessageCache>,
}

#[derive(Debug, Deserialize)]
struct MessageCache {
    read: Option<i64>,
    write: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessagePath {
    cwd: Option<String>,
    root: Option<String>,
}

// ============================================================================
// Cursor state (persisted per workspace to track replayed messages)
// ============================================================================

pub(crate) fn replayed_path(workspace_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id)
        .join("opencode")
        .join("replayed.txt")
}

pub(crate) fn load_replayed(workspace_id: &str) -> HashSet<String> {
    let path = replayed_path(workspace_id);
    let data = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return HashSet::new(),
    };
    data.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

pub(crate) fn append_replayed(workspace_id: &str, ids: &[String]) {
    if ids.is_empty() {
        return;
    }
    let path = replayed_path(workspace_id);
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
    for id in ids {
        if id.trim().is_empty() {
            continue;
        }
        let _ = writeln!(f, "{}", id.trim());
    }
}

// ============================================================================
// Storage discovery
// ============================================================================

fn xdg_data_home() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".local").join("share");
    }
    PathBuf::from(".")
}

fn opencode_storage_dir() -> PathBuf {
    xdg_data_home()
        .join("opencode")
        .join("storage")
}

/// Find all sessions that map to the given workspace directory.
fn find_sessions_for_workspace(workspace_dir: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let storage = opencode_storage_dir();
    let session_root = storage.join("session");

    if !session_root.exists() {
        return Ok(Vec::new());
    }

    let workspace_canonical = workspace_dir.canonicalize().unwrap_or_else(|_| workspace_dir.to_path_buf());
    let mut results: Vec<(String, String)> = Vec::new(); // (session_id, project_id)

    // Iterate over project hash directories
    for project_entry in std::fs::read_dir(&session_root)? {
        let project_entry = project_entry?;
        let project_path = project_entry.path();
        if !project_path.is_dir() {
            continue;
        }

        // Iterate over session files in this project
        for session_entry in std::fs::read_dir(&project_path)? {
            let session_entry = session_entry?;
            let session_file_path = session_entry.path();
            if session_file_path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }

            // Parse session file
            let content = match std::fs::read_to_string(&session_file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let session: SessionFile = match serde_json::from_str(&content) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Check if this session's directory matches our workspace
            let session_dir = Path::new(&session.directory);
            let session_canonical = session_dir.canonicalize().unwrap_or_else(|_| session_dir.to_path_buf());

            if session_canonical == workspace_canonical {
                results.push((session.id, session.project_id));
            }
        }
    }

    Ok(results)
}

/// Load all messages for the given session IDs.
fn load_messages_for_sessions(
    session_ids: &HashSet<String>,
) -> anyhow::Result<HashMap<String, Vec<MessageFile>>> {
    let storage = opencode_storage_dir();
    let message_root = storage.join("message");

    if !message_root.exists() {
        return Ok(HashMap::new());
    }

    let mut results: HashMap<String, Vec<MessageFile>> = HashMap::new();

    // Iterate over session directories in message/
    for entry in std::fs::read_dir(&message_root)? {
        let entry = entry?;
        let session_dir = entry.path();
        if !session_dir.is_dir() {
            continue;
        }

        // Check if this session directory matches one of our target sessions
        let dir_name = match session_dir.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };

        if !session_ids.contains(dir_name) {
            continue;
        }

        // Load all messages in this session
        let mut messages: Vec<MessageFile> = Vec::new();
        for msg_entry in std::fs::read_dir(&session_dir)? {
            let msg_entry = msg_entry?;
            let msg_path = msg_entry.path();
            if msg_path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }

            let content = match std::fs::read_to_string(&msg_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let msg: MessageFile = match serde_json::from_str(&content) {
                Ok(m) => m,
                Err(_) => continue,
            };

            messages.push(msg);
        }

        // Sort by creation time
        messages.sort_by_key(|m| m.time.created);
        results.insert(dir_name.to_string(), messages);
    }

    Ok(results)
}

/// A parsed turn from OpenCode messages (user + assistant).
struct ParsedTurn {
    user_text: String,
    assistant_text: String,
    usage: Option<UsageEvent>,
    touched_paths: Vec<String>,
    turn_key: String, // user_msg_id:assistant_msg_id
    timestamp_ms: i64,
}

/// Read full message text from parts if available.
fn read_message_full_text(message_id: &str) -> Option<String> {
    let storage = opencode_storage_dir();
    let part_dir = storage.join("part").join(message_id);
    if !part_dir.exists() {
        return None;
    }

    let mut parts = Vec::new();
    if let Ok(entries) = std::fs::read_dir(part_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                        if v.get("type").and_then(|t| t.as_str()) == Some("text") {
                            let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("");
                            let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
                            if !text.is_empty() {
                                parts.push((id.to_string(), text.to_string()));
                            }
                        }
                    }
                }
            }
        }
    }

    if parts.is_empty() {
        return None;
    }

    // Sort by part ID to maintain order
    parts.sort_by(|a, b| a.0.cmp(&b.0));
    Some(parts.into_iter().map(|p| p.1).collect::<Vec<_>>().join("\n").trim().to_string())
}

/// Extract turns from a list of messages.
fn extract_turns(messages: Vec<MessageFile>) -> Vec<ParsedTurn> {
    // Build a map of message_id -> message for quick lookup
    let msg_map: HashMap<String, &MessageFile> = messages.iter().map(|m| (m.id.clone(), m)).collect();

    let mut turns: Vec<ParsedTurn> = Vec::new();

    // Find all assistant messages and pair with their parent user message
    for msg in &messages {
        if msg.role != "assistant" {
            continue;
        }

        let parent_id = match &msg.parent_id {
            Some(p) => p,
            None => continue,
        };

        let user_msg = match msg_map.get(parent_id) {
            Some(m) => m,
            None => continue,
        };

        if user_msg.role != "user" {
            continue;
        }

        // Try reading full text from parts first, fallback to summary.title
        let user_text = read_message_full_text(&user_msg.id)
            .unwrap_or_else(|| {
                user_msg
                    .summary
                    .as_ref()
                    .and_then(|s| s.title.as_ref())
                    .cloned()
                    .unwrap_or_default()
            });

        if user_text.trim().is_empty() {
            continue;
        }

        let assistant_text = read_message_full_text(&msg.id)
            .unwrap_or_else(|| {
                msg
                    .summary
                    .as_ref()
                    .and_then(|s| s.title.as_ref())
                    .cloned()
                    .unwrap_or_default()
            });

        // Extract touched paths from diffs
        let mut touched_paths: Vec<String> = Vec::new();
        if let Some(summary) = &user_msg.summary {
            if let Some(diffs) = &summary.diffs {
                for diff in diffs {
                    if let Some(file) = &diff.file {
                        if !file.trim().is_empty() {
                            touched_paths.push(file.clone());
                        }
                    }
                }
            }
        }
        if let Some(summary) = &msg.summary {
            if let Some(diffs) = &summary.diffs {
                for diff in diffs {
                    if let Some(file) = &diff.file {
                        if !file.trim().is_empty() && !touched_paths.contains(file) {
                            touched_paths.push(file.clone());
                        }
                    }
                }
            }
        }

        // Build usage from assistant message
        let usage = msg.tokens.as_ref().map(|t| UsageEvent {
            provider_id: msg.provider_id.clone(),
            model_id: msg.model_id.clone(),
            cost: msg.cost,
            tokens_input: t.input,
            tokens_output: t.output,
            tokens_reasoning: t.reasoning,
            tokens_cache_read: t.cache.as_ref().and_then(|c| c.read),
            tokens_cache_write: t.cache.as_ref().and_then(|c| c.write),
        });

        let turn_key = format!("{}:{}", user_msg.id, msg.id);

        turns.push(ParsedTurn {
            user_text,
            assistant_text,
            usage,
            touched_paths,
            turn_key,
            timestamp_ms: msg.time.created,
        });
    }

    turns
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
fn print_cost_warning(turn_count: usize, mode: crate::types::ExtractionMode, use_color: bool) {
    use crate::workspace::load_workspace_config;
    
    let cfg = load_workspace_config();
    
    let (provider, model) = match &cfg.llm {
        Some(crate::config::LlmConfig::Openai { model, .. }) => ("openai", model.as_str()),
        Some(crate::config::LlmConfig::Anthropic { model, .. }) => ("anthropic", model.as_str()),
        Some(crate::config::LlmConfig::Ollama { model, .. }) => ("ollama", model.as_str()),
        Some(crate::config::LlmConfig::Custom { model, .. }) => ("custom", model.as_str()),
        None => {
            if use_color {
                println!("\x1b[33m!\x1b[0m No LLM provider configured. Replay will proceed with heuristic extraction.");
            } else {
                println!("! No LLM provider configured. Replay will proceed with heuristic extraction.");
            }
            return;
        }
    };

    // Only warn for cloud providers
    if provider == "ollama" {
        return;
    }

    let is_expensive = is_expensive_model(model);
    
    if use_color {
        println!("\x1b[34m?\x1b[0m Replaying \x1b[1m{}\x1b[0m turns using \x1b[32m{}/{}\x1b[0m", turn_count, provider, model);
        match mode {
            crate::types::ExtractionMode::Hybrid => {
                println!("  \x1b[34m*\x1b[0m Hybrid Mode: Local search indexing for all turns; LLM analysis only for pivotal moments.");
            }
            crate::types::ExtractionMode::Full => {
                println!("  \x1b[33m!\x1b[0m Full Extraction: Extracting every turn (highest quality, highest cost).");
            }
            _ => {}
        }
    } else {
        println!("? Replaying {} turns using {}/{}", turn_count, provider, model);
        match mode {
            crate::types::ExtractionMode::Hybrid => {
                println!("  * Hybrid Mode: Local search indexing for all turns; LLM analysis only for pivotal moments.");
            }
            crate::types::ExtractionMode::Full => {
                println!("  ! Full Extraction: Extracting every turn (highest quality, highest cost).");
            }
            _ => {}
        }
    }

    if is_expensive {
        if use_color {
            println!("\x1b[33m!\x1b[0m \x1b[1mWarning:\x1b[0m This model is expensive for bulk replay.");
            if let Some((suggested, cmd)) = suggest_cheaper_model(provider, model) {
                println!("  Consider using \x1b[32m{}\x1b[0m for faster, cheaper replays:", suggested);
                println!("  \x1b[36m{}\x1b[0m", cmd);
            }
        } else {
            println!("! Warning: This model is expensive for bulk replay.");
            if let Some((suggested, cmd)) = suggest_cheaper_model(provider, model) {
                println!("  Consider using {} for faster, cheaper replays:", suggested);
                println!("  {}", cmd);
            }
        }
    }
    
    println!();
}

// ============================================================================
// Replay entry point
// ============================================================================

pub async fn replay(
    path: String,
    dedupe: bool,
    clear: bool,
    extraction_mode: crate::types::ExtractionMode,
    embed_model: String,
    embed_cache_dir: Option<String>,
    git_grounding: bool,
) -> anyhow::Result<()> {
    let dir_path = Path::new(&path);
    let ws = get_or_create_workspace_paths(dir_path)?;

    if clear {
        clear_replay_data(&ws, "opencode", None)?;
    }

    let repo_root = crate::workspace::git_toplevel(dir_path);

    let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();

    // Find all sessions for this workspace
    let sessions = find_sessions_for_workspace(dir_path)?;
    if sessions.is_empty() {
        if use_color {
            println!("\x1b[33m!\x1b[0m No OpenCode sessions found for this workspace");
        } else {
            println!("No OpenCode sessions found for this workspace");
        }
        println!("  Workspace: {}", dir_path.display());
        println!("  Storage: {}", opencode_storage_dir().display());
        return Ok(());
    }

    let session_ids: HashSet<String> = sessions.iter().map(|(s, _)| s.clone()).collect();

    // Load messages for all sessions first (to count turns for cost warning)
    let mut messages_by_session = load_messages_for_sessions(&session_ids)?;

    // Count total turns across all sessions for cost warning
    let total_turns: usize = messages_by_session
        .values()
        .map(|msgs| {
            // Count assistant messages with parent_id (each is a turn)
            msgs.iter()
                .filter(|m| m.role == "assistant" && m.parent_id.is_some())
                .count()
        })
        .sum();

    // Show cost warning before processing
    if total_turns > 0 && extraction_mode != crate::types::ExtractionMode::None {
        print_cost_warning(total_turns, extraction_mode, use_color);
    }

    // Spinner setup
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    let sessions_done = Arc::new(AtomicU64::new(0));
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

    let total_sessions = session_ids.len() as u64;
    let spinner_task = if let Some(ref pb) = spinner {
        let pb = pb.clone();
        let sessions_done = sessions_done.clone();
        let done = done.clone();
        Some(tokio::spawn(async move {
            while !done.load(Ordering::Relaxed) {
                tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
                let done_count = sessions_done.load(Ordering::Relaxed);
                pb.set_message(format!(
                    "Replaying {} sessions ({} done)",
                    total_sessions, done_count
                ));
                pb.tick();
            }
        }))
    } else {
        None
    };

    // Process sessions in parallel
    use futures_util::future::join_all;

    let max_concurrency = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));

    let mut handles = Vec::new();

    for (session_id, _project_id) in sessions {
        let messages = match messages_by_session.remove(&session_id) {
            Some(m) => m,
            None => continue,
        };

        let ws_id = ws.id.clone();
        let path = path.clone();
        let embed_model = embed_model.clone();
        let embed_cache_dir = embed_cache_dir.clone();
        let sessions_done = sessions_done.clone();
        let semaphore = semaphore.clone();
        let session_id_clone = session_id.clone();
        let repo_root_clone = repo_root.clone();

        let handle = tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow::anyhow!("replay semaphore closed"))?;

            let config = FlowConfig {
                embed_model,
                embed_cache_dir,
                extraction_mode,
            };
            let mut flow = Flow::new(config);

            let turns = extract_turns(messages);
            
            // If git grounding is enabled, fetch commits for the session range
            let commits = if git_grounding {
                if let Some(ref root) = repo_root_clone {
                    let min_ts = turns.iter().map(|t| t.timestamp_ms).min().unwrap_or(0);
                    let max_ts = turns.iter().map(|t| t.timestamp_ms).max().unwrap_or(0);
                    // Look up to 15 minutes after the last turn
                    crate::git::get_commits_for_range(root, min_ts, max_ts + 15 * 60 * 1000).ok()
                } else {
                    None
                }
            } else {
                None
            };

            let mut recorded = 0usize;

            let mut seen = if dedupe {
                load_replayed(&ws_id)
            } else {
                HashSet::new()
            };
            let mut new_keys: Vec<String> = Vec::new();

            for turn in turns {
                if dedupe {
                    if seen.contains(&turn.turn_key) {
                        continue;
                    }
                    seen.insert(turn.turn_key.clone());
                    new_keys.push(turn.turn_key.clone());
                }

                // Grounding from git logs
                let grounding_note = if let Some(ref available) = commits {
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
                };

                let event = RecordTurnEvent {
                    directory: path.clone(),
                    user_text: turn.user_text,
                    assistant_text: turn.assistant_text,
                    touched_paths: turn.touched_paths,
                    agent_kind: AgentKind::OpenCode,
                    agent_session_id: Some(session_id_clone.clone()),
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
                append_replayed(&ws_id, &new_keys);
            }

            sessions_done.fetch_add(1, Ordering::Relaxed);

            Ok::<_, anyhow::Error>((session_id_clone, recorded, flow.llm_calls().await))
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
    let mut total_llm_calls = 0u64;

    for result in results {
        match result {
            Ok(Ok((_session_id, recorded, llm_calls))) => {
                grand_recorded += recorded;
                total_llm_calls += llm_calls;
            }

            Ok(Err(e)) => {
                eprintln!("Error processing session: {}", e);
            }
            Err(e) => {
                eprintln!("Task panicked: {}", e);
            }
        }
    }

    println!();
    if use_color {
        print!(
            "\x1b[1;32m✓\x1b[0m Replay complete: \x1b[1;36m{}\x1b[0m turns indexed locally",
            grand_recorded
        );
        if total_llm_calls > 0 && grand_recorded > 0 {
            let pct = (total_llm_calls as f64 / grand_recorded as f64) * 100.0;
            let saved = 100.0 - pct;
            if saved > 0.1 {
                print!(". The LLM was only needed for \x1b[1;33m{}\x1b[0m pivotal moments (saving \x1b[1;36m{:.1}%\x1b[0m in API calls)", total_llm_calls, saved);
            } else {
                print!(". The LLM analyzed \x1b[1;33m{}\x1b[0m pivotal moments", total_llm_calls);
            }
        }
        println!();
    } else {
        print!(
            "Replay complete: {} turns indexed locally",
            grand_recorded
        );
        if total_llm_calls > 0 && grand_recorded > 0 {
            let pct = (total_llm_calls as f64 / grand_recorded as f64) * 100.0;
            let saved = 100.0 - pct;
            if saved > 0.1 {
                print!(". The LLM was only needed for {} pivotal moments (saving {:.1}% in API calls)", total_llm_calls, saved);
            } else {
                print!(". The LLM analyzed {} pivotal moments", total_llm_calls);
            }
        }
        println!();
    }

    // Always ingest git history after replaying agent sessions — zero extra cost,
    // deduplicates on hash, and makes `unlost brief` significantly more useful.
    if let Some(ref root) = repo_root {
        let embedder = crate::embed::load_embedder(
            &embed_model,
            embed_cache_dir.as_deref().map(std::path::PathBuf::from),
            false,
        )
        .await;
        if let Ok(embedder) = embedder {
            let _ = crate::git::ingest_git_commits(&ws, root, &embedder, 500, use_color).await;
        }
    }

    Ok(())
}

fn clear_replay_data(ws: &crate::WorkspacePaths, _kind: &str, _session_id: Option<&str>) -> anyhow::Result<()> {
    println!("Clearing existing replay data for workspace: {}", ws.id);
    
    // 1. Clear LanceDB
    if ws.db_dir.exists() {
        std::fs::remove_dir_all(&ws.db_dir)?;
    }
    
    // 2. Clear JSONL
    if ws.capsules_jsonl.exists() {
        std::fs::remove_file(&ws.capsules_jsonl)?;
    }
    
    // 3. Clear replayed trackers
    let opencode_dir = crate::workspace::unlost_workspace_dir(&ws.id).join("opencode");
    if opencode_dir.exists() {
        std::fs::remove_dir_all(&opencode_dir)?;
    }
    
    Ok(())
}
