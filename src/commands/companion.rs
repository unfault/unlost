//! Companion mode for unlost.
//!
//! Runs as a child process of an agent plugin. Reads JSON requests from stdin,
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
//! The plugin spawns `unlost companion` on init and communicates over stdio.
//!
//! IMPORTANT: `record` returns immediately after enqueueing; heavy work (LLM extraction,
//! embedding, LanceDB insert) happens in a background task so we never block the agent.

use crate::embed::Embedder;
use crate::emotion::{apply_context_heuristics, map_go_emotions, EmotionConfig, EmotionModel};
use crate::governor::evaluate_friction;
use crate::recording::{looks_like_commit_or_pr, ChunkInput, FlushJob, WorkspaceChunker};
use crate::storage::{ensure_capsules_table, insert_capsule_row};
use crate::workspace::get_or_create_workspace_paths;
use crate::IntentCapsule;
use lancedb::connection::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

static CONN_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
struct Request {
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CheckParams {
    /// User's last message text
    #[serde(default)]
    text: String,
    /// Workspace directory (absolute path)
    #[serde(default)]
    directory: String,
}

#[derive(Debug, Deserialize)]
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

/// Shared state for background processing (accessed from the worker task).
struct BackgroundState {
    emotion_model: Option<EmotionModel>,
    embedder: Option<Embedder>,
    db_cache: HashMap<String, Connection>,
    embed_model: String,
    embed_cache_dir: Option<String>,
}

impl BackgroundState {
    fn new(embed_model: String, embed_cache_dir: Option<String>) -> Self {
        Self {
            emotion_model: None,
            embedder: None,
            db_cache: HashMap::new(),
            embed_model,
            embed_cache_dir,
        }
    }

    async fn ensure_emotion_model(&mut self) -> anyhow::Result<&mut EmotionModel> {
        if self.emotion_model.is_none() {
            let model = EmotionModel::load(EmotionConfig::default()).await?;
            self.emotion_model = Some(model);
        }
        Ok(self.emotion_model.as_mut().unwrap())
    }

    async fn ensure_embedder(&mut self) -> anyhow::Result<Embedder> {
        if self.embedder.is_none() {
            let cache_path = self.embed_cache_dir.as_deref().map(std::path::PathBuf::from);
            let embedder = crate::embed::load_embedder(&self.embed_model, cache_path, false).await?;
            self.embedder = Some(embedder.clone());
            Ok(embedder)
        } else {
            Ok(self.embedder.clone().unwrap())
        }
    }

    async fn db_for(&mut self, workspace_id: &str, db_dir: &std::path::Path) -> anyhow::Result<Connection> {
        if let Some(db) = self.db_cache.get(workspace_id) {
            return Ok(db.clone());
        }

        std::fs::create_dir_all(db_dir)?;
        let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
            .execute()
            .await?;
        let _ = ensure_capsules_table(&db).await?;

        self.db_cache.insert(workspace_id.to_string(), db.clone());
        Ok(db)
    }
}

/// Companion state held in the main request loop.
/// Heavy processing is offloaded to BackgroundState via channel.
struct CompanionState {
    /// For friction checks only (needs recent capsules).
    emotion_model: Option<EmotionModel>,
    /// Chunker that buffers exchanges and emits FlushJobs.
    chunker: WorkspaceChunker,
}

impl CompanionState {
    fn new(job_tx: kanal::AsyncSender<FlushJob>) -> Self {
        // The chunker sends jobs to the channel.
        let chunker = WorkspaceChunker::new(job_tx);
        Self {
            emotion_model: None,
            chunker,
        }
    }

    async fn ensure_emotion_model(&mut self) -> anyhow::Result<&mut EmotionModel> {
        if self.emotion_model.is_none() {
            let model = EmotionModel::load(EmotionConfig::default()).await?;
            self.emotion_model = Some(model);
        }
        Ok(self.emotion_model.as_mut().unwrap())
    }
}

async fn handle_check(state: &mut CompanionState, params: CheckParams) -> Response {
    if params.directory.is_empty() {
        return Response::Check(CheckResponse {
            note: None,
            error: Some("missing directory param".to_string()),
        });
    }

    let dir_path = std::path::Path::new(&params.directory);
    if !dir_path.exists() {
        return Response::Check(CheckResponse {
            note: None,
            error: Some(format!("directory does not exist: {}", params.directory)),
        });
    }

    // Get workspace paths
    let ws = match get_or_create_workspace_paths(dir_path) {
        Ok(ws) => ws,
        Err(e) => {
            return Response::Check(CheckResponse {
                note: None,
                error: Some(format!("workspace error: {e}")),
            });
        }
    };

    // Load recent capsules
    let history = match crate::storage::scan_capsules_lancedb(&ws, 5, None, None, None, None, None).await {
        Ok(h) => h,
        Err(e) => {
            tracing::debug!("scan_capsules_lancedb failed: {e}");
            vec![]
        }
    };

    // If we have no history, nothing to check friction against
    if history.is_empty() {
        return Response::Check(CheckResponse { note: None, error: None });
    }

    // Extract symbols from the text
    let symbols = crate::net::extract_symbols_from_text(&params.text);

    // Classify user emotion (used as a signal for friction warning injection).
    let user_emotion = if !params.text.is_empty() {
        match state.ensure_emotion_model().await {
            Ok(model) => match model.classify_one(&params.text) {
                Ok((raw_label, score)) => {
                    let model_meta = map_go_emotions(&raw_label, score);
                    Some(apply_context_heuristics(&params.text, model_meta))
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    // Create current capsule for friction evaluation
    let current = IntentCapsule {
        category: String::new(),
        intent: params.text.clone(),
        decision: String::new(),
        rationale: String::new(),
        next_steps: vec![],
        symbols,
    };

    let note = evaluate_friction(&current, user_emotion.as_ref(), &history);
    Response::Check(CheckResponse { note, error: None })
}

/// Enqueue the exchange for background processing. Returns immediately.
async fn handle_record(state: &mut CompanionState, params: RecordParams) -> Response {
    if params.directory.is_empty() {
        return Response::Record(RecordResponse {
            ok: false,
            error: Some("missing directory param".to_string()),
        });
    }

    let dir_path = std::path::Path::new(&params.directory);
    if !dir_path.exists() {
        return Response::Record(RecordResponse {
            ok: false,
            error: Some(format!("directory does not exist: {}", params.directory)),
        });
    }

    // Get workspace paths
    let ws = match get_or_create_workspace_paths(dir_path) {
        Ok(ws) => ws,
        Err(e) => {
            return Response::Record(RecordResponse {
                ok: false,
                error: Some(format!("workspace error: {e}")),
            });
        }
    };

    // Skip if both texts are empty
    if params.user_text.trim().is_empty() && params.assistant_text.trim().is_empty() {
        return Response::Record(RecordResponse { ok: true, error: None });
    }

    // Build exchange text in the format expected by the chunker
    let mut exchange_text = String::new();
    if !params.user_text.trim().is_empty() {
        exchange_text.push_str("User:\n");
        exchange_text.push_str(params.user_text.trim());
        exchange_text.push_str("\n\n");
    }
    if !params.assistant_text.trim().is_empty() {
        exchange_text.push_str("Assistant:\n");
        exchange_text.push_str(params.assistant_text.trim());
    }

    let commit_mentioned = looks_like_commit_or_pr(&exchange_text);
    let conn_id = CONN_SEQ.fetch_add(1, Ordering::Relaxed);

    let usage = params.usage.map(|u| crate::types::UsageMeta {
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

    let item = ChunkInput {
        conn_id,
        upstream_host: "companion".to_string(),
        request_path: "/companion".to_string(),
        http_status: 200,
        exchange_text,
        commit_mentioned,
        agent_session_id: params.agent_session_id.clone(),
        usage,
    };

    // Ingest into chunker (may or may not produce a flush job depending on boundaries).
    state.chunker.ingest(ws.id.clone(), item).await;

    // Force-flush this workspace so we don't lose data if the companion exits soon.
    // This sends a FlushJob to the background worker via the channel.
    state.chunker.flush_workspace(&ws.id).await;

    // Return immediately; the background worker will process the job asynchronously.
    Response::Record(RecordResponse { ok: true, error: None })
}

/// Background worker that processes FlushJobs without blocking the request loop.
async fn background_worker(rx: kanal::AsyncReceiver<FlushJob>, state: Arc<Mutex<BackgroundState>>) {
    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break, // channel closed
        };

        if let Err(e) = process_flush_job(&state, job).await {
            tracing::warn!("background flush job failed: {e}");
        }
    }
}

async fn process_flush_job(state: &Arc<Mutex<BackgroundState>>, job: FlushJob) -> anyhow::Result<()> {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON only with fields: {category, intent, decision, rationale, next_steps (array), symbols (array)}.\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.";

    // Extract user/assistant text for emotion classification
    let (user_text, assistant_text) = crate::emotion::extract_user_and_assistant_text(&job.input);

    // Classify emotions (requires mutable access to state)
    let user_emotion = if !user_text.trim().is_empty() {
        let mut st = state.lock().await;
        match st.ensure_emotion_model().await {
            Ok(model) => match model.classify_one(&user_text) {
                Ok((raw, score)) => {
                    let meta = map_go_emotions(&raw, score);
                    Some(apply_context_heuristics(&user_text, meta))
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    let assistant_emotion = if !assistant_text.trim().is_empty() {
        let mut st = state.lock().await;
        match st.ensure_emotion_model().await {
            Ok(model) => match model.classify_one(&assistant_text) {
                Ok((raw, score)) => Some(map_go_emotions(&raw, score)),
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    // Extract capsule using LLM (this is the expensive network call)
    let capsule = crate::llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await?;

    // Get workspace paths from the job
    let ws_dir = crate::unlost_workspace_dir(&job.workspace_id);
    let ws = crate::WorkspacePaths {
        id: job.workspace_id.clone(),
        db_dir: ws_dir.join("lancedb"),
        capsules_jsonl: ws_dir.join("capsules.jsonl"),
    };

    // Append to JSONL (cheap, local)
    append_capsule_jsonl(&ws.capsules_jsonl, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, &capsule)?;

    // Insert into LanceDB
    let mut st = state.lock().await;
    let embedder = st.ensure_embedder().await?;
    let db = st.db_for(&job.workspace_id, &ws.db_dir).await?;
    drop(st); // release lock before the potentially slow insert

    insert_capsule_row(
        &db,
        &embedder,
        job.conn_id,
        job.exchange_seq,
        job.ts_ms,
        &job.meta,
        user_emotion.as_ref(),
        assistant_emotion.as_ref(),
        &capsule,
    )
    .await?;

    tracing::info!(
        workspace_id = %job.workspace_id,
        exchange_seq = job.exchange_seq,
        "recorded capsule"
    );

    Ok(())
}

fn append_capsule_jsonl(
    path: &std::path::Path,
    ts_ms: i64,
    conn_id: u64,
    exchange_seq: u64,
    meta: &crate::ResponseMeta,
    capsule: &IntentCapsule,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let usage = meta.usage.as_ref().map(|u| {
        serde_json::json!({
            "provider_id": u.provider_id,
            "model_id": u.model_id,
            "cost": u.cost,
            "tokens": {
                "input": u.tokens_input,
                "output": u.tokens_output,
                "reasoning": u.tokens_reasoning,
                "cache": {
                    "read": u.tokens_cache_read,
                    "write": u.tokens_cache_write,
                }
            }
        })
    });

    let record = serde_json::json!({
        "ts_ms": ts_ms,
        "conn_id": conn_id,
        "exchange_seq": exchange_seq,
        "source": meta.source,
        "upstream_host": meta.upstream_host,
        "request_path": meta.request_path,
        "http_status": meta.http_status,
        "agent_session_id": meta.agent_session_id,
        "usage": usage,
        "capsule": capsule,
    });
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{}\n", record).as_bytes())?;
    Ok(())
}

pub(crate) async fn run(embed_model: String, embed_cache_dir: Option<String>) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    // Create an unbounded channel so we never block on send.
    // We accept potential memory growth under extreme load rather than blocking the agent.
    let (job_tx, job_rx) = kanal::unbounded_async::<FlushJob>();

    let mut state = CompanionState::new(job_tx);
    let bg_state = Arc::new(Mutex::new(BackgroundState::new(embed_model, embed_cache_dir)));

    // Spawn background worker
    let bg_state_clone = bg_state.clone();
    tokio::spawn(async move {
        background_worker(job_rx, bg_state_clone).await;
    });

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
                handle_check(&mut state, params).await
            }
            "record" => {
                let params: RecordParams = serde_json::from_value(req.params).unwrap_or_default();
                handle_record(&mut state, params).await
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

impl Default for CheckParams {
    fn default() -> Self {
        Self {
            text: String::new(),
            directory: String::new(),
        }
    }
}

impl Default for RecordParams {
    fn default() -> Self {
        Self {
            user_text: String::new(),
            assistant_text: String::new(),
            directory: String::new(),
            agent_session_id: None,
            usage: None,
        }
    }
}
