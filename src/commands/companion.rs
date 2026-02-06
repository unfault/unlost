//! Companion mode: stdio JSON-RPC for OpenCode plugins.
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

use crate::embed::Embedder;
use crate::emotion::{apply_context_heuristics, map_go_emotions, EmotionConfig, EmotionModel};
use crate::governor::evaluate_friction;
use crate::recording::{looks_like_commit_or_pr, ChunkInput, FlushJob, WorkspaceChunker};
use crate::storage::{ensure_capsules_table, insert_capsule_row, scan_capsules_lancedb};
use crate::workspace::get_or_create_workspace_paths;
use crate::IntentCapsule;
use lancedb::connection::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Companion state held across requests.
struct CompanionState {
    emotion_model: Option<EmotionModel>,
    embedder: Option<Embedder>,
    db_cache: HashMap<String, Connection>,
    chunker: Option<WorkspaceChunker>,
    flush_rx: Option<kanal::AsyncReceiver<FlushJob>>,
}

impl CompanionState {
    fn new() -> Self {
        Self {
            emotion_model: None,
            embedder: None,
            db_cache: HashMap::new(),
            chunker: None,
            flush_rx: None,
        }
    }

    async fn ensure_emotion_model(&mut self) -> anyhow::Result<&mut EmotionModel> {
        if self.emotion_model.is_none() {
            let model = EmotionModel::load(EmotionConfig::default()).await?;
            self.emotion_model = Some(model);
        }
        Ok(self.emotion_model.as_mut().unwrap())
    }

    async fn ensure_embedder(&mut self, embed_model: &str, embed_cache_dir: Option<&str>) -> anyhow::Result<Embedder> {
        if self.embedder.is_none() {
            let cache_path = embed_cache_dir.map(std::path::PathBuf::from);
            let embedder = crate::embed::load_embedder(embed_model, cache_path, false).await?;
            self.embedder = Some(embedder.clone());
            Ok(embedder)
        } else {
            Ok(self.embedder.clone().unwrap())
        }
    }

    fn ensure_chunker(&mut self) -> WorkspaceChunker {
        if self.chunker.is_none() {
            let (flush_tx, flush_rx) = kanal::bounded_async(64);
            self.chunker = Some(WorkspaceChunker::new(flush_tx));
            self.flush_rx = Some(flush_rx);
        }
        self.chunker.clone().unwrap()
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
    let history = match scan_capsules_lancedb(&ws, 5, None).await {
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

    // Classify user emotion
    let _user_emotion = if !params.text.is_empty() {
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

    let note = evaluate_friction(&current, &history);
    Response::Check(CheckResponse { note, error: None })
}

async fn handle_record(
    state: &mut CompanionState,
    params: RecordParams,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> Response {
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

    let chunker = state.ensure_chunker();
    let item = ChunkInput {
        conn_id,
        upstream_host: "companion".to_string(),
        request_path: "/companion".to_string(),
        http_status: 200,
        exchange_text,
        commit_mentioned,
    };

    chunker.ingest(ws.id.clone(), item).await;

    // Collect pending flush jobs
    let mut jobs: Vec<FlushJob> = Vec::new();
    if let Some(rx) = state.flush_rx.as_ref() {
        while let Ok(Some(job)) = rx.try_recv() {
            jobs.push(job);
        }
    }

    // Also flush idle buffers
    chunker.flush_idle().await;

    // Collect any jobs that were just flushed
    if let Some(rx) = state.flush_rx.as_ref() {
        while let Ok(Some(job)) = rx.try_recv() {
            jobs.push(job);
        }
    }

    // Process all collected jobs
    for job in jobs {
        if let Err(e) = process_flush_job(state, job, embed_model, embed_cache_dir).await {
            tracing::warn!("flush job failed: {e}");
        }
    }

    Response::Record(RecordResponse { ok: true, error: None })
}

async fn process_flush_job(
    state: &mut CompanionState,
    job: FlushJob,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
) -> anyhow::Result<()> {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON only with fields: {category, intent, decision, rationale, next_steps (array), symbols (array)}.\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.";

    // Extract user/assistant text for emotion classification
    let (user_text, assistant_text) = crate::emotion::extract_user_and_assistant_text(&job.input);

    // Classify emotions
    let user_emotion = if !user_text.trim().is_empty() {
        match state.ensure_emotion_model().await {
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
        match state.ensure_emotion_model().await {
            Ok(model) => match model.classify_one(&assistant_text) {
                Ok((raw, score)) => Some(map_go_emotions(&raw, score)),
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    // Extract capsule using LLM
    let capsule = crate::llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await?;

    // Get workspace paths from the job
    let ws_dir = crate::unlost_workspace_dir(&job.workspace_id);
    let ws = crate::WorkspacePaths {
        id: job.workspace_id.clone(),
        db_dir: ws_dir.join("lancedb"),
        capsules_jsonl: ws_dir.join("capsules.jsonl"),
    };

    // Append to JSONL
    append_capsule_jsonl(&ws.capsules_jsonl, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, &capsule)?;

    // Insert into LanceDB
    let embedder = state.ensure_embedder(embed_model, embed_cache_dir).await?;
    let db = state.db_for(&job.workspace_id, &ws.db_dir).await?;

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
    let record = serde_json::json!({
        "ts_ms": ts_ms,
        "conn_id": conn_id,
        "exchange_seq": exchange_seq,
        "source": meta.source,
        "upstream_host": meta.upstream_host,
        "request_path": meta.request_path,
        "http_status": meta.http_status,
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
    let mut state = CompanionState::new();

    let embed_cache_ref = embed_cache_dir.as_deref();

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
                handle_record(&mut state, params, &embed_model, embed_cache_ref).await
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_params_parsing() {
        let json = r#"{"text": "hello", "directory": "/tmp"}"#;
        let params: CheckParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.text, "hello");
        assert_eq!(params.directory, "/tmp");
    }

    #[test]
    fn test_record_params_parsing() {
        let json = r#"{"user_text": "hello", "assistant_text": "hi there", "directory": "/tmp"}"#;
        let params: RecordParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.user_text, "hello");
        assert_eq!(params.assistant_text, "hi there");
        assert_eq!(params.directory, "/tmp");
    }

    #[test]
    fn test_response_serialization() {
        let resp = Response::Check(CheckResponse {
            note: Some("warning".to_string()),
            error: None,
        });
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""note":"warning""#));

        let resp = Response::Record(RecordResponse {
            ok: true,
            error: None,
        });
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true"#));
    }
}
