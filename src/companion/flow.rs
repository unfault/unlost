//! Core flow for agent integrations.
//!
//! Provides `CheckEvent` and `RecordTurnEvent` as the internal event model,
//! plus `Flow` which handles all business logic: friction checking, chunking,
//! background flush jobs, LLM extraction, embedding, and LanceDB insertion.
//!
//! Shims (e.g., `opencode_stdio`) translate external protocols into these events
//! and call the flow methods.

use crate::IntentCapsule;
use crate::embed::Embedder;
use crate::emotion::{EmotionConfig, EmotionModel, apply_context_heuristics, map_go_emotions};
use crate::governor::{
    evaluate_decision_conflict, evaluate_failure_modes, evaluate_friction,
    evaluate_stateless_friction,
};
use crate::recording::{ChunkInput, FlushJob, WorkspaceChunker, looks_like_commit_or_pr};
use crate::storage::{ensure_capsules_table, insert_capsule_row};
use crate::types::UsageMeta;
use crate::workspace::get_or_create_workspace_paths;
use lancedb::connection::Connection;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

static CONN_SEQ: AtomicU64 = AtomicU64::new(1);

/// Identifies which agent platform originated an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentKind {
    OpenCode,
    Claude,
}

impl AgentKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            AgentKind::OpenCode => "opencode",
            AgentKind::Claude => "claude",
        }
    }
}

/// Event for checking friction before an LLM call.
#[derive(Debug)]
pub(crate) struct CheckEvent {
    /// Workspace directory (absolute path)
    pub directory: String,
    /// User's message text
    pub text: String,
    /// Which agent platform this came from
    #[allow(dead_code)]
    pub agent_kind: AgentKind,
    /// Optional session ID for grouping
    #[allow(dead_code)]
    pub agent_session_id: Option<String>,
}

/// Result of a check operation.
#[derive(Debug)]
pub(crate) struct CheckResult {
    /// Warning note to inject, or None if no friction detected
    pub note: Option<String>,
    /// Error message if something went wrong (note may still be None)
    pub error: Option<String>,
}

/// Token usage metadata for recording.
#[derive(Debug, Clone, Default)]
pub(crate) struct UsageEvent {
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub cost: Option<f64>,
    pub tokens_input: Option<i64>,
    pub tokens_output: Option<i64>,
    pub tokens_reasoning: Option<i64>,
    pub tokens_cache_read: Option<i64>,
    pub tokens_cache_write: Option<i64>,
}

impl From<UsageEvent> for UsageMeta {
    fn from(u: UsageEvent) -> Self {
        UsageMeta {
            provider_id: u.provider_id,
            model_id: u.model_id,
            cost: u.cost,
            tokens_input: u.tokens_input,
            tokens_output: u.tokens_output,
            tokens_reasoning: u.tokens_reasoning,
            tokens_cache_read: u.tokens_cache_read,
            tokens_cache_write: u.tokens_cache_write,
        }
    }
}

/// Event for recording a conversation turn after an LLM call.
#[derive(Debug)]
pub(crate) struct RecordTurnEvent {
    /// Workspace directory (absolute path)
    pub directory: String,
    /// User's message text
    pub user_text: String,
    /// Assistant's response text
    pub assistant_text: String,
    /// Best-effort list of touched paths (workspace-relative). Optional.
    pub touched_paths: Vec<String>,
    /// Which agent platform this came from
    pub agent_kind: AgentKind,
    /// Optional session ID for grouping conversations
    pub agent_session_id: Option<String>,
    /// Optional usage metrics
    pub usage: Option<UsageEvent>,
}

/// Result of a record operation.
#[derive(Debug)]
pub(crate) struct RecordResult {
    /// Whether the operation succeeded
    pub ok: bool,
    /// Error message if something went wrong
    pub error: Option<String>,
}

/// Configuration for the flow.
pub(crate) struct FlowConfig {
    pub embed_model: String,
    pub embed_cache_dir: Option<String>,
    pub no_llm: bool,
}

/// Shared state for background processing (accessed from the worker task).
struct BackgroundState {
    emotion_model: Option<EmotionModel>,
    embedder: Option<Embedder>,
    db_cache: HashMap<String, Connection>,
    embed_model: String,
    embed_cache_dir: Option<String>,
    no_llm: bool,
}

impl BackgroundState {
    fn new(embed_model: String, embed_cache_dir: Option<String>, no_llm: bool) -> Self {
        Self {
            emotion_model: None,
            embedder: None,
            db_cache: HashMap::new(),
            embed_model,
            embed_cache_dir,
            no_llm,
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
            let cache_path = self
                .embed_cache_dir
                .as_deref()
                .map(std::path::PathBuf::from);
            let embedder =
                crate::embed::load_embedder(&self.embed_model, cache_path, false).await?;
            self.embedder = Some(embedder.clone());
            Ok(embedder)
        } else {
            Ok(self.embedder.clone().unwrap())
        }
    }

    async fn db_for(
        &mut self,
        workspace_id: &str,
        db_dir: &std::path::Path,
    ) -> anyhow::Result<Connection> {
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

/// Flow state held in the main request loop.
/// Heavy processing is offloaded to BackgroundState via channel.
struct FlowState {
    /// For friction checks only (needs recent capsules).
    emotion_model: Option<EmotionModel>,
    /// Chunker that buffers exchanges and emits FlushJobs.
    chunker: WorkspaceChunker,
    /// Proactive friction regulation controllers, keyed by workspace ID.
    controllers: HashMap<String, crate::governor::TrajectoryController>,
}

impl FlowState {
    fn new(job_tx: kanal::AsyncSender<FlushJob>) -> Self {
        let chunker = WorkspaceChunker::new(job_tx);
        Self {
            emotion_model: None,
            chunker,
            controllers: HashMap::new(),
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

/// The main flow handler for agent integrations.
///
/// Owns the check + record_turn pipeline, chunking, and background worker.
pub(crate) struct Flow {
    state: FlowState,
    #[allow(dead_code)]
    bg_state: Arc<Mutex<BackgroundState>>,
    /// Sender half kept to control shutdown - dropping closes the channel
    job_tx: kanal::AsyncSender<FlushJob>,
    /// Handle to the background worker task
    worker_handle: tokio::task::JoinHandle<()>,
}

impl Flow {
    /// Create a new flow with the given configuration.
    /// Spawns a background worker for processing flush jobs.
    pub(crate) fn new(config: FlowConfig) -> Self {
        let (job_tx, job_rx) = kanal::unbounded_async::<FlushJob>();
        let state = FlowState::new(job_tx.clone());
        let bg_state = Arc::new(Mutex::new(BackgroundState::new(
            config.embed_model,
            config.embed_cache_dir,
            config.no_llm,
        )));

        // Spawn background worker
        let bg_state_clone = bg_state.clone();
        let worker_handle = tokio::spawn(async move {
            background_worker(job_rx, bg_state_clone).await;
        });

        Self {
            state,
            bg_state,
            job_tx,
            worker_handle,
        }
    }

    /// Drain pending jobs and shut down the background worker.
    ///
    /// This closes the job channel (signaling the worker to exit after processing
    /// remaining jobs) and waits for the worker to finish.
    ///
    /// Call this before dropping the Flow if you need to ensure all enqueued
    /// jobs are processed (e.g., in short-lived processes like hooks).
    pub(crate) async fn drain(self) {
        // Drop our sender to close the channel - worker will exit after draining
        drop(self.job_tx);
        // Also drop the chunker's sender
        drop(self.state);
        // Wait for worker to finish processing remaining jobs
        let _ = self.worker_handle.await;
    }

    /// Check for friction before an LLM call.
    ///
    /// Returns a note to inject if friction is detected, or None otherwise.
    pub(crate) async fn check(&mut self, event: CheckEvent) -> CheckResult {
        tracing::info!(
            directory = %event.directory,
            text_len = event.text.len(),
            session = event.agent_session_id.as_deref().unwrap_or("-"),
            "check called"
        );

        if event.directory.is_empty() {
            return CheckResult {
                note: None,
                error: Some("missing directory param".to_string()),
            };
        }

        let dir_path = std::path::Path::new(&event.directory);
        if !dir_path.exists() {
            return CheckResult {
                note: None,
                error: Some(format!("directory does not exist: {}", event.directory)),
            };
        }

        // Get workspace paths
        let ws = match get_or_create_workspace_paths(dir_path) {
            Ok(ws) => ws,
            Err(e) => {
                return CheckResult {
                    note: None,
                    error: Some(format!("workspace error: {e}")),
                };
            }
        };

        // Decision/constraint intervention ("I told you NOT to")
        // Runs before friction checks so it can interrupt immediately.
        if !event.text.trim().is_empty() {
            let embedder = {
                let mut st = self.bg_state.lock().await;
                st.ensure_embedder().await.ok()
            };

            if let Some(embedder) = embedder {
                // Semantic search for relevant prior decisions.
                let matches = crate::storage::query_capsules_lancedb(
                    &event.text,
                    8,
                    None,
                    None,
                    None,
                    None,
                    None,
                    embedder,
                    &ws,
                )
                .await;

                if let Ok(hits) = matches {
                    if let Some(note) = evaluate_decision_conflict(&event.text, &hits) {
                        tracing::info!(
                            workspace = %ws.id,
                            session = event.agent_session_id.as_deref().unwrap_or("-"),
                            "decision conflict intervention will be injected"
                        );
                        return CheckResult {
                            note: Some(note),
                            error: None,
                        };
                    }
                }
            }
        }

        // Load recent capsules (most recent first)
        let history = match crate::storage::scan_capsules_lancedb_recent(
            &ws, 5, None, None, None, None, None,
        )
        .await
        {
            Ok(h) => h,
            Err(e) => {
                tracing::debug!("scan_capsules_lancedb_recent failed: {e}");
                vec![]
            }
        };

        // Extract symbols from the text
        let symbols = crate::net::extract_symbols_from_text(&event.text);

        // Classify user emotion (used as a signal for friction warning injection).
        let user_emotion = if !event.text.is_empty() {
            match self.state.ensure_emotion_model().await {
                Ok(model) => match model.classify_one(&event.text) {
                    Ok((raw_label, score)) => {
                        let model_meta = map_go_emotions(&raw_label, score);
                        Some(apply_context_heuristics(&event.text, model_meta))
                    }
                    Err(_) => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };

        // If we have no history yet, we can't run repetition-based heuristics.
        // Still inject a small stateless note when the user is clearly upset.
        if history.is_empty() {
            return CheckResult {
                note: evaluate_stateless_friction(user_emotion.as_ref()),
                error: None,
            };
        }

        // Detect failure modes from keywords in the current message
        let failure_mode = crate::governor::detect_failure_keywords(&event.text).unwrap_or(crate::types::FailureMode::None);

        // Create current capsule for friction evaluation
        let current = IntentCapsule {
            category: String::new(),
            intent: event.text.clone(),
            decision: String::new(),
            rationale: String::new(),
            next_steps: vec![],
            symbols,
            failure_mode,
            failure_signals: None,
        };

        // NEW: Trajectory-based proactive friction regulation
        let (trajectory_state, trajectory_note) = {
            let controller = self.state.controllers.entry(ws.id.clone()).or_default();
            controller.update(
                &ws.id,
                &current,
                user_emotion.as_ref(),
                &history,
                crate::workspace::now_ms(),
            )
        };

        if let Some(note) = trajectory_note {
            tracing::info!(
                workspace = %ws.id,
                session = event.agent_session_id.as_deref().unwrap_or("-"),
                state = ?trajectory_state,
                "trajectory controller returned proactive warning"
            );
            return CheckResult { note: Some(note), error: None };
        }

        // Fallback to legacy friction check (emotion + symbol repetition)
        let note = evaluate_friction(&current, user_emotion.as_ref(), &history);

        if note.is_some() {
            tracing::info!(
                workspace = %ws.id,
                session = event.agent_session_id.as_deref().unwrap_or("-"),
                history_size = history.len(),
                "friction check returned warning"
            );
            return CheckResult { note, error: None };
        }

        // If no friction, check for LLM-detected failure modes (drift, false_progress, rediscovery)
        let note = evaluate_failure_modes(&history);

        if note.is_some() {
            tracing::info!(
                workspace = %ws.id,
                session = event.agent_session_id.as_deref().unwrap_or("-"),
                history_size = history.len(),
                "failure mode check returned warning"
            );
        }

        CheckResult { note, error: None }
    }

    /// Record a conversation turn after an LLM call.
    ///
    /// Enqueues the exchange for background processing and returns immediately.
    pub(crate) async fn record_turn(&mut self, event: RecordTurnEvent) -> RecordResult {
        if event.directory.is_empty() {
            return RecordResult {
                ok: false,
                error: Some("missing directory param".to_string()),
            };
        }

        let dir_path = std::path::Path::new(&event.directory);
        if !dir_path.exists() {
            return RecordResult {
                ok: false,
                error: Some(format!("directory does not exist: {}", event.directory)),
            };
        }

        // Get workspace paths
        let ws = match get_or_create_workspace_paths(dir_path) {
            Ok(ws) => ws,
            Err(e) => {
                return RecordResult {
                    ok: false,
                    error: Some(format!("workspace error: {e}")),
                };
            }
        };

        // Skip if both texts are empty
        if event.user_text.trim().is_empty() && event.assistant_text.trim().is_empty() {
            return RecordResult {
                ok: true,
                error: None,
            };
        }

        // Build exchange text in the format expected by the chunker
        let mut exchange_text = String::new();

        if !event.touched_paths.is_empty() {
            exchange_text.push_str("Touched paths:\n");
            for p in event.touched_paths.iter().take(32) {
                let p = p.trim();
                if p.is_empty() {
                    continue;
                }
                exchange_text.push_str(p);
                exchange_text.push('\n');
            }
            exchange_text.push('\n');
        }
        if !event.user_text.trim().is_empty() {
            exchange_text.push_str("User:\n");
            exchange_text.push_str(event.user_text.trim());
            exchange_text.push_str("\n\n");
        }
        if !event.assistant_text.trim().is_empty() {
            exchange_text.push_str("Assistant:\n");
            exchange_text.push_str(event.assistant_text.trim());
        }

        let commit_mentioned = looks_like_commit_or_pr(&exchange_text);
        let conn_id = CONN_SEQ.fetch_add(1, Ordering::Relaxed);

        let usage = event.usage.map(|u| {
            tracing::debug!(
                "record_turn usage: provider={:?} model={:?} cost={:?} tokens_input={:?} tokens_output={:?}",
                u.provider_id,
                u.model_id,
                u.cost,
                u.tokens_input,
                u.tokens_output,
            );
            u.into()
        });

        if usage.is_none() {
            tracing::debug!("record_turn has no usage data");
        }

        let upstream_host = format!("shim-{}", event.agent_kind.as_str());
        let request_path = format!("/{}/record", event.agent_kind.as_str());

        let item = ChunkInput {
            conn_id,
            upstream_host,
            request_path,
            http_status: 200,
            exchange_text,
            commit_mentioned,
            agent_session_id: event.agent_session_id,
            usage,
        };

        // Ingest into chunker (may or may not produce a flush job depending on boundaries).
        self.state.chunker.ingest(ws.id.clone(), item).await;

        // Force-flush this workspace so we don't lose data if the process exits soon.
        // This sends a FlushJob to the background worker via the channel.
        self.state.chunker.flush_workspace(&ws.id).await;

        // Return immediately; the background worker will process the job asynchronously.
        RecordResult {
            ok: true,
            error: None,
        }
    }
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

async fn process_flush_job(
    state: &Arc<Mutex<BackgroundState>>,
    job: FlushJob,
) -> anyhow::Result<()> {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
 Return JSON with fields: {category, intent, decision, rationale, next_steps (array), symbols (array), failure_mode, failure_signals}.\n\
 \n\
 Rules:\n\
 - Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
 - Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
 - Keep each field concise; next_steps max 3.\n\
 - symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned. If a 'Touched paths:' section is present, include those paths in symbols.\n\
 \n\
 Failure mode detection - set failure_mode to one of:\n\
- none: No failure mode detected, conversation is productive.\n\
- drift: Agent has wrong mental model of the codebase. Signs: user corrects factual errors about code structure, APIs, or file locations; agent references non-existent symbols/paths.\n\
- rediscovery: Same ground being covered again. Signs: user re-explains constraints or decisions from earlier; \"we already discussed this\"; \"remember when we decided\".\n\
- decision_conflict: Agent proposes or starts an approach that conflicts with an established project decision/constraint. Signs: user says \"we decided against that\", \"I told you not to\", \"that's not how we do it\"; a prior decision capsule forbids the approach.\n\
- retry_spiral: Agent stuck in a loop. Signs: user frustration (\"same error\", \"you already tried that\", \"going in circles\"); same symbols appear repeatedly; agent apologizes then repeats similar approach.\n\
- false_progress: Agent claims done but isn't. Signs: user says \"that's still not working\", \"the error is still there\"; agent declared completion but user disputes it.\n\
- unbounded_horizon: Agent wandering off-task. Signs: \"while I'm here\" tangents; refactoring unrelated code; user redirects back to original task.\n\
\n\
Set failure_signals to a brief explanation (1 sentence) if failure_mode is not 'none', otherwise null.";

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

    // Extract capsule metadata locally (fast, zero cost)
    let symbols = crate::net::extract_symbols_from_text(&job.input);
    let failure_mode = crate::governor::detect_failure_keywords(&job.input).unwrap_or(crate::types::FailureMode::None);

    let no_llm = state.lock().await.no_llm;

    // Extract capsule using LLM (this is the expensive network call)
    let mut capsule = if !no_llm {
        match crate::llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(mut c) => {
                // Augment LLM capsule with local symbols if LLM missed any
                for s in symbols {
                    if !c.symbols.contains(&s) {
                        c.symbols.push(s);
                    }
                }
                c
            }
            Err(e) => {
                tracing::warn!("LLM extraction failed, falling back to heuristic: {e}");
                IntentCapsule {
                    category: "unknown".to_string(),
                    intent: user_text.lines().next().unwrap_or("").to_string(),
                    decision: assistant_text.lines().next().unwrap_or("").to_string(),
                    rationale: String::new(),
                    next_steps: vec![],
                    symbols,
                    failure_mode,
                    failure_signals: Some("Heuristic extraction (LLM failed)".to_string()),
                }
            }
        }
    } else {
        // Ghost Mode (Option B)
        IntentCapsule {
            category: "replay".to_string(),
            intent: user_text.lines().next().unwrap_or("").to_string(),
            decision: assistant_text.lines().next().unwrap_or("").to_string(),
            rationale: String::new(),
            next_steps: vec![],
            symbols,
            failure_mode,
            failure_signals: Some("Ghost extraction (no-LLM)".to_string()),
        }
    };

    crate::util::augment_capsule_symbols_from_input(&mut capsule, &job.input);

    // Log if a failure mode was detected by the LLM
    if capsule.failure_mode != crate::types::FailureMode::None {
        tracing::info!(
            workspace_id = %job.workspace_id,
            failure_mode = ?capsule.failure_mode,
            failure_signals = capsule.failure_signals.as_deref().unwrap_or("-"),
            category = %capsule.category,
            symbols = ?capsule.symbols,
            "LLM detected failure mode"
        );
    }

    // Get workspace paths from the job
    let ws_dir = crate::unlost_workspace_dir(&job.workspace_id);
    let ws = crate::WorkspacePaths {
        id: job.workspace_id.clone(),
        db_dir: ws_dir.join("lancedb"),
        capsules_jsonl: ws_dir.join("capsules.jsonl"),
        metrics_jsonl: ws_dir.join("metrics.jsonl"),
    };

    // Append to JSONL (cheap, local)
    append_capsule_jsonl(
        &ws.capsules_jsonl,
        job.ts_ms,
        job.conn_id,
        job.exchange_seq,
        &job.meta,
        &capsule,
    )?;

    let _ = crate::metrics::record_capsule_saved(
        &ws,
        job.ts_ms,
        job.conn_id,
        job.exchange_seq,
        &job.meta,
        user_emotion.as_ref(),
        assistant_emotion.as_ref(),
        &capsule,
    );

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
        Some(&job.input),
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
    use std::io::Write;

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
