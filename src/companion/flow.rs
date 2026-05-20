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
use crate::emotion::{
    EmotionConfig, EmotionModel, apply_context_heuristics, extract_user_and_assistant_text,
    map_go_emotions,
};
use crate::governor::{evaluate_failure_modes, evaluate_friction};
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
    Copilot,
    Cowork,
}

impl AgentKind {
    #[allow(dead_code)]
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            AgentKind::OpenCode => "opencode",
            AgentKind::Claude => "claude",
            AgentKind::Copilot => "copilot",
            AgentKind::Cowork => "cowork",
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
    /// Normalized outcomes from significant tool calls (build, test, publish, git, etc.).
    /// Each entry is a short fact: "succeeded" or "failed: <snippet>". Empty = none captured.
    pub tool_calls: Vec<crate::types::ToolCall>,
    /// Which agent platform this came from
    #[allow(dead_code)]
    pub agent_kind: AgentKind,
    /// Optional session ID for grouping conversations
    pub agent_session_id: Option<String>,
    /// Optional usage metrics
    pub usage: Option<UsageEvent>,
    /// Optional grounding info (e.g. verified git commits)
    pub grounding_note: Option<String>,
    /// Original message timestamp (ms since epoch). When set (replay path),
    /// this overrides the wall-clock time so capsules sort correctly.
    pub source_ts_ms: Option<i64>,
    /// Optional URI back to the source system of record for this turn.
    /// See `internal/SOURCE_POINTERS.md` for the scheme registry.
    pub source_pointer: Option<String>,
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
    pub extraction_mode: crate::types::ExtractionMode,
}

/// Shared state for background processing (accessed from the worker task).
struct BackgroundState {
    emotion_model: Option<EmotionModel>,
    embedder: Option<Embedder>,
    db_cache: HashMap<String, Connection>,
    embed_model: String,
    embed_cache_dir: Option<String>,
    extraction_mode: crate::types::ExtractionMode,
    pub(crate) llm_calls: Arc<std::sync::atomic::AtomicU64>,
    /// Most recent extracted decision per workspace, for embedding causal continuity.
    /// Updated after each successful capsule insertion. The next job for that workspace
    /// uses this as `prior_decision` in the embed text, even if the chunker's buffer
    /// was flushed before the worker finished.
    last_decisions: HashMap<String, String>,
    /// Cumulative turn count per agent session ID, used to compute context_freshness.
    session_turn_counts: HashMap<String, usize>,
}

impl BackgroundState {
    fn new(
        embed_model: String,
        embed_cache_dir: Option<String>,
        extraction_mode: crate::types::ExtractionMode,
    ) -> Self {
        Self {
            emotion_model: None,
            embedder: None,
            db_cache: HashMap::new(),
            embed_model,
            embed_cache_dir,
            extraction_mode,
            llm_calls: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_decisions: HashMap::new(),
            session_turn_counts: HashMap::new(),
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

pub(crate) struct Flow {
    state: FlowState,
    background_state: Arc<Mutex<BackgroundState>>,
    worker_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Flow {
    pub(crate) fn new(config: FlowConfig) -> Self {
        let (job_tx, job_rx) = kanal::bounded_async::<FlushJob>(64);
        let background_state = Arc::new(Mutex::new(BackgroundState::new(
            config.embed_model,
            config.embed_cache_dir,
            config.extraction_mode,
        )));

        // Start background worker
        let worker_handle = tokio::spawn(background_worker(job_rx, background_state.clone()));

        Self {
            state: FlowState::new(job_tx),
            background_state,
            worker_handle: Some(worker_handle),
        }
    }

    /// Flush all buffered exchanges and wait for the background worker to finish
    /// writing every capsule before returning. Safe to call only once; subsequent
    /// calls are no-ops.
    pub(crate) async fn drain(&mut self) {
        // Send any buffered turns as flush jobs.
        self.state.chunker.flush_all().await;
        // Close the sender so the background worker exits after draining the queue.
        self.state.chunker.close_sender();
        // Wait for the worker to finish processing all queued jobs.
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.await;
        }
    }

    pub(crate) async fn llm_calls(&self) -> u64 {
        self.background_state
            .lock()
            .await
            .llm_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Inspect a turn before the agent's LLM call.
    ///
    /// Mirrors [`Flow::record_turn`] (the post-call counterpart). Runs in the
    /// hot path; everything inside must be cheap. Today this does:
    ///
    /// 1. Local emotion classification on the user text.
    /// 2. A small chronological scan of recent capsules (last 5).
    /// 3. An ANN query for dormant candidates feeding the recurrence channel.
    /// 4. A pass through `TrajectoryController::update_with_candidates`, which
    ///    may emit a SYSTEM NOTE for an existing basin (drift / spec / loop /
    ///    anger) or a standalone resurfacing note from the recurrence channel.
    /// 5. Fallback heuristics (`evaluate_friction`, `evaluate_failure_modes`).
    ///
    /// See `internal/SOURCE_POINTERS.md` §Recurrence Channel for the
    /// resurfacing contract specifically.
    pub(crate) async fn check_turn(&mut self, event: CheckEvent) -> CheckResult {
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

        let ws = match get_or_create_workspace_paths(dir_path) {
            Ok(ws) => ws,
            Err(e) => {
                return CheckResult {
                    note: None,
                    error: Some(format!("workspace error: {e}")),
                };
            }
        };

        // Classify user emotion (fast, local)
        let user_emotion = match self.state.ensure_emotion_model().await {
            Ok(model) => match model.classify_one(&event.text) {
                Ok((raw, score)) => {
                    let meta = map_go_emotions(&raw, score);
                    Some(apply_context_heuristics(&event.text, meta))
                }
                Err(_) => None,
            },
            Err(_) => None,
        };

        // Query recent history — exclude git capsules: they are facts about merged code,
        // not conversation turns, and their thin signals (no usage, no emotion, failure_mode=None)
        // would skew friction scoring if they land in the last-5 window.
        let history: Vec<crate::CapsuleHit> =
            match crate::storage::scan_capsules_lancedb(&ws, 5, None, None, None, None, None).await
            {
                Ok(h) => h.into_iter().filter(|c| c.meta.source != "git").collect(),
                Err(_) => vec![],
            };

        // Dormant candidates for the recurrence channel: ANN against the current
        // user text, fanned out across all registered workspaces. We embed
        // best-effort and fail silently — friction must keep working even if
        // the embedder is cold or any workspace table is empty.
        // See `internal/SOURCE_POINTERS.md` §Recurrence Channel.
        let dormant_candidates: Vec<crate::CapsuleHit> = if event.text.trim().is_empty() {
            Vec::new()
        } else {
            let embedder = {
                let mut st = self.background_state.lock().await;
                st.ensure_embedder().await.ok()
            };
            match embedder {
                Some(emb) => {
                    crate::storage::query_capsules_cross_workspace(
                        &event.text,
                        emb,
                        &ws,
                        // per-workspace pull: enough to catch the strongest match
                        // even if a workspace has many near-duplicates.
                        4,
                        // total cap after merge: the controller only picks one,
                        // but more candidates means dormancy/structural tiebreaks
                        // have more to choose from.
                        12,
                    )
                    .await
                }
                None => Vec::new(),
            }
        };

        let symbols = crate::net::extract_symbols_from_text(&event.text);
        let user_symbols = symbols.clone();
        let failure_mode = crate::governor::detect_failure_keywords(&event.text)
            .unwrap_or(crate::types::FailureMode::None);

        // Create current capsule for friction evaluation
        let current = IntentCapsule {
            category: String::new(),
            intent: crate::util::strip_llm_boilerplate(event.text.clone()),
            decision: String::new(),
            rationale: String::new(),
            next_steps: vec![],
            symbols,
            user_symbols,
            failure_mode,
            failure_signals: None,
            extraction_mode: crate::types::ExtractionMode::None,
            questions: vec![],
        };

        // NEW: Trajectory-based proactive friction regulation
        let (update, final_session_id) = {
            let controller = self.state.controllers.entry(ws.id.clone()).or_default();

            // If event has a session ID, stick it. Otherwise try to use the last one.
            if let Some(ref sid) = event.agent_session_id {
                controller.last_agent_session_id = Some(sid.clone());
            }
            let sid = event
                .agent_session_id
                .clone()
                .or_else(|| controller.last_agent_session_id.clone());

            let update = controller.update_with_candidates(
                &ws.id,
                &current,
                user_emotion.as_ref(),
                &history,
                &dormant_candidates,
                sid.as_deref(),
                crate::workspace::now_ms(),
            );
            (update, sid)
        };

        if let Some(note) = update.note {
            tracing::info!(
                workspace = %ws.id,
                session = final_session_id.as_deref().unwrap_or("-"),
                state = ?update.state,
                cause = %update.cause,
                intensity = %update.intensity,
                "trajectory controller returned proactive warning"
            );

            // Resurfacing telemetry: log every time a recurrence match landed in the
            // emitted note, including modifier-mode appendings to other basins.
            if let Some(m) = update.recurrence_match.as_ref() {
                let mode = if update.cause == "resurfacing" {
                    "standalone"
                } else {
                    "modifier"
                };
                let age_days = ((crate::workspace::now_ms() - m.ts_ms) / (24 * 60 * 60 * 1000))
                    .max(0);
                let _ = crate::metrics::record_resurfacing_emitted(
                    &ws.id,
                    final_session_id.clone(),
                    &m.capsule_id,
                    m.similarity,
                    mode,
                    age_days,
                );
            }

            // Record the intervention metric
            let _ = crate::metrics::record_friction_warning_injected(
                &ws.id,
                0, // conn_id not available in CheckEvent
                final_session_id,
                current.symbols.clone(),
                user_emotion.as_ref(),
                update.intensity,
                update.cause,
                Some(&update.channels),
                Some(current.intent.clone()),
                update.watch_start_ts,
            );

            return CheckResult {
                note: Some(note),
                error: None,
            };
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
        let note = evaluate_failure_modes(&history, final_session_id.as_deref());

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

        // Append normalized tool outcomes (build/test/publish/git facts only).
        // Each entry is already a short fact string: "succeeded" or "failed: <snippet>".
        if !event.tool_calls.is_empty() {
            exchange_text.push_str("\n\nTool outcomes:\n");
            for tc in &event.tool_calls {
                exchange_text.push_str(&tc.name);
                exchange_text.push_str(" -> ");
                exchange_text.push_str(&tc.output);
                exchange_text.push('\n');
            }
        }

        let commit_mentioned = looks_like_commit_or_pr(&exchange_text);
        let conn_id = CONN_SEQ.fetch_add(1, Ordering::Relaxed);

        // Sticky session ID for the controller + session transition detection.
        // Also grab the last governor channels snapshot for TurnEval construction.
        let (final_session_id, turn_eval_channels) = {
            let controller = self.state.controllers.entry(ws.id.clone()).or_default();
            let prev_session = controller.last_agent_session_id.clone();
            let channels = controller.last_channels.take();
            if let Some(ref sid) = event.agent_session_id {
                controller.last_agent_session_id = Some(sid.clone());
            }

            // If the session ID has changed and we had a previous session, spawn a
            // background checkpoint task for the session that just ended.
            // We do this asynchronously so the current record_turn call is not blocked.
            if let (Some(prev), Some(cur)) = (&prev_session, &event.agent_session_id) {
                if prev != cur {
                    let ws_clone = ws.clone();
                    let prev_clone = prev.clone();
                    tokio::spawn(async move {
                        match crate::storage_checkpoint::maybe_create_checkpoint(
                            &ws_clone,
                            Some(&prev_clone),
                            "new_session_opencode",
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
            }

            let sid = event
                .agent_session_id
                .clone()
                .or_else(|| controller.last_agent_session_id.clone());
            (sid, channels)
        };

        let item = ChunkInput {
            conn_id,
            upstream_host: String::new(),
            request_path: String::new(),
            http_status: 200,
            exchange_text,
            commit_mentioned,
            agent_session_id: final_session_id,
            source_pointer: event.source_pointer,
            usage: event.usage.map(UsageMeta::from),
            grounding_note: event.grounding_note,
            source_ts_ms: event.source_ts_ms,
            workspace_root: ws.root.clone(),
            turn_eval_channels,
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

        if let Err(e) = process_flush_job(job, state.clone()).await {
            tracing::warn!("background flush job failed: {e}");
        }
    }
}

fn is_pivotal(
    user_text: &str,
    user_emotion: Option<&crate::emotion::EmotionMeta>,
    symbols: &[String],
) -> bool {
    // 1. Emotional Friction
    if let Some(e) = user_emotion {
        if e.valence < -0.3 {
            return true;
        }
        match e.label.as_str() {
            "frustration" | "anger" | "doubt" | "disapproval" | "confused" => return true,
            _ => {}
        }
    }

    // 2. Corrective Keywords
    if crate::governor::detect_correction(user_text, user_emotion) > 0.1 {
        return true;
    }

    // 3. Structural Signal (high churn)
    if symbols.len() > 3 {
        return true;
    }

    // 4. Substantial content (long turns often carry complex intent)
    if user_text.len() > 1500 {
        return true;
    }

    false
}

async fn process_flush_job(
    job: FlushJob,
    state: Arc<Mutex<BackgroundState>>,
) -> anyhow::Result<()> {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\nThen provide `capsules`: up to the requested max. Each capsule must be short and high-signal.\n\nStructure the capsule as an IntentCapsule: category, intent (why), decision (what), rationale (why that), next_steps, symbols (file paths), and questions.\n\n`questions`: 2-3 natural language questions that a developer would ask to find this capsule later (e.g. \"Why is the timeout set to 30 seconds?\", \"How does the proxy route requests upstream?\"). These are used for semantic retrieval — make them specific and grounded in the actual decision.\n\nThe slice may include a 'Tool outcomes' section listing normalized facts from build/test/publish/git tool calls (e.g. 'cargo build -> succeeded'). Use these facts as ground truth when filling decision and next_steps — do not infer build status from prose alone.";

    // Use the authoritative last decision from state (more reliable than the chunker's
    // snapshot, which may have been taken before the previous job was processed).
    let prior_decision = {
        let st = state.lock().await;
        st.last_decisions
            .get(&job.workspace_id)
            .cloned()
            .or_else(|| job.prior_decision.clone())
    };

    let (user_text, assistant_text) = extract_user_and_assistant_text(&job.input);

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
    let user_symbols = crate::net::extract_symbols_from_text(&user_text);
    let failure_mode = crate::governor::detect_failure_keywords(&job.input)
        .unwrap_or(crate::types::FailureMode::None);

    let extraction_mode = state.lock().await.extraction_mode;

    let use_llm = match extraction_mode {
        crate::types::ExtractionMode::Full => true,
        crate::types::ExtractionMode::None => false,
        crate::types::ExtractionMode::Hybrid => {
            is_pivotal(&user_text, user_emotion.as_ref(), &symbols)
        }
    };

    if use_llm {
        state
            .lock()
            .await
            .llm_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // Extract capsule using LLM (this is the expensive network call)
    let mut capsule = if use_llm {
        match crate::llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(mut c) => {
                // Augment LLM capsule with local symbols if LLM missed any
                for s in symbols {
                    if !c.symbols.contains(&s) {
                        c.symbols.push(s);
                    }
                }
                c.user_symbols = user_symbols;
                c.extraction_mode = extraction_mode;
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
                    user_symbols,
                    failure_mode,
                    failure_signals: Some("Heuristic extraction (LLM failed)".to_string()),
                    extraction_mode: crate::types::ExtractionMode::None, // Effectively none if it failed
                    questions: vec![],
                }
            }
        }
    } else {
        // Ghost Mode (Option B) or skipped by Hybrid
        IntentCapsule {
            category: "replay".to_string(),
            intent: user_text.lines().next().unwrap_or("").to_string(),
            decision: assistant_text.lines().next().unwrap_or("").to_string(),
            rationale: String::new(),
            next_steps: vec![],
            symbols,
            user_symbols,
            failure_mode,
            failure_signals: if extraction_mode == crate::types::ExtractionMode::Hybrid {
                Some("Ghost extraction (Hybrid: non-pivotal)".to_string())
            } else {
                Some("Ghost extraction (no-LLM)".to_string())
            },
            extraction_mode: crate::types::ExtractionMode::None,
            questions: vec![],
        }
    };

    crate::util::augment_capsule_symbols_from_input(&mut capsule, &job.input);

    if let Some(note) = job.grounding_note {
        let current = capsule.failure_signals.unwrap_or_default();
        if current.is_empty() {
            capsule.failure_signals = Some(format!("Grounding: {}", note));
        } else {
            capsule.failure_signals = Some(format!("{} | Grounding: {}", current, note));
        }
    }

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
        root: job.workspace_root.clone(),
        db_dir: ws_dir.join("lancedb"),
        capsules_jsonl: ws_dir.join("capsules.jsonl"),
        metrics_jsonl: ws_dir.join("metrics.jsonl"),
    };

    // ── Build TurnEval ────────────────────────────────────────────────────────
    // All heuristic — zero LLM calls. Uses:
    // (a) Governor channels snapshot carried in from check_turn via ChunkInput.
    // (b) Coach scores derived from the capsule + prior history + exchange text.
    let turn_eval = {
        // Query recent capsule history for coach scoring. Best-effort; empty on error.
        // We take the last 8 capsules regardless of session — recent history is what matters
        // for scoring clarity, scope, and progress.
        let history = match crate::storage::scan_capsules_lancedb(
            &ws, 8, None, None, None, None, None,
        )
        .await
        {
            Ok(h) => h,
            Err(_) => vec![],
        };

        // Resolve session turn count from background state.
        let session_turn_count = {
            let mut st = state.lock().await;
            let sid = job.meta.agent_session_id.as_deref().unwrap_or("__none__");
            let count = st.session_turn_counts.entry(sid.to_string()).or_insert(0);
            *count += 1;
            *count
        };

        let coach_input = crate::governor::CoachInput {
            capsule: &capsule,
            history: &history,
            exchange_text: &job.input,
            usage: job.meta.usage.as_ref(),
            session_turn_count,
        };
        let coach = crate::governor::compute_coach_scores(&coach_input);

        // Unpack governor channels (may be None for replayed/non-live turns).
        let (channels, traj_intensity, traj_state) = job
            .turn_eval_channels
            .clone()
            .unwrap_or_default();

        let flags = crate::governor::compute_flags(
            &channels,
            traj_intensity,
            &coach,
            session_turn_count,
        );

        // Combine evidence from coach + any flag evidence.
        let mut evidence = coach.evidence.clone();
        if channels.path_hallucination > 0.5 {
            evidence.push(format!(
                "path_hallucination={:.2} (agent referenced non-existent files)",
                channels.path_hallucination
            ));
        }
        if channels.alignment_debt > 0.4 {
            evidence.push(format!(
                "alignment_debt={:.2} (repeated user corrections detected)",
                channels.alignment_debt
            ));
        }

        crate::types::TurnEval {
            version: "v1".to_string(),
            // Agent tuning (tune)
            repetition: channels.repetition,
            novelty_collapse: channels.novelty_collapse,
            semantic_stall: channels.semantic_stall,
            effort_spike: channels.effort_spike,
            alignment_debt: channels.alignment_debt,
            path_hallucination: channels.path_hallucination,
            grounding_stall: channels.grounding_stall,
            instruction_staticness: channels.instruction_staticness,
            logic_churn: channels.logic_churn,
            fluency: channels.fluency,
            trajectory_intensity: traj_intensity,
            trajectory_state: traj_state,
            // Developer coaching
            clarity: coach.clarity,
            context_freshness: coach.context_freshness,
            verification_rigor: coach.verification_rigor,
            decision_progress: coach.decision_progress,
            scope_discipline: coach.scope_discipline,
            cost_acceleration: coach.cost_acceleration,
            // Flags + outcome
            flags,
            outcome_hint: "unclear".to_string(),
            evidence,
        }
    };

    // Append to JSONL (cheap, local)
    append_capsule_jsonl(
        &ws.capsules_jsonl,
        job.ts_ms,
        job.conn_id,
        job.exchange_seq,
        &job.meta,
        &capsule,
        &turn_eval,
        job.head_sha.as_deref(),
        job.commit_sha.as_deref(),
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
        &turn_eval,
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
        &turn_eval,
        prior_decision.as_deref(),
        job.head_sha.as_deref(),
        job.commit_sha.as_deref(),
    )
    .await?;

    // Record the decision so the next capsule in this workspace can encode causal continuity.
    if !capsule.decision.trim().is_empty() {
        state
            .lock()
            .await
            .last_decisions
            .insert(job.workspace_id.clone(), capsule.decision.clone());
    }

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
    turn_eval: &crate::types::TurnEval,
    head_sha: Option<&str>,
    commit_sha: Option<&str>,
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
        "source_pointer": meta.source_pointer,
        "usage": usage,
        "capsule": capsule,
        "turn_eval": turn_eval,
        "head_sha": head_sha,
        "commit_sha": commit_sha,
    });
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{}\n", record).as_bytes())?;
    Ok(())
}
