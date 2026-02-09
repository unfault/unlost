use crate::analysis::{AnalysisMeta, AnalysisMsg};
use crate::embed::Embedder;
use crate::emotion::{
    EmotionModel, apply_context_heuristics, extract_user_and_assistant_text, map_go_emotions,
};
use crate::storage::ensure_capsules_table;
use bytes::Bytes;
use kanal::AsyncReceiver;
use lancedb::connection::Connection;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};
use tracing::{debug, warn};

fn append_capsule_jsonl(
    path: &std::path::Path,
    ts_ms: i64,
    conn_id: u64,
    exchange_seq: u64,
    meta: &crate::ResponseMeta,
    capsule: &crate::IntentCapsule,
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

fn contains_hex_hash(s: &str) -> bool {
    for tok in s
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        if tok.len() < 7 || tok.len() > 40 {
            continue;
        }
        if tok.chars().all(|c: char| c.is_ascii_hexdigit()) {
            return true;
        }
    }
    false
}

pub(crate) fn looks_like_commit_or_pr(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    if s.contains("git commit") || s.contains("commit:") || s.contains("commit ") {
        return true;
    }
    if s.contains("pull request") || s.contains("merge request") {
        return true;
    }
    if s.contains("pr #") || s.contains("pr#") {
        return true;
    }

    // cheap hash heuristic (7+ hex) preceded by common tokens
    if s.contains("sha") || s.contains("hash") || s.contains("commit") {
        if contains_hex_hash(&s) {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
pub(crate) struct ChunkInput {
    pub(crate) conn_id: u64,
    pub(crate) upstream_host: String,
    pub(crate) request_path: String,
    pub(crate) http_status: u16,
    pub(crate) exchange_text: String,
    pub(crate) commit_mentioned: bool,
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    pub(crate) agent_session_id: Option<String>,
    /// Best-effort usage metrics (tokens/cost). Not always present.
    pub(crate) usage: Option<crate::types::UsageMeta>,
}

#[derive(Debug, Clone)]
pub(crate) struct FlushJob {
    pub(crate) workspace_id: String,
    pub(crate) conn_id: u64,
    pub(crate) exchange_seq: u64,
    pub(crate) ts_ms: i64,
    pub(crate) meta: crate::ResponseMeta,
    pub(crate) input: String,
}

struct WorkspaceBuffer {
    next_seq: u64,
    last_activity: Instant,
    last_conn_id: u64,
    last_upstream_host: String,
    last_request_path: String,
    last_http_status: u16,
    last_agent_session_id: Option<String>,
    last_usage: Option<crate::types::UsageMeta>,
    total_chars: usize,
    turns: Vec<String>,
    saw_commit: bool,
}

impl WorkspaceBuffer {
    fn new(now: Instant) -> Self {
        Self {
            next_seq: 0,
            last_activity: now,
            last_conn_id: 0,
            last_upstream_host: String::new(),
            last_request_path: String::new(),
            last_http_status: 0,
            last_agent_session_id: None,
            last_usage: None,
            total_chars: 0,
            turns: Vec::new(),
            saw_commit: false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct WorkspaceChunker {
    buffers: Arc<Mutex<HashMap<String, WorkspaceBuffer>>>,
    flush_tx: kanal::AsyncSender<FlushJob>,
}

impl WorkspaceChunker {
    const IDLE_FLUSH_AFTER: Duration = Duration::from_secs(2);
    const MAX_TOTAL_CHARS: usize = 16 * 1024;
    const MAX_TURNS: usize = 8;

    pub(crate) fn new(flush_tx: kanal::AsyncSender<FlushJob>) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            flush_tx,
        }
    }

    pub(crate) async fn ingest(&self, workspace_id: String, item: ChunkInput) {
        let now = Instant::now();
        let mut maybe_flush: Option<FlushJob> = None;

        {
            let mut map = self.buffers.lock().await;
            let buf = map
                .entry(workspace_id.clone())
                .or_insert_with(|| WorkspaceBuffer::new(now));

            // If this buffer has been idle, flush it before appending new content.
            if !buf.turns.is_empty()
                && now.duration_since(buf.last_activity) >= Self::IDLE_FLUSH_AFTER
            {
                maybe_flush = Some(build_flush_job(workspace_id.clone(), buf));
                *buf = WorkspaceBuffer::new(now);
            }

            buf.last_activity = now;
            buf.last_conn_id = item.conn_id;
            buf.last_upstream_host = item.upstream_host;
            buf.last_request_path = item.request_path;
            buf.last_http_status = item.http_status;
            buf.last_agent_session_id = item.agent_session_id;
            buf.last_usage = item.usage;
            buf.saw_commit |= item.commit_mentioned;

            buf.total_chars = buf.total_chars.saturating_add(item.exchange_text.len());
            buf.turns.push(item.exchange_text);

            // Force flush boundaries.
            let too_big = buf.total_chars >= Self::MAX_TOTAL_CHARS;
            let too_many = buf.turns.len() >= Self::MAX_TURNS;
            let milestone = item.commit_mentioned;
            if too_big || too_many || milestone {
                if maybe_flush.is_none() {
                    maybe_flush = Some(build_flush_job(workspace_id.clone(), buf));
                    *buf = WorkspaceBuffer::new(now);
                }
            }
        }

        if let Some(job) = maybe_flush {
            let _ = self.flush_tx.send(job).await;
        }
    }

    pub(crate) async fn flush_idle(&self) {
        let now = Instant::now();
        let mut jobs: Vec<FlushJob> = Vec::new();

        {
            let mut map = self.buffers.lock().await;
            for (ws_id, buf) in map.iter_mut() {
                if buf.turns.is_empty() {
                    continue;
                }
                if now.duration_since(buf.last_activity) < Self::IDLE_FLUSH_AFTER {
                    continue;
                }
                jobs.push(build_flush_job(ws_id.clone(), buf));
                *buf = WorkspaceBuffer::new(now);
            }
        }

        for j in jobs {
            let _ = self.flush_tx.send(j).await;
        }
    }

    /// Force-flush the buffer for a single workspace, regardless of idle time.
    pub(crate) async fn flush_workspace(&self, workspace_id: &str) {
        let now = Instant::now();
        let mut job: Option<FlushJob> = None;

        {
            let mut map = self.buffers.lock().await;
            if let Some(buf) = map.get_mut(workspace_id) {
                if !buf.turns.is_empty() {
                    job = Some(build_flush_job(workspace_id.to_string(), buf));
                    *buf = WorkspaceBuffer::new(now);
                }
            }
        }

        if let Some(j) = job {
            let _ = self.flush_tx.send(j).await;
        }
    }
}

fn build_flush_job(workspace_id: String, buf: &mut WorkspaceBuffer) -> FlushJob {
    buf.next_seq += 1;
    let exchange_seq = buf.next_seq;
    let ts_ms = crate::now_ms();

    let mut input = String::new();
    input.push_str("Signals:\n");
    input.push_str(&format!("commit_mentioned={}\n\n", buf.saw_commit));
    input.push_str("Conversation slice (newest last):\n\n");
    for (i, t) in buf.turns.iter().enumerate() {
        input.push_str(&format!("Turn {}:\n", i + 1));
        input.push_str(t);
        input.push_str("\n\n");
    }

    let meta = crate::ResponseMeta {
        source: "record".to_string(),
        upstream_host: buf.last_upstream_host.clone(),
        request_path: buf.last_request_path.clone(),
        http_status: buf.last_http_status,
        agent_session_id: buf.last_agent_session_id.clone(),
        usage: buf.last_usage.clone(),
    };

    FlushJob {
        workspace_id,
        conn_id: buf.last_conn_id,
        exchange_seq,
        ts_ms,
        meta,
        input,
    }
}

#[derive(Clone)]
pub(crate) struct ServeState {
    pub(crate) embedder: Embedder,
    db_cache: Arc<Mutex<HashMap<String, Connection>>>,
    pub(crate) emotion: Arc<std::sync::Mutex<EmotionModel>>,
}

impl ServeState {
    pub(crate) fn new(embedder: Embedder, emotion: EmotionModel) -> Self {
        Self {
            embedder,
            db_cache: Arc::new(Mutex::new(HashMap::new())),
            emotion: Arc::new(std::sync::Mutex::new(emotion)),
        }
    }

    pub(crate) fn workspace_paths(&self, workspace_id: &str) -> crate::WorkspacePaths {
        let ws_dir = crate::unlost_workspace_dir(workspace_id);
        crate::WorkspacePaths {
            id: workspace_id.to_string(),
            db_dir: ws_dir.join("lancedb"),
            capsules_jsonl: ws_dir.join("capsules.jsonl"),
            metrics_jsonl: ws_dir.join("metrics.jsonl"),
        }
    }

    pub(crate) async fn db_for(&self, workspace_id: &str) -> anyhow::Result<Connection> {
        {
            let cache = self.db_cache.lock().await;
            if let Some(c) = cache.get(workspace_id) {
                return Ok(c.clone());
            }
        }

        let ws = self.workspace_paths(workspace_id);
        std::fs::create_dir_all(&ws.db_dir)?;
        let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
            .execute()
            .await?;
        let _ = ensure_capsules_table(&db).await?;

        let mut cache = self.db_cache.lock().await;
        cache.insert(workspace_id.to_string(), db.clone());
        Ok(db)
    }

    /// Fetch recent capsules for friction detection. Returns empty vec on error.
    pub(crate) async fn get_recent_capsules(
        &self,
        workspace_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::CapsuleHit>> {
        let ws = self.workspace_paths(workspace_id);
        crate::storage::scan_capsules_lancedb(&ws, limit, None, None, None, None, None).await
    }
}

pub(crate) async fn process_flush_jobs_serve(rx: AsyncReceiver<FlushJob>, state: ServeState) {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON with fields: {category, intent, decision, rationale, next_steps (array), symbols (array), failure_mode, failure_signals}.\n\
\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.\n\
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

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let (user_text, assistant_text) = extract_user_and_assistant_text(&job.input);
        let emotion_handle = state.emotion.clone();
        let user_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if user_text.trim().is_empty() {
                return None;
            }
            let (raw, score) = model.classify_one(&user_text).ok()?;
            let meta = map_go_emotions(&raw, score);
            Some(apply_context_heuristics(&user_text, meta))
        })
        .await
        .ok()
        .flatten();

        let emotion_handle = state.emotion.clone();
        let assistant_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if assistant_text.trim().is_empty() {
                return None;
            }
            let (raw, score) = model.classify_one(&assistant_text).ok()?;
            // No heuristics for assistant - we trust the model there
            Some(map_go_emotions(&raw, score))
        })
        .await
        .ok()
        .flatten();

        let capsule =
            match crate::llm_extract::<crate::IntentCapsule>(None, PREAMBLE, &job.input).await {
                Ok(c) => c,
                Err(e) => {
                    warn!(
                        workspace_id = %job.workspace_id,
                        conn_id = job.conn_id,
                        exchange_seq = job.exchange_seq,
                        error = ?e,
                        "capsule extraction failed"
                    );
                    continue;
                }
            };

        let ws_paths = state.workspace_paths(&job.workspace_id);
        if let Err(e) = append_capsule_jsonl(
            &ws_paths.capsules_jsonl,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to append capsule jsonl");
        }

        if let Err(e) = crate::metrics::record_capsule_saved(
            &ws_paths,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            user_emotion.as_ref(),
            assistant_emotion.as_ref(),
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to record metrics event");
        }

        match state.db_for(&job.workspace_id).await {
            Ok(db) => {
                if let Err(e) = crate::storage::insert_capsule_row(
                    &db,
                    &state.embedder,
                    job.conn_id,
                    job.exchange_seq,
                    job.ts_ms,
                    &job.meta,
                    user_emotion.as_ref(),
                    assistant_emotion.as_ref(),
                    &capsule,
                )
                .await
                {
                    warn!(workspace_id = %job.workspace_id, error = ?e, "failed to insert capsule into lancedb");
                }
            }
            Err(e) => {
                warn!(workspace_id = %job.workspace_id, error = ?e, "failed to open workspace db");
            }
        }
    }
}

pub(crate) async fn process_flush_jobs_proxy(
    rx: AsyncReceiver<FlushJob>,
    ws: crate::WorkspacePaths,
    db: Connection,
    embedder: Embedder,
    emotion: Arc<std::sync::Mutex<EmotionModel>>,
) {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON with fields: {category, intent, decision, rationale, next_steps (array), symbols (array), failure_mode, failure_signals}.\n\
\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.\n\
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

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let (user_text, assistant_text) = extract_user_and_assistant_text(&job.input);
        let emotion_handle = emotion.clone();
        let user_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if user_text.trim().is_empty() {
                return None;
            }
            let (raw, score) = model.classify_one(&user_text).ok()?;
            let meta = map_go_emotions(&raw, score);
            Some(apply_context_heuristics(&user_text, meta))
        })
        .await
        .ok()
        .flatten();

        let emotion_handle = emotion.clone();
        let assistant_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if assistant_text.trim().is_empty() {
                return None;
            }
            let (raw, score) = model.classify_one(&assistant_text).ok()?;
            // No heuristics for assistant - we trust the model there
            Some(map_go_emotions(&raw, score))
        })
        .await
        .ok()
        .flatten();

        let capsule =
            match crate::llm_extract::<crate::IntentCapsule>(None, PREAMBLE, &job.input).await {
                Ok(c) => c,
                Err(e) => {
                    warn!(
                        workspace_id = %job.workspace_id,
                        conn_id = job.conn_id,
                        exchange_seq = job.exchange_seq,
                        error = ?e,
                        "capsule extraction failed"
                    );
                    continue;
                }
            };

        if let Err(e) = append_capsule_jsonl(
            &ws.capsules_jsonl,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to append capsule jsonl");
        }

        if let Err(e) = crate::metrics::record_capsule_saved(
            &ws,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            user_emotion.as_ref(),
            assistant_emotion.as_ref(),
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to record metrics event");
        }

        if let Err(e) = crate::storage::insert_capsule_row(
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
        .await
        {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to insert capsule into lancedb");
        }
    }
}

pub(crate) async fn analysis_worker_multiplex(
    rx: AsyncReceiver<AnalysisMsg>,
    chunker: WorkspaceChunker,
    conn_id: u64,
) {
    debug!(conn_id, "analysis worker started");
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;

    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() {
            p
        } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                Ok(_) => continue,
                Err(_) => break,
            }
        };

        let user_text = crate::net::decode_json_lossy(&request_body)
            .and_then(|v| {
                if meta.request_path.contains("/v1/messages") {
                    crate::net::extract_anthropic_user_text(&v)
                        .or_else(|| crate::net::extract_openai_message_text(&v))
                } else {
                    crate::net::extract_openai_message_text(&v)
                        .or_else(|| crate::net::extract_anthropic_user_text(&v))
                }
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = crate::net::is_event_stream(meta.content_type.as_deref());

        loop {
            let msg = match rx.recv().await {
                Ok(m) => m,
                Err(_) => break,
            };

            match msg {
                AnalysisMsg::ResponseEnd => break,
                AnalysisMsg::ExchangeStart {
                    meta: next_meta,
                    request_body: next_req,
                } => {
                    pending_start = Some((next_meta, next_req));
                    break;
                }
                AnalysisMsg::ResponseChunk(b) => {
                    if sse {
                        sse_buf.extend_from_slice(&b);
                        crate::net::sse_extract_deltas(&mut sse_buf, &mut assistant_text);
                    } else {
                        raw_buf.extend_from_slice(&b);
                    }

                    if assistant_text.len() > 512 * 1024 {
                        assistant_text.truncate(512 * 1024);
                    }
                    if raw_buf.len() > 2 * 1024 * 1024 {
                        raw_buf.truncate(2 * 1024 * 1024);
                    }
                }
            }
        }

        if !sse {
            if let Some(v) = crate::net::decode_json_lossy(&raw_buf) {
                if meta.request_path.contains("/v1/messages") {
                    assistant_text = crate::net::extract_anthropic_assistant_text_from_json(&v)
                        .or_else(|| crate::net::extract_openai_assistant_text_from_json(&v))
                        .unwrap_or_default();
                } else {
                    assistant_text = crate::net::extract_openai_assistant_text_from_json(&v)
                        .or_else(|| crate::net::extract_anthropic_assistant_text_from_json(&v))
                        .unwrap_or_default();
                }
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() {
            continue;
        }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() {
            input.push_str("User:\n");
            input.push_str(u);
            input.push_str("\n\n");
        }
        if !assistant_text.is_empty() {
            input.push_str("Assistant:\n");
            input.push_str(&assistant_text);
        }

        let commit_mentioned = looks_like_commit_or_pr(&input);
        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input,
            commit_mentioned,
            agent_session_id: None, // HTTP proxy doesn't have agent session context
            usage: None,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }

    debug!(conn_id, "analysis worker finished");
}

pub(crate) async fn analysis_worker(
    rx: AsyncReceiver<AnalysisMsg>,
    chunker: WorkspaceChunker,
    conn_id: u64,
) {
    debug!(conn_id, "analysis worker started");
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;

    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() {
            p
        } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                Ok(_) => continue,
                Err(_) => break,
            }
        };

        let user_text = crate::net::decode_json_lossy(&request_body)
            .and_then(|v| {
                if meta.request_path.contains("/v1/messages") {
                    crate::net::extract_anthropic_user_text(&v)
                        .or_else(|| crate::net::extract_openai_message_text(&v))
                } else {
                    crate::net::extract_openai_message_text(&v)
                        .or_else(|| crate::net::extract_anthropic_user_text(&v))
                }
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = crate::net::is_event_stream(meta.content_type.as_deref());

        loop {
            let msg = match rx.recv().await {
                Ok(m) => m,
                Err(_) => {
                    // channel closed mid-exchange
                    break;
                }
            };

            match msg {
                AnalysisMsg::ResponseEnd => break,
                AnalysisMsg::ExchangeStart {
                    meta: next_meta,
                    request_body: next_req,
                } => {
                    pending_start = Some((next_meta, next_req));
                    break;
                }
                AnalysisMsg::ResponseChunk(b) => {
                    if sse {
                        sse_buf.extend_from_slice(&b);
                        crate::net::sse_extract_deltas(&mut sse_buf, &mut assistant_text);
                    } else {
                        raw_buf.extend_from_slice(&b);
                    }

                    // Hard bound: we do not want to hold huge transient buffers.
                    if assistant_text.len() > 512 * 1024 {
                        assistant_text.truncate(512 * 1024);
                    }
                    if raw_buf.len() > 2 * 1024 * 1024 {
                        raw_buf.truncate(2 * 1024 * 1024);
                    }
                }
            }
        }

        if !sse {
            if let Some(v) = crate::net::decode_json_lossy(&raw_buf) {
                if meta.request_path.contains("/v1/messages") {
                    assistant_text = crate::net::extract_anthropic_assistant_text_from_json(&v)
                        .or_else(|| crate::net::extract_openai_assistant_text_from_json(&v))
                        .unwrap_or_default();
                } else {
                    assistant_text = crate::net::extract_openai_assistant_text_from_json(&v)
                        .or_else(|| crate::net::extract_anthropic_assistant_text_from_json(&v))
                        .unwrap_or_default();
                }
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() {
            continue;
        }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() {
            input.push_str("User:\n");
            input.push_str(u);
            input.push_str("\n\n");
        }
        if !assistant_text.is_empty() {
            input.push_str("Assistant:\n");
            input.push_str(&assistant_text);
        }

        let commit_mentioned = looks_like_commit_or_pr(&input);
        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input,
            commit_mentioned,
            agent_session_id: None, // HTTP proxy doesn't have agent session context
            usage: None,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }

    debug!(conn_id, "analysis worker finished");
}
