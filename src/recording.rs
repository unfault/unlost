use crate::analysis::{AnalysisMeta, AnalysisMsg};
use crate::embed::Embedder;
use crate::emotion::{
    EmotionModel, apply_context_heuristics, extract_user_and_assistant_text, map_go_emotions,
};
use crate::storage::ensure_capsules_table;
use bytes::Bytes;
use kanal::AsyncReceiver;
use lancedb::connection::Connection;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

struct RecentJobDeduper {
    ttl: std::time::Duration,
    max_entries: usize,
    seen: HashMap<String, std::time::Instant>,
}

impl RecentJobDeduper {
    fn new(ttl: std::time::Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries,
            seen: HashMap::new(),
        }
    }

    fn should_skip_and_mark(&mut self, key: String) -> bool {
        let now = std::time::Instant::now();
        self.prune(now);

        if let Some(prev) = self.seen.get(&key) {
            if now.duration_since(*prev) <= self.ttl {
                // Refresh the timestamp so rapid repeated duplicates remain suppressed.
                self.seen.insert(key, now);
                return true;
            }
        }

        self.seen.insert(key, now);
        false
    }

    fn prune(&mut self, now: std::time::Instant) {
        self.seen
            .retain(|_, t| now.duration_since(*t) <= self.ttl);
        if self.seen.len() <= self.max_entries {
            return;
        }

        // Keep the most recent max_entries.
        let mut items = self.seen.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.max_entries);
        self.seen = items.into_iter().collect();
    }
}

fn flush_job_dedupe_key(job: &FlushJob) -> String {
    let mut h = Sha256::new();
    h.update(job.workspace_id.as_bytes());
    h.update(&[0u8]);
    h.update(job.conn_id.to_le_bytes());
    h.update(&[0u8]);
    h.update(job.meta.upstream_host.as_bytes());
    h.update(&[0u8]);
    h.update(job.meta.request_path.as_bytes());
    h.update(&[0u8]);
    h.update(job.meta.agent_session_id.as_deref().unwrap_or("").as_bytes());
    h.update(&[0u8]);
    h.update(job.input.as_bytes());
    format!(
        "{}|{}|{}|{}|{}",
        job.workspace_id,
        job.conn_id,
        job.meta.request_path,
        job.meta.agent_session_id.as_deref().unwrap_or(""),
        hex::encode(h.finalize())
    )
}

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
    /// Optional grounding info (e.g. verified git commits)
    pub(crate) grounding_note: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct FlushJob {
    pub(crate) workspace_id: String,
    pub(crate) conn_id: u64,
    pub(crate) exchange_seq: u64,
    pub(crate) ts_ms: i64,
    pub(crate) meta: crate::ResponseMeta,
    pub(crate) input: String,
    pub(crate) grounding_note: Option<String>,
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
    last_grounding_note: Option<String>,
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
            last_grounding_note: None,
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
            buf.last_grounding_note = item.grounding_note;
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

    /// Force-flush all buffers for all workspaces.
    pub(crate) async fn flush_all(&self) {
        let now = Instant::now();
        let mut jobs: Vec<FlushJob> = Vec::new();

        {
            let mut map = self.buffers.lock().await;
            for (ws_id, buf) in map.iter_mut() {
                if !buf.turns.is_empty() {
                    jobs.push(build_flush_job(ws_id.clone(), buf));
                    *buf = WorkspaceBuffer::new(now);
                }
            }
        }

        for j in jobs {
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
        grounding_note: buf.last_grounding_note.clone(),
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
 Return JSON with fields: {category, intent, decision, rationale, next_steps (array), symbols (array), failure_mode, failure_signals}.\n\n\
 Rules:\n\
 - Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
 - Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
 - Keep each field concise; next_steps max 3.\n\
 - symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.\n\n\
 Failure mode detection: none, drift, rediscovery, decision_conflict, retry_spiral, false_progress, unbounded_horizon.";

    let mut deduper = RecentJobDeduper::new(std::time::Duration::from_secs(45), 2048);

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let dedupe_key = flush_job_dedupe_key(&job);
        if deduper.should_skip_and_mark(dedupe_key) {
            continue;
        }

        let (user_text, assistant_text) = extract_user_and_assistant_text(&job.input);
        
        let emotion_handle = state.emotion.clone();
        let user_text_clone = user_text.clone();
        let user_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if user_text_clone.trim().is_empty() { return None; }
            let (raw, score) = model.classify_one(&user_text_clone).ok()?;
            let meta = map_go_emotions(&raw, score);
            Some(apply_context_heuristics(&user_text_clone, meta))
        }).await.ok().flatten();

        let emotion_handle = state.emotion.clone();
        let assistant_text_clone = assistant_text.clone();
        let assistant_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if assistant_text_clone.trim().is_empty() { return None; }
            let (raw, score) = model.classify_one(&assistant_text_clone).ok()?;
            Some(map_go_emotions(&raw, score))
        }).await.ok().flatten();

        let symbols = crate::net::extract_symbols_from_text(&job.input);
        let user_symbols = crate::net::extract_symbols_from_text(&user_text);
        let failure_mode = crate::governor::detect_failure_keywords(&job.input).unwrap_or(crate::types::FailureMode::None);

        let mut capsule = match crate::llm_extract::<crate::IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(mut c) => {
                for s in symbols { if !c.symbols.contains(&s) { c.symbols.push(s); } }
                c.user_symbols = user_symbols;
                c
            }
            Err(_) => crate::IntentCapsule {
                category: "unknown".to_string(),
                intent: user_text.lines().next().unwrap_or("").to_string(),
                decision: assistant_text.lines().next().unwrap_or("").to_string(),
                rationale: String::new(),
                next_steps: vec![],
                symbols,
                user_symbols,
                failure_mode,
                failure_signals: Some("Heuristic extraction (LLM failed)".to_string()),
            },
        };

        crate::util::augment_capsule_symbols_from_input(&mut capsule, &job.input);

        let ws_paths = state.workspace_paths(&job.workspace_id);
        let _ = append_capsule_jsonl(&ws_paths.capsules_jsonl, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, &capsule);
        let _ = crate::metrics::record_capsule_saved(&ws_paths, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, user_emotion.as_ref(), assistant_emotion.as_ref(), &capsule);

        if let Ok(db) = state.db_for(&job.workspace_id).await {
            let _ = crate::storage::insert_capsule_row(&db, &state.embedder, job.conn_id, job.exchange_seq, job.ts_ms, &job.meta, user_emotion.as_ref(), assistant_emotion.as_ref(), &capsule, Some(&job.input)).await;
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
 Return JSON with fields: {category, intent, decision, rationale, next_steps (array), symbols (array), failure_mode, failure_signals}.\n\n\
 Failure mode detection: none, drift, rediscovery, decision_conflict, retry_spiral, false_progress, unbounded_horizon.";

    let mut deduper = RecentJobDeduper::new(std::time::Duration::from_secs(45), 2048);

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let dedupe_key = flush_job_dedupe_key(&job);
        if deduper.should_skip_and_mark(dedupe_key) { continue; }

        let (user_text, assistant_text) = extract_user_and_assistant_text(&job.input);
        
        let emotion_handle = emotion.clone();
        let user_text_clone = user_text.clone();
        let user_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if user_text_clone.trim().is_empty() { return None; }
            let (raw, score) = model.classify_one(&user_text_clone).ok()?;
            let meta = map_go_emotions(&raw, score);
            Some(apply_context_heuristics(&user_text_clone, meta))
        }).await.ok().flatten();

        let emotion_handle = emotion.clone();
        let assistant_text_clone = assistant_text.clone();
        let assistant_emotion = tokio::task::spawn_blocking(move || {
            let mut model = emotion_handle.lock().ok()?;
            if assistant_text_clone.trim().is_empty() { return None; }
            let (raw, score) = model.classify_one(&assistant_text_clone).ok()?;
            Some(map_go_emotions(&raw, score))
        }).await.ok().flatten();

        let symbols = crate::net::extract_symbols_from_text(&job.input);
        let user_symbols = crate::net::extract_symbols_from_text(&user_text);
        let failure_mode = crate::governor::detect_failure_keywords(&job.input).unwrap_or(crate::types::FailureMode::None);

        let mut capsule = match crate::llm_extract::<crate::IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(mut c) => {
                for s in symbols { if !c.symbols.contains(&s) { c.symbols.push(s); } }
                c.user_symbols = user_symbols;
                c
            }
            Err(_) => crate::IntentCapsule {
                category: "unknown".to_string(),
                intent: user_text.lines().next().unwrap_or("").to_string(),
                decision: assistant_text.lines().next().unwrap_or("").to_string(),
                rationale: String::new(),
                next_steps: vec![],
                symbols,
                user_symbols,
                failure_mode,
                failure_signals: Some("Heuristic extraction (LLM failed)".to_string()),
            },
        };

        crate::util::augment_capsule_symbols_from_input(&mut capsule, &job.input);

        let _ = append_capsule_jsonl(&ws.capsules_jsonl, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, &capsule);
        let _ = crate::metrics::record_capsule_saved(&ws, job.ts_ms, job.conn_id, job.exchange_seq, &job.meta, user_emotion.as_ref(), assistant_emotion.as_ref(), &capsule);
        let _ = crate::storage::insert_capsule_row(&db, &embedder, job.conn_id, job.exchange_seq, job.ts_ms, &job.meta, user_emotion.as_ref(), assistant_emotion.as_ref(), &capsule, Some(&job.input)).await;
    }
}

pub(crate) async fn analysis_worker_multiplex(
    rx: AsyncReceiver<AnalysisMsg>,
    chunker: WorkspaceChunker,
    conn_id: u64,
) {
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;
    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() { p } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                _ => break,
            }
        };

        let user_text = crate::net::decode_json_lossy(&request_body).and_then(|v| {
            if meta.request_path.contains("/v1/messages") { crate::net::extract_anthropic_user_text(&v) }
            else { crate::net::extract_openai_message_text(&v) }
        }).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = crate::net::is_event_stream(meta.content_type.as_deref());

        loop {
            match rx.recv().await {
                Ok(AnalysisMsg::ResponseEnd) => break,
                Ok(AnalysisMsg::ExchangeStart { meta: n_meta, request_body: n_req }) => {
                    pending_start = Some((n_meta, n_req));
                    break;
                }
                Ok(AnalysisMsg::ResponseChunk(b)) => {
                    if sse { sse_buf.extend_from_slice(&b); crate::net::sse_extract_deltas(&mut sse_buf, &mut assistant_text); }
                    else { raw_buf.extend_from_slice(&b); }
                }
                Err(_) => break,
            }
        }

        if !sse {
            if let Some(v) = crate::net::decode_json_lossy(&raw_buf) {
                assistant_text = if meta.request_path.contains("/v1/messages") { crate::net::extract_anthropic_assistant_text_from_json(&v) }
                else { crate::net::extract_openai_assistant_text_from_json(&v) }.unwrap_or_default();
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() { continue; }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() { input.push_str("User:\n"); input.push_str(u); input.push_str("\n\n"); }
        if !assistant_text.is_empty() { input.push_str("Assistant:\n"); input.push_str(&assistant_text); }

        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input.clone(),
            commit_mentioned: looks_like_commit_or_pr(&input),
            agent_session_id: None,
            usage: None,
            grounding_note: None,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }
}

pub(crate) async fn analysis_worker(
    rx: AsyncReceiver<AnalysisMsg>,
    chunker: WorkspaceChunker,
    conn_id: u64,
) {
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;
    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() { p } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                _ => break,
            }
        };

        let user_text = crate::net::decode_json_lossy(&request_body).and_then(|v| {
            if meta.request_path.contains("/v1/messages") { crate::net::extract_anthropic_user_text(&v) }
            else { crate::net::extract_openai_message_text(&v) }
        }).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = crate::net::is_event_stream(meta.content_type.as_deref());

        loop {
            match rx.recv().await {
                Ok(AnalysisMsg::ResponseEnd) => break,
                Ok(AnalysisMsg::ExchangeStart { meta: n_meta, request_body: n_req }) => {
                    pending_start = Some((n_meta, n_req));
                    break;
                }
                Ok(AnalysisMsg::ResponseChunk(b)) => {
                    if sse { sse_buf.extend_from_slice(&b); crate::net::sse_extract_deltas(&mut sse_buf, &mut assistant_text); }
                    else { raw_buf.extend_from_slice(&b); }
                }
                Err(_) => break,
            }
        }

        if !sse {
            if let Some(v) = crate::net::decode_json_lossy(&raw_buf) {
                assistant_text = if meta.request_path.contains("/v1/messages") { crate::net::extract_anthropic_assistant_text_from_json(&v) }
                else { crate::net::extract_openai_assistant_text_from_json(&v) }.unwrap_or_default();
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() { continue; }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() { input.push_str("User:\n"); input.push_str(u); input.push_str("\n\n"); }
        if !assistant_text.is_empty() { input.push_str("Assistant:\n"); input.push_str(&assistant_text); }

        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input.clone(),
            commit_mentioned: looks_like_commit_or_pr(&input),
            agent_session_id: None,
            usage: None,
            grounding_note: None,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }
}
