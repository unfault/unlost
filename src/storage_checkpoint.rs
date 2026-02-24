//! Storage layer for the `checkpoints_v1` LanceDB table.
//!
//! Checkpoints are pre-synthesized story segments generated at session
//! boundaries. They serve as a fast-path for recall, brief, and pr-comment
//! so those commands don't have to re-run raw capsules through the LLM every
//! time.
//!
//! Schema mirrors the design choices in storage.rs: Arrow-typed columns,
//! best-effort schema evolution via `add_columns`, and a separate constant
//! for the table name to allow future versioning.

use anyhow::Context;
use arrow_array::{
    Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    builder::{ListBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema};
use futures_util::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

pub(crate) const CHECKPOINTS_TABLE: &str = "checkpoints_v1";

// ── Schema ────────────────────────────────────────────────────────────────────

fn checkpoints_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("ts_ms", DataType::Int64, false),
        Field::new("workspace_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, true),
        Field::new("from_ts_ms", DataType::Int64, false),
        Field::new("to_ts_ms", DataType::Int64, false),
        Field::new(
            "capsule_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("capsule_count", DataType::Int32, false),
        Field::new("narrative", DataType::Utf8, false),
        Field::new("model_used", DataType::Utf8, false),
        // trigger: "new_session_opencode" | "new_session_claude" | "manual"
        Field::new("trigger", DataType::Utf8, false),
    ]))
}

// ── Table bootstrap ───────────────────────────────────────────────────────────

pub(crate) async fn ensure_checkpoints_table(db: &Connection) -> anyhow::Result<lancedb::Table> {
    match db.open_table(CHECKPOINTS_TABLE).execute().await {
        Ok(t) => Ok(t),
        Err(_) => {
            let schema = checkpoints_schema();
            let empty = RecordBatchIterator::new(
                vec![],
                schema.clone(),
            );
            let t = db
                .create_table(CHECKPOINTS_TABLE, Box::new(empty))
                .execute()
                .await
                .context("failed to create checkpoints_v1 table")?;
            // Index on ts_ms for range queries
            t.create_index(&["ts_ms"], lancedb::index::Index::Auto)
                .execute()
                .await
                .ok();
            Ok(t)
        }
    }
}

// ── Row type ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CheckpointRow {
    pub id: String,
    pub ts_ms: i64,
    pub workspace_id: String,
    pub session_id: Option<String>,
    pub from_ts_ms: i64,
    pub to_ts_ms: i64,
    pub capsule_ids: Vec<String>,
    pub capsule_count: i32,
    pub narrative: String,
    pub model_used: String,
    pub trigger: String,
}

// ── Insert ────────────────────────────────────────────────────────────────────

pub(crate) async fn insert_checkpoint(
    db: &Connection,
    row: &CheckpointRow,
) -> anyhow::Result<()> {
    let table = ensure_checkpoints_table(db).await?;

    let schema = checkpoints_schema();
    let n = 1usize;

    let id_arr = Arc::new(StringArray::from(vec![row.id.as_str()]));
    let ts_arr = Arc::new(Int64Array::from(vec![row.ts_ms]));
    let ws_arr = Arc::new(StringArray::from(vec![row.workspace_id.as_str()]));
    let session_arr = Arc::new(StringArray::from(vec![row.session_id.as_deref()]));
    let from_arr = Arc::new(Int64Array::from(vec![row.from_ts_ms]));
    let to_arr = Arc::new(Int64Array::from(vec![row.to_ts_ms]));

    // capsule_ids as List<Utf8>
    let mut ids_builder = ListBuilder::new(StringBuilder::new());
    for cid in &row.capsule_ids {
        ids_builder.values().append_value(cid);
    }
    ids_builder.append(true);
    let capsule_ids_arr = Arc::new(ids_builder.finish());

    let count_arr = Arc::new(Int32Array::from(vec![row.capsule_count]));
    let narrative_arr = Arc::new(StringArray::from(vec![row.narrative.as_str()]));
    let model_arr = Arc::new(StringArray::from(vec![row.model_used.as_str()]));
    let trigger_arr = Arc::new(StringArray::from(vec![row.trigger.as_str()]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            id_arr,
            ts_arr,
            ws_arr,
            session_arr,
            from_arr,
            to_arr,
            capsule_ids_arr,
            count_arr,
            narrative_arr,
            model_arr,
            trigger_arr,
        ],
    )
    .context("failed to build checkpoint record batch")?;

    let iter = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table
        .add(Box::new(iter))
        .execute()
        .await
        .context("failed to insert checkpoint row")?;

    let _ = n; // suppress unused warning
    Ok(())
}

// ── Query helpers ─────────────────────────────────────────────────────────────

/// Return the most recent checkpoints for a workspace, ordered newest-first.
pub(crate) async fn get_recent_checkpoints(
    db: &Connection,
    workspace_id: &str,
    limit: usize,
) -> anyhow::Result<Vec<CheckpointRow>> {
    let table = ensure_checkpoints_table(db).await?;
    let ws_escaped = workspace_id.replace('\'', "\\'");
    let filter = format!("workspace_id = '{ws_escaped}'");

    let batches: Vec<RecordBatch> = table
        .query()
        .only_if(&filter)
        .limit(limit * 3) // over-fetch then sort
        .execute()
        .await
        .context("checkpoint query failed")?
        .try_collect()
        .await
        .context("checkpoint collect failed")?;

    let mut rows = batches_to_rows(&batches)?;
    rows.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
    rows.truncate(limit);
    Ok(rows)
}

/// Return checkpoints whose [from_ts_ms, to_ts_ms] window overlaps [since, until].
/// Used by pr-comment to find relevant session stories for a commit range.
#[allow(dead_code)]
pub(crate) async fn get_checkpoints_in_range(
    db: &Connection,
    workspace_id: &str,
    since_ms: i64,
    until_ms: i64,
) -> anyhow::Result<Vec<CheckpointRow>> {
    let table = ensure_checkpoints_table(db).await?;
    let ws_escaped = workspace_id.replace('\'', "\\'");
    // Overlap condition: checkpoint starts before until AND ends after since
    let filter = format!(
        "workspace_id = '{ws_escaped}' AND CAST(from_ts_ms AS BIGINT) <= CAST({until_ms} AS BIGINT) AND CAST(to_ts_ms AS BIGINT) >= CAST({since_ms} AS BIGINT)"
    );

    let batches: Vec<RecordBatch> = table
        .query()
        .only_if(&filter)
        .execute()
        .await
        .context("checkpoint range query failed")?
        .try_collect()
        .await
        .context("checkpoint range collect failed")?;

    let mut rows = batches_to_rows(&batches)?;
    rows.sort_by(|a, b| a.from_ts_ms.cmp(&b.from_ts_ms));
    Ok(rows)
}

/// Check whether an up-to-date checkpoint already exists for this workspace so
/// we can skip redundant generation.
///
/// "Up to date" means: there's a checkpoint whose `to_ts_ms` is within 60s of
/// `latest_capsule_ts` AND whose `capsule_count` is within 2 of
/// `current_capsule_count`. This guards against crash-restart loops and
/// rapid session cycling.
pub(crate) async fn checkpoint_is_current(
    db: &Connection,
    workspace_id: &str,
    latest_capsule_ts: i64,
    current_capsule_count: i32,
) -> bool {
    let threshold_ts = latest_capsule_ts - 60_000; // 60s window
    let ws_escaped = workspace_id.replace('\'', "\\'");
    let filter = format!(
        "workspace_id = '{ws_escaped}' AND CAST(to_ts_ms AS BIGINT) >= CAST({threshold_ts} AS BIGINT)"
    );

    let table = match ensure_checkpoints_table(db).await {
        Ok(t) => t,
        Err(_) => return false,
    };

    let stream = match table.query().only_if(&filter).limit(5).execute().await {
        Ok(s) => s,
        Err(_) => return false,
    };
    let batches: Vec<RecordBatch> = match stream.try_collect().await {
        Ok(b) => b,
        Err(_) => return false,
    };

    let rows = match batches_to_rows(&batches) {
        Ok(r) => r,
        Err(_) => return false,
    };

    rows.iter().any(|r| {
        (r.capsule_count - current_capsule_count).abs() <= 2
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn col_str<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a arrow_array::StringArray> {
    let idx = batch.schema().index_of(name).ok()?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
}

fn col_i64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a arrow_array::Int64Array> {
    let idx = batch.schema().index_of(name).ok()?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
}

fn col_i32<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a arrow_array::Int32Array> {
    let idx = batch.schema().index_of(name).ok()?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
}

fn col_list<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a arrow_array::ListArray> {
    let idx = batch.schema().index_of(name).ok()?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow_array::ListArray>()
}

fn batches_to_rows(batches: &[RecordBatch]) -> anyhow::Result<Vec<CheckpointRow>> {
    let mut rows = Vec::new();
    for batch in batches {
        let id_col = col_str(batch, "id");
        let ts_col = col_i64(batch, "ts_ms");
        let ws_col = col_str(batch, "workspace_id");
        let session_col = col_str(batch, "session_id");
        let from_col = col_i64(batch, "from_ts_ms");
        let to_col = col_i64(batch, "to_ts_ms");
        let ids_col = col_list(batch, "capsule_ids");
        let count_col = col_i32(batch, "capsule_count");
        let narrative_col = col_str(batch, "narrative");
        let model_col = col_str(batch, "model_used");
        let trigger_col = col_str(batch, "trigger");

        for row in 0..batch.num_rows() {
            let id = id_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()))
                .unwrap_or_default();
            let ts_ms = ts_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row)))
                .unwrap_or(0);
            let workspace_id = ws_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()))
                .unwrap_or_default();
            let session_id = session_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()));
            let from_ts_ms = from_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row)))
                .unwrap_or(0);
            let to_ts_ms = to_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row)))
                .unwrap_or(0);
            let capsule_count = count_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row)))
                .unwrap_or(0);
            let narrative = narrative_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()))
                .unwrap_or_default();
            let model_used = model_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()))
                .unwrap_or_default();
            let trigger = trigger_col
                .and_then(|c| (!c.is_null(row)).then(|| c.value(row).to_string()))
                .unwrap_or_default();

            // Decode capsule_ids list
            let capsule_ids: Vec<String> = ids_col
                .and_then(|c| {
                    if c.is_null(row) {
                        return None;
                    }
                    let list = c.value(row);
                    let str_arr = list
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()?;
                    Some(
                        (0..str_arr.len())
                            .filter_map(|i| {
                                (!str_arr.is_null(i)).then(|| str_arr.value(i).to_string())
                            })
                            .collect(),
                    )
                })
                .unwrap_or_default();

            rows.push(CheckpointRow {
                id,
                ts_ms,
                workspace_id,
                session_id,
                from_ts_ms,
                to_ts_ms,
                capsule_ids,
                capsule_count,
                narrative,
                model_used,
                trigger,
            });
        }
    }
    Ok(rows)
}

// ── Generation ────────────────────────────────────────────────────────────────

/// LLM-generated narrative for a session checkpoint.
#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CheckpointNarrativeOutput {
    /// The story segment (max ~300 words).
    pub narrative: String,
}

/// Build the LLM prompt context string from a set of capsule hits and any
/// associated git capsules (source="git") that landed in the same time window.
pub(crate) fn build_checkpoint_context(
    session_id: Option<&str>,
    capsules: &[crate::CapsuleHit],
    git_capsules: &[crate::CapsuleHit],
) -> String {
    use chrono::{SecondsFormat, TimeZone};
    let fmt_ts = |ts_ms: i64| -> String {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
            .unwrap_or_else(|| ts_ms.to_string())
    };

    let mut ctx = String::new();

    if let Some(sid) = session_id {
        ctx.push_str(&format!("Session: {sid}\n\n"));
    }

    if !capsules.is_empty() {
        ctx.push_str(&format!(
            "Conversation capsules ({}):\n\n",
            capsules.len()
        ));
        for (i, hit) in capsules.iter().enumerate() {
            let cap = &hit.capsule;
            ctx.push_str(&format!("#{} [{}]\n", i + 1, fmt_ts(hit.ts_ms)));
            if !cap.category.is_empty() {
                ctx.push_str(&format!("category: {}\n", cap.category));
            }
            if !cap.intent.trim().is_empty() {
                ctx.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
            }
            if !cap.decision.trim().is_empty() {
                ctx.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
            }
            if !cap.rationale.trim().is_empty() {
                ctx.push_str(&format!("rationale: {}\n", cap.rationale.replace('\n', " ")));
            }
            if cap.failure_mode != crate::types::FailureMode::None {
                let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
                ctx.push_str(&format!("failure_mode: {}\n", fm.trim_matches('"')));
            }
            if !cap.symbols.is_empty() {
                let syms = cap.symbols.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
                ctx.push_str(&format!("symbols: {syms}\n"));
            }
            ctx.push('\n');
        }
    }

    if !git_capsules.is_empty() {
        ctx.push_str(&format!("Commits that landed during this session ({}):\n\n", git_capsules.len()));
        for hit in git_capsules {
            let cap = &hit.capsule;
            ctx.push_str(&format!("[{}] ", fmt_ts(hit.ts_ms)));
            if !hit.meta.request_path.is_empty() {
                ctx.push_str(&format!("commit:{} ", hit.meta.request_path));
            }
            if !cap.intent.trim().is_empty() {
                ctx.push_str(&cap.intent.replace('\n', " "));
            }
            ctx.push('\n');
        }
        ctx.push('\n');
    }

    ctx
}

const CHECKPOINT_PREAMBLE: &str = "\
You are recording the story of a work session for future use by humans and AI agents.

Given a set of conversation capsules (intent, decision, rationale, failure_mode, symbols) \
and any git commits that landed during this session, produce a compact structured story segment.

Format your response as:
WHAT WAS WORKED ON: (1-2 sentences summarising the focus)
KEY DECISIONS: (2-4 bullet points, each with the decision and the rationale behind it; \
backtick any symbol names or file paths)
GOTCHAS / FAILURES: (only if there are capsules with a non-None failure_mode — be concrete, \
not generic; omit this section entirely if there are none)
COMMITS: (brief one-line summary per commit that landed; omit this section if there are none)

Rules:
- Max 300 words total.
- Anchor every claim to evidence in the capsules. Never invent facts.
- Write for a developer or AI agent reading this later to understand WHY the code exists.
- Return the full formatted text in the `narrative` field.";

/// Generate a checkpoint narrative via LLM and return the text.
pub(crate) async fn generate_checkpoint_narrative(
    llm_model_override: Option<&str>,
    session_id: Option<&str>,
    capsules: &[crate::CapsuleHit],
    git_capsules: &[crate::CapsuleHit],
) -> anyhow::Result<String> {
    let context = build_checkpoint_context(session_id, capsules, git_capsules);
    let result = crate::llm_extract::<CheckpointNarrativeOutput>(
        llm_model_override,
        CHECKPOINT_PREAMBLE,
        &context,
    )
    .await?;
    Ok(result.narrative)
}

/// Full pipeline: fetch unchecked capsules for a session, check dedup, generate
/// narrative, and store. Returns the new CheckpointRow on success.
///
/// `trigger` should be one of: "new_session_opencode", "new_session_claude", "manual".
/// Reason a checkpoint was skipped — returned so callers can give specific feedback.
#[derive(Debug)]
pub enum CheckpointSkipReason {
    TooFewCapsules { found: usize, minimum: usize },
    AlreadyCurrent,
    NoConversationalCapsules,
}

pub(crate) async fn maybe_create_checkpoint(
    ws: &crate::WorkspacePaths,
    session_id: Option<&str>,
    trigger: &str,
    llm_model_override: Option<&str>,
) -> anyhow::Result<Result<CheckpointRow, CheckpointSkipReason>> {
    // Open (or create) the LanceDB database
    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await
        .context("failed to open lancedb for checkpoint")?;

    // Fetch the capsules for this session (or all recent if session_id is None).
    let all_capsules = fetch_session_capsules(&db, &ws.id, session_id).await?;

    const MIN_CAPSULES: usize = 3;
    if all_capsules.len() < MIN_CAPSULES {
        tracing::debug!(
            "checkpoint: skipping — only {} capsules for session {:?}",
            all_capsules.len(),
            session_id
        );
        return Ok(Err(CheckpointSkipReason::TooFewCapsules {
            found: all_capsules.len(),
            minimum: MIN_CAPSULES,
        }));
    }

    let capsule_count = all_capsules.len() as i32;
    let from_ts_ms = all_capsules.iter().map(|h| h.ts_ms).min().unwrap_or(0);
    let to_ts_ms = all_capsules.iter().map(|h| h.ts_ms).max().unwrap_or(0);

    // Dedup check: skip if a recent checkpoint already covers this work.
    // Manual trigger ("manual") bypasses this — if the user explicitly asks
    // for a checkpoint, they get one regardless.
    if trigger != "manual" && checkpoint_is_current(&db, &ws.id, to_ts_ms, capsule_count).await {
        tracing::debug!(
            "checkpoint: skipping — current checkpoint already covers session {:?}",
            session_id
        );
        return Ok(Err(CheckpointSkipReason::AlreadyCurrent));
    }

    // Split into conversational and git capsules
    let conv_capsules: Vec<_> = all_capsules
        .iter()
        .filter(|h| h.meta.source != "git")
        .cloned()
        .collect();
    let git_capsules: Vec<_> = all_capsules
        .iter()
        .filter(|h| h.meta.source == "git")
        .cloned()
        .collect();

    if conv_capsules.is_empty() {
        tracing::debug!("checkpoint: skipping — no conversational capsules");
        return Ok(Err(CheckpointSkipReason::NoConversationalCapsules));
    }

    // Generate narrative
    let narrative = generate_checkpoint_narrative(
        llm_model_override,
        session_id,
        &conv_capsules,
        &git_capsules,
    )
    .await?;

    // Determine model name for audit
    let model_used = if let Some(m) = llm_model_override {
        m.to_string()
    } else if let Some(cfg) = crate::llm::get_llm_config() {
        match cfg {
            crate::config::LlmConfig::Openai { model, .. } => model,
            crate::config::LlmConfig::Anthropic { model, .. } => model,
            crate::config::LlmConfig::Ollama { model, .. } => model,
            crate::config::LlmConfig::Custom { model, .. } => model,
        }
    } else {
        "gpt-4o-mini".to_string()
    };

    let capsule_ids: Vec<String> = all_capsules.iter().map(|h| h.id.clone()).collect();

    let row = CheckpointRow {
        id: Uuid::new_v4().to_string(),
        ts_ms: crate::workspace::now_ms(),
        workspace_id: ws.id.clone(),
        session_id: session_id.map(|s| s.to_string()),
        from_ts_ms,
        to_ts_ms,
        capsule_ids,
        capsule_count,
        narrative,
        model_used,
        trigger: trigger.to_string(),
    };

    insert_checkpoint(&db, &row).await?;

    tracing::info!(
        "checkpoint: created for session {:?} ({} capsules, trigger={})",
        session_id,
        capsule_count,
        trigger
    );

    Ok(Ok(row))
}

/// Fetch all capsules (conversational + git) for a given session within a
/// workspace. If session_id is None, fetches all recent capsules (unscoped).
async fn fetch_session_capsules(
    db: &Connection,
    workspace_id: &str,
    session_id: Option<&str>,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    // We use the capsules_v4 table directly via LanceDB query.
    let table = crate::storage::open_capsules_table(db).await?;

    let filter = match session_id {
        Some(sid) => {
            let sid_escaped = sid.replace('\'', "\\'");
            // Include git capsules by time proximity: we'll filter them after
            // fetching by doing a two-pass query below.
            format!("agent_session_id = '{sid_escaped}'")
        }
        None => {
            // No session: fetch the most recent 200 capsules
            String::new()
        }
    };

    let mut batches_query = table.query();
    if !filter.is_empty() {
        batches_query = batches_query.only_if(&filter);
    }
    let batches: Vec<RecordBatch> = batches_query
        .limit(500)
        .execute()
        .await
        .context("session capsule query failed")?
        .try_collect()
        .await
        .context("session capsule collect failed")?;

    let mut hits = crate::storage::record_batches_to_hits(&batches, workspace_id)?;

    // If we have a session_id, also fetch git capsules in the session's time window
    if session_id.is_some() && !hits.is_empty() {
        let from_ts = hits.iter().map(|h| h.ts_ms).min().unwrap_or(0);
        let to_ts = hits.iter().map(|h| h.ts_ms).max().unwrap_or(0);
        let git_filter = format!(
            "source = 'git' AND CAST(ts_ms AS BIGINT) >= CAST({from_ts} AS BIGINT) AND CAST(ts_ms AS BIGINT) <= CAST({to_ts} AS BIGINT)"
        );
        if let Ok(stream) = table.query().only_if(&git_filter).limit(50).execute().await {
            if let Ok(git_batches) = stream.try_collect::<Vec<RecordBatch>>().await {
                if let Ok(git_hits) =
                    crate::storage::record_batches_to_hits(&git_batches, workspace_id)
                {
                    hits.extend(git_hits);
                }
            }
        }
    }

    // Deduplicate by id
    let mut by_id: std::collections::HashMap<String, crate::CapsuleHit> =
        std::collections::HashMap::new();
    for h in hits {
        by_id.entry(h.id.clone()).or_insert(h);
    }

    let mut result: Vec<_> = by_id.into_values().collect();
    result.sort_by_key(|h| h.ts_ms);
    Ok(result)
}
