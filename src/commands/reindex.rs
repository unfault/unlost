use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Deserialize)]
struct JsonCapsule {
    #[serde(default)]
    agent_session_id: String,
    ts_ms: i64,
    conn_id: u64,
    exchange_seq: u64,
    request_path: String,
    source: String,
    #[serde(default)]
    usage: Option<Usage>,
    capsule: Caps,
    #[serde(default)]
    turn_eval: Option<crate::types::TurnEval>,
    #[serde(default)]
    head_sha: Option<String>,
    #[serde(default)]
    commit_sha: Option<String>,
}

#[derive(Deserialize, Default)]
struct Usage {
    provider_id: Option<String>,
    model_id: Option<String>,
    cost: Option<f64>,
    tokens: Option<Tokens>,
}

#[derive(Deserialize, Default)]
struct Tokens {
    input: Option<i64>,
    output: Option<i64>,
    reasoning: Option<i64>,
    cache: Option<Cache>,
}

#[derive(Deserialize, Default)]
struct Cache {
    read: Option<i64>,
    write: Option<i64>,
}

#[derive(Deserialize)]
struct Caps {
    category: String,
    intent: String,
    decision: String,
    rationale: String,
    next_steps: Vec<String>,
    symbols: Vec<String>,
    #[serde(default)]
    failure_mode: Option<String>,
    #[serde(default)]
    failure_signals: Option<String>,
}

pub async fn run(path: String, yes: bool) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(Path::new(&path))?;

    let jsonl_path = &ws.capsules_jsonl;
    if !jsonl_path.exists() {
        anyhow::bail!("capsules.jsonl not found at {}", jsonl_path.display());
    }

    let capsules_count = count_capsules(jsonl_path).await?;
    println!(
        "Found {} capsules in {}",
        capsules_count,
        jsonl_path.display()
    );

    if !yes {
        let mut stdout = tokio::io::stdout();
        stdout
            .write_all(b"Reindex will delete LanceDB and rebuild from JSONL. Continue? [y/N] ")
            .await?;
        stdout.flush().await?;

        let mut input = String::new();
        let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
        stdin.read_line(&mut input).await?;
        if !input.trim().eq_ignore_ascii_case("y") && !input.trim().eq_ignore_ascii_case("yes") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    println!("Deleting LanceDB data...");
    if ws.db_dir.exists() {
        std::fs::remove_dir_all(&ws.db_dir)?;
    }

    // Number of capsules to embed + write per LanceDB flush.
    // Larger batches = fewer ONNX calls and fewer Parquet fragments.
    const BATCH_SIZE: usize = 64;

    let mut stdout = tokio::io::stdout();
    let mut count = 0usize;

    println!("Rebuilding LanceDB from JSONL...");
    println!("Loading embedding model...");

    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await
        .context("failed to connect to lancedb")?;
    let embedder =
        crate::embed::load_embedder(crate::constants::DEFAULT_EMBED_MODEL, None, false).await?;

    // Open/create the table once so all batches share the same handle
    // and there's no risk of schema mismatch from repeated ensure_capsules_table calls.
    let table = crate::storage::ensure_capsules_table(&db).await?;

    let file = File::open(&jsonl_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Print initial progress now that we're actually about to start indexing
    let msg = format!("\r  [  0%] 0/{} capsules", capsules_count);
    stdout.write_all(msg.as_bytes()).await?;
    stdout.flush().await?;

    // Pending batch accumulators
    let mut batch_texts: Vec<String> = Vec::with_capacity(BATCH_SIZE);
    let mut batch_rows: Vec<BatchRow> = Vec::with_capacity(BATCH_SIZE);

    // Rolling history buffer for coach score computation (newest first, max 8).
    // Built from capsules processed so far so compute_coach_scores has lookahead.
    let mut reindex_history: std::collections::VecDeque<crate::CapsuleHit> =
        std::collections::VecDeque::with_capacity(9);

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let capsule: JsonCapsule = serde_json::from_str(&line)?;

        let meta = crate::ResponseMeta {
            source: capsule.source,
            upstream_host: "companion".to_string(),
            request_path: capsule.request_path,
            http_status: 200,
            agent_session_id: Some(capsule.agent_session_id),
            usage: capsule.usage.map(|u| crate::types::UsageMeta {
                provider_id: u.provider_id,
                model_id: u.model_id,
                cost: u.cost,
                tokens_input: u.tokens.as_ref().and_then(|t| t.input),
                tokens_output: u.tokens.as_ref().and_then(|t| t.output),
                tokens_reasoning: u.tokens.as_ref().and_then(|t| t.reasoning),
                tokens_cache_read: u
                    .tokens
                    .as_ref()
                    .and_then(|t| t.cache.as_ref().and_then(|c| c.read)),
                tokens_cache_write: u
                    .tokens
                    .as_ref()
                    .and_then(|t| t.cache.as_ref().and_then(|c| c.write)),
            }),
        };

        let failure_mode = match capsule.capsule.failure_mode.as_deref() {
            Some("drift") => crate::types::FailureMode::Drift,
            Some("rediscovery") => crate::types::FailureMode::Rediscovery,
            Some("decision_conflict") => crate::types::FailureMode::DecisionConflict,
            Some("retry_spiral") => crate::types::FailureMode::RetrySpiral,
            Some("false_progress") => crate::types::FailureMode::FalseProgress,
            Some("unbounded_horizon") => crate::types::FailureMode::UnboundedHorizon,
            _ => crate::types::FailureMode::None,
        };

        let intent_capsule = crate::IntentCapsule {
            category: capsule.capsule.category,
            intent: capsule.capsule.intent,
            decision: capsule.capsule.decision,
            rationale: capsule.capsule.rationale,
            next_steps: capsule.capsule.next_steps,
            symbols: capsule.capsule.symbols.clone(),
            user_symbols: vec![],
            failure_mode,
            failure_signals: capsule.capsule.failure_signals,
            extraction_mode: crate::types::ExtractionMode::None,
            questions: vec![],
        };

        // Determine TurnEval for this capsule:
        // (a) If the JSONL already has turn_eval (post-v0.13), use it directly.
        // (b) Otherwise compute the coach dimensions from accumulated history.
        //     Tune channels (governor EMA) cannot be recovered and stay at 0.
        let turn_eval = match capsule.turn_eval {
            Some(te) => Some(te),
            None => {
                // Only compute for conversational capsules (not git/replay sources)
                // since coach scores only make sense for human+agent turns.
                if meta.source != "git" && meta.source != "changelog" {
                    let history_slice: Vec<crate::CapsuleHit> =
                        reindex_history.iter().cloned().collect();
                    let usage_ref: Option<crate::types::UsageMeta> = meta.usage.clone();
                    let coach_input = crate::governor::CoachInput {
                        capsule: &intent_capsule,
                        history: &history_slice,
                        exchange_text: "", // exchange text not stored in JSONL
                        usage: usage_ref.as_ref(),
                        session_turn_count: reindex_history.len() + 1,
                    };
                    let coach = crate::governor::compute_coach_scores(&coach_input);
                    // Build a partial TurnEval: coach dimensions only, tune channels = 0
                    let flags = crate::governor::compute_flags(
                        &crate::types::SymptomChannels::default(),
                        0.0,
                        &coach,
                        reindex_history.len() + 1,
                    );
                    let mut evidence = coach.evidence.clone();
                    evidence.push("reindexed: tune channels unavailable".to_string());
                    Some(crate::types::TurnEval {
                        version: "v1-reindex".to_string(),
                        // Tune channels: not recoverable during reindex
                        repetition: 0.0,
                        novelty_collapse: 0.0,
                        semantic_stall: 0.0,
                        effort_spike: 0.0,
                        alignment_debt: 0.0,
                        path_hallucination: 0.0,
                        grounding_stall: 0.0,
                        instruction_staticness: 0.0,
                        logic_churn: 0.0,
                        fluency: 0.0,
                        trajectory_intensity: 0.0,
                        trajectory_state: crate::types::TrajectoryState::Stable,
                        // Coach dimensions: computed from capsule content + history
                        clarity: coach.clarity,
                        context_freshness: coach.context_freshness,
                        verification_rigor: coach.verification_rigor,
                        decision_progress: coach.decision_progress,
                        scope_discipline: coach.scope_discipline,
                        cost_acceleration: coach.cost_acceleration,
                        flags,
                        outcome_hint: "unclear".to_string(),
                        evidence,
                    })
                } else {
                    None
                }
            }
        };

        // Update rolling history (newest first, max 8 entries)
        let history_hit = crate::CapsuleHit {
            id: format!("reindex-{}-{}", capsule.ts_ms, capsule.conn_id),
            ts_ms: capsule.ts_ms,
            conn_id: capsule.conn_id as i64,
            exchange_seq: capsule.exchange_seq as i64,
            distance: 0.0,
            capsule: intent_capsule.clone(),
            meta: meta.clone(),
            user_emotion: None,
            assistant_emotion: None,
            head_sha: capsule.head_sha.clone(),
            commit_sha: capsule.commit_sha.clone(),
            turn_eval: turn_eval.clone(),
        };
        reindex_history.push_front(history_hit);
        if reindex_history.len() > 8 {
            reindex_history.pop_back();
        }

        let embed_text = crate::storage::capsule_embed_text_with_prior(&intent_capsule, None);
        batch_texts.push(embed_text);
        batch_rows.push(BatchRow {
            conn_id: capsule.conn_id,
            exchange_seq: capsule.exchange_seq,
            ts_ms: capsule.ts_ms,
            meta,
            capsule: intent_capsule,
            head_sha: capsule.head_sha,
            commit_sha: capsule.commit_sha,
            turn_eval,
        });

        if batch_rows.len() >= BATCH_SIZE {
            count += flush_batch(&table, &embedder, &mut batch_texts, &mut batch_rows).await?;
            let pct = count * 100 / capsules_count;
            let msg = format!("\r  [{:3}%] {}/{} capsules", pct, count, capsules_count);
            stdout.write_all(msg.as_bytes()).await?;
            stdout.flush().await?;
        }
    }

    // Flush any remaining capsules
    if !batch_rows.is_empty() {
        count += flush_batch(&table, &embedder, &mut batch_texts, &mut batch_rows).await?;
        let pct = count * 100 / capsules_count;
        let msg = format!("\r  [{:3}%] {}/{} capsules", pct, count, capsules_count);
        stdout.write_all(msg.as_bytes()).await?;
        stdout.flush().await?;
    }

    // Move to a new line after the in-place progress
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;

    println!("Done! Reindexed {} capsules.", count);
    Ok(())
}

struct BatchRow {
    conn_id: u64,
    exchange_seq: u64,
    ts_ms: i64,
    meta: crate::ResponseMeta,
    capsule: crate::IntentCapsule,
    head_sha: Option<String>,
    commit_sha: Option<String>,
    turn_eval: Option<crate::types::TurnEval>,
}

/// Embed `batch_texts` in one ONNX call, pair with `batch_rows`,
/// write to LanceDB as a single RecordBatch, then clear both vecs.
/// Returns the number of rows written.
async fn flush_batch(
    table: &lancedb::Table,
    embedder: &crate::embed::Embedder,
    batch_texts: &mut Vec<String>,
    batch_rows: &mut Vec<BatchRow>,
) -> anyhow::Result<usize> {
    let n = batch_rows.len();
    let embeddings = crate::embed::embed_texts_batch(embedder, std::mem::take(batch_texts)).await?;
    if embeddings.len() != n {
        anyhow::bail!(
            "embed batch size mismatch: expected {}, got {}",
            n,
            embeddings.len()
        );
    }

    let rows: Vec<crate::storage::CapsuleRow> = std::mem::take(batch_rows)
        .into_iter()
        .zip(embeddings)
        .map(|(r, emb)| crate::storage::CapsuleRow {
            conn_id: r.conn_id,
            exchange_seq: r.exchange_seq,
            ts_ms: r.ts_ms,
            meta: r.meta,
            capsule: r.capsule,
            embedding: emb,
            head_sha: r.head_sha,
            commit_sha: r.commit_sha,
            turn_eval: r.turn_eval,
        })
        .collect();

    crate::storage::insert_capsule_batch(table, &rows).await?;
    Ok(n)
}

async fn count_capsules(path: &Path) -> anyhow::Result<usize> {
    let file = File::open(path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut count = 0;
    while let Some(line) = lines.next_line().await? {
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}
