use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

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

        let mut input = Vec::new();
        let mut stdin = tokio::io::stdin();
        stdin.read_to_end(&mut input).await?;
        let input_str = String::from_utf8(input)?;
        if !input_str.trim().eq_ignore_ascii_case("y")
            && !input_str.trim().eq_ignore_ascii_case("yes")
        {
            println!("Cancelled.");
            return Ok(());
        }
    }

    println!("Deleting LanceDB data...");
    let lancedb_path = ws.db_dir.join("lancedb");
    if lancedb_path.exists() {
        std::fs::remove_dir_all(&lancedb_path)?;
    }

    println!("Rebuilding LanceDB from JSONL...");

    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await
        .context("failed to connect to lancedb")?;
    let embedder =
        crate::embed::load_embedder(crate::constants::DEFAULT_EMBED_MODEL, None, false).await?;

    let file = File::open(&jsonl_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut count = 0;
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
            user_symbols: vec![], // Can't recover from JSONL easily without parsing
            failure_mode,
            failure_signals: capsule.capsule.failure_signals,
            extraction_mode: crate::types::ExtractionMode::None,
        };

        crate::storage::insert_capsule_row(
            &db,
            &embedder,
            capsule.conn_id,
            capsule.exchange_seq,
            capsule.ts_ms,
            &meta,
            None,
            None,
            &intent_capsule,
            None,
        )
        .await?;

        count += 1;
        if count % 100 == 0 {
            println!("  Reindexed {} / {} capsules...", count, capsules_count);
        }
    }

    println!("Done! Reindexed {} capsules.", count);
    Ok(())
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
