use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use lancedb::query::{ExecutableQuery, QueryBase};
use std::time::Duration;

pub async fn run(
    mode: crate::cli::ReflectMode,
    session: Option<String>,
    since: Option<String>,
    llm_model: Option<String>,
    output: OutputFormat,
    path: String,
) -> anyhow::Result<()> {
    let dir_path = std::path::Path::new(&path);
    let ws = crate::workspace::get_or_create_workspace_paths(dir_path)?;

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Loading capsules...");
        Some(pb)
    } else {
        None
    };

    // Parse time filters
    let since_ms: Option<i64> = if let Some(ref s) = since {
        crate::util::parse_time_filter(s)?
    } else {
        None
    };

    // Fetch conversational capsules for the session/period
    let capsules = fetch_reflect_capsules(&ws, session.as_deref(), since_ms).await?;

    if capsules.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No capsules found for this workspace.");
        println!("Record a session first: run `unlost record` or use a supported agent.");
        return Ok(());
    }

    // Filter to only capsules that have TurnEval data (produced after v0.13)
    let eval_capsules: Vec<_> = capsules
        .iter()
        .filter(|h| h.turn_eval.is_some())
        .cloned()
        .collect();

    if eval_capsules.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No turn evaluation data found yet.");
        println!(
            "TurnEval is collected on new sessions going forward. \
             Try again after recording a new session."
        );
        return Ok(());
    }

    let model_name = if let Some(ref m) = llm_model {
        m.clone()
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

    if let Some(pb) = spinner.as_ref() {
        pb.set_message(format!("Reflecting with {} ({} turns)...", model_name, eval_capsules.len()));
    }

    let narrative = crate::narrative::llm_reflect_narrative(
        llm_model.as_deref(),
        mode,
        &eval_capsules,
        session.as_deref(),
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }

    let out = crate::narrative::render_narrative(output, &narrative);
    println!("{}", out);
    println!();

    Ok(())
}

/// Fetch conversational capsules for reflect, filtered by session or time window.
async fn fetch_reflect_capsules(
    ws: &crate::WorkspacePaths,
    session_id: Option<&str>,
    since_ms: Option<i64>,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let table = match crate::storage::open_capsules_table(&db).await {
        Ok(t) => t,
        Err(_) => return Ok(vec![]),
    };

    use futures_util::TryStreamExt;
    use arrow_array::RecordBatch;

    let mut q = table.query();

    let mut filters: Vec<String> = Vec::new();

    // Only conversational capsules (not git, not replay)
    filters.push("source != 'git'".to_string());

    if let Some(sid) = session_id {
        let sid_esc = sid.replace('\'', "\\'");
        filters.push(format!("agent_session_id = '{sid_esc}'"));
    }

    if let Some(ms) = since_ms {
        filters.push(format!(
            "CAST(ts_ms AS BIGINT) >= CAST({ms} AS BIGINT)"
        ));
    }

    let combined = filters.join(" AND ");
    q = q.only_if(&combined);

    let batches: Vec<RecordBatch> = q
        .limit(500)
        .execute()
        .await?
        .try_collect()
        .await?;

    let mut hits = crate::storage::record_batches_to_hits(&batches, &ws.id)?;
    hits.sort_by_key(|h| h.ts_ms);
    Ok(hits)
}
