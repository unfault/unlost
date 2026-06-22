use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Score a capsule for exploration context.
///
/// Prioritises recorded rationale, cross-session recurrence (recurring knowledge),
/// and failure modes (recorded pain = future risk signal). Mirrors `brief`'s
/// scoring philosophy: importance over recency.
fn score_hit_for_exploration(
    h: &crate::CapsuleHit,
    symbol_session_counts: &HashMap<String, usize>,
) -> f32 {
    let mut score = 0.0f32;

    // Recorded pain is a forward risk signal
    if h.capsule.failure_mode != crate::types::FailureMode::None {
        score += 3.0;
    }
    if matches!(
        h.capsule.failure_mode,
        crate::types::FailureMode::RetrySpiral | crate::types::FailureMode::DecisionConflict
    ) {
        score += 2.0;
    }

    // Someone recorded *why* — high signal for evaluating paths
    if !h.capsule.rationale.trim().is_empty() {
        score += 2.0;
    }

    // An explicit choice was made
    if !h.capsule.decision.trim().is_empty() {
        score += 1.0;
    }

    // Recurring symbols across sessions = load-bearing knowledge
    for sym in &h.capsule.symbols {
        if symbol_session_counts.get(sym).copied().unwrap_or(0) >= 2 {
            score += 1.0;
        }
    }

    score
}

fn select_hits_for_exploration(
    sem_hits: Vec<crate::CapsuleHit>,
    broad_hits: Vec<crate::CapsuleHit>,
    limit: usize,
) -> Vec<crate::CapsuleHit> {
    // Merge and dedup by id, preferring the most recent version
    let mut by_id: HashMap<String, crate::CapsuleHit> = HashMap::new();
    for h in sem_hits.into_iter().chain(broad_hits) {
        match by_id.get(&h.id) {
            Some(existing) if existing.ts_ms >= h.ts_ms => {}
            _ => {
                by_id.insert(h.id.clone(), h);
            }
        }
    }
    let hits: Vec<crate::CapsuleHit> = by_id.into_values().collect();

    if hits.is_empty() {
        return hits;
    }

    // Build cross-session symbol counts
    let mut symbol_sessions: HashMap<String, HashSet<String>> = HashMap::new();
    for h in &hits {
        let session_key = h
            .meta
            .agent_session_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("ses:{s}"))
            .unwrap_or_else(|| format!("conn:{}", h.conn_id));
        for sym in &h.capsule.symbols {
            symbol_sessions
                .entry(sym.clone())
                .or_default()
                .insert(session_key.clone());
        }
    }
    let symbol_session_counts: HashMap<String, usize> = symbol_sessions
        .into_iter()
        .map(|(sym, sessions)| (sym, sessions.len()))
        .collect();

    let mut scored: Vec<(f32, crate::CapsuleHit)> = hits
        .into_iter()
        .map(|h| {
            let s = score_hit_for_exploration(&h, &symbol_session_counts);
            (s, h)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().take(limit).map(|(_, h)| h).collect()
}

pub async fn run(
    query: Vec<String>,
    no_llm: bool,
    llm_model: Option<String>,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let query = query.join(" ");
    let query = query.trim().to_string();

    if query.is_empty() {
        println!("Explore future paths grounded in your workspace memory.\n");
        println!("Examples:");
        println!("  unlost explore \"should we keep lancedb or move to sqlite+fts?\"");
        println!("  unlost explore \"what happens if we remove the proxy recorder?\"");
        println!("  unlost explore \"what would multi-workspace federation require?\"");
        println!("  unlost explore \"is our embedding model still the right choice?\"");
        return Ok(());
    }

    let cwd = std::env::current_dir()?;
    let ws = crate::workspace::get_or_create_workspace_paths(&cwd)?;
    let workspace_root = crate::workspace::git_toplevel(&cwd)
        .unwrap_or_else(|| crate::workspace::canonicalize_dir(&cwd).unwrap_or(cwd.clone()));
    let workspace_root =
        crate::workspace::canonicalize_dir(&workspace_root).unwrap_or(workspace_root);
    let workspace_root = workspace_root.to_string_lossy().to_string();

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Loading memory...");
        Some(pb)
    } else {
        None
    };

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.set_message("Searching relevant capsules...");
    }

    // Pool 1: semantic search on the query (topically relevant capsules).
    // Frame the query with the explore intent so the embedding aligns with HyPE
    // question vectors stored at indexing time (question-to-question match).
    let framed_query = crate::storage::frame_query_for_command(
        &query,
        crate::storage::QueryIntent::Explore,
    );
    let sem_hits = crate::storage::query_capsules_lancedb(
        &framed_query,
        40,
        None,
        None,
        None,
        None,
        None,
        embedder,
        &ws,
    )
    .await
    .unwrap_or_default();

    // Pool 2: full scan scored by importance (failure modes, rationale, recurrence)
    let broad_hits =
        crate::storage::scan_capsules_lancedb(&ws, 200, None, None, None, None, None)
            .await
            .unwrap_or_default();

    let hits = select_hits_for_exploration(sem_hits, broad_hits, 30);

    if hits.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No capsules found yet for this workspace.");
        println!("Run `unlost init` or record a few coding sessions first.");
        return Ok(());
    }

    if no_llm {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        if output == OutputFormat::Json {
            println!("{}", crate::util::hits_to_json_pretty(&hits));
        } else {
            for hit in &hits {
                let cap = &hit.capsule;
                println!("---");
                println!("category:  {}", cap.category);
                if !cap.intent.trim().is_empty() {
                    println!("intent:    {}", cap.intent);
                }
                if !cap.decision.trim().is_empty() {
                    println!("decision:  {}", cap.decision);
                }
                if !cap.rationale.trim().is_empty() {
                    println!("rationale: {}", cap.rationale);
                }
                if !cap.next_steps.is_empty() {
                    println!("next:      {:?}", cap.next_steps);
                }
                println!("symbols:   {:?}", cap.symbols);
                println!();
            }
            println!();
        }
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
        pb.set_message(format!("Projecting with {}...", model_name));
    }

    let narrative = crate::narrative::llm_explore_narrative(
        llm_model.as_deref(),
        &query,
        &workspace_root,
        &hits,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }

    let out = crate::narrative::render_structured(output, &narrative);
    println!("{}", out);
    println!();

    Ok(())
}
