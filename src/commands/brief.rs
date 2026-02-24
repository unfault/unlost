use crate::cli::OutputFormat;
use crate::types::FailureMode;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Score a capsule by its briefing value.
///
/// This is the inversion of recall's recency-first selection: repetition across
/// sessions is signal, recency is irrelevant. Failure modes and explicit
/// rationale are the highest-value capsules for a "what do I need to know" brief.
fn score_hit_for_brief(
    h: &crate::CapsuleHit,
    symbol_session_counts: &HashMap<String, usize>,
) -> f32 {
    let mut score = 0.0f32;

    // Recorded pain is must-know knowledge
    if h.capsule.failure_mode != FailureMode::None {
        score += 3.0;
    }
    // The worst traps get extra weight
    if matches!(
        h.capsule.failure_mode,
        FailureMode::RetrySpiral | FailureMode::DecisionConflict
    ) {
        score += 2.0;
    }

    // Someone recorded *why* — high signal
    if !h.capsule.rationale.trim().is_empty() {
        score += 1.5;
    }

    // An explicit choice was made
    if !h.capsule.decision.trim().is_empty() {
        score += 1.0;
    }

    // Symbols that recur across multiple sessions = recurring knowledge worth surfacing
    for sym in &h.capsule.symbols {
        if symbol_session_counts.get(sym).copied().unwrap_or(0) >= 2 {
            score += 1.0;
        }
    }

    score
}

/// Select the most briefing-valuable capsules from a pool.
///
/// Unlike select_hits_for_recall (recency-first, per-session capped),
/// this function:
/// - Scores by importance (failure modes, rationale, cross-session recurrence)
/// - Has no recency bias whatsoever
/// - Has no per-session cap — a session with many high-scoring capsules gets them all
/// - Deduplicates by id only (not by fingerprint — repetition IS signal here)
fn select_hits_for_brief(hits: Vec<crate::CapsuleHit>, limit: usize) -> Vec<crate::CapsuleHit> {
    if hits.is_empty() {
        return hits;
    }

    // Dedup by id first
    let mut by_id: HashMap<String, crate::CapsuleHit> = HashMap::new();
    for h in hits {
        match by_id.get(&h.id) {
            Some(existing) if existing.ts_ms >= h.ts_ms => {}
            _ => {
                by_id.insert(h.id.clone(), h);
            }
        }
    }
    let hits: Vec<crate::CapsuleHit> = by_id.into_values().collect();

    // Build cross-session symbol counts: for each symbol, how many distinct sessions
    // mention it? This identifies knowledge that had to be re-learned or re-established.
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

    // Score and sort — highest importance first
    let mut scored: Vec<(f32, crate::CapsuleHit)> = hits
        .into_iter()
        .map(|h| {
            let s = score_hit_for_brief(&h, &symbol_session_counts);
            (s, h)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().take(limit).map(|(_, h)| h).collect()
}

/// Try to serve brief from stored checkpoint narratives.
/// Returns Some(rendered text) on success, None to fall back to full capsule scan.
async fn try_checkpoint_brief(
    ws: &crate::WorkspacePaths,
    llm_model: Option<&str>,
    _output: OutputFormat,
) -> Option<String> {
    std::fs::create_dir_all(&ws.db_dir).ok()?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await
        .ok()?;

    let checkpoints =
        crate::storage_checkpoint::get_recent_checkpoints(&db, &ws.id, 10)
            .await
            .ok()?;

    if checkpoints.is_empty() {
        return None;
    }

    // If there's only one checkpoint, return its narrative directly — no LLM needed.
    if checkpoints.len() == 1 {
        return Some(checkpoints[0].narrative.clone());
    }

    // Multiple checkpoints: ask LLM to synthesize them into a brief.
    // This is far cheaper than scanning 200 raw capsules.
    let result = synthesize_checkpoints_for_brief(llm_model, &checkpoints).await.ok()?;
    Some(result)
}

/// Synthesize multiple checkpoint narratives into a staff-engineer brief.
async fn synthesize_checkpoints_for_brief(
    llm_model: Option<&str>,
    checkpoints: &[crate::storage_checkpoint::CheckpointRow],
) -> anyhow::Result<String> {
    use chrono::TimeZone;
    let mut context = String::new();
    context.push_str(&format!(
        "The following are {} checkpoint story segments from this workspace, \
         ordered from most recent to oldest:\n\n",
        checkpoints.len()
    ));

    for (i, cp) in checkpoints.iter().enumerate() {
        let ts_str = chrono::Utc
            .timestamp_millis_opt(cp.ts_ms)
            .single()
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| cp.ts_ms.to_string());
        context.push_str(&format!("--- Segment {} ({})\n", i + 1, ts_str));
        context.push_str(&cp.narrative);
        context.push_str("\n\n");
    }

    let preamble = "You are unlost, acting as a staff engineer giving a debrief. \
        Given multiple session story segments from a workspace, synthesize them into \
        a single staff-engineer-level brief that covers: \
        (1) What is being built and the current state, \
        (2) The key decisions and trade-offs made (with rationale), \
        (3) Known failure modes, gotchas, and debt, \
        (4) What to focus on next. \
        Be direct and concrete. Cite specific decisions. Max 400 words. \
        Return the brief in the `narrative` field.";

    let result = crate::llm_extract::<crate::storage_checkpoint::CheckpointNarrativeOutput>(
        llm_model,
        preamble,
        &context,
    )
    .await?;
    Ok(result.narrative)
}

pub async fn run(
    target: Vec<String>,
    llm_model: Option<String>,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
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
        pb.set_message("Reading the room...");
        Some(pb)
    } else {
        None
    };

    let scope = target.join(" ");
    let scope = scope.trim().to_string();
    let scope_opt = (!scope.is_empty()).then_some(scope.as_str());

    // Load embedder only when scoped (needed for semantic search)
    let embedder = if scope_opt.is_some() {
        if let Some(pb) = spinner.as_ref() {
            pb.set_message("Loading memory...");
        }
        Some(
            crate::embed::load_embedder(
                &embed_model,
                embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                false,
            )
            .await?,
        )
    } else {
        None
    };

    if let Some(pb) = spinner.as_ref() {
        pb.set_message("Weighing what matters...");
    }

    // ── Checkpoint fast path (unscoped only) ─────────────────────────────────
    // When not scoped, try to synthesize from stored checkpoint narratives.
    // This is significantly cheaper than processing 200 raw capsules.
    if scope_opt.is_none() {
        if let Some(result) = try_checkpoint_brief(&ws, llm_model.as_deref(), output).await {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }
            let wrap = output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
            let out = if wrap {
                crate::util::wrap_plain_text(&result, 80)
            } else {
                result
            };
            println!("{}", out);
            println!();
            return Ok(());
        }
    }

    // Full scan — no recency bias, no emotion/provider/time filters.
    // brief is deliberately opinionated: it looks at all recorded history.
    let mut hits: Vec<crate::CapsuleHit> = Vec::new();

    if let Ok(all) =
        crate::storage::scan_capsules_lancedb(&ws, 200, None, None, None, None, None).await
    {
        hits.extend(all);
    }

    // When scoped, also run semantic search to catch conceptually-related capsules
    // that don't literally mention the scope string in their symbols field.
    // Frame the query with the brief intent so the embedding aligns with HyPE
    // question vectors stored at indexing time (question-to-question match).
    if let (Some(scope), Some(embedder)) = (scope_opt, embedder) {
        let framed = crate::storage::frame_query_for_command(
            scope,
            crate::storage::QueryIntent::Brief,
        );
        if let Ok(sem) = crate::storage::query_capsules_lancedb(
            &framed, 60, None, None, None, None, None, embedder, &ws,
        )
        .await
        {
            hits.extend(sem);
        }
    }

    let hits = select_hits_for_brief(hits, 40);

    if hits.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        if let Some(s) = scope_opt {
            println!("No capsules found yet for: {s}");
        } else {
            println!("No capsules found yet for this workspace.");
            println!("Run `unlost recall` after a few coding sessions to build up memory.");
        }
        return Ok(());
    }

    // Resolve model name for spinner message
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
        pb.set_message(format!("Briefing you with {}...", model_name));
    }

    let narrative = crate::narrative::llm_brief_narrative(
        llm_model.as_deref(),
        scope_opt,
        &ws.id,
        &workspace_root,
        &hits,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }

    let mut out = crate::narrative::render_brief(output, &narrative);
    let wrap = output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
    if wrap {
        out = crate::util::wrap_plain_text(&out, 80);
    }
    println!("{}", out);
    println!();

    Ok(())
}
