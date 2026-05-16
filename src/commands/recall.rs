use crate::cli::OutputFormat;
use chrono::TimeZone;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

fn fingerprint_tokens(s: &str, max_tokens: usize) -> String {
    let mut out = String::with_capacity(s.len().min(96));
    let mut cur = String::new();
    let mut n = 0usize;

    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
            continue;
        }

        if cur.is_empty() {
            continue;
        }

        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&cur);
        n += 1;
        if n >= max_tokens {
            return out;
        }
        cur.clear();
    }

    if !cur.is_empty() && n < max_tokens {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&cur);
    }
    out
}

fn hit_fingerprint(h: &crate::CapsuleHit) -> String {
    // Goal: collapse near-duplicates that differ only slightly in phrasing.
    // We intentionally use only the first few tokens from each field.
    let cap = &h.capsule;
    let mut fp = String::with_capacity(256);
    fp.push_str(&fingerprint_tokens(&cap.category, 6));
    fp.push('|');
    fp.push_str(&fingerprint_tokens(&cap.intent, 14));
    fp.push('|');
    fp.push_str(&fingerprint_tokens(&cap.decision, 12));

    if !cap.symbols.is_empty() {
        fp.push('|');
        fp.push_str(&fingerprint_tokens(&cap.symbols.join(" "), 10));
    }
    fp
}

fn hit_session_key(h: &crate::CapsuleHit) -> String {
    if let Some(s) = h.meta.agent_session_id.as_deref() {
        if !s.trim().is_empty() {
            return format!("ses:{s}");
        }
    }
    format!("conn:{}", h.conn_id)
}

fn is_low_signal_for_recall(h: &crate::CapsuleHit) -> bool {
    let cap = &h.capsule;
    let category = cap.category.trim().to_ascii_lowercase();
    if category == "replay" {
        return true;
    }

    // These are explicit signals that we did not get a real LLM-derived capsule.
    // They tend to be short echoes like "yes" and crowd out real decisions.
    if let Some(sig) = cap.failure_signals.as_deref() {
        if sig.contains("Ghost extraction") {
            return true;
        }
        if sig.contains("Heuristic extraction (LLM failed)") {
            return true;
        }
    }

    // Also drop tiny ack-like capsules that carry no structure.
    let intent = cap.intent.trim();
    let decision = cap.decision.trim();
    let no_structure =
        cap.rationale.trim().is_empty() && cap.next_steps.is_empty() && cap.symbols.is_empty();
    if no_structure && intent.len() <= 3 && decision.len() <= 3 {
        return true;
    }

    false
}

fn select_hits_for_recall(
    mut hits: Vec<crate::CapsuleHit>,
    limit: usize,
) -> Vec<crate::CapsuleHit> {
    if hits.len() <= limit {
        return hits;
    }

    hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));

    // Keep recency, but collapse repetitive capsules so we don't crowd out
    // older-but-important decisions.
    let mut seen_fp: HashSet<String> = HashSet::new();
    let mut per_session: HashMap<String, usize> = HashMap::new();

    // We want a mix:
    // 1. High recency (the absolute latest things, even if repetitive)
    // 2. Historical breadth (decisions from older sessions)

    // Increase recency priority: take more from the absolute latest window.
    let recent_threshold = (limit / 2).max(10);
    let max_per_old_session = 3; // Reduce historical crowding

    let mut selected = Vec::with_capacity(limit);
    let now = crate::workspace::now_ms();
    let thirty_mins_ms = 30 * 60 * 1000;

    for h in hits {
        if selected.len() >= limit {
            break;
        }

        let sk = hit_session_key(&h);
        let fp = hit_fingerprint(&h);
        let k = format!("{sk}|{fp}");

        // Absolute recency: things from the last 30 minutes get a pass
        // regardless of the recent_threshold index, but still deduplicated.
        let is_very_recent = (now - h.ts_ms).abs() < thirty_mins_ms;

        if is_very_recent || selected.len() < recent_threshold {
            if seen_fp.contains(&k) {
                continue;
            }
            seen_fp.insert(k);
            *per_session.entry(sk).or_insert(0) += 1;
            selected.push(h);
            continue;
        }

        // For older history, we are stricter to ensure variety.
        let cnt = per_session.entry(sk.clone()).or_insert(0);
        if *cnt >= max_per_old_session {
            continue;
        }

        if seen_fp.contains(&k) {
            continue;
        }

        seen_fp.insert(k);
        *cnt += 1;
        selected.push(h);
    }
    selected
}

/// Try to serve recall from the most recent checkpoint + delta capsules.
///
/// Instead of printing the raw checkpoint narrative (which uses a different
/// structured format), we feed the checkpoint as pre-digested context into
/// `llm_recall_narrative` — the same function and prompt that normal recall
/// uses. Output format is therefore identical regardless of which path was taken.
///
/// Returns Some(rendered output) on success, None to fall back to full recall.
async fn try_checkpoint_recall(
    ws: &crate::WorkspacePaths,
    workspace_root: &str,
    llm_model: Option<&str>,
    output: OutputFormat,
) -> Option<String> {
    std::fs::create_dir_all(&ws.db_dir).ok()?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await
        .ok()?;

    let checkpoints =
        crate::storage_checkpoint::get_recent_checkpoints(&db, &ws.id, 5)
            .await
            .ok()?;

    let latest = checkpoints.into_iter().next()?; // newest first

    // Fetch capsules newer than the checkpoint's to_ts_ms (the delta)
    let delta = crate::storage::scan_capsules_lancedb_recent(
        ws,
        20,
        None,
        None,
        None,
        Some(latest.to_ts_ms + 1),
        None,
    )
    .await
    .ok()?;

    if delta.len() > 5 {
        // Too much new content — fall back to full recall so nothing is missed
        return None;
    }

    // Build a synthetic hit list: a single "checkpoint" CapsuleHit carrying the
    // pre-digested narrative as its intent, followed by any delta capsules.
    // llm_recall_narrative will see these as its capsule input and produce the
    // same styled output it always does.
    let checkpoint_hit = checkpoint_as_capsule_hit(&latest);
    let mut hits = vec![checkpoint_hit];
    hits.extend(delta);

    let narrative = crate::narrative::llm_recall_narrative(
        llm_model,
        None, // unscoped
        &ws.id,
        workspace_root,
        &hits,
        &[], // no interventions in checkpoint path
        false,
        false,
        false,
    )
    .await
    .ok()?;

    let mut out = crate::narrative::render_narrative(output, &narrative);
    out = out.replace("Suggested next steps:", "Next steps (if any):");
    let wrap = output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
    if wrap {
        out = crate::util::wrap_plain_text(&out, 80);
    }
    Some(out)
}

/// Convert a CheckpointRow into a synthetic CapsuleHit so it can be passed to
/// `llm_recall_narrative` as pre-digested context alongside any delta capsules.
fn checkpoint_as_capsule_hit(
    cp: &crate::storage_checkpoint::CheckpointRow,
) -> crate::CapsuleHit {
    use crate::types::{ExtractionMode, FailureMode, IntentCapsule, ResponseMeta};
    crate::CapsuleHit {
        id: cp.id.clone(),
        ts_ms: cp.to_ts_ms, // use the end of the checkpoint window as its timestamp
        conn_id: 0,
        exchange_seq: 0,
        capsule: IntentCapsule {
            category: "checkpoint".to_string(),
            // Put the full checkpoint narrative in intent — this is what
            // llm_recall_narrative will read as context
            intent: cp.narrative.clone(),
            decision: String::new(),
            rationale: String::new(),
            next_steps: vec![],
            symbols: vec![],
            user_symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
            extraction_mode: ExtractionMode::default(),
            questions: vec![],
        },
        meta: ResponseMeta {
            source: "checkpoint".to_string(),
            upstream_host: String::new(),
            request_path: String::new(),
            http_status: 200,
            agent_session_id: cp.session_id.clone(),
            source_pointer: None,
            usage: None,
        },
        distance: 0.0,
        user_emotion: None,
        assistant_emotion: None,
        head_sha: None,
        commit_sha: None,
        turn_eval: None,
        origin_workspace_id: None,
    }
}

pub async fn run(
    target: Vec<String>,
    limit: usize,
    emotion: Option<crate::cli::EmotionType>,
    provider: Option<crate::cli::ProviderType>,
    since: Option<String>,
    until: Option<String>,
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
        pb.set_message("Let me recall...");
        Some(pb)
    } else {
        None
    };

    let scope = target.join(" ");
    let scope = scope.trim().to_string();
    let scope_opt = (!scope.is_empty()).then_some(scope);

    let emotion_label = emotion.map(|e| match e {
        crate::cli::EmotionType::Joy => "joy",
        crate::cli::EmotionType::Anger => "anger",
        crate::cli::EmotionType::Frustration => "frustration",
        crate::cli::EmotionType::Sad => "sad",
        crate::cli::EmotionType::Confused => "confused",
        crate::cli::EmotionType::Neutral => "neutral",
    });
    let provider_label = provider.map(|p| match p {
        crate::cli::ProviderType::Openai => "openai",
        crate::cli::ProviderType::Anthropic => "anthropic",
        crate::cli::ProviderType::Opencode => "opencode",
    });
    let since_ms = match since {
        Some(ref s) => crate::util::parse_time_filter(s)?,
        None => None,
    };
    let until_ms = match until {
        Some(ref u) => crate::util::parse_time_filter(u)?,
        None => None,
    };

    let _ = crate::metrics::record_command_recall(
        &ws,
        scope_opt.as_deref().unwrap_or(""),
        limit,
        emotion_label,
        provider_label,
    );

    // ── Checkpoint fast path (unscoped only) ─────────────────────────────────
    // When there's no scope filter, try to serve from the most recent checkpoint
    // plus a small delta of new capsules since the checkpoint. This avoids an
    // LLM call for every recall invocation.
    if scope_opt.is_none()
        && since_ms.is_none()
        && until_ms.is_none()
        && emotion_label.is_none()
        && provider_label.is_none()
    {
        if let Some(result) = try_checkpoint_recall(&ws, &workspace_root, llm_model.as_deref(), output).await {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }
            println!("{}", result);
            println!();
            return Ok(());
        }
    }

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.set_message("Browsing our memory...");
    }

    let mut hits: Vec<crate::CapsuleHit> = Vec::new();
    let want = limit.min(40);

    if let Some(scope) = scope_opt.as_deref() {
        // When scoped, prioritize capsules that explicitly mention or relate to the scope.
        // Frame the query with the recall intent so the embedding aligns with HyPE
        // question vectors stored at indexing time (question-to-question match).
        let framed_scope = crate::storage::frame_query_for_command(
            scope,
            crate::storage::QueryIntent::Recall,
        );
        if let Ok(mut sem) = crate::storage::query_capsules_lancedb(
            &framed_scope,
            50,
            None,
            emotion_label.as_deref(),
            provider_label.as_deref(),
            since_ms,
            until_ms,
            embedder.clone(),
            &ws,
        )
        .await
        {
            sem.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            hits.extend(sem);
        }

        // Also fetch by symbols field for direct references
        if let Some(expr) = crate::util::scope_filter_expr(scope) {
            if let Ok(scoped) = crate::storage::scan_capsules_lancedb_recent(
                &ws,
                80,
                Some(&expr),
                emotion_label.as_deref(),
                provider_label.as_deref(),
                since_ms,
                until_ms,
            )
            .await
            {
                hits.extend(scoped);
            }
        }

        // Only backfill with recent capsules if we're under the limit
        // This ensures scoped results dominate the narrative
        if hits.len() < want {
            if let Ok(recent) = crate::storage::scan_capsules_lancedb_recent(
                &ws,
                want.saturating_sub(hits.len()),
                None,
                emotion_label.as_deref(),
                provider_label.as_deref(),
                since_ms,
                until_ms,
            )
            .await
            {
                hits.extend(recent);
            }
        }
    } else {
        // No scope: fetch a larger recency window so low-signal "ghost" capsules
        // don't crowd out the last real decisions.
        let scan_n = (want.saturating_mul(12)).clamp(120, 600);
        if let Ok(recent) = crate::storage::scan_capsules_lancedb_recent(
            &ws,
            scan_n,
            None,
            emotion_label.as_deref(),
            provider_label.as_deref(),
            since_ms,
            until_ms,
        )
        .await
        {
            hits.extend(recent);
        }
    }

    let mut by_id: HashMap<String, crate::CapsuleHit> = HashMap::new();
    for h in hits {
        match by_id.get(&h.id) {
            Some(existing) if existing.ts_ms >= h.ts_ms => {}
            _ => {
                by_id.insert(h.id.clone(), h);
            }
        }
    }
    // Include git capsules in recall by default. Commits are high-signal decisions
    // and help ground the recent story when conversational capsules are thin.
    let hits = by_id.into_values().collect::<Vec<_>>();

    // Prefer high-signal capsules for narrative recall. If we don't have enough,
    // fall back to the full set rather than returning nothing.
    let filtered = hits
        .iter()
        .cloned()
        .filter(|h| !is_low_signal_for_recall(h))
        .collect::<Vec<_>>();
    let mut hits = if filtered.len() >= 6 { filtered } else { hits };

    hits = select_hits_for_recall(hits, want);

    if hits.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        if let Some(s) = scope_opt {
            println!("No capsules found yet for: {s}");
        } else {
            println!("No capsules found yet for this workspace.");
        }
        return Ok(());
    }

    // Determine which model will be used for the narrative
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
        pb.set_message(format!("Weaving threads with {}...", model_name));
    }

    // Recent friction interventions are useful to display to the user, but they
    // shouldn't steer the LLM narrative by default.
    let hide_interventions = std::env::var_os("UNLOST_RECALL_HIDE_INTERVENTIONS").is_some();
    let include_interventions_in_context =
        std::env::var_os("UNLOST_RECALL_INTERVENTIONS_IN_CONTEXT").is_some();

    let mut recent_interventions = if hide_interventions {
        Vec::new()
    } else {
        crate::metrics::get_recent_interventions(&ws.metrics_jsonl, 3).unwrap_or_default()
    };

    // Backfill topics from hits if missing
    for iv in &mut recent_interventions {
        if iv.topic.is_none() {
            iv.topic = hits
                .iter()
                .find(|h| (h.ts_ms - iv.ts_ms).abs() < 30000)
                .map(|h| h.capsule.intent.clone());
        }
    }

    let interventions_for_context: Vec<crate::metrics::Intervention> = if include_interventions_in_context {
        recent_interventions.clone()
    } else {
        Vec::new()
    };

    let narrative = crate::narrative::llm_recall_narrative(
        llm_model.as_deref(),
        scope_opt.as_deref(),
        &ws.id,
        &workspace_root,
        &hits,
        &interventions_for_context,
        !recent_interventions.is_empty(),
        include_interventions_in_context,
        true,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }
    let mut out = crate::narrative::render_narrative(output, &narrative);

    // Make the "next steps" heading less misleading in common cases where the
    // narrative is primarily a retrospective recap.
    out = out.replace("Suggested next steps:", "Next steps (if any):");

    let wrap = output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
    if wrap {
        out = crate::util::wrap_plain_text(&out, 80);
    }
    println!("{}", out);

    // Display recent interventions after the narrative
    if !recent_interventions.is_empty() {
        println!();
        println!("\x1b[2mRecent friction interventions:\x1b[0m");
        let now = crate::workspace::now_ms();
        for (i, iv) in recent_interventions.iter().enumerate() {
            let ts_str = chrono::Utc
                .timestamp_millis_opt(iv.ts_ms)
                .single()
                .map(|dt| dt.format("%H:%M").to_string())
                .unwrap_or_else(|| "--:--".to_string());

            let ago_str = crate::util::format_elapsed_time(iv.ts_ms, now);

            let duration_str = if let Some(start) = iv.watch_start_ts {
                let dur_mins = (iv.ts_ms - start) / 60000;
                format!("Intervened after {}m", dur_mins)
            } else {
                "Intervened".to_string()
            };

            let diagnosis = crate::metrics::get_diagnosis(&iv.cause, &iv.top_channels);
            let severity = crate::metrics::get_severity_label(iv.intensity);

            let emotion_str = if let Some(ref e) = iv.user_emotion {
                if crate::governor::FRICTION_EMOTIONS.contains(&e.as_str()) {
                    format!(" (user {})", e)
                } else {
                    "".to_string()
                }
            } else {
                "".to_string()
            };

            let mut clean_symbols = Vec::new();
            let blacklist = [
                "NOTE",
                "REASON",
                "DECISION",
                "INTENT",
                "NEXT_STEPS",
                "RATIONALE",
                "SYMBOLS",
                "USER",
                "ASSISTANT",
                "SYSTEM",
                "SUCCESS",
                "FAILURE",
                "ERROR",
                "WARNING",
            ];
            for s in &iv.symbols {
                let trimmed = s.trim_matches(|c: char| c == ':' || c == '.' || c == ',');
                if blacklist.iter().any(|&b| b == trimmed.to_uppercase()) {
                    continue;
                }
                if trimmed.contains('/') && !trimmed.contains('.') {
                    let parts: Vec<&str> = trimmed.split('/').collect();
                    if parts.iter().any(|&p| {
                        p == "read" || p == "write" || p == "inspect" || p == "call" || p == "tool"
                    }) {
                        continue;
                    }
                }
                if !trimmed.is_empty() {
                    clean_symbols.push(trimmed.to_string());
                }
            }

            let symbols_str = if clean_symbols.is_empty() {
                "—".to_string()
            } else if clean_symbols.len() > 3 {
                format!(
                    "{}, {} and {} others",
                    clean_symbols[0],
                    clean_symbols[1],
                    clean_symbols.len() - 2
                )
            } else {
                clean_symbols.join(", ")
            };

            println!(
                "\x1b[2m  {}. {} ({}) | {}: {} - {}{} \x1b[0m",
                i + 1,
                ts_str,
                ago_str,
                duration_str,
                severity,
                diagnosis,
                emotion_str
            );

            // Try to find topic in existing hits if not in intervention record (backfill)
            let topic = iv.topic.as_ref().or_else(|| {
                // Look for a hit within 30s of the intervention
                hits.iter()
                    .find(|h| (h.ts_ms - iv.ts_ms).abs() < 30000)
                    .map(|h| &h.capsule.intent)
            });

            if let Some(topic) = topic {
                let clean_topic = topic.replace('\n', " ");
                let truncated = if clean_topic.len() > 80 {
                    format!("{}...", &clean_topic[..77])
                } else {
                    clean_topic
                };
                println!("\x1b[2m     Topic: \"{}\"\x1b[0m", truncated);
            }

            println!("\x1b[2m     Symbols: {}\x1b[0m", symbols_str);
        }
    }

    println!();
    Ok(())
}
