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

fn select_hits_for_recall(mut hits: Vec<crate::CapsuleHit>, limit: usize) -> Vec<crate::CapsuleHit> {
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
    let recent_threshold = (limit / 3).max(5);
    let max_per_old_session = 5;

    let mut selected = Vec::with_capacity(limit);
    for h in hits {
        if selected.len() >= limit {
            break;
        }

        let sk = hit_session_key(&h);
        let fp = hit_fingerprint(&h);
        let k = format!("{sk}|{fp}");

        // If we are in the "high recency" window, we are less strict about repetition
        // and session limits, but we still collapse exact duplicates (same fingerprint).
        if selected.len() < recent_threshold {
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
        // When scoped, prioritize capsules that explicitly mention or relate to the scope
        // Use semantic search to catch text mentions (increased from 18 to 50)
        if let Ok(mut sem) = crate::storage::query_capsules_lancedb(
            scope,
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
            if let Ok(mut scoped) = crate::storage::scan_capsules_lancedb(
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
                scoped.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
                hits.extend(scoped);
            }
        }

        // Only backfill with recent capsules if we're under the limit
        // This ensures scoped results dominate the narrative
        if hits.len() < want {
            if let Ok(mut recent) = crate::storage::scan_capsules_lancedb(
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
                recent.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
                hits.extend(recent);
            }
        }
    } else {
        // No scope: fetch recent capsules for general workspace context
        if let Ok(mut recent) = crate::storage::scan_capsules_lancedb(
            &ws,
            120,
            None,
            emotion_label.as_deref(),
            provider_label.as_deref(),
            since_ms,
            until_ms,
        )
        .await
        {
            recent.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
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
    let mut hits = by_id.into_values().collect::<Vec<_>>();
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

    // Fetch recent interventions to include in context
    let mut recent_interventions =
        crate::metrics::get_recent_interventions(&ws.metrics_jsonl, 3).unwrap_or_default();

    // Backfill topics from hits if missing
    for iv in &mut recent_interventions {
        if iv.topic.is_none() {
            iv.topic = hits
                .iter()
                .find(|h| (h.ts_ms - iv.ts_ms).abs() < 30000)
                .map(|h| h.capsule.intent.clone());
        }
    }

    let narrative = crate::narrative::llm_recall_narrative(
        llm_model.as_deref(),
        scope_opt.as_deref(),
        &ws.id,
        &workspace_root,
        &hits,
        &recent_interventions,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }
    let mut out = crate::narrative::render_narrative(output, &narrative);
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
                "NOTE", "REASON", "DECISION", "INTENT", "NEXT_STEPS", "RATIONALE", "SYMBOLS",
                "USER", "ASSISTANT", "SYSTEM", "SUCCESS", "FAILURE", "ERROR", "WARNING",
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
                format!("{}, {} and {} others", clean_symbols[0], clean_symbols[1], clean_symbols.len() - 2)
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
