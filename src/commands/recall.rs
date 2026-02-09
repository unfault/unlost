use crate::cli::OutputFormat;
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
    // Only de-dup within a session/connection. Cross-session repetition can still be useful
    // evidence (e.g., the same decision reaffirmed later).
    let mut seen_fp: HashSet<String> = HashSet::new();
    let mut per_session: HashMap<String, usize> = HashMap::new();
    let max_per_session = (limit / 3).clamp(2, 8);

    let mut selected = Vec::with_capacity(limit);

    // Always keep the most recent capsule as an anchor.
    let mut it = hits.into_iter();
    let Some(first) = it.next() else {
        return Vec::new();
    };
    {
        let sk = hit_session_key(&first);
        let fp = hit_fingerprint(&first);
        seen_fp.insert(format!("{sk}|{fp}"));
        *per_session.entry(sk).or_insert(0) += 1;
        selected.push(first);
    }

    for h in it {
        if selected.len() >= limit {
            break;
        }

        let sk = hit_session_key(&h);
        let cnt = per_session.entry(sk.clone()).or_insert(0);
        if *cnt >= max_per_session {
            continue;
        }

        let fp = hit_fingerprint(&h);
        let k = format!("{sk}|{fp}");
        if seen_fp.contains(&k) {
            continue;
        }

        seen_fp.insert(k);
        *cnt += 1;
        selected.push(h);
    }
    selected
}

pub(crate) async fn run(
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

    let mut hits: Vec<crate::CapsuleHit> = Vec::new();
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

    if let Some(scope) = scope_opt.as_deref() {
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

        if let Ok(mut sem) = crate::storage::query_capsules_lancedb(
            scope,
            18,
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
    let want = limit.min(40);
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

    if let Some(pb) = spinner.as_ref() {
        pb.set_message("Weaving threads...");
    }
    let narrative = crate::narrative::llm_recall_narrative(
        llm_model.as_deref(),
        scope_opt.as_deref(),
        &ws.id,
        &workspace_root,
        &hits,
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
    println!("{}\n", out);
    Ok(())
}
