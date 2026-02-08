use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::Duration;

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
    let ws = crate::workspace::get_or_create_workspace_paths(&std::env::current_dir()?)?;

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

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    let mut hits: Vec<crate::CapsuleHit> = Vec::new();
    if let Ok(mut recent) = crate::storage::scan_capsules_lancedb(
        &ws, 120, None, emotion_label.as_deref(), provider_label.as_deref(), since_ms, until_ms,
    )
    .await
    {
        recent.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        hits.extend(recent.into_iter().take(limit.min(40)));
    }

    if let Some(scope) = scope_opt.as_deref() {
        if let Some(expr) = crate::util::scope_filter_expr(scope) {
            if let Ok(mut scoped) = crate::storage::scan_capsules_lancedb(
                &ws, 80, Some(&expr), emotion_label.as_deref(), provider_label.as_deref(), since_ms, until_ms,
            )
            .await
            {
                scoped.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
                hits.extend(scoped);
            }
        }

        if let Ok(mut sem) = crate::storage::query_capsules_lancedb(
            scope, 18, None, emotion_label.as_deref(), provider_label.as_deref(), since_ms, until_ms,
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
    hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
    if hits.len() > limit {
        hits.truncate(limit);
    }

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
