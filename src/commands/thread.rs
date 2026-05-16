use crate::cli::OutputFormat;
use chrono::TimeZone;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

const SESSION_GAP_MS: i64 = 4 * 60 * 60 * 1000;
const DORMANCY_THRESHOLD_MS: i64 = 7 * 24 * 60 * 60 * 1000;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    topic: Vec<String>,
    limit: usize,
    since: Option<String>,
    no_llm: bool,
    llm_model: Option<String>,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let query = topic.join(" ");
    let query = query.trim().to_string();
    if query.is_empty() {
        println!("Usage: unlost thread <topic>");
        println!("  Examples:");
        println!("    unlost thread \"passive resurfacing of memory\"");
        println!("    unlost thread \"local-first storage decisions\" --since 6m");
        return Ok(());
    }

    let since_ms = match since {
        Some(ref s) => crate::util::parse_time_filter(s)?,
        None => None,
    };

    let cwd = std::env::current_dir()?;
    let ws = crate::workspace::get_or_create_workspace_paths(&cwd)?;

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Mapping the thread...");
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

    let framed = crate::storage::frame_query_for_command(
        &query,
        crate::storage::QueryIntent::Trace,
    );

    let mut hits = crate::storage::query_capsules_cross_workspace(
        &framed,
        embedder,
        &ws,
        10,
        limit,
    )
    .await;

    if let Some(since_ms) = since_ms {
        hits.retain(|h| h.ts_ms >= since_ms);
    }

    if let Some(ref spinner) = spinner {
        spinner.finish_and_clear();
    }

    if hits.is_empty() {
        println!("unlost: no moments found for this topic in any workspace.");
        return Ok(());
    }

    hits.sort_by_key(|h| h.ts_ms);

    let rendered = render_thread_map(&hits, output, &ws);
    print!("{rendered}");

    if !no_llm {
        if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            println!("\n\x1b[2m{}\x1b[0m", "─".repeat(72));
        } else {
            println!("\n{}", "─".repeat(72));
        }

        let narrative = crate::narrative::llm_thread_narrative(
            llm_model.as_deref(),
            &query,
            &hits,
        )
        .await?;

        let rendered = crate::narrative::render_narrative(output, &narrative);
        println!("{rendered}");
    }

    Ok(())
}

fn render_thread_map(hits: &[crate::CapsuleHit], output: OutputFormat, _current_ws: &crate::WorkspacePaths) -> String {
    let mut out = String::new();

    let project_ids: std::collections::BTreeSet<&str> = hits
        .iter()
        .filter_map(|h| h.origin_workspace_id.as_deref())
        .collect();
    let project_count = if project_ids.is_empty() { 1 } else { project_ids.len() };
    let project_label = if project_ids.is_empty() || project_count == 1 {
        String::new()
    } else {
        format!(" across {} projects", project_count)
    };

    let earliest = fmt_date(hits.first().unwrap().ts_ms);
    let latest = fmt_date(hits.last().unwrap().ts_ms);

    out.push_str(&format!(
        "unlost: {} moments{} ({} → {})\n",
        hits.len(),
        project_label,
        earliest,
        latest,
    ));

    let sessions = group_into_sessions(hits);

    for (si, session) in sessions.iter().enumerate() {
        if si > 0 {
            let prev_last = sessions[si - 1].last().unwrap().ts_ms;
            let curr_first = session.first().unwrap().ts_ms;
            let gap = curr_first - prev_last;
            if gap >= DORMANCY_THRESHOLD_MS {
                let days = gap / (24 * 60 * 60 * 1000);
                if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
                    out.push_str(&format!("  \x1b[2m· · · {} days dormant · · ·\x1b[0m\n\n", days));
                } else {
                    out.push_str(&format!("  · · · {} days dormant · · ·\n\n", days));
                }
            }
        }

        let first_hit = session.first().unwrap();
        let date_str = fmt_date(first_hit.ts_ms);

        let ws_labels: std::collections::BTreeSet<String> = session
            .iter()
            .filter_map(|h| h.origin_workspace_id.as_deref())
            .filter_map(|id| crate::workspace::workspace_label_by_id(id))
            .collect();
        let ws_suffix = if ws_labels.len() == 1 {
            let label = ws_labels.iter().next().unwrap();
            if !label.is_empty() {
                format!("  [{}]", label)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let line = if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            let dashes = "─".repeat(60usize.saturating_sub(date_str.len() + ws_suffix.len()));
            format!("\x1b[2m── {} {ws_suffix} {dashes}\x1b[0m\n", date_str)
        } else {
            let dashes = "─".repeat(60usize.saturating_sub(date_str.len() + ws_suffix.len()));
            format!("── {} {ws_suffix} {dashes}\n", date_str)
        };
        out.push_str(&line);

        for (hi, hit) in session.iter().enumerate() {
            let entry = render_capsule_entry(hit, hi + 1, output);
            out.push_str(&entry);
            out.push('\n');
        }
    }

    out
}

fn group_into_sessions<'a>(hits: &'a [crate::CapsuleHit]) -> Vec<Vec<&'a crate::CapsuleHit>> {
    let mut sessions: Vec<Vec<&crate::CapsuleHit>> = Vec::new();
    let mut current: Vec<&crate::CapsuleHit> = Vec::new();

    for hit in hits {
        if current.is_empty() {
            current.push(hit);
            continue;
        }
        let prev_ts = current.last().unwrap().ts_ms;
        if hit.ts_ms - prev_ts <= SESSION_GAP_MS {
            current.push(hit);
        } else {
            sessions.push(std::mem::take(&mut current));
            current.push(hit);
        }
    }
    if !current.is_empty() {
        sessions.push(current);
    }
    sessions
}

fn render_capsule_entry(hit: &crate::CapsuleHit, index: usize, output: OutputFormat) -> String {
    let cap = &hit.capsule;
    let mut lines = Vec::new();

    let category = if cap.category.trim().is_empty() {
        "note".to_string()
    } else {
        cap.category.trim().to_string()
    };
    let decision = truncate(cap.decision.trim(), 80);

    if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
        lines.push(format!(
            "  #{}  \x1b[1m{}\x1b[0m  \"{}\"",
            index, category, decision
        ));
    } else {
        lines.push(format!("  #{}  {}  \"{}\"", index, category, decision));
    }

    if !cap.rationale.trim().is_empty() {
        let rationale = first_sentence(cap.rationale.trim());
        if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            lines.push(format!("      \x1b[2mRationale:\x1b[0m {}", rationale));
        } else {
            lines.push(format!("      Rationale: {}", rationale));
        }
    }

    if cap.failure_mode != crate::types::FailureMode::None {
        let fm = match cap.failure_mode {
            crate::types::FailureMode::None => "none",
            crate::types::FailureMode::Drift => "Drift",
            crate::types::FailureMode::Rediscovery => "Rediscovery",
            crate::types::FailureMode::DecisionConflict => "DecisionConflict",
            crate::types::FailureMode::RetrySpiral => "RetrySpiral",
            crate::types::FailureMode::FalseProgress => "FalseProgress",
            crate::types::FailureMode::UnboundedHorizon => "UnboundedHorizon",
        };
        if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            lines.push(format!("      \x1b[31m▲\x1b[0m {}", fm));
        } else {
            lines.push(format!("      ▲ {}", fm));
        }
    }

    if !cap.next_steps.is_empty() {
        let next = truncate(&cap.next_steps[0], 60);
        if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            lines.push(format!("      \x1b[36m○\x1b[0m next: {}", next));
        } else {
            lines.push(format!("      ○ next: {}", next));
        }
    }

    if !cap.symbols.is_empty() {
        let syms: Vec<&str> = cap.symbols.iter().map(|s| s.as_str()).take(3).collect();
        let sym_str = syms.join(", ");
        if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
            lines.push(format!("      \x1b[2m@\x1b[0m {}", sym_str));
        } else {
            lines.push(format!("      @ {}", sym_str));
        }
    }

    if let Some(ref sp) = hit.meta.source_pointer {
        let label = crate::workspace::resolve_source_label(sp);
        if let Some(l) = label {
            if !l.is_empty() {
                if output == OutputFormat::Ansi && !std::env::var_os("NO_COLOR").is_some() {
                    lines.push(format!("      \x1b[2m↗\x1b[0m {}", l));
                } else {
                    lines.push(format!("      ↗ {}", l));
                }
            }
        }
    }

    lines.join("\n")
}

fn fmt_date(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts_ms.to_string())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

fn first_sentence(s: &str) -> String {
    if let Some(pos) = s.find('.') {
        let candidate = s[..=pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    truncate(s, 80)
}
