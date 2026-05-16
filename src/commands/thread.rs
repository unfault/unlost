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
        if output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none() {
            println!("\x1b[2m{}\x1b[0m", "─".repeat(72));
        } else {
            println!("{}", "─".repeat(72));
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

fn is_ansi(output: OutputFormat) -> bool {
    output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none()
}

fn render_thread_map(
    hits: &[crate::CapsuleHit],
    output: OutputFormat,
    _current_ws: &crate::WorkspacePaths,
) -> String {
    let mut out = String::new();

    // Header line
    let project_ids: std::collections::BTreeSet<&str> = hits
        .iter()
        .filter_map(|h| h.origin_workspace_id.as_deref())
        .collect();
    let project_count = if project_ids.is_empty() { 1 } else { project_ids.len() };
    let project_label = if project_count > 1 {
        format!(" across {} projects", project_count)
    } else {
        String::new()
    };

    let earliest = fmt_date(hits.first().unwrap().ts_ms);
    let latest = fmt_date(hits.last().unwrap().ts_ms);

    if is_ansi(output) {
        out.push_str(&format!(
            "\x1b[2m{} moment{}{} · {} → {}\x1b[0m\n\n",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            project_label,
            earliest,
            latest,
        ));
    } else {
        out.push_str(&format!(
            "{} moment{}{} · {} → {}\n\n",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            project_label,
            earliest,
            latest,
        ));
    }

    let sessions = group_into_sessions(hits);
    let mut global_idx = 0usize;

    for (si, session) in sessions.iter().enumerate() {
        // Blank line + dormancy gap between sessions
        if si > 0 {
            let prev_last = sessions[si - 1].last().unwrap().ts_ms;
            let curr_first = session.first().unwrap().ts_ms;
            let gap = curr_first - prev_last;
            if gap >= DORMANCY_THRESHOLD_MS {
                let days = gap / (24 * 60 * 60 * 1000);
                if is_ansi(output) {
                    out.push_str(&format!("\x1b[2m  · · ·  {} days  · · ·\x1b[0m\n\n", days));
                } else {
                    out.push_str(&format!("  · · ·  {} days  · · ·\n\n", days));
                }
            } else {
                out.push('\n');
            }
        }

        // Session date header
        let first_hit = session.first().unwrap();
        let date_str = fmt_date(first_hit.ts_ms);

        // Workspace label — only show when there's a single distinct project for this session
        let ws_labels: std::collections::BTreeSet<String> = session
            .iter()
            .filter_map(|h| h.origin_workspace_id.as_deref())
            .filter_map(|id| crate::workspace::workspace_label_by_id(id))
            .collect();
        let ws_tag = if ws_labels.len() == 1 {
            let label = ws_labels.iter().next().unwrap();
            if !label.is_empty() {
                Some(label.clone())
            } else {
                None
            }
        } else {
            None
        };

        if is_ansi(output) {
            // Bold white date, cyan project tag, dim trailing dashes
            let tag_part = match &ws_tag {
                Some(t) => format!("  \x1b[0;36m{}\x1b[0m", t),
                None => String::new(),
            };
            let tag_visible_len = ws_tag.as_deref().map(|t| t.len() + 2).unwrap_or(0);
            let dash_count = 52usize.saturating_sub(date_str.len() + tag_visible_len);
            out.push_str(&format!(
                "\x1b[1;97m{}\x1b[0m{}\x1b[2m  {}\x1b[0m\n",
                date_str,
                tag_part,
                "─".repeat(dash_count),
            ));
        } else {
            let tag_part = match &ws_tag {
                Some(t) => format!("  {}", t),
                None => String::new(),
            };
            out.push_str(&format!("{}{}\n", date_str, tag_part));
        }

        for hit in session.iter() {
            global_idx += 1;
            let entry = render_capsule_entry(hit, global_idx, output);
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

    // ── Decision line — the primary signal, full width ──────────────────────
    let decision = cap.decision.trim();
    let decision_display = if decision.is_empty() {
        // Fall back to intent if no decision was extracted
        truncate(cap.intent.trim(), 120)
    } else {
        decision.to_string()
    };

    if is_ansi(output) {
        // Index dim, decision bold white
        lines.push(format!(
            "\n  \x1b[2m{:>2}\x1b[0m  \x1b[1m{}\x1b[0m",
            index, decision_display
        ));
    } else {
        lines.push(format!("\n  {:>2}  {}", index, decision_display));
    }

    // ── Support lines — all dim, no labels, compact ──────────────────────────

    // Rationale: first sentence, max 80 chars, no "Rationale:" prefix
    if !cap.rationale.trim().is_empty() {
        let rationale = truncate(&first_sentence(cap.rationale.trim()), 100);
        if is_ansi(output) {
            lines.push(format!("      \x1b[2m{}\x1b[0m", rationale));
        } else {
            lines.push(format!("      {}", rationale));
        }
    }

    // Failure mode — amber warning glyph, stays visible
    if cap.failure_mode != crate::types::FailureMode::None {
        let fm = match cap.failure_mode {
            crate::types::FailureMode::None => "",
            crate::types::FailureMode::Drift => "drift",
            crate::types::FailureMode::Rediscovery => "rediscovery",
            crate::types::FailureMode::DecisionConflict => "decision conflict",
            crate::types::FailureMode::RetrySpiral => "retry spiral",
            crate::types::FailureMode::FalseProgress => "false progress",
            crate::types::FailureMode::UnboundedHorizon => "unbounded horizon",
        };
        if !fm.is_empty() {
            if is_ansi(output) {
                lines.push(format!("      \x1b[33m▲ {}\x1b[0m", fm));
            } else {
                lines.push(format!("      ▲ {}", fm));
            }
        }
    }

    // Symbols — dim, no prefix glyph
    if !cap.symbols.is_empty() {
        let syms: Vec<&str> = cap.symbols.iter().map(|s| s.as_str()).take(4).collect();
        let sym_str = syms.join("  ");
        if is_ansi(output) {
            lines.push(format!("      \x1b[2m{}\x1b[0m", sym_str));
        } else {
            lines.push(format!("      {}", sym_str));
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
    // Truncate on a char boundary to avoid splitting multi-byte chars
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let end = s.char_indices().nth(max.saturating_sub(1)).map(|(i, _)| i).unwrap_or(s.len());
        format!("{}…", &s[..end])
    }
}

fn first_sentence(s: &str) -> String {
    // Find the first sentence-ending period that isn't inside an abbreviation.
    // Simple heuristic: period followed by space or end-of-string.
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'.' {
            let after = i + 1;
            if after >= bytes.len() || bytes[after] == b' ' || bytes[after] == b'\n' {
                let candidate = s[..=i].trim();
                if !candidate.is_empty() {
                    return candidate.to_string();
                }
            }
        }
    }
    s.to_string()
}
