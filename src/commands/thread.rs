use crate::cli::OutputFormat;
use chrono::TimeZone;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::BTreeSet;
use std::time::Duration;

const WRAP_WIDTH: usize = 80;
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

    let (narrative, narrative_warning) = if no_llm {
        (None, None)
    } else {
        match crate::narrative::llm_thread_narrative(llm_model.as_deref(), &query, &hits).await {
            Ok(n) => (Some(n), None),
            Err(e) => {
                tracing::debug!(error = %e, "thread narrative unavailable; falling back to extracted notes");
                (
                    None,
                    Some(
                        "No LLM configured; showing extracted notes only. Configure with `unlost config llm ollama --model <model>`."
                            .to_string(),
                    ),
                )
            }
        }
    };

    let rendered = render_thread_map(
        &query,
        narrative.as_deref(),
        narrative_warning.as_deref(),
        &hits,
        output,
        &ws,
    );
    print!("{rendered}");

    Ok(())
}

fn is_ansi(output: OutputFormat) -> bool {
    output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none()
}

fn render_thread_map(
    topic: &str,
    narrative: Option<&str>,
    narrative_warning: Option<&str>,
    hits: &[crate::CapsuleHit],
    output: OutputFormat,
    _current_ws: &crate::WorkspacePaths,
) -> String {
    let mut out = String::new();

    let project_ids: BTreeSet<&str> = hits
        .iter()
        .filter_map(|h| h.origin_workspace_id.as_deref())
        .collect();
    let project_count = if project_ids.is_empty() { 1 } else { project_ids.len() };
    let project_label = if project_count > 1 {
        format!(" across {} projects", project_count)
    } else {
        String::new()
    };

    let days = group_into_days(hits);
    let display_note_count: usize = days
        .iter()
        .map(|day| display_notes_for_day(day).len())
        .sum();
    let earliest = fmt_range_date(hits.first().unwrap().ts_ms);
    let latest = fmt_range_date(hits.last().unwrap().ts_ms);
    let span_days = ((hits.last().unwrap().ts_ms - hits.first().unwrap().ts_ms)
        / (24 * 60 * 60 * 1000))
        .max(0);

    render_title(&mut out, topic, output);

    if is_ansi(output) {
        out.push_str(&format!(
            "\x1b[2m{} moment{} · {} note{}{} · {} back to {}{}\x1b[0m\n\n",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            display_note_count,
            if display_note_count == 1 { "" } else { "s" },
            project_label,
            latest,
            earliest,
            span_suffix(span_days),
        ));
    } else {
        out.push_str(&format!(
            "{} moment{} · {} note{}{} · {} back to {}{}\n\n",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            display_note_count,
            if display_note_count == 1 { "" } else { "s" },
            project_label,
            latest,
            earliest,
            span_suffix(span_days),
        ));
    }

    if let Some(n) = narrative {
        let rendered = crate::narrative::render_narrative(output, n);
        for line in rendered.lines() {
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    } else if let Some(warning) = narrative_warning {
        if is_ansi(output) {
            push_wrapped_ansi(
                &mut out,
                "\x1b[2m",
                "\x1b[0m",
                "\x1b[2m",
                "\x1b[0m",
                warning,
                WRAP_WIDTH,
            );
        } else {
            push_wrapped_plain(&mut out, "", "", warning, WRAP_WIDTH);
        }
        out.push_str("\n\n");
    }

    let mut global_idx = 0usize;
    let display_days: Vec<Vec<&crate::CapsuleHit>> = days
        .iter()
        .rev()
        .map(|day| day.iter().rev().copied().collect())
        .collect();

    for (di, day) in display_days.iter().enumerate() {
        if di > 0 {
            let newer_day_oldest = display_days[di - 1].last().unwrap().ts_ms;
            let older_day_newest = day.first().unwrap().ts_ms;
            let gap = newer_day_oldest - older_day_newest;
            if gap >= DORMANCY_THRESHOLD_MS {
                let label = gap_label(gap);
                if is_ansi(output) {
                    out.push_str(&format!("\n\x1b[2m        {}\x1b[0m\n\n", label));
                } else {
                    out.push_str(&format!("\n        {}\n\n", label));
                }
            } else {
                out.push('\n');
            }
        }

        let first_hit = day.first().unwrap();
        let date_str = fmt_day_label(first_hit.ts_ms);

        let ws_labels: BTreeSet<String> = day
            .iter()
            .filter_map(|h| h.origin_workspace_id.as_deref())
            .filter_map(|id| crate::workspace::workspace_label_by_id(id))
            .collect();
        let ws_tag = if project_count > 1 && ws_labels.len() == 1 {
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
            let tag_part = match &ws_tag {
                Some(t) => format!("  \x1b[2;36m{}\x1b[0m", t),
                None => String::new(),
            };
            out.push_str(&format!(
                "\x1b[1;97m{}\x1b[0m{}\n",
                date_str,
                tag_part,
            ));
        } else {
            let tag_part = match &ws_tag {
                Some(t) => format!("  {}", t),
                None => String::new(),
            };
            out.push_str(&format!("{}{}\n", date_str, tag_part));
        }

        for note in display_notes_for_day(day) {
            global_idx += 1;
            let entry = render_capsule_entry(note.hit, note.echoes, global_idx, output);
            out.push_str(&entry);
            out.push('\n');
        }
    }

    out
}

fn render_title(out: &mut String, topic: &str, output: OutputFormat) {
    let topic = topic.trim();
    if is_ansi(output) {
        out.push_str("\x1b[1m");
        out.push_str(topic);
        out.push_str("\x1b[0m\n");
    } else {
        out.push_str(topic);
        out.push('\n');
    }
}

fn span_suffix(days: i64) -> String {
    if days >= 90 {
        format!(" · {}-month arc", (days / 30).max(3))
    } else if days >= 30 {
        format!(" · {}-week arc", (days / 7).max(4))
    } else if days >= 7 {
        format!(" · {}-day arc", days)
    } else {
        String::new()
    }
}

fn gap_label(gap_ms: i64) -> String {
    let days = (gap_ms / (24 * 60 * 60 * 1000)).max(1);
    if days >= 90 {
        format!("{} months earlier · long return", (days / 30).max(3))
    } else if days >= 30 {
        format!("{} weeks earlier · shelved for a while", (days / 7).max(4))
    } else {
        format!("{} days earlier", days)
    }
}

fn group_into_days<'a>(hits: &'a [crate::CapsuleHit]) -> Vec<Vec<&'a crate::CapsuleHit>> {
    let mut days: Vec<Vec<&crate::CapsuleHit>> = Vec::new();
    let mut current: Vec<&crate::CapsuleHit> = Vec::new();
    let mut current_day = String::new();

    for hit in hits {
        let day = fmt_day_key(hit.ts_ms);
        if current.is_empty() {
            current_day = day;
            current.push(hit);
            continue;
        }
        if day == current_day {
            current.push(hit);
        } else {
            days.push(std::mem::take(&mut current));
            current_day = day;
            current.push(hit);
        }
    }
    if !current.is_empty() {
        days.push(current);
    }
    days
}

struct DisplayNote<'a> {
    hit: &'a crate::CapsuleHit,
    echoes: usize,
}

fn display_notes_for_day<'a>(hits: &[&'a crate::CapsuleHit]) -> Vec<DisplayNote<'a>> {
    let mut notes: Vec<DisplayNote<'a>> = Vec::new();
    for hit in hits {
        let decision = note_text(hit);
        if let Some(existing) = notes
            .iter_mut()
            .find(|note| similar_notes(&decision, &note_text(note.hit)))
        {
            existing.echoes += 1;
        } else {
            notes.push(DisplayNote { hit, echoes: 0 });
        }
    }
    notes
}

fn render_capsule_entry(
    hit: &crate::CapsuleHit,
    echoes: usize,
    index: usize,
    output: OutputFormat,
) -> String {
    let cap = &hit.capsule;
    let mut out = String::new();

    let decision_display = note_text(hit);

    if is_ansi(output) {
        push_wrapped_ansi(
            &mut out,
            &format!("\n  \x1b[2m{:>2}\x1b[0m  \x1b[1m", index),
            "\x1b[0m",
            "      \x1b[1m",
            "\x1b[0m",
            &decision_display,
            WRAP_WIDTH - 6,
        );
    } else {
        push_wrapped_plain(
            &mut out,
            &format!("\n  {:>2}  ", index),
            "      ",
            &decision_display,
            WRAP_WIDTH - 6,
        );
    }

    if !cap.rationale.trim().is_empty() {
        let rationale = truncate(&humanize_rationale(&first_sentence(cap.rationale.trim())), 112);
        if is_ansi(output) {
            push_wrapped_ansi(
                &mut out,
                "\n      \x1b[2m",
                "\x1b[0m",
                "      \x1b[2m",
                "\x1b[0m",
                &rationale,
                WRAP_WIDTH - 6,
            );
        } else {
            push_wrapped_plain(&mut out, "\n      ", "      ", &rationale, WRAP_WIDTH - 6);
        }
    }

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
                out.push_str(&format!("\n      \x1b[33m▲ {}\x1b[0m", fm));
            } else {
                out.push_str(&format!("\n      ▲ {}", fm));
            }
        }
    }

    if !cap.symbols.is_empty() {
        let syms: Vec<&str> = cap.symbols.iter().map(|s| s.as_str()).take(4).collect();
        let sym_str = syms.join("  ");
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2m{}\x1b[0m", sym_str));
        } else {
            out.push_str(&format!("\n      {}", sym_str));
        }
    }

    if echoes > 0 {
        let line = if echoes == 1 {
            "same idea appears once more".to_string()
        } else {
            format!("same idea appears {} more times", echoes)
        };
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2m{}\x1b[0m", line));
        } else {
            out.push_str(&format!("\n      {}", line));
        }
    }

    out
}

fn note_text(hit: &crate::CapsuleHit) -> String {
    let decision = hit.capsule.decision.trim();
    let text = if decision.is_empty() {
        truncate(hit.capsule.intent.trim(), 160)
    } else {
        decision.to_string()
    };
    humanize_note_text(&text)
}

fn fmt_range_date(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts_ms.to_string())
}

fn fmt_day_key(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts_ms.to_string())
}

fn fmt_day_label(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.format("%b %d").to_string())
        .unwrap_or_else(|| ts_ms.to_string())
}

fn push_wrapped_plain(out: &mut String, first_prefix: &str, cont_prefix: &str, text: &str, width: usize) {
    let lines = wrap_words(text, width);
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            out.push_str(first_prefix);
        } else {
            out.push('\n');
            out.push_str(cont_prefix);
        }
        out.push_str(line);
    }
}

fn push_wrapped_ansi(
    out: &mut String,
    first_prefix: &str,
    first_suffix: &str,
    cont_prefix: &str,
    cont_suffix: &str,
    text: &str,
    width: usize,
) {
    let lines = wrap_words(text, width);
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            out.push_str(first_prefix);
            out.push_str(line);
            out.push_str(first_suffix);
        } else {
            out.push('\n');
            out.push_str(cont_prefix);
            out.push_str(line);
            out.push_str(cont_suffix);
        }
    }
}

fn wrap_words(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;
    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current.is_empty() {
            current.push_str(word);
            current_len = word_len;
        } else if current_len + 1 + word_len <= width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            lines.push(current);
            current = word.to_string();
            current_len = word_len;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn similar_notes(a: &str, b: &str) -> bool {
    let a_tokens = note_tokens(a);
    let b_tokens = note_tokens(b);
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return false;
    }
    let intersection = a_tokens.intersection(&b_tokens).count() as f32;
    let union = a_tokens.union(&b_tokens).count() as f32;
    (intersection / union) >= 0.48
}

fn note_tokens(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|t| t.len() >= 4)
        .collect()
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

fn humanize_rationale(s: &str) -> String {
    let s = s.trim();
    let replacements = [
        ("The user wants ", "Wanted "),
        ("The user wanted ", "Wanted "),
        ("The user needs ", "Needed "),
        ("The user needed ", "Needed "),
        ("The user expects ", "Expected "),
        ("The user expected ", "Expected "),
        ("The user expresses ", "Expressed "),
        ("The user expressed ", "Expressed "),
        ("The user is assessing ", "Assessing "),
        ("The user was assessing ", "Assessing "),
        ("The user agrees ", "Agreed "),
        ("The user agreed ", "Agreed "),
        ("The user highlighted ", "Flagged "),
        ("The user asked ", "Asked "),
        ("The user is interested in ", "Wanted to understand "),
        ("The user was interested in ", "Wanted to understand "),
        ("The user seeks to understand ", "Wanted to understand "),
        ("The user sought to understand ", "Wanted to understand "),
        ("User initiated ", "Started "),
        ("User initiates ", "Started "),
        ("User seeks to understand ", "Wanted to understand "),
        ("User seeks ", "Wanted "),
        ("User is interested in ", "Wanted to understand "),
        ("User was interested in ", "Wanted to understand "),
        ("User wants ", "Wanted "),
        ("User wanted ", "Wanted "),
        ("User needs ", "Needed "),
        ("User needed ", "Needed "),
        ("User expects ", "Expected "),
        ("User expected ", "Expected "),
        ("User expresses ", "Expressed "),
        ("User expressed ", "Expressed "),
        ("User agrees ", "Agreed "),
        ("User agreed ", "Agreed "),
        ("User highlighted ", "Flagged "),
        ("User asked ", "Asked "),
        ("The conversation focuses on understanding how ", "Wanted to understand how "),
        ("The conversation focused on understanding how ", "Wanted to understand how "),
        ("The conversation focuses on ", "Looked at "),
        ("The conversation focused on ", "Looked at "),
    ];
    for (from, to) in replacements {
        if let Some(rest) = s.strip_prefix(from) {
            return polish_rationale(format!("{to}{rest}"));
        }
    }
    polish_rationale(s.to_string())
}

fn polish_rationale(s: String) -> String {
    s.replace(" but wants ", " but wanted ")
        .replace("user expects", "expected")
        .replace("User expects", "Expected")
        .replace(" and seeks clarity ", " and needed clarity ")
        .replace(" seeks clarity ", " needed clarity ")
        .replace("indicating a need for", "pointing toward")
        .replace("their knowledge or project", "the design")
        .replace("Wanted to understand analyzing ", "Wanted better ways to analyze ")
        .replace("Wanted to understand expanding ", "Wanted to expand ")
        .replace("Needed code-level", "Needed a code-level")
}

fn humanize_note_text(s: &str) -> String {
    let s = s.trim();
    let replacements = [
        ("User requests ", "Requested "),
        ("User requested ", "Requested "),
        ("User initiates ", "Started "),
        ("User initiated ", "Started "),
        ("The user requests ", "Requested "),
        ("The user requested ", "Requested "),
    ];
    for (from, to) in replacements {
        if let Some(rest) = s.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    s.to_string()
}
