use crate::cli::OutputFormat;
use chrono::TimeZone;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::BTreeSet;
use std::time::Duration;

const WRAP_WIDTH: usize = 80;
const DORMANCY_THRESHOLD_MS: i64 = 7 * 24 * 60 * 60 * 1000;
/// Two notes within this gap are part of the same cluster.
const CLUSTER_GAP_MS: i64 = 4 * 60 * 60 * 1000;

// ── Public entry point ──────────────────────────────────────────────────────

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
    timeline: bool,
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

    // Sort chronologically for analysis; renderers handle display order.
    hits.sort_by_key(|h| h.ts_ms);

    let view = ThreadView::from_hits(&hits, &ws);

    let (narrative, narrative_warning) = if no_llm {
        (None, None)
    } else {
        match crate::narrative::llm_thread_narrative(llm_model.as_deref(), &query, &view).await {
            Ok(n) => (Some(n), None),
            Err(e) => {
                tracing::debug!(error = %e, "thread narrative unavailable");
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

    let rendered = if timeline {
        render_timeline(&query, narrative.as_deref(), narrative_warning.as_deref(), &view, output)
    } else {
        render_trail(&query, narrative.as_deref(), narrative_warning.as_deref(), &view, output)
    };
    print!("{rendered}");

    Ok(())
}

// ── ThreadView: pre-analyzed structure ──────────────────────────────────────

/// A cluster of nearby notes (within CLUSTER_GAP_MS of each other).
pub struct Cluster {
    pub notes: Vec<DisplayNote>,
    pub earliest_ts: i64,
    pub latest_ts: i64,
    pub provenance: String,
}

pub struct DisplayNote {
    pub decision: String,
    pub rationale: Option<String>,
    pub failure_mode: Option<String>,
    pub symbols: Vec<String>,
    pub echoes: usize,
    pub ts_ms: i64,
}

pub struct LongGap {
    pub from_ts: i64,
    pub to_ts: i64,
    pub days: i64,
}

pub struct ThreadView {
    /// All clusters, sorted chronologically (oldest first).
    pub clusters: Vec<Cluster>,
    /// Gaps >=7d between consecutive clusters, chronological.
    pub long_gaps: Vec<LongGap>,
    /// Total moments before dedup.
    pub total_moments: usize,
    /// Total notes after dedup.
    pub total_notes: usize,
    pub earliest_ts: i64,
    pub latest_ts: i64,
    pub span_days: i64,
    pub project_count: usize,
}

impl ThreadView {
    pub fn from_hits(hits: &[crate::CapsuleHit], ws: &crate::WorkspacePaths) -> Self {
        let project_ids: BTreeSet<&str> = hits
            .iter()
            .filter_map(|h| h.origin_workspace_id.as_deref())
            .collect();
        let project_count = project_ids.len().max(1);

        // Group into clusters by temporal proximity.
        let raw_clusters = cluster_hits(hits);
        let mut clusters: Vec<Cluster> = Vec::new();
        let mut total_notes = 0usize;

        for raw in &raw_clusters {
            let folded = fold_similar(raw);
            total_notes += folded.len();

            let prov = cluster_provenance(raw, ws);
            let earliest = raw.first().unwrap().ts_ms;
            let latest = raw.last().unwrap().ts_ms;

            clusters.push(Cluster {
                notes: folded,
                earliest_ts: earliest,
                latest_ts: latest,
                provenance: prov,
            });
        }

        // Find long gaps between consecutive clusters.
        let mut long_gaps: Vec<LongGap> = Vec::new();
        for i in 1..clusters.len() {
            let from_ts = clusters[i - 1].latest_ts;
            let to_ts = clusters[i].earliest_ts;
            let gap = to_ts - from_ts;
            if gap >= DORMANCY_THRESHOLD_MS {
                let days = (gap / (24 * 60 * 60 * 1000)).max(1);
                long_gaps.push(LongGap { from_ts, to_ts, days });
            }
        }

        let earliest_ts = hits.first().unwrap().ts_ms;
        let latest_ts = hits.last().unwrap().ts_ms;
        let span_days = ((latest_ts - earliest_ts) / (24 * 60 * 60 * 1000)).max(0);

        ThreadView {
            clusters,
            long_gaps,
            total_moments: hits.len(),
            total_notes,
            earliest_ts,
            latest_ts,
            span_days,
            project_count,
        }
    }
}

fn cluster_hits<'a>(hits: &'a [crate::CapsuleHit]) -> Vec<Vec<&'a crate::CapsuleHit>> {
    let mut clusters: Vec<Vec<&crate::CapsuleHit>> = Vec::new();
    let mut current: Vec<&crate::CapsuleHit> = Vec::new();

    for hit in hits {
        if current.is_empty() {
            current.push(hit);
            continue;
        }
        let prev_ts = current.last().unwrap().ts_ms;
        if hit.ts_ms - prev_ts <= CLUSTER_GAP_MS {
            current.push(hit);
        } else {
            clusters.push(std::mem::take(&mut current));
            current.push(hit);
        }
    }
    if !current.is_empty() {
        clusters.push(current);
    }
    clusters
}

fn fold_similar(hits: &[&crate::CapsuleHit]) -> Vec<DisplayNote> {
    let mut notes: Vec<DisplayNote> = Vec::new();
    for hit in hits {
        let text = note_text(hit);
        if let Some(existing) = notes.iter_mut().find(|n| similar_notes(&text, &n.decision)) {
            existing.echoes += 1;
        } else {
            let cap = &hit.capsule;
            let rationale = if cap.rationale.trim().is_empty() {
                None
            } else {
                Some(truncate(
                    &humanize_rationale(&first_sentence(cap.rationale.trim())),
                    112,
                ))
            };
            let failure_mode = match cap.failure_mode {
                crate::types::FailureMode::None => None,
                crate::types::FailureMode::Drift => Some("drift".into()),
                crate::types::FailureMode::Rediscovery => Some("rediscovery".into()),
                crate::types::FailureMode::DecisionConflict => Some("decision conflict".into()),
                crate::types::FailureMode::RetrySpiral => Some("retry spiral".into()),
                crate::types::FailureMode::FalseProgress => Some("false progress".into()),
                crate::types::FailureMode::UnboundedHorizon => Some("unbounded horizon".into()),
            };
            let symbols: Vec<String> = cap.symbols.iter().take(4).cloned().collect();
            notes.push(DisplayNote {
                decision: text,
                rationale,
                failure_mode,
                symbols,
                echoes: 0,
                ts_ms: hit.ts_ms,
            });
        }
    }
    notes
}

fn cluster_provenance(hits: &[&crate::CapsuleHit], ws: &crate::WorkspacePaths) -> String {
    // Collect unique provenances across all notes in this cluster.
    let mut labels: BTreeSet<String> = BTreeSet::new();
    for hit in hits {
        if let Some(label) = provenance_label(hit, ws) {
            labels.insert(label);
        }
    }
    if labels.len() == 1 {
        labels.into_iter().next().unwrap()
    } else if labels.is_empty() {
        workspace_label_from_root(&ws.root).unwrap_or_default()
    } else {
        // Multiple sources within the cluster — just show workspace.
        let ws_name = workspace_label_from_root(&ws.root).unwrap_or_default();
        let sources: Vec<&String> = labels.iter().collect();
        if sources.iter().all(|s| s.starts_with(&ws_name)) {
            ws_name
        } else {
            labels.into_iter().collect::<Vec<_>>().join(", ")
        }
    }
}

// ── Trail renderer (default) ────────────────────────────────────────────────

fn render_trail(
    topic: &str,
    narrative: Option<&str>,
    narrative_warning: Option<&str>,
    view: &ThreadView,
    output: OutputFormat,
) -> String {
    let mut out = String::new();

    render_header(&mut out, topic, view, output);
    render_narrative_block(&mut out, narrative, narrative_warning, output);

    // Trail: most recent cluster first, oldest last.
    let cluster_count = view.clusters.len();
    for (ci, cluster) in view.clusters.iter().rev().enumerate() {
        let is_most_recent = ci == 0;
        let is_origin = ci == cluster_count - 1;

        // Section label
        let date_str = fmt_day_label(cluster.earliest_ts);
        let section_label = if is_most_recent && is_origin {
            format!("{}", date_str)
        } else if is_most_recent {
            format!("Current shape · {}", date_str)
        } else if is_origin {
            format!("Origin · {}", date_str)
        } else {
            format!("Earlier · {}", date_str)
        };

        // Gap from previous (newer) cluster
        if ci > 0 {
            let newer_cluster = &view.clusters[cluster_count - ci];
            let gap_ms = newer_cluster.earliest_ts - cluster.latest_ts;
            if gap_ms >= DORMANCY_THRESHOLD_MS {
                let label = gap_label(gap_ms);
                if is_ansi(output) {
                    out.push_str(&format!("\n\x1b[2m        {}\x1b[0m\n", label));
                } else {
                    out.push_str(&format!("\n        {}\n", label));
                }
            }
        }

        out.push('\n');
        if is_ansi(output) {
            out.push_str(&format!("\x1b[1;97m{}\x1b[0m", section_label));
        } else {
            out.push_str(&section_label);
        }

        // Provenance for the cluster — dim, on the same line or next
        if !cluster.provenance.is_empty() {
            if is_ansi(output) {
                out.push_str(&format!("  \x1b[2m{}\x1b[0m", cluster.provenance));
            } else {
                out.push_str(&format!("  {}", cluster.provenance));
            }
        }
        out.push('\n');

        for (ni, note) in cluster.notes.iter().enumerate() {
            render_note(&mut out, note, ni + 1, output);
        }
    }

    out.push('\n');
    out
}

// ── Timeline renderer (--timeline) ──────────────────────────────────────────

fn render_timeline(
    topic: &str,
    narrative: Option<&str>,
    narrative_warning: Option<&str>,
    view: &ThreadView,
    output: OutputFormat,
) -> String {
    let mut out = String::new();

    render_header(&mut out, topic, view, output);
    render_narrative_block(&mut out, narrative, narrative_warning, output);

    // Timeline: most recent first (reverse chronological).
    let cluster_count = view.clusters.len();
    for (ci, cluster) in view.clusters.iter().rev().enumerate() {
        // Gap from the previous (newer) cluster
        if ci > 0 {
            let newer_cluster = &view.clusters[cluster_count - ci];
            let gap_ms = newer_cluster.earliest_ts - cluster.latest_ts;
            if gap_ms >= DORMANCY_THRESHOLD_MS {
                let label = gap_label(gap_ms);
                if is_ansi(output) {
                    out.push_str(&format!("\n\x1b[2m        {}\x1b[0m\n", label));
                } else {
                    out.push_str(&format!("\n        {}\n", label));
                }
            }
        }

        let date_str = fmt_day_label(cluster.earliest_ts);
        out.push('\n');
        if is_ansi(output) {
            out.push_str(&format!("\x1b[1;97m{}\x1b[0m", date_str));
            if !cluster.provenance.is_empty() {
                out.push_str(&format!("  \x1b[2m{}\x1b[0m", cluster.provenance));
            }
        } else {
            out.push_str(&date_str);
            if !cluster.provenance.is_empty() {
                out.push_str(&format!("  {}", cluster.provenance));
            }
        }
        out.push('\n');

        for (ni, note) in cluster.notes.iter().rev().enumerate() {
            render_note(&mut out, note, ni + 1, output);
        }
    }

    out.push('\n');
    out
}

// ── Shared rendering helpers ────────────────────────────────────────────────

fn render_header(out: &mut String, topic: &str, view: &ThreadView, output: OutputFormat) {
    let topic = topic.trim();
    if is_ansi(output) {
        out.push_str(&format!("\x1b[1m{}\x1b[0m\n", topic));
    } else {
        out.push_str(topic);
        out.push('\n');
    }

    let project_label = if view.project_count > 1 {
        format!(" across {} projects", view.project_count)
    } else {
        String::new()
    };

    let earliest = fmt_range_date(view.earliest_ts);
    let latest = fmt_range_date(view.latest_ts);

    let meta = format!(
        "{} moment{} · {} note{}{} · {} back to {}{}",
        view.total_moments,
        if view.total_moments == 1 { "" } else { "s" },
        view.total_notes,
        if view.total_notes == 1 { "" } else { "s" },
        project_label,
        latest,
        earliest,
        span_suffix(view.span_days),
    );

    if is_ansi(output) {
        out.push_str(&format!("\x1b[2m{}\x1b[0m\n", meta));
    } else {
        out.push_str(&meta);
        out.push('\n');
    }
}

fn render_narrative_block(
    out: &mut String,
    narrative: Option<&str>,
    warning: Option<&str>,
    output: OutputFormat,
) {
    if let Some(n) = narrative {
        out.push('\n');
        let rendered = crate::narrative::render_narrative(output, n);
        for line in rendered.lines() {
            out.push_str(line);
            out.push('\n');
        }
    } else if let Some(w) = warning {
        out.push('\n');
        if is_ansi(output) {
            push_wrapped_ansi(out, "\x1b[2m", "\x1b[0m", "\x1b[2m", "\x1b[0m", w, WRAP_WIDTH);
        } else {
            push_wrapped_plain(out, "", "", w, WRAP_WIDTH);
        }
        out.push('\n');
    }
}

fn render_note(out: &mut String, note: &DisplayNote, index: usize, output: OutputFormat) {
    // Decision — the primary signal
    if is_ansi(output) {
        push_wrapped_ansi(
            out,
            &format!("\n  \x1b[2m{:>2}\x1b[0m  ", index),
            "",
            "      ",
            "",
            &note.decision,
            WRAP_WIDTH - 6,
        );
    } else {
        push_wrapped_plain(
            out,
            &format!("\n  {:>2}  ", index),
            "      ",
            &note.decision,
            WRAP_WIDTH - 6,
        );
    }

    // Rationale — dim
    if let Some(ref rationale) = note.rationale {
        if is_ansi(output) {
            push_wrapped_ansi(
                out,
                "\n      \x1b[2m",
                "\x1b[0m",
                "      \x1b[2m",
                "\x1b[0m",
                rationale,
                WRAP_WIDTH - 6,
            );
        } else {
            push_wrapped_plain(out, "\n      ", "      ", rationale, WRAP_WIDTH - 6);
        }
    }

    // Failure mode
    if let Some(ref fm) = note.failure_mode {
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[33m▲ {}\x1b[0m", fm));
        } else {
            out.push_str(&format!("\n      ▲ {}", fm));
        }
    }

    // Symbols
    if !note.symbols.is_empty() {
        let sym_str = note.symbols.join("  ");
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2m{}\x1b[0m", sym_str));
        } else {
            out.push_str(&format!("\n      {}", sym_str));
        }
    }

    // Echoes
    if note.echoes > 0 {
        let line = if note.echoes == 1 {
            "same idea appears once more"
        } else {
            &format!("same idea appears {} more times", note.echoes)
        };
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2m{}\x1b[0m", line));
        } else {
            out.push_str(&format!("\n      {}", line));
        }
    }

    out.push('\n');
}

fn is_ansi(output: OutputFormat) -> bool {
    output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none()
}

// ── ThreadView to LLM context ───────────────────────────────────────────────

impl ThreadView {
    /// Build the structured context string for the LLM, giving it pre-analyzed
    /// clusters, gaps, and motifs instead of raw hit rows.
    pub fn to_llm_context(&self) -> String {
        let mut ctx = String::new();

        ctx.push_str(&format!(
            "Thread: {} moments, {} notes, {} clusters, spanning {} days\n\n",
            self.total_moments,
            self.total_notes,
            self.clusters.len(),
            self.span_days,
        ));

        for (ci, cluster) in self.clusters.iter().enumerate() {
            let date = fmt_range_date(cluster.earliest_ts);
            let gap_ctx = if ci > 0 {
                let prev = &self.clusters[ci - 1];
                let gap_days =
                    (cluster.earliest_ts - prev.latest_ts) / (24 * 60 * 60 * 1000);
                if gap_days >= 90 {
                    format!(" (returned after {} months)", (gap_days / 30).max(3))
                } else if gap_days >= 30 {
                    format!(" (returned after {} weeks)", (gap_days / 7).max(4))
                } else if gap_days >= 7 {
                    format!(" (gap: {} days)", gap_days)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            ctx.push_str(&format!(
                "Cluster {} · {}{} · {}\n",
                ci + 1,
                date,
                gap_ctx,
                cluster.provenance,
            ));
            for note in &cluster.notes {
                ctx.push_str(&format!("  decision: {}\n", note.decision));
                if let Some(ref r) = note.rationale {
                    ctx.push_str(&format!("  rationale: {}\n", r));
                }
                if let Some(ref fm) = note.failure_mode {
                    ctx.push_str(&format!("  failure_mode: {}\n", fm));
                }
                if !note.symbols.is_empty() {
                    ctx.push_str(&format!("  symbols: {}\n", note.symbols.join(", ")));
                }
                if note.echoes > 0 {
                    ctx.push_str(&format!("  echoes: {} similar notes folded\n", note.echoes));
                }
            }
            ctx.push('\n');
        }

        if !self.long_gaps.is_empty() {
            ctx.push_str("Long gaps:\n");
            for g in &self.long_gaps {
                ctx.push_str(&format!(
                    "  {} → {} ({} days)\n",
                    fmt_range_date(g.from_ts),
                    fmt_range_date(g.to_ts),
                    g.days,
                ));
            }
            ctx.push('\n');
        }

        ctx
    }
}

// ── Text helpers ────────────────────────────────────────────────────────────

fn provenance_label(hit: &crate::CapsuleHit, ws: &crate::WorkspacePaths) -> Option<String> {
    let workspace = hit
        .origin_workspace_id
        .as_deref()
        .and_then(crate::workspace::workspace_label_by_id)
        .or_else(|| workspace_label_from_root(&ws.root))?;

    let source = hit
        .meta
        .source_pointer
        .as_deref()
        .and_then(crate::workspace::resolve_source_label)
        .or_else(|| fallback_source_label(hit));

    match source {
        Some(source) if !source.trim().is_empty() => Some(format!("{workspace} · {source}")),
        _ => Some(workspace),
    }
}

fn workspace_label_from_root(root: &std::path::Path) -> Option<String> {
    root.file_name()
        .and_then(|n| n.to_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn fallback_source_label(hit: &crate::CapsuleHit) -> Option<String> {
    let source = hit.meta.source.trim();
    let path = hit.meta.request_path.trim();
    if source.is_empty() {
        return None;
    }
    match (source, path.is_empty()) {
        ("changelog", false) => Some(format!("CHANGELOG {path}")),
        ("git", false) => Some(format!("git {path}")),
        (_, true) => Some(source.to_string()),
        _ => Some(format!("{source} · {path}")),
    }
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

fn fmt_range_date(ts_ms: i64) -> String {
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

// ── Word wrapping ───────────────────────────────────────────────────────────

fn push_wrapped_plain(
    out: &mut String,
    first_prefix: &str,
    cont_prefix: &str,
    text: &str,
    width: usize,
) {
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

// ── Similarity ──────────────────────────────────────────────────────────────

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

// ── Text normalization ──────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let end = s
            .char_indices()
            .nth(max.saturating_sub(1))
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        format!("{}…", &s[..end])
    }
}

fn first_sentence(s: &str) -> String {
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
