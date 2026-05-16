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
        pb.set_message("pulling on the thread...");
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

    let mut view = ThreadView::from_hits(&hits, &ws);

    // Collect all hit ids so we can exclude them from neighbor scans.
    let thread_hit_ids: BTreeSet<String> = hits.iter().map(|h| h.id.clone()).collect();
    view.enrich_spatial_context(&ws, &thread_hit_ids).await;

    let llm_spinner = if !no_llm {
        if let Some(target) = crate::narrative::spinner_draw_target(output) {
            let pb = ProgressBar::new_spinner();
            pb.set_draw_target(target);
            pb.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                    .unwrap()
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
            );
            pb.enable_steady_tick(Duration::from_millis(80));
            pb.set_message("reading between the lines...");
            Some(pb)
        } else {
            None
        }
    } else {
        None
    };

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

    if let Some(ref sp) = llm_spinner {
        sp.finish_and_clear();
    }

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
    /// Distinct source_pointer URIs found in this cluster (for linking back).
    pub source_links: Vec<String>,
    /// What the broader session was about (from checkpoint narrative).
    pub session_context: Option<String>,
    /// Other topics that were being discussed nearby in time (not in this thread).
    pub nearby_topics: Vec<String>,
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

            let mut source_links: Vec<String> = raw
                .iter()
                .filter_map(|h| h.meta.source_pointer.as_deref())
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            source_links.truncate(3);

            clusters.push(Cluster {
                notes: folded,
                earliest_ts: earliest,
                latest_ts: latest,
                provenance: prov,
                source_links,
                session_context: None,
                nearby_topics: Vec::new(),
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

    /// Enrich clusters with spatial context: what session they belonged to
    /// and what other topics were being discussed nearby.
    pub async fn enrich_spatial_context(
        &mut self,
        ws: &crate::WorkspacePaths,
        thread_hit_ids: &BTreeSet<String>,
    ) {
        const NEIGHBOR_WINDOW_MS: i64 = 30 * 60 * 1000; // ±30 minutes

        for cluster in &mut self.clusters {
            // 1. Checkpoint: find the session narrative covering this cluster's time range.
            if let Ok(db) = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
                .execute()
                .await
            {
                if let Ok(checkpoints) =
                    crate::storage_checkpoint::get_checkpoints_in_range(
                        &db,
                        &ws.id,
                        cluster.earliest_ts,
                        cluster.latest_ts,
                    )
                    .await
                {
                    // Take the first checkpoint that covers this time range.
                    if let Some(cp) = checkpoints.first() {
                        cluster.session_context = extract_session_topic(&cp.narrative);
                    }
                }
            }

            // 2. Nearby topics: scan a time window around the cluster for
            //    capsules not in the thread.
            let since = cluster.earliest_ts - NEIGHBOR_WINDOW_MS;
            let until = cluster.latest_ts + NEIGHBOR_WINDOW_MS;
            if let Ok(nearby) = crate::storage::scan_capsules_lancedb(
                ws,
                20,
                None,
                None,
                None,
                Some(since),
                Some(until),
            )
            .await
            {
                let mut topics: Vec<String> = Vec::new();
                let mut seen = BTreeSet::new();
                for hit in &nearby {
                    // Skip notes already in the thread.
                    if thread_hit_ids.contains(&hit.id) {
                        continue;
                    }
                    // Skip git/changelog/init — those are facts, not conversation.
                    if matches!(hit.meta.source.as_str(), "git" | "changelog" | "init") {
                        continue;
                    }
                    let text = hit.capsule.decision.trim();
                    if text.is_empty() || text.len() < 20 {
                        continue;
                    }
                    if is_placeholder_decision(text) {
                        continue;
                    }
                    // Skip low-signal neighbor patterns.
                    let lower = text.to_ascii_lowercase();
                    if lower.starts_with("no action")
                        || lower.starts_with("no actionable")
                        || lower.starts_with("no specific")
                        || lower.starts_with("user simply")
                        || lower.starts_with("user rejects")
                        || lower.starts_with("user disputes")
                        || lower.starts_with("user is unsure")
                        || lower.starts_with("acknowledge")
                        || lower.contains("conversation ended")
                        || lower.contains("conversation effectively")
                        || lower.contains("no further request")
                        || lower.contains("end interaction")
                        || lower.contains("close interaction")
                        || lower.contains("awaiting context")
                        || lower.starts_with("none;")
                        || lower.starts_with("none.")
                    {
                        continue;
                    }
                    // Skip single-word non-space decisions under 30 chars.
                    if !text.contains(' ') && text.len() < 30 {
                        continue;
                    }
                    let short = truncate(text, 60);
                    // Deduplicate similar neighbor topics.
                    let key = short.to_ascii_lowercase();
                    if seen.contains(&key) {
                        continue;
                    }
                    // Skip if this topic overlaps with any note in the cluster.
                    let dominated = cluster
                        .notes
                        .iter()
                        .any(|n| similar_notes_loose(&short, &n.decision));
                    if dominated {
                        continue;
                    }
                    seen.insert(key);
                    topics.push(short);
                    if topics.len() >= 3 {
                        break;
                    }
                }
                cluster.nearby_topics = topics;
            }
        }
    }
}

/// Extract the "WHAT WAS WORKED ON" section from a checkpoint narrative,
/// or fall back to the first non-empty line.
fn extract_session_topic(narrative: &str) -> Option<String> {
    let narrative = narrative.trim();
    if narrative.is_empty() {
        return None;
    }
    // Look for the structured section header.
    let marker = "WHAT WAS WORKED ON";
    if let Some(start) = narrative.find(marker) {
        let after = &narrative[start + marker.len()..];
        // Skip the header line itself and any blank lines.
        let content: String = after
            .lines()
            .skip_while(|l| l.trim().is_empty() || l.trim() == marker)
            .take(2)
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !content.is_empty() {
            return Some(truncate(&content, 120));
        }
    }
    // Fallback: first non-empty line.
    for line in narrative.lines() {
        let l = line.trim();
        if !l.is_empty() && !l.chars().all(|c| c == '─' || c == '-' || c == '=') {
            return Some(truncate(l, 120));
        }
    }
    None
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
        if is_placeholder_decision(&text) {
            continue;
        }
        if let Some(existing) = notes.iter_mut().find(|n| similar_notes(&text, &n.decision)) {
            existing.echoes += 1;
        } else {
            let cap = &hit.capsule;
            let rationale = if cap.rationale.trim().is_empty() {
                None
            } else {
                Some(truncate(
                    &humanize_rationale(&first_sentence(cap.rationale.trim())),
                    80,
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
            let symbols: Vec<String> = cap
                .symbols
                .iter()
                .filter(|s| s.contains('/') || s.contains('.'))
                .take(2)
                .cloned()
                .collect();
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

/// Default view: narrative-first thread. The story is the content;
/// anchor clusters are quote blocks that ground it. Use --timeline for
/// the full reverse-chronological cluster dump.
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

    // Pick at most 3 anchor clusters: the turning points worth showing.
    let anchors = pick_thread_anchors(view);
    let anchors_total = view.clusters.len();
    let anchors_shown = anchors.len();

    for (i, idx) in anchors.iter().enumerate() {
        let cluster = &view.clusters[*idx];

        // Anchor role label, picked per position not per chronology.
        let role = anchor_role(i, anchors_shown, *idx, anchors_total);

        // Time-distance phrase relative to "today" / latest cluster.
        let when_phrase = relative_when(cluster.earliest_ts, view.latest_ts);

        out.push('\n');
        render_anchor_header(&mut out, role, &when_phrase, cluster, output);

        // Spatial context: what session this was part of, what else was nearby
        render_spatial_context_compact(&mut out, cluster, output);

        // Show up to 2 representative notes per anchor (the longest = most signal).
        let notes_to_show = pick_representative_notes(&cluster.notes, 2);
        for note in &notes_to_show {
            render_note_anchor(&mut out, note, output);
        }

        // Source links — dim, under the notes
        render_source_links_inline(&mut out, cluster, output);
    }

    // Hint: more notes available in --timeline
    let hidden_clusters = anchors_total.saturating_sub(anchors_shown);
    if hidden_clusters > 0 {
        out.push('\n');
        let hint = format!(
            "  {} more cluster{} in `unlost thread \"{}\" --timeline`",
            hidden_clusters,
            if hidden_clusters == 1 { "" } else { "s" },
            topic,
        );
        if is_ansi(output) {
            out.push_str(&format!("\x1b[2;3m{}\x1b[0m\n", hint));
        } else {
            out.push_str(&format!("{}\n", hint));
        }
    }

    out.push('\n');
    out
}

/// Pick at most 3 cluster indices that represent the thread's turning points.
/// Strategy: always include most-recent and origin; if there's room, pick the
/// "pivot" — the cluster with the most notes, a failure mode, or sitting just
/// after the longest gap.
fn pick_thread_anchors(view: &ThreadView) -> Vec<usize> {
    let n = view.clusters.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= 3 {
        return (0..n).rev().collect(); // newest first
    }

    let newest = n - 1;
    let origin = 0;

    // Find the pivot among middle clusters: largest gap-after-previous wins,
    // tiebreak by note count.
    let mut best_pivot: Option<(usize, i64, usize)> = None;
    for i in 1..n - 1 {
        let gap = view.clusters[i].earliest_ts - view.clusters[i - 1].latest_ts;
        let notes = view.clusters[i].notes.len();
        let score = (gap, notes);
        if best_pivot.map(|(_, g, n2)| (gap, notes) > (g, n2)).unwrap_or(true) {
            best_pivot = Some((i, score.0, score.1));
        }
    }

    let mut picks = vec![newest, origin];
    if let Some((idx, _, _)) = best_pivot {
        picks.push(idx);
    }
    picks.sort_by(|a, b| b.cmp(a)); // newest first
    picks.dedup();
    picks
}

/// Decide what to label this anchor based on its position in the chosen set.
fn anchor_role(position: usize, total_shown: usize, cluster_idx: usize, total_clusters: usize) -> &'static str {
    let is_newest = cluster_idx == total_clusters - 1;
    let is_origin = cluster_idx == 0;
    if total_shown == 1 {
        ""
    } else if is_newest {
        "where it landed"
    } else if is_origin {
        "where it started"
    } else {
        let _ = position;
        "the turn"
    }
}

fn relative_when(ts_ms: i64, latest_ts: i64) -> String {
    let days = ((latest_ts - ts_ms) / (24 * 60 * 60 * 1000)).max(0);
    if days == 0 {
        "today".to_string()
    } else if days == 1 {
        "yesterday".to_string()
    } else if days < 7 {
        format!("{} days ago", days)
    } else if days < 30 {
        let weeks = (days / 7).max(1);
        format!("{} week{} ago", weeks, if weeks == 1 { "" } else { "s" })
    } else if days < 90 {
        let weeks = days / 7;
        format!("{} weeks ago", weeks)
    } else {
        let months = (days / 30).max(3);
        format!("{} months ago", months)
    }
}

fn render_anchor_header(
    out: &mut String,
    role: &str,
    when_phrase: &str,
    cluster: &Cluster,
    output: OutputFormat,
) {
    let date_str = fmt_day_label(cluster.earliest_ts);

    if is_ansi(output) {
        // Role keyword in cyan, then "—" date, then dim "when_phrase"
        if !role.is_empty() {
            out.push_str(&format!("\x1b[36m{}\x1b[0m  ", role));
        }
        out.push_str(&format!("\x1b[1;97m{}\x1b[0m", date_str));
        if !when_phrase.is_empty() {
            out.push_str(&format!("  \x1b[2m({})\x1b[0m", when_phrase));
        }
        if !cluster.provenance.is_empty() {
            out.push_str(&format!("  \x1b[2;36m{}\x1b[0m", cluster.provenance));
        }
    } else {
        if !role.is_empty() {
            out.push_str(&format!("{}  ", role));
        }
        out.push_str(&date_str);
        if !when_phrase.is_empty() {
            out.push_str(&format!("  ({})", when_phrase));
        }
        if !cluster.provenance.is_empty() {
            out.push_str(&format!("  {}", cluster.provenance));
        }
    }
    out.push('\n');
}

/// Render spatial context as a single short line (not stacked).
fn render_spatial_context_compact(out: &mut String, cluster: &Cluster, output: OutputFormat) {
    if let Some(ref ctx) = cluster.session_context {
        let line = format!("inside: {}", truncate(ctx, 72));
        if is_ansi(output) {
            push_wrapped_ansi(
                out,
                "  \x1b[2;3m",
                "\x1b[0m",
                "  \x1b[2;3m",
                "\x1b[0m",
                &line,
                WRAP_WIDTH - 4,
            );
        } else {
            push_wrapped_plain(out, "  ", "  ", &line, WRAP_WIDTH - 4);
        }
        out.push('\n');
    }
    if !cluster.nearby_topics.is_empty() {
        // Single line, comma-separated, hard-truncated.
        let joined = cluster
            .nearby_topics
            .iter()
            .map(|t| truncate(t, 40))
            .collect::<Vec<_>>()
            .join(", ");
        let line = format!("alongside: {}", truncate(&joined, 100));
        if is_ansi(output) {
            push_wrapped_ansi(
                out,
                "  \x1b[2;3m",
                "\x1b[0m",
                "  \x1b[2;3m",
                "\x1b[0m",
                &line,
                WRAP_WIDTH - 4,
            );
        } else {
            push_wrapped_plain(out, "  ", "  ", &line, WRAP_WIDTH - 4);
        }
        out.push('\n');
    }
}

/// Pick up to `max` representative notes from a cluster: longest decisions
/// (more signal), preferring ones with failure modes.
fn pick_representative_notes(notes: &[DisplayNote], max: usize) -> Vec<&DisplayNote> {
    let mut scored: Vec<(usize, &DisplayNote)> = notes
        .iter()
        .map(|n| {
            let mut score = n.decision.len();
            if n.failure_mode.is_some() {
                score += 1000;
            }
            score += n.echoes * 20;
            (score, n)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(max).map(|(_, n)| n).collect()
}

/// Render a note inside an anchor — quote-style, indented with a bar.
fn render_note_anchor(out: &mut String, note: &DisplayNote, output: OutputFormat) {
    let text = truncate(&note.decision, 200);
    let echo_suffix = if note.echoes > 0 {
        if note.echoes == 1 {
            " (+1)".to_string()
        } else {
            format!(" (+{})", note.echoes)
        }
    } else {
        String::new()
    };
    let fm_prefix = if note.failure_mode.is_some() { "▲ " } else { "" };
    let body = format!("{fm_prefix}{text}{echo_suffix}");

    if is_ansi(output) {
        push_wrapped_ansi(
            out,
            "  \x1b[36m│\x1b[0m \x1b[36m",
            "\x1b[0m",
            "  \x1b[36m│\x1b[0m \x1b[36m",
            "\x1b[0m",
            &body,
            WRAP_WIDTH - 4,
        );
    } else {
        push_wrapped_plain(out, "  │ ", "  │ ", &body, WRAP_WIDTH - 4);
    }
    out.push('\n');
}

fn render_source_links_inline(out: &mut String, cluster: &Cluster, output: OutputFormat) {
    if cluster.source_links.is_empty() {
        return;
    }
    // First link only — keep it tight.
    let link = &cluster.source_links[0];
    if is_ansi(output) {
        out.push_str(&format!("  \x1b[2;36m→ {}\x1b[0m\n", link));
    } else {
        out.push_str(&format!("  → {}\n", link));
    }
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
                    out.push_str(&format!("\n\x1b[2m        ~ {}\x1b[0m\n", label));
                } else {
                    out.push_str(&format!("\n        ~ {}\n", label));
                }
            }
        }

        let date_str = fmt_day_label(cluster.earliest_ts);
        out.push('\n');
        if is_ansi(output) {
            out.push_str(&format!("\x1b[1;97m{}\x1b[0m", date_str));
            if !cluster.provenance.is_empty() {
                out.push_str(&format!("  \x1b[2;36m{}\x1b[0m", cluster.provenance));
            }
        } else {
            out.push_str(&date_str);
            if !cluster.provenance.is_empty() {
                out.push_str(&format!("  {}", cluster.provenance));
            }
        }
        out.push('\n');

        render_source_links(&mut out, cluster, output);
        render_spatial_context(&mut out, cluster, output);

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

fn render_spatial_context(out: &mut String, cluster: &Cluster, output: OutputFormat) {
    // Session context — what the broader conversation was about
    if let Some(ref ctx) = cluster.session_context {
        if is_ansi(output) {
            push_wrapped_ansi(
                out,
                "  \x1b[2;3mpart of: ",
                "\x1b[0m\n",
                "          \x1b[2;3m",
                "\x1b[0m\n",
                ctx,
                WRAP_WIDTH - 10,
            );
        } else {
            push_wrapped_plain(out, "  part of: ", "          ", ctx, WRAP_WIDTH - 10);
            out.push('\n');
        }
    }

    // Nearby topics — what else was being discussed at the same time
    if !cluster.nearby_topics.is_empty() {
        let label = if cluster.nearby_topics.len() == 1 {
            "  also discussing: "
        } else {
            "  also discussing: "
        };
        for (i, topic) in cluster.nearby_topics.iter().enumerate() {
            let prefix = if i == 0 { label } else { "                    " };
            if is_ansi(output) {
                out.push_str(&format!("\x1b[2m{}{}\x1b[0m\n", prefix, truncate(topic, 58)));
            } else {
                out.push_str(&format!("{}{}\n", prefix, truncate(topic, 58)));
            }
        }
    }
}

fn render_source_links(out: &mut String, cluster: &Cluster, output: OutputFormat) {
    if cluster.source_links.is_empty() {
        return;
    }
    for link in &cluster.source_links {
        if is_ansi(output) {
            out.push_str(&format!("  \x1b[2;36m{}\x1b[0m\n", link));
        } else {
            out.push_str(&format!("  {}\n", link));
        }
    }
}

/// Full multi-line note for --timeline view.
fn render_note(out: &mut String, note: &DisplayNote, index: usize, output: OutputFormat) {
    // Decision — the primary signal, cyan tint
    if is_ansi(output) {
        push_wrapped_ansi(
            out,
            &format!("\n  \x1b[2m{:>2}\x1b[0m  \x1b[36m", index),
            "\x1b[0m",
            "      \x1b[36m",
            "\x1b[0m",
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

    // Failure mode — yellow
    if let Some(ref fm) = note.failure_mode {
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[33m▲ {}\x1b[0m", fm));
        } else {
            out.push_str(&format!("\n      ▲ {}", fm));
        }
    }

    // Symbols — dim green
    if !note.symbols.is_empty() {
        let sym_str = note.symbols.join("  ");
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2;32m{}\x1b[0m", sym_str));
        } else {
            out.push_str(&format!("\n      {}", sym_str));
        }
    }

    // Echoes — dim italic
    if note.echoes > 0 {
        let line = if note.echoes == 1 {
            "same idea appears once more".to_string()
        } else {
            format!("same idea appears {} more times", note.echoes)
        };
        if is_ansi(output) {
            out.push_str(&format!("\n      \x1b[2;3m{}\x1b[0m", line));
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
            if let Some(ref session_ctx) = cluster.session_context {
                ctx.push_str(&format!("  session_topic: {}\n", session_ctx));
            }
            if !cluster.nearby_topics.is_empty() {
                ctx.push_str(&format!(
                    "  also_discussing: {}\n",
                    cluster.nearby_topics.join("; "),
                ));
            }
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
    // Map internal source/path combos to human-readable labels.
    if source == "changelog" {
        return if path.is_empty() {
            Some("CHANGELOG".into())
        } else {
            Some(format!("CHANGELOG {path}"))
        };
    }
    if source == "git" {
        return if path.is_empty() {
            Some("git".into())
        } else {
            Some(format!("git {path}"))
        };
    }
    if source == "record" {
        // /opencode/record, /companion, /v1/chat/completions, etc.
        if path.contains("opencode") || path.contains("companion") {
            return Some("OpenCode".into());
        }
        if path.contains("claude") {
            return Some("Claude".into());
        }
        if path.contains("copilot") {
            return Some("Copilot".into());
        }
        if path.contains("chat/completions") {
            return Some("chat".into());
        }
        return None; // Just show workspace, not "record"
    }
    if source == "init" {
        return Some("init".into());
    }
    None
}

fn note_text(hit: &crate::CapsuleHit) -> String {
    let decision = hit.capsule.decision.trim();
    let text = if decision.is_empty() {
        hit.capsule.intent.trim().to_string()
    } else {
        decision.to_string()
    };
    humanize_note_text(&text)
}

/// Returns true if a decision looks like a machine-generated placeholder
/// that carries no real signal.
fn is_placeholder_decision(s: &str) -> bool {
    let lower = s.trim().to_ascii_lowercase();
    let placeholders = [
        "proceed",
        "defer",
        "pause",
        "continue",
        "no_action_required",
        "accepted/implemented",
        "awaiting_commit_instruction",
        "awaiting_user_instruction",
        "proceed_with_planned_action",
        "awaiting context to determine next action",
        "no specific task",
    ];
    for p in placeholders {
        if lower == p || lower.starts_with(p) {
            return true;
        }
    }
    // Single word with underscores = likely a status marker
    if !lower.contains(' ') && lower.contains('_') {
        return true;
    }
    false
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

/// Looser similarity check for excluding neighbor topics that overlap with cluster notes.
fn similar_notes_loose(a: &str, b: &str) -> bool {
    let a_tokens = note_tokens(a);
    let b_tokens = note_tokens(b);
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return false;
    }
    let intersection = a_tokens.intersection(&b_tokens).count() as f32;
    let smaller = (a_tokens.len().min(b_tokens.len())) as f32;
    // If more than 30% of the smaller set's tokens appear in the other, it's dominated.
    (intersection / smaller) >= 0.3
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
