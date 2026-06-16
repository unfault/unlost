/// Pure rendering functions for the second-brain markdown export.
///
/// No I/O happens here — all functions take data and return `String`.
/// The command handler in `commands/export.rs` owns the file system work.
use std::fmt::Write as FmtWrite;

use crate::types::{FailureMode, IntentCapsule};

// ─────────────────────────────────────────────────────────────────────────────
// Fixed taxonomy
// ─────────────────────────────────────────────────────────────────────────────

/// The canonical set of export categories.
/// LLM-extracted free-text categories are mapped onto these buckets.
pub const TAXONOMY: &[(&str, &str)] = &[
    ("architecture",  "System design, data models, high-level decisions, ADRs"),
    ("debugging",     "Bug investigation, root-cause analysis, crash fixes"),
    ("devops",        "CI/CD, deployment, infrastructure, build systems"),
    ("documentation", "Docs writing, README, changelog, comments"),
    ("feature",       "New feature design or implementation"),
    ("meta",          "Check-ins, greetings, continuations, session bookkeeping"),
    ("notes",         "Manual notes, ideas, observations, ad-hoc thoughts"),
    ("performance",   "Profiling, optimisation, latency, throughput"),
    ("planning",      "Roadmap, sprint planning, scoping, project management"),
    ("refactoring",   "Code clean-up, restructuring, renaming, no behaviour change"),
    ("release",       "Versioning, publishing, changelog prep, tagging"),
    ("review",        "Code review, feedback, discussion of existing code"),
    ("testing",       "Unit tests, integration tests, test coverage, assertions"),
    ("ux",            "UI, CLI output, formatting, user-facing behaviour"),
    ("other",         "Anything that doesn't fit the above buckets"),
];

/// Return the taxonomy bucket names.
pub fn taxonomy_names() -> Vec<&'static str> {
    TAXONOMY.iter().map(|(name, _)| *name).collect()
}

/// Fast deterministic fallback: map a raw category string to the nearest
/// taxonomy bucket using keyword matching.
pub fn map_category_fallback(raw: &str) -> &'static str {
    let s = raw.to_lowercase();
    let s = s.trim();

    for (name, _) in TAXONOMY {
        if s == *name {
            return name;
        }
    }

    if s.contains("debug") || s.contains("bug") || s.contains("fix") || s.contains("crash")
        || s.contains("error") || s.contains("issue") || s.contains("investig")
        || s.contains("troubleshoot") || s.contains("diagnos") || s.contains("root cause")
    {
        return "debugging";
    }
    if s.contains("architect") || s.contains("design") || s.contains("adr")
        || s.contains("schema") || s.contains("model") || s.contains("structure")
        || s.contains("system")
    {
        return "architecture";
    }
    if s.contains("refactor") || s.contains("clean") || s.contains("restructur")
        || s.contains("rename") || s.contains("reorg") || s.contains("maintenance")
        || s.contains("maintenan")
    {
        return "refactoring";
    }
    if s.contains("test") || s.contains("coverage") || s.contains("assert")
        || s.contains("spec") || s.contains("unit test") || s.contains("integration test")
    {
        return "testing";
    }
    if s.contains("doc") || s.contains("readme") || s.contains("changelog")
        || s.contains("comment") || s.contains("write-up") || s.contains("writeup")
    {
        return "documentation";
    }
    if s.contains("feature") || s.contains("implement") || s.contains("develop")
        || s.contains("add ") || s.contains("new ")
    {
        return "feature";
    }
    if s.contains("release") || s.contains("publish") || s.contains("version")
        || s.contains("semver") || s.contains("bump")
        || s.contains("version_control") || s.contains("version control")
    {
        return "release";
    }
    if s.contains("deploy") || s.contains("build") || s.contains("infra")
        || s.contains("pipeline") || s.contains("devops") || s.contains("docker")
        || s.contains("k8s") || s.contains("github action") || s.contains("ci/cd")
    {
        return "devops";
    }
    if s.contains("perf") || s.contains("optim") || s.contains("latency")
        || s.contains("throughput") || s.contains("speed") || s.contains("slow")
        || s.contains("memory") || s.contains("cpu")
    {
        return "performance";
    }
    if s.contains("plan") || s.contains("roadmap") || s.contains("sprint")
        || s.contains("scope") || s.contains("project") || s.contains("milestone")
        || s.contains("management") || s.contains("priorit")
    {
        return "planning";
    }
    if s.contains("review") || s.contains("feedback") || s.contains("code quality")
        || s.contains("pr ") || s.contains("pull request") || s.contains("critique")
    {
        return "review";
    }
    if s.contains("ux") || s.contains("ui") || s.contains("cli")
        || s.contains("format") || s.contains("output") || s.contains("display")
        || s.contains("user interface") || s.contains("user experience")
        || s.contains("styling")
    {
        return "ux";
    }
    if s.contains("note") || s.contains("idea") || s.contains("thought")
        || s.contains("observation") || s.contains("memo")
    {
        return "notes";
    }
    if s.contains("check") || s.contains("ack") || s.contains("confirm")
        || s.contains("greeting") || s.contains("continuation") || s.contains("meta")
        || s.contains("replay") || s.contains("unknown") || s.contains("conversation")
        || s.contains("chat") || s.contains("check-in") || s.contains("checkin")
        || s.contains("status")
    {
        return "meta";
    }

    "other"
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal filtering
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum decision length (chars) to be considered substantive.
const MIN_DECISION_LEN: usize = 25;
/// Minimum rationale length (chars) to be considered substantive.
const MIN_RATIONALE_LEN: usize = 15;

/// Returns true if a capsule carries enough signal to be worth exporting.
///
/// A capsule is substantive if:
/// - It has a meaningful decision AND rationale, OR
/// - It has a non-None failure mode (regardless of text quality)
pub fn is_substantive(cap: &ExportCapsule) -> bool {
    let d = cap.capsule.decision.trim();
    let r = cap.capsule.rationale.trim();
    let has_content = d.len() >= MIN_DECISION_LEN && r.len() >= MIN_RATIONALE_LEN;
    let has_failure = cap.capsule.failure_mode != FailureMode::None;
    has_content || has_failure
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight capsule row read from `capsules.jsonl`.
#[derive(Debug, Clone)]
pub struct ExportCapsule {
    pub id: String,
    pub ts_ms: i64,
    pub agent_session_id: Option<String>,
    pub head_sha: Option<String>,
    pub commit_sha: Option<String>,
    pub source: String,
    /// Derived project name (e.g. "unlost", "unfault") — not a UUID.
    pub project: String,
    pub capsule: IntentCapsule,
}

/// Per-category stats used by the index page.
#[derive(Debug, Clone)]
pub struct CategoryStat {
    pub category: String,
    pub count: usize,
}

/// Per-project stats used by the index page.
#[derive(Debug, Clone)]
pub struct ProjectStat {
    pub project: String,
    pub total_capsules: usize,
    pub substantive_capsules: usize,
    pub symbol_pages: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// File-name helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Derive a filesystem-safe slug from an arbitrary string.
/// Lowercases, replaces non-alphanumeric runs with hyphens, truncates at 60 chars.
pub fn slugify(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut slug = String::new();
    let mut last_was_hyphen = true;
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            slug.push(ch);
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    slug.truncate(60);
    if slug.is_empty() {
        slug = "capsule".to_string();
    }
    slug
}

/// Build the filename for a symbol knowledge page: `<symbol-slug>.md`
/// e.g. `src-storage-rs.md`, `cli-src-main-rs.md`
pub fn symbol_page_filename(symbol: &str) -> String {
    format!("{}.md", slugify(symbol))
}

/// Derive a human-readable project name from a workspace root path.
/// Uses the git remote name if available, otherwise the directory basename.
/// e.g. `/home/user/dev/unlost` → `unlost`
///      `/home/user/dev/my-project` → `my-project`
pub fn project_name_from_root(root: &str) -> String {
    let path = std::path::Path::new(root);
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| slugify(s))
        .unwrap_or_else(|| "workspace".to_string())
}

/// Render `ts_ms` as `YYYY-MM-DD`.
pub fn ts_ms_to_date(ts_ms: i64) -> String {
    let secs = ts_ms / 1000;
    let z = secs / 86400 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Render `ts_ms` as ISO-8601 UTC `YYYY-MM-DDTHH:MM:SSZ`.
pub fn ts_ms_to_iso(ts_ms: i64) -> String {
    let secs = ts_ms / 1000;
    let date = ts_ms_to_date(ts_ms);
    let hms = secs % 86400;
    let h = hms / 3600;
    let m = (hms % 3600) / 60;
    let s = hms % 60;
    format!("{date}T{h:02}:{m:02}:{s:02}Z")
}

pub fn failure_mode_str(fm: &FailureMode) -> &'static str {
    match fm {
        FailureMode::None => "none",
        FailureMode::Drift => "drift",
        FailureMode::Rediscovery => "rediscovery",
        FailureMode::DecisionConflict => "decision_conflict",
        FailureMode::RetrySpiral => "retry_spiral",
        FailureMode::FalseProgress => "false_progress",
        FailureMode::UnboundedHorizon => "unbounded_horizon",
    }
}

fn failure_mode_emoji(fm: &FailureMode) -> &'static str {
    match fm {
        FailureMode::None => "",
        FailureMode::Drift => "⚠️ drift",
        FailureMode::Rediscovery => "🔁 rediscovery",
        FailureMode::DecisionConflict => "⚡ conflict",
        FailureMode::RetrySpiral => "🌀 retry spiral",
        FailureMode::FalseProgress => "🚩 false progress",
        FailureMode::UnboundedHorizon => "🌊 scope creep",
    }
}

/// Escape a string for inline YAML (double-quoted value).
fn escape_yaml(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = chars[..max_chars - 1].iter().collect();
        format!("{}…", truncated)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbol knowledge page  (symbols/<project>/<symbol-slug>.md)
// ─────────────────────────────────────────────────────────────────────────────

/// Render a symbol knowledge page aggregating all substantive decisions
/// that touched this symbol, across time.
pub fn symbol_page(symbol: &str, project: &str, caps: &[&ExportCapsule]) -> String {
    let mut out = String::new();

    // Front-matter
    out.push_str("---\n");
    let _ = writeln!(out, "symbol: \"{}\"", escape_yaml(symbol));
    let _ = writeln!(out, "project: \"{}\"", escape_yaml(project));
    let _ = writeln!(out, "capsule_count: {}", caps.len());
    let failure_count = caps.iter()
        .filter(|c| c.capsule.failure_mode != FailureMode::None)
        .count();
    if failure_count > 0 {
        let _ = writeln!(out, "failure_signals: {failure_count}");
    }
    let first = caps.iter().map(|c| c.ts_ms).min().unwrap_or(0);
    let last  = caps.iter().map(|c| c.ts_ms).max().unwrap_or(0);
    let _ = writeln!(out, "first_seen: \"{}\"", ts_ms_to_date(first));
    let _ = writeln!(out, "last_seen: \"{}\"", ts_ms_to_date(last));
    let _ = writeln!(out, "tags:\n  - symbols\n  - \"{}\"", escape_yaml(project));
    out.push_str("---\n\n");

    // Header
    let _ = writeln!(out, "# `{symbol}`\n");
    let _ = writeln!(out, "> `{project}` · {} decision{} · {} → {}",
        caps.len(),
        if caps.len() == 1 { "" } else { "s" },
        ts_ms_to_date(first),
        ts_ms_to_date(last));
    if failure_count > 0 {
        let _ = writeln!(out, "> ⚠️ {failure_count} failure signal{} on this file\n",
            if failure_count == 1 { "" } else { "s" });
    } else {
        out.push('\n');
    }

    // Sort newest-first so the top of the page is the most relevant view.
    let mut sorted: Vec<&&ExportCapsule> = caps.iter().collect();
    sorted.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));

    // Quick-scan: outstanding next-steps from the most recent 5 capsules.
    let pending: Vec<String> = sorted.iter()
        .take(5)
        .flat_map(|c| c.capsule.next_steps.iter().cloned())
        .take(8)
        .collect();
    if !pending.is_empty() {
        out.push_str("## Outstanding\n\n");
        for step in &pending {
            let _ = writeln!(out, "- [ ] {step}");
        }
        out.push('\n');
    }

    out.push_str("## Decisions\n\n");

    for cap in sorted {
        let fm = &cap.capsule.failure_mode;
        let date = ts_ms_to_date(cap.ts_ms);

        let fm_badge = if *fm != FailureMode::None {
            format!(" · {}", failure_mode_emoji(fm))
        } else {
            String::new()
        };

        let _ = writeln!(out, "### {date}{fm_badge}\n");
        let _ = writeln!(out, "_{}_\n", truncate_str(cap.capsule.intent.trim(), 100));

        let d = cap.capsule.decision.trim();
        let r = cap.capsule.rationale.trim();

        if !d.is_empty() {
            let _ = writeln!(out, "{d}\n");
        }
        if !r.is_empty() {
            let _ = writeln!(out, "> {}\n", r.replace('\n', "\n> "));
        }

        if let Some(ref sig) = cap.capsule.failure_signals {
            if !sig.trim().is_empty() && *fm != FailureMode::None {
                let _ = writeln!(out, "**Signal**: {}\n", sig.trim());
            }
        }

        // Wikilinks to other symbols touched in the same capsule
        let other_syms: Vec<&String> = cap.capsule.symbols.iter()
            .take(MAX_PRIMARY_POSITION)
            .filter(|s| s.as_str() != symbol)
            .collect();
        if !other_syms.is_empty() {
            let links = other_syms.iter()
                .map(|s| format!("[[{}]]", symbol_page_filename(s).trim_end_matches(".md")))
                .collect::<Vec<_>>()
                .join("  ");
            let _ = writeln!(out, "_↳ {links}_\n");
        }
    }

    out
}

/// Symbol position ≤ this is treated as primary focus of the capsule.
pub const MAX_PRIMARY_POSITION: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// decisions.md  (root-level curated decision log)
// ─────────────────────────────────────────────────────────────────────────────

/// Render the root `decisions.md` — a curated, chronological log of every
/// substantive decision and every failure-mode signal across all projects.
pub fn decisions_md(caps: &[&ExportCapsule]) -> String {
    let mut out = String::new();

    let failure_caps: Vec<&&ExportCapsule> = caps.iter()
        .filter(|c| c.capsule.failure_mode != FailureMode::None)
        .collect();

    let _ = writeln!(out, "# Decision Log\n");
    let _ = writeln!(out, "> {} substantive decisions · {} failure signals across {} project{}\n",
        caps.len(),
        failure_caps.len(),
        {
            let mut projs: Vec<&str> = caps.iter().map(|c| c.project.as_str()).collect();
            projs.sort_unstable();
            projs.dedup();
            projs.len()
        },
        if caps.len() == 1 { "" } else { "s" });

    if !failure_caps.is_empty() {
        out.push_str("## Failure Signals\n\n");
        out.push_str("> Patterns worth reviewing — situations where the AI drifted, looped, or made false progress.\n\n");
        out.push_str("| Date | Project | Mode | Decision |\n");
        out.push_str("|------|---------|------|----------|\n");
        for cap in &failure_caps {
            let date = ts_ms_to_date(cap.ts_ms);
            let mode = failure_mode_emoji(&cap.capsule.failure_mode);
            let decision = truncate_str(cap.capsule.decision.trim(), 70);
            let _ = writeln!(out, "| {date} | `{}` | {mode} | {} |",
                cap.project, decision);
        }
        out.push('\n');
    }

    out.push_str("## Chronological Log\n\n");

    let mut current_month = String::new();
    for cap in caps {
        let date = ts_ms_to_date(cap.ts_ms);
        let month = &date[..7]; // "YYYY-MM"

        if month != current_month {
            current_month = month.to_string();
            let _ = writeln!(out, "### {current_month}\n");
        }

        let fm = &cap.capsule.failure_mode;
        let fm_badge = if *fm != FailureMode::None {
            format!(" · {}", failure_mode_emoji(fm))
        } else {
            String::new()
        };

        let _ = writeln!(out, "#### {} · `{}`{}\n", date, cap.project, fm_badge);
        let _ = writeln!(out, "**{}**\n", cap.capsule.intent);

        let d = cap.capsule.decision.trim();
        if !d.is_empty() {
            let _ = writeln!(out, "{d}\n");
        }

        let r = cap.capsule.rationale.trim();
        if !r.is_empty() {
            let _ = writeln!(out, "> {}\n", r.replace('\n', "\n> "));
        }

        if !cap.capsule.symbols.is_empty() {
            let syms = cap.capsule.symbols.iter()
                .take(4)
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "_Symbols: {syms}_\n");
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Category README  (categories/<bucket>/README.md)
// ─────────────────────────────────────────────────────────────────────────────

pub fn category_readme_static(category: &str, caps: &[&ExportCapsule]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {} ({} {})\n",
        category, caps.len(),
        if caps.len() == 1 { "decision" } else { "decisions" });

    if caps.is_empty() {
        out.push_str("_No substantive decisions in this category._\n");
        return out;
    }

    // Group by project for readability
    let mut by_project: std::collections::BTreeMap<&str, Vec<&&ExportCapsule>> =
        std::collections::BTreeMap::new();
    for cap in caps {
        by_project.entry(cap.project.as_str()).or_default().push(cap);
    }

    for (proj, entries) in &by_project {
        let _ = writeln!(out, "## {proj} ({})\n", entries.len());
        out.push_str("| Date | Decision | Symbol | Failure |\n");
        out.push_str("|------|----------|--------|---------|\n");
        for cap in entries {
            let date = ts_ms_to_date(cap.ts_ms);
            let decision = truncate_str(cap.capsule.decision.trim(), 70);
            // Link to the primary symbol's knowledge page if available
            let primary = cap.capsule.symbols.first()
                .map(|s| {
                    let fname = symbol_page_filename(s);
                    format!("[{}](../../symbols/{}/{})",
                        truncate_str(s, 30), proj, fname)
                })
                .unwrap_or_default();
            let fm = if cap.capsule.failure_mode != FailureMode::None {
                failure_mode_emoji(&cap.capsule.failure_mode).to_string()
            } else {
                String::new()
            };
            let _ = writeln!(out, "| {date} | {decision} | {primary} | {fm} |");
        }
        out.push('\n');
    }

    out
}

pub fn category_readme_with_narrative(
    category: &str,
    caps: &[&ExportCapsule],
    narrative: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {} ({} {})\n",
        category, caps.len(),
        if caps.len() == 1 { "decision" } else { "decisions" });

    out.push_str("## Summary\n\n");
    let _ = writeln!(out, "{}\n", narrative.trim());

    let static_part = category_readme_static(category, caps);
    let after_header = static_part.lines().skip(1).collect::<Vec<_>>().join("\n");
    out.push_str(&after_header);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// INDEX.md  (root)
// ─────────────────────────────────────────────────────────────────────────────

pub fn index_md(
    exported_at: &str,
    project_stats: &[ProjectStat],
    category_stats: &[CategoryStat],
    total_decisions: usize,
    total_failure_signals: usize,
    total_symbol_pages: usize,
) -> String {
    let total_caps: usize = project_stats.iter().map(|p| p.substantive_capsules).sum();
    let mut out = String::new();

    let _ = writeln!(out, "# Second Brain\n");
    let _ = writeln!(out, "> Exported from [unlost](https://unlost.dev) · {exported_at}\n");

    out.push_str("## At a Glance\n\n");
    let _ = writeln!(out, "| | |\n|---|---|\n\
        | Projects | {} |\n\
        | Substantive decisions | {} |\n\
        | Failure signals | {} |\n\
        | Symbol knowledge pages | {} |\n\
        | Categories | {} |\n",
        project_stats.len(),
        total_caps,
        total_failure_signals,
        total_symbol_pages,
        category_stats.iter().filter(|s| s.count > 0).count());

    out.push_str("## Entry Points\n\n");
    out.push_str("| Document | What's in it |\n");
    out.push_str("|----------|--------------|\n");
    let _ = writeln!(out, "| [decisions.md](decisions.md) | {} chronological decisions + {} failure signals |",
        total_decisions, total_failure_signals);
    out.push_str("| [symbols/](symbols/) | Per-file knowledge pages grouped by project |\n");
    out.push_str("| [categories/](categories/) | Decisions grouped by type |\n\n");

    out.push_str("## Projects\n\n");
    out.push_str("| Project | Capsules | Substantive | Symbol Pages |\n");
    out.push_str("|---------|----------|-------------|---------------|\n");
    for p in project_stats {
        let _ = writeln!(out, "| `{}` | {} | {} | [{}](symbols/{}) |",
            p.project, p.total_capsules, p.substantive_capsules,
            p.symbol_pages, p.project);
    }
    out.push('\n');

    out.push_str("## Categories\n\n");
    out.push_str("| Category | Capsules |\n");
    out.push_str("|----------|----------|\n");
    for stat in category_stats {
        if stat.count > 0 {
            let _ = writeln!(out, "| [{}](categories/{}/README.md) | {} |",
                stat.category, stat.category, stat.count);
        }
    }
    out.push('\n');

    out.push_str("> Re-run `unlost export` to refresh.\n");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// LLM narrative prompt
// ─────────────────────────────────────────────────────────────────────────────

pub fn category_narrative_prompt(category: &str, caps: &[&ExportCapsule]) -> String {
    let mut prompt = format!(
        "You are writing a concise narrative summary for a second-brain knowledge base.\n\
         Category: \"{category}\"\n\
         There are {} capsules. Here is a structured summary of each:\n\n",
        caps.len()
    );

    for (i, cap) in caps.iter().enumerate() {
        let _ = write!(prompt,
            "--- Capsule {} ({} / {}) ---\nIntent: {}\nDecision: {}\nRationale: {}\n",
            i + 1,
            ts_ms_to_date(cap.ts_ms),
            cap.project,
            cap.capsule.intent,
            cap.capsule.decision.trim(),
            cap.capsule.rationale.trim(),
        );
        if !cap.capsule.next_steps.is_empty() {
            let _ = writeln!(prompt, "Next steps: {}", cap.capsule.next_steps.join("; "));
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "Write a 2–4 paragraph narrative summary in plain prose.\n\
         Focus on the evolution of decisions, recurring themes, and key insights.\n\
         Use past tense. Do not repeat the structured data verbatim — synthesise it.\n\
         Output only the narrative text, no headers, no lists.",
    );
    prompt
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Fix auth token expiry"), "fix-auth-token-expiry");
        assert_eq!(slugify("  hello  world  "), "hello-world");
        assert_eq!(slugify(""), "capsule");
        assert_eq!(slugify("---"), "capsule");
        assert_eq!(slugify("A+B=C"), "a-b-c");
    }

    #[test]
    fn test_ts_ms_to_date() {
        // 2026-06-15 00:00:00 UTC = 1781481600 secs
        let ts = 1781481600_i64 * 1000;
        assert_eq!(ts_ms_to_date(ts), "2026-06-15");
    }

    #[test]
    fn test_ts_ms_to_iso() {
        let ts = 1781481600_i64 * 1000 + 3723_i64 * 1000;
        assert_eq!(ts_ms_to_iso(ts), "2026-06-15T01:02:03Z");
    }

    #[test]
    fn test_is_substantive() {
        let mut cap = make_test_capsule();
        assert!(is_substantive(&cap));

        cap.capsule.decision = "ok".to_string();
        cap.capsule.rationale = "".to_string();
        assert!(!is_substantive(&cap));

        cap.capsule.failure_mode = FailureMode::Drift;
        assert!(is_substantive(&cap), "failure mode overrides content check");
    }

    #[test]
    fn test_project_name_from_root() {
        assert_eq!(project_name_from_root("/home/user/dev/unlost"), "unlost");
        assert_eq!(project_name_from_root("/home/user/my-project"), "my-project");
        assert_eq!(project_name_from_root("/"), "workspace");
    }

    #[test]
    fn test_symbol_page_filename() {
        assert_eq!(symbol_page_filename("src/storage.rs"), "src-storage-rs.md");
        assert_eq!(symbol_page_filename("CHANGELOG.md"), "changelog-md.md");
    }

    #[test]
    fn test_escape_yaml() {
        assert_eq!(escape_yaml(r#"say "hello""#), r#"say \"hello\""#);
        assert_eq!(escape_yaml(r"back\slash"), r"back\\slash");
    }

    fn make_test_capsule() -> ExportCapsule {
        ExportCapsule {
            id: "cap_test123".to_string(),
            ts_ms: 1781481600_i64 * 1000,
            agent_session_id: Some("ses_abc".to_string()),
            head_sha: None,
            commit_sha: None,
            source: "opencode".to_string(),
            project: "unlost".to_string(),
            capsule: IntentCapsule {
                category: "debugging".to_string(),
                intent: "Fix auth token expiry race condition".to_string(),
                decision: "Switch to monotonic clock for all expiry checks".to_string(),
                rationale: "Wall-clock skews on cloud VMs by up to 2s".to_string(),
                next_steps: vec!["Add integration test".to_string()],
                symbols: vec!["src/auth.rs".to_string()],
                user_symbols: vec![],
                failure_mode: FailureMode::None,
                failure_signals: None,
                extraction_mode: crate::types::ExtractionMode::Hybrid,
                questions: vec![],
            },
        }
    }
}
