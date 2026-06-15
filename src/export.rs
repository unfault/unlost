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

/// Return the taxonomy bucket name for display (folder name = first element).
pub fn taxonomy_names() -> Vec<&'static str> {
    TAXONOMY.iter().map(|(name, _)| *name).collect()
}

/// Fast deterministic fallback: map a raw category string to the nearest
/// taxonomy bucket using keyword matching. Used when no LLM is configured
/// or when `--no-llm` is passed.
pub fn map_category_fallback(raw: &str) -> &'static str {
    let s = raw.to_lowercase();
    let s = s.trim();

    // Exact match on taxonomy names first
    for (name, _) in TAXONOMY {
        if s == *name {
            return name;
        }
    }

    // Keyword matching
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
        || s.contains("tag") || s.contains("deploy") && s.contains("release")
        || s.contains("version_control") || s.contains("version control")
        || s.contains("semver") || s.contains("bump")
    {
        return "release";
    }
    if s.contains("deploy") || s.contains("ci") || s.contains("cd")
        || s.contains("build") || s.contains("infra") || s.contains("pipeline")
        || s.contains("devops") || s.contains("docker") || s.contains("k8s")
        || s.contains("github action")
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

/// Lightweight capsule row read from `capsules.jsonl`.
/// Only the fields we actually render are populated.
#[derive(Debug, Clone)]
pub struct ExportCapsule {
    pub id: String,
    pub ts_ms: i64,
    pub agent_session_id: Option<String>,
    pub head_sha: Option<String>,
    pub commit_sha: Option<String>,
    pub source: String,
    pub capsule: IntentCapsule,
}

/// Per-category stats used by the index page.
#[derive(Debug, Clone)]
pub struct CategoryStat {
    pub category: String,
    pub count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// File-name helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Derive a filesystem-safe slug from an arbitrary string.
/// Lowercases, replaces non-alphanumeric runs with hyphens, truncates at 60 chars.
pub fn slugify(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut slug = String::new();
    let mut last_was_hyphen = true; // suppress leading hyphen
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            slug.push(ch);
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }
    // strip trailing hyphen
    while slug.ends_with('-') {
        slug.pop();
    }
    slug.truncate(60);
    if slug.is_empty() {
        slug = "capsule".to_string();
    }
    slug
}

/// Build the filename for a capsule: `<YYYY-MM-DD>-<intent-slug>.md`
pub fn capsule_filename(cap: &ExportCapsule) -> String {
    let date = ts_ms_to_date(cap.ts_ms);
    let slug = slugify(&cap.capsule.intent);
    format!("{date}-{slug}.md")
}

/// Render `ts_ms` as `YYYY-MM-DD`.
pub fn ts_ms_to_date(ts_ms: i64) -> String {
    // Simple implementation: seconds since epoch → date
    let secs = ts_ms / 1000;
    // Rough Julian-day decomposition (no external dep needed).
    // Using the algorithm from https://www.researchgate.net/publication/316558298
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

fn failure_mode_str(fm: &FailureMode) -> &'static str {
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

// ─────────────────────────────────────────────────────────────────────────────
// Individual capsule file
// ─────────────────────────────────────────────────────────────────────────────

/// Render a full markdown file for a single capsule, including YAML front-matter.
pub fn capsule_to_markdown(cap: &ExportCapsule) -> String {
    let mut out = String::new();

    // ── YAML front-matter ──────────────────────────────────────────────────
    out.push_str("---\n");
    let _ = writeln!(out, "id: \"{}\"", cap.id);
    let _ = writeln!(out, "date: \"{}\"", ts_ms_to_iso(cap.ts_ms));
    let _ = writeln!(out, "category: \"{}\"", escape_yaml(&cap.capsule.category));

    // symbols list
    if !cap.capsule.symbols.is_empty() {
        out.push_str("symbols:\n");
        for sym in &cap.capsule.symbols {
            let _ = writeln!(out, "  - \"{}\"", escape_yaml(sym));
        }
    } else {
        out.push_str("symbols: []\n");
    }

    let _ = writeln!(
        out,
        "failure_mode: \"{}\"",
        failure_mode_str(&cap.capsule.failure_mode)
    );

    // tags = category
    let _ = writeln!(out, "tags:\n  - \"{}\"", escape_yaml(&cap.capsule.category));

    if let Some(ref sha) = cap.commit_sha {
        let _ = writeln!(out, "commit_sha: \"{}\"", sha);
    }
    if let Some(ref sha) = cap.head_sha {
        let _ = writeln!(out, "head_sha: \"{}\"", sha);
    }
    if let Some(ref sid) = cap.agent_session_id {
        let _ = writeln!(out, "session_id: \"{}\"", escape_yaml(sid));
    }
    let _ = writeln!(out, "source: \"{}\"", escape_yaml(&cap.source));

    out.push_str("---\n\n");

    // ── Title ──────────────────────────────────────────────────────────────
    let _ = writeln!(out, "# {}\n", cap.capsule.intent);

    // ── Intent block ───────────────────────────────────────────────────────
    let _ = writeln!(out, "**Intent**: {}\n", cap.capsule.intent);

    // ── Decision ───────────────────────────────────────────────────────────
    out.push_str("## Decision\n\n");
    let _ = writeln!(out, "{}\n", cap.capsule.decision);

    // ── Rationale ──────────────────────────────────────────────────────────
    out.push_str("## Rationale\n\n");
    let _ = writeln!(out, "{}\n", cap.capsule.rationale);

    // ── Next Steps ─────────────────────────────────────────────────────────
    if !cap.capsule.next_steps.is_empty() {
        out.push_str("## Next Steps\n\n");
        for step in &cap.capsule.next_steps {
            let _ = writeln!(out, "- [ ] {}", step);
        }
        out.push('\n');
    }

    // ── Symbols ────────────────────────────────────────────────────────────
    if !cap.capsule.symbols.is_empty() {
        out.push_str("## Symbols\n\n");
        for sym in &cap.capsule.symbols {
            let _ = writeln!(out, "- `{}`", sym);
        }
        out.push('\n');
    }

    // ── Failure mode note ──────────────────────────────────────────────────
    if cap.capsule.failure_mode != FailureMode::None {
        out.push_str("## Failure Signal\n\n");
        let _ = writeln!(
            out,
            "**Mode**: `{}`\n",
            failure_mode_str(&cap.capsule.failure_mode)
        );
        if let Some(ref sig) = cap.capsule.failure_signals {
            let _ = writeln!(out, "{}\n", sig);
        }
    }

    // ── Metadata footer ────────────────────────────────────────────────────
    out.push_str("---\n\n");
    let _ = writeln!(out, "_Recorded: {}_", ts_ms_to_iso(cap.ts_ms));

    out
}

/// Escape a string for inline YAML (double-quoted value): escape `"` and `\`.
fn escape_yaml(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ─────────────────────────────────────────────────────────────────────────────
// Category README
// ─────────────────────────────────────────────────────────────────────────────

/// Render a static category README with a table of all capsules in that category.
/// `caps` should be sorted oldest-first; filenames must already be computed.
pub fn category_readme_static(category: &str, caps: &[(&ExportCapsule, &str)]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# {} ({} {})\n",
        category,
        caps.len(),
        if caps.len() == 1 { "capsule" } else { "capsules" }
    );

    if caps.is_empty() {
        out.push_str("_No capsules yet._\n");
        return out;
    }

    out.push_str("| Date | Intent | Failure Mode |\n");
    out.push_str("|------|--------|--------------|\n");
    for (cap, filename) in caps {
        let date = ts_ms_to_date(cap.ts_ms);
        let intent_truncated = truncate_str(&cap.capsule.intent, 60);
        let fm = failure_mode_str(&cap.capsule.failure_mode);
        let _ = writeln!(
            out,
            "| {} | [{}]({}) | `{}` |",
            date, intent_truncated, filename, fm
        );
    }
    out.push('\n');
    out.push_str("> _Run `unlost export --narrative` to generate an LLM-written narrative summary for this category._\n");
    out
}

/// Render a category README that has been augmented with an LLM-generated narrative.
pub fn category_readme_with_narrative(
    category: &str,
    caps: &[(&ExportCapsule, &str)],
    narrative: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# {} ({} {})\n",
        category,
        caps.len(),
        if caps.len() == 1 { "capsule" } else { "capsules" }
    );

    out.push_str("## Summary\n\n");
    let _ = writeln!(out, "{}\n", narrative.trim());

    out.push_str("## Capsules\n\n");
    if caps.is_empty() {
        out.push_str("_No capsules._\n");
    } else {
        out.push_str("| Date | Intent | Failure Mode |\n");
        out.push_str("|------|--------|--------------|\n");
        for (cap, filename) in caps {
            let date = ts_ms_to_date(cap.ts_ms);
            let intent_truncated = truncate_str(&cap.capsule.intent, 60);
            let fm = failure_mode_str(&cap.capsule.failure_mode);
            let _ = writeln!(
                out,
                "| {} | [{}]({}) | `{}` |",
                date, intent_truncated, filename, fm
            );
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level INDEX.md
// ─────────────────────────────────────────────────────────────────────────────

/// Render the top-level `INDEX.md` for the export directory.
pub fn index_md(workspace_root: &str, exported_at: &str, stats: &[CategoryStat]) -> String {
    let total: usize = stats.iter().map(|s| s.count).sum();
    let mut out = String::new();

    let _ = writeln!(out, "# unlost: second brain export\n");
    let _ = writeln!(out, "**Workspace**: `{}`  ", workspace_root);
    let _ = writeln!(out, "**Exported**: {}  ", exported_at);
    let _ = writeln!(
        out,
        "**Total capsules**: {}  \n",
        total
    );

    if stats.is_empty() {
        out.push_str("_No capsules recorded yet._\n");
        return out;
    }

    out.push_str("## Categories\n\n");
    out.push_str("| Category | Capsules |\n");
    out.push_str("|----------|----------|\n");
    for stat in stats {
        let _ = writeln!(
            out,
            "| [{}]({}/README.md) | {} |",
            stat.category, stat.category, stat.count
        );
    }
    out.push('\n');
    out.push_str("> Generated by [unlost](https://unlost.dev). Re-run `unlost export` to refresh.\n");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// LLM narrative prompt
// ─────────────────────────────────────────────────────────────────────────────

/// Build the LLM prompt for a category narrative summary.
/// The caller is responsible for sending this to the configured LLM.
pub fn category_narrative_prompt(category: &str, caps: &[&ExportCapsule]) -> String {
    let mut prompt = format!(
        "You are writing a concise narrative summary for a second-brain knowledge base.\n\
         Category: \"{category}\"\n\
         There are {} capsules in this category. Here is a structured summary of each:\n\n",
        caps.len()
    );

    for (i, cap) in caps.iter().enumerate() {
        let _ = write!(
            prompt,
            "--- Capsule {} ({}) ---\n\
             Intent: {}\n\
             Decision: {}\n\
             Rationale: {}\n",
            i + 1,
            ts_ms_to_date(cap.ts_ms),
            cap.capsule.intent,
            cap.capsule.decision,
            cap.capsule.rationale,
        );
        if !cap.capsule.next_steps.is_empty() {
            let steps = cap.capsule.next_steps.join("; ");
            let _ = writeln!(prompt, "Next steps: {steps}");
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "Write a 2–4 paragraph narrative summary of this category in plain prose.\n\
         Focus on the evolution of decisions, recurring themes, and key insights.\n\
         Use past tense. Do not repeat the structured data verbatim — synthesise it.\n\
         Output only the narrative text, no headers, no lists.",
    );
    prompt
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

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
        // 2026-06-15 00:00:00 UTC  =  1781481600 secs
        let ts = 1781481600_i64 * 1000;
        assert_eq!(ts_ms_to_date(ts), "2026-06-15");
    }

    #[test]
    fn test_ts_ms_to_iso() {
        let ts = 1781481600_i64 * 1000 + 3723_i64 * 1000; // +1h 2m 3s
        assert_eq!(ts_ms_to_iso(ts), "2026-06-15T01:02:03Z");
    }

    #[test]
    fn test_capsule_filename() {
        let cap = make_test_capsule();
        let name = capsule_filename(&cap);
        assert!(name.starts_with("2026-"), "filename should start with year: {name}");
        assert!(name.ends_with(".md"), "filename should end with .md: {name}");
    }

    #[test]
    fn test_capsule_to_markdown_contains_frontmatter() {
        let cap = make_test_capsule();
        let md = capsule_to_markdown(&cap);
        assert!(md.starts_with("---\n"), "should start with YAML front-matter");
        assert!(md.contains("id:"), "front-matter should have id");
        assert!(md.contains("category:"), "front-matter should have category");
        assert!(md.contains("failure_mode:"), "front-matter should have failure_mode");
        assert!(md.contains("## Decision"), "body should have Decision section");
        assert!(md.contains("## Rationale"), "body should have Rationale section");
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
            capsule: IntentCapsule {
                category: "debugging".to_string(),
                intent: "Fix auth token expiry race condition".to_string(),
                decision: "Switch to monotonic clock".to_string(),
                rationale: "Wall-clock skews on cloud VMs".to_string(),
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
