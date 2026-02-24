use anyhow::Context;
use std::io::Write;
use std::path::Path;

// ============================================================================
// CHANGELOG.md ingestion as capsules
// ============================================================================

/// A parsed version entry from a Keep-a-Changelog-style CHANGELOG.md.
#[derive(Debug, Clone)]
struct ChangelogEntry {
    /// Semver string, e.g. "0.7.0"
    version: String,
    /// Release date string as written in the file, e.g. "2026-02-19"
    date: String,
    /// Unix timestamp in milliseconds derived from `date`, or 0 if unparseable.
    timestamp_ms: i64,
    /// First bullet point of the section — used as the capsule `decision`.
    first_bullet: String,
    /// Full section body (all lines after the `## [version]` header).
    body: String,
}

/// Parse a CHANGELOG.md and return one entry per `## [version]` section.
///
/// Handles the Keep-a-Changelog format:
/// ```markdown
/// ## [0.7.0] - 2026-02-19
/// ### Added
/// - **Feature**: Description.
/// ```
///
/// Entries are returned oldest-first (ascending timestamp) so that capsule
/// timestamps are monotonically increasing, matching the git commit pattern.
fn parse_changelog(content: &str) -> Vec<ChangelogEntry> {
    let mut entries: Vec<ChangelogEntry> = Vec::new();

    let mut current_version: Option<String> = None;
    let mut current_date: Option<String> = None;
    let mut current_lines: Vec<String> = Vec::new();

    for line in content.lines() {
        // Detect version header: `## [0.7.0] - 2026-02-19` (or without date)
        if line.starts_with("## [") {
            // Flush previous entry
            if let (Some(ver), Some(date)) = (current_version.take(), current_date.take()) {
                if let Some(entry) = build_entry(ver, date, std::mem::take(&mut current_lines)) {
                    entries.push(entry);
                }
            }
            current_lines.clear();

            // Parse: `## [VERSION] - DATE`
            let inner = &line[3..]; // strip "## "
            // inner looks like "[0.7.0] - 2026-02-19" or "[Unreleased]"
            if let Some(close) = inner.find(']') {
                let ver = inner[1..close].trim().to_string();
                if ver.eq_ignore_ascii_case("unreleased") {
                    // Skip unreleased sections — no date, not yet shipped
                    continue;
                }
                current_version = Some(ver);
                // Date part after `] - `
                let rest = inner[close + 1..].trim();
                let date = if let Some(stripped) = rest.strip_prefix('-') {
                    stripped.trim().to_string()
                } else {
                    String::new()
                };
                current_date = Some(date);
            }
        } else if current_version.is_some() {
            // Accumulate body lines under the current version
            current_lines.push(line.to_string());
        }
    }

    // Flush last entry
    if let (Some(ver), Some(date)) = (current_version, current_date) {
        if let Some(entry) = build_entry(ver, date, current_lines) {
            entries.push(entry);
        }
    }

    // Sort oldest-first so LanceDB timestamps are monotonically increasing
    entries.sort_by_key(|e| e.timestamp_ms);
    entries
}

/// Build a `ChangelogEntry` from the accumulated lines of one version block.
fn build_entry(version: String, date: String, lines: Vec<String>) -> Option<ChangelogEntry> {
    // Collect non-empty lines for the body, stripping leading/trailing blank lines.
    let body_lines: Vec<&str> = lines
        .iter()
        .map(|l| l.as_str())
        .collect::<Vec<_>>()
        .into_iter()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>();

    // Trim trailing blank lines
    let body_lines: Vec<&str> = {
        let mut v = body_lines;
        while v
            .last()
            .map(|l: &&str| l.trim().is_empty())
            .unwrap_or(false)
        {
            v.pop();
        }
        v
    };

    if body_lines.is_empty() {
        return None;
    }

    let body = body_lines.join("\n");

    // First bullet: find the first line starting with `- ` (possibly after a `### ` header)
    let first_bullet = body_lines
        .iter()
        .find(|l| l.trim_start().starts_with("- "))
        .map(|l| {
            // Strip leading `- ` and any bold markers like `**Title**: `
            let trimmed = l.trim_start().trim_start_matches("- ");
            // Strip **bold**: prefix (common in this changelog style)
            if trimmed.starts_with("**") {
                if let Some(end) = trimmed.find("**: ") {
                    return trimmed[end + 4..].trim().to_string();
                }
                if let Some(end) = trimmed.find("**:") {
                    return trimmed[end + 3..].trim().to_string();
                }
            }
            trimmed.to_string()
        })
        .unwrap_or_else(|| body_lines[0].trim().to_string());

    // Parse timestamp from "YYYY-MM-DD"
    let timestamp_ms = parse_date_ms(&date);

    Some(ChangelogEntry {
        version,
        date,
        timestamp_ms,
        first_bullet,
        body,
    })
}

/// Parse "YYYY-MM-DD" into Unix milliseconds (midnight UTC). Returns 0 on failure.
fn parse_date_ms(date: &str) -> i64 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return 0;
    }
    let year: i64 = parts[0].parse().unwrap_or(0);
    let month: i64 = parts[1].parse().unwrap_or(0);
    let day: i64 = parts[2].parse().unwrap_or(0);
    if year == 0 || month == 0 || day == 0 {
        return 0;
    }
    // Days since Unix epoch via the civil-date algorithm (no external deps)
    // Reference: https://howardhinnant.github.io/date_algorithms.html
    let y = if month <= 2 { year - 1 } else { year };
    let m = month as i64;
    let d = day as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86_400 * 1000
}

// ── Deduplication ────────────────────────────────────────────────────────────

fn ingested_versions_path(ws: &crate::WorkspacePaths) -> std::path::PathBuf {
    crate::workspace::unlost_workspace_dir(&ws.id)
        .join("changelog")
        .join("ingested.txt")
}

fn load_ingested_versions(ws: &crate::WorkspacePaths) -> std::collections::HashSet<String> {
    let path = ingested_versions_path(ws);
    match std::fs::read_to_string(&path) {
        Ok(s) => s
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => std::collections::HashSet::new(),
    }
}

fn append_ingested_versions(ws: &crate::WorkspacePaths, versions: &[String]) {
    if versions.is_empty() {
        return;
    }
    let path = ingested_versions_path(ws);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    for v in versions {
        let _ = writeln!(f, "{}", v.trim());
    }
}

// ── Public ingestion entry point ──────────────────────────────────────────────

/// Ingest a `CHANGELOG.md` file as capsules into the workspace's LanceDB.
///
/// Each `## [version]` section becomes one `IntentCapsule` with
/// `category: "Changelog"`. Deduplicates by version string across runs so
/// re-running `unlost init` or `unlost replay` is safe.
///
/// Returns the number of newly ingested version entries.
pub async fn ingest_changelog(
    ws: &crate::WorkspacePaths,
    changelog_path: &Path,
    embedder: &crate::embed::Embedder,
    use_color: bool,
) -> anyhow::Result<usize> {
    let content = match std::fs::read_to_string(changelog_path) {
        Ok(c) => c,
        Err(_) => return Ok(0), // No CHANGELOG.md — silently skip
    };

    let already_ingested = load_ingested_versions(ws);
    let entries = parse_changelog(&content);

    let new_entries: Vec<ChangelogEntry> = entries
        .into_iter()
        .filter(|e| !already_ingested.contains(&e.version))
        .collect();

    if new_entries.is_empty() {
        return Ok(0);
    }

    std::fs::create_dir_all(&ws.db_dir).context("create db_dir")?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let mut ingested = 0usize;
    let mut new_versions: Vec<String> = Vec::new();

    for entry in &new_entries {
        let capsule = entry_to_capsule(entry);

        let meta = crate::ResponseMeta {
            source: "changelog".to_string(),
            upstream_host: "changelog".to_string(),
            request_path: entry.version.clone(),
            http_status: 0,
            agent_session_id: None,
            usage: None,
        };

        match crate::storage::insert_capsule_row(
            &db,
            embedder,
            0,
            0,
            entry.timestamp_ms,
            &meta,
            None,
            None,
            &capsule,
            None,
            None, // head_sha: not applicable for changelog ingestion
            None, // commit_sha: not applicable for changelog ingestion
        )
        .await
        {
            Ok(_) => {
                new_versions.push(entry.version.clone());
                ingested += 1;
            }
            Err(e) => {
                tracing::warn!(
                    version = %entry.version,
                    error = %e,
                    "failed to insert changelog capsule"
                );
            }
        }
    }

    append_ingested_versions(ws, &new_versions);

    if ingested > 0 && use_color {
        println!(
            "\x1b[2m  + {} changelog version{} indexed\x1b[0m",
            ingested,
            if ingested == 1 { "" } else { "s" }
        );
    } else if ingested > 0 {
        println!("  + {} changelog version(s) indexed", ingested);
    }

    Ok(ingested)
}

/// Convert a `ChangelogEntry` into an `IntentCapsule`.
fn entry_to_capsule(e: &ChangelogEntry) -> crate::IntentCapsule {
    crate::IntentCapsule {
        category: "Changelog".to_string(),
        intent: format!("Release {} on {}", e.version, e.date),
        decision: e.first_bullet.clone(),
        rationale: e.body.clone(),
        next_steps: Vec::new(),
        symbols: Vec::new(),
        user_symbols: Vec::new(),
        failure_mode: crate::types::FailureMode::None,
        failure_signals: None,
        extraction_mode: crate::types::ExtractionMode::None,
        questions: vec![],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"# Changelog

## [0.7.0] - 2026-02-19

### Added
- **`unlost brief`**: New command that produces a staff-engineer-style codebase debrief.
- **Git commit ingestion**: Git commits are now first-class capsules.

### Changed
- **Git capsule routing**: Git capsules are included in `brief` and `query`.

## [0.6.5] - 2026-02-18

### Fixed
- **LLM Schema Compatibility**: Fixed invalid JSON schema for `extraction_mode`.

## [Unreleased]

### Added
- Work in progress.
"#;

    #[test]
    fn test_parse_count() {
        let entries = parse_changelog(SAMPLE);
        // Unreleased should be skipped; 0.6.5 comes before 0.7.0 (oldest-first)
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].version, "0.6.5");
        assert_eq!(entries[1].version, "0.7.0");
    }

    #[test]
    fn test_parse_fields() {
        let entries = parse_changelog(SAMPLE);
        let v07 = &entries[1];
        assert_eq!(v07.date, "2026-02-19");
        assert!(v07.timestamp_ms > 0);
        assert!(v07.body.contains("unlost brief"));
        // first_bullet should be the description after stripping bold prefix
        assert!(
            v07.first_bullet.contains("codebase debrief") || v07.first_bullet.contains("brief"),
            "unexpected first_bullet: {}",
            v07.first_bullet
        );
    }

    #[test]
    fn test_parse_date_ms() {
        // 1970-01-01 = 0
        assert_eq!(parse_date_ms("1970-01-01"), 0);
        // 2026-02-19 should be positive
        assert!(parse_date_ms("2026-02-19") > 0);
        // Invalid date
        assert_eq!(parse_date_ms("not-a-date"), 0);
    }

    #[test]
    fn test_no_entries_for_empty() {
        assert!(parse_changelog("# Changelog\n").is_empty());
        assert!(parse_changelog("").is_empty());
    }
}
