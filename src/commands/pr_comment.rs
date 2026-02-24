//! `unlost pr-comment` — post an unlost context comment on a GitHub PR.
//!
//! Stealth mode: the shim detects when the agent runs `gh pr create` and spawns
//! this command automatically in the background. Can also be invoked manually:
//!
//!   unlost pr-comment 42
//!   unlost pr-comment https://github.com/owner/repo/pull/42 --session-id ses_abc

use anyhow::Context;
use std::path::Path;

/// Entry point for `unlost pr-comment`.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    pr: String,
    session_id: Option<String>,
    from_commit: Option<String>,
    llm_model: Option<String>,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()?;
    let workspace_root = crate::workspace::git_toplevel(&cwd)
        .unwrap_or_else(|| crate::workspace::canonicalize_dir(&cwd).unwrap_or(cwd.clone()));

    // ── 1. Resolve PR reference ───────────────────────────────────────────────
    // Accept full URL or bare number.
    let pr_ref = pr.trim().to_string();

    // ── 2. Fetch PR metadata via gh ───────────────────────────────────────────
    let pr_meta = fetch_pr_meta(&pr_ref)
        .context("could not fetch PR metadata via gh — is gh installed and authenticated?")?;

    eprintln!(
        "unlost: generating PR comment for PR #{} ({} changed files)",
        pr_meta.number,
        pr_meta.changed_files.len()
    );

    // ── 3. Build the trace chain scoped to changed files + session ────────────
    let ws = crate::workspace::get_or_create_workspace_paths(&cwd)?;

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    // Resolve commit timestamps for scoping.
    let from_ts = from_commit
        .as_deref()
        .or(Some(pr_meta.base_sha.as_str()))
        .and_then(|c| crate::commands::trace::resolve_commit_timestamp(&workspace_root, c));
    let to_ts =
        crate::commands::trace::resolve_commit_timestamp(&workspace_root, &pr_meta.head_sha);

    // Build one composite query from all changed files (up to 20).
    let changed_files_query = pr_meta
        .changed_files
        .iter()
        .take(20)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let query = format!("changes in {}", changed_files_query);
    let framed = crate::storage::frame_query_for_command(&query, crate::storage::QueryIntent::Trace);

    let chain = crate::storage::trace_capsules_lancedb(
        &framed,
        10,  // more seeds for a PR scope
        12,  // wider fan-out
        0.7, // slightly tighter threshold to keep noise down
        from_ts,
        to_ts,
        session_id.as_deref(),
        embedder,
        &ws,
    )
    .await?;

    // ── 4. Build "Worth noting" section via unfault-core graph ────────────────
    let worth_noting =
        build_worth_noting_section(&workspace_root, &pr_meta.changed_files);

    // ── 5. Generate the markdown comment body via LLM ─────────────────────────
    let comment_body = build_pr_comment_markdown(
        llm_model.as_deref(),
        &pr_meta,
        &chain,
        &worth_noting,
        &workspace_root.to_string_lossy(),
    )
    .await?;

    // ── 6. Post via gh pr comment ─────────────────────────────────────────────
    post_pr_comment(&pr_ref, &comment_body)
        .context("could not post PR comment via gh")?;

    eprintln!("unlost: comment posted on PR #{}", pr_meta.number);
    Ok(())
}

// ============================================================================
// PR metadata via gh CLI
// ============================================================================

struct PrMeta {
    number: u64,
    title: String,
    base_sha: String,
    head_sha: String,
    base_branch: String,
    changed_files: Vec<String>,
    /// Raw unified diff (truncated to ~8 KB to keep context manageable).
    diff_summary: String,
    /// One-line summaries of each commit in the PR.
    commits: Vec<String>,
    /// Extracted runtime-impact signals from the diff.
    runtime_signals: RuntimeSignals,
}

/// Signals extracted from the diff text that indicate runtime behaviour changes.
#[derive(Default)]
struct RuntimeSignals {
    has_retry_or_fallback: bool,
    has_extra_io: bool,
    has_logging_changes: bool,
    has_error_handling_changes: bool,
    has_feature_flags: bool,
    /// Correctness-sensitive patterns found (filter semantics, ordering/limit changes, fallback triggers).
    correctness_hints: Vec<String>,
}

fn fetch_pr_meta(pr_ref: &str) -> anyhow::Result<PrMeta> {
    // headRefOid is available; baseRefOid is not in all gh versions — resolve base SHA via git.
    let output = std::process::Command::new("gh")
        .args([
            "pr",
            "view",
            pr_ref,
            "--json",
            "number,title,baseRefName,headRefOid,files",
        ])
        .output()
        .context("failed to run gh pr view")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("gh pr view failed: {stderr}");
    }

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("gh pr view: invalid JSON")?;

    let number = json["number"].as_u64().unwrap_or(0);
    let title = json["title"].as_str().unwrap_or("").to_string();
    let base_branch = json["baseRefName"].as_str().unwrap_or("main").to_string();
    let head_sha = json["headRefOid"].as_str().unwrap_or("").to_string();

    // Resolve base branch to a SHA via git (works even if the branch is remote-only).
    let base_sha = {
        let out = std::process::Command::new("git")
            .args(["rev-parse", &format!("origin/{}", base_branch)])
            .output()
            .ok()
            .filter(|o| o.status.success());
        out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };

    let changed_files: Vec<String> = json["files"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|f| f["path"].as_str().map(|s| s.to_string()))
        .collect();

    // ── Fetch unified diff (capped at ~8 KB) ─────────────────────────────────
    let diff_summary = fetch_pr_diff(pr_ref).unwrap_or_default();

    // ── Fetch commit list ─────────────────────────────────────────────────────
    let commits = fetch_pr_commits(pr_ref).unwrap_or_default();

    // ── Extract runtime-impact signals ────────────────────────────────────────
    let runtime_signals = extract_runtime_signals(&diff_summary);

    Ok(PrMeta {
        number,
        title,
        base_sha,
        head_sha,
        base_branch,
        changed_files,
        diff_summary,
        commits,
        runtime_signals,
    })
}

/// Fetch the unified diff for the PR, truncated to ~8 KB.
fn fetch_pr_diff(pr_ref: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("gh")
        .args(["pr", "diff", pr_ref])
        .output()
        .context("failed to run gh pr diff")?;

    if !output.status.success() {
        return Ok(String::new());
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    // Keep at most 8 192 bytes; trim at a hunk boundary when possible.
    const LIMIT: usize = 8_192;
    if raw.len() <= LIMIT {
        return Ok(raw.into_owned());
    }
    let truncated = &raw[..LIMIT];
    // Try to cut at the last hunk header so we don't leave a torn line.
    let cut = truncated.rfind("\n@@").unwrap_or(LIMIT);
    Ok(format!("{}\n[diff truncated]", &truncated[..cut]))
}

/// Fetch one-line commit messages for the PR.
fn fetch_pr_commits(pr_ref: &str) -> anyhow::Result<Vec<String>> {
    let output = std::process::Command::new("gh")
        .args(["pr", "view", pr_ref, "--json", "commits"])
        .output()
        .context("failed to run gh pr view --json commits")?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let commits = json["commits"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|c| {
            c["messageHeadline"]
                .as_str()
                .map(|s| s.to_string())
        })
        .collect();

    Ok(commits)
}

/// Scan the diff text for patterns that signal runtime behaviour changes.
fn extract_runtime_signals(diff: &str) -> RuntimeSignals {
    let mut s = RuntimeSignals::default();
    if diff.is_empty() {
        return s;
    }

    // Only look at added/changed lines (lines starting with '+').
    let added: String = diff
        .lines()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
        .map(|l| &l[1..])
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    // Retry / fallback paths
    s.has_retry_or_fallback = added.contains("retry")
        || added.contains("fallback")
        || added.contains("backoff")
        || added.contains("attempt");

    // Extra I/O: new network calls, file ops, DB queries
    s.has_extra_io = added.contains("fetch(")
        || added.contains("reqwest")
        || added.contains("sqlx")
        || added.contains("tokio::fs")
        || added.contains("std::fs")
        || added.contains("open(")
        || added.contains("query!(")
        || added.contains("execute(");

    // Logging changes
    s.has_logging_changes = added.contains("tracing::")
        || added.contains("log::")
        || added.contains("eprintln!")
        || added.contains("println!");

    // Error handling changes
    s.has_error_handling_changes = added.contains("anyhow::")
        || added.contains("bail!(")
        || added.contains("unwrap_or")
        || added.contains("map_err")
        || added.contains("context(")
        || added.contains("expect(");

    // Feature flags
    s.has_feature_flags = added.contains("feature_flag")
        || added.contains("flag::")
        || added.contains("cfg!(feature");

    // Correctness-sensitive patterns
    if added.contains("filter(") || added.contains(".filter_map(") {
        s.correctness_hints.push("filter/filter_map semantics changed".to_string());
    }
    if added.contains(".limit(") || added.contains("take(") || added.contains("truncat") {
        s.correctness_hints.push("result truncation/limit behaviour changed".to_string());
    }
    if added.contains(".sort") || added.contains("order_by") {
        s.correctness_hints.push("ordering behaviour changed".to_string());
    }
    if s.has_retry_or_fallback {
        s.correctness_hints.push("fallback/retry path introduced or modified".to_string());
    }

    s
}

// ============================================================================
// Worth noting: code graph analysis of changed files
// ============================================================================

fn build_worth_noting_section(
    workspace_root: &Path,
    changed_files: &[String],
) -> String {
    if changed_files.is_empty() {
        return String::new();
    }

    let ctx = match crate::workspace::build_graph_context_for_workspace(workspace_root) {
        Some(c) => c,
        None => return String::new(),
    };

    let mut notes: Vec<String> = Vec::new();

    // Find which hotspot files intersect with changed files.
    let changed_set: std::collections::HashSet<&str> =
        changed_files.iter().map(|s| s.as_str()).collect();

    // Check if any changed file is a hotspot (highly depended-on).
    let hotspot_hits: Vec<String> = ctx
        .hotspots
        .iter()
        .filter(|(callers, path)| *callers >= 2 && changed_set.contains(path.as_str()))
        .map(|(callers, path)| format!("`{}` (imported by {} other files)", path, callers))
        .collect();

    if !hotspot_hits.is_empty() {
        notes.push(format!(
            "**High-dependency files changed** — these are imported by multiple other files, so \
             changes here may ripple further than they appear:\n{}",
            hotspot_hits
                .iter()
                .map(|s| format!("  - {}", s))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // Find files that import the changed files (direct dependents).
    let dependents: Vec<String> = ctx
        .hotspots
        .iter()
        .filter(|(_, path)| changed_set.contains(path.as_str()))
        .flat_map(|(_, path)| {
            // deps is the dep list of the most central file only; use hotspots as proxy
            let _ = path;
            std::iter::empty::<String>()
        })
        .collect();
    let _ = dependents; // placeholder — full dep traversal would need CodeGraph API

    if notes.is_empty() {
        return String::new();
    }

    notes.join("\n\n")
}

// ============================================================================
// LLM narrative for the PR comment
// ============================================================================

async fn build_pr_comment_markdown(
    llm_model_override: Option<&str>,
    pr_meta: &PrMeta,
    chain: &[crate::CapsuleHit],
    worth_noting: &str,
    _workspace_root: &str,
) -> anyhow::Result<String> {
    use chrono::{SecondsFormat, TimeZone};

    if chain.is_empty() && worth_noting.is_empty() && pr_meta.diff_summary.is_empty() {
        return Ok(format!(
            "## unlost context\n\n\
             _No recorded decisions or diff found for the files changed in this PR. \
             If unlost was not active during this session, run `unlost replay opencode` \
             or `unlost replay git` to seed memory._\n"
        ));
    }

    let fmt_ts = |ts_ms: i64| -> String {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
            .unwrap_or_else(|| ts_ms.to_string())
    };

    let mut context = String::new();

    // ── PR header ─────────────────────────────────────────────────────────────
    context.push_str(&format!("PR: #{} — {}\n", pr_meta.number, pr_meta.title));
    context.push_str(&format!("Base branch: {}\n", pr_meta.base_branch));
    context.push_str(&format!(
        "Changed files ({}): {}\n",
        pr_meta.changed_files.len(),
        pr_meta.changed_files.iter().take(15).cloned().collect::<Vec<_>>().join(", ")
    ));

    // ── Commit list ───────────────────────────────────────────────────────────
    if !pr_meta.commits.is_empty() {
        context.push_str("\nCommits:\n");
        for msg in &pr_meta.commits {
            context.push_str(&format!("  - {}\n", msg));
        }
    }

    // ── Diff ─────────────────────────────────────────────────────────────────
    if !pr_meta.diff_summary.is_empty() {
        context.push_str("\n--- diff (key hunks) ---\n");
        context.push_str(&pr_meta.diff_summary);
        context.push_str("\n--- end diff ---\n");
    }

    // ── Runtime-impact signals ────────────────────────────────────────────────
    let sig = &pr_meta.runtime_signals;
    let mut signals: Vec<&str> = Vec::new();
    if sig.has_retry_or_fallback { signals.push("retry/fallback path added or modified"); }
    if sig.has_extra_io          { signals.push("new or changed I/O (network/file/DB)"); }
    if sig.has_logging_changes   { signals.push("logging/tracing statements changed"); }
    if sig.has_error_handling_changes { signals.push("error handling changed"); }
    if sig.has_feature_flags     { signals.push("feature flag usage present"); }
    if !signals.is_empty() {
        context.push_str("\nRuntime-impact signals detected:\n");
        for s in &signals {
            context.push_str(&format!("  - {}\n", s));
        }
    }
    if !sig.correctness_hints.is_empty() {
        context.push_str("\nCorrectness-sensitive patterns:\n");
        for h in &sig.correctness_hints {
            context.push_str(&format!("  - {}\n", h));
        }
    }

    // ── Causal history from unlost memory ────────────────────────────────────
    if !chain.is_empty() {
        context.push_str(&format!(
            "\nCausal history: {} recorded decisions\n\n",
            chain.len()
        ));
        for (i, hit) in chain.iter().enumerate() {
            let cap = &hit.capsule;
            let meta = &hit.meta;
            context.push_str(&format!(
                "#{} [{}] source={} category={}\n",
                i + 1,
                fmt_ts(hit.ts_ms),
                meta.source,
                cap.category,
            ));
            if cap.failure_mode != crate::types::FailureMode::None {
                let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
                context.push_str(&format!("failure: {}\n", fm.trim_matches('"')));
            }
            if !cap.intent.trim().is_empty() {
                context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
            }
            if !cap.decision.trim().is_empty() {
                context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
            }
            if !cap.rationale.trim().is_empty() {
                context.push_str(&format!("rationale: {}\n", cap.rationale.replace('\n', " ")));
            }
            if !cap.symbols.is_empty() {
                let syms = cap.symbols.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
                context.push_str(&format!("symbols: {syms}\n"));
            }
            context.push('\n');
        }
    }

    // ── Graph worth-noting ────────────────────────────────────────────────────
    if !worth_noting.is_empty() {
        context.push_str("\nGraph analysis:\n");
        context.push_str(worth_noting);
        context.push('\n');
    }

    // ── Prompt ────────────────────────────────────────────────────────────────
    let preamble = "\
You are unlost, a staff engineer writing a review-guidance comment on a pull request.\n\
Your output is read by the developer before a human reviewer sees the PR.\n\
Your role: surface what changed, why it matters operationally, and what the reviewer must verify.\n\
You are NOT writing process notes or style feedback.\n\n\
RULES:\n\
- Total output: 200–300 words maximum.\n\
- Use bullet points with actionable verbs: Verify…, Check…, Confirm…, Ensure…\n\
- Every risk or check must cite a specific file, function, or codepath from the diff.\n\
- If you lack evidence for a claim, write: \"Cannot confirm from history — would need [specific artifact].\"\n\
- Omit sections that have nothing concrete to say.\n\
- Do NOT write generic warnings (e.g. \"make sure tests pass\").\n\
- Do NOT include process/workflow notes unless they affect correctness (migration, data rewrite, backwards compat).\n\n\
OUTPUT FORMAT (strict — use exactly these section headings):\n\
## unlost context\n\
### What Changed\n\
1–2 sentences: what the code does differently after this PR.\n\
### Why\n\
1–2 sentences: the decision/constraint that drove this change (cite recorded decisions if available).\n\
### Behavioral Impact\n\
Bullets: concrete runtime differences a reviewer must understand (latency, ordering, error modes, data shape).\n\
### Risks / Trade-offs\n\
Bullets: specific risks tied to exact codepaths. Each bullet names a file or symbol.\n\
### How To Verify\n\
Bullets starting with Verify/Check/Confirm: what the reviewer should test or inspect.\n\
### Rollout / Recovery\n\
Only include if there is a migration, data rewrite, flag, or backwards-compat concern. Otherwise omit entirely.\n\n\
Return the markdown in the `narrative` field.";

    let result = crate::llm_extract::<crate::QueryNarrativeOutput>(
        llm_model_override,
        preamble,
        &context,
    )
    .await?;

    let mut body = result.narrative;

    // Footer
    body.push_str("\n\n---\n_Generated by [unlost](https://unlost.unfault.dev) — \
                   local-first agent memory._\n");

    Ok(body)
}

// ============================================================================
// Post the comment via gh
// ============================================================================

fn post_pr_comment(pr_ref: &str, body: &str) -> anyhow::Result<()> {
    let output = std::process::Command::new("gh")
        .args(["pr", "comment", pr_ref, "--body", body])
        .output()
        .context("failed to run gh pr comment")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("gh pr comment failed: {stderr}");
    }

    Ok(())
}
