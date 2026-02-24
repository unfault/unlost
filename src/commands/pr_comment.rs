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

    Ok(PrMeta {
        number,
        title,
        base_sha,
        head_sha,
        base_branch,
        changed_files,
    })
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

    if chain.is_empty() && worth_noting.is_empty() {
        return Ok(format!(
            "## unlost context\n\n\
             _No recorded decisions found for the files changed in this PR. \
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

    context.push_str(&format!(
        "PR: #{} — {}\n",
        pr_meta.number, pr_meta.title
    ));
    context.push_str(&format!(
        "Base branch: {}\n",
        pr_meta.base_branch
    ));
    context.push_str(&format!(
        "Changed files ({}): {}\n\n",
        pr_meta.changed_files.len(),
        pr_meta.changed_files.iter().take(15).cloned().collect::<Vec<_>>().join(", ")
    ));

    if !chain.is_empty() {
        context.push_str(&format!(
            "Causal history: {} recorded decisions relevant to these changes\n\n",
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
                context.push_str(&format!(
                    "rationale: {}\n",
                    cap.rationale.replace('\n', " ")
                ));
            }
            if !cap.symbols.is_empty() {
                let syms = cap.symbols.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
                context.push_str(&format!("symbols: {syms}\n"));
            }
            context.push('\n');
        }
    }

    let preamble = "\
You are unlost, acting as a staff engineer reviewing a pull request. \
Your job is not to find bugs — that is what code reviewers do. \
Your job is to give the developer context: why the changed code exists, \
what decisions shaped it, and what else might be affected. \
Be concise, direct, and honest. Write in plain markdown. \
Never hallucinate — if you do not have evidence for a claim, say so or omit it.\n\n\
Produce a markdown comment with these sections:\n\
## unlost context\n\
### What happened here\n\
(1-3 sentences: what this change is about based on recorded decisions and intents)\n\
### Where this comes from\n\
(The key decisions and constraints that led to this change. \
Cite specific decisions from the history. \
If a decision had a recorded failure mode, mention it.)\n\
### Worth noting\n\
(Anything the reviewer should be aware of that is not obvious from the diff. \
Cross-cutting concerns, related past failures, patterns that have been changed before. \
Omit this section entirely if nothing notable.)\n\n\
Keep the entire comment under 400 words. \
Write for a developer reading this PR for the first time. \
Return the markdown in the `narrative` field.";

    let result = crate::llm_extract::<crate::QueryNarrativeOutput>(
        llm_model_override,
        preamble,
        &context,
    )
    .await?;

    let mut body = result.narrative;

    // Append "Worth noting" from graph analysis if LLM didn't cover it and we have data.
    if !worth_noting.is_empty() && !body.contains("Worth noting") {
        body.push_str("\n\n### Worth noting\n\n");
        body.push_str(worth_noting);
    }

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
