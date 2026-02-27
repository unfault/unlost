use anyhow::Context;
use serde::Serialize;
use std::io::Write;
use std::path::Path;

// ============================================================================
// Commit ingestion as capsules
// ============================================================================

/// A richer commit record used for capsule ingestion (includes body text).
#[derive(Debug, Clone)]
struct CommitDetail {
    hash: String,
    timestamp_ms: i64,
    subject: String,
    body: String,
    files: Vec<String>,
    author: String,
}

/// Fetch commits with full subject + body for capsule ingestion.
///
/// Uses `--since` when provided to avoid reprocessing old commits on incremental runs.
fn fetch_commits_for_ingest(
    repo_root: &Path,
    since_ms: Option<i64>,
    max_commits: usize,
) -> anyhow::Result<Vec<CommitDetail>> {
    // Format: hash RS timestamp RS author RS subject RS body GS
    // We use ASCII record/group separators to avoid conflicts with commit text.
    let mut args = vec![
        "log".to_string(),
        format!("-n{}", max_commits),
        "--pretty=format:%H%x1f%ct%x1f%an%x1f%s%x1f%b%x1e".to_string(),
    ];
    if let Some(since) = since_ms {
        args.push(format!("--since={}", since / 1000));
    }

    let output = std::process::Command::new("git")
        .current_dir(repo_root)
        .args(&args)
        .output()
        .context("failed to run git log")?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let mut commits = Vec::new();

    for rec in raw.split('\x1e') {
        let rec = rec.trim();
        if rec.is_empty() {
            continue;
        }
        let mut parts = rec.splitn(5, '\x1f');
        let hash = parts.next().unwrap_or("").trim().to_string();
        let ts_str = parts.next().unwrap_or("").trim();
        let author = parts.next().unwrap_or("").trim().to_string();
        let subject = parts.next().unwrap_or("").trim().to_string();
        let body = parts.next().unwrap_or("").trim().to_string();

        if hash.is_empty() || subject.is_empty() {
            continue;
        }
        // Skip merge commits and trivial subjects
        if subject.starts_with("Merge ") || subject.starts_with("merge ") {
            continue;
        }

        let ts_ms = ts_str.parse::<i64>().unwrap_or(0) * 1000;

        let files = std::process::Command::new("git")
            .current_dir(repo_root)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", &hash])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        commits.push(CommitDetail {
            hash,
            timestamp_ms: ts_ms,
            subject,
            body,
            files,
            author,
        });
    }

    // Oldest-first so capsule timestamps are monotonically increasing
    commits.sort_by_key(|c| c.timestamp_ms);
    Ok(commits)
}

/// Convert a commit detail into an `IntentCapsule`.
fn commit_to_capsule(c: &CommitDetail) -> crate::IntentCapsule {
    // Treat commit subject as the decision; body (if any) as rationale.
    // This maps well to the capsule schema: a commit IS a recorded decision.
    let decision = c.subject.clone();
    let rationale = c.body.clone();

    // Symbols = touched files. For large commits cap at 20.
    let symbols: Vec<String> = c.files.iter().take(20).cloned().collect();

    crate::IntentCapsule {
        category: "GitCommit".to_string(),
        intent: format!("Commit by {}: {}", c.author, c.subject),
        decision,
        rationale,
        next_steps: Vec::new(),
        symbols,
        user_symbols: Vec::new(),
        failure_mode: crate::types::FailureMode::None,
        failure_signals: None,
        extraction_mode: crate::types::ExtractionMode::None,
        questions: vec![],
    }
}

/// Load the set of already-ingested commit hashes for a workspace.
fn load_ingested_hashes(ws: &crate::WorkspacePaths) -> std::collections::HashSet<String> {
    let path = ingested_hashes_path(ws);
    match std::fs::read_to_string(&path) {
        Ok(s) => s
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => std::collections::HashSet::new(),
    }
}

fn ingested_hashes_path(ws: &crate::WorkspacePaths) -> std::path::PathBuf {
    crate::workspace::unlost_workspace_dir(&ws.id)
        .join("git")
        .join("ingested.txt")
}

fn append_ingested_hashes(ws: &crate::WorkspacePaths, hashes: &[String]) {
    if hashes.is_empty() {
        return;
    }
    let path = ingested_hashes_path(ws);
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
    for h in hashes {
        let _ = writeln!(f, "{}", h.trim());
    }
}

/// Ingest git commits from `repo_root` as capsules into the workspace's LanceDB.
///
/// - Deduplicates by commit hash (persisted in `git/ingested.txt`).
/// - Skips merge commits and empty subjects.
/// - Does not call an LLM — the commit message IS the capsule.
/// - Returns the number of newly ingested commits.
///
/// This is intentionally zero-LLM: every commit is already a human-authored
/// intent capsule. We embed `subject + body` for semantic search.
pub async fn ingest_git_commits(
    ws: &crate::WorkspacePaths,
    repo_root: &Path,
    embedder: &crate::embed::Embedder,
    max_commits: usize,
    use_color: bool,
) -> anyhow::Result<usize> {
    if crate::workspace::git_toplevel(repo_root).is_none() {
        return Ok(0);
    }

    let already_ingested = load_ingested_hashes(ws);

    // Fetch up to max_commits; we'll dedup client-side.
    let commits = fetch_commits_for_ingest(repo_root, None, max_commits)?;

    let new_commits: Vec<CommitDetail> = commits
        .into_iter()
        .filter(|c| !already_ingested.contains(&c.hash))
        .collect();

    if new_commits.is_empty() {
        return Ok(0);
    }

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let mut ingested = 0usize;
    let mut new_hashes: Vec<String> = Vec::new();

    for commit in &new_commits {
        let capsule = commit_to_capsule(commit);

        let meta = crate::ResponseMeta {
            source: "git".to_string(),
            upstream_host: "git".to_string(),
            request_path: commit.hash[..7.min(commit.hash.len())].to_string(),
            http_status: 0,
            agent_session_id: None,
            usage: None,
        };

        // Embed subject + body for rich semantic search
        let embed_text = if commit.body.trim().is_empty() {
            commit.subject.clone()
        } else {
            format!("{}\n\n{}", commit.subject, commit.body)
        };

        match crate::storage::insert_capsule_row(
            &db,
            embedder,
            0,
            0,
            commit.timestamp_ms,
            &meta,
            None,
            None,
            &capsule,
            &crate::types::TurnEval::default(),
            Some(&embed_text),
            None, // head_sha: not applicable for git history ingestion
            None, // commit_sha: not applicable for git history ingestion
        )
        .await
        {
            Ok(_) => {
                new_hashes.push(commit.hash.clone());
                ingested += 1;
            }
            Err(e) => {
                tracing::warn!(hash = %commit.hash, error = %e, "failed to insert git commit capsule");
            }
        }
    }

    append_ingested_hashes(ws, &new_hashes);

    if ingested > 0 && use_color {
        println!(
            "\x1b[2m  + {} git commit{} indexed\x1b[0m",
            ingested,
            if ingested == 1 { "" } else { "s" }
        );
    } else if ingested > 0 {
        println!("  + {} git commit(s) indexed", ingested);
    }

    Ok(ingested)
}

// ============================================================================
// Git tag ingestion as capsules
// ============================================================================

#[derive(Debug, Clone)]
struct TagDetail {
    name: String,
    /// The commit SHA the tag points to (peeled for annotated tags).
    commit_hash: String,
    /// Tag creation timestamp in ms. For annotated tags this is the tagger date;
    /// for lightweight tags it falls back to the commit date.
    timestamp_ms: i64,
    /// Annotated tag message (empty for lightweight tags).
    message: String,
    /// Files touched by the tagged commit (for boundary reconstruction).
    files: Vec<String>,
}

/// Fetch all tags with their commit SHA, date, and message.
///
/// Uses `git tag -l --sort=creatordate --format=...` which works for both
/// annotated and lightweight tags.
fn fetch_tags(repo_root: &Path) -> anyhow::Result<Vec<TagDetail>> {
    // Format: name RS commit_hash RS creator_date_unix RS subject GS
    // %(objecttype) lets us detect annotated vs lightweight tags later if needed.
    let output = std::process::Command::new("git")
        .current_dir(repo_root)
        .args([
            "tag",
            "-l",
            "--sort=creatordate",
            // *objectname dereferences annotated tags to their commit SHA.
            "--format=%(refname:short)\x1f%(*objectname)%(objectname)\x1f%(creatordate:unix)\x1f%(contents:subject)\x1e",
        ])
        .output()
        .context("failed to run git tag")?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let mut tags = Vec::new();

    for rec in raw.split('\x1e') {
        let rec = rec.trim();
        if rec.is_empty() {
            continue;
        }
        let mut parts = rec.splitn(4, '\x1f');
        let name = parts.next().unwrap_or("").trim().to_string();
        let commit_hash = parts.next().unwrap_or("").trim().to_string();
        let ts_str = parts.next().unwrap_or("").trim();
        let message = parts.next().unwrap_or("").trim().to_string();

        if name.is_empty() || commit_hash.is_empty() {
            continue;
        }

        let timestamp_ms = ts_str.parse::<i64>().unwrap_or(0) * 1000;

        // Fetch files touched by the tagged commit so we can reconstruct work boundaries.
        let files = std::process::Command::new("git")
            .current_dir(repo_root)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", &commit_hash])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        tags.push(TagDetail {
            name,
            commit_hash,
            timestamp_ms,
            message,
            files,
        });
    }

    // Oldest-first so capsule timestamps are monotonically increasing
    tags.sort_by_key(|t| t.timestamp_ms);
    Ok(tags)
}

fn tag_to_capsule(t: &TagDetail) -> crate::IntentCapsule {
    let sha7 = &t.commit_hash[..7.min(t.commit_hash.len())];
    let intent = format!("Tagged {} at commit {}", t.name, sha7);
    let decision = t.name.clone();
    let rationale = if t.message.is_empty() {
        format!("Lightweight tag {} pinned at {}", t.name, sha7)
    } else {
        format!("{}\n\nTagged commit: {}", t.message, sha7)
    };
    let symbols: Vec<String> = t.files.iter().take(20).cloned().collect();

    crate::IntentCapsule {
        category: "GitTag".to_string(),
        intent,
        decision,
        rationale,
        next_steps: Vec::new(),
        symbols,
        user_symbols: Vec::new(),
        failure_mode: crate::types::FailureMode::None,
        failure_signals: None,
        extraction_mode: crate::types::ExtractionMode::None,
        questions: vec![],
    }
}

fn ingested_tags_path(ws: &crate::WorkspacePaths) -> std::path::PathBuf {
    crate::workspace::unlost_workspace_dir(&ws.id)
        .join("git")
        .join("ingested_tags.txt")
}

fn load_ingested_tags(ws: &crate::WorkspacePaths) -> std::collections::HashSet<String> {
    let path = ingested_tags_path(ws);
    match std::fs::read_to_string(&path) {
        Ok(s) => s
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => std::collections::HashSet::new(),
    }
}

fn append_ingested_tags(ws: &crate::WorkspacePaths, tags: &[String]) {
    if tags.is_empty() {
        return;
    }
    let path = ingested_tags_path(ws);
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
    for t in tags {
        let _ = writeln!(f, "{}", t.trim());
    }
}

/// Ingest git tags from `repo_root` as capsules into the workspace's LanceDB.
///
/// Each tag becomes a `GitTag` capsule linked to its commit SHA, so session
/// boundaries can be reconstructed (e.g. "what changed between v0.8.0 and v0.9.0?").
/// Deduplicates by tag name in `git/ingested_tags.txt`. Zero LLM cost.
pub async fn ingest_git_tags(
    ws: &crate::WorkspacePaths,
    repo_root: &Path,
    embedder: &crate::embed::Embedder,
    use_color: bool,
) -> anyhow::Result<usize> {
    if crate::workspace::git_toplevel(repo_root).is_none() {
        return Ok(0);
    }

    let already_ingested = load_ingested_tags(ws);
    let tags = fetch_tags(repo_root)?;

    let new_tags: Vec<TagDetail> = tags
        .into_iter()
        .filter(|t| !already_ingested.contains(&t.name))
        .collect();

    if new_tags.is_empty() {
        return Ok(0);
    }

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let mut ingested = 0usize;
    let mut new_names: Vec<String> = Vec::new();

    for tag in &new_tags {
        let capsule = tag_to_capsule(tag);

        let meta = crate::ResponseMeta {
            source: "git".to_string(),
            upstream_host: "git".to_string(),
            // Store "tag:<name>" so narrative.rs can emit "ref=tag:<name>" citations.
            request_path: format!("tag:{}", tag.name),
            http_status: 0,
            agent_session_id: None,
            usage: None,
        };

        let embed_text = if tag.message.is_empty() {
            format!("Tag {} at commit {}", tag.name, &tag.commit_hash[..7.min(tag.commit_hash.len())])
        } else {
            format!("Tag {}: {}", tag.name, tag.message)
        };

        match crate::storage::insert_capsule_row(
            &db,
            embedder,
            0,
            0,
            tag.timestamp_ms,
            &meta,
            None,
            None,
            &capsule,
            &crate::types::TurnEval::default(),
            Some(&embed_text),
            None, // head_sha: not applicable for git tag ingestion
            None, // commit_sha: not applicable for git tag ingestion
        )
        .await
        {
            Ok(_) => {
                new_names.push(tag.name.clone());
                ingested += 1;
            }
            Err(e) => {
                tracing::warn!(tag = %tag.name, error = %e, "failed to insert git tag capsule");
            }
        }
    }

    append_ingested_tags(ws, &new_names);

    if ingested > 0 && use_color {
        println!(
            "\x1b[2m  + {} git tag{} indexed\x1b[0m",
            ingested,
            if ingested == 1 { "" } else { "s" }
        );
    } else if ingested > 0 {
        println!("  + {} git tag(s) indexed", ingested);
    }

    Ok(ingested)
}

/// Standalone entry point for `unlost replay git`.
pub async fn replay_git(
    path: String,
    max_commits: usize,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let dir_path = std::path::Path::new(&path);
    let ws = crate::workspace::get_or_create_workspace_paths(dir_path)?;

    let repo_root = crate::workspace::git_toplevel(dir_path).ok_or_else(|| {
        anyhow::anyhow!("no git repository found at or above {}", dir_path.display())
    })?;

    let use_color = std::io::IsTerminal::is_terminal(&std::io::stdout())
        && std::env::var_os("NO_COLOR").is_none();

    if use_color {
        println!(
            "\x1b[36mi\x1b[0m Ingesting git history from \x1b[1m{}\x1b[0m",
            repo_root.display()
        );
    } else {
        println!("Ingesting git history from {}", repo_root.display());
    }

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    let ingested = ingest_git_commits(&ws, &repo_root, &embedder, max_commits, use_color).await?;

    if use_color {
        println!(
            "\x1b[1;32m✓\x1b[0m Git replay complete: \x1b[1;36m{}\x1b[0m commit{} ingested",
            ingested,
            if ingested == 1 { "" } else { "s" }
        );
    } else {
        println!("Git replay complete: {} commit(s) ingested", ingested);
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct GitCommit {
    pub hash: String,
    pub timestamp_ms: i64,
    pub subject: String,
    pub files: Vec<String>,
}

pub fn get_commits_for_range(
    repo_root: &Path,
    since_ms: i64,
    until_ms: i64,
) -> anyhow::Result<Vec<GitCommit>> {
    let since_s = since_ms / 1000;
    let until_s = until_ms / 1000;

    let log_args: Vec<String> = vec![
        "log".to_string(),
        format!("--since={}", since_s),
        format!("--until={}", until_s),
        "--pretty=format:%H%x1f%ct%x1f%s%x1e".to_string(),
    ];

    let output = std::process::Command::new("git")
        .current_dir(repo_root)
        .args(&log_args)
        .output()
        .context("failed to run git log")?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let mut commits = Vec::new();

    for rec in raw.split('\x1e') {
        let rec = rec.trim();
        if rec.is_empty() {
            continue;
        }
        let mut parts = rec.split('\x1f');
        let hash = parts.next().unwrap_or("").trim().to_string();
        let ts_str = parts.next().unwrap_or("").trim();
        let subj = parts.next().unwrap_or("").trim().to_string();

        if hash.is_empty() {
            continue;
        }

        let ts_ms = ts_str.parse::<i64>().unwrap_or(0) * 1000;

        // Fetch files for this commit
        let files = std::process::Command::new("git")
            .current_dir(repo_root)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", &hash])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        commits.push(GitCommit {
            hash,
            timestamp_ms: ts_ms,
            subject: subj,
            files,
        });
    }

    // Sort by timestamp ascending
    commits.sort_by_key(|c| c.timestamp_ms);

    Ok(commits)
}

/// Find commits that likely correspond to a turn based on timestamp and touched files.
pub fn find_corresponding_commits(
    turn_timestamp_ms: i64,
    touched_paths: &[String],
    available_commits: &[GitCommit],
    window_ms: i64,
) -> Vec<GitCommit> {
    let mut matches = Vec::new();

    for commit in available_commits {
        // Commit must be within the window (e.g. 5 minutes after the turn)
        let diff = commit.timestamp_ms - turn_timestamp_ms;
        if diff >= 0 && diff <= window_ms {
            // Check if there is overlap in touched files
            let commit_files: std::collections::HashSet<_> = commit.files.iter().collect();
            let turn_files: std::collections::HashSet<_> = touched_paths.iter().collect();

            let has_overlap = turn_files.iter().any(|f| commit_files.contains(f));

            if has_overlap || touched_paths.is_empty() {
                matches.push(commit.clone());
            }
        }
    }

    matches
}
