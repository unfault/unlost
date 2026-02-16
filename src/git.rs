use anyhow::Context;
use serde::Serialize;
use std::path::Path;

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
