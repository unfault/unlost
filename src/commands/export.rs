use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::export::{
    capsule_filename, capsule_to_markdown, category_narrative_prompt, category_readme_static,
    category_readme_with_narrative, index_md, ts_ms_to_iso, CategoryStat, ExportCapsule,
};

// ─── local JSONL deserialization types (mirrors reindex.rs) ─────────────────

#[derive(Deserialize)]
struct JsonCapsule {
    #[serde(default)]
    id: Option<String>,
    ts_ms: i64,
    #[serde(default)]
    conn_id: Option<u64>,
    #[serde(default)]
    agent_session_id: String,
    source: String,
    #[serde(default)]
    head_sha: Option<String>,
    #[serde(default)]
    commit_sha: Option<String>,
    capsule: CapsuleFields,
}

#[derive(Deserialize)]
struct CapsuleFields {
    category: String,
    intent: String,
    decision: String,
    rationale: String,
    #[serde(default)]
    next_steps: Vec<String>,
    #[serde(default)]
    symbols: Vec<String>,
    #[serde(default)]
    user_symbols: Vec<String>,
    #[serde(default)]
    failure_mode: Option<String>,
    #[serde(default)]
    failure_signals: Option<String>,
    #[serde(default)]
    questions: Vec<String>,
}

fn parse_failure_mode(s: Option<&str>) -> crate::types::FailureMode {
    match s {
        Some("drift") => crate::types::FailureMode::Drift,
        Some("rediscovery") => crate::types::FailureMode::Rediscovery,
        Some("decision_conflict") => crate::types::FailureMode::DecisionConflict,
        Some("retry_spiral") => crate::types::FailureMode::RetrySpiral,
        Some("false_progress") => crate::types::FailureMode::FalseProgress,
        Some("unbounded_horizon") => crate::types::FailureMode::UnboundedHorizon,
        _ => crate::types::FailureMode::None,
    }
}

// ─── command entry point ─────────────────────────────────────────────────────

pub async fn run(
    dir: Option<String>,
    path: String,
    narrative: bool,
    force: bool,
    llm_model: Option<String>,
) -> anyhow::Result<()> {
    // 1. Resolve workspace
    let ws = crate::workspace::get_or_create_workspace_paths(Path::new(&path))?;

    // 2. Resolve export directory: CLI flag > config default > error
    let export_dir: PathBuf = if let Some(d) = dir {
        expand_tilde(&d)
    } else {
        let cfg = crate::workspace::load_workspace_config();
        match cfg.export_dir {
            Some(d) => expand_tilde(&d),
            None => {
                anyhow::bail!(
                    "No export directory specified.\n\
                     Use --dir <path>, or set a default with:\n\
                     \n  unlost config export-dir ~/notes/unlost\n"
                );
            }
        }
    };

    // 3. Read capsules from JSONL
    let jsonl_path = &ws.capsules_jsonl;
    if !jsonl_path.exists() {
        println!("No capsules.jsonl found at {}", jsonl_path.display());
        println!("Nothing to export.");
        return Ok(());
    }

    let capsules = load_capsules(jsonl_path).await?;
    if capsules.is_empty() {
        println!("No capsules found. Nothing to export.");
        return Ok(());
    }

    // 4. Group by category (preserving insertion/chronological order)
    let mut by_category: BTreeMap<String, Vec<ExportCapsule>> = BTreeMap::new();
    for cap in capsules {
        by_category
            .entry(cap.capsule.category.clone())
            .or_default()
            .push(cap);
    }

    // 5. Create top-level export directory
    std::fs::create_dir_all(&export_dir)
        .with_context(|| format!("failed to create export dir {}", export_dir.display()))?;

    let mut total_written = 0usize;
    let mut total_skipped = 0usize;
    let mut stats: Vec<CategoryStat> = Vec::new();

    // 6. For each category: create dir, write capsule files, write README
    for (category, caps) in &by_category {
        let cat_dir = export_dir.join(category);
        std::fs::create_dir_all(&cat_dir)
            .with_context(|| format!("failed to create category dir {}", cat_dir.display()))?;

        // Compute all filenames up-front; deduplicate by appending counter on collision
        let mut filenames: Vec<String> = Vec::with_capacity(caps.len());
        let mut seen: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for cap in caps {
            let raw = capsule_filename(cap);
            let count = seen.entry(raw.clone()).or_insert(0);
            let fname = if *count == 0 {
                raw.clone()
            } else {
                // Insert counter before extension: foo.md → foo-2.md
                let stem = raw.trim_end_matches(".md");
                format!("{}-{}.md", stem, *count + 1)
            };
            *count += 1;
            filenames.push(fname);
        }

        // Write individual capsule files (incremental by default)
        for (cap, filename) in caps.iter().zip(filenames.iter()) {
            let file_path = cat_dir.join(filename);
            if file_path.exists() && !force {
                total_skipped += 1;
                continue;
            }
            let content = capsule_to_markdown(cap);
            std::fs::write(&file_path, content)
                .with_context(|| format!("failed to write {}", file_path.display()))?;
            total_written += 1;
        }

        // Write README (always regenerated — it's a derived index)
        let pairs: Vec<(&ExportCapsule, &str)> = caps
            .iter()
            .zip(filenames.iter().map(|s| s.as_str()))
            .collect();

        let readme_content = if narrative {
            let prompt = category_narrative_prompt(category, &caps.iter().collect::<Vec<_>>());
            match generate_narrative(llm_model.as_deref(), &prompt).await {
                Ok(narr) => category_readme_with_narrative(category, &pairs, &narr),
                Err(e) => {
                    eprintln!(
                        "warning: LLM narrative for category '{}' failed: {e}. Using static README.",
                        category
                    );
                    category_readme_static(category, &pairs)
                }
            }
        } else {
            category_readme_static(category, &pairs)
        };

        let readme_path = cat_dir.join("README.md");
        std::fs::write(&readme_path, readme_content)
            .with_context(|| format!("failed to write {}", readme_path.display()))?;

        stats.push(CategoryStat {
            category: category.clone(),
            count: caps.len(),
        });
    }

    // 7. Write top-level INDEX.md (always regenerated)
    let workspace_root = ws.root.to_string_lossy().to_string();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let exported_at = ts_ms_to_iso(now_ms);
    let index_content = index_md(&workspace_root, &exported_at, &stats);
    let index_path = export_dir.join("INDEX.md");
    std::fs::write(&index_path, index_content)
        .with_context(|| format!("failed to write {}", index_path.display()))?;

    // 8. Summary
    let total_capsules: usize = by_category.values().map(|v| v.len()).sum();
    println!(
        "Exported {} capsule{} across {} categor{} to {}",
        total_written,
        if total_written == 1 { "" } else { "s" },
        stats.len(),
        if stats.len() == 1 { "y" } else { "ies" },
        export_dir.display()
    );
    if total_skipped > 0 {
        println!(
            "  {} file{} already existed and were skipped (use --force to overwrite)",
            total_skipped,
            if total_skipped == 1 { "" } else { "s" }
        );
    }
    let total_files = total_written + total_skipped;
    println!(
        "  Total: {} capsule file{}, {} categor{}, INDEX.md",
        total_files,
        if total_files == 1 { "" } else { "s" },
        stats.len(),
        if stats.len() == 1 { "y" } else { "ies" },
    );
    if total_capsules > 0 && !narrative {
        println!("  Tip: run with --narrative to generate LLM-written category summaries");
    }

    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

async fn load_capsules(jsonl_path: &Path) -> anyhow::Result<Vec<ExportCapsule>> {
    let file = File::open(jsonl_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut capsules = Vec::new();
    let mut line_num = 0usize;

    while let Some(line) = lines.next_line().await? {
        line_num += 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw: JsonCapsule = serde_json::from_str(line)
            .with_context(|| format!("failed to parse capsules.jsonl line {line_num}"))?;

        // Derive a stable ID: use the stored one if present, otherwise synthesize from ts+conn
        let id = raw.id.clone().unwrap_or_else(|| {
            format!(
                "cap-{}-{}",
                raw.ts_ms,
                raw.conn_id.unwrap_or(0)
            )
        });

        let failure_mode = parse_failure_mode(raw.capsule.failure_mode.as_deref());

        let cap = ExportCapsule {
            id,
            ts_ms: raw.ts_ms,
            agent_session_id: if raw.agent_session_id.is_empty() {
                None
            } else {
                Some(raw.agent_session_id)
            },
            head_sha: raw.head_sha,
            commit_sha: raw.commit_sha,
            source: raw.source,
            capsule: crate::types::IntentCapsule {
                category: raw.capsule.category,
                intent: raw.capsule.intent,
                decision: raw.capsule.decision,
                rationale: raw.capsule.rationale,
                next_steps: raw.capsule.next_steps,
                symbols: raw.capsule.symbols,
                user_symbols: raw.capsule.user_symbols,
                failure_mode,
                failure_signals: raw.capsule.failure_signals,
                extraction_mode: crate::types::ExtractionMode::None,
                questions: raw.capsule.questions,
            },
        };
        capsules.push(cap);
    }

    Ok(capsules)
}

/// Expand a leading `~` to the home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return PathBuf::from(home).join(rest);
        }
    } else if path == "~" {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

/// Call the configured LLM with a free-text prompt and return the narrative string.
async fn generate_narrative(
    model_override: Option<&str>,
    prompt: &str,
) -> anyhow::Result<String> {
    let result = crate::llm::llm_extract::<crate::types::QueryNarrativeOutput>(
        model_override,
        "You are a technical writer summarising software development decisions for a knowledge base. \
         Output only the requested narrative text in the `narrative` field.",
        prompt,
    )
    .await?;
    Ok(result.narrative)
}
