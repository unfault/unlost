use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use lancedb::query::{ExecutableQuery, QueryBase};
use std::time::Duration;

/// A skill installed in the workspace.
#[derive(Debug, Clone)]
pub struct InstalledSkill {
    pub name: String,
    pub description: String,
    /// Relative path from workspace root, for display
    pub path: String,
}

/// Scan common skill locations under `workspace_root` and return all discovered skills.
///
/// Locations checked (ordered by priority):
/// - `.opencode/skills/<name>/SKILL.md`
/// - `.claude/skills/<name>/SKILL.md`
/// - `.cursor/skills/<name>/SKILL.md`
/// - `.aider/skills/<name>/SKILL.md`
///
/// Each SKILL.md is expected to have YAML frontmatter with `name` and `description`.
/// Falls back to the directory name if frontmatter is absent.
pub fn discover_installed_skills(workspace_root: &std::path::Path) -> Vec<InstalledSkill> {
    let skill_dirs = [
        ".opencode/skills",
        ".claude/skills",
        ".cursor/skills",
        ".aider/skills",
    ];

    let mut skills: Vec<InstalledSkill> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for base in &skill_dirs {
        let base_path = workspace_root.join(base);
        if !base_path.is_dir() {
            continue;
        }
        let entries = match std::fs::read_dir(&base_path) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let skill_dir = entry.path();
            if !skill_dir.is_dir() {
                continue;
            }
            let skill_md = skill_dir.join("SKILL.md");
            if !skill_md.exists() {
                continue;
            }

            let dir_name = skill_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            // Deduplicate by name — first occurrence wins
            if seen.contains(&dir_name) {
                continue;
            }
            seen.insert(dir_name.clone());

            let (name, description) = parse_skill_frontmatter(&skill_md, &dir_name);
            let rel_path = format!(
                "{}/{}",
                base,
                skill_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            );

            skills.push(InstalledSkill {
                name,
                description,
                path: rel_path,
            });
        }
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Parse `name` and `description` from YAML frontmatter in a SKILL.md file.
/// Returns `(dir_name, "")` if frontmatter is missing or unparseable.
fn parse_skill_frontmatter(path: &std::path::Path, fallback_name: &str) -> (String, String) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return (fallback_name.to_string(), String::new()),
    };

    // Expect frontmatter delimited by `---`
    if !content.starts_with("---") {
        return (fallback_name.to_string(), String::new());
    }
    let after_open = &content[3..];
    let close = match after_open.find("\n---") {
        Some(p) => p,
        None => return (fallback_name.to_string(), String::new()),
    };
    let frontmatter = &after_open[..close];

    let mut name = fallback_name.to_string();
    let mut description = String::new();

    for line in frontmatter.lines() {
        if let Some(val) = line.strip_prefix("name:") {
            name = val.trim().trim_matches('"').trim_matches('\'').to_string();
        } else if let Some(val) = line.strip_prefix("description:") {
            description = val.trim().trim_matches('"').trim_matches('\'').to_string();
        }
    }

    (name, description)
}

pub async fn run(
    mode: crate::cli::ReflectMode,
    session: Option<String>,
    since: Option<String>,
    llm_model: Option<String>,
    output: OutputFormat,
    path: String,
) -> anyhow::Result<()> {
    let dir_path = std::path::Path::new(&path);
    let ws = crate::workspace::get_or_create_workspace_paths(dir_path)?;

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Loading capsules...");
        Some(pb)
    } else {
        None
    };

    // Parse time filters
    let since_ms: Option<i64> = if let Some(ref s) = since {
        crate::util::parse_time_filter(s)?
    } else {
        None
    };

    // Fetch conversational capsules for the session/period
    let capsules = fetch_reflect_capsules(&ws, session.as_deref(), since_ms).await?;

    if capsules.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No capsules found for this workspace.");
        println!("Record a session first: run `unlost record` or use a supported agent.");
        return Ok(());
    }

    // Filter to only capsules that have TurnEval data (produced after v0.13)
    let eval_capsules: Vec<_> = capsules
        .iter()
        .filter(|h| h.turn_eval.is_some())
        .cloned()
        .collect();

    if eval_capsules.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No turn evaluation data found yet.");
        println!(
            "TurnEval is collected on new sessions going forward. \
             Try again after recording a new session."
        );
        return Ok(());
    }

    let model_name = if let Some(ref m) = llm_model {
        m.clone()
    } else if let Some(cfg) = crate::llm::get_llm_config() {
        match cfg {
            crate::config::LlmConfig::Openai { model, .. } => model,
            crate::config::LlmConfig::Anthropic { model, .. } => model,
            crate::config::LlmConfig::Ollama { model, .. } => model,
            crate::config::LlmConfig::Custom { model, .. } => model,
        }
    } else {
        "gpt-4o-mini".to_string()
    };

    // Discover installed skills — only relevant for tune/both modes
    let installed_skills = match mode {
        crate::cli::ReflectMode::Tune | crate::cli::ReflectMode::Both => {
            let workspace_root = crate::workspace::git_toplevel(dir_path)
                .unwrap_or_else(|| dir_path.to_path_buf());
            discover_installed_skills(&workspace_root)
        }
        crate::cli::ReflectMode::Coach => vec![],
    };

    if let Some(pb) = spinner.as_ref() {
        pb.set_message(format!("Reflecting with {} ({} turns)...", model_name, eval_capsules.len()));
    }

    let narrative = crate::narrative::llm_reflect_narrative(
        llm_model.as_deref(),
        mode,
        &eval_capsules,
        session.as_deref(),
        &installed_skills,
    )
    .await?;

    if let Some(pb) = spinner.as_ref() {
        pb.finish_and_clear();
    }

    let out = crate::narrative::render_reflect(output, mode, &narrative);
    print!("{}", out);

    Ok(())
}

/// Fetch conversational capsules for reflect, filtered by session or time window.
async fn fetch_reflect_capsules(
    ws: &crate::WorkspacePaths,
    session_id: Option<&str>,
    since_ms: Option<i64>,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let table = match crate::storage::open_capsules_table(&db).await {
        Ok(t) => t,
        Err(_) => return Ok(vec![]),
    };

    use futures_util::TryStreamExt;
    use arrow_array::RecordBatch;

    let mut q = table.query();

    let mut filters: Vec<String> = Vec::new();

    // Only conversational capsules (not git, not replay)
    filters.push("source != 'git'".to_string());

    if let Some(sid) = session_id {
        let sid_esc = sid.replace('\'', "\\'");
        filters.push(format!("agent_session_id = '{sid_esc}'"));
    }

    if let Some(ms) = since_ms {
        filters.push(format!(
            "CAST(ts_ms AS BIGINT) >= CAST({ms} AS BIGINT)"
        ));
    }

    let combined = filters.join(" AND ");
    q = q.only_if(&combined);

    let batches: Vec<RecordBatch> = q
        .limit(500)
        .execute()
        .await?
        .try_collect()
        .await?;

    let mut hits = crate::storage::record_batches_to_hits(&batches, &ws.id)?;
    hits.sort_by_key(|h| h.ts_ms);
    Ok(hits)
}
