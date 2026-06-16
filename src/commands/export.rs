use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::export::{
    category_narrative_prompt, category_readme_static, category_readme_with_narrative,
    decisions_md, index_md, is_substantive, map_category_fallback, project_name_from_root,
    symbol_page, symbol_page_filename, ts_ms_to_iso, CategoryStat, ExportCapsule, ProjectStat,
    TAXONOMY,
};

/// Categories that are filtered noise and don't deserve their own folder.
/// Substantive decisions in these buckets still appear in decisions.md.
const SKIP_CATEGORIES: &[&str] = &["meta", "other"];

// ─── JSONL deserialization ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JsonCapsule {
    #[serde(default)]
    id: Option<String>,
    ts_ms: i64,
    #[serde(default)]
    conn_id: Option<u64>,
    #[serde(default)]
    agent_session_id: Option<String>,
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

// ─── Minimum symbol-page threshold ───────────────────────────────────────────

/// Only create a symbol page if a file appears in at least this many
/// substantive capsules. Keeps noise symbols (e.g. lock files) out.
/// A symbol must appear in at least this many substantive capsules (at top-3
/// position) to earn a knowledge page. Lower values explode the file count
/// with marginal symbols; higher values lose useful smaller files.
const MIN_CAPSULES_FOR_SYMBOL_PAGE: usize = 8;

/// Skip symbol pages for these patterns — they're infrastructure, not knowledge.
const SYMBOL_PAGE_SKIPLIST: &[&str] = &[
    "Cargo.lock", "package-lock.json", "yarn.lock",
    "go.sum", "poetry.lock", ".gitignore", ".env",
];

/// Cheap structural checks: discard things that obviously can't be a path
/// or a code identifier (prose, flags, attributes, env vars, quoted strings,
/// runtime IDs). Returns `true` to **skip**.
fn is_structural_noise(sym: &str) -> bool {
    if sym.len() < 4 { return true; }
    if sym.contains(' ') { return true; }

    // Starts with a non-alphanumeric that signals CLI flag / attribute / token.
    let first = sym.chars().next().unwrap_or(' ');
    if matches!(first, '-' | '#' | '$' | '*' | '%' | '"' | '\'' | '`' | '@' | '!') {
        return true;
    }

    if sym.contains('=') { return true; }   // env-var assignment
    if sym.starts_with("#[") { return true; } // rust attribute
    if sym.contains('\\') { return true; }   // Windows path or escape — almost always machine state

    // Runtime tokens that look like identifiers but aren't knowledge anchors.
    // Workspace IDs (wks_*), session IDs (ses_*), connection IDs etc. are
    // generated at runtime and have no semantic meaning to a human reader.
    if sym.starts_with("wks_") || sym.starts_with("ses_") || sym.starts_with("conn_") {
        return true;
    }

    // Explicit lock-file blocklist (basename only).
    let basename = sym.rsplit('/').next().unwrap_or(sym);
    if SYMBOL_PAGE_SKIPLIST.contains(&basename) { return true; }

    false
}

/// For non-path symbols (no `/`), is this string a plausible code identifier?
/// Accepts qualified names (`Foo::bar`, `pkg.Class`), CamelCase types,
/// snake_case identifiers ≥ 8 chars. Rejects plain English words.
fn is_identifier_like(sym: &str) -> bool {
    if sym.contains("::") { return true; }
    if sym.contains('.') && !sym.starts_with('.') { return true; }

    let chars: Vec<char> = sym.chars().collect();
    let has_camel = chars.windows(2).any(|w| w[0].is_lowercase() && w[1].is_uppercase());
    if has_camel { return true; }

    if sym.contains('_')
        && sym.len() >= 8
        && sym.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        return true;
    }

    false
}

// ─── Command entry point ─────────────────────────────────────────────────────

pub async fn run(
    dir: Option<String>,
    path: String,
    narrative: bool,
    force: bool,
    llm_model: Option<String>,
) -> anyhow::Result<()> {
    // 1. Resolve export directory
    let export_dir: PathBuf = if let Some(d) = dir {
        expand_tilde(&d)
    } else {
        let cfg = crate::workspace::load_workspace_config();
        match cfg.export_dir {
            Some(d) => expand_tilde(&d),
            None => anyhow::bail!(
                "No export directory specified.\n\
                 Use --dir <path>, or set a default with:\n\
                 \n  unlost config export-dir ~/notes/unlost\n"
            ),
        }
    };

    // 2. Collect capsules from all registered workspaces
    //    (falls back to current workspace only if config has no registered workspaces)
    let workspace_data = collect_all_workspaces(&path).await?;

    if workspace_data.is_empty() {
        println!("No capsules found. Nothing to export.");
        return Ok(());
    }

    let total_raw: usize = workspace_data.iter().map(|w| w.capsules.len()).sum();
    println!("Loaded {} capsules from {} workspace{}",
        total_raw,
        workspace_data.len(),
        if workspace_data.len() == 1 { "" } else { "s" });

    // Build per-project git-tracked-file set once. Used to filter LLM-extracted
    // symbols: a path is a knowledge anchor only if git tracks it. This is
    // self-tuning to whatever the user's `.gitignore` excludes — no hardcoded
    // build-artefact lists, no per-toolchain assumptions.
    let mut tracked_by_project: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    for w in &workspace_data {
        if let Some(root) = &w.root {
            tracked_by_project.insert(w.project.clone(), tracked_files(root));
        }
    }

    // 3. Map raw categories → taxonomy buckets (single batch per workspace)
    //    Build the full flat list with project name and bucket assigned.
    let mut all_capsules: Vec<ExportCapsule> = Vec::with_capacity(total_raw);
    for w in &workspace_data {
        let raw_cats: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            w.capsules.iter().map(|c| c.capsule.category.clone())
                .filter(|c| seen.insert(c.clone()))
                .collect()
        };
        let cat_map = build_category_map(&raw_cats, llm_model.as_deref()).await;

        for mut cap in w.capsules.clone() {
            let bucket = cat_map.get(&cap.capsule.category)
                .cloned()
                .unwrap_or_else(|| map_category_fallback(&cap.capsule.category).to_string());
            cap.project = w.project.clone();
            cap.capsule.category = bucket;
            all_capsules.push(cap);
        }
    }

    // Sort globally by timestamp
    all_capsules.sort_by_key(|c| c.ts_ms);

    // 4. Partition into substantive / noise
    let substantive: Vec<&ExportCapsule> = all_capsules.iter()
        .filter(|c| is_substantive(c))
        .collect();

    println!("  {} substantive ({} filtered as noise)",
        substantive.len(),
        all_capsules.len() - substantive.len());

    // 5. Create directory structure
    std::fs::create_dir_all(&export_dir)
        .with_context(|| format!("failed to create {}", export_dir.display()))?;
    let categories_dir = export_dir.join("categories");
    let symbols_dir = export_dir.join("symbols");
    std::fs::create_dir_all(&categories_dir)?;
    std::fs::create_dir_all(&symbols_dir)?;

    // 6. Write decisions.md (cross-project, all substantive)
    {
        let decisions_path = export_dir.join("decisions.md");
        let content = decisions_md(&substantive);
        std::fs::write(&decisions_path, content)?;
        println!("  decisions.md ({} entries)", substantive.len());
    }

    // 7. Write symbol knowledge pages
    //    Group substantive capsules by (project, symbol)
    let mut sym_written = 0usize;
    let mut project_symbol_counts: HashMap<String, usize> = HashMap::new();

    // Collect: project → symbol → [&ExportCapsule]
    // Only attribute a capsule to a symbol if it appears in the first 3 positions
    // of the symbol list. Position encodes LLM-assigned relevance: hub files like
    // src/main.rs appear at position 10+ on capsules that aren't about them at all.
    const MAX_SYMBOL_POSITION: usize = 3;
    // Look up workspace root for each project (None for un-rooted workspaces).
    let project_roots: HashMap<&str, &Path> = workspace_data.iter()
        .filter_map(|w| w.root.as_deref().map(|r| (w.project.as_str(), r)))
        .collect();

    let mut by_project_symbol: BTreeMap<String, BTreeMap<String, Vec<&ExportCapsule>>> =
        BTreeMap::new();
    for cap in &substantive {
        for (pos, sym) in cap.capsule.symbols.iter().enumerate() {
            if pos >= MAX_SYMBOL_POSITION { break; }
            if is_structural_noise(sym) { continue; }

            // A symbol is path-shaped if it contains a separator OR ends in a
            // file extension. Both must resolve to a git-tracked file under
            // the project root to earn a page; canonicalised key avoids
            // duplicate pages when the same file appears as relative and
            // absolute paths in different capsules.
            let looks_like_path = sym.contains('/') || has_file_extension(sym);
            let canonical: String = if looks_like_path {
                match (project_roots.get(cap.project.as_str()),
                       tracked_by_project.get(cap.project.as_str()))
                {
                    (Some(root), Some(tracked)) => match canonical_tracked_path(sym, root, tracked) {
                        Some(c) => c,
                        None => continue, // not tracked → drop
                    },
                    _ => continue, // can't validate → drop
                }
            } else if is_identifier_like(sym) {
                sym.clone()
            } else {
                continue;
            };

            by_project_symbol
                .entry(cap.project.clone())
                .or_default()
                .entry(canonical)
                .or_default()
                .push(cap);
        }
    }

    for (project, symbols) in &by_project_symbol {
        let proj_sym_dir = symbols_dir.join(project);
        std::fs::create_dir_all(&proj_sym_dir)?;

        for (sym, caps) in symbols {
            if caps.len() < MIN_CAPSULES_FOR_SYMBOL_PAGE { continue; }

            let filename = symbol_page_filename(sym);
            let file_path = proj_sym_dir.join(&filename);

            if file_path.exists() && !force { continue; }

            let content = symbol_page(sym, project, caps);
            std::fs::write(&file_path, content)
                .with_context(|| format!("failed to write {}", file_path.display()))?;
            sym_written += 1;
            *project_symbol_counts.entry(project.clone()).or_insert(0) += 1;
        }
    }
    println!("  symbols/ → {} pages across {} project{}",
        sym_written,
        project_symbol_counts.len(),
        if project_symbol_counts.len() == 1 { "" } else { "s" });

    // 8. Write category READMEs (one file per category — no per-capsule files).
    //    Skip noise buckets (meta, other) entirely. Substantive failure-mode
    //    capsules from those buckets still appear in decisions.md.
    let mut by_category: BTreeMap<String, Vec<&ExportCapsule>> = BTreeMap::new();
    for cap in &substantive {
        let bucket = &cap.capsule.category;
        if SKIP_CATEGORIES.contains(&bucket.as_str()) { continue; }
        by_category.entry(bucket.clone()).or_default().push(cap);
    }

    let mut cat_stats: Vec<CategoryStat> = Vec::new();

    for (bucket, cats) in &by_category {
        let cat_dir = categories_dir.join(bucket);
        std::fs::create_dir_all(&cat_dir)?;

        // Sort newest-first within the README
        let mut sorted = cats.clone();
        sorted.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));

        let readme = if narrative {
            let cats_for_prompt: Vec<&ExportCapsule> = sorted.iter().copied().collect();
            let prompt = category_narrative_prompt(bucket, &cats_for_prompt);
            match generate_narrative(llm_model.as_deref(), &prompt).await {
                Ok(narr) => category_readme_with_narrative(bucket, &sorted, &narr),
                Err(e) => {
                    eprintln!("warning: narrative for '{bucket}' failed: {e}");
                    category_readme_static(bucket, &sorted)
                }
            }
        } else {
            category_readme_static(bucket, &sorted)
        };

        std::fs::write(cat_dir.join("README.md"), readme)?;
        cat_stats.push(CategoryStat { category: bucket.clone(), count: cats.len() });
    }

    // Fill in zero-count buckets (excluding skipped) so INDEX.md is complete
    for (bucket, _) in TAXONOMY {
        if SKIP_CATEGORIES.contains(bucket) { continue; }
        if !cat_stats.iter().any(|s| s.category == *bucket) {
            cat_stats.push(CategoryStat { category: bucket.to_string(), count: 0 });
        }
    }
    cat_stats.sort_by(|a, b| b.count.cmp(&a.count));

    println!("  categories/ → {} READMEs", by_category.len());

    // 9. Write INDEX.md
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0);
    let exported_at = ts_ms_to_iso(now_ms);

    let failure_count = substantive.iter()
        .filter(|c| c.capsule.failure_mode != crate::types::FailureMode::None)
        .count();

    let proj_stats: Vec<ProjectStat> = workspace_data.iter().map(|w| {
        let sub = w.capsules.iter()
            .filter(|c| is_substantive(c))
            .count();
        let sym_pages = project_symbol_counts.get(&w.project).copied().unwrap_or(0);
        ProjectStat {
            project: w.project.clone(),
            total_capsules: w.capsules.len(),
            substantive_capsules: sub,
            symbol_pages: sym_pages,
        }
    }).collect();

    let index = index_md(
        &exported_at,
        &proj_stats,
        &cat_stats,
        substantive.len(),
        failure_count,
        sym_written,
    );
    std::fs::write(export_dir.join("INDEX.md"), index)?;

    println!("\nExported to {}", export_dir.display());
    if !narrative {
        println!("  Tip: run with --narrative to add LLM-written category summaries");
    }

    Ok(())
}

// ─── Workspace collection ─────────────────────────────────────────────────────

/// One row of workspace data needed for export: the human project name,
/// the workspace root on disk (for git tracked-file lookup), and the loaded
/// capsules. The root is `None` when we couldn't recover it (rare path).
struct WorkspaceData {
    project: String,
    root: Option<PathBuf>,
    capsules: Vec<ExportCapsule>,
}

/// Load capsules from all registered workspaces.
async fn collect_all_workspaces(current_path: &str) -> anyhow::Result<Vec<WorkspaceData>> {
    let cfg = crate::workspace::load_workspace_config();
    let mut result: Vec<WorkspaceData> = Vec::new();

    if cfg.workspaces.is_empty() {
        let ws = crate::workspace::get_or_create_workspace_paths(Path::new(current_path))?;
        let project = project_name_from_root(&ws.root.to_string_lossy());
        if ws.capsules_jsonl.exists() {
            let caps = load_capsules(&ws.capsules_jsonl, &project).await?;
            result.push(WorkspaceData { project, root: Some(ws.root.clone()), capsules: caps });
        }
        return Ok(result);
    }

    for (_, info) in &cfg.workspaces {
        let jsonl = PathBuf::from(&info.capsules_jsonl);
        if !jsonl.exists() { continue; }

        let project = project_name_from_root(&info.root);
        let root = Some(PathBuf::from(&info.root));
        match load_capsules(&jsonl, &project).await {
            Ok(caps) if !caps.is_empty() => {
                result.push(WorkspaceData { project, root, capsules: caps });
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("warning: skipping workspace {} ({}): {e}", info.id, info.root);
            }
        }
    }

    result.sort_by(|a, b| a.project.cmp(&b.project));
    Ok(result)
}

/// Return the set of files tracked by git in `root`, as relative paths.
/// Returns an empty set if `root` is not a git repo or git is unavailable.
fn tracked_files(root: &Path) -> std::collections::HashSet<String> {
    use std::process::Command;
    let output = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["ls-files", "-z"])
        .output();
    let Ok(output) = output else { return Default::default() };
    if !output.status.success() {
        return Default::default();
    }
    output
        .stdout
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| std::str::from_utf8(s).ok())
        .map(|s| s.to_string())
        .collect()
}

/// Returns true if the symbol looks like it has a file extension
/// (e.g. `metrics.jsonl`, `CHANGELOG.md`, `Cargo.toml`).
/// Used to detect bare-filename "path-shaped" symbols that contain no `/`.
fn has_file_extension(sym: &str) -> bool {
    if let Some(dot_pos) = sym.rfind('.') {
        // Skip dotfiles like `.gitignore` (no real extension).
        if dot_pos == 0 { return false; }
        let ext = &sym[dot_pos + 1..];
        // Extension must be 1–6 alphanumeric chars.
        !ext.is_empty()
            && ext.len() <= 6
            && ext.chars().all(|c| c.is_ascii_alphanumeric())
    } else {
        false
    }
}

/// Resolve `sym` to a canonical, git-tracked relative path under `root`,
/// or `None` if it isn't tracked. Handles absolute paths (stripped against
/// root) and relative paths (used as-is). Also collapses leading `./`.
fn canonical_tracked_path(
    sym: &str,
    root: &Path,
    tracked: &std::collections::HashSet<String>,
) -> Option<String> {
    let rel: String = if sym.starts_with('/') {
        Path::new(sym).strip_prefix(root).ok()?.to_string_lossy().to_string()
    } else {
        sym.strip_prefix("./").unwrap_or(sym).to_string()
    };
    if tracked.contains(&rel) {
        Some(rel)
    } else {
        None
    }
}

async fn load_capsules(jsonl_path: &Path, project: &str) -> anyhow::Result<Vec<ExportCapsule>> {
    let file = File::open(jsonl_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut capsules = Vec::new();
    let mut line_num = 0usize;

    while let Some(line) = lines.next_line().await? {
        line_num += 1;
        let line = line.trim();
        if line.is_empty() { continue; }

        let raw: JsonCapsule = serde_json::from_str(line)
            .with_context(|| format!("failed to parse {} line {line_num}", jsonl_path.display()))?;

        let id = raw.id.clone().unwrap_or_else(|| {
            format!("cap-{}-{}", raw.ts_ms, raw.conn_id.unwrap_or(0))
        });

        let cap = ExportCapsule {
            id,
            ts_ms: raw.ts_ms,
            agent_session_id: raw.agent_session_id.filter(|s| !s.is_empty()),
            head_sha: raw.head_sha,
            commit_sha: raw.commit_sha,
            source: raw.source,
            project: project.to_string(),
            capsule: crate::types::IntentCapsule {
                category: raw.capsule.category,
                intent: raw.capsule.intent,
                decision: raw.capsule.decision,
                rationale: raw.capsule.rationale,
                next_steps: raw.capsule.next_steps,
                symbols: raw.capsule.symbols,
                user_symbols: raw.capsule.user_symbols,
                failure_mode: parse_failure_mode(raw.capsule.failure_mode.as_deref()),
                failure_signals: raw.capsule.failure_signals,
                extraction_mode: crate::types::ExtractionMode::None,
                questions: raw.capsule.questions,
            },
        };
        capsules.push(cap);
    }

    Ok(capsules)
}

// ─── Category mapping ─────────────────────────────────────────────────────────

#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema, Debug)]
struct CategoryMapping {
    /// Map from raw category string to canonical taxonomy bucket.
    mapping: HashMap<String, String>,
}

async fn build_category_map(
    raw_categories: &[String],
    model_override: Option<&str>,
) -> HashMap<String, String> {
    if crate::llm::get_llm_config().is_none() {
        return fallback_map(raw_categories);
    }

    let bucket_list = TAXONOMY.iter()
        .map(|(name, desc)| format!("  - {name}: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");

    const CHUNK: usize = 200;
    let mut result: HashMap<String, String> = HashMap::new();

    for chunk in raw_categories.chunks(CHUNK) {
        let items = chunk.iter()
            .map(|c| format!("  - {c:?}"))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Map each raw category string to exactly one canonical bucket.\n\
             Return a JSON object with a `mapping` key.\n\
             Use only the bucket names listed — nothing else. Use `other` as fallback.\n\
             \nBuckets:\n{bucket_list}\n\nRaw categories:\n{items}"
        );

        match crate::llm::llm_extract::<CategoryMapping>(
            model_override,
            "You are a precise classifier. Respond only with the requested JSON.",
            &prompt,
        ).await {
            Ok(mapped) => {
                let valid: std::collections::HashSet<&str> =
                    TAXONOMY.iter().map(|(n, _)| *n).collect();
                for (raw, bucket) in mapped.mapping {
                    if valid.contains(bucket.as_str()) {
                        result.insert(raw, bucket);
                    } else {
                        result.insert(raw.clone(), map_category_fallback(&raw).to_string());
                    }
                }
            }
            Err(e) => {
                tracing::warn!("category mapping failed: {e}; using keyword fallback");
                for raw in chunk {
                    result.insert(raw.clone(), map_category_fallback(raw).to_string());
                }
            }
        }
    }

    for raw in raw_categories {
        result.entry(raw.clone())
            .or_insert_with(|| map_category_fallback(raw).to_string());
    }

    result
}

fn fallback_map(raw_categories: &[String]) -> HashMap<String, String> {
    raw_categories.iter()
        .map(|raw| (raw.clone(), map_category_fallback(raw).to_string()))
        .collect()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

async fn generate_narrative(model_override: Option<&str>, prompt: &str) -> anyhow::Result<String> {
    let result = crate::llm::llm_extract::<crate::types::QueryNarrativeOutput>(
        model_override,
        "You are a technical writer summarising software development decisions for a knowledge base. \
         Output only the requested narrative text in the `narrative` field.",
        prompt,
    ).await?;
    Ok(result.narrative)
}
