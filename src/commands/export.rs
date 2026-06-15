use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::export::{
    capsule_filename, capsule_to_markdown, category_narrative_prompt, category_readme_static,
    category_readme_with_narrative, decisions_md, index_md, is_substantive,
    is_worth_categorising, map_category_fallback, project_name_from_root, symbol_page,
    symbol_page_filename, ts_ms_to_iso, CategoryStat, ExportCapsule, ProjectStat, TAXONOMY,
};

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
const MIN_CAPSULES_FOR_SYMBOL_PAGE: usize = 3;

/// Skip symbol pages for these patterns — they're infrastructure, not knowledge.
const SYMBOL_PAGE_SKIPLIST: &[&str] = &[
    "Cargo.lock", "package-lock.json", "yarn.lock",
    "go.sum", "poetry.lock", ".gitignore", ".env",
];

/// Returns true if a symbol string is worth a dedicated knowledge page.
///
/// Accepted:
/// - File paths: contain `/` or have a recognised source-file extension
/// - Qualified identifiers: `Module::function`, `package.Class`
///
/// Rejected:
/// - Plain English words / phrases
/// - Single tokens without extension that look like prose
/// - Very short strings
/// - Environment variable assignments (`KEY=value`)
fn should_skip_symbol(sym: &str) -> bool {
    // Explicit skiplist (lock files etc.)
    let basename = sym.rsplit('/').next().unwrap_or(sym);
    if SYMBOL_PAGE_SKIPLIST.contains(&basename) {
        return true;
    }

    // Too short to be meaningful
    if sym.len() < 4 {
        return true;
    }

    // Contains spaces → prose, not a symbol
    if sym.contains(' ') {
        return true;
    }

    // Starts with non-alphanumeric that signals non-path content
    // (CLI flags, Rust attributes, CSS tokens, glob patterns, env vars, quoted strings)
    let first = sym.chars().next().unwrap_or(' ');
    if matches!(first, '-' | '#' | '$' | '*' | '%' | '"' | '\'' | '`' | '@' | '!') {
        return true;
    }

    // Contains `=` → env var assignment, not a symbol
    if sym.contains('=') {
        return true;
    }

    // Rust attribute syntax
    if sym.starts_with("#[") {
        return true;
    }

    // Looks like a file path (has slash) → keep
    if sym.contains('/') {
        return false;
    }

    // Has a recognised source-file extension → keep
    const SOURCE_EXTS: &[&str] = &[
        ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".kt",
        ".swift", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".cs", ".toml",
        ".json", ".yaml", ".yml", ".md", ".html", ".css", ".sh", ".sql",
    ];
    if SOURCE_EXTS.iter().any(|ext| sym.ends_with(ext)) {
        return false;
    }

    // Qualified identifier with `::` or `.` separator (e.g. `TrajectoryController`, `Module::fn`) → keep
    if sym.contains("::") || (sym.contains('.') && !sym.starts_with('.')) {
        return false;
    }

    // CamelCase identifier (at least one uppercase after a lowercase) → keep
    let chars: Vec<char> = sym.chars().collect();
    let has_camel = chars.windows(2).any(|w| w[0].is_lowercase() && w[1].is_uppercase());
    if has_camel {
        return false;
    }

    // snake_case identifier (has underscore, all word chars) → keep if ≥8 chars
    if sym.contains('_') && sym.len() >= 8
        && sym.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        return false;
    }

    // Everything else: plain word, number, abbrev → skip
    true
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

    let total_raw: usize = workspace_data.iter().map(|(_, caps)| caps.len()).sum();
    println!("Loaded {} capsules from {} workspace{}",
        total_raw,
        workspace_data.len(),
        if workspace_data.len() == 1 { "" } else { "s" });

    // 3. Map raw categories → taxonomy buckets (single batch per workspace)
    //    Build the full flat list with project name and bucket assigned.
    let mut all_capsules: Vec<ExportCapsule> = Vec::with_capacity(total_raw);
    for (project, caps) in &workspace_data {
        let raw_cats: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            caps.iter().map(|c| c.capsule.category.clone())
                .filter(|c| seen.insert(c.clone()))
                .collect()
        };
        let cat_map = build_category_map(&raw_cats, llm_model.as_deref()).await;

        for mut cap in caps.clone() {
            let bucket = cat_map.get(&cap.capsule.category)
                .cloned()
                .unwrap_or_else(|| map_category_fallback(&cap.capsule.category).to_string());
            cap.project = project.clone();
            // Store resolved bucket in the capsule category field for downstream use
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
    let mut by_project_symbol: BTreeMap<String, BTreeMap<String, Vec<&ExportCapsule>>> =
        BTreeMap::new();
    for cap in &substantive {
        for (pos, sym) in cap.capsule.symbols.iter().enumerate() {
            if pos >= MAX_SYMBOL_POSITION { break; }
            if should_skip_symbol(sym) { continue; }
            by_project_symbol
                .entry(cap.project.clone())
                .or_default()
                .entry(sym.clone())
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

    // 8. Write category folders
    //    Only include capsules that pass `is_worth_categorising`
    let mut by_category: BTreeMap<String, Vec<&ExportCapsule>> = BTreeMap::new();
    for cap in &substantive {
        let bucket = &cap.capsule.category;
        if is_worth_categorising(cap, bucket) {
            by_category.entry(bucket.clone()).or_default().push(cap);
        }
    }

    let mut cat_written = 0usize;
    let mut cat_stats: Vec<CategoryStat> = Vec::new();

    for (bucket, cats) in &by_category {
        let cat_dir = categories_dir.join(bucket);
        std::fs::create_dir_all(&cat_dir)?;

        // Deduplicate filenames
        let mut filenames: Vec<String> = Vec::with_capacity(cats.len());
        let mut seen: HashMap<String, usize> = HashMap::new();
        for cap in cats.iter() {
            let raw = capsule_filename(cap);
            let count = seen.entry(raw.clone()).or_insert(0);
            let fname = if *count == 0 {
                raw.clone()
            } else {
                let stem = raw.trim_end_matches(".md");
                format!("{stem}-{}.md", *count + 1)
            };
            *count += 1;
            filenames.push(fname);
        }

        for (cap, filename) in cats.iter().zip(filenames.iter()) {
            let file_path = cat_dir.join(filename);
            if file_path.exists() && !force { continue; }
            let content = capsule_to_markdown(cap, bucket);
            std::fs::write(&file_path, content)?;
            cat_written += 1;
        }

        // README
        let pairs: Vec<(&ExportCapsule, &str)> = cats.iter()
            .copied()
            .zip(filenames.iter().map(|s| s.as_str()))
            .collect();

        let readme = if narrative {
            let prompt = category_narrative_prompt(bucket, &cats.iter().map(|c| *c).collect::<Vec<_>>());
            match generate_narrative(llm_model.as_deref(), &prompt).await {
                Ok(narr) => category_readme_with_narrative(bucket, &pairs, &narr),
                Err(e) => {
                    eprintln!("warning: narrative for '{bucket}' failed: {e}");
                    category_readme_static(bucket, &pairs)
                }
            }
        } else {
            category_readme_static(bucket, &pairs)
        };

        std::fs::write(cat_dir.join("README.md"), readme)?;
        cat_stats.push(CategoryStat { category: bucket.clone(), count: cats.len() });
    }

    // Fill in zero-count categories so INDEX.md is complete
    let all_buckets: Vec<&str> = TAXONOMY.iter().map(|(n, _)| *n).collect();
    for bucket in &all_buckets {
        if !cat_stats.iter().any(|s| s.category == *bucket) {
            cat_stats.push(CategoryStat { category: bucket.to_string(), count: 0 });
        }
    }
    cat_stats.sort_by(|a, b| b.count.cmp(&a.count));

    println!("  categories/ → {} files across {} categories",
        cat_written,
        by_category.len());

    // 9. Write INDEX.md
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0);
    let exported_at = ts_ms_to_iso(now_ms);

    let failure_count = substantive.iter()
        .filter(|c| c.capsule.failure_mode != crate::types::FailureMode::None)
        .count();

    let proj_stats: Vec<ProjectStat> = workspace_data.iter().map(|(proj, caps)| {
        let sub = caps.iter()
            .filter(|c| is_substantive(c))
            .count();
        let sym_pages = project_symbol_counts.get(proj).copied().unwrap_or(0);
        ProjectStat {
            project: proj.clone(),
            total_capsules: caps.len(),
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

/// Load capsules from all registered workspaces.
/// Returns a list of (project_name, capsules) pairs.
async fn collect_all_workspaces(
    current_path: &str,
) -> anyhow::Result<Vec<(String, Vec<ExportCapsule>)>> {
    let cfg = crate::workspace::load_workspace_config();
    let mut result: Vec<(String, Vec<ExportCapsule>)> = Vec::new();

    if cfg.workspaces.is_empty() {
        // Fallback: just the current workspace
        let ws = crate::workspace::get_or_create_workspace_paths(Path::new(current_path))?;
        let project = project_name_from_root(&ws.root.to_string_lossy());
        if ws.capsules_jsonl.exists() {
            let caps = load_capsules(&ws.capsules_jsonl, &project).await?;
            result.push((project, caps));
        }
        return Ok(result);
    }

    for (_, info) in &cfg.workspaces {
        let jsonl = PathBuf::from(&info.capsules_jsonl);
        if !jsonl.exists() { continue; }

        let project = project_name_from_root(&info.root);
        match load_capsules(&jsonl, &project).await {
            Ok(caps) if !caps.is_empty() => result.push((project, caps)),
            Ok(_) => {}
            Err(e) => {
                eprintln!("warning: skipping workspace {} ({}): {e}", info.id, info.root);
            }
        }
    }

    // Sort by project name for deterministic output
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
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
