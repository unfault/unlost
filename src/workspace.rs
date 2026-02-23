use crate::config::{WorkspaceConfig, WorkspaceInfo};
use anyhow::Context;
use sha2::{Digest, Sha256};
use std::io::IsTerminal;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct WorkspacePaths {
    pub id: String,
    pub db_dir: std::path::PathBuf,
    pub capsules_jsonl: std::path::PathBuf,
    pub metrics_jsonl: std::path::PathBuf,
}

fn xdg_config_home() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        return std::path::PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".config");
    }
    std::path::PathBuf::from(".")
}

fn xdg_data_home() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        return std::path::PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".local").join("share");
    }
    std::path::PathBuf::from(".")
}

pub fn unlost_config_path() -> std::path::PathBuf {
    xdg_config_home().join("unlost").join("config.json")
}

pub fn unlost_data_root() -> std::path::PathBuf {
    xdg_data_home().join("unlost")
}

pub fn unlost_workspace_dir(workspace_id: &str) -> std::path::PathBuf {
    unlost_data_root().join("workspaces").join(workspace_id)
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

pub(crate) fn canonicalize_dir(path: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    std::fs::canonicalize(path).context("failed to canonicalize path")
}

pub(crate) fn git_toplevel(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        return None;
    }
    std::fs::canonicalize(s).ok()
}

fn normalize_git_remote(remote: &str) -> String {
    let mut remote = remote.trim().to_string();

    if remote.starts_with("git@") {
        remote = remote[4..].to_string();
        remote = remote.replacen(":", "/", 1);
    } else if remote.starts_with("ssh://") {
        remote = remote[6..].to_string();
        if remote.starts_with("git@") {
            remote = remote[4..].to_string();
        }
    } else if let Some(pos) = remote.find("://") {
        remote = remote[(pos + 3)..].to_string();
        if let Some(at_pos) = remote.find('@') {
            if at_pos < remote.find('/').unwrap_or(remote.len()) {
                remote = remote[(at_pos + 1)..].to_string();
            }
        }
    }

    if remote.ends_with(".git") {
        remote = remote[..remote.len() - 4].to_string();
    }

    remote = remote.trim_end_matches('/').to_string();
    remote.to_lowercase()
}

fn compute_hash16(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..8])
}

fn get_git_remote(workspace_root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if output.status.success() {
        let remote = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !remote.is_empty() {
            return Some(remote);
        }
    }
    None
}

fn read_meta_files(workspace_root: &std::path::Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let candidates = [
        ("pyproject", "pyproject.toml"),
        ("package_json", "package.json"),
        ("cargo_toml", "Cargo.toml"),
        ("go_mod", "go.mod"),
    ];
    for (kind, name) in candidates {
        let p = workspace_root.join(name);
        if let Ok(s) = std::fs::read_to_string(&p) {
            out.push((kind.to_string(), s));
        }
    }
    out
}

fn extract_project_name_from_meta_files(meta_files: &[(String, String)]) -> Option<String> {
    let re_pyproject_project =
        regex::Regex::new(r#"\[project\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_pyproject_poetry =
        regex::Regex::new(r#"\[tool\.poetry\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_cargo =
        regex::Regex::new(r#"\[package\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_go_mod = regex::Regex::new(r#"^module\s+(\S+)"#).ok();

    for (kind, contents) in meta_files {
        match kind.as_str() {
            "package_json" => {
                let json: serde_json::Value = serde_json::from_str(contents).ok()?;
                if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
                    return Some(name.to_string());
                }
            }
            "pyproject" => {
                if let Some(re) = &re_pyproject_project {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
                if let Some(re) = &re_pyproject_poetry {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
            }
            "cargo_toml" => {
                if let Some(re) = &re_cargo {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
            }
            "go_mod" => {
                if let Some(re) = &re_go_mod {
                    for line in contents.lines() {
                        if let Some(caps) = re.captures(line) {
                            return Some(caps.get(1)?.as_str().to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn compute_workspace_id(workspace_root: &std::path::Path) -> Option<(String, String)> {
    if let Some(remote) = get_git_remote(workspace_root) {
        let norm = normalize_git_remote(&remote);
        if !norm.is_empty() {
            return Some((
                format!("wks_{}", compute_hash16(&format!("git:{norm}"))),
                "git".to_string(),
            ));
        }
    }

    let meta = read_meta_files(workspace_root);
    if let Some(name) = extract_project_name_from_meta_files(&meta) {
        return Some((
            format!("wks_{}", compute_hash16(&format!("manifest:{name}"))),
            "manifest".to_string(),
        ));
    }

    let label = workspace_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace");
    Some((
        format!("wks_{}", compute_hash16(&format!("label:cli:{label}"))),
        "label".to_string(),
    ))
}

pub(crate) fn load_workspace_config() -> WorkspaceConfig {
    let p = unlost_config_path();
    if let Ok(s) = std::fs::read_to_string(&p) {
        if let Ok(cfg) = serde_json::from_str::<WorkspaceConfig>(&s) {
            return cfg;
        }
    }
    WorkspaceConfig {
        version: 1,
        path_index: Default::default(),
        workspaces: Default::default(),
        llm: None,
    }
}

pub(crate) fn save_workspace_config(cfg: &WorkspaceConfig) -> anyhow::Result<()> {
    let p = unlost_config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = serde_json::to_string_pretty(cfg)?;
    std::fs::write(&p, s)?;
    Ok(())
}

pub(crate) fn get_or_create_workspace_paths(
    workspace_root: &std::path::Path,
) -> anyhow::Result<WorkspacePaths> {
    let root = git_toplevel(workspace_root).unwrap_or_else(|| {
        canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf())
    });
    let root = canonicalize_dir(&root)?;
    let root_str = root.to_string_lossy().to_string();
    let mut cfg = load_workspace_config();

    if let Some(existing_id) = cfg.path_index.get(&root_str).cloned() {
        if let Some(info) = cfg.workspaces.get_mut(&existing_id) {
            info.updated_ts_ms = now_ms();
            let _ = save_workspace_config(&cfg);

            let ws_dir = unlost_workspace_dir(&existing_id);
            return Ok(WorkspacePaths {
                id: existing_id,
                db_dir: ws_dir.join("lancedb"),
                capsules_jsonl: ws_dir.join("capsules.jsonl"),
                metrics_jsonl: ws_dir.join("metrics.jsonl"),
            });
        }
    }

    let (id, source) = compute_workspace_id(&root)
        .ok_or_else(|| anyhow::anyhow!("unable to compute workspace id"))?;

    let ws_dir = unlost_workspace_dir(&id);
    let db_dir = ws_dir.join("lancedb");
    let capsules_jsonl = ws_dir.join("capsules.jsonl");

    let t = now_ms();
    cfg.path_index.insert(root_str.clone(), id.clone());
    cfg.workspaces.insert(
        id.clone(),
        WorkspaceInfo {
            id: id.clone(),
            root: root_str,
            source,
            db_dir: db_dir.to_string_lossy().to_string(),
            capsules_jsonl: capsules_jsonl.to_string_lossy().to_string(),
            created_ts_ms: t,
            updated_ts_ms: t,
        },
    );
    let _ = save_workspace_config(&cfg);

    Ok(WorkspacePaths {
        id,
        db_dir,
        capsules_jsonl,
        metrics_jsonl: ws_dir.join("metrics.jsonl"),
    })
}

pub(crate) fn clear_workspace(workspace_root: &std::path::Path, yes: bool) -> anyhow::Result<()> {
    let root = git_toplevel(workspace_root).unwrap_or_else(|| {
        canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf())
    });
    let root = canonicalize_dir(&root)?;
    let root_str = root.to_string_lossy().to_string();

    let mut cfg = load_workspace_config();

    let workspace_id = cfg
        .path_index
        .get(&root_str)
        .cloned()
        .or_else(|| compute_workspace_id(&root).map(|(id, _src)| id));

    let Some(workspace_id) = workspace_id else {
        println!("No workspace mapping found for: {root_str}");
        return Ok(());
    };

    let ws_dir = unlost_workspace_dir(&workspace_id);

    if !yes {
        if !std::io::stdin().is_terminal() {
            anyhow::bail!("refusing to clear without --yes in non-interactive mode");
        }

        let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let (warn_on, warn_off) = if use_color {
            ("\x1b[33;1m", "\x1b[0m")
        } else {
            ("", "")
        };
        let (dim_on, dim_off) = if use_color {
            ("\x1b[2m", "\x1b[0m")
        } else {
            ("", "")
        };

        println!("{warn_on}This will permanently delete unlost data{warn_off}");
        println!("workspace: {workspace_id}");
        println!("{dim_on}path:{dim_off} {root_str}");
        println!("{dim_on}data:{dim_off} {}", ws_dir.display());
        print!("{warn_on}Continue?{warn_off} [y/N]: ");
        std::io::stdout().flush().ok();

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        let ans = line.trim().to_ascii_lowercase();
        if ans != "y" && ans != "yes" {
            println!("aborted");
            return Ok(());
        }
    }

    if ws_dir.exists() {
        std::fs::remove_dir_all(&ws_dir)
            .with_context(|| format!("failed to delete {}", ws_dir.display()))?;
        println!("deleted: {}", ws_dir.display());
    } else {
        println!(
            "no data dir for workspace {workspace_id} (expected {})",
            ws_dir.display()
        );
    }

    // Remove config mappings pointing to this id.
    cfg.workspaces.remove(&workspace_id);
    cfg.path_index.retain(|_k, v| v != &workspace_id);
    save_workspace_config(&cfg)?;

    println!("cleared workspace: {workspace_id}");
    Ok(())
}

pub fn looks_like_rel_path(sym: &str) -> bool {
    // Mirror (loosely) the extraction heuristic in net.rs.
    sym.contains('/')
        || sym.starts_with("./")
        || sym.ends_with(".rs")
        || sym.ends_with(".ts")
        || sym.ends_with(".py")
        || sym.ends_with(".js")
        || sym.ends_with(".go")
}

pub fn normalize_rel_path(sym: &str) -> &str {
    sym.strip_prefix("./").unwrap_or(sym)
}

pub fn workspace_root_for_id(workspace_id: &str) -> Option<std::path::PathBuf> {
    let cfg = load_workspace_config();
    cfg.workspaces
        .get(workspace_id)
        .map(|w| std::path::PathBuf::from(w.root.clone()))
}

pub fn validate_paths(workspace_id: &str, symbols: &[String]) -> (usize, usize) {
    let Some(root) = workspace_root_for_id(workspace_id) else {
        return (0, 0);
    };
    let mut checked = 0usize;
    let mut missing = 0usize;
    for s in symbols {
        if !looks_like_rel_path(s) {
            continue;
        }
        checked += 1;
        let rel = normalize_rel_path(s);
        if rel.is_empty() {
            missing += 1;
            continue;
        }
        if !root.join(rel).exists() {
            missing += 1;
        }
    }
    (checked, missing)
}

/// Validates symbols against the actual codebase structure using unfault-core.
/// Returns (total_identifiers_checked, missing_identifiers).
pub fn validate_identifiers(workspace_id: &str, symbols: &[String]) -> (usize, usize) {
    let Some(root) = workspace_root_for_id(workspace_id) else {
        return (0, 0);
    };

    // For now, we do a live parse of the most recently modified files to keep it fast.
    let files = match collect_recent_source_files(&root) {
        Ok(f) => f,
        Err(_) => return (0, 0),
    };

    if files.is_empty() {
        return (0, 0);
    }

    let mut sem_entries = Vec::new();
    let mut next_id = 1u64;

    for sf in files {
        let file_id = unfault_core::FileId(next_id);
        next_id += 1;
        if let Ok(parsed) = unfault_core::parse::parse_source_file(file_id, &sf) {
            if let Ok(Some(sem)) = unfault_core::semantics::build_source_semantics(&parsed) {
                sem_entries.push((file_id, std::sync::Arc::new(sem)));
            }
        }
    }

    if sem_entries.is_empty() {
        return (0, 0);
    }

    let cg = unfault_core::build_code_graph(&sem_entries);
    let mut valid_identifiers = std::collections::HashSet::new();

    for node in cg.graph.node_weights() {
        match node {
            unfault_core::GraphNode::Function { name, .. } => {
                valid_identifiers.insert(name.clone());
            }
            unfault_core::GraphNode::Class { name, .. } => {
                valid_identifiers.insert(name.clone());
            }
            _ => {}
        }
    }

    let mut checked = 0;
    let mut missing = 0;

    for s in symbols {
        // Skip paths, we only care about plain identifiers here
        if looks_like_rel_path(s) {
            continue;
        }
        // Basic sanity check for identifiers (alphanumeric + underscore, at least 3 chars)
        if s.len() >= 3 && s.chars().all(|c| c.is_alphanumeric() || c == '_') {
            checked += 1;
            if !valid_identifiers.contains(s) {
                missing += 1;
            }
        }
    }

    (checked, missing)
}

fn collect_recent_source_files(
    root: &std::path::Path,
) -> anyhow::Result<Vec<unfault_core::SourceFile>> {
    use ignore::WalkBuilder;
    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .max_depth(Some(3)) // Shallow walk for speed
        .build();

    let mut out = Vec::new();
    for entry in walker {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let Some(lang) = detect_language(path) else {
            continue;
        };

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel = path.strip_prefix(root).unwrap_or(path);
        out.push(unfault_core::SourceFile {
            path: rel.to_string_lossy().to_string(),
            language: lang,
            content,
        });

        if out.len() > 50 {
            break;
        } // Limit to 50 files for "live" check speed
    }

    Ok(out)
}

fn detect_language(path: &std::path::Path) -> Option<unfault_core::Language> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "py" => Some(unfault_core::Language::Python),
        "rs" => Some(unfault_core::Language::Rust),
        "go" => Some(unfault_core::Language::Go),
        "ts" | "tsx" => Some(unfault_core::Language::Typescript),
        "js" | "jsx" => Some(unfault_core::Language::Javascript),
        "java" => Some(unfault_core::Language::Java),
        _ => None,
    }
}

/// Build a `CodeGraph` from the full workspace, with no depth or file count limit.
///
/// Returns `None` on any failure — callers should log a warning and proceed
/// without graph data rather than surfacing an error to the user.
pub fn build_graph_for_workspace(root: &std::path::Path) -> Option<unfault_core::CodeGraph> {
    use ignore::WalkBuilder;
    use std::sync::Arc;
    use unfault_core::{
        build_code_graph, parse::parse_source_file, semantics::build_source_semantics, FileId,
        SourceFile,
    };

    let walker = WalkBuilder::new(root).hidden(true).git_ignore(true).build();

    let mut sem_entries = Vec::new();
    let mut file_id_counter: u64 = 0;

    for entry in walker {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(lang) = detect_language(path) else {
            continue;
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let rel = path.strip_prefix(root).unwrap_or(path);
        let source_file = SourceFile {
            path: rel.to_string_lossy().to_string(),
            language: lang,
            content,
        };
        let file_id = FileId(file_id_counter);
        file_id_counter += 1;

        let parsed = match parse_source_file(file_id, &source_file) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let sem = match build_source_semantics(&parsed) {
            Ok(Some(s)) => s,
            _ => continue,
        };
        sem_entries.push((file_id, Arc::new(sem)));
    }

    if sem_entries.is_empty() {
        return None;
    }

    Some(build_code_graph(&sem_entries))
}

/// Rich graph context for use in LLM prompts that need to reason about code structure.
pub struct GraphContext {
    pub stats: unfault_core::GraphStats,
    /// (callers, symbol_path) ordered by callers descending — most-imported files first
    pub hotspots: Vec<(usize, String)>,
    /// Dependencies of the most central file
    pub deps: Vec<String>,
    /// Route patterns discovered in the codebase
    pub routes: Vec<(String, String)>,
    /// All source file paths (relative to root)
    pub file_paths: Vec<String>,
}

/// Build a `CodeGraph` plus derived structural context (hotspots, deps, routes, files).
///
/// Returns `None` on any failure. Callers should log a warning and proceed
/// without graph data rather than surfacing an error to the user.
pub fn build_graph_context_for_workspace(root: &std::path::Path) -> Option<GraphContext> {
    use ignore::WalkBuilder;
    use std::sync::Arc;
    use unfault_core::{
        build_code_graph,
        parse::parse_source_file,
        semantics::{build_source_semantics, CommonSemantics, SourceSemantics},
        FileId, SourceFile,
    };

    let walker = WalkBuilder::new(root).hidden(true).git_ignore(true).build();

    let mut sem_entries: Vec<(FileId, Arc<SourceSemantics>)> = Vec::new();
    let mut file_paths: Vec<String> = Vec::new();
    let mut file_id_counter: u64 = 0;

    for entry in walker {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(lang) = detect_language(path) else {
            continue;
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().to_string();
        file_paths.push(rel_str.clone());
        let source_file = SourceFile {
            path: rel_str,
            language: lang,
            content,
        };
        let file_id = FileId(file_id_counter);
        file_id_counter += 1;

        let parsed = match parse_source_file(file_id, &source_file) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let sem = match build_source_semantics(&parsed) {
            Ok(Some(s)) => s,
            _ => continue,
        };
        sem_entries.push((file_id, Arc::new(sem)));
    }

    if sem_entries.is_empty() {
        return None;
    }

    let cg = build_code_graph(&sem_entries);
    let stats = cg.stats();

    // Hotspots: most-imported files by centrality
    let centrality = unfault_core::graph::traversal::get_centrality(&cg, 25);
    let hotspots: Vec<(usize, String)> = centrality
        .central_files
        .iter()
        .map(|(path, score)| (*score as usize, path.clone()))
        .collect();

    // Dependencies of the top hub file
    let deps: Vec<String> = if let Some((top_path, _)) = centrality.central_files.first() {
        let dep_ctx = unfault_core::graph::traversal::get_dependencies(&cg, top_path);
        let mut all: Vec<String> = dep_ctx
            .dependencies
            .into_iter()
            .chain(dep_ctx.library_users)
            .collect();
        all.sort();
        all.dedup();
        all.truncate(25);
        all
    } else {
        Vec::new()
    };

    // Routes
    let mut routes: Vec<(String, String)> = Vec::new();
    for (_file_id, sem) in &sem_entries {
        let common: &dyn CommonSemantics = match sem.as_ref() {
            SourceSemantics::Python(s) => s,
            SourceSemantics::Go(s) => s,
            SourceSemantics::Rust(s) => s,
            SourceSemantics::Typescript(s) => s,
        };
        for route in common.route_patterns() {
            let method = route.method.as_str();
            let path = route.path.as_str();
            let handler = route.handler_name.as_deref().unwrap_or(&route.handler_file);
            routes.push((format!("{method} {path}"), handler.to_string()));
        }
    }
    routes.sort_by(|a, b| a.0.cmp(&b.0));
    routes.dedup_by(|a, b| a.0 == b.0);
    routes.truncate(25);

    file_paths.sort();

    Some(GraphContext {
        stats,
        hotspots,
        deps,
        routes,
        file_paths,
    })
}

/// Given a `CodeGraph` and a list of symbol strings (file paths, identifiers,
/// anything from a capsule's `symbols` field), return a compact one-line
/// summary of the import relationships for symbols that resolve to file nodes.
///
/// Format: `a.rs → [b.rs, c.rs] | b.rs ← [a.rs]`
///
/// Symbols that don't resolve (identifiers, markdown, unsupported languages)
/// are silently skipped. Returns `None` if nothing resolved.
pub fn relationships_for_symbols(
    cg: &unfault_core::CodeGraph,
    symbols: &[String],
) -> Option<String> {
    use unfault_core::graph::traversal::{get_dependencies, get_impact};

    let mut parts: Vec<String> = Vec::new();

    for sym in symbols {
        let Some(_idx) = cg.find_file_by_path(sym) else {
            continue;
        };

        let deps = get_dependencies(cg, sym);
        let impact = get_impact(cg, sym, 1); // depth 1: direct dependents only

        let has_deps = !deps.dependencies.is_empty() || !deps.library_users.is_empty();
        let has_impact = !impact.affected_files.is_empty();

        if !has_deps && !has_impact {
            continue;
        }

        // Use just the filename for readability, not the full path.
        let label = std::path::Path::new(sym)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| sym.clone());

        let mut entry = label;

        if has_deps {
            let names: Vec<String> = deps
                .dependencies
                .iter()
                .chain(deps.library_users.iter())
                .take(4)
                .map(|d| {
                    std::path::Path::new(d)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| d.clone())
                })
                .collect();
            entry.push_str(&format!(" → [{}]", names.join(", ")));
        }

        if has_impact {
            let names: Vec<String> = impact
                .affected_files
                .iter()
                .take(4)
                .map(|d| {
                    std::path::Path::new(d)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| d.clone())
                })
                .collect();
            entry.push_str(&format!(" ← [{}]", names.join(", ")));
        }

        parts.push(entry);
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

/// Build a compact semantic grounding string for a set of touched file paths.
///
/// For each path that can be parsed with unfault-core, extracts function names,
/// async flags, and route patterns. Returns a single string suitable for
/// appending to a capsule's `grounding_note`, or `None` if no files yielded
/// useful data.
///
/// Designed to be called synchronously at replay time, once per turn. Files
/// that don't exist on disk or whose language is not supported are silently
/// skipped. Capped at 10 files to bound latency.
pub fn semantic_grounding_for_paths(
    workspace_root: &std::path::Path,
    touched_paths: &[String],
) -> Option<String> {
    use unfault_core::{
        parse::parse_source_file,
        semantics::{build_source_semantics, CommonSemantics, SourceSemantics},
        FileId, SourceFile,
    };

    let mut parts: Vec<String> = Vec::new();

    for (i, rel_path) in touched_paths.iter().take(10).enumerate() {
        let abs = workspace_root.join(rel_path);
        let lang = match detect_language(&abs) {
            Some(l) => l,
            None => continue,
        };
        let content = match std::fs::read_to_string(&abs) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_id = FileId(i as u64);
        let source_file = SourceFile {
            path: rel_path.clone(),
            language: lang,
            content,
        };

        let parsed = match parse_source_file(file_id, &source_file) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let sem = match build_source_semantics(&parsed) {
            Ok(Some(s)) => s,
            _ => continue,
        };

        let common: &dyn CommonSemantics = match &sem {
            SourceSemantics::Python(s) => s,
            SourceSemantics::Go(s) => s,
            SourceSemantics::Rust(s) => s,
            SourceSemantics::Typescript(s) => s,
        };

        let fns = common.functions();
        let routes = common.route_patterns();

        if fns.is_empty() && routes.is_empty() {
            continue;
        }

        let mut file_parts: Vec<String> = Vec::new();

        if !fns.is_empty() {
            let fn_descs: Vec<String> = fns
                .iter()
                .take(8)
                .map(|f| {
                    if f.is_async {
                        format!("async {}", f.name)
                    } else {
                        f.name.clone()
                    }
                })
                .collect();
            file_parts.push(format!("fns=[{}]", fn_descs.join(", ")));
        }

        if !routes.is_empty() {
            let route_descs: Vec<String> = routes
                .iter()
                .take(4)
                .map(|r| format!("{} {}", r.method.as_str(), r.path.as_str()))
                .collect();
            file_parts.push(format!("routes=[{}]", route_descs.join(", ")));
        }

        if !file_parts.is_empty() {
            parts.push(format!("{}: {}", rel_path, file_parts.join("; ")));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(format!("Semantics: {}", parts.join(" | ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::test_support::ENV_LOCK;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, val: &std::ffi::OsStr) -> Self {
            let prev = std::env::var_os(key);
            // Rust 2024: modifying process env is `unsafe` (can race with other threads).
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn test_xdg_config_home() {
        let _lock = ENV_LOCK.lock().unwrap();

        // Test with XDG_CONFIG_HOME set
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());
        assert_eq!(xdg_config_home(), temp_dir.path());

        // Test fallback to HOME/.config
        let home_dir = TempDir::new().unwrap();
        let _h = EnvVarGuard::set("HOME", home_dir.path().as_os_str());
        let _r = EnvVarGuard::remove("XDG_CONFIG_HOME");
        assert_eq!(xdg_config_home(), home_dir.path().join(".config"));
    }

    #[test]
    fn test_xdg_data_home() {
        let _lock = ENV_LOCK.lock().unwrap();

        // Test with XDG_DATA_HOME set
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_DATA_HOME", temp_dir.path().as_os_str());
        assert_eq!(xdg_data_home(), temp_dir.path());

        // Test fallback to HOME/.local/share
        let home_dir = TempDir::new().unwrap();
        let _h = EnvVarGuard::set("HOME", home_dir.path().as_os_str());
        let _r = EnvVarGuard::remove("XDG_DATA_HOME");
        assert_eq!(
            xdg_data_home(),
            home_dir.path().join(".local").join("share")
        );
    }

    #[test]
    fn test_now_ms() {
        let t1 = now_ms();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t2 = now_ms();
        assert!(t2 >= t1);
    }

    #[test]
    fn test_normalize_git_remote() {
        assert_eq!(
            normalize_git_remote("git@github.com:user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("ssh://git@github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://user@github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://github.com/user/repo"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_compute_hash16() {
        let hash1 = compute_hash16("test");
        let hash2 = compute_hash16("test");
        let hash3 = compute_hash16("different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_extract_project_name_from_meta_files() {
        let package_json = r#"
        {
            "name": "my-project",
            "version": "1.0.0"
        }
        "#;
        let meta = vec![("package_json".to_string(), package_json.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-project".to_string())
        );

        let pyproject = r#"
        [project]
        name = "my-py-project"
        version = "0.1.0"
        "#;
        let meta = vec![("pyproject".to_string(), pyproject.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-py-project".to_string())
        );

        let cargo_toml = r#"
        [package]
        name = "my-rust-project"
        version = "0.1.0"
        "#;
        let meta = vec![("cargo_toml".to_string(), cargo_toml.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-rust-project".to_string())
        );

        let go_mod = r#"
module github.com/user/my-go-project
        "#;
        let meta = vec![("go_mod".to_string(), go_mod.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("github.com/user/my-go-project".to_string())
        );
    }

    #[test]
    fn test_compute_workspace_id() {
        // Test with a mock directory that doesn't exist
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path();

        let (id, source) = compute_workspace_id(workspace_root).unwrap();
        assert!(id.starts_with("wks_"));
        assert_eq!(source, "label");

        // Test that the same path produces the same ID
        let (id2, source2) = compute_workspace_id(workspace_root).unwrap();
        assert_eq!(id, id2);
        assert_eq!(source, source2);
    }

    #[test]
    fn test_load_and_save_workspace_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        // Load non-existent config
        let config = load_workspace_config();
        assert_eq!(config.version, 1);
        assert!(config.path_index.is_empty());
        assert!(config.workspaces.is_empty());
        assert!(config.llm.is_none());

        // Save and load config
        let mut new_config = config.clone();
        new_config.version = 2;
        new_config
            .path_index
            .insert("/test/path".to_string(), "test_id".to_string());
        save_workspace_config(&new_config).unwrap();

        let loaded = load_workspace_config();
        assert_eq!(loaded.version, 2);
        assert!(loaded.path_index.contains_key("/test/path"));
    }

    #[test]
    fn test_unlost_paths() {
        let _lock = ENV_LOCK.lock().unwrap();

        let temp_dir = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());
        let _g2 = EnvVarGuard::set("XDG_DATA_HOME", temp_dir.path().as_os_str());

        let config_path = unlost_config_path();
        assert!(config_path.starts_with(temp_dir.path()));
        assert!(config_path.ends_with("unlost/config.json"));

        let data_root = unlost_data_root();
        assert!(data_root.starts_with(temp_dir.path()));
        assert!(data_root.ends_with("unlost"));

        let workspace_dir = unlost_workspace_dir("test_id");
        assert!(workspace_dir.starts_with(&data_root));
        assert!(workspace_dir.ends_with("test_id"));
    }
}
