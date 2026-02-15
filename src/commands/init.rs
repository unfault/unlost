use anyhow::Context;
use ignore::WalkBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;
use unfault_core::{
    FileId, Language as UfLanguage, SourceFile, build_code_graph,
    parse::parse_source_file,
    semantics::{SourceSemantics, build_source_semantics},
};

fn detect_language(path: &std::path::Path) -> Option<UfLanguage> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "py" => Some(UfLanguage::Python),
        "rs" => Some(UfLanguage::Rust),
        "go" => Some(UfLanguage::Go),
        "ts" | "tsx" => Some(UfLanguage::Typescript),
        "js" | "jsx" => Some(UfLanguage::Javascript),
        "java" => Some(UfLanguage::Java),
        _ => None,
    }
}

fn should_skip_directory(name: &str) -> bool {
    name == "node_modules"
        || name == "__pycache__"
        || name == "target"
        || name == "venv"
        || name == "env"
        || name == ".venv"
        || name == ".env"
        || name == "dist"
        || name == "build"
        || name == "vendor"
        || name == "site-packages"
        || name == ".git"
}

fn collect_source_files(root: &std::path::Path) -> anyhow::Result<Vec<SourceFile>> {
    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .require_git(false)
        .add_custom_ignore_filename(".gitignore")
        .add_custom_ignore_filename(".dockerignore")
        .filter_entry(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|n| !should_skip_directory(n))
                .unwrap_or(true)
        })
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

        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.len() > 256 * 1024 {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel = rel.to_string_lossy().replace('\\', "/");
        out.push(SourceFile {
            path: rel,
            language: lang,
            content,
        });
    }

    Ok(out)
}

fn collect_git_history_summary(
    repo_root: &std::path::Path,
    scope_rel: Option<&str>,
    max_commits: usize,
) -> anyhow::Result<Option<String>> {
    if max_commits == 0 {
        return Ok(None);
    }
    if crate::workspace::git_toplevel(repo_root).is_none() {
        return Ok(None);
    }

    let n = max_commits.min(50);
    let n_str = n.to_string();
    let mut log_args: Vec<String> = vec![
        "log".to_string(),
        "-n".to_string(),
        n_str,
        "--pretty=format:%H%x1f%ct%x1f%s%x1e".to_string(),
    ];
    if let Some(scope) = scope_rel {
        if !scope.trim().is_empty() {
            log_args.push("--".to_string());
            log_args.push(scope.to_string());
        }
    }

    let output = std::process::Command::new("git")
        .current_dir(repo_root)
        .args(&log_args)
        .output()
        .context("failed to run git log")?;
    if !output.status.success() {
        return Ok(None);
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let mut out = String::new();
    out.push_str(&format!("commits: {n}\n"));

    let mut total = 0usize;
    for rec in raw.split('\x1e') {
        let rec = rec.trim();
        if rec.is_empty() {
            continue;
        }
        let mut parts = rec.split('\x1f');
        let hash = parts.next().unwrap_or("").trim();
        let ts = parts.next().unwrap_or("").trim();
        let subj = parts.next().unwrap_or("").trim();
        if hash.is_empty() {
            continue;
        }

        let files = std::process::Command::new("git")
            .current_dir(repo_root)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", hash])
            .args(
                scope_rel
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| vec!["--", s])
                    .unwrap_or_default(),
            )
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

        let mut files = files;
        if files.len() > 10 {
            files.truncate(10);
        }

        let short = if hash.len() >= 7 { &hash[..7] } else { hash };
        let line = if files.is_empty() {
            format!("- {short} ts={ts} {subj}\n")
        } else {
            format!("- {short} ts={ts} {subj} [files: {}]\n", files.join(", "))
        };
        total += line.len();
        if total > 20_000 {
            out.push_str("- ... (truncated)\n");
            break;
        }
        out.push_str(&line);
    }

    if out.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

async fn llm_init_capsules(
    llm_model_override: Option<&str>,
    llm_max_capsules: usize,
    workspace_root: &std::path::Path,
    stats: &unfault_core::GraphStats,
    routes: &[(String, String)],
    hotspots: &[(usize, String)],
    deps: &[(usize, String)],
    file_paths: &[String],
    git_history: Option<&str>,
) -> anyhow::Result<(String, Vec<crate::IntentCapsule>)> {
    let mut context = String::new();
    context.push_str("Workspace init context\n\n");
    context.push_str(&format!("root: {}\n", workspace_root.to_string_lossy()));
    context.push_str(&format!(
        "graph: files={}, functions={}, call_edges={}, import_edges={}, external_modules={}\n\n",
        stats.file_count,
        stats.function_count,
        stats.calls_edge_count,
        stats.import_edge_count,
        stats.external_module_count,
    ));

    if !routes.is_empty() {
        context.push_str("routes:\n");
        for (route, handler) in routes.iter().take(25) {
            context.push_str(&format!("- {route} -> {handler}\n"));
        }
        context.push('\n');
    }
    if !hotspots.is_empty() {
        context.push_str("hotspots:\n");
        for (callers, sym) in hotspots.iter().take(25) {
            context.push_str(&format!("- {sym} (callers={callers})\n"));
        }
        context.push('\n');
    }
    if !deps.is_empty() {
        context.push_str("dependencies:\n");
        for (uses, dep) in deps.iter().take(25) {
            context.push_str(&format!("- {dep} (uses={uses})\n"));
        }
        context.push('\n');
    }
    if !file_paths.is_empty() {
        context.push_str("files (sample):\n");
        for p in file_paths.iter().take(120) {
            context.push_str(&format!("- {p}\n"));
        }
        context.push('\n');
    }

    if let Some(gh) = git_history {
        if !gh.trim().is_empty() {
            context.push_str("recent git history (bounded):\n");
            context.push_str(gh.trim());
            context.push_str("\n\n");
        }
    }

    let preamble = "You are unlost init. Write a compact, colleague-like baseline understanding of this codebase.\n\
 Return JSON only.\n\
 Provide a `debrief` field: 6-10 sentences, conversational, no bullets, no headings.\n\
 Then provide `capsules`: up to the requested max. Each capsule must be short and high-signal.\n\
 Each capsule schema: {category, intent, decision, rationale, next_steps (array), symbols (array)}.\n\
 Use categories like: Snapshot:Project, Snapshot:Architecture, Snapshot:DataModel, Snapshot:Runtime, Snapshot:Risks, Snapshot:NextSteps.\n\
 If git history is provided, add a couple of Snapshot:History capsules about recent evolution and intent (no code excerpts).\n\
 Do not include long excerpts; do not include any tool/system boilerplate.\n\
 Populate symbols with real identifiers, file paths, and endpoints when available.";

    let crate::InitCapsulesOutput {
        debrief,
        mut capsules,
    } = crate::llm_extract::<crate::InitCapsulesOutput>(
        llm_model_override,
        preamble,
        &format!("max_capsules: {llm_max_capsules}\n\n{context}"),
    )
    .await?;

    if capsules.len() > llm_max_capsules {
        capsules.truncate(llm_max_capsules);
    }

    Ok((debrief, capsules))
}

pub async fn run(
    path: String,
    embed_model: String,
    embed_cache_dir: Option<String>,
    max_capsules: usize,
    no_llm: bool,
    git_history: bool,
    git_commits: usize,
    git_path: Option<String>,
    llm_model: Option<String>,
    llm_max_capsules: usize,
) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;

    let root_path = crate::workspace::canonicalize_dir(std::path::Path::new(&path))
        .unwrap_or_else(|_| std::path::PathBuf::from(&path));
    let files = collect_source_files(&root_path)?;
    if files.is_empty() {
        anyhow::bail!("no supported source files found under {path}");
    }

    let file_paths = files.iter().map(|sf| sf.path.clone()).collect::<Vec<_>>();

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        true,
    )
    .await?;

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let mut sem_entries: Vec<(FileId, Arc<SourceSemantics>)> = Vec::new();
    let mut next_id: u64 = 1;
    for sf in files {
        let file_id = FileId(next_id);
        next_id += 1;
        let parsed = match parse_source_file(file_id, &sf) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let Some(sem) = build_source_semantics(&parsed)? else {
            continue;
        };
        sem_entries.push((file_id, Arc::new(sem)));
    }

    if sem_entries.is_empty() {
        anyhow::bail!("no parsable semantics produced (supported: py/rs/go/ts)");
    }

    let cg = build_code_graph(&sem_entries);
    let stats = cg.stats();
    info!(?stats, "built unfault code graph");

    let mut lang_counts: HashMap<&'static str, usize> = HashMap::new();
    for node in cg.graph.node_weights() {
        if let unfault_core::GraphNode::File { language, .. } = node {
            let k = match language {
                UfLanguage::Python => "python",
                UfLanguage::Rust => "rust",
                UfLanguage::Go => "go",
                UfLanguage::Typescript => "typescript",
                UfLanguage::Javascript => "javascript",
                UfLanguage::Java => "java",
            };
            *lang_counts.entry(k).or_insert(0) += 1;
        }
    }

    let mut top_routes: Vec<(String, String)> = Vec::new();
    for (idx, path, method) in cg.get_http_route_handlers() {
        let node = &cg.graph[idx];
        let qualified = match node {
            unfault_core::GraphNode::Function { qualified_name, .. } => qualified_name.clone(),
            _ => continue,
        };
        let m = method.unwrap_or("ANY");
        top_routes.push((format!("{m} {path}"), qualified));
    }
    top_routes.sort_by(|a, b| a.0.cmp(&b.0));
    if top_routes.len() > 5 {
        top_routes.truncate(5);
    }

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let git_history_summary = if git_history {
        let repo_root = crate::workspace::git_toplevel(&root_path);
        if let Some(repo_root) = repo_root {
            let scope_abs = if let Some(p) = git_path.as_deref() {
                let p = std::path::Path::new(p);
                if p.is_absolute() {
                    crate::workspace::canonicalize_dir(p).unwrap_or_else(|_| p.to_path_buf())
                } else {
                    repo_root.join(p)
                }
            } else {
                root_path.to_path_buf()
            };

            let scope_rel = scope_abs
                .strip_prefix(&repo_root)
                .ok()
                .and_then(|p| p.to_str())
                .map(|s| s.trim_matches('/'))
                .filter(|s| !s.is_empty());

            collect_git_history_summary(&repo_root, scope_rel, git_commits)
                .ok()
                .flatten()
        } else {
            None
        }
    } else {
        None
    };

    let mut capsules: Vec<(crate::IntentCapsule, crate::ResponseMeta)> = Vec::new();

    capsules.push((
        crate::IntentCapsule {
            category: "Snapshot:Project".to_string(),
            intent: "Establish a baseline picture of the codebase".to_string(),
            decision: format!(
                "Project snapshot: {} files, {} functions, {} classes, {} external modules. Import edges: {}, call edges: {}.",
                stats.file_count,
                stats.function_count,
                stats.class_count,
                stats.external_module_count,
                stats.import_edge_count,
                stats.calls_edge_count,
            ),
            rationale: String::new(),
            next_steps: Vec::new(),
            symbols: vec![crate::storage::CAPSULES_TABLE.to_string()],
            user_symbols: vec![],
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        },
        crate::ResponseMeta {
            source: "init".to_string(),
            upstream_host: "init".to_string(),
            request_path: "init".to_string(),
            http_status: 0,
            agent_session_id: None,
            usage: None,
        },
    ));

    for (idx, path, method) in cg.get_http_route_handlers() {
        if capsules.len() >= max_capsules {
            break;
        }

        let node = &cg.graph[idx];
        let (qualified, file_id) = match node {
            unfault_core::GraphNode::Function {
                qualified_name,
                file_id,
                ..
            } => (qualified_name.clone(), *file_id),
            _ => continue,
        };

        let method = method.unwrap_or("ANY");
        let file_path = cg
            .file_nodes
            .get(&file_id)
            .and_then(|fidx| match &cg.graph[*fidx] {
                unfault_core::GraphNode::File { path, .. } => Some(path.clone()),
                _ => None,
            })
            .unwrap_or_default();

        capsules.push((
            crate::IntentCapsule {
                category: "Snapshot:Route".to_string(),
                intent: "Identify the request surface area".to_string(),
                decision: format!(
                    "HTTP handler {method} {path} implemented by {qualified} ({file_path})."
                ),
                rationale: String::new(),
                next_steps: Vec::new(),
                symbols: vec![
                    qualified.clone(),
                    file_path.clone(),
                    format!("{method} {path}"),
                ],
                user_symbols: vec![],
                failure_mode: crate::types::FailureMode::None,
                failure_signals: None,
            },
            crate::ResponseMeta {
                source: "init".to_string(),
                upstream_host: "init".to_string(),
                request_path: file_path,
                http_status: 0,
                agent_session_id: None,
                usage: None,
            },
        ));
    }

    if !no_llm {
        if let Ok((debrief, more)) = llm_init_capsules(
            llm_model.as_deref(),
            llm_max_capsules,
            &root_path,
            &stats,
            &top_routes,
            &Vec::new(),
            &Vec::new(),
            &file_paths,
            git_history_summary.as_deref(),
        )
        .await
        {
            println!("{debrief}\n");
            for c in more {
                capsules.push((
                    c,
                    crate::ResponseMeta {
                        source: "init".to_string(),
                        upstream_host: "init".to_string(),
                        request_path: "init".to_string(),
                        http_status: 0,
                        agent_session_id: None,
                        usage: None,
                    },
                ));
                if capsules.len() >= max_capsules {
                    break;
                }
            }
        }
    }

    for (cap, meta) in capsules {
        crate::storage::insert_capsule_row(&db, &embedder, 0, 0, now_ms, &meta, None, None, &cap, None)
            .await
            .ok();
    }

    println!("workspace: {}", ws.id);
    if !top_routes.is_empty() {
        println!("routes:");
        for (r, h) in top_routes {
            println!("- {r} -> {h}");
        }
    }
    if !lang_counts.is_empty() {
        let mut langs = lang_counts.into_iter().collect::<Vec<_>>();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        let s = langs
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("languages: {s}");
    }

    println!("\ntry:");
    println!("- unlost query \"what is the auth flow?\"");
    println!("- unlost query \"why does this exist?\" --symbol src/main.rs");
    println!("- unlost query \"where is proxying implemented?\" --symbol proxy_request");
    Ok(())
}
