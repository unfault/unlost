use anyhow::Context;
use arrow_array::{
    Array,
    ListArray,
    FixedSizeListArray,
    Int32Array,
    Int64Array,
    RecordBatch,
    RecordBatchIterator,
    StringArray,
    builder::{ListBuilder, StringBuilder},
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use clap::{CommandFactory, Parser, Subcommand};
use clap::ValueEnum;
use futures_util::TryStreamExt;
use http_body_util::{
    BodyExt,
    BodyStream,
    StreamBody,
    combinators::BoxBody,
    Full,
};
use hyper::{
    Request,
    Response,
    StatusCode,
    Uri,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use hyper::server::conn::http1;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::{
    client::legacy::Client,
    rt::{TokioExecutor, TokioIo},
};
use kanal::AsyncReceiver;
use lancedb::connection::Connection;
use lancedb::index::{Index, scalar::LabelListIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use rig::providers::{openai, anthropic};
use rig::client::ProviderClient;
use rig::client::CompletionClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::io::Write;
use std::io::IsTerminal;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::Duration;
use tokio::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;
use std::collections::HashMap;
use petgraph::visit::EdgeRef;
use sha2::{Digest, Sha256};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};

use fastembed::{EmbeddingModel, InitOptions as FastEmbedInitOptions, TextEmbedding};
use unfault_core::{
    FileId,
    SourceFile,
    Language as UfLanguage,
    build_code_graph,
    parse::parse_source_file,
    semantics::{SourceSemantics, build_source_semantics},
};


// fastembed uses "model_code" strings (e.g. Xenova/*, Qdrant/*) for FromStr.
// For BGE small, fastembed's model_code is Xenova/bge-small-en-v1.5.
const DEFAULT_EMBED_MODEL: &str = "Xenova/bge-small-en-v1.5";
const DEFAULT_EMBED_DIM: usize = 384;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkspaceConfig {
    version: u32,
    // Map canonical workspace root -> workspace_id
    path_index: std::collections::BTreeMap<String, String>,
    // Map workspace_id -> info
    workspaces: std::collections::BTreeMap<String, WorkspaceInfo>,

    #[serde(default)]
    llm: Option<LlmConfig>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
enum LlmConfig {
    Openai {
        api_key: String,
        #[serde(default)]
        base_url: Option<String>,
        model: String,
    },
    Anthropic {
        api_key: String,
        #[serde(default)]
        base_url: Option<String>,
        model: String,
    },
    Ollama {
        #[serde(default = "default_ollama_base_url")]
        base_url: String,
        model: String,
    },
    Custom {
        base_url: String,
        #[serde(default)]
        api_key: Option<String>,
        model: String,
    },
}

fn default_ollama_base_url() -> String {
    "http://127.0.0.1:11434/v1".to_string()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkspaceInfo {
    id: String,
    root: String,
    source: String,
    db_dir: String,
    capsules_jsonl: String,
    created_ts_ms: i64,
    updated_ts_ms: i64,
}

#[derive(Debug, Clone)]
struct WorkspacePaths {
    id: String,
    db_dir: std::path::PathBuf,
    capsules_jsonl: std::path::PathBuf,
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

fn unlost_config_path() -> std::path::PathBuf {
    xdg_config_home().join("unlost").join("config.json")
}

fn unlost_data_root() -> std::path::PathBuf {
    xdg_data_home().join("unlost")
}

fn unlost_workspace_dir(workspace_id: &str) -> std::path::PathBuf {
    unlost_data_root().join("workspaces").join(workspace_id)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn canonicalize_dir(path: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    std::fs::canonicalize(path).context("failed to canonicalize path")
}

fn git_toplevel(dir: &std::path::Path) -> Option<std::path::PathBuf> {
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
    for (kind, contents) in meta_files {
        match kind.as_str() {
            "package_json" => {
                let json: serde_json::Value = serde_json::from_str(contents).ok()?;
                if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
                    return Some(name.to_string());
                }
            }
            "pyproject" => {
                let re = regex::Regex::new(
                    r#"\[project\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#,
                )
                .ok()?;
                if let Some(caps) = re.captures(contents) {
                    return Some(caps.get(1)?.as_str().to_string());
                }
                let re = regex::Regex::new(
                    r#"\[tool\.poetry\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#,
                )
                .ok()?;
                if let Some(caps) = re.captures(contents) {
                    return Some(caps.get(1)?.as_str().to_string());
                }
            }
            "cargo_toml" => {
                let re =
                    regex::Regex::new(r#"\[package\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#)
                        .ok()?;
                if let Some(caps) = re.captures(contents) {
                    return Some(caps.get(1)?.as_str().to_string());
                }
            }
            "go_mod" => {
                let re = regex::Regex::new(r#"^module\s+(\S+)"#).ok()?;
                for line in contents.lines() {
                    if let Some(caps) = re.captures(line) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn compute_workspace_id(workspace_root: &std::path::Path) -> Option<(String, String)> {
    if let Some(remote) = get_git_remote(workspace_root) {
        let norm = normalize_git_remote(&remote);
        if !norm.is_empty() {
            return Some((format!("wks_{}", compute_hash16(&format!("git:{norm}"))), "git".to_string()));
        }
    }

    let meta = read_meta_files(workspace_root);
    if let Some(name) = extract_project_name_from_meta_files(&meta) {
        return Some((
            format!("wks_{}", compute_hash16(&format!("manifest:{name}"))),
            "manifest".to_string(),
        ));
    }

    let label = workspace_root.file_name().and_then(|s| s.to_str()).unwrap_or("workspace");
    Some((
        format!("wks_{}", compute_hash16(&format!("label:cli:{label}"))),
        "label".to_string(),
    ))
}

fn collect_git_history_summary(
    root: &std::path::Path,
    max_commits: usize,
) -> anyhow::Result<Option<String>> {
    if max_commits == 0 {
        return Ok(None);
    }
    if git_toplevel(root).is_none() {
        return Ok(None);
    }

    let n = max_commits.min(50);
    let n_str = n.to_string();
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(["log", "-n", n_str.as_str(), "--pretty=format:%H%x1f%ct%x1f%s%x1e"])
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
            .current_dir(root)
            .args(["diff-tree", "--no-commit-id", "--name-only", "-r", hash])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect::<Vec<_>>())
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

fn load_workspace_config() -> WorkspaceConfig {
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

fn save_workspace_config(cfg: &WorkspaceConfig) -> anyhow::Result<()> {
    let p = unlost_config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = serde_json::to_string_pretty(cfg)?;
    std::fs::write(&p, s)?;
    Ok(())
}

async fn serve_request(
    client: Client<hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>, BoxBody<Bytes, hyper::Error>>,
    conn_id: u64,
    analysis_tx: kanal::Sender<AnalysisMsg>,
    analysis_drops_logged: Arc<AtomicBool>,
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
    let (workspace_id, provider, stripped_pq) = match parse_multiplexed_uri(req.uri()) {
        Ok(v) => v,
        Err(e) => {
            warn!(conn_id, error = ?e, "bad multiplexed uri");
            let resp = Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(text_body(b"expected /w/<workspace_id>/<provider>/..."))
                .unwrap();
            return Ok(resp);
        }
    };

    let upstream_host = provider.upstream_host().to_string();
    let upstream_port: u16 = 443;
    let upstream_uri = match Uri::builder()
        .scheme("https")
        .authority(format!("{}:{}", upstream_host, upstream_port))
        .path_and_query(stripped_pq.as_str())
        .build()
    {
        Ok(u) => u,
        Err(e) => {
            warn!(conn_id, error = ?e, "failed to build upstream uri");
            let resp = Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(text_body(b"bad request"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (mut parts, body) = req.into_parts();
    let method = parts.method.clone();
    let version = parts.version;
    parts.uri = Uri::builder()
        .path_and_query(stripped_pq.as_str())
        .build()
        .unwrap_or_else(|_| Uri::from_static("/"));
    let request_path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_string();
    let headers = sanitize_request_headers(parts.headers, &upstream_host);

    const MAX_REQ_BODY: usize = 2 * 1024 * 1024;
    let req_body_bytes = match read_incoming_body_limited(body, MAX_REQ_BODY).await {
        Ok(b) => b,
        Err(e) => {
            warn!(conn_id, error = ?e, "request body capture failed");
            let resp = Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(text_body(b"payload too large"))
                .unwrap();
            return Ok(resp);
        }
    };

    let mut out_req = Request::builder()
        .method(method)
        .uri(upstream_uri)
        .version(version)
        .body(bytes_body(req_body_bytes.clone()))
        .unwrap();
    *out_req.headers_mut() = headers;

    let res = match client.request(out_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(conn_id, error = ?e, "upstream request failed");
            let resp = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(text_body(b"bad gateway"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (res_parts, res_body) = res.into_parts();
    let res_headers = sanitize_response_headers(res_parts.headers);
    let content_type = res_headers
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let meta = AnalysisMeta {
        workspace_id,
        upstream_host: upstream_host.clone(),
        request_path: request_path.clone(),
        http_status: res_parts.status.as_u16(),
        content_type,
    };
    let _ = analysis_tx.try_send(AnalysisMsg::ExchangeStart {
        meta,
        request_body: req_body_bytes,
    });

    let tx = analysis_tx;
    let drops_logged = analysis_drops_logged;
    let stream = futures_util::stream::unfold(
        (BodyStream::new(res_body), tx, drops_logged, false),
        move |(mut bs, tx, drops_logged, ended)| async move {
            if ended {
                return None;
            }
            match bs.try_next().await {
                Ok(Some(frame)) => {
                    if let Some(data) = frame.data_ref() {
                        match tx.try_send(AnalysisMsg::ResponseChunk(data.clone())) {
                            Ok(true) => {}
                            Ok(false) => {
                                if !drops_logged.swap(true, Ordering::Relaxed) {
                                    info!(conn_id, "analysis channel full; dropping response bytes");
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    Some((Ok(frame), (bs, tx, drops_logged, false)))
                }
                Ok(None) => {
                    let _ = tx.try_send(AnalysisMsg::ResponseEnd);
                    None
                }
                Err(e) => {
                    let _ = tx.try_send(AnalysisMsg::ResponseEnd);
                    Some((Err(e), (bs, tx, drops_logged, true)))
                }
            }
        },
    );

    let out_body = StreamBody::new(stream).boxed();
    let mut out_res = Response::new(out_body);
    *out_res.status_mut() = res_parts.status;
    *out_res.version_mut() = res_parts.version;
    *out_res.headers_mut() = res_headers;
    Ok(out_res)
}

fn get_llm_config() -> Option<LlmConfig> {
    load_workspace_config().llm
}

fn set_llm_config(new_cfg: Option<LlmConfig>) -> anyhow::Result<()> {
    let mut cfg = load_workspace_config();
    cfg.llm = new_cfg;
    save_workspace_config(&cfg)
}

async fn llm_extract<T>(
    model_override: Option<&str>,
    preamble: &str,
    input: &str,
) -> anyhow::Result<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    let cfg = get_llm_config();

    match cfg {
        Some(LlmConfig::Openai { api_key, base_url, model }) => {
            let model = model_override.unwrap_or(&model);
            let mut builder: openai::ClientBuilder<reqwest::Client> =
                openai::Client::builder().api_key(&api_key);
            if let Some(base) = base_url.as_deref() {
                builder = builder.base_url(base);
            }
            let client = builder.build().context("failed to build OpenAI client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Anthropic { api_key, base_url, model }) => {
            let model = model_override.unwrap_or(&model);
            let mut builder: anthropic::ClientBuilder<reqwest::Client> =
                anthropic::Client::builder().api_key(api_key);
            if let Some(base) = base_url.as_deref() {
                builder = builder.base_url(base);
            }
            let client = builder.build().context("failed to build Anthropic client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Ollama { base_url, model }) => {
            // Ollama provides an OpenAI-compatible endpoint. Use a dummy key.
            let model = model_override.unwrap_or(&model);
            let mut builder: openai::ClientBuilder<reqwest::Client> =
                openai::Client::builder().api_key("ollama");
            builder = builder.base_url(&base_url);
            let client = builder
                .build()
                .context("failed to build Ollama (OpenAI-compatible) client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Custom { base_url, api_key, model }) => {
            let model = model_override.unwrap_or(&model);
            let key = api_key.as_deref().unwrap_or("custom");
            let mut builder: openai::ClientBuilder<reqwest::Client> =
                openai::Client::builder().api_key(key);
            builder = builder.base_url(&base_url);
            let client = builder
                .build()
                .context("failed to build custom OpenAI-compatible client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        None => {
            // Default: OpenAI from env.
            let model = model_override.unwrap_or("gpt-4o-mini");
            let client = openai::Client::from_env();
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
    }
}

fn get_or_create_workspace_paths(workspace_root: &std::path::Path) -> anyhow::Result<WorkspacePaths> {
    let root = git_toplevel(workspace_root)
        .unwrap_or_else(|| canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf()));
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
    })
}

fn clear_workspace(workspace_root: &std::path::Path) -> anyhow::Result<()> {
    let root = git_toplevel(workspace_root)
        .unwrap_or_else(|| canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf()));
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
    if ws_dir.exists() {
        std::fs::remove_dir_all(&ws_dir)
            .with_context(|| format!("failed to delete {}", ws_dir.display()))?;
        println!("deleted: {}", ws_dir.display());
    } else {
        println!("no data dir for workspace {workspace_id} (expected {})", ws_dir.display());
    }

    // Remove config mappings pointing to this id.
    cfg.workspaces.remove(&workspace_id);
    cfg.path_index.retain(|_k, v| v != &workspace_id);
    save_workspace_config(&cfg)?;

    println!("cleared workspace: {workspace_id}");
    Ok(())
}

fn show_llm_config() {
    let cfg = get_llm_config();
    match cfg {
        None => {
            println!("LLM: not configured");
        }
        Some(LlmConfig::Openai { base_url, model, .. }) => {
            println!("LLM: openai");
            println!("model: {model}");
            if let Some(b) = base_url {
                println!("base_url: {b}");
            }
        }
        Some(LlmConfig::Anthropic { base_url, model, .. }) => {
            println!("LLM: anthropic");
            println!("model: {model}");
            if let Some(b) = base_url {
                println!("base_url: {b}");
            }
        }
        Some(LlmConfig::Ollama { base_url, model }) => {
            println!("LLM: ollama");
            println!("model: {model}");
            println!("base_url: {base_url}");
        }
        Some(LlmConfig::Custom { base_url, model, .. }) => {
            println!("LLM: custom");
            println!("model: {model}");
            println!("base_url: {base_url}");
        }
    }
}

fn handle_llm_command(cmd: LlmCommand) -> anyhow::Result<()> {
    match cmd {
        LlmCommand::Openai { api_key, base_url, model } => {
            set_llm_config(Some(LlmConfig::Openai { api_key, base_url, model }))?;
            println!("LLM provider set to OpenAI");
        }
        LlmCommand::Anthropic { api_key, base_url, model } => {
            set_llm_config(Some(LlmConfig::Anthropic { api_key, base_url, model }))?;
            println!("LLM provider set to Anthropic");
        }
        LlmCommand::Ollama { base_url, model } => {
            set_llm_config(Some(LlmConfig::Ollama { base_url, model }))?;
            println!("LLM provider set to Ollama");
        }
        LlmCommand::Custom { base_url, api_key, model } => {
            set_llm_config(Some(LlmConfig::Custom { base_url, api_key, model }))?;
            println!("LLM provider set to custom endpoint");
        }
        LlmCommand::Show => {
            show_llm_config();
        }
        LlmCommand::Remove => {
            set_llm_config(None)?;
            println!("LLM configuration removed");
        }
    }

    Ok(())
}

fn ensure_object(v: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    if !v.is_object() {
        *v = serde_json::Value::Object(serde_json::Map::new());
    }
    v.as_object_mut().unwrap()
}

fn set_nested_string(root: &mut serde_json::Value, path: &[&str], value: String) {
    let mut cur = root;
    for (i, key) in path.iter().enumerate() {
        let is_last = i + 1 == path.len();
        let obj = ensure_object(cur);
        if is_last {
            obj.insert((*key).to_string(), serde_json::Value::String(value));
            return;
        }
        cur = obj.entry((*key).to_string()).or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    }
}

fn handle_agent_command(cmd: AgentCommand) -> anyhow::Result<()> {
    match cmd {
        AgentCommand::Opencode { path, server } => {
            let root = git_toplevel(std::path::Path::new(&path))
                .unwrap_or_else(|| canonicalize_dir(std::path::Path::new(&path)).unwrap_or_else(|_| std::path::PathBuf::from(&path)));
            let root = canonicalize_dir(&root)?;

            let (workspace_id, _src) = compute_workspace_id(&root)
                .ok_or_else(|| anyhow::anyhow!("unable to compute workspace id"))?;

            let server = server.trim_end_matches('/');
            let base_url = format!("{server}/w/{workspace_id}/opencode/zen/v1");

            let cfg_path = root.join("opencode.json");
            let mut json = match std::fs::read_to_string(&cfg_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s).unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Object(serde_json::Map::new()),
                Err(e) => return Err(e.into()),
            };

            set_nested_string(
                &mut json,
                &["$schema"],
                "https://opencode.ai/config.json".to_string(),
            );
            set_nested_string(
                &mut json,
                &["provider", "opencode", "options", "baseURL"],
                base_url,
            );

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;
            println!("configured: {}", cfg_path.display());
        }
    }
    Ok(())
}

static CONN_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct ResponseMeta {
    source: String,
    upstream_host: String,
    request_path: String,
    http_status: u16,
}

#[derive(Debug, Clone)]
struct AnalysisMeta {
    workspace_id: String,
    upstream_host: String,
    request_path: String,
    http_status: u16,
    content_type: Option<String>,
}

#[derive(Debug)]
enum AnalysisMsg {
    ExchangeStart { meta: AnalysisMeta, request_body: Bytes },
    ResponseChunk(Bytes),
    ResponseEnd,
}

fn escape_sql_string(s: &str) -> String {
    s.replace("'", "''")
}

type Embedder = Arc<std::sync::Mutex<TextEmbedding>>;

fn default_embed_cache_dir() -> std::path::PathBuf {
    // Large model artifacts belong in XDG_DATA_HOME.
    // Linux default: ~/.local/share/unlost/models/fastembed
    if let Some(dir) = std::env::var_os("UNLOST_EMBED_CACHE_DIR") {
        return std::path::PathBuf::from(dir);
    }

    let base = if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        std::path::PathBuf::from(xdg)
    } else if let Some(home) = std::env::var_os("HOME") {
        std::path::PathBuf::from(home).join(".local").join("share")
    } else {
        std::path::PathBuf::from(".")
    };

    base.join("unlost").join("models").join("fastembed")
}

fn parse_embed_model(s: &str) -> anyhow::Result<EmbeddingModel> {
    let raw = s.trim();

    // Friendly aliases
    let aliased = match raw.to_ascii_lowercase().as_str() {
        // Common HF name used in docs/blog posts
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" | "bge-small-en" => DEFAULT_EMBED_MODEL,

        // Enum variant names people might guess
        "bgesmallenv15" => DEFAULT_EMBED_MODEL,
        "bgesmallenv15q" => "Qdrant/bge-small-en-v1.5-onnx-Q",
        _ => raw,
    };

    aliased
        .parse::<EmbeddingModel>()
        .map_err(|e| anyhow::anyhow!("unknown embedding model '{raw}': {e}"))
}

async fn load_embedder(
    model: &str,
    cache_dir: Option<std::path::PathBuf>,
    show_progress: bool,
) -> anyhow::Result<Embedder> {
    let model = parse_embed_model(model)?;
    let cache_dir = cache_dir
        .unwrap_or_else(default_embed_cache_dir);

    let info = TextEmbedding::get_model_info(&model)?;
    if info.dim != DEFAULT_EMBED_DIM {
        anyhow::bail!("embedding dimension mismatch: {} (expected {DEFAULT_EMBED_DIM})", info.dim);
    }

    info!(model = %model, cache_dir = %cache_dir.display(), "loading local embedder");

    let options = FastEmbedInitOptions::new(model)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(show_progress);

    let embedder = tokio::task::spawn_blocking(move || TextEmbedding::try_new(options))
        .await
        .context("embedding init task failed")??;

    Ok(Arc::new(std::sync::Mutex::new(embedder)))
}

async fn embed_text(embedder: &Embedder, text: &str) -> anyhow::Result<Vec<f32>> {
    let embedder = embedder.clone();
    let text = text.to_string();
    tokio::task::spawn_blocking(move || {
        let mut model = embedder.lock().map_err(|_| anyhow::anyhow!("embedder lock poisoned"))?;
        let mut out = model.embed([text], Some(1))?;
        out.pop().ok_or_else(|| anyhow::anyhow!("embedding model returned no vectors"))
    })
    .await
    .context("embedding task failed")?
}

async fn download_model(model: &str, cache_dir: Option<&str>, force: bool) -> anyhow::Result<std::path::PathBuf> {
    let cache_dir = cache_dir
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_embed_cache_dir);

    if force && cache_dir.exists() {
        tokio::fs::remove_dir_all(&cache_dir).await.ok();
    }

    // Trigger download by initializing the model.
    let _ = load_embedder(model, Some(cache_dir.clone()), true).await?;
    Ok(cache_dir)
}

const CAPSULES_TABLE: &str = "capsules";

fn capsules_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("ts_ms", DataType::Int64, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("upstream_host", DataType::Utf8, false),
        Field::new("request_path", DataType::Utf8, false),
        Field::new("http_status", DataType::Int32, false),
        Field::new("conn_id", DataType::Int64, false),
        Field::new("exchange_seq", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("intent", DataType::Utf8, false),
        Field::new("decision", DataType::Utf8, false),
        Field::new("rationale", DataType::Utf8, false),
        Field::new(
            "next_steps",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "symbols",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
            false,
        ),
    ]))
}

async fn ensure_capsules_table(db: &Connection) -> anyhow::Result<lancedb::Table> {
    match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => Ok(t),
        Err(_) => {
            info!(table = CAPSULES_TABLE, "creating lancedb table");
            let schema = capsules_schema();

            let id = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let ts_ms = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let source = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let upstream_host = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let request_path = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let http_status = Arc::new(Int32Array::from_iter_values(std::iter::empty::<i32>()));
            let conn_id = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let exchange_seq = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let category = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let intent = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let decision = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let rationale = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));

            let mut next_steps_builder = ListBuilder::new(StringBuilder::new());
            let next_steps = Arc::new(next_steps_builder.finish());

            let mut symbols_builder = ListBuilder::new(StringBuilder::new());
            let symbols = Arc::new(symbols_builder.finish());

            let embedding = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                std::iter::empty::<Option<Vec<Option<f32>>>>(),
                384,
            ));

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    id,
                    ts_ms,
                    source,
                    upstream_host,
                    request_path,
                    http_status,
                    conn_id,
                    exchange_seq,
                    category,
                    intent,
                    decision,
                    rationale,
                    next_steps,
                    symbols,
                    embedding,
                ],
            )
            .context("failed to build empty schema batch")?;

            let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
            let table = db
                .create_table(CAPSULES_TABLE, Box::new(batches))
                .execute()
                .await
                .with_context(|| format!("failed to create {CAPSULES_TABLE}"))?;

            table.create_index(&["embedding"], Index::Auto).execute().await.ok();
            table
                .create_index(&["symbols"], Index::LabelList(LabelListIndexBuilder::default()))
                .execute()
                .await
                .ok();

            Ok(table)
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "unlost",
    version,
    about = "Local-first code memory (record, init, query)"
)]
struct Cli {
    /// Logging level for unlost (overrides RUST_LOG when set)
    #[arg(long, global = true, value_enum, alias = "log-level")]
    log: Option<LogLevel>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_tracing_str(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    /// Default terminal-friendly output (ANSI colors)
    Ansi,
    /// No ANSI colors (useful for piping)
    Plain,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Global recorder that multiplexes workspaces via base URL
    Serve {
        /// Bind address. Accepts either `port` or `ip:port`.
        /// Examples: `3000`, `127.0.0.1:3000`.
        #[arg(long, default_value = "127.0.0.1:3000")]
        bind: String,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Record live LLM conversations (captures and summarizes)
    #[command(alias = "proxy")]
    Record {
        /// Bind address. Accepts either `port` or `ip:port`.
        /// Examples: `3000`, `0.0.0.0:3000`.
        #[arg(long, default_value = "3000")]
        bind: String,

        /// Upstream host (or set UNLOST_UPSTREAM_HOST)
        #[arg(long, env = "UNLOST_UPSTREAM_HOST")]
        upstream_host: String,

        /// Upstream port (or set UNLOST_UPSTREAM_PORT)
        #[arg(long, env = "UNLOST_UPSTREAM_PORT", default_value_t = 443)]
        upstream_port: u16,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Semantic search across recorded capsules
    Query {
        /// Query text
        query: Vec<String>,

        /// Max results
        #[arg(long, default_value_t = 5)]
        limit: usize,

        /// Filter results to a symbol
        #[arg(long)]
        symbol: Option<String>,

        /// Disable LLM narrative (prints raw matches)
        #[arg(long, default_value_t = false)]
        no_llm: bool,

        /// LLM model to use for query narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Print raw match facts after the narrative
        #[arg(long, default_value_t = false)]
        facts: bool,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Path to capsules JSONL (fallback mode only). Defaults to the workspace's JSONL.
        #[arg(long, default_value = "")]
        file: String,
    },

    /// Recall the story so far (proactive overview)
    Recall {
        /// Optional scope (file path or symbol/function name)
        target: Vec<String>,

        /// Max capsules to use
        #[arg(long, default_value_t = 24)]
        limit: usize,

        /// LLM model to use for recall narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Inspect stored capsules for this workspace
    Inspect {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Max rows to print
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Optional Lance filter expression (DataFusion SQL)
        #[arg(long)]
        filter: Option<String>,
    },

    /// Seed LanceDB from the current codebase (unfault-core graph)
    Init {
        /// Root directory to scan
        #[arg(long, default_value = ".")]
        path: String,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Max number of capsules to insert
        #[arg(long, default_value_t = 120)]
        max_capsules: usize,

        /// Disable LLM summaries for init
        #[arg(long, default_value_t = false)]
        no_llm: bool,

        /// Include recent git history (commit subjects + touched files) when available
        #[arg(long, default_value_t = true)]
        git_history: bool,

        /// Max commits to consider for git history (bounded)
        #[arg(long, default_value_t = 50)]
        git_commits: usize,

        /// LLM model to use for init summaries
        #[arg(long)]
        llm_model: Option<String>,

        /// Max LLM-generated capsules
        #[arg(long, default_value_t = 12)]
        llm_max_capsules: usize,
    },

    /// Manage local models (download, etc.)
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },

    /// Manage configuration (LLM provider, etc.)
    #[command(alias = "configure")]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Delete all generated data for the current workspace
    Clear {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    /// Manage LLM configuration for init/query narratives
    Llm {
        #[command(subcommand)]
        command: LlmCommand,
    },

    /// Configure an agent workspace to talk to unlost
    Agent {
        #[command(subcommand)]
        command: AgentCommand,
    },
}

#[derive(Debug, Subcommand)]
enum AgentCommand {
    /// Configure OpenCode via opencode.json in the workspace
    Opencode {
        /// Workspace path (defaults to current directory; uses git toplevel)
        #[arg(long, default_value = ".")]
        path: String,

        /// unlost server base URL (loopback)
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        server: String,
    },
}

#[derive(Debug, Subcommand)]
enum LlmCommand {
    /// Configure OpenAI as LLM provider
    Openai {
        /// OpenAI API key
        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: String,

        /// Default model to use
        #[arg(long, default_value = "gpt-4o-mini")]
        model: String,

        /// Optional base URL override (OpenAI-compatible)
        #[arg(long)]
        base_url: Option<String>,
    },

    /// Configure Anthropic as LLM provider
    Anthropic {
        /// Anthropic API key
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        api_key: String,

        /// Default model to use
        #[arg(long, default_value = "claude-3-5-sonnet-20241022")]
        model: String,

        /// Optional base URL override
        #[arg(long)]
        base_url: Option<String>,
    },

    /// Configure local Ollama as LLM provider (OpenAI-compatible endpoint)
    Ollama {
        /// Ollama model name (e.g. llama3.2:3b)
        #[arg(long)]
        model: String,

        /// OpenAI-compatible base URL (default: http://127.0.0.1:11434/v1)
        #[arg(long, default_value = "http://127.0.0.1:11434/v1")]
        base_url: String,
    },

    /// Configure a custom OpenAI-compatible endpoint
    Custom {
        /// Base URL (e.g. https://my-endpoint/v1)
        #[arg(long)]
        base_url: String,

        /// API key (if required)
        #[arg(long)]
        api_key: Option<String>,

        /// Default model to use
        #[arg(long)]
        model: String,
    },

    /// Show current LLM configuration
    Show,

    /// Remove LLM configuration
    Remove,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download embedding model files into the local cache
    Download {
        /// Embedding model (fastembed)
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        cache_dir: Option<String>,

        /// Delete cache dir before downloading
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

fn parse_bind(s: &str) -> anyhow::Result<SocketAddr> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("bind cannot be empty");
    }

    // `:3000`
    if let Some(port_str) = s.strip_prefix(':') {
        let port: u16 = port_str.parse().context("invalid port")?;
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port));
    }

    // `3000`
    if s.chars().all(|c| c.is_ascii_digit()) {
        let port: u16 = s.parse().context("invalid port")?;
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port));
    }

    // `ip:port`
    Ok(s.parse().context("invalid bind address")?)
}

fn hop_by_hop_names(headers: &HeaderMap) -> Vec<HeaderName> {
    // RFC 7230 §6.1: headers listed in `Connection` are hop-by-hop.
    let mut out = Vec::new();
    for val in headers.get_all(hyper::header::CONNECTION).iter() {
        if let Ok(s) = val.to_str() {
            for token in s.split(',') {
                let token = token.trim();
                if token.is_empty() {
                    continue;
                }
                if let Ok(name) = HeaderName::from_bytes(token.as_bytes()) {
                    out.push(name);
                }
            }
        }
    }
    out
}

fn sanitize_request_headers(mut headers: HeaderMap, upstream_host: &str) -> HeaderMap {
    // Remove hop-by-hop headers
    let mut remove = hop_by_hop_names(&headers);
    remove.extend([
        hyper::header::CONNECTION,
        HeaderName::from_static("proxy-connection"),
        HeaderName::from_static("keep-alive"),
        hyper::header::TE,
        hyper::header::TRAILER,
        hyper::header::TRANSFER_ENCODING,
        hyper::header::UPGRADE,
    ]);
    for name in remove {
        headers.remove(name);
    }

    // Force upstream Host
    headers.remove(hyper::header::HOST);
    if let Ok(v) = HeaderValue::from_str(upstream_host) {
        headers.insert(hyper::header::HOST, v);
    }

    // Make response bodies easier to parse (SSE/JSON) by avoiding compression.
    headers.remove(hyper::header::ACCEPT_ENCODING);
    headers.insert(hyper::header::ACCEPT_ENCODING, HeaderValue::from_static("identity"));

    headers
}

fn sanitize_response_headers(mut headers: HeaderMap) -> HeaderMap {
    let mut remove = hop_by_hop_names(&headers);
    remove.extend([
        hyper::header::CONNECTION,
        HeaderName::from_static("proxy-connection"),
        HeaderName::from_static("keep-alive"),
        hyper::header::TE,
        hyper::header::TRAILER,
        hyper::header::TRANSFER_ENCODING,
        hyper::header::UPGRADE,
    ]);
    for name in remove {
        headers.remove(name);
    }
    headers
}

fn build_upstream_uri(upstream_host: &str, upstream_port: u16, uri: &Uri) -> anyhow::Result<Uri> {
    let pq = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let authority = format!("{upstream_host}:{upstream_port}");
    Ok(Uri::builder()
        .scheme("https")
        .authority(authority)
        .path_and_query(pq)
        .build()
        .context("failed to build upstream URI")?)
}

fn text_body(msg: &'static [u8]) -> BoxBody<Bytes, hyper::Error> {
    Full::new(Bytes::from_static(msg))
        .map_err(|never| match never {})
        .boxed()
}

fn bytes_body(b: Bytes) -> BoxBody<Bytes, hyper::Error> {
    Full::new(b).map_err(|never| match never {}).boxed()
}

async fn read_incoming_body_limited(
    body: hyper::body::Incoming,
    max_bytes: usize,
) -> anyhow::Result<Bytes> {
    let mut bs = BodyStream::new(body);
    let mut out: Vec<u8> = Vec::new();
    while let Some(frame) = bs.try_next().await? {
        if let Some(data) = frame.data_ref() {
            if out.len() + data.len() > max_bytes {
                anyhow::bail!("request body too large (>{} bytes)", max_bytes);
            }
            out.extend_from_slice(data);
        }
    }
    Ok(Bytes::from(out))
}

fn is_event_stream(content_type: Option<&str>) -> bool {
    content_type
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("text/event-stream")
}

fn decode_json_lossy(bytes: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(bytes).ok()
}

fn extract_openai_message_text(v: &serde_json::Value) -> Option<String> {
    // Supports both request bodies (messages/input) and response bodies (choices/message).
    // We intentionally take the most recent user message only.
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        let mut out: Vec<String> = Vec::new();
        for msg in messages.iter().rev() {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role != "user" {
                if !out.is_empty() {
                    break;
                }
                continue;
            }

            // content can be string or an array of parts like {type:"text", text:"..."}
            let content = msg.get("content");
            if let Some(s) = content.and_then(|c| c.as_str()) {
                out.push(s.to_string());
            } else if let Some(parts) = content.and_then(|c| c.as_array()) {
                let mut buf = String::new();
                for p in parts {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        }
                    }
                }
                if !buf.trim().is_empty() {
                    out.push(buf);
                }
            }
        }

        if out.is_empty() {
            return None;
        }
        out.reverse();
        return Some(out.join("\n\n"));
    }

    if let Some(input) = v.get("input") {
        if let Some(s) = input.as_str() {
            return Some(s.to_string());
        }
        if let Some(arr) = input.as_array() {
            for item in arr.iter().rev() {
                // best-effort: look for user role content
                let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role != "user" {
                    continue;
                }
                if let Some(s) = item.get("content").and_then(|c| c.as_str()) {
                    return Some(s.to_string());
                }
            }
        }
    }

    None
}

fn extract_anthropic_user_text(v: &serde_json::Value) -> Option<String> {
    let Some(messages) = v.get("messages").and_then(|m| m.as_array()) else {
        return None;
    };

    let mut out: Vec<String> = Vec::new();
    for msg in messages.iter().rev() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "user" {
            if !out.is_empty() {
                break;
            }
            continue;
        }
        if let Some(content) = msg.get("content") {
            if let Some(s) = content.as_str() {
                out.push(s.to_string());
            } else if let Some(parts) = content.as_array() {
                let mut buf = String::new();
                for p in parts {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        }
                    }
                }
                if !buf.trim().is_empty() {
                    out.push(buf);
                }
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        out.reverse();
        Some(out.join("\n\n"))
    }
}

fn extract_openai_assistant_text_from_json(v: &serde_json::Value) -> Option<String> {
    // chat.completions style
    if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice0) = choices.first() {
            if let Some(s) = choice0
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                return Some(s.to_string());
            }
        }
    }

    // responses API style
    if let Some(s) = v.get("output_text").and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }

    None
}

fn extract_anthropic_assistant_text_from_json(v: &serde_json::Value) -> Option<String> {
    let Some(content) = v.get("content") else {
        return None;
    };
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    if let Some(parts) = content.as_array() {
        let mut buf = String::new();
        for p in parts {
            if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                    buf.push_str(t);
                }
            }
        }
        if !buf.trim().is_empty() {
            return Some(buf);
        }
    }
    None
}

fn sse_extract_deltas(sse_buf: &mut Vec<u8>, assistant: &mut String) {
    // Parse SSE frames split by blank line. We only keep text deltas.
    // NOTE: We do not keep any transcript/evidence in storage; assistant text is transient.
    loop {
        let Some(pos) = sse_buf.windows(2).position(|w| w == b"\n\n") else {
            break;
        };
        let block = sse_buf.drain(..pos + 2).collect::<Vec<u8>>();
        let block = String::from_utf8_lossy(&block);
        let mut data_lines: Vec<&str> = Vec::new();
        for line in block.lines() {
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim());
            }
        }

        if data_lines.is_empty() {
            continue;
        }

        let data = data_lines.join("\n");
        if data == "[DONE]" {
            continue;
        }

        let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) else {
            continue;
        };

        // OpenAI chat completions stream
        if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
            if let Some(delta) = choices
                .first()
                .and_then(|c0| c0.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                assistant.push_str(delta);
                continue;
            }
        }

        // OpenAI responses API stream
        if v.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta") {
            if let Some(delta) = v.get("delta").and_then(|d| d.as_str()) {
                assistant.push_str(delta);
                continue;
            }
        }

        // Anthropic messages stream
        if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
            if let Some(delta) = v
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
            {
                assistant.push_str(delta);
                continue;
            }
        }

        // fallback: some variants include nested delta text
        if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
            if let Some(delta) = v
                .get("delta")
                .and_then(|d| d.get("delta"))
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
            {
                assistant.push_str(delta);
                continue;
            }
        }

        // ignored event
    }
}

fn append_capsule_jsonl(
    path: &std::path::Path,
    ts_ms: i64,
    conn_id: u64,
    exchange_seq: u64,
    meta: &ResponseMeta,
    capsule: &IntentCapsule,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let record = serde_json::json!({
        "ts_ms": ts_ms,
        "conn_id": conn_id,
        "exchange_seq": exchange_seq,
        "source": meta.source,
        "upstream_host": meta.upstream_host,
        "request_path": meta.request_path,
        "http_status": meta.http_status,
        "capsule": capsule,
    });
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{}\n", record).as_bytes())?;
    Ok(())
}

fn contains_hex_hash(s: &str) -> bool {
    for tok in s
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        if tok.len() < 7 || tok.len() > 40 {
            continue;
        }
        if tok.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f' | 'A'..='F')) {
            return true;
        }
    }
    false
}

fn looks_like_commit_or_pr(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    if s.contains("git commit") || s.contains("commit:") || s.contains("commit ") {
        return true;
    }
    if s.contains("pull request") || s.contains("merge request") {
        return true;
    }
    if s.contains("pr #") || s.contains("pr#") {
        return true;
    }

    // cheap hash heuristic (7+ hex) preceded by common tokens
    if s.contains("sha") || s.contains("hash") || s.contains("commit") {
        if contains_hex_hash(&s) {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
struct ChunkInput {
    conn_id: u64,
    upstream_host: String,
    request_path: String,
    http_status: u16,
    exchange_text: String,
    commit_mentioned: bool,
}

#[derive(Debug, Clone)]
struct FlushJob {
    workspace_id: String,
    conn_id: u64,
    exchange_seq: u64,
    ts_ms: i64,
    meta: ResponseMeta,
    input: String,
}

struct WorkspaceBuffer {
    next_seq: u64,
    last_activity: Instant,
    last_conn_id: u64,
    last_upstream_host: String,
    last_request_path: String,
    last_http_status: u16,
    total_chars: usize,
    turns: Vec<String>,
    saw_commit: bool,
}

impl WorkspaceBuffer {
    fn new(now: Instant) -> Self {
        Self {
            next_seq: 0,
            last_activity: now,
            last_conn_id: 0,
            last_upstream_host: String::new(),
            last_request_path: String::new(),
            last_http_status: 0,
            total_chars: 0,
            turns: Vec::new(),
            saw_commit: false,
        }
    }
}

#[derive(Clone)]
struct WorkspaceChunker {
    buffers: Arc<Mutex<HashMap<String, WorkspaceBuffer>>>,
    flush_tx: kanal::AsyncSender<FlushJob>,
}

impl WorkspaceChunker {
    const IDLE_FLUSH_AFTER: Duration = Duration::from_secs(2);
    const MAX_TOTAL_CHARS: usize = 16 * 1024;
    const MAX_TURNS: usize = 8;

    fn new(flush_tx: kanal::AsyncSender<FlushJob>) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            flush_tx,
        }
    }

    async fn ingest(&self, workspace_id: String, item: ChunkInput) {
        let now = Instant::now();
        let mut maybe_flush: Option<FlushJob> = None;

        {
            let mut map = self.buffers.lock().await;
            let buf = map.entry(workspace_id.clone()).or_insert_with(|| WorkspaceBuffer::new(now));

            // If this buffer has been idle, flush it before appending new content.
            if !buf.turns.is_empty() && now.duration_since(buf.last_activity) >= Self::IDLE_FLUSH_AFTER {
                maybe_flush = Some(build_flush_job(workspace_id.clone(), buf));
                *buf = WorkspaceBuffer::new(now);
            }

            buf.last_activity = now;
            buf.last_conn_id = item.conn_id;
            buf.last_upstream_host = item.upstream_host;
            buf.last_request_path = item.request_path;
            buf.last_http_status = item.http_status;
            buf.saw_commit |= item.commit_mentioned;

            buf.total_chars = buf.total_chars.saturating_add(item.exchange_text.len());
            buf.turns.push(item.exchange_text);

            // Force flush boundaries.
            let too_big = buf.total_chars >= Self::MAX_TOTAL_CHARS;
            let too_many = buf.turns.len() >= Self::MAX_TURNS;
            let milestone = item.commit_mentioned;
            if too_big || too_many || milestone {
                if maybe_flush.is_none() {
                    maybe_flush = Some(build_flush_job(workspace_id.clone(), buf));
                    *buf = WorkspaceBuffer::new(now);
                }
            }
        }

        if let Some(job) = maybe_flush {
            let _ = self.flush_tx.send(job).await;
        }
    }

    async fn flush_idle(&self) {
        let now = Instant::now();
        let mut jobs: Vec<FlushJob> = Vec::new();

        {
            let mut map = self.buffers.lock().await;
            for (ws_id, buf) in map.iter_mut() {
                if buf.turns.is_empty() {
                    continue;
                }
                if now.duration_since(buf.last_activity) < Self::IDLE_FLUSH_AFTER {
                    continue;
                }
                jobs.push(build_flush_job(ws_id.clone(), buf));
                *buf = WorkspaceBuffer::new(now);
            }
        }

        for j in jobs {
            let _ = self.flush_tx.send(j).await;
        }
    }
}

fn build_flush_job(workspace_id: String, buf: &mut WorkspaceBuffer) -> FlushJob {
    buf.next_seq += 1;
    let exchange_seq = buf.next_seq;
    let ts_ms = now_ms();

    let mut input = String::new();
    input.push_str("Signals:\n");
    input.push_str(&format!("commit_mentioned={}\n\n", buf.saw_commit));
    input.push_str("Conversation slice (newest last):\n\n");
    for (i, t) in buf.turns.iter().enumerate() {
        input.push_str(&format!("Turn {}:\n", i + 1));
        input.push_str(t);
        input.push_str("\n\n");
    }

    let meta = ResponseMeta {
        source: "record".to_string(),
        upstream_host: buf.last_upstream_host.clone(),
        request_path: buf.last_request_path.clone(),
        http_status: buf.last_http_status,
    };

    FlushJob {
        workspace_id,
        conn_id: buf.last_conn_id,
        exchange_seq,
        ts_ms,
        meta,
        input,
    }
}

fn query_capsules_jsonl(path: &str, query: &str, limit: usize) -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    struct Row {
        ts_ms: Option<i64>,
        conn_id: Option<u64>,
        exchange_seq: Option<u64>,
        capsule: IntentCapsule,
    }

    let q = query.to_lowercase();
    let data = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("No capsules file found at: {path}");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    let mut matches: Vec<Row> = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let hay = format!(
            "{}\n{}\n{}",
            row.capsule.intent,
            row.capsule.decision,
            row.capsule.rationale
        )
        .to_lowercase();
        if hay.contains(&q) {
            matches.push(row);
        }
    }

    matches.reverse();
    let out = matches.into_iter().take(limit).collect::<Vec<_>>();
    if out.is_empty() {
        println!("No matches for: {query}");
        return Ok(());
    }

    println!("Found {} matches:\n", out.len());
    for row in out {
        println!("---");
        if let Some(ts_ms) = row.ts_ms {
            println!("ts_ms:   {ts_ms}");
        }
        if let Some(conn_id) = row.conn_id {
            println!("conn_id: {conn_id}");
        }
        if let Some(exchange_seq) = row.exchange_seq {
            println!("exchange: {exchange_seq}");
        }
        println!("category:  {}", row.capsule.category);
        if !row.capsule.intent.trim().is_empty() {
            println!("intent:    {}", row.capsule.intent);
        }
        if !row.capsule.decision.trim().is_empty() {
            println!("decision:  {}", row.capsule.decision);
        }
        if !row.capsule.rationale.trim().is_empty() {
            println!("rationale: {}", row.capsule.rationale);
        }
        if !row.capsule.next_steps.is_empty() {
            println!("next:      {:?}", row.capsule.next_steps);
        }
        println!("symbols:   {:?}\n", row.capsule.symbols);
    }

    Ok(())
}

async fn analysis_worker(
    rx: AsyncReceiver<AnalysisMsg>,
    chunker: WorkspaceChunker,
    conn_id: u64,
) {
    debug!(conn_id, "analysis worker started");
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;

    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() {
            p
        } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                Ok(_) => continue,
                Err(_) => break,
            }
        };

        let user_text = decode_json_lossy(&request_body)
            .and_then(|v| {
                if meta.request_path.contains("/v1/messages") {
                    extract_anthropic_user_text(&v).or_else(|| extract_openai_message_text(&v))
                } else {
                    extract_openai_message_text(&v).or_else(|| extract_anthropic_user_text(&v))
                }
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = is_event_stream(meta.content_type.as_deref());

        loop {
            let msg = match rx.recv().await {
                Ok(m) => m,
                Err(_) => {
                    // channel closed mid-exchange
                    break;
                }
            };

            match msg {
                AnalysisMsg::ResponseEnd => break,
                AnalysisMsg::ExchangeStart { meta: next_meta, request_body: next_req } => {
                    pending_start = Some((next_meta, next_req));
                    break;
                }
                AnalysisMsg::ResponseChunk(b) => {
                    if sse {
                        sse_buf.extend_from_slice(&b);
                        sse_extract_deltas(&mut sse_buf, &mut assistant_text);
                    } else {
                        raw_buf.extend_from_slice(&b);
                    }

                    // Hard bound: we do not want to hold huge transient buffers.
                    if assistant_text.len() > 512 * 1024 {
                        assistant_text.truncate(512 * 1024);
                    }
                    if raw_buf.len() > 2 * 1024 * 1024 {
                        raw_buf.truncate(2 * 1024 * 1024);
                    }
                }
            }
        }

        if !sse {
            if let Some(v) = decode_json_lossy(&raw_buf) {
                if meta.request_path.contains("/v1/messages") {
                    assistant_text = extract_anthropic_assistant_text_from_json(&v)
                        .or_else(|| extract_openai_assistant_text_from_json(&v))
                        .unwrap_or_default();
                } else {
                    assistant_text = extract_openai_assistant_text_from_json(&v)
                        .or_else(|| extract_anthropic_assistant_text_from_json(&v))
                        .unwrap_or_default();
                }
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() {
            continue;
        }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() {
            input.push_str("User:\n");
            input.push_str(u);
            input.push_str("\n\n");
        }
        if !assistant_text.is_empty() {
            input.push_str("Assistant:\n");
            input.push_str(&assistant_text);
        }

        let commit_mentioned = looks_like_commit_or_pr(&input);
        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input,
            commit_mentioned,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }

    debug!(conn_id, "analysis worker finished");
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
pub struct IntentCapsule {
    pub category: String,
    pub intent: String,
    pub decision: String,
    pub rationale: String,
    pub next_steps: Vec<String>,
    pub symbols: Vec<String>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
struct InitCapsulesOutput {
    /// Short, colleague-style debrief to print to the user.
    debrief: String,
    capsules: Vec<IntentCapsule>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
struct QueryNarrativeOutput {
    narrative: String,
}

#[derive(Debug, Clone)]
struct CapsuleHit {
    id: String,
    ts_ms: i64,
    distance: f32,
    capsule: IntentCapsule,
    meta: ResponseMeta,
}

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
    // Mirrors unfault's workspace scanning defaults, plus a couple of obvious extras.
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
    // Use ignore crate to respect .gitignore/.dockerignore and global ignores.
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
        let Some(lang) = detect_language(path) else { continue };

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

async fn query_capsules_lancedb(
    query_text: &str,
    limit: usize,
    symbol: Option<&str>,
    embedder: Embedder,
    ws: &WorkspacePaths,
) -> anyhow::Result<Vec<CapsuleHit>> {
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let table = match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => t,
        Err(_) => {
            println!("No LanceDB table found for workspace {}.", ws.id);
            return Ok(vec![]);
        }
    };

    let q_embedding = embed_text(&embedder, query_text).await?;

    let mut q = table
        .query()
        .nearest_to(q_embedding.as_slice())?
        .column("embedding")
        .limit(limit);

    if let Some(sym) = symbol {
        let sym = escape_sql_string(sym);
        // lance-datafusion suggests `array_contains` for list columns.
        q = q.only_if(format!("array_contains(symbols, '{sym}')"));
    }

    let batches = q.execute().await?.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let mut out: Vec<CapsuleHit> = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let idx = |name: &str| schema.index_of(name).ok();
        let col_str = |name: &str| -> Option<&StringArray> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
        };

        let id_col = col_str("id");
        let ts_ms_col = idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());

        let source = col_str("source");
        let intent = col_str("intent");
        let decision = col_str("decision");
        let rationale = col_str("rationale");
        let category = col_str("category");
        let upstream_host = col_str("upstream_host");
        let request_path = col_str("request_path");

        let distance = idx("_distance")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<arrow_array::Float32Array>());

        let next_steps = idx("next_steps")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());

        let symbols = idx("symbols")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());

        for row in 0..batch.num_rows() {
            if out.len() >= limit {
                break;
            }

            let dist = distance
                .and_then(|d| (!d.is_null(row)).then(|| d.value(row)))
                .unwrap_or_default();
            let id = id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let ts_ms = ts_ms_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let cat = category
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let src = source
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let up = upstream_host
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let path = request_path
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let i_text = intent
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let d_text = decision
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let r_text = rationale
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");

            let mut syms: Vec<String> = Vec::new();
            if let Some(sym_arr) = symbols {
                if !sym_arr.is_null(row) {
                    let values = sym_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        syms = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            let mut steps: Vec<String> = Vec::new();
            if let Some(ns_arr) = next_steps {
                if !ns_arr.is_null(row) {
                    let values = ns_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        steps = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            out.push(CapsuleHit {
                id: id.to_string(),
                ts_ms,
                distance: dist,
                capsule: IntentCapsule {
                    category: cat.to_string(),
                    intent: i_text.to_string(),
                    decision: d_text.to_string(),
                    rationale: r_text.to_string(),
                    next_steps: steps,
                    symbols: syms,
                },
                meta: ResponseMeta {
                    source: src.to_string(),
                    upstream_host: up.to_string(),
                    request_path: path.to_string(),
                    http_status: 0,
                },
            });
        }
    }

    Ok(out)
}

async fn scan_capsules_lancedb(
    ws: &WorkspacePaths,
    limit: usize,
    filter: Option<&str>,
) -> anyhow::Result<Vec<CapsuleHit>> {
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => t,
        Err(_) => return Ok(vec![]),
    };

    let mut q = table.query().limit(limit);
    if let Some(f) = filter {
        q = q.only_if(f.to_string());
    }

    let batches = q.execute().await?.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return Ok(vec![]);
    }

    // Reuse the same extraction shape as query results; distance will be 0.
    let mut out: Vec<CapsuleHit> = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let idx = |name: &str| schema.index_of(name).ok();
        let col_str = |name: &str| -> Option<&StringArray> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
        };

        let id_col = col_str("id");
        let ts_ms_col = idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());

        let source = col_str("source");
        let intent = col_str("intent");
        let decision = col_str("decision");
        let rationale = col_str("rationale");
        let category = col_str("category");
        let upstream_host = col_str("upstream_host");
        let request_path = col_str("request_path");
        let next_steps = idx("next_steps")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());

        let symbols = idx("symbols")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());

        for row in 0..batch.num_rows() {
            if out.len() >= limit {
                break;
            }
            let cat = category
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let up = upstream_host
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let path = request_path
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let src = source
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let id = id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let ts_ms = ts_ms_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let i_text = intent
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let d_text = decision
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let r_text = rationale
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");

            let mut syms: Vec<String> = Vec::new();
            if let Some(sym_arr) = symbols {
                if !sym_arr.is_null(row) {
                    let values = sym_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        syms = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            let mut steps: Vec<String> = Vec::new();
            if let Some(ns_arr) = next_steps {
                if !ns_arr.is_null(row) {
                    let values = ns_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        steps = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            out.push(CapsuleHit {
                id: id.to_string(),
                ts_ms,
                distance: 0.0,
                capsule: IntentCapsule {
                    category: cat.to_string(),
                    intent: i_text.to_string(),
                    decision: d_text.to_string(),
                    rationale: r_text.to_string(),
                    next_steps: steps,
                    symbols: syms,
                },
                meta: ResponseMeta {
                    source: src.to_string(),
                    upstream_host: up.to_string(),
                    request_path: path.to_string(),
                    http_status: 0,
                },
            });
        }
    }

    Ok(out)
}

async fn llm_query_narrative(
    llm_model_override: Option<&str>,
    query_text: &str,
    symbol: Option<&str>,
    matches: &[CapsuleHit],
) -> anyhow::Result<String> {
    let mut context = String::new();
    context.push_str("Query:\n");
    context.push_str(query_text);
    context.push('\n');
    if let Some(sym) = symbol {
        context.push_str("Symbol filter: ");
        context.push_str(sym);
        context.push('\n');
    }
    context.push_str("Matches (lower distance = closer):\n");
    for (i, hit) in matches.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        context.push_str(&format!(
            "#{} distance={} source={} category={} upstream={} path={}\n",
            i + 1,
            hit.distance,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path
        ));
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
            let syms = cap.symbols.iter().take(12).cloned().collect::<Vec<_>>().join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
        }
        context.push('\n');
        if i >= 9 {
            break;
        }
    }

    let preamble = r#"You are unlost. Talk like a teammate discussing the codebase with the user.

Grounding rules:
- Base your answer ONLY on the provided matches. Don't invent files, symbols, routes, frameworks, or auth mechanisms.
- When you make a claim, anchor it to concrete evidence by mentioning 1-3 specific backticked tokens pulled from the matches (paths, symbols, or routes).

Clarity rules:
- The FIRST sentence must be an explicit verdict: "Yes", "No", or "I don't know yet".
- If you say "I don't know yet", immediately say what is missing in one sentence.

Style rules:
- First person, conversational, concise: 4-6 sentences.
- No headings, no bullets, no "report" language.
- Never output internal/system/tool boilerplate (e.g. anything like `<system-reminder>...</system-reminder>`).
- Wrap code identifiers in backticks (e.g. `proxy_request`), file paths in backticks (e.g. `src/main.rs`, `main.py`), and routes in backticks (e.g. `GET /inventory`).
- End with ONE actionable next step, phrased as a concrete `unlost query ...` suggestion (not grep/file search)."#;

    let mut out = llm_extract::<QueryNarrativeOutput>(llm_model_override, preamble, &context)
        .await?
        .narrative;

    // Defensive: if the LLM includes any leaked system/tool boilerplate, strip it.
    // We do a case-insensitive search for common tag prefixes.
    let lower = out.to_ascii_lowercase();
    if let Some(i) = lower
        .find("<system-reminder")
        .or_else(|| lower.find("<system"))
        .or_else(|| lower.find("<commentary"))
        .or_else(|| lower.find("<tool"))
    {
        out.truncate(i);
    }
    Ok(out)
}

fn colorize_backticks(input: &str) -> String {
    // Very small, dependency-free ANSI highlighting pass.
    // - `GET /foo` etc -> yellow
    // - `src/main.rs` or `main.py` -> green
    // - everything else -> cyan
    let methods = [
        "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "CONNECT", "TRACE",
    ];
    let exts = [
        ".rs", ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".toml", ".json",
        ".yaml", ".yml", ".md",
    ];

    let mut out = String::with_capacity(input.len() + 32);
    let mut in_tick = false;
    let mut buf = String::new();

    for ch in input.chars() {
        if ch == '`' {
            if in_tick {
                let t = buf.trim();
                let is_route = methods
                    .iter()
                    .any(|m| t.starts_with(m) && t.contains(" /"));
                let is_path = t.contains('/') || exts.iter().any(|e| t.ends_with(e));

                let color = if is_route {
                    "\x1b[33m" // yellow
                } else if is_path {
                    "\x1b[32m" // green
                } else {
                    "\x1b[36m" // cyan
                };

                out.push('`');
                out.push_str(color);
                out.push_str(t);
                out.push_str("\x1b[0m");
                out.push('`');

                buf.clear();
                in_tick = false;
            } else {
                in_tick = true;
            }
            continue;
        }

        if in_tick {
            buf.push(ch);
        } else {
            out.push(ch);
        }
    }

    // Unbalanced backtick: just append it back.
    if in_tick {
        out.push('`');
        out.push_str(&buf);
    }

    out
}

fn scope_filter_expr(scope: &str) -> Option<String> {
    let s = scope.trim();
    if s.is_empty() {
        return None;
    }
    let s = escape_sql_string(s);
    Some(format!("array_contains(symbols, '{s}')"))
}

fn strip_llm_boilerplate(mut s: String) -> String {
    // Defensive: if the LLM includes any leaked system/tool boilerplate, strip it.
    // Do a case-insensitive search for common tag prefixes.
    let lower = s.to_ascii_lowercase();
    if let Some(i) = lower
        .find("<system-reminder")
        .or_else(|| lower.find("<system"))
        .or_else(|| lower.find("<commentary"))
        .or_else(|| lower.find("<tool"))
    {
        s.truncate(i);
    }
    s
}

fn render_narrative(output: OutputFormat, s: &str) -> String {
    let output = if std::env::var_os("NO_COLOR").is_some() {
        OutputFormat::Plain
    } else {
        output
    };

    let s = strip_llm_boilerplate(s.trim().to_string());

    match output {
        OutputFormat::Plain => s.trim().to_string(),
        OutputFormat::Ansi => {
            // Dim “tips” lines so they read as guidance, not facts.
            // We intentionally skip backtick-coloring inside dimmed lines, so dim stays consistent.
            let mut out = String::with_capacity(s.len() + 32);
            for (i, line) in s.lines().enumerate() {
                if i > 0 {
                    out.push('\n');
                }

                let l = line.trim_end();
                let lower = l.to_ascii_lowercase();
                let is_tip = lower.starts_with("evidence note:")
                    || lower.starts_with("follow-up query:")
                    || lower.starts_with("follow up query:")
                    || lower.starts_with("next step:");

                if is_tip {
                    out.push_str("\x1b[2m");
                    out.push_str(l);
                    out.push_str("\x1b[0m");
                } else {
                    out.push_str(&colorize_backticks(l));
                }
            }
            out
        }
    }
}

async fn llm_recall_narrative(
    llm_model_override: Option<&str>,
    scope: Option<&str>,
    hits: &[CapsuleHit],
) -> anyhow::Result<String> {
    let mut context = String::new();
    context.push_str("Recall context\n\n");
    if let Some(s) = scope {
        context.push_str("Scope:\n");
        context.push_str(s);
        context.push_str("\n\n");
    } else {
        context.push_str("Scope:\n<workspace>\n\n");
    }
    context.push_str("Capsules (most recent first):\n");
    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        context.push_str(&format!(
            "#{} ts_ms={} source={} category={} upstream={} path={}\n",
            i + 1,
            hit.ts_ms,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path
        ));
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!("rationale: {}\n", cap.rationale.replace('\n', " ")));
        }
        if !cap.next_steps.is_empty() {
            let steps = cap.next_steps.iter().take(3).cloned().collect::<Vec<_>>().join(" | ");
            context.push_str(&format!("next: {steps}\n"));
        }
        if !cap.symbols.is_empty() {
            let syms = cap.symbols.iter().take(16).cloned().collect::<Vec<_>>().join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
        }
        context.push('\n');
        if i >= 39 {
            break;
        }
    }

    let preamble = r#"You are unlost recall. Your job is to proactively reconstruct the story so far.

Rules:
- Base your output ONLY on the provided capsules.
- Do NOT quote or excerpt the conversation.
- If scoped (a file path or symbol), focus on that scope but explicitly call out cross-scope impacts: any important symbols or files outside the scope that appear connected.
- Keep it high-signal: intent, decisions, rationale, and what's next.

Output format:
- 2-3 sentences: overall state of the work.
- Then 3-6 short bullets: key decisions (with 1-2 backticked tokens each).
- Then 2-4 short bullets: suggested next steps (as actions).
- If the evidence is thin, say so plainly and recommend ONE follow-up `unlost query ...`.
"#;

    Ok(llm_extract::<QueryNarrativeOutput>(llm_model_override, preamble, &context)
        .await?
        .narrative)
}

fn query_spinner_enabled(output: OutputFormat) -> bool {
    if output != OutputFormat::Ansi {
        return false;
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    std::io::stdout().is_terminal()
}

fn capsule_embed_text(c: &IntentCapsule) -> String {
    let mut s = String::new();
    if !c.intent.trim().is_empty() {
        s.push_str("intent: ");
        s.push_str(c.intent.trim());
        s.push('\n');
    }
    if !c.decision.trim().is_empty() {
        s.push_str("decision: ");
        s.push_str(c.decision.trim());
        s.push('\n');
    }
    if !c.rationale.trim().is_empty() {
        s.push_str("rationale: ");
        s.push_str(c.rationale.trim());
        s.push('\n');
    }
    s
}

async fn insert_capsule_row(
    db: &Connection,
    embedder: &Embedder,
    conn_id: u64,
    exchange_seq: u64,
    ts_ms: i64,
    meta: &ResponseMeta,
    capsule: &IntentCapsule,
) -> anyhow::Result<()> {
    debug!(
        conn_id,
        exchange_seq,
        decision_bytes = capsule.decision.len(),
        symbols = capsule.symbols.len(),
        "inserting capsule"
    );
    let table = ensure_capsules_table(db).await?;
    let schema = capsules_schema();

    let embedding = embed_text(embedder, &capsule_embed_text(capsule)).await?;
    if embedding.len() != 384 {
        anyhow::bail!("embedding dimension mismatch: {}", embedding.len());
    }

    let id = Uuid::new_v4().to_string();

    let id_arr = Arc::new(StringArray::from(vec![id.as_str()]));
    let ts_ms_arr = Arc::new(Int64Array::from(vec![ts_ms]));
    let source_arr = Arc::new(StringArray::from(vec![meta.source.as_str()]));
    let upstream_host_arr = Arc::new(StringArray::from(vec![meta.upstream_host.as_str()]));
    let request_path_arr = Arc::new(StringArray::from(vec![meta.request_path.as_str()]));
    let http_status_arr = Arc::new(Int32Array::from(vec![meta.http_status as i32]));
    let conn_id_arr = Arc::new(Int64Array::from(vec![conn_id as i64]));
    let exchange_seq_arr = Arc::new(Int64Array::from(vec![exchange_seq as i64]));
    let category_arr = Arc::new(StringArray::from(vec![capsule.category.as_str()]));
    let intent_arr = Arc::new(StringArray::from(vec![capsule.intent.as_str()]));
    let decision_arr = Arc::new(StringArray::from(vec![capsule.decision.as_str()]));
    let rationale_arr = Arc::new(StringArray::from(vec![capsule.rationale.as_str()]));

    let mut next_steps_builder = ListBuilder::new(StringBuilder::new());
    for step in &capsule.next_steps {
        next_steps_builder.values().append_value(step);
    }
    next_steps_builder.append(true);
    let next_steps_arr = Arc::new(next_steps_builder.finish());

    let mut symbols_builder = ListBuilder::new(StringBuilder::new());
    for sym in &capsule.symbols {
        symbols_builder.values().append_value(sym);
    }
    symbols_builder.append(true);
    let symbols_arr = Arc::new(symbols_builder.finish());

    let embedding_arr = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        std::iter::once(Some(
            embedding.into_iter().map(Some).collect::<Vec<_>>(),
        )),
        384,
    ));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            id_arr,
            ts_ms_arr,
            source_arr,
            upstream_host_arr,
            request_path_arr,
            http_status_arr,
            conn_id_arr,
            exchange_seq_arr,
            category_arr,
            intent_arr,
            decision_arr,
            rationale_arr,
            next_steps_arr,
            symbols_arr,
            embedding_arr,
        ],
    )
    .context("failed to build insert batch")?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    table.add(batches).execute().await.context("lancedb insert failed")?;
    Ok(())
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
) -> anyhow::Result<(String, Vec<IntentCapsule>)> {
    // Keep this tightly bounded: small prompt, small output.
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

    let InitCapsulesOutput { debrief, mut capsules } = llm_extract::<InitCapsulesOutput>(
        llm_model_override,
        preamble,
        &format!("max_capsules: {llm_max_capsules}\n\n{context}"),
    )
    .await?;

    // Enforce hard bound.
    if capsules.len() > llm_max_capsules {
        capsules.truncate(llm_max_capsules);
    }

    Ok((debrief, capsules))
}

async fn run_init(
    root: &str,
    embed_model: &str,
    embed_cache_dir: Option<&str>,
    no_llm: bool,
    git_history: bool,
    git_commits: usize,
    llm_model: Option<&str>,
    llm_max_capsules: usize,
    max_capsules: usize,
    ws: WorkspacePaths,
) -> anyhow::Result<()> {
    let root_path = std::path::Path::new(root);
    let files = collect_source_files(root_path)?;
    if files.is_empty() {
        anyhow::bail!("no supported source files found under {root}");
    }

    let file_paths = files.iter().map(|sf| sf.path.clone()).collect::<Vec<_>>();

    let embedder = load_embedder(
        embed_model,
        embed_cache_dir.map(std::path::PathBuf::from),
        true,
    )
    .await?;

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = ensure_capsules_table(&db).await?;

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

    // Quick overview for the user
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

    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64;

    let git_history_summary = if git_history {
        collect_git_history_summary(root_path, git_commits).ok().flatten()
    } else {
        None
    };

    let mut capsules: Vec<(IntentCapsule, ResponseMeta)> = Vec::new();

    // Project capsule
    capsules.push((
        IntentCapsule {
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
            symbols: vec![CAPSULES_TABLE.to_string()],
        },
        ResponseMeta {
            source: "init".to_string(),
            upstream_host: "init".to_string(),
            request_path: "init".to_string(),
            http_status: 0,
        },
    ));

    // Route handler capsules (if any)
    for (idx, path, method) in cg.get_http_route_handlers() {
        if capsules.len() >= max_capsules {
            break;
        }

        let node = &cg.graph[idx];
        let (qualified, file_id) = match node {
            unfault_core::GraphNode::Function { qualified_name, file_id, .. } => (qualified_name.clone(), *file_id),
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
            IntentCapsule {
                category: "Snapshot:Route".to_string(),
                intent: "Identify the request surface area".to_string(),
                decision: format!("HTTP handler {method} {path} implemented by {qualified} ({file_path})."),
                rationale: String::new(),
                next_steps: Vec::new(),
                symbols: vec![qualified.clone(), file_path.clone(), format!("{method} {path}")],
            },
            ResponseMeta {
                source: "init".to_string(),
                upstream_host: "init".to_string(),
                request_path: file_path,
                http_status: 0,
            },
        ));
    }

    // High fan-in functions (call hotspots)
    let mut funcs: Vec<(usize, String, String)> = Vec::new();
    for idx in cg.graph.node_indices() {
        if let unfault_core::GraphNode::Function { qualified_name, file_id, .. } = &cg.graph[idx] {
            let callers = cg
                .graph
                .edges_directed(idx, petgraph::Direction::Incoming)
                .filter(|e| matches!(e.weight(), unfault_core::GraphEdgeKind::Calls))
                .count();
            if callers == 0 {
                continue;
            }

            let file_path = match cg.file_nodes.get(file_id).copied() {
                Some(fidx) => match &cg.graph[fidx] {
                    unfault_core::GraphNode::File { path, .. } => path.clone(),
                    _ => "".to_string(),
                },
                None => "".to_string(),
            };
            funcs.push((callers, qualified_name.clone(), file_path));
        }
    }
    funcs.sort_by(|a, b| b.0.cmp(&a.0));

    let top_hotspots: Vec<(usize, String)> = funcs
        .iter()
        .take(25)
        .map(|(callers, qualified, _file_path)| (*callers, qualified.clone()))
        .collect();

    // Dependency frequency (best-effort)
    let mut dep_counts: HashMap<String, usize> = HashMap::new();
    for edge in cg.graph.edge_references() {
        if !matches!(edge.weight(), unfault_core::GraphEdgeKind::UsesLibrary) {
            continue;
        }
        if let unfault_core::GraphNode::ExternalModule { name, .. } = &cg.graph[edge.target()] {
            *dep_counts.entry(name.clone()).or_insert(0) += 1;
        }
    }
    let mut deps: Vec<(usize, String)> = dep_counts.into_iter().map(|(n, c)| (c, n)).collect();
    deps.sort_by(|a, b| b.0.cmp(&a.0));
    if deps.len() > 25 {
        deps.truncate(25);
    }

    // Optional LLM summaries (bounded)
    let mut llm_debrief: Option<String> = None;
    if !no_llm {
        let llm_limit = llm_max_capsules.min(max_capsules.saturating_sub(capsules.len()));
        if llm_limit > 0 {
            info!(llm_model = ?llm_model, llm_limit, "generating init summaries via LLM");
            match llm_init_capsules(
                llm_model,
                llm_limit,
                root_path,
                &stats,
                &top_routes,
                &top_hotspots,
                &deps,
                &file_paths,
                git_history_summary.as_deref(),
            )
            .await
            {
                Ok((debrief, llm_caps)) => {
                    llm_debrief = Some(debrief);
                    // Insert right after the project capsule, keep original ordering.
                    let mut i = 1usize;
                    for c in llm_caps {
                        capsules.insert(
                            i,
                            (
                                c,
                                ResponseMeta {
                                    source: "init".to_string(),
                                    upstream_host: "init".to_string(),
                                    request_path: "init".to_string(),
                                    http_status: 0,
                                },
                            ),
                        );
                        i += 1;
                    }
                }
                Err(e) => {
                    warn!(error = ?e, "init LLM summaries failed; continuing with structural init only");
                }
            }
        }
    }

    // If we have git history and no LLM, still seed one lightweight capsule.
    if no_llm {
        if let Some(gh) = git_history_summary.as_deref() {
            if capsules.len() < max_capsules {
                let preview = gh.lines().take(12).collect::<Vec<_>>().join("\n");
                capsules.push((
                    IntentCapsule {
                        category: "Snapshot:History".to_string(),
                        intent: "Capture recent evolution signals".to_string(),
                        decision: preview,
                        rationale: String::new(),
                        next_steps: Vec::new(),
                        symbols: vec!["git".to_string()],
                    },
                    ResponseMeta {
                        source: "init".to_string(),
                        upstream_host: "init".to_string(),
                        request_path: "init".to_string(),
                        http_status: 0,
                    },
                ));
            }
        }
    }

    for (callers, qualified, file_path) in funcs.into_iter().take(60) {
        if capsules.len() >= max_capsules {
            break;
        }
        capsules.push((
            IntentCapsule {
                category: "Snapshot:Hotspot".to_string(),
                intent: "Highlight high fan-in code paths".to_string(),
                decision: format!("{qualified} is called by {callers} functions (hotspot)."),
                rationale: String::new(),
                next_steps: Vec::new(),
                symbols: vec![qualified.clone(), file_path.clone()],
            },
            ResponseMeta {
                source: "init".to_string(),
                upstream_host: "init".to_string(),
                request_path: file_path,
                http_status: 0,
            },
        ));
    }

    // Insert
    let mut seq: u64 = 1;
    for (capsule, meta) in capsules {
        if seq as usize > max_capsules {
            break;
        }
        if seq == 1 || seq % 25 == 0 {
            info!(seq, max_capsules, "init seeding progress");
        }
        insert_capsule_row(&db, &embedder, 0, seq, now_ms, &meta, &capsule).await?;
        seq += 1;
    }

    // Use deps already computed above (trim for printing)
    let mut deps_print = deps.clone();
    if deps_print.len() > 5 {
        deps_print.truncate(5);
    }

    println!("\nunlost init complete");
    if let Some(debrief) = llm_debrief {
        println!("\n{debrief}\n");
    }
    println!("stored {} capsules for workspace {}", (seq - 1), ws.id);

    // Keep some lightweight context (not a report).
    println!(
        "graph: {} files, {} functions, {} call edges",
        stats.file_count,
        stats.function_count,
        stats.calls_edge_count
    );
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpstreamProvider {
    Openai,
    Anthropic,
    Opencode,
}

impl UpstreamProvider {
    fn from_path_segment(s: &str) -> Option<Self> {
        match s {
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            "opencode" => Some(Self::Opencode),
            _ => None,
        }
    }

    fn upstream_host(self) -> &'static str {
        match self {
            Self::Openai => "api.openai.com",
            Self::Anthropic => "api.anthropic.com",
            Self::Opencode => "opencode.ai",
        }
    }
}

fn parse_multiplexed_uri(uri: &Uri) -> anyhow::Result<(String, UpstreamProvider, String)> {
    let path = uri.path();
    let segs = path.split('/').filter(|s| !s.is_empty()).collect::<Vec<_>>();
    if segs.len() < 3 || segs[0] != "w" {
        anyhow::bail!("expected path /w/<workspace_id>/<provider>/... (got {path})");
    }
    let workspace_id = segs[1].to_string();
    let provider = UpstreamProvider::from_path_segment(segs[2])
        .ok_or_else(|| anyhow::anyhow!("unknown provider '{}'", segs[2]))?;

    let rest = &segs[3..];
    let mut new_path = String::from("/");
    if !rest.is_empty() {
        new_path.push_str(&rest.join("/"));
    }
    if let Some(q) = uri.query() {
        new_path.push('?');
        new_path.push_str(q);
    }

    Ok((workspace_id, provider, new_path))
}

#[derive(Clone)]
struct ServeState {
    embedder: Embedder,
    db_cache: Arc<Mutex<HashMap<String, Connection>>>,
}

impl ServeState {
    fn new(embedder: Embedder) -> Self {
        Self {
            embedder,
            db_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn workspace_paths(&self, workspace_id: &str) -> WorkspacePaths {
        let ws_dir = unlost_workspace_dir(workspace_id);
        WorkspacePaths {
            id: workspace_id.to_string(),
            db_dir: ws_dir.join("lancedb"),
            capsules_jsonl: ws_dir.join("capsules.jsonl"),
        }
    }

    async fn db_for(&self, workspace_id: &str) -> anyhow::Result<Connection> {
        {
            let cache = self.db_cache.lock().await;
            if let Some(c) = cache.get(workspace_id) {
                return Ok(c.clone());
            }
        }

        let ws = self.workspace_paths(workspace_id);
        std::fs::create_dir_all(&ws.db_dir)?;
        let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
            .execute()
            .await?;
        let _ = ensure_capsules_table(&db).await?;

        let mut cache = self.db_cache.lock().await;
        cache.insert(workspace_id.to_string(), db.clone());
        Ok(db)
    }
}

async fn process_flush_jobs_serve(rx: AsyncReceiver<FlushJob>, state: ServeState) {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON only with fields: {category, intent, decision, rationale, next_steps (array), symbols (array)}.\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.";

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let capsule = match llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(c) => c,
            Err(e) => {
                warn!(workspace_id = %job.workspace_id, conn_id = job.conn_id, exchange_seq = job.exchange_seq, error = ?e, "capsule extraction failed");
                continue;
            }
        };

        let ws_paths = state.workspace_paths(&job.workspace_id);
        if let Err(e) = append_capsule_jsonl(
            &ws_paths.capsules_jsonl,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to append capsule jsonl");
        }

        match state.db_for(&job.workspace_id).await {
            Ok(db) => {
                if let Err(e) = insert_capsule_row(
                    &db,
                    &state.embedder,
                    job.conn_id,
                    job.exchange_seq,
                    job.ts_ms,
                    &job.meta,
                    &capsule,
                )
                .await
                {
                    warn!(workspace_id = %job.workspace_id, error = ?e, "failed to insert capsule into lancedb");
                }
            }
            Err(e) => {
                warn!(workspace_id = %job.workspace_id, error = ?e, "failed to open workspace db");
            }
        }
    }
}

async fn process_flush_jobs_proxy(
    rx: AsyncReceiver<FlushJob>,
    ws: WorkspacePaths,
    db: Connection,
    embedder: Embedder,
) {
    const PREAMBLE: &str = "You are unlost. Extract a short, high-signal intent capsule from this multi-turn conversation slice.\n\
Return JSON only with fields: {category, intent, decision, rationale, next_steps (array), symbols (array)}.\n\
Rules:\n\
- Do NOT include quotes or excerpts from the conversation. No evidence snippets.\n\
- Keep it grounded in what happened: intent, decisions, rationale, and what's next.\n\
- Keep each field concise; next_steps max 3.\n\
- symbols: identifiers, file paths, endpoints, commit/PR refs if explicitly mentioned.";

    loop {
        let job = match rx.recv().await {
            Ok(j) => j,
            Err(_) => break,
        };

        let capsule = match llm_extract::<IntentCapsule>(None, PREAMBLE, &job.input).await {
            Ok(c) => c,
            Err(e) => {
                warn!(workspace_id = %job.workspace_id, conn_id = job.conn_id, exchange_seq = job.exchange_seq, error = ?e, "capsule extraction failed");
                continue;
            }
        };

        if let Err(e) = append_capsule_jsonl(
            &ws.capsules_jsonl,
            job.ts_ms,
            job.conn_id,
            job.exchange_seq,
            &job.meta,
            &capsule,
        ) {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to append capsule jsonl");
        }

        if let Err(e) = insert_capsule_row(
            &db,
            &embedder,
            job.conn_id,
            job.exchange_seq,
            job.ts_ms,
            &job.meta,
            &capsule,
        )
        .await
        {
            warn!(workspace_id = %job.workspace_id, error = ?e, "failed to insert capsule into lancedb");
        }
    }
}

async fn analysis_worker_multiplex(rx: AsyncReceiver<AnalysisMsg>, chunker: WorkspaceChunker, conn_id: u64) {
    debug!(conn_id, "analysis worker started");
    let mut pending_start: Option<(AnalysisMeta, Bytes)> = None;

    loop {
        let (meta, request_body) = if let Some(p) = pending_start.take() {
            p
        } else {
            match rx.recv().await {
                Ok(AnalysisMsg::ExchangeStart { meta, request_body }) => (meta, request_body),
                Ok(_) => continue,
                Err(_) => break,
            }
        };

        let user_text = decode_json_lossy(&request_body)
            .and_then(|v| {
                if meta.request_path.contains("/v1/messages") {
                    extract_anthropic_user_text(&v).or_else(|| extract_openai_message_text(&v))
                } else {
                    extract_openai_message_text(&v).or_else(|| extract_anthropic_user_text(&v))
                }
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let mut assistant_text = String::new();
        let mut sse_buf: Vec<u8> = Vec::new();
        let mut raw_buf: Vec<u8> = Vec::new();
        let sse = is_event_stream(meta.content_type.as_deref());

        loop {
            let msg = match rx.recv().await {
                Ok(m) => m,
                Err(_) => break,
            };

            match msg {
                AnalysisMsg::ResponseEnd => break,
                AnalysisMsg::ExchangeStart { meta: next_meta, request_body: next_req } => {
                    pending_start = Some((next_meta, next_req));
                    break;
                }
                AnalysisMsg::ResponseChunk(b) => {
                    if sse {
                        sse_buf.extend_from_slice(&b);
                        sse_extract_deltas(&mut sse_buf, &mut assistant_text);
                    } else {
                        raw_buf.extend_from_slice(&b);
                    }

                    if assistant_text.len() > 512 * 1024 {
                        assistant_text.truncate(512 * 1024);
                    }
                    if raw_buf.len() > 2 * 1024 * 1024 {
                        raw_buf.truncate(2 * 1024 * 1024);
                    }
                }
            }
        }

        if !sse {
            if let Some(v) = decode_json_lossy(&raw_buf) {
                if meta.request_path.contains("/v1/messages") {
                    assistant_text = extract_anthropic_assistant_text_from_json(&v)
                        .or_else(|| extract_openai_assistant_text_from_json(&v))
                        .unwrap_or_default();
                } else {
                    assistant_text = extract_openai_assistant_text_from_json(&v)
                        .or_else(|| extract_anthropic_assistant_text_from_json(&v))
                        .unwrap_or_default();
                }
            }
        }

        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_none() && assistant_text.is_empty() {
            continue;
        }

        let mut input = String::new();
        if let Some(u) = user_text.as_deref() {
            input.push_str("User:\n");
            input.push_str(u);
            input.push_str("\n\n");
        }
        if !assistant_text.is_empty() {
            input.push_str("Assistant:\n");
            input.push_str(&assistant_text);
        }

        let commit_mentioned = looks_like_commit_or_pr(&input);
        let item = ChunkInput {
            conn_id,
            upstream_host: meta.upstream_host.clone(),
            request_path: meta.request_path.clone(),
            http_status: meta.http_status,
            exchange_text: input,
            commit_mentioned,
        };
        chunker.ingest(meta.workspace_id.clone(), item).await;
    }

    debug!(conn_id, "analysis worker finished");
}

async fn proxy_request(
    client: Client<hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>, BoxBody<Bytes, hyper::Error>>,
    workspace_id: String,
    upstream_host: String,
    upstream_port: u16,
    conn_id: u64,
    analysis_tx: kanal::Sender<AnalysisMsg>,
    analysis_drops_logged: Arc<AtomicBool>,
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
    let upstream_uri = match build_upstream_uri(&upstream_host, upstream_port, req.uri()) {
        Ok(u) => u,
        Err(e) => {
            warn!(conn_id, error = ?e, "bad request URI");
            let resp = Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(text_body(b"bad request"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (parts, body) = req.into_parts();
    let request_path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_string();
    let method = parts.method.clone();
    let version = parts.version;
    let headers = sanitize_request_headers(parts.headers, &upstream_host);

    // Buffer request body so we can extract user text (no transcript storage).
    // This is intentionally bounded.
    const MAX_REQ_BODY: usize = 2 * 1024 * 1024;
    let req_body_bytes = match read_incoming_body_limited(body, MAX_REQ_BODY).await {
        Ok(b) => b,
        Err(e) => {
            warn!(conn_id, error = ?e, "request body capture failed");
            let resp = Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(text_body(b"payload too large"))
                .unwrap();
            return Ok(resp);
        }
    };

    let mut out_req = Request::builder()
        .method(method)
        .uri(upstream_uri)
        .version(version)
        .body(bytes_body(req_body_bytes.clone()))
        .unwrap();
    *out_req.headers_mut() = headers;

    let res = match client.request(out_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(conn_id, error = ?e, "upstream request failed");
            let resp = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(text_body(b"bad gateway"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (res_parts, res_body) = res.into_parts();
    let res_headers = sanitize_response_headers(res_parts.headers);

    let content_type = res_headers
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Notify analyzer of a new exchange. No headers, no SSE framing, just structured text extraction.
    let meta = AnalysisMeta {
        workspace_id,
        upstream_host: upstream_host.clone(),
        request_path: request_path.clone(),
        http_status: res_parts.status.as_u16(),
        content_type,
    };
    let _ = analysis_tx.try_send(AnalysisMsg::ExchangeStart {
        meta,
        request_body: req_body_bytes,
    });

    let tx = analysis_tx;
    let drops_logged = analysis_drops_logged;
    let stream = futures_util::stream::unfold(
        (BodyStream::new(res_body), tx, drops_logged, false),
        move |(mut bs, tx, drops_logged, ended)| async move {
            if ended {
                return None;
            }

            match bs.try_next().await {
                Ok(Some(frame)) => {
                    if let Some(data) = frame.data_ref() {
                        match tx.try_send(AnalysisMsg::ResponseChunk(data.clone())) {
                            Ok(true) => {}
                            Ok(false) => {
                                if !drops_logged.swap(true, Ordering::Relaxed) {
                                    info!(conn_id, "analysis channel full; dropping response bytes");
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    Some((Ok(frame), (bs, tx, drops_logged, false)))
                }
                Ok(None) => {
                    let _ = tx.try_send(AnalysisMsg::ResponseEnd);
                    None
                }
                Err(e) => {
                    let _ = tx.try_send(AnalysisMsg::ResponseEnd);
                    Some((Err(e), (bs, tx, drops_logged, true)))
                }
            }
        },
    );

    let out_body = StreamBody::new(stream).boxed();
    let mut out_res = Response::new(out_body);
    *out_res.status_mut() = res_parts.status;
    *out_res.version_mut() = res_parts.version;
    *out_res.headers_mut() = res_headers;

    Ok(out_res)
}

async fn run_serve(bind: SocketAddr, embedder: Embedder) -> anyhow::Result<()> {
    let https = HttpsConnectorBuilder::new()
        .with_native_roots()
        .context("no native root CA certificates found")?
        .https_only()
        .enable_http1()
        .build();

    let client: Client<_, BoxBody<Bytes, hyper::Error>> =
        Client::builder(TokioExecutor::new()).build(https);

    let listener = TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;
    info!(%addr, "unlost serve active");
    println!("unlost serve active on {}", addr);
    println!("expected base URLs like: http://{}:{}/w/<workspace_id>/anthropic/v1/messages", addr.ip(), addr.port());

    let state = ServeState::new(embedder);

    const FLUSH_CHAN_CAP: usize = 256;
    let (flush_tx, flush_rx) = kanal::bounded::<FlushJob>(FLUSH_CHAN_CAP);
    let chunker = WorkspaceChunker::new(flush_tx.to_async());

    // Periodically flush idle buffers (workspace-level).
    {
        let chunker = chunker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                chunker.flush_idle().await;
            }
        });
    }

    tokio::spawn(process_flush_jobs_serve(flush_rx.to_async(), state.clone()));

    loop {
        let (stream, peer) = listener.accept().await?;
        let conn_id = CONN_SEQ.fetch_add(1, Ordering::Relaxed);
        info!(conn_id, ?peer, "connection accepted");

        let io = TokioIo::new(stream);
        let client = client.clone();
        let chunker = chunker.clone();

        tokio::spawn(async move {
            const ANALYSIS_CHAN_CAP: usize = 256;
            let (analysis_tx, analysis_rx) = kanal::bounded::<AnalysisMsg>(ANALYSIS_CHAN_CAP);
            tokio::spawn(analysis_worker_multiplex(analysis_rx.to_async(), chunker.clone(), conn_id));
            let drops_logged = Arc::new(AtomicBool::new(false));

            let service = hyper::service::service_fn(move |req| {
                serve_request(
                    client.clone(),
                    conn_id,
                    analysis_tx.clone(),
                    drops_logged.clone(),
                    req,
                )
            });

            let res = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service)
                .await;
            if let Err(e) = res {
                warn!(conn_id, error = ?e, "connection error");
            }
        });
    }
}

async fn run_proxy(
    bind: SocketAddr,
    upstream_host: String,
    upstream_port: u16,
    embedder: Embedder,
    ws: WorkspacePaths,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = ensure_capsules_table(&db).await?;

    let https = HttpsConnectorBuilder::new()
        .with_native_roots()
        .context("no native root CA certificates found")?
        .https_only()
        .enable_http1()
        .build();

    let client: Client<_, BoxBody<Bytes, hyper::Error>> =
        Client::builder(TokioExecutor::new()).build(https);

    let listener = TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;
    info!(%addr, %upstream_host, upstream_port, "unlost recording active");
    println!("unlost recording active on :{}", addr.port());

    const FLUSH_CHAN_CAP: usize = 256;
    let (flush_tx, flush_rx) = kanal::bounded::<FlushJob>(FLUSH_CHAN_CAP);
    let chunker = WorkspaceChunker::new(flush_tx.to_async());

    {
        let chunker = chunker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                chunker.flush_idle().await;
            }
        });
    }
    tokio::spawn(process_flush_jobs_proxy(
        flush_rx.to_async(),
        ws.clone(),
        db.clone(),
        embedder.clone(),
    ));

    let workspace_id = ws.id.clone();

    loop {
        let (stream, peer) = listener.accept().await?;
        let conn_id = CONN_SEQ.fetch_add(1, Ordering::Relaxed);
        info!(conn_id, ?peer, "connection accepted");

        let io = TokioIo::new(stream);
        let client = client.clone();
        let upstream_host = upstream_host.clone();
        let chunker = chunker.clone();
        let workspace_id = workspace_id.clone();
        tokio::spawn(async move {
            // One analysis worker per TCP connection. Requests on the same
            // connection (HTTP keep-alive) share this worker.
            const ANALYSIS_CHAN_CAP: usize = 256;
            let (analysis_tx, analysis_rx) = kanal::bounded::<AnalysisMsg>(ANALYSIS_CHAN_CAP);
            tokio::spawn(analysis_worker(analysis_rx.to_async(), chunker.clone(), conn_id));
            let drops_logged = Arc::new(AtomicBool::new(false));

            let service = hyper::service::service_fn(move |req| {
                proxy_request(
                    client.clone(),
                    workspace_id.clone(),
                    upstream_host.clone(),
                    upstream_port,
                    conn_id,
                    analysis_tx.clone(),
                    drops_logged.clone(),
                    req,
                )
            });

            let res = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service)
                .await;

            if let Err(e) = res {
                warn!(conn_id, error = ?e, "connection error");
            }
        });
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.command.is_none() {
        Cli::command().print_help()?;
        println!("\n\nTry:\n- unlost serve --bind 127.0.0.1:3000\n- unlost configure agent opencode --path . --server http://127.0.0.1:3000\n- unlost config llm anthropic --model claude-3-5-sonnet-20241022\n- unlost init --path .\n- unlost recall\n- unlost query \"what are the routes available?\"\n");
        return Ok(());
    }

    let filter = if let Some(level) = cli.log {
        // Keep dependency noise low unless user opts in via RUST_LOG.
        tracing_subscriber::EnvFilter::new(format!(
            "unlost={},lance=warn,lancedb=warn",
            level.as_tracing_str()
        ))
    } else {
        tracing_subscriber::EnvFilter::try_from_default_env()
            // Stay quiet by default; opt into verbosity via RUST_LOG.
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"))
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // rustls 0.23 requires selecting a process-level CryptoProvider.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    match cli.command.unwrap() {
        Command::Query {
            query,
            limit,
            symbol,
            no_llm,
            llm_model,
            facts,
            output,
            embed_model,
            embed_cache_dir,
            file,
        } => {
            let query = query.join(" ");
            let ws = get_or_create_workspace_paths(&std::env::current_dir()?)?;
            let embedder = load_embedder(
                &embed_model,
                embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                false,
            )
            .await?;

            let spinner = if query_spinner_enabled(output) {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::with_template("{spinner} {msg}")
                        .unwrap()
                        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
                );
                pb.enable_steady_tick(Duration::from_millis(80));
                pb.set_message("Let me check...");
                Some(pb)
            } else {
                None
            };

            match query_capsules_lancedb(&query, limit, symbol.as_deref(), embedder.clone(), &ws).await {
                Ok(mut matches) if !matches.is_empty() => {
                    matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
                    if !no_llm {
                        if let Some(pb) = spinner.as_ref() {
                            pb.set_message("Putting it together...");
                        }
                        match llm_query_narrative(llm_model.as_deref(), &query, symbol.as_deref(), &matches).await {
                            Ok(n) => {
                                let rendered = render_narrative(output, &n);
                                if let Some(pb) = spinner.as_ref() {
                                    pb.finish_and_clear();
                                }
                                println!("{}\n", rendered);
                            }
                            Err(e) => {
                                warn!(error = ?e, "query narrative failed; printing raw matches");
                            }
                        }
                    }

                    if let Some(pb) = spinner.as_ref() {
                        pb.finish_and_clear();
                    }

                    if no_llm || facts {
                        for hit in matches {
                            let dist = hit.distance;
                            let cap = hit.capsule;
                            let meta = hit.meta;
                            println!("---");
                            println!("distance:  {dist}");
                            println!("source:    {}", meta.source);
                            println!("category:  {}", cap.category);
                            println!("upstream:  {}", meta.upstream_host);
                            println!("path:      {}", meta.request_path);
                            if !cap.intent.trim().is_empty() {
                                println!("intent:    {}", cap.intent);
                            }
                            if !cap.decision.trim().is_empty() {
                                println!("decision:  {}", cap.decision);
                            }
                            if !cap.rationale.trim().is_empty() {
                                println!("rationale: {}", cap.rationale);
                            }
                            if !cap.next_steps.is_empty() {
                                println!("next:      {:?}", cap.next_steps);
                            }
                            println!("symbols:   {:?}\n", cap.symbols);
                        }
                    }
                }
                Ok(_) => {
                    if let Some(pb) = spinner.as_ref() {
                        pb.finish_and_clear();
                    }
                    println!("No matches for: {query}");
                }
                Err(e) => {
                    if let Some(pb) = spinner.as_ref() {
                        pb.finish_and_clear();
                    }
                    warn!(error = ?e, "lancedb query failed; falling back to jsonl");
                    let fallback = if file.trim().is_empty() {
                        ws.capsules_jsonl.to_string_lossy().to_string()
                    } else {
                        file
                    };
                    query_capsules_jsonl(&fallback, &query, limit)?;
                }
            }
        }
        Command::Recall {
            target,
            limit,
            llm_model,
            output,
            embed_model,
            embed_cache_dir,
        } => {
            let ws = get_or_create_workspace_paths(&std::env::current_dir()?)?;

            let scope = target.join(" ");
            let scope = scope.trim().to_string();
            let scope_opt = (!scope.is_empty()).then_some(scope);

            let embedder = load_embedder(
                &embed_model,
                embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                false,
            )
            .await?;

            // Start with recent capsules (for story), then add scoped/semantic for relevance.
            let mut hits: Vec<CapsuleHit> = Vec::new();
            if let Ok(mut recent) = scan_capsules_lancedb(&ws, 120, None).await {
                recent.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
                hits.extend(recent.into_iter().take(limit.min(40)));
            }

            if let Some(scope) = scope_opt.as_deref() {
                // Symbol-scoped scan (exact match in symbols list)
                if let Some(expr) = scope_filter_expr(scope) {
                    if let Ok(mut scoped) = scan_capsules_lancedb(&ws, 80, Some(&expr)).await {
                        scoped.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
                        hits.extend(scoped);
                    }
                }

                // Semantic recall for the scope (helps surface cross-file impacts)
                if let Ok(mut sem) = query_capsules_lancedb(scope, 18, None, embedder.clone(), &ws).await {
                    sem.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
                    hits.extend(sem);
                }
            }

            // Dedupe by id, keep most recent version.
            let mut by_id: HashMap<String, CapsuleHit> = HashMap::new();
            for h in hits {
                match by_id.get(&h.id) {
                    Some(existing) if existing.ts_ms >= h.ts_ms => {}
                    _ => {
                        by_id.insert(h.id.clone(), h);
                    }
                }
            }
            let mut hits = by_id.into_values().collect::<Vec<_>>();
            hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
            if hits.len() > limit {
                hits.truncate(limit);
            }

            if hits.is_empty() {
                if let Some(s) = scope_opt {
                    println!("No capsules found yet for: {s}");
                } else {
                    println!("No capsules found yet for this workspace.");
                }
                return Ok(());
            }

            let narrative = llm_recall_narrative(llm_model.as_deref(), scope_opt.as_deref(), &hits).await?;
            println!("{}\n", render_narrative(output, &narrative));
        }
        Command::Init {
            path,
            embed_model,
            embed_cache_dir,
            max_capsules,
            no_llm,
            git_history,
            git_commits,
            llm_model,
            llm_max_capsules,
        } => {
            let ws = get_or_create_workspace_paths(std::path::Path::new(&path))?;
            run_init(
                &path,
                &embed_model,
                embed_cache_dir.as_deref(),
                no_llm,
                git_history,
                git_commits,
                llm_model.as_deref(),
                llm_max_capsules,
                max_capsules,
                ws,
            )
            .await?;
        }
        Command::Model { command } => match command {
            ModelCommand::Download { embed_model, cache_dir, force } => {
                let dir = download_model(&embed_model, cache_dir.as_deref(), force).await?;
                println!("downloaded into: {}", dir.display());
            }
        },
        Command::Config { command } => match command {
            ConfigCommand::Llm { command } => {
                handle_llm_command(command)?;
            }
            ConfigCommand::Agent { command } => {
                handle_agent_command(command)?;
            }
        },
        Command::Clear { path } => {
            clear_workspace(std::path::Path::new(&path))?;
        }
        Command::Inspect { path, limit, filter } => {
            let ws = get_or_create_workspace_paths(std::path::Path::new(&path))?;
            match scan_capsules_lancedb(&ws, limit, filter.as_deref()).await {
                Ok(rows) if !rows.is_empty() => {
                    println!("workspace: {}", ws.id);
                    for hit in rows {
                        let cap = hit.capsule;
                        let meta = hit.meta;
                        println!("---");
                        println!("source:    {}", meta.source);
                        println!("category:  {}", cap.category);
                        println!("upstream:  {}", meta.upstream_host);
                        println!("path:      {}", meta.request_path);
                        if !cap.intent.trim().is_empty() {
                            println!("intent:    {}", cap.intent);
                        }
                        if !cap.decision.trim().is_empty() {
                            println!("decision:  {}", cap.decision);
                        }
                        if !cap.rationale.trim().is_empty() {
                            println!("rationale: {}", cap.rationale);
                        }
                        if !cap.next_steps.is_empty() {
                            println!("next:      {:?}", cap.next_steps);
                        }
                        println!("symbols:   {:?}\n", cap.symbols);
                    }
                }
                Ok(_) => {
                    println!("workspace: {}", ws.id);
                    println!("no rows found");
                }
                Err(e) => {
                    warn!(error = ?e, "inspect failed");
                    println!("inspect failed: {e}");
                }
            }
        }
        Command::Serve { bind, embed_model, embed_cache_dir } => {
            let bind = parse_bind(&bind)?;
            let embedder = load_embedder(
                &embed_model,
                embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                false,
            )
            .await?;
            run_serve(bind, embedder).await?;
        }
        Command::Record { bind, upstream_host, upstream_port, embed_model, embed_cache_dir } => {
            if upstream_host.trim().is_empty() {
                anyhow::bail!("missing upstream host (set --upstream-host or UNLOST_UPSTREAM_HOST)");
            }
            let bind = parse_bind(&bind)?;
            let embedder = load_embedder(
                &embed_model,
                embed_cache_dir.as_deref().map(std::path::PathBuf::from),
                false,
            )
            .await?;
            let ws = get_or_create_workspace_paths(&std::env::current_dir()?)?;
            run_proxy(bind, upstream_host, upstream_port, embedder, ws).await?;
        }
    }

    Ok(())
}
