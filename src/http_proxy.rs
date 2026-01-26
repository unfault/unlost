use anyhow::Context;
use bytes::Bytes;
use http_body_util::{
    BodyExt,
    BodyStream,
    StreamBody,
    combinators::BoxBody,
};
use hyper::{
    Request,
    Response,
    StatusCode,
    Uri,
};
use hyper::server::conn::http1;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::{
    client::legacy::Client,
    rt::{TokioExecutor, TokioIo},
};
use futures_util::TryStreamExt;
use std::io::IsTerminal;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::net::TcpListener;
use tokio::time::Duration;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpstreamProvider {
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

async fn serve_request(
    client: Client<hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>, BoxBody<Bytes, hyper::Error>>,
    conn_id: u64,
    analysis_tx: kanal::Sender<crate::analysis::AnalysisMsg>,
    analysis_drops_logged: Arc<AtomicBool>,
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
    let (workspace_id, provider, stripped_pq) = match parse_multiplexed_uri(req.uri()) {
        Ok(v) => v,
        Err(e) => {
            warn!(conn_id, error = ?e, "bad multiplexed uri");
            let resp = Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(crate::net::text_body(b"expected /w/<workspace_id>/<provider>/..."))
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
                .body(crate::net::text_body(b"bad request"))
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
    let headers = crate::net::sanitize_request_headers(parts.headers, &upstream_host);

    const MAX_REQ_BODY: usize = 2 * 1024 * 1024;
    let req_body_bytes = match crate::net::read_incoming_body_limited(body, MAX_REQ_BODY).await {
        Ok(b) => b,
        Err(e) => {
            warn!(conn_id, error = ?e, "request body capture failed");
            let resp = Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(crate::net::text_body(b"payload too large"))
                .unwrap();
            return Ok(resp);
        }
    };

    let mut out_req = Request::builder()
        .method(method)
        .uri(upstream_uri)
        .version(version)
        .body(crate::net::bytes_body(req_body_bytes.clone()))
        .unwrap();
    *out_req.headers_mut() = headers;

    let res = match client.request(out_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(conn_id, error = ?e, "upstream request failed");
            let resp = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(crate::net::text_body(b"bad gateway"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (res_parts, res_body) = res.into_parts();
    let res_headers = crate::net::sanitize_response_headers(res_parts.headers);
    let content_type = res_headers
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let meta = crate::analysis::AnalysisMeta {
        workspace_id,
        upstream_host: upstream_host.clone(),
        request_path: request_path.clone(),
        http_status: res_parts.status.as_u16(),
        content_type,
    };
    let _ = analysis_tx.try_send(crate::analysis::AnalysisMsg::ExchangeStart {
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
                        match tx.try_send(crate::analysis::AnalysisMsg::ResponseChunk(data.clone())) {
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
                    let _ = tx.try_send(crate::analysis::AnalysisMsg::ResponseEnd);
                    None
                }
                Err(e) => {
                    let _ = tx.try_send(crate::analysis::AnalysisMsg::ResponseEnd);
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

async fn proxy_request(
    client: Client<hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>, BoxBody<Bytes, hyper::Error>>,
    workspace_id: String,
    upstream_host: String,
    upstream_port: u16,
    conn_id: u64,
    analysis_tx: kanal::Sender<crate::analysis::AnalysisMsg>,
    analysis_drops_logged: Arc<AtomicBool>,
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
    let upstream_uri = match crate::net::build_upstream_uri(&upstream_host, upstream_port, req.uri()) {
        Ok(u) => u,
        Err(e) => {
            warn!(conn_id, error = ?e, "bad request URI");
            let resp = Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(crate::net::text_body(b"bad request"))
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
    let headers = crate::net::sanitize_request_headers(parts.headers, &upstream_host);

    const MAX_REQ_BODY: usize = 2 * 1024 * 1024;
    let req_body_bytes = match crate::net::read_incoming_body_limited(body, MAX_REQ_BODY).await {
        Ok(b) => b,
        Err(e) => {
            warn!(conn_id, error = ?e, "request body capture failed");
            let resp = Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(crate::net::text_body(b"payload too large"))
                .unwrap();
            return Ok(resp);
        }
    };

    let mut out_req = Request::builder()
        .method(method)
        .uri(upstream_uri)
        .version(version)
        .body(crate::net::bytes_body(req_body_bytes.clone()))
        .unwrap();
    *out_req.headers_mut() = headers;

    let res = match client.request(out_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(conn_id, error = ?e, "upstream request failed");
            let resp = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(crate::net::text_body(b"bad gateway"))
                .unwrap();
            return Ok(resp);
        }
    };

    let (res_parts, res_body) = res.into_parts();
    let res_headers = crate::net::sanitize_response_headers(res_parts.headers);

    let content_type = res_headers
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let meta = crate::analysis::AnalysisMeta {
        workspace_id,
        upstream_host: upstream_host.clone(),
        request_path: request_path.clone(),
        http_status: res_parts.status.as_u16(),
        content_type,
    };
    let _ = analysis_tx.try_send(crate::analysis::AnalysisMsg::ExchangeStart {
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
                        match tx.try_send(crate::analysis::AnalysisMsg::ResponseChunk(data.clone())) {
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
                    let _ = tx.try_send(crate::analysis::AnalysisMsg::ResponseEnd);
                    None
                }
                Err(e) => {
                    let _ = tx.try_send(crate::analysis::AnalysisMsg::ResponseEnd);
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

pub(crate) async fn run_serve(bind: SocketAddr, embedder: crate::embed::Embedder) -> anyhow::Result<()> {
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

    let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let (title_on, title_off) = if use_color {
        ("\x1b[36;1m", "\x1b[0m") // bold cyan
    } else {
        ("", "")
    };
    let (ok_on, ok_off) = if use_color {
        ("\x1b[32;1m", "\x1b[0m") // bold green
    } else {
        ("", "")
    };
    let (dim_on, dim_off) = if use_color { ("\x1b[2m", "\x1b[0m") } else { ("", "") };

    let host = if addr.ip().is_unspecified() {
        "127.0.0.1".to_string()
    } else {
        addr.ip().to_string()
    };
    let base = format!("http://{host}:{}", addr.port());

    println!("{title_on}unlost serve{title_off} {ok_on}listening{ok_off} on {base}");
    println!("{dim_on}Multiplexing via URL: /w/<workspace_id>/<provider>/...{dim_off}");
    println!("{dim_on}Providers: openai | anthropic | opencode{dim_off}");
    println!();
    println!("Examples:");
    println!("  {dim_on}Anthropic:{dim_off} {base}/w/<workspace_id>/anthropic/v1/messages");
    println!("  {dim_on}OpenAI:{dim_off}    {base}/w/<workspace_id>/openai/v1/chat/completions");
    println!("  {dim_on}OpenCode:{dim_off}  {base}/w/<workspace_id>/opencode/zen/v1/responses");
    println!();
    println!("Next:");
    println!(
        "  {dim_on}Write agent config:{dim_off} unlost configure agent opencode --path . --server {base}"
    );

    if use_color {
        let openai_env = std::env::var("OPENAI_BASE_URL")
            .ok()
            .or_else(|| std::env::var("OPENAI_API_BASE").ok());
        let anthropic_env = std::env::var("ANTHROPIC_BASE_URL").ok();
        if openai_env.is_some() || anthropic_env.is_some() {
            println!();
            println!("{dim_on}Note:{dim_off} if you set base-url env vars in this shell (e.g. OPENAI_BASE_URL), unlost will ignore them for its own extractor calls.");
        }
    }

    let emotion = crate::emotion::EmotionModel::load(crate::emotion::EmotionConfig::default()).await?;
    let state = crate::recording::ServeState::new(embedder, emotion);

    const FLUSH_CHAN_CAP: usize = 256;
    let (flush_tx, flush_rx) = kanal::bounded::<crate::recording::FlushJob>(FLUSH_CHAN_CAP);
    let chunker = crate::recording::WorkspaceChunker::new(flush_tx.to_async());

    {
        let chunker = chunker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                chunker.flush_idle().await;
            }
        });
    }

    tokio::spawn(crate::recording::process_flush_jobs_serve(flush_rx.to_async(), state.clone()));

    loop {
        let (stream, peer) = listener.accept().await?;
        let conn_id = crate::analysis::CONN_SEQ.fetch_add(1, Ordering::Relaxed);
        info!(conn_id, ?peer, "connection accepted");

        let io = TokioIo::new(stream);
        let client = client.clone();
        let chunker = chunker.clone();

        tokio::spawn(async move {
            const ANALYSIS_CHAN_CAP: usize = 256;
            let (analysis_tx, analysis_rx) =
                kanal::bounded::<crate::analysis::AnalysisMsg>(ANALYSIS_CHAN_CAP);
            tokio::spawn(crate::recording::analysis_worker_multiplex(
                analysis_rx.to_async(),
                chunker.clone(),
                conn_id,
            ));
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

pub(crate) async fn run_proxy(
    bind: SocketAddr,
    upstream_host: String,
    upstream_port: u16,
    embedder: crate::embed::Embedder,
    ws: crate::WorkspacePaths,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let emotion = crate::emotion::EmotionModel::load(crate::emotion::EmotionConfig::default()).await?;
    let emotion = Arc::new(std::sync::Mutex::new(emotion));

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
    let (flush_tx, flush_rx) = kanal::bounded::<crate::recording::FlushJob>(FLUSH_CHAN_CAP);
    let chunker = crate::recording::WorkspaceChunker::new(flush_tx.to_async());

    {
        let chunker = chunker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                chunker.flush_idle().await;
            }
        });
    }
    tokio::spawn(crate::recording::process_flush_jobs_proxy(
        flush_rx.to_async(),
        ws.clone(),
        db.clone(),
        embedder.clone(),
        emotion.clone(),
    ));

    let workspace_id = ws.id.clone();

    loop {
        let (stream, peer) = listener.accept().await?;
        let conn_id = crate::analysis::CONN_SEQ.fetch_add(1, Ordering::Relaxed);
        info!(conn_id, ?peer, "connection accepted");

        let io = TokioIo::new(stream);
        let client = client.clone();
        let upstream_host = upstream_host.clone();
        let chunker = chunker.clone();
        let workspace_id = workspace_id.clone();
        tokio::spawn(async move {
            const ANALYSIS_CHAN_CAP: usize = 256;
            let (analysis_tx, analysis_rx) =
                kanal::bounded::<crate::analysis::AnalysisMsg>(ANALYSIS_CHAN_CAP);
            tokio::spawn(crate::recording::analysis_worker(
                analysis_rx.to_async(),
                chunker.clone(),
                conn_id,
            ));
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
