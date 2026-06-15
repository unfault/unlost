//! Anthropic SSO login via OAuth 2.0 PKCE.
//!
//! Flow:
//!   1. Generate a PKCE code verifier + challenge (S256).
//!   2. Start a local HTTP server on a random port to receive the callback.
//!   3. Open the browser to `https://console.anthropic.com/oauth/authorize`.
//!   4. Anthropic redirects to `http://127.0.0.1:<port>/callback?code=...`.
//!   5. Exchange the code for an access token at
//!      `https://console.anthropic.com/v1/oauth/token`.
//!   6. POST to `https://api.anthropic.com/api/oauth/claude_cli/create_api_key`
//!      with the Bearer token to mint a permanent `sk-ant-...` API key.
//!   7. Store the API key via `set_llm_config(LlmConfig::Anthropic { ... })`.
//!
//! The temporary OAuth access token is discarded after step 6; only the
//! permanent API key is persisted. This means the rest of the codebase
//! (llm_extract, show_llm_config, etc.) requires zero changes.

use anyhow::Context;
use base64::Engine as _;
use sha2::Digest;
use std::net::TcpListener;

use crate::config::LlmConfig;

// OAuth app credentials used by the Claude CLI / opencode-anthropic-auth.
// These are the same public client_id that Anthropic ships with their own
// first-party tools; there is no client_secret (it is a public PKCE client).
const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const AUTH_URL: &str = "https://console.anthropic.com/oauth/authorize";
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const CREATE_KEY_URL: &str =
    "https://api.anthropic.com/api/oauth/claude_cli/create_api_key";
const SCOPES: &str = "org:create_api_key user:profile user:inference";

/// Run the full SSO login flow and store a permanent Anthropic API key.
pub async fn run(model: String, base_url: Option<String>) -> anyhow::Result<()> {
    // ── 1. PKCE ────────────────────────────────────────────────────────────────
    let verifier = generate_verifier();
    let challenge = pkce_challenge(&verifier);
    let state = uuid::Uuid::new_v4().to_string();

    // ── 2. Local callback server ───────────────────────────────────────────────
    // Bind to port 0 so the OS picks a free port.
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to bind local callback server")?;
    let port = listener.local_addr()?.port();
    let _local_redirect = format!("http://127.0.0.1:{port}/callback");

    // ── 3. Build auth URL and open browser ────────────────────────────────────
    let auth_url = format!(
        "{AUTH_URL}?response_type=code\
         &client_id={CLIENT_ID}\
         &redirect_uri={}\
         &scope={}\
         &code_challenge={challenge}\
         &code_challenge_method=S256\
         &state={state}",
        urlencoding::encode(REDIRECT_URI),
        urlencoding::encode(SCOPES),
    );

    println!("Opening your browser for Anthropic login...");
    println!("If the browser does not open, visit:\n  {auth_url}");
    println!();

    open_browser(&auth_url);

    // ── 4. Receive the callback ────────────────────────────────────────────────
    println!("Waiting for Anthropic to redirect back (port {port})...");

    // The browser will be redirected to REDIRECT_URI (console.anthropic.com),
    // which in turn redirects to our local server. We wait for that local hit.
    let code = wait_for_code(listener, &state)
        .await
        .context("did not receive OAuth callback")?;

    println!("Received authorization code. Exchanging for access token...");

    // ── 5. Exchange code for access token ─────────────────────────────────────
    let client = reqwest::Client::new();
    let token_resp = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        }))
        .send()
        .await
        .context("token exchange request failed")?;

    let status = token_resp.status();
    let body = token_resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("token exchange failed ({status}): {body}");
    }
    let token_json: serde_json::Value =
        serde_json::from_str(&body).context("invalid JSON in token response")?;
    let access_token = token_json["access_token"]
        .as_str()
        .context("missing access_token in token response")?
        .to_string();

    println!("Token obtained. Creating permanent API key...");

    // ── 6. Create permanent API key ────────────────────────────────────────────
    let key_resp = client
        .post(CREATE_KEY_URL)
        .header("Authorization", format!("Bearer {access_token}"))
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({}))
        .send()
        .await
        .context("API key creation request failed")?;

    let key_status = key_resp.status();
    let key_body = key_resp.text().await.unwrap_or_default();
    if !key_status.is_success() {
        anyhow::bail!("API key creation failed ({key_status}): {key_body}");
    }
    let key_json: serde_json::Value =
        serde_json::from_str(&key_body).context("invalid JSON in key creation response")?;
    let api_key = key_json["raw_key"]
        .as_str()
        .context("missing raw_key in API key creation response")?
        .to_string();

    // ── 7. Persist to unlost config ───────────────────────────────────────────
    crate::llm::set_llm_config(Some(LlmConfig::Anthropic {
        api_key,
        base_url,
        model,
    }))
    .context("failed to save LLM config")?;

    println!("Logged in. LLM provider set to Anthropic (SSO).");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn generate_verifier() -> String {
    // 32 random bytes → base64url (no padding) → 43-char verifier.
    let bytes: Vec<u8> = (0..32).map(|_| rand_byte()).collect();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes)
}

fn pkce_challenge(verifier: &str) -> String {
    let hash = sha2::Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

fn open_browser(url: &str) {
    // Try xdg-open (Linux), then open (macOS), then start (Windows).
    let _ = std::process::Command::new("xdg-open").arg(url).spawn()
        .or_else(|_| std::process::Command::new("open").arg(url).spawn())
        .or_else(|_| std::process::Command::new("cmd").args(["/c", "start", url]).spawn());
}

/// Minimal inline random byte (no rand crate needed — getrandom is available).
fn rand_byte() -> u8 {
    let mut buf = [0u8; 1];
    // getrandom v0.3 uses `fill`.
    getrandom::fill(&mut buf).expect("getrandom failed");
    buf[0]
}

/// Spin up a one-shot HTTP server that captures `?code=` from the redirect.
async fn wait_for_code(listener: TcpListener, expected_state: &str) -> anyhow::Result<String> {
    use http_body_util::Full;
    use hyper::{Request, Response, body::Bytes, server::conn::http1};
    use hyper_util::rt::TokioIo;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener as TokioListener;

    let code_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let code_slot_inner = Arc::clone(&code_slot);
    let state_str = expected_state.to_string();

    // Convert the already-bound std listener to a tokio one.
    listener.set_nonblocking(true)?;
    let tokio_listener = TokioListener::from_std(listener)?;

    // Accept exactly one connection.
    let (stream, _) = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        tokio_listener.accept(),
    )
    .await
    .context("timed out waiting for browser redirect")?
    .context("accept error")?;

    let io = TokioIo::new(stream);
    let code_slot_for_svc = Arc::clone(&code_slot_inner);
    let state_for_svc = state_str.clone();

    let svc = hyper::service::service_fn(move |req: Request<hyper::body::Incoming>| {
        let slot = Arc::clone(&code_slot_for_svc);
        let st = state_for_svc.clone();
        async move {
            let path_and_query = req.uri().path_and_query()
                .map(|pq| pq.as_str())
                .unwrap_or("");

            let (code, msg) = extract_code_from_query(path_and_query, &st);
            if let Some(c) = code {
                *slot.lock().unwrap() = Some(c);
            }

            Ok::<_, std::convert::Infallible>(
                Response::new(Full::new(Bytes::from(msg))),
            )
        }
    });

    http1::Builder::new()
        .serve_connection(io, svc)
        .await
        .context("error serving callback connection")?;

    let code = code_slot
        .lock()
        .unwrap()
        .take()
        .context("no authorization code received in callback")?;

    Ok(code)
}

fn extract_code_from_query(path_and_query: &str, expected_state: &str) -> (Option<String>, String) {
    // path_and_query looks like "/callback?code=XXX&state=YYY"
    let query = path_and_query
        .splitn(2, '?')
        .nth(1)
        .unwrap_or("");

    let mut code = None;
    let mut state = None;
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        match k {
            "code" => code = Some(urlencoding::decode(v).unwrap_or_default().into_owned()),
            "state" => state = Some(v.to_string()),
            _ => {}
        }
    }

    // Validate state to guard against CSRF.
    if state.as_deref() != Some(expected_state) {
        return (
            None,
            "Error: state mismatch. Please retry.".to_string(),
        );
    }

    if code.is_some() {
        (
            code,
            "<html><body><h2>Login successful!</h2><p>You can close this tab.</p></body></html>"
                .to_string(),
        )
    } else {
        (
            None,
            "Error: no authorization code received.".to_string(),
        )
    }
}
