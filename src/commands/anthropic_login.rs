//! Anthropic SSO login via OAuth 2.0 PKCE — manual copy-code flow.
//!
//! Flow:
//!   1. Generate a PKCE code verifier + challenge (S256) and CSRF state.
//!   2. Build an authorize URL with `code=true` so Anthropic shows the
//!      copy-code page instead of attempting a redirect.
//!   3. Open the user's browser to that URL (or print it for them to copy).
//!   4. The user authenticates on claude.ai, picks the org/workspace they
//!      want, and Anthropic shows them a `<authcode>#<state>` blob to copy.
//!   5. The user pastes that blob into the terminal.
//!   6. Exchange the code (form-encoded POST, 120 s timeout) for an access
//!      token at `console.anthropic.com/v1/oauth/token`.
//!   7. POST to `api.anthropic.com/api/oauth/claude_cli/create_api_key`
//!      with the Bearer token to mint a permanent `sk-ant-...` API key.
//!   8. Persist via `set_llm_config(LlmConfig::Anthropic { ... })`.
//!
//! Notes on account selection:
//!   - The browser flow uses whatever claude.ai session the user is already
//!     signed into. To pick a different account (e.g. team vs personal), the
//!     user must log out of claude.ai first or open the URL in a private /
//!     incognito window — there is no `prompt=select_account` parameter on
//!     this flow.
//!
//! The temporary OAuth access token is discarded after step 7; only the
//! permanent API key is persisted, so the rest of the codebase
//! (llm_extract, show_llm_config, etc.) requires zero changes.

use anyhow::Context;
use base64::Engine as _;
use sha2::Digest;
use std::io::{BufRead, Write};
use std::time::Duration;

use crate::config::LlmConfig;

// OAuth app credentials used by the Claude CLI / opencode-anthropic-auth.
// Public PKCE client — no client_secret.
const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// The redirect target Anthropic uses for the copy-code flow. The user is
/// shown a page that displays the auth code; nothing is actually redirected
/// to a localhost callback.
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const AUTH_URL: &str = "https://claude.ai/oauth/authorize";
/// `console.anthropic.com` and `platform.claude.com` both accept token
/// exchanges for this `client_id`. We use the console host because the
/// account/admin endpoints (incl. `create_api_key`) live there too.
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const CREATE_KEY_URL: &str = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key";
const SCOPES: &str = "org:create_api_key user:profile user:inference";

/// Run the full SSO login flow and store a permanent Anthropic API key.
pub async fn run(model: String, base_url: Option<String>) -> anyhow::Result<()> {
    // ── 1. PKCE + CSRF state ──────────────────────────────────────────────────
    let verifier = generate_verifier();
    let challenge = pkce_challenge(&verifier);
    let state = generate_verifier(); // reuse same generator → 43-char base64url

    // ── 2. Build the authorize URL with `code=true` so Anthropic shows the
    //       copy-code page after the user completes login.
    let auth_url = format!(
        "{AUTH_URL}?code=true\
         &response_type=code\
         &client_id={CLIENT_ID}\
         &redirect_uri={}\
         &scope={}\
         &code_challenge={challenge}\
         &code_challenge_method=S256\
         &state={state}",
        urlencoding::encode(REDIRECT_URI),
        urlencoding::encode(SCOPES),
    );

    // ── 3. User-facing instructions ───────────────────────────────────────────
    println!();
    println!("Anthropic OAuth login");
    println!("─────────────────────");
    println!();
    println!("1. A browser window will open. Sign in with the Anthropic account");
    println!("   you want to use, then pick the organisation / workspace.");
    println!();
    println!("   ⚠  Anthropic uses whatever account you are already signed in to");
    println!("      on claude.ai. If you have the wrong account active, log out");
    println!("      first (or open the URL in a private / incognito window).");
    println!();
    println!("2. After authorising, the page will display an authorization code.");
    println!("   Copy the entire code (including the part after `#`) and paste");
    println!("   it back here when prompted.");
    println!();
    println!("URL:");
    println!("  {auth_url}");
    println!();

    open_browser(&auth_url);

    // ── 4. Prompt the user for the code they were shown ───────────────────────
    let raw_code = read_line_prompt("Paste the authorization code here: ")?;
    let raw_code = raw_code.trim();
    if raw_code.is_empty() {
        anyhow::bail!("no authorization code provided");
    }

    // The displayed code is `<authcode>#<state>` — split if present.
    // If the user paste-trims either half we still want to handle both forms.
    let (auth_code, returned_state) = match raw_code.split_once('#') {
        Some((c, s)) => (c.to_string(), s.to_string()),
        None => (raw_code.to_string(), state.clone()),
    };

    if !returned_state.is_empty() && returned_state != state {
        // Soft warn; the server will validate. Don't hard-fail because some
        // claude.ai variants don't include the state in the displayed blob.
        eprintln!("warning: returned state did not match (continuing anyway)");
    }

    println!();
    println!("Exchanging code for access token (this can take up to a minute)…");

    // ── 5. Exchange for access token (form-encoded, 120 s timeout). ──────────
    //       The token endpoint can take 40–60 s during platform issues; the
    //       Claude CLI's hard-coded 15 s timeout has bitten users.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

    let mut form = std::collections::HashMap::new();
    form.insert("grant_type", "authorization_code");
    form.insert("client_id", CLIENT_ID);
    form.insert("code", &auth_code);
    form.insert("redirect_uri", REDIRECT_URI);
    form.insert("code_verifier", &verifier);
    form.insert("state", &returned_state);

    let token_resp = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&form)
        .send()
        .await
        .context("token exchange request failed")?;

    let status = token_resp.status();
    let body = token_resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "token exchange failed ({status}). \
             Likely causes: code expired, wrong code copied, or the code was \
             already used. Try again.\n\nResponse body: {body}"
        );
    }
    let token_json: serde_json::Value =
        serde_json::from_str(&body).context("invalid JSON in token response")?;
    let access_token = token_json["access_token"]
        .as_str()
        .context("missing access_token in token response")?
        .to_string();

    println!("Token obtained. Creating permanent API key…");

    // ── 6. Mint a permanent API key. ──────────────────────────────────────────
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
        anyhow::bail!(
            "API key creation failed ({key_status}).\n\n\
             This usually means the OAuth account does not have permission to \
             create API keys in the chosen workspace. Try logging in again \
             with an account that has admin rights, or pick a workspace where \
             you can issue keys.\n\nResponse body: {key_body}"
        );
    }
    let key_json: serde_json::Value =
        serde_json::from_str(&key_body).context("invalid JSON in key creation response")?;
    let api_key = key_json["raw_key"]
        .as_str()
        .context("missing raw_key in API key creation response")?
        .to_string();

    // ── 7. Persist to unlost config. ─────────────────────────────────────────
    crate::llm::set_llm_config(Some(LlmConfig::Anthropic {
        api_key,
        base_url,
        model,
    }))
    .context("failed to save LLM config")?;

    println!();
    println!("Logged in. LLM provider set to Anthropic (SSO).");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn generate_verifier() -> String {
    // 32 random bytes → base64url (no padding) → 43-char string.
    let bytes: Vec<u8> = (0..32).map(|_| rand_byte()).collect();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn pkce_challenge(verifier: &str) -> String {
    let hash = sha2::Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

fn open_browser(url: &str) {
    // Best-effort: try xdg-open (Linux), then open (macOS), then start (Windows).
    let _ = std::process::Command::new("xdg-open")
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .or_else(|_| {
            std::process::Command::new("open")
                .arg(url)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        })
        .or_else(|_| {
            std::process::Command::new("cmd")
                .args(["/c", "start", "", url])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        });
}

fn rand_byte() -> u8 {
    let mut buf = [0u8; 1];
    getrandom::fill(&mut buf).expect("getrandom failed");
    buf[0]
}

fn read_line_prompt(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    let stdin = std::io::stdin();
    stdin
        .lock()
        .read_line(&mut line)
        .context("failed to read from stdin")?;
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_verifier_and_challenge_are_deterministic() {
        // The challenge for a known verifier per RFC 7636 §4.6 example.
        let v = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let c = pkce_challenge(v);
        assert_eq!(c, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn auth_code_blob_splits_on_hash() {
        let raw = "abc123#xyz789";
        let (code, state) = match raw.split_once('#') {
            Some((c, s)) => (c.to_string(), s.to_string()),
            None => (raw.to_string(), String::new()),
        };
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }
}
