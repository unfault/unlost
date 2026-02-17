use anyhow::Context;
use bytes::Bytes;
use futures_util::TryStreamExt;
use http_body_util::{BodyExt, BodyStream, Full, combinators::BoxBody};
use hyper::Uri;
use hyper::header::{HeaderMap, HeaderName, HeaderValue};

pub(crate) fn hop_by_hop_names(headers: &HeaderMap) -> Vec<HeaderName> {
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

pub(crate) fn sanitize_request_headers(mut headers: HeaderMap, upstream_host: &str) -> HeaderMap {
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
    headers.insert(
        hyper::header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );

    headers
}

pub(crate) fn sanitize_response_headers(mut headers: HeaderMap) -> HeaderMap {
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

pub(crate) fn build_upstream_uri(
    upstream_host: &str,
    upstream_port: u16,
    uri: &Uri,
) -> anyhow::Result<Uri> {
    let pq = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let authority = format!("{upstream_host}:{upstream_port}");
    Uri::builder()
        .scheme("https")
        .authority(authority)
        .path_and_query(pq)
        .build()
        .context("failed to build upstream URI")
}

pub(crate) fn text_body(msg: &'static [u8]) -> BoxBody<Bytes, hyper::Error> {
    Full::new(Bytes::from_static(msg))
        .map_err(|never| match never {})
        .boxed()
}

pub(crate) fn bytes_body(b: Bytes) -> BoxBody<Bytes, hyper::Error> {
    Full::new(b).map_err(|never| match never {}).boxed()
}

pub(crate) async fn read_incoming_body_limited(
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

pub(crate) fn is_event_stream(content_type: Option<&str>) -> bool {
    content_type
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("text/event-stream")
}

pub(crate) fn decode_json_lossy(bytes: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(bytes).ok()
}

pub(crate) fn extract_openai_message_text(v: &serde_json::Value) -> Option<String> {
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

pub(crate) fn extract_anthropic_user_text(v: &serde_json::Value) -> Option<String> {
    let messages = v.get("messages").and_then(|m| m.as_array())?;

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

pub(crate) fn extract_openai_assistant_text_from_json(v: &serde_json::Value) -> Option<String> {
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

pub(crate) fn extract_anthropic_assistant_text_from_json(v: &serde_json::Value) -> Option<String> {
    let content = v.get("content")?;
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

// ---------------------------------------------------------------------------
// Friction detection helpers
// ---------------------------------------------------------------------------

/// Extract file paths and identifiers from request body for friction checking.
pub(crate) fn extract_symbols_from_body(body: &[u8], path: &str) -> Vec<String> {
    let json = match decode_json_lossy(body) {
        Some(j) => j,
        None => return vec![],
    };

    let text = if path.contains("/v1/messages") {
        extract_anthropic_user_text(&json)
    } else {
        extract_openai_message_text(&json)
    };

    text.map(|t| extract_symbols_from_text(&t))
        .unwrap_or_default()
}

/// Extract file paths and identifiers from text.
pub fn extract_symbols_from_text(text: &str) -> Vec<String> {
    let mut out = vec![];

    let blacklist = [
        "NOTE", "REASON", "DECISION", "INTENT", "NEXT_STEPS", "RATIONALE", "SYMBOLS",
        "USER", "ASSISTANT", "SYSTEM", "SUCCESS", "FAILURE", "ERROR", "WARNING",
    ];

    // File paths: src/foo.rs, ./lib/bar.py, auth.ts
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| {
            c == '`' || c == '\'' || c == '"' || c == '(' || c == ')' || c == ',' || c == ':' || c == '.'
        });

        if word.is_empty() {
            continue;
        }

        // Ignore if all caps and in blacklist
        if blacklist.iter().any(|&b| b == word.to_uppercase()) {
            continue;
        }

        // Check if it looks like a file path
        let looks_like_path = (word.contains('.') || word.contains('/'))
            && (word.contains('/')
                || word.ends_with("rs")
                || word.ends_with("ts")
                || word.ends_with("tsx")
                || word.ends_with("py")
                || word.ends_with("js")
                || word.ends_with("jsx")
                || word.ends_with("go")
                || word.ends_with("astro")
                || word.ends_with("md")
                || word.ends_with("toml")
                || word.ends_with("yaml")
                || word.ends_with("yml")
                || word.ends_with("json"));

        if looks_like_path {
            // Exclude common tool-call noise: read/inspect, write/file, etc.
            if word.contains('/') && !word.contains('.') {
                let parts: Vec<&str> = word.split('/').collect();
                if parts.iter().any(|&p| p == "read" || p == "write" || p == "inspect" || p == "call" || p == "tool") {
                    continue;
                }
            }

            // Basic sanity: not too long, has reasonable characters
            if word.len() < 200
                && word
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '/' || c == '.' || c == '_' || c == '-')
            {
                out.push(word.to_string());
            }
        }
    }

    // Backtick identifiers: `validateAuth`, `MyClass`
    let mut in_backtick = false;
    let mut current = String::new();
    for c in text.chars() {
        if c == '`' {
            if in_backtick && current.len() >= 3 {
                // If it contains spaces, split it and treat each word as a potential symbol
                if current.contains(' ') {
                    for part in current.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.' && c != '/') {
                        if part.len() >= 3 {
                            out.push(part.to_string());
                        }
                    }
                } else {
                    // Single word in backticks - keep it if it looks reasonable
                    if current.len() < 100 {
                        out.push(current.clone());
                    }
                }
            }
            current.clear();
            in_backtick = !in_backtick;
        } else if in_backtick {
            current.push(c);
        }
    }

    // Bold identifiers: **validateAuth**
    let mut in_bold = false;
    current.clear();
    let mut last_c = ' ';
    for c in text.chars() {
        if c == '*' && last_c == '*' {
            if in_bold && current.len() >= 3 {
                // If it contains spaces, split it
                if current.contains(' ') {
                    for part in current.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.' && c != '/') {
                        if part.len() >= 3 {
                            out.push(part.to_string());
                        }
                    }
                } else {
                    if current.len() < 100 {
                        out.push(current.clone());
                    }
                }
            }
            current.clear();
            in_bold = !in_bold;
        } else if c != '*' {
            if in_bold {
                current.push(c);
            }
        }
        last_c = c;
    }

    out.sort();
    out.dedup();
    out
}

/// Inject warning by prepending to last user message in request body.
pub(crate) fn inject_warning(body: &Bytes, warning: &str, _path: &str) -> anyhow::Result<Bytes> {
    let mut json: serde_json::Value = serde_json::from_slice(body)?;

    let messages = json
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
        .ok_or_else(|| anyhow::anyhow!("no messages array in request"))?;

    // Find last user message and prepend warning
    for msg in messages.iter_mut().rev() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            if let Some(content) = msg.get_mut("content") {
                if let Some(s) = content.as_str() {
                    *content = serde_json::json!(format!("{}{}", warning, s));
                    break;
                } else if let Some(arr) = content.as_array_mut() {
                    // Anthropic/OpenAI array format: prepend as text block
                    arr.insert(0, serde_json::json!({"type": "text", "text": warning}));
                    break;
                }
            }
        }
    }

    Ok(Bytes::from(serde_json::to_vec(&json)?))
}

pub(crate) fn sse_extract_deltas(sse_buf: &mut Vec<u8>, assistant: &mut String) {
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

        // Anthropic SSE stream
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
    }
}
