//! Minimal integration tests for the unlost MCP server.
//!
//! Run with: cargo test --test mcp_tools

use tempfile::TempDir;

fn make_server(tmp: &TempDir, allow_writes: bool) -> unlost::companion::mcp::UnlostMcpServer {
    let workspace_root = tmp.path().to_path_buf();
    let ws_data = workspace_root.join(".unlost_test_mcp");
    std::fs::create_dir_all(ws_data.join("lancedb")).unwrap();

    let ws = unlost::WorkspacePaths {
        id: "test-workspace-mcp".to_string(),
        root: workspace_root.clone(),
        db_dir: ws_data.join("lancedb"),
        capsules_jsonl: ws_data.join("capsules.jsonl"),
        metrics_jsonl: ws_data.join("metrics.jsonl"),
    };

    let config = unlost::companion::mcp::server::McpServerConfig {
        workspace_root,
        ws,
        allow_writes,
        cross_workspace: false,
        embed_model: unlost::DEFAULT_EMBED_MODEL.to_string(),
        embed_cache_dir: None,
    };

    unlost::companion::mcp::UnlostMcpServer::new(config)
}

/// Server info must identify as "unlost" with the crate version.
#[test]
fn test_server_get_info() {
    use rmcp::ServerHandler;
    let tmp = TempDir::new().unwrap();
    let info = make_server(&tmp, false).get_info();
    assert_eq!(info.server_info.name, "unlost");
    assert!(!info.server_info.version.is_empty());
    let instructions = info.instructions.unwrap_or_default();
    assert!(instructions.contains("unlost_recall"));
}

/// The config allow_writes flag is honoured — visible via the config field.
#[test]
fn test_writes_gate_via_config() {
    let tmp = TempDir::new().unwrap();
    // Reads-only server
    let server = make_server(&tmp, false);
    assert!(!server.allow_writes(), "expect writes disabled");
    // Writes-enabled server
    let server = make_server(&tmp, true);
    assert!(server.allow_writes(), "expect writes enabled");
}

/// `unlost mcp serve` starts, accepts an initialize request, and returns a response.
#[test]
fn test_mcp_serve_initialize() {
    use std::io::{BufRead, BufReader, Write};
    use std::time::Duration;
    use wait_timeout::ChildExt;

    let bin = env!("CARGO_BIN_EXE_unlost");
    let tmp = TempDir::new().unwrap();

    let mut child = std::process::Command::new(bin)
        .args(["mcp", "serve"])
        .env("XDG_DATA_HOME", tmp.path())
        .env("XDG_CONFIG_HOME", tmp.path())
        .current_dir(tmp.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn unlost mcp serve");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let init_req = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0" }
        }
    });
    writeln!(stdin, "{init_req}").ok();
    drop(stdin);

    // Read until we get a JSON-RPC response or 8 seconds pass.
    let deadline = std::time::Instant::now() + Duration::from_secs(8);
    let mut got_response = false;
    loop {
        if std::time::Instant::now() > deadline { break; }
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 { break; }
        let t = line.trim();
        if t.is_empty() || t.starts_with("Content-") { continue; }
        if t.starts_with('{') {
            let v: serde_json::Value = serde_json::from_str(t).unwrap_or_default();
            if v.get("result").is_some() || v.get("error").is_some() {
                got_response = true;
                // Verify the result identifies as "unlost"
                if let Some(result) = v.get("result") {
                    let name = result
                        .get("serverInfo")
                        .and_then(|i| i.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    assert_eq!(name, "unlost", "server name mismatch in initialize response");
                }
            }
            break;
        }
    }

    let _ = child.wait_timeout(Duration::from_secs(3));
    let _ = child.kill();

    assert!(got_response, "expected a JSON-RPC response to initialize");
}
