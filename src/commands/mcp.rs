//! `unlost mcp serve` — MCP stdio server entry point.
//!
//! Starts an MCP server on stdin/stdout. Each workspace invocation is scoped
//! to the git toplevel of the directory where the server is launched.

use crate::companion::mcp::{UnlostMcpServer, server::McpServerConfig};

pub async fn run(
    allow_writes: bool,
    no_cross_workspace: bool,
    workspace: String,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    // Resolve workspace root.
    let workspace_root = if workspace == "." {
        let cwd = std::env::current_dir()?;
        crate::workspace::git_toplevel(&cwd)
            .unwrap_or_else(|| crate::workspace::canonicalize_dir(&cwd).unwrap_or(cwd))
    } else {
        let p = std::path::PathBuf::from(&workspace);
        crate::workspace::git_toplevel(&p)
            .unwrap_or_else(|| crate::workspace::canonicalize_dir(&p).unwrap_or(p))
    };

    let ws = crate::workspace::get_or_create_workspace_paths(&workspace_root)?;

    let config = McpServerConfig {
        workspace_root,
        ws,
        allow_writes,
        cross_workspace: !no_cross_workspace,
        embed_model,
        embed_cache_dir: embed_cache_dir.map(std::path::PathBuf::from),
    };

    let server = UnlostMcpServer::new(config);

    tracing::info!(
        allow_writes,
        cross_workspace = !no_cross_workspace,
        "starting unlost MCP server (stdio)"
    );

    // Run MCP server over stdio.
    // rmcp::transport::io::stdio() returns (Stdin, Stdout) as a ready-to-use transport pair.
    let transport = rmcp::transport::io::stdio();

    let running = rmcp::serve_server(server, transport).await?;

    // Wait for the client to disconnect (returns when stdin closes).
    running.waiting().await.ok();

    Ok(())
}
