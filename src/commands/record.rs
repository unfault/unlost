pub(crate) async fn run(
    bind: String,
    upstream_host: String,
    upstream_port: u16,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    if upstream_host.trim().is_empty() {
        anyhow::bail!("missing upstream host (set --upstream-host or UNLOST_UPSTREAM_HOST)");
    }
    let bind = crate::cli::parse_bind(&bind)?;
    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;
    let ws = crate::workspace::get_or_create_workspace_paths(&std::env::current_dir()?)?;
    crate::http_proxy::run_proxy(bind, upstream_host, upstream_port, embedder, ws).await
}
