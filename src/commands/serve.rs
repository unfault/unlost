pub(crate) async fn run(
    bind: String,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let bind = crate::cli::parse_bind(&bind)?;
    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;
    crate::http_proxy::run_serve(bind, embedder).await
}
