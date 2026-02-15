use crate::cli::ModelCommand;

pub async fn run(command: ModelCommand) -> anyhow::Result<()> {
    match command {
        ModelCommand::Download {
            embed_model,
            cache_dir,
            force,
        } => {
            let dir =
                crate::embed::download_model(&embed_model, cache_dir.as_deref(), force).await?;
            println!("downloaded into: {}", dir.display());
            Ok(())
        }
    }
}
