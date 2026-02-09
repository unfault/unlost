use anyhow::Context;
use fastembed::{EmbeddingModel, InitOptions as FastEmbedInitOptions, TextEmbedding};
use std::sync::Arc;

pub(crate) type Embedder = Arc<std::sync::Mutex<TextEmbedding>>;

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
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" | "bge-small-en" => {
            crate::constants::DEFAULT_EMBED_MODEL
        }

        // Enum variant names people might guess
        "bgesmallenv15" => crate::constants::DEFAULT_EMBED_MODEL,
        "bgesmallenv15q" => "Qdrant/bge-small-en-v1.5-onnx-Q",
        _ => raw,
    };

    aliased
        .parse::<EmbeddingModel>()
        .map_err(|e| anyhow::anyhow!("unknown embedding model '{raw}': {e}"))
}

pub(crate) async fn load_embedder(
    model: &str,
    cache_dir: Option<std::path::PathBuf>,
    show_progress: bool,
) -> anyhow::Result<Embedder> {
    let model = parse_embed_model(model)?;
    let cache_dir = cache_dir.unwrap_or_else(default_embed_cache_dir);

    let info = TextEmbedding::get_model_info(&model)?;
    if info.dim != crate::constants::DEFAULT_EMBED_DIM {
        anyhow::bail!(
            "embedding dimension mismatch: {} (expected {})",
            info.dim,
            crate::constants::DEFAULT_EMBED_DIM
        );
    }

    tracing::info!(model = %model, cache_dir = %cache_dir.display(), "loading local embedder");

    let options = FastEmbedInitOptions::new(model)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(show_progress);

    let embedder = tokio::task::spawn_blocking(move || TextEmbedding::try_new(options))
        .await
        .context("embedding init task failed")??;

    Ok(Arc::new(std::sync::Mutex::new(embedder)))
}

pub(crate) async fn embed_text(embedder: &Embedder, text: &str) -> anyhow::Result<Vec<f32>> {
    let embedder = embedder.clone();
    let text = text.to_string();
    tokio::task::spawn_blocking(move || {
        let mut model = embedder
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder lock poisoned"))?;
        let mut out = model.embed([text], Some(1))?;
        out.pop()
            .ok_or_else(|| anyhow::anyhow!("embedding model returned no vectors"))
    })
    .await
    .context("embedding task failed")?
}

pub(crate) async fn download_model(
    model: &str,
    cache_dir: Option<&str>,
    force: bool,
) -> anyhow::Result<std::path::PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::test_support::ENV_LOCK;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, val: &std::ffi::OsStr) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn test_default_embed_cache_dir_with_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("UNLOST_EMBED_CACHE_DIR", temp_dir.path().as_os_str());
        assert_eq!(default_embed_cache_dir(), temp_dir.path());
    }

    #[test]
    fn test_default_embed_cache_dir_xdg_fallback() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("XDG_DATA_HOME", temp_dir.path().as_os_str());
        let _g2 = EnvVarGuard::remove("UNLOST_EMBED_CACHE_DIR");
        let expected = temp_dir
            .path()
            .join("unlost")
            .join("models")
            .join("fastembed");
        assert_eq!(default_embed_cache_dir(), expected);
    }

    #[test]
    fn test_default_embed_cache_dir_home_fallback() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("HOME", temp_dir.path().as_os_str());
        let _g2 = EnvVarGuard::remove("XDG_DATA_HOME");
        let _g3 = EnvVarGuard::remove("UNLOST_EMBED_CACHE_DIR");
        let expected = temp_dir
            .path()
            .join(".local")
            .join("share")
            .join("unlost")
            .join("models")
            .join("fastembed");
        assert_eq!(default_embed_cache_dir(), expected);
    }

    #[test]
    fn test_parse_embed_model_aliases() {
        assert_eq!(
            parse_embed_model("bge-small-en-v1.5").unwrap(),
            parse_embed_model("baai/bge-small-en-v1.5").unwrap()
        );
        assert_eq!(
            parse_embed_model("BGE-small-en").unwrap(),
            parse_embed_model("Xenova/bge-small-en-v1.5").unwrap()
        );
    }

    #[test]
    fn test_parse_embed_model_case_insensitive() {
        let result1 = parse_embed_model("Xenova/bge-small-en-v1.5").unwrap();
        let result2 = parse_embed_model("xenova/bge-small-en-v1.5").unwrap();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_parse_embed_model_q_variant() {
        let q_model = parse_embed_model("bgesmallenv15q").unwrap();
        let expected = "Qdrant/bge-small-en-v1.5-onnx-Q";
        assert_eq!(format!("{}", q_model), expected);
    }

    #[test]
    fn test_parse_embed_model_invalid() {
        let result = parse_embed_model("nonexistent-model-xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_embed_model_whitespace() {
        let result = parse_embed_model("  Xenova/bge-small-en-v1.5  ");
        assert!(result.is_ok());
    }
}
