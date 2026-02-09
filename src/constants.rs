// fastembed uses "model_code" strings (e.g. Xenova/*, Qdrant/*) for FromStr.
// For BGE small, fastembed's model_code is Xenova/bge-small-en-v1.5.
pub(crate) const DEFAULT_EMBED_MODEL: &str = "Xenova/bge-small-en-v1.5";
pub(crate) const DEFAULT_EMBED_DIM: usize = 384;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_embed_model() {
        assert!(!DEFAULT_EMBED_MODEL.is_empty());
        assert_eq!(DEFAULT_EMBED_MODEL, "Xenova/bge-small-en-v1.5");
    }

    #[test]
    fn test_default_embed_dim() {
        assert_eq!(DEFAULT_EMBED_DIM, 384);
    }

    #[test]
    fn test_constants_consistency() {
        // Verify that the model name and dimension are consistent
        // For BGE small model, dimension should be 384
        assert_eq!(DEFAULT_EMBED_MODEL, "Xenova/bge-small-en-v1.5");
        assert_eq!(DEFAULT_EMBED_DIM, 384);
    }
}
