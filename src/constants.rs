// fastembed uses "model_code" strings (e.g. Xenova/*, Qdrant/*) for FromStr.
// For BGE small, fastembed's model_code is Xenova/bge-small-en-v1.5.
pub(crate) const DEFAULT_EMBED_MODEL: &str = "Xenova/bge-small-en-v1.5";
pub(crate) const DEFAULT_EMBED_DIM: usize = 384;
