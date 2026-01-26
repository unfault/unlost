use bytes::Bytes;
use std::sync::atomic::AtomicU64;

pub(crate) static CONN_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub(crate) struct AnalysisMeta {
    pub(crate) workspace_id: String,
    pub(crate) upstream_host: String,
    pub(crate) request_path: String,
    pub(crate) http_status: u16,
    pub(crate) content_type: Option<String>,
}

#[derive(Debug)]
pub(crate) enum AnalysisMsg {
    ExchangeStart {
        meta: AnalysisMeta,
        request_body: Bytes,
    },
    ResponseChunk(Bytes),
    ResponseEnd,
}
