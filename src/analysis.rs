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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_meta_creation() {
        let meta = AnalysisMeta {
            workspace_id: "wks_test123".to_string(),
            upstream_host: "api.openai.com".to_string(),
            request_path: "/v1/chat/completions".to_string(),
            http_status: 200,
            content_type: Some("application/json".to_string()),
        };

        assert_eq!(meta.workspace_id, "wks_test123");
        assert_eq!(meta.upstream_host, "api.openai.com");
        assert_eq!(meta.http_status, 200);
        assert_eq!(meta.content_type, Some("application/json".to_string()));
    }

    #[test]
    fn test_analysis_meta_no_content_type() {
        let meta = AnalysisMeta {
            workspace_id: "wks_abc".to_string(),
            upstream_host: "api.anthropic.com".to_string(),
            request_path: "/v1/messages".to_string(),
            http_status: 404,
            content_type: None,
        };

        assert!(meta.content_type.is_none());
    }

    #[test]
    fn test_conn_seq_atomic() {
        let initial = CONN_SEQ.load(std::sync::atomic::Ordering::SeqCst);
        CONN_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let after = CONN_SEQ.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, initial + 1);
    }

    #[test]
    fn test_analysis_msg_exchange_start() {
        let meta = AnalysisMeta {
            workspace_id: "wks_test".to_string(),
            upstream_host: "test.com".to_string(),
            request_path: "/test".to_string(),
            http_status: 200,
            content_type: None,
        };
        let body = Bytes::from(r#"{"test": "value"}"#);

        let msg = AnalysisMsg::ExchangeStart {
            meta: meta.clone(),
            request_body: body.clone(),
        };

        match msg {
            AnalysisMsg::ExchangeStart {
                meta: m,
                request_body: b,
            } => {
                assert_eq!(m.workspace_id, "wks_test");
                assert_eq!(b, body);
            }
            _ => panic!("Expected ExchangeStart variant"),
        }
    }

    #[test]
    fn test_analysis_msg_response_chunk() {
        let chunk = Bytes::from("test chunk data");
        let msg = AnalysisMsg::ResponseChunk(chunk.clone());
        match msg {
            AnalysisMsg::ResponseChunk(data) => {
                assert_eq!(data, chunk);
            }
            _ => panic!("Expected ResponseChunk variant"),
        }
    }

    #[test]
    fn test_analysis_msg_response_end() {
        let msg = AnalysisMsg::ResponseEnd;
        match msg {
            AnalysisMsg::ResponseEnd => {}
            _ => panic!("Expected ResponseEnd variant"),
        }
    }

    #[test]
    fn test_analysis_meta_clone() {
        let meta = AnalysisMeta {
            workspace_id: "wks_clone".to_string(),
            upstream_host: "clone.com".to_string(),
            request_path: "/clone".to_string(),
            http_status: 201,
            content_type: Some("text/plain".to_string()),
        };
        let cloned = meta.clone();
        assert_eq!(meta.workspace_id, cloned.workspace_id);
        assert_eq!(meta.http_status, cloned.http_status);
    }
}
