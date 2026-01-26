use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub(crate) struct ResponseMeta {
    pub(crate) source: String,
    pub(crate) upstream_host: String,
    pub(crate) request_path: String,
    pub(crate) http_status: u16,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
pub struct IntentCapsule {
    pub category: String,
    pub intent: String,
    pub decision: String,
    pub rationale: String,
    pub next_steps: Vec<String>,
    pub symbols: Vec<String>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub(crate) struct InitCapsulesOutput {
    /// Short, colleague-style debrief to print to the user.
    pub(crate) debrief: String,
    pub(crate) capsules: Vec<IntentCapsule>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub(crate) struct QueryNarrativeOutput {
    pub(crate) narrative: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CapsuleHit {
    pub(crate) id: String,
    pub(crate) ts_ms: i64,
    pub(crate) conn_id: i64,
    pub(crate) exchange_seq: i64,
    pub(crate) distance: f32,
    pub(crate) user_emotion: Option<crate::emotion::EmotionMeta>,
    pub(crate) assistant_emotion: Option<crate::emotion::EmotionMeta>,
    pub(crate) capsule: IntentCapsule,
    pub(crate) meta: ResponseMeta,
}
