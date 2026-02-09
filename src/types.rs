use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub(crate) struct UsageMeta {
    pub(crate) provider_id: Option<String>,
    pub(crate) model_id: Option<String>,
    pub(crate) cost: Option<f64>,
    pub(crate) tokens_input: Option<i64>,
    pub(crate) tokens_output: Option<i64>,
    pub(crate) tokens_reasoning: Option<i64>,
    pub(crate) tokens_cache_read: Option<i64>,
    pub(crate) tokens_cache_write: Option<i64>,
}

impl UsageMeta {
    pub(crate) fn tokens_total(&self) -> Option<i64> {
        let sum = self.tokens_input.unwrap_or(0)
            + self.tokens_output.unwrap_or(0)
            + self.tokens_reasoning.unwrap_or(0);
        if sum > 0 {
            Some(sum)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResponseMeta {
    pub(crate) source: String,
    pub(crate) upstream_host: String,
    pub(crate) request_path: String,
    pub(crate) http_status: u16,
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    pub(crate) agent_session_id: Option<String>,
    /// Best-effort usage metrics (tokens/cost). Not always present.
    pub(crate) usage: Option<UsageMeta>,
}

/// Failure modes that unlost can detect in agent conversations.
/// See internal/DEVELOPMENT.md for detailed definitions.
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureMode {
    /// No failure mode detected
    None,
    /// Agent believes the system works one way, but code says otherwise
    Drift,
    /// Same lessons/decisions being re-explained across sessions
    Rediscovery,
    /// Agent attempts an approach that conflicts with an established project decision
    DecisionConflict,
    /// Agent trying the same failed approach repeatedly
    RetrySpiral,
    /// Agent claims done but verification would fail
    FalseProgress,
    /// Agent wandering into unrelated side-quests
    UnboundedHorizon,
}

impl Default for FailureMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
pub struct IntentCapsule {
    pub category: String,
    pub intent: String,
    pub decision: String,
    pub rationale: String,
    pub next_steps: Vec<String>,
    pub symbols: Vec<String>,
    /// Detected failure mode: none, drift, rediscovery, decision_conflict, retry_spiral, false_progress, or unbounded_horizon
    #[serde(default)]
    #[schemars(schema_with = "failure_mode_schema")]
    pub failure_mode: FailureMode,
    /// Brief explanation of why this failure mode was detected (null if failure_mode is none)
    #[serde(default)]
    pub failure_signals: Option<String>,
}

fn failure_mode_schema(_gen: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": "string",
        "enum": ["none", "drift", "rediscovery", "decision_conflict", "retry_spiral", "false_progress", "unbounded_horizon"]
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_intent_capsule_serialization() {
        let capsule = IntentCapsule {
            category: "test_category".to_string(),
            intent: "test_intent".to_string(),
            decision: "test_decision".to_string(),
            rationale: "test_rationale".to_string(),
            next_steps: vec!["step1".to_string(), "step2".to_string()],
            symbols: vec!["symbol1".to_string(), "symbol2".to_string()],
            failure_mode: FailureMode::None,
            failure_signals: None,
        };

        let json = serde_json::to_string(&capsule).unwrap();
        let parsed: IntentCapsule = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.category, "test_category");
        assert_eq!(parsed.intent, "test_intent");
        assert_eq!(parsed.decision, "test_decision");
        assert_eq!(parsed.rationale, "test_rationale");
        assert_eq!(
            parsed.next_steps,
            vec!["step1".to_string(), "step2".to_string()]
        );
        assert_eq!(
            parsed.symbols,
            vec!["symbol1".to_string(), "symbol2".to_string()]
        );
        assert_eq!(parsed.failure_mode, FailureMode::None);
        assert!(parsed.failure_signals.is_none());
    }

    #[test]
    fn test_intent_capsule_with_failure_mode() {
        let capsule = IntentCapsule {
            category: "debugging".to_string(),
            intent: "fix auth bug".to_string(),
            decision: "retry same approach".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["auth.rs".to_string()],
            failure_mode: FailureMode::RetrySpiral,
            failure_signals: Some(
                "User expressed frustration, same symbols touched 3 times".to_string(),
            ),
        };

        let json = serde_json::to_string(&capsule).unwrap();
        let parsed: IntentCapsule = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.failure_mode, FailureMode::RetrySpiral);
        assert_eq!(
            parsed.failure_signals,
            Some("User expressed frustration, same symbols touched 3 times".to_string())
        );
    }

    #[test]
    fn test_failure_mode_serialization() {
        // Test all variants serialize correctly
        assert_eq!(
            serde_json::to_string(&FailureMode::None).unwrap(),
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::Drift).unwrap(),
            "\"drift\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::Rediscovery).unwrap(),
            "\"rediscovery\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::DecisionConflict).unwrap(),
            "\"decision_conflict\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::RetrySpiral).unwrap(),
            "\"retry_spiral\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::FalseProgress).unwrap(),
            "\"false_progress\""
        );
        assert_eq!(
            serde_json::to_string(&FailureMode::UnboundedHorizon).unwrap(),
            "\"unbounded_horizon\""
        );
    }

    #[test]
    fn test_failure_mode_deserialization() {
        assert_eq!(
            serde_json::from_str::<FailureMode>("\"none\"").unwrap(),
            FailureMode::None
        );
        assert_eq!(
            serde_json::from_str::<FailureMode>("\"drift\"").unwrap(),
            FailureMode::Drift
        );
        assert_eq!(
            serde_json::from_str::<FailureMode>("\"decision_conflict\"").unwrap(),
            FailureMode::DecisionConflict
        );
        assert_eq!(
            serde_json::from_str::<FailureMode>("\"retry_spiral\"").unwrap(),
            FailureMode::RetrySpiral
        );
    }

    #[test]
    fn test_init_capsules_output_serialization() {
        let capsules = vec![IntentCapsule {
            category: "category1".to_string(),
            intent: "intent1".to_string(),
            decision: "decision1".to_string(),
            rationale: "rationale1".to_string(),
            next_steps: vec!["action1".to_string()],
            symbols: vec!["symbol1".to_string()],
            failure_mode: FailureMode::None,
            failure_signals: None,
        }];

        let output = InitCapsulesOutput {
            debrief: "Test debrief".to_string(),
            capsules,
        };

        let json = serde_json::to_string(&output).unwrap();
        let parsed: InitCapsulesOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.debrief, "Test debrief");
        assert_eq!(parsed.capsules.len(), 1);
        assert_eq!(parsed.capsules[0].category, "category1");
    }

    #[test]
    fn test_query_narrative_output_serialization() {
        let output = QueryNarrativeOutput {
            narrative: "Test narrative response".to_string(),
        };

        let json = serde_json::to_string(&output).unwrap();
        let parsed: QueryNarrativeOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.narrative, "Test narrative response");
    }

    #[test]
    fn test_response_meta_creation() {
        let meta = ResponseMeta {
            source: "test_source".to_string(),
            upstream_host: "test.example.com".to_string(),
            request_path: "/api/test".to_string(),
            http_status: 200,
            agent_session_id: None,
            usage: None,
        };

        assert_eq!(meta.source, "test_source");
        assert_eq!(meta.upstream_host, "test.example.com");
        assert_eq!(meta.request_path, "/api/test");
        assert_eq!(meta.http_status, 200);
        assert!(meta.agent_session_id.is_none());
    }

    #[test]
    fn test_capsule_hit_creation() {
        let capsule = IntentCapsule {
            category: "test".to_string(),
            intent: "test".to_string(),
            decision: "test".to_string(),
            rationale: "test".to_string(),
            next_steps: vec![],
            symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
        };

        let meta = ResponseMeta {
            source: "test".to_string(),
            upstream_host: "test.com".to_string(),
            request_path: "/test".to_string(),
            http_status: 200,
            agent_session_id: Some("test_session".to_string()),
            usage: None,
        };

        let hit = CapsuleHit {
            id: "test_id".to_string(),
            ts_ms: 1234567890,
            conn_id: 1,
            exchange_seq: 1,
            distance: 0.5,
            user_emotion: None,
            assistant_emotion: None,
            capsule,
            meta,
        };

        assert_eq!(hit.id, "test_id");
        assert_eq!(hit.ts_ms, 1234567890);
        assert_eq!(hit.conn_id, 1);
        assert_eq!(hit.exchange_seq, 1);
        assert_eq!(hit.distance, 0.5);
        assert!(hit.user_emotion.is_none());
        assert!(hit.assistant_emotion.is_none());
    }

    #[test]
    fn test_intent_capsule_empty_fields() {
        let capsule = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
        };

        let json = serde_json::to_string(&capsule).unwrap();
        let parsed: IntentCapsule = serde_json::from_str(&json).unwrap();

        assert!(parsed.category.is_empty());
        assert!(parsed.intent.is_empty());
        assert!(parsed.decision.is_empty());
        assert!(parsed.rationale.is_empty());
        assert!(parsed.next_steps.is_empty());
        assert!(parsed.symbols.is_empty());
        assert_eq!(parsed.failure_mode, FailureMode::None);
    }

    #[test]
    fn test_intent_capsule_deserialize_without_failure_mode() {
        // Old capsules without failure_mode fields should still deserialize
        let json = r#"{
            "category": "test",
            "intent": "test intent",
            "decision": "test decision",
            "rationale": "test rationale",
            "next_steps": [],
            "symbols": []
        }"#;
        let parsed: IntentCapsule = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.failure_mode, FailureMode::None);
        assert!(parsed.failure_signals.is_none());
    }
}
