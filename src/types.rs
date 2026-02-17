use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct UsageMeta {
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub cost: Option<f64>,
    pub tokens_input: Option<i64>,
    pub tokens_output: Option<i64>,
    pub tokens_reasoning: Option<i64>,
    pub tokens_cache_read: Option<i64>,
    pub tokens_cache_write: Option<i64>,
}

impl UsageMeta {
    pub fn tokens_total(&self) -> Option<i64> {
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
pub struct ResponseMeta {
    pub source: String,
    pub upstream_host: String,
    pub request_path: String,
    pub http_status: u16,
    /// Agent session ID (e.g., OpenCode session) for grouping conversations
    pub agent_session_id: Option<String>,
    /// Best-effort usage metrics (tokens/cost). Not always present.
    pub usage: Option<UsageMeta>,
}

/// Failure modes that unlost can detect in agent conversations.
/// See internal/DEVELOPMENT.md for detailed definitions.
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FailureMode {
    /// No failure mode detected
    #[default]
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

/// State of the proactive friction regulation controller.
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryState {
    /// Normal iteration.
    #[default]
    Stable,
    /// Instability is rising; monitoring closely.
    Watch,
    /// High confidence loop detected; intervention required.
    Intervene,
}

/// Detailed instability metrics for a conversation turn.
#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, Default)]
pub struct InstabilityIntensity {
    /// Overall intensity score (0..1).
    pub intensity: f32,
    /// Rate of change in intensity.
    pub slope: f32,
    /// Individual symptom channel scores.
    pub channels: SymptomChannels,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, Default)]
pub struct SymptomChannels {
    pub repetition: f32,
    pub novelty_collapse: f32,
    pub semantic_stall: f32,
    pub effort_spike: f32,
    /// Measure of interaction volatility (repeated corrections).
    pub alignment_debt: f32,
    /// Measure of path hallucinations (referencing non-existent files).
    pub path_hallucination: f32,
    /// Measure of agent ignoring user-mentioned paths.
    pub grounding_stall: f32,
    /// Measure of user repeating long structural instructions.
    pub instruction_staticness: f32,
    /// Measure of rapid plan/decision changes without progress.
    pub logic_churn: f32,
    /// Measure of assistant verbosity vs user input (fluency/blind acceptance signal).
    pub fluency: f32,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMode {
    /// Skip all LLM extraction (fast, zero cost)
    None,
    /// LLM extract only pivotal turns (default research-backed mode)
    #[default]
    Hybrid,
    /// LLM extract every turn (slow, high quality)
    Full,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
pub struct IntentCapsule {
    pub category: String,
    pub intent: String,
    pub decision: String,
    pub rationale: String,
    pub next_steps: Vec<String>,
    pub symbols: Vec<String>,
    /// Symbols mentioned by the user in this turn
    #[serde(default)]
    pub user_symbols: Vec<String>,
    /// Detected failure mode: none, drift, rediscovery, decision_conflict, retry_spiral, false_progress, or unbounded_horizon
    #[serde(default)]
    #[schemars(schema_with = "failure_mode_schema")]
    pub failure_mode: FailureMode,
    /// Brief explanation of why this failure mode was detected (null if failure_mode is none)
    #[serde(default)]
    pub failure_signals: Option<String>,
    /// The extraction mode used for this capsule
    #[serde(default)]
    pub extraction_mode: ExtractionMode,
}

fn failure_mode_schema(_gen: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": "string",
        "enum": ["none", "drift", "rediscovery", "decision_conflict", "retry_spiral", "false_progress", "unbounded_horizon"]
    })
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub struct InitCapsulesOutput {
    /// Short, colleague-style debrief to print to the user.
    pub debrief: String,
    pub capsules: Vec<IntentCapsule>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub struct QueryNarrativeOutput {
    pub narrative: String,
}

#[derive(Debug, Clone)]
pub struct CapsuleHit {
    pub id: String,
    pub ts_ms: i64,
    pub conn_id: i64,
    pub exchange_seq: i64,
    pub distance: f32,
    pub user_emotion: Option<crate::emotion::EmotionMeta>,
    pub assistant_emotion: Option<crate::emotion::EmotionMeta>,
    pub capsule: IntentCapsule,
    pub meta: ResponseMeta,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_capsule_serialization() {
        let capsule = IntentCapsule {
            category: "test_category".to_string(),
            intent: "test_intent".to_string(),
            decision: "test_decision".to_string(),
            rationale: "test_rationale".to_string(),
            next_steps: vec!["step1".to_string(), "step2".to_string()],
            symbols: vec!["symbol1".to_string(), "symbol2".to_string()],
            user_symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
            extraction_mode: ExtractionMode::None,
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
            user_symbols: vec![],
            failure_mode: FailureMode::RetrySpiral,
            failure_signals: Some(
                "User expressed frustration, same symbols touched 3 times".to_string(),
            ),
            extraction_mode: ExtractionMode::None,
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
            user_symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
            extraction_mode: ExtractionMode::None,
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
            user_symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
            extraction_mode: ExtractionMode::None,
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
            user_symbols: vec![],
            failure_mode: FailureMode::None,
            failure_signals: None,
            extraction_mode: ExtractionMode::None,
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
