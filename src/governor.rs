//! Friction-based loop detection using existing signals.
//!
//! No frameworks, no state machines - just cheap heuristics:
//! - User emotion (frustration/anger/annoyance/disapproval from ONNX model)
//! - Symbol repetition (same files touched repeatedly)
//!
//! Total overhead: <15ms (local emotion + LanceDB query + matching)

use crate::CapsuleHit;
use crate::IntentCapsule;

/// Emotions that indicate friction - user is frustrated, disapproves, doubts, or expresses dissatisfaction
/// Note: "sad" in agent context usually means "not happy with this" = dissatisfaction, not actual sadness
const FRICTION_EMOTIONS: &[&str] = &[
    "frustration",
    "anger",
    "annoyance",
    "disapproval",
    "doubt",
    "sad",
];

/// Check if the current request is repeating a failed pattern.
///
/// Returns `Some(warning)` to inject into the request, or `None` if all clear.
pub fn evaluate_friction(current: &IntentCapsule, history: &[CapsuleHit]) -> Option<String> {
    if history.is_empty() || current.symbols.is_empty() {
        return None;
    }

    // Signal 1: User frustration in recent turns
    let frustrated = history.iter().take(3).any(|h| {
        h.user_emotion
            .as_ref()
            .map(|e| FRICTION_EMOTIONS.contains(&e.label.as_str()))
            .unwrap_or(false)
    });

    // Signal 2: Same symbols touched 2+ times recently
    let symbol_repeats = history
        .iter()
        .take(3)
        .filter(|h| {
            h.capsule
                .symbols
                .iter()
                .any(|s| current.symbols.contains(s))
        })
        .count();

    // Trigger: User is frustrated AND agent is stuck in the same code area
    if frustrated && symbol_repeats >= 2 {
        let symbols_str = current.symbols.join(", ");
        return Some(format!(
            "[SYSTEM NOTE: Your previous attempts involving {} have not succeeded \
and the user is frustrated. Stop this approach. Propose a different strategy.]\n\n",
            symbols_str
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emotion::EmotionMeta;
    use crate::types::ResponseMeta;

    fn make_hit(symbols: Vec<&str>, emotion: Option<&str>) -> CapsuleHit {
        CapsuleHit {
            id: "test".to_string(),
            ts_ms: 0,
            conn_id: 0,
            exchange_seq: 0,
            distance: 0.0,
            user_emotion: emotion.map(|label| EmotionMeta {
                label: label.to_string(),
                valence: -0.5,
                intensity: 0.8,
                confidence: 0.9,
            }),
            assistant_emotion: None,
            capsule: IntentCapsule {
                category: "bugfix".to_string(),
                intent: "fix the bug".to_string(),
                decision: "modify auth".to_string(),
                rationale: "".to_string(),
                next_steps: vec![],
                symbols: symbols.into_iter().map(String::from).collect(),
            },
            meta: ResponseMeta {
                source: "test".to_string(),
                upstream_host: "test".to_string(),
                request_path: "test".to_string(),
                http_status: 200,
            },
        }
    }

    #[test]
    fn test_no_friction_empty_history() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["auth.ts".to_string()],
        };
        assert!(evaluate_friction(&current, &[]).is_none());
    }

    #[test]
    fn test_no_friction_empty_symbols() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec![],
        };
        let history = vec![make_hit(vec!["auth.ts"], Some("frustration"))];
        assert!(evaluate_friction(&current, &history).is_none());
    }

    #[test]
    fn test_no_friction_happy_user() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["auth.ts".to_string()],
        };
        let history = vec![
            make_hit(vec!["auth.ts"], Some("joy")),
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], None),
        ];
        assert!(evaluate_friction(&current, &history).is_none());
    }

    #[test]
    fn test_no_friction_different_symbols() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["utils.ts".to_string()],
        };
        let history = vec![
            make_hit(vec!["auth.ts"], Some("frustration")),
            make_hit(vec!["auth.ts"], Some("anger")),
        ];
        assert!(evaluate_friction(&current, &history).is_none());
    }

    #[test]
    fn test_friction_detected() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["auth.ts".to_string()],
        };
        let history = vec![
            make_hit(vec!["auth.ts"], Some("frustration")),
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], None),
        ];
        let result = evaluate_friction(&current, &history);
        assert!(result.is_some());
        let warning = result.unwrap();
        assert!(warning.contains("auth.ts"));
        assert!(warning.contains("SYSTEM NOTE"));
    }

    #[test]
    fn test_friction_with_annoyance() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["db.rs".to_string()],
        };
        let history = vec![
            make_hit(vec!["db.rs"], Some("annoyance")),
            make_hit(vec!["db.rs", "models.rs"], None),
        ];
        let result = evaluate_friction(&current, &history);
        assert!(result.is_some());
    }
}
