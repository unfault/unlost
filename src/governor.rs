//! Friction-based loop detection using existing signals.
//!
//! No frameworks, no state machines - just cheap heuristics:
//! - User emotion (frustration/anger/annoyance/disapproval from ONNX model)
//! - Symbol repetition (same files touched repeatedly)
//!
//! Total overhead: <15ms (local emotion + LanceDB query + matching)

use crate::types::{SymptomChannels, TrajectoryState};
use crate::CapsuleHit;
use crate::IntentCapsule;

/// Constants for the Trajectory Model (Calibrated Feb 15, 2026)
const WEIGHT_EFFORT: f32 = 0.34;
const WEIGHT_REPETITION: f32 = 0.24;
const WEIGHT_NOVELTY: f32 = 0.24;
const WEIGHT_SEMANTIC: f32 = 0.18;
const WEIGHT_ALIGNMENT_DEBT: f32 = 0.45;
const WEIGHT_PATH_HALLUCINATION: f32 = 0.60;
const WEIGHT_GROUNDING_STALL: f32 = 0.30;
const WEIGHT_INSTRUCTION_STATICNESS: f32 = 0.25;

const THRESHOLD_WATCH: f32 = 0.5;
const THRESHOLD_INTERVENE: f32 = 0.8;
const THRESHOLD_STABLE_OFF: f32 = 0.4;

const EMA_ALPHA: f32 = 0.3;
const COFFEE_PAUSE_MS: i64 = 30 * 60 * 1000;
const PERSISTENCE_WINDOW: usize = 3;
const PERSISTENCE_THRESHOLD: f32 = 0.75;
const COFFEE_PAUSE_DECAY: f32 = 0.3;

#[derive(Debug, Clone, Default)]
pub struct TrajectoryController {
    pub state: TrajectoryState,
    pub intensity: f32,
    pub smoothed_channels: SymptomChannels,
    pub last_ts_ms: i64,
    pub turns_since_intervention: usize,
    /// History of intensity values for persistence checking.
    pub intensity_history: std::collections::VecDeque<f32>,
    /// Tracks the last intervention type to avoid repetitive "nagging" oscillations.
    pub last_intervention_type: Option<String>,
    /// Persistence counter for grounding stall (consecutive turns).
    pub stall_streak: usize,
    /// Persistence counter for instruction staticness.
    pub static_streak: usize,
}

const CORRECTION_PATTERNS: &[&str] = &[
    "no",
    "not that",
    "i meant",
    "actually",
    "that's not what i asked",
    "you misunderstood",
    "wrong",
    "incorrect",
];

fn detect_correction(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut score = 0.0;
    for p in CORRECTION_PATTERNS {
        if lower.contains(p) {
            if *p == "actually" {
                score += 0.5;
            } else {
                score += 1.0;
            }
        }
    }
    (score / 1.5_f32).min(1.0_f32)
}

const SUMMARY_CUES: &[&str] = &[
    "summary",
    "recap",
    "summarize",
    "consolidate",
    "overview",
    "in short",
    "to conclude",
];

fn detect_summary_intent(text: &str) -> f32 {
    let lower = text.to_lowercase();
    for cue in SUMMARY_CUES {
        if lower.contains(cue) {
            return 1.0;
        }
    }
    0.0
}

fn extract_paths(text: &str) -> std::collections::HashSet<String> {
    let mut paths = std::collections::HashSet::new();
    // Simplified regex for files/paths
    let re = regex::Regex::new(r"[\w\-\./]+\.[a-z]{2,5}").unwrap();
    for cap in re.captures_iter(text) {
        paths.insert(cap[0].to_string());
    }
    paths
}

impl TrajectoryController {
    pub fn update(
        &mut self,
        workspace_id: &str,
        current: &IntentCapsule,
        current_emotion: Option<&crate::emotion::EmotionMeta>,
        history: &[CapsuleHit],
        ts_ms: i64,
    ) -> (TrajectoryState, Option<String>) {
        let mut reset_note = None;

        // 1. Check for Coffee Pause (Soft Decay Reset)
        if self.last_ts_ms > 0 && (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS {
            self.state = TrajectoryState::Stable;
            self.intensity *= COFFEE_PAUSE_DECAY;
            reset_note = render_resumption_brief(history);
        }
        self.last_ts_ms = ts_ms;

        // 2. Calculate raw symptoms
        let s_rep = calculate_repetition(current, history);
        let s_nov = calculate_novelty_collapse(current, history);
        let s_sem = calculate_semantic_stall(current, history);
        let s_eff = calculate_effort_spike(current, history);
        let s_corr = detect_correction(&current.intent);
        let s_path = calculate_path_hallucination(workspace_id, current);
        let s_summary = detect_summary_intent(&current.decision);

        // 2.1 Deep Drift Sensors
        let user_paths = extract_paths(&current.intent);
        let assistant_symbols: std::collections::HashSet<_> =
            current.symbols.iter().cloned().collect();

        // Grounding Stall: user mentions paths, agent touches disjoint set
        let has_stall =
            !user_paths.is_empty() && user_paths.intersection(&assistant_symbols).next().is_none();
        if has_stall {
            self.stall_streak += 1;
        } else {
            self.stall_streak = 0;
        }
        let s_stall = if self.stall_streak >= 2 { 1.0 } else { 0.0 };

        // Instruction Staticness: user repeats same long message
        let is_static = history.first().map_or(false, |h| {
            current.intent.len() > 50 && current.intent.trim() == h.capsule.intent.trim()
        });
        if is_static {
            self.static_streak += 1;
        } else {
            self.static_streak = 0;
        }
        let s_stat = if self.static_streak >= 2 { 1.0 } else { 0.0 };

        // 3. EMA Smoothing
        self.smoothed_channels.repetition =
            EMA_ALPHA * s_rep + (1.0 - EMA_ALPHA) * self.smoothed_channels.repetition;
        self.smoothed_channels.novelty_collapse =
            EMA_ALPHA * s_nov + (1.0 - EMA_ALPHA) * self.smoothed_channels.novelty_collapse;
        self.smoothed_channels.semantic_stall =
            EMA_ALPHA * s_sem + (1.0 - EMA_ALPHA) * self.smoothed_channels.semantic_stall;
        self.smoothed_channels.effort_spike =
            EMA_ALPHA * s_eff + (1.0 - EMA_ALPHA) * self.smoothed_channels.effort_spike;
        self.smoothed_channels.alignment_debt =
            EMA_ALPHA * s_corr + (1.0 - EMA_ALPHA) * self.smoothed_channels.alignment_debt;
        self.smoothed_channels.path_hallucination =
            EMA_ALPHA * s_path + (1.0 - EMA_ALPHA) * self.smoothed_channels.path_hallucination;
        self.smoothed_channels.grounding_stall =
            EMA_ALPHA * s_stall + (1.0 - EMA_ALPHA) * self.smoothed_channels.grounding_stall;
        self.smoothed_channels.instruction_staticness =
            EMA_ALPHA * s_stat + (1.0 - EMA_ALPHA) * self.smoothed_channels.instruction_staticness;

        // 4. Intensity Calculation
        let loop_intensity = WEIGHT_REPETITION * self.smoothed_channels.repetition
            + WEIGHT_NOVELTY * self.smoothed_channels.novelty_collapse
            + WEIGHT_SEMANTIC * self.smoothed_channels.semantic_stall
            + WEIGHT_EFFORT * self.smoothed_channels.effort_spike;

        let spec_intensity = WEIGHT_ALIGNMENT_DEBT * self.smoothed_channels.alignment_debt
            + WEIGHT_INSTRUCTION_STATICNESS * self.smoothed_channels.instruction_staticness;

        let drift_intensity = WEIGHT_PATH_HALLUCINATION * self.smoothed_channels.path_hallucination
            + WEIGHT_GROUNDING_STALL * self.smoothed_channels.grounding_stall;

        let mut raw_intensity = loop_intensity + spec_intensity + drift_intensity;

        // NEW: Summary Intent Damping (Prevents false positives during consolidation)
        if s_summary > 0.5 {
            raw_intensity *= 0.6;
        }

        // 5. Affective Modulation (The "Emotional Wave")
        if let Some(e) = current_emotion {
            match e.label.as_str() {
                "joy" if e.confidence > 0.7 => {
                    raw_intensity *= 0.5;
                }
                "anger" if e.confidence > 0.6 => {
                    raw_intensity = (raw_intensity + 0.3).min(1.0);
                }
                _ => {}
            }
        }

        let old_intensity = self.intensity;
        self.intensity = raw_intensity;
        let slope = self.intensity - old_intensity;

        // Track history for persistence
        self.intensity_history.push_back(self.intensity);
        if self.intensity_history.len() > PERSISTENCE_WINDOW {
            self.intensity_history.pop_front();
        }
        let is_persistent = self.intensity_history.len() == PERSISTENCE_WINDOW
            && self
                .intensity_history
                .iter()
                .all(|&v| v > PERSISTENCE_THRESHOLD);

        // 6. State Transitions
        let prev_state = self.state;
        match self.state {
            TrajectoryState::Stable => {
                if self.intensity > THRESHOLD_WATCH && slope > 0.0 {
                    self.state = TrajectoryState::Watch;
                }
            }
            TrajectoryState::Watch => {
                if (self.intensity > THRESHOLD_INTERVENE && slope > 0.05) || is_persistent {
                    self.state = TrajectoryState::Intervene;
                } else if self.intensity < THRESHOLD_STABLE_OFF {
                    self.state = TrajectoryState::Stable;
                }
            }
            TrajectoryState::Intervene => {
                self.turns_since_intervention = 0;
                self.state = TrajectoryState::Watch;
                self.intensity_history.clear();
            }
        }

        self.turns_since_intervention += 1;

        // 7. Select Intervention
        let mut note = if self.state != prev_state || self.state == TrajectoryState::Intervene {
            let cause = if drift_intensity > spec_intensity && drift_intensity > loop_intensity {
                "drift"
            } else if spec_intensity > loop_intensity {
                "spec"
            } else {
                "loop"
            };

            let intervention_type = format!("{}:{}", cause, self.state as u8);
            if self.last_intervention_type.as_ref() == Some(&intervention_type) {
                // One-Shot Rule: Don't repeat the exact same intervention type within the same episode
                None
            } else {
                self.last_intervention_type = Some(intervention_type);
                select_intervention_with_substance(
                    self.state,
                    current_emotion,
                    cause,
                    workspace_id,
                    current,
                    history,
                )
            }
        } else {
            None
        };

        if self.state == TrajectoryState::Stable {
            self.last_intervention_type = None;
        }

        // Prioritize reset note (Resumption Brief) over trajectory warnings if it just happened
        if reset_note.is_some() {
            note = reset_note;
        }

        (self.state, note)
    }

    pub fn reset(&mut self) {
        self.state = TrajectoryState::Stable;
        self.intensity = 0.0;
        self.smoothed_channels = SymptomChannels::default();
        self.turns_since_intervention = 0;
        self.intensity_history.clear();
        self.last_intervention_type = None;
    }
}

fn calculate_repetition(current: &IntentCapsule, history: &[CapsuleHit]) -> f32 {
    if current.symbols.is_empty() || history.is_empty() {
        return 0.0;
    }
    let mut recent_symbols = std::collections::HashSet::new();
    for h in history.iter().take(8) {
        for s in &h.capsule.symbols {
            recent_symbols.insert(s);
        }
    }
    let overlap = current
        .symbols
        .iter()
        .filter(|s| recent_symbols.contains(s))
        .count();
    overlap as f32 / current.symbols.len() as f32
}

fn calculate_novelty_collapse(current: &IntentCapsule, history: &[CapsuleHit]) -> f32 {
    1.0 - (1.0 - calculate_repetition(current, history))
}

fn calculate_path_hallucination(workspace_id: &str, current: &IntentCapsule) -> f32 {
    let (_checked, missing) = crate::workspace::validate_paths(workspace_id, &current.symbols);
    if current.symbols.is_empty() {
        return 0.0;
    }
    if missing > 0 {
        (missing as f32 / current.symbols.len() as f32).max(0.5)
    } else {
        0.0
    }
}

fn calculate_semantic_stall(_current: &IntentCapsule, history: &[CapsuleHit]) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    0.0
}

fn calculate_effort_spike(current: &IntentCapsule, history: &[CapsuleHit]) -> f32 {
    if history.is_empty() {
        return 0.0;
    }

    let current_eff = (current.symbols.len() * 100 + current.intent.len()) as f32;

    let mut total_prev_eff = 0.0;
    let count = history.iter().take(8).count();
    for h in history.iter().take(8) {
        total_prev_eff += (h.capsule.symbols.len() * 100 + h.capsule.intent.len()) as f32;
    }

    let avg_eff = total_prev_eff / count as f32;
    if avg_eff > 0.0 {
        (current_eff / avg_eff).min(2.0) / 2.0
    } else {
        0.5
    }
}

fn select_intervention_with_substance(
    state: TrajectoryState,
    emotion: Option<&crate::emotion::EmotionMeta>,
    cause: &str,
    workspace_id: &str,
    current: &IntentCapsule,
    history: &[CapsuleHit],
) -> Option<String> {
    let label = emotion.map(|e| e.label.as_str()).unwrap_or("neutral");
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    match (state, cause, label) {
        // --- Specification Basin (Staff Engineer Voice) ---
        (TrajectoryState::Watch, "spec", "confused" | "doubt") => {
            let rationale = current.rationale.trim();
            let rational_part = if !rationale.is_empty() {
                format!(" (Rationale was: \"{rationale}\")")
            } else {
                String::new()
            };
            Some(format!(
                "[SYSTEM NOTE: User appears confused by the direction{}. Briefly explain the 'why' behind your current approach and ask if this aligns with their intent before proceeding.]",
                rational_part
            ))
        }
        (TrajectoryState::Watch, "spec", _) => {
            let intent = current.intent.trim();
            let decision = current.decision.trim();
            Some(format!(
                "[SYSTEM NOTE: I noticed a few corrections recently. To ensure we're aligned: my current understanding is \"{}\". Next I'll do \"{}\". Does that sound right?]",
                intent, decision
            ))
        }
        (TrajectoryState::Intervene, "spec", _) => {
            let corrections: Vec<String> = recent
                .iter()
                .take(5)
                .filter(|h| detect_correction(&h.capsule.intent) > 0.5)
                .map(|h| h.capsule.intent.trim().to_string())
                .collect();

            let north_star = recent
                .iter()
                .rev()
                .find(|h| !h.capsule.intent.trim().is_empty())
                .map(|h| h.capsule.intent.trim());

            let mut note = "[SYSTEM NOTE: Alignment debt is high. Stop and restate the current objective. Ask the user to confirm or pivot before any more code is written.]".to_string();

            if let Some(ns) = north_star {
                note.push_str(&format!("\nOriginal Goal: \"{}\"", ns));
            }

            if !corrections.is_empty() {
                note.push_str("\nRecent corrections:\n- ");
                note.push_str(&corrections.join("\n- "));
            }
            Some(note)
        }

        // --- Drift Basin (Grounding/Hallucination) ---
        (TrajectoryState::Watch, "drift", _) => {
            let (_, missing_count) = crate::workspace::validate_paths(workspace_id, &current.symbols);
            if missing_count > 0 {
                Some("[SYSTEM NOTE: Potential drift detected. Some mentioned paths do not exist. Verify the workspace state before proceeding.]".to_string())
            } else {
                Some("[SYSTEM NOTE: High assumption load or grounding mismatch detected. Verify your facts about the codebase.]".to_string())
            }
        }
        (TrajectoryState::Intervene, "drift", _) => {
            Some("[SYSTEM NOTE: Factual drift is high. Stop. Re-read the relevant files and list 3 verified facts about the current code structure before continuing.]".to_string())
        }

        // --- Loop Basin (Hydration/Attempt Log) ---
        (TrajectoryState::Watch, "loop", "frustration") => Some(
            "[SYSTEM NOTE: User frustration detected in a potential loop. Pause to clarify the immediate blocker.]"
                .to_string(),
        ),
        (TrajectoryState::Watch, "loop", _) => {
            let syms = current.symbols.join(", ");
            Some(format!(
                "[SYSTEM NOTE: A lot of repeat activity detected in [{}]. If this approach is stalling, consider proposing an alternative.]",
                syms
            ))
        }
        (TrajectoryState::Intervene, "loop", "anger") => Some(
            "[SYSTEM NOTE: CRITICAL: Anger detected. Stop all execution. Apologize and await explicit instructions.]"
                .to_string(),
        ),
        (TrajectoryState::Intervene, "loop", _) => {
            let w = FrictionWeights::default();
            let packet = build_hydration_packet(current, emotion, &recent, &w);
            let symbols_str = current.symbols.join(", ");
            Some(render_hydration_warning(
                &symbols_str,
                String::new(),
                packet.as_ref(),
            ))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FrictionWeights {
    pub(crate) recent_window: usize,
    pub(crate) symbol_repeat_threshold: usize,
    pub(crate) graveyard_window: usize,
    pub(crate) map_window: usize,
    pub(crate) logic_recency_tau: f32,
    pub(crate) overlap_scale: f32,
    pub(crate) emotion_boost: f32,
    pub(crate) effort_tokens_norm: f32,
    pub(crate) effort_scale: f32,
    pub(crate) effort_tokens_warn_threshold: i64,
}

impl Default for FrictionWeights {
    fn default() -> Self {
        Self {
            recent_window: 3,
            symbol_repeat_threshold: 2,
            graveyard_window: 10,
            map_window: 10,
            logic_recency_tau: 3.0,
            overlap_scale: 0.85,
            emotion_boost: 1.35,
            effort_tokens_norm: 1500.0,
            effort_scale: 0.25,
            effort_tokens_warn_threshold: 1500,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HydrationPacket {
    pub(crate) north_star: Option<String>,
    pub(crate) graveyard: Vec<HydrationNode>,
    pub(crate) map: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct HydrationNode {
    pub(crate) symbols: Vec<String>,
    pub(crate) intent: String,
    pub(crate) decision: String,
    pub(crate) user_emotion: Option<String>,
    pub(crate) tokens_total: Option<i64>,
    pub(crate) failure_mode: crate::types::FailureMode,
}

const FRICTION_EMOTIONS: &[&str] = &[
    "frustration",
    "anger",
    "annoyance",
    "disapproval",
    "doubt",
    "sad",
    "confused",
];

pub fn detect_failure_keywords(text: &str) -> Option<crate::types::FailureMode> {
    let lower = text.to_lowercase();
    if lower.contains("try again")
        || lower.contains("circles")
        || lower.contains("same error")
        || lower.contains("retry")
    {
        return Some(crate::types::FailureMode::RetrySpiral);
    }
    if lower.contains("never mind")
        || lower.contains("forget it")
        || lower.contains("actually")
        || lower.contains("instead")
    {
        if lower.contains("wrong") || lower.contains("incorrect") || lower.contains("fact") {
            return Some(crate::types::FailureMode::Drift);
        }
    }
    None
}

pub(crate) fn evaluate_stateless_friction(
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
) -> Option<String> {
    let e = current_user_emotion?;
    if !FRICTION_EMOTIONS.contains(&e.label.as_str()) {
        return None;
    }
    if e.confidence < 0.45 || e.intensity < 0.35 {
        return None;
    }
    Some(
        "[SYSTEM NOTE: User appears frustrated or confused. Pause to acknowledge the concern and ask what they want to achieve next before continuing.]"
            .to_string(),
    )
}

pub fn evaluate_friction(
    current: &IntentCapsule,
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
    history: &[CapsuleHit],
) -> Option<String> {
    evaluate_friction_with_weights(
        current,
        current_user_emotion,
        history,
        &FrictionWeights::default(),
    )
}

pub(crate) fn evaluate_friction_with_weights(
    current: &IntentCapsule,
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
    history: &[CapsuleHit],
    w: &FrictionWeights,
) -> Option<String> {
    if history.is_empty() {
        return None;
    }
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));
    let current_friction = current_user_emotion
        .map(|e| FRICTION_EMOTIONS.contains(&e.label.as_str()))
        .unwrap_or(false);
    let recent_friction = recent.iter().take(w.recent_window).any(|h| {
        h.user_emotion
            .as_ref()
            .map(|e| FRICTION_EMOTIONS.contains(&e.label.as_str()))
            .unwrap_or(false)
    });
    let frustrated = current_friction || recent_friction;
    let symbol_repeats = if current.symbols.is_empty() {
        0
    } else {
        recent
            .iter()
            .take(w.recent_window)
            .filter(|h| {
                h.capsule
                    .symbols
                    .iter()
                    .any(|s| current.symbols.contains(s))
            })
            .count()
    };
    let effort_tokens: i64 = if current.symbols.is_empty() {
        0
    } else {
        recent
            .iter()
            .take(w.recent_window)
            .filter(|h| {
                h.capsule
                    .symbols
                    .iter()
                    .any(|s| current.symbols.contains(s))
            })
            .filter_map(|h| h.meta.usage.as_ref().and_then(|u| u.tokens_total()))
            .sum()
    };
    let keyword_failure = detect_failure_keywords(&current.intent)
        .or_else(|| detect_failure_keywords(&current.decision));
    if current.failure_mode != crate::types::FailureMode::None || keyword_failure.is_some() {
        let mode = if current.failure_mode != crate::types::FailureMode::None {
            current.failure_mode.clone()
        } else {
            keyword_failure.unwrap()
        };
        let mode_str = match mode {
            crate::types::FailureMode::Drift => "Drift detected (incorrect fact/structure).",
            crate::types::FailureMode::Rediscovery => {
                "Rediscovery detected (repeating old decisions)."
            }
            crate::types::FailureMode::DecisionConflict => {
                "Decision conflict (conflicts with project constraint)."
            }
            crate::types::FailureMode::RetrySpiral => "Retry spiral (stuck in a loop).",
            crate::types::FailureMode::FalseProgress => "False progress (claims done but broken).",
            crate::types::FailureMode::UnboundedHorizon => "Unbounded horizon (off-task tangents).",
            _ => "Failure mode detected.",
        };
        let _signals = current.failure_signals.as_deref().unwrap_or("");
        return Some(format!(
            "[SYSTEM NOTE: {} Pause to re-align with the user.]",
            mode_str
        ));
    }
    if !current.symbols.is_empty() && frustrated && symbol_repeats >= w.symbol_repeat_threshold {
        let packet = build_hydration_packet(current, current_user_emotion, &recent, w);
        let symbols_str = current.symbols.join(", ");
        let amplifier = if effort_tokens >= w.effort_tokens_warn_threshold {
            format!(
                " This loop likely already consumed ~{} tokens. Prioritize a different approach.",
                effort_tokens
            )
        } else {
            String::new()
        };
        return Some(render_hydration_warning(
            &symbols_str,
            amplifier,
            packet.as_ref(),
        ));
    }
    evaluate_conversational_friction_with_current(current_user_emotion, history, w.recent_window)
}

const SOFT_FRICTION_WARNING: &str =
    "The user seems frustrated or confused. Consider pausing to acknowledge their concern before continuing.";
const FIRM_FRICTION_WARNING: &str = "The user has expressed repeated frustration or confusion. Stop and ask what's wrong or how you can help differently.";

pub(crate) fn evaluate_conversational_friction_with_current(
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
    history: &[CapsuleHit],
    window: usize,
) -> Option<String> {
    if history.is_empty() && current_user_emotion.is_none() {
        return None;
    }
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));
    let current_negative = current_user_emotion
        .map(|e| crate::emotion::is_negative_emotion(&e.label))
        .unwrap_or(false);
    if current_user_emotion.is_some() && !current_negative {
        return None;
    }
    let include_current = current_user_emotion.is_some();
    let take_n = if include_current {
        window.saturating_sub(1)
    } else {
        window
    };
    let history_negative_count = recent
        .iter()
        .take(take_n)
        .filter(|h| h.user_emotion.is_some())
        .filter(|h| {
            h.user_emotion
                .as_ref()
                .map(|e| crate::emotion::is_negative_emotion(&e.label))
                .unwrap_or(false)
        })
        .count();
    let total_negative = history_negative_count + (current_negative as usize);
    if total_negative >= 3 {
        return Some(FIRM_FRICTION_WARNING.to_string());
    }
    if total_negative >= 2 {
        return Some(SOFT_FRICTION_WARNING.to_string());
    }
    None
}

const DRIFT_WARNING: &str = "Previous context may be stale or incorrect. The last exchange showed signs of drift (wrong mental model). Verify your assumptions about the codebase before proceeding.";
const FALSE_PROGRESS_WARNING: &str = "The user disputed completion in a recent exchange. Verify that your changes actually work before claiming the task is done.";
const REDISCOVERY_WARNING: &str = "This topic was already discussed recently. Check the prior decision before re-exploring the same ground.";
const DECISION_CONFLICT_WARNING: &str = "Intervention: This approach conflicts with a prior project decision. Re-route to the compliant pattern.";

pub(crate) fn evaluate_failure_modes(history: &[CapsuleHit]) -> Option<String> {
    if history.is_empty() {
        return None;
    }
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));
    let latest = recent.first()?;
    let failure_mode = &latest.capsule.failure_mode;
    let symbols = &latest.capsule.symbols;
    match failure_mode {
        crate::types::FailureMode::Drift => {
            let symbols_str = if symbols.is_empty() {
                String::new()
            } else {
                format!(" Relevant symbols: {}", symbols.join(", "))
            };
            Some(format!("{}{}", DRIFT_WARNING, symbols_str))
        }
        crate::types::FailureMode::DecisionConflict => {
            let prior_decision = if !latest.capsule.decision.is_empty() {
                format!(" Prior decision: \"{}\"", latest.capsule.decision)
            } else {
                String::new()
            };
            Some(format!("{}{}", DECISION_CONFLICT_WARNING, prior_decision))
        }
        crate::types::FailureMode::FalseProgress => Some(FALSE_PROGRESS_WARNING.to_string()),
        crate::types::FailureMode::Rediscovery => {
            let prior_decision = if !latest.capsule.decision.is_empty() {
                format!(" Prior decision: \"{}\"", latest.capsule.decision)
            } else {
                String::new()
            };
            Some(format!("{}{}", REDISCOVERY_WARNING, prior_decision))
        }
        _ => None,
    }
}

fn has_any_marker(s: &str, markers: &[&str]) -> bool {
    markers.iter().any(|m| s.contains(m))
}

fn tokenize_keywords(s: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for raw in s
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '/' && c != '.' && c != '-')
        .filter(|t| !t.is_empty())
    {
        let t = raw.to_ascii_lowercase();
        if t.len() < 4 {
            continue;
        }
        if t.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if matches!(
            t.as_str(),
            "that"
                | "this"
                | "with"
                | "from"
                | "have"
                | "will"
                | "into"
                | "your"
                | "then"
                | "than"
                | "when"
                | "where"
                | "what"
                | "which"
                | "should"
                | "could"
                | "would"
                | "about"
                | "please"
                | "using"
                | "because"
                | "avoid"
        ) {
            continue;
        }
        out.insert(t);
    }
    out
}

fn window_contains(hay: &str, needle: &str, idx: usize, window: usize) -> bool {
    let start = idx.saturating_sub(window);
    let end = (idx + needle.len() + window).min(hay.len());
    hay.get(start..end).unwrap_or("").contains(needle)
}

fn token_is_negated(prompt_lc: &str, token: &str) -> bool {
    const NEG: &[&str] = &[
        "do not",
        "don't",
        "dont",
        "avoid",
        "never",
        "must not",
        "should not",
        "no ",
    ];
    if let Some(idx) = prompt_lc.find(token) {
        for m in NEG {
            if window_contains(prompt_lc, m, idx, 28) {
                return true;
            }
        }
    }
    false
}

fn decision_prohibits_token(decision_lc: &str, token: &str) -> bool {
    const PROHIBIT: &[&str] = &[
        "do not",
        "don't",
        "dont",
        "avoid",
        "never",
        "must not",
        "should not",
        "not allowed",
        "forbidden",
        "ban",
        "banned",
        "no ",
    ];
    if let Some(idx) = decision_lc.find(token) {
        for m in PROHIBIT {
            if window_contains(decision_lc, m, idx, 40) {
                return true;
            }
        }
        if decision_lc.contains(&format!("no {token}")) {
            return true;
        }
    }
    false
}

fn prompt_requests_action(prompt_lc: &str) -> bool {
    const ACTION: &[&str] = &[
        "use ",
        "add ",
        "introduce ",
        "implement ",
        "switch ",
        "migrate ",
        "refactor ",
        "create ",
        "make ",
        "move ",
        "build ",
        "wire ",
        "route ",
        "rewrite ",
        "replace ",
    ];
    has_any_marker(prompt_lc, ACTION)
}

pub(crate) fn evaluate_decision_conflict(prompt: &str, hits: &[CapsuleHit]) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    if prompt_lc.trim().is_empty() {
        return None;
    }
    if !prompt_requests_action(&prompt_lc) {
        return None;
    }
    let best = hits
        .iter()
        .find(|h| !h.capsule.decision.trim().is_empty())?;
    let decision = best.capsule.decision.trim();
    let decision_lc = decision.to_ascii_lowercase();
    const PROHIBIT: &[&str] = &[
        "do not",
        "don't",
        "dont",
        "avoid",
        "never",
        "must not",
        "should not",
        "not allowed",
        "forbidden",
        "ban",
        "banned",
        "no ",
    ];
    if !has_any_marker(&decision_lc, PROHIBIT) {
        return None;
    }
    let p_tok = tokenize_keywords(&prompt_lc);
    let d_tok = tokenize_keywords(&decision_lc);
    let shared: Vec<String> = p_tok.intersection(&d_tok).cloned().collect();
    if shared.is_empty() {
        return None;
    }
    let mut conflict_token: Option<String> = None;
    for t in shared {
        if token_is_negated(&prompt_lc, &t) {
            continue;
        }
        if decision_prohibits_token(&decision_lc, &t) {
            conflict_token = Some(t);
            break;
        }
    }
    let _ = conflict_token?;
    let cat = best.capsule.category.trim();
    let cat_part = if cat.is_empty() {
        String::new()
    } else {
        format!(" ({cat})")
    };
    Some(format!(
        "{DECISION_CONFLICT_WARNING}{cat_part} Prior decision: \"{decision}\"\n\n"
    ))
}

fn is_friction_label(label: &str) -> bool {
    FRICTION_EMOTIONS.contains(&label)
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let uni = sa.union(&sb).count() as f32;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

fn build_hydration_packet(
    current: &IntentCapsule,
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
    recent: &[&CapsuleHit],
    w: &FrictionWeights,
) -> Option<HydrationPacket> {
    if recent.is_empty() {
        return None;
    }
    let north_star = if !current.intent.trim().is_empty() {
        Some(current.intent.trim().to_string())
    } else {
        recent.iter().find_map(|h| {
            (!h.capsule.intent.trim().is_empty()).then(|| h.capsule.intent.trim().to_string())
        })
    };
    let mut candidates: Vec<(f32, HydrationNode)> = Vec::new();
    for (i, h) in recent.iter().take(w.graveyard_window).enumerate() {
        let overlap = jaccard(&current.symbols, &h.capsule.symbols);
        if overlap <= 0.0 {
            continue;
        }
        let tau = if w.logic_recency_tau <= 0.1 {
            0.1
        } else {
            w.logic_recency_tau
        };
        let logic_recency = (-((i as f32) / tau)).exp();
        let emo = h.user_emotion.as_ref().map(|e| e.label.clone());
        let emo_is_friction = emo.as_deref().map(is_friction_label).unwrap_or(false)
            || current_user_emotion
                .map(|e| is_friction_label(&e.label))
                .unwrap_or(false);
        let emo_boost = if emo_is_friction {
            w.emotion_boost.max(1.0)
        } else {
            1.0
        };
        let overlap_boost = 1.0 + (overlap * w.overlap_scale.max(0.0));
        let tok = h.meta.usage.as_ref().and_then(|u| u.tokens_total());
        let norm = w.effort_tokens_norm.max(1.0);
        let effort_boost = tok
            .map(|t| 1.0 + ((t as f32) / norm).clamp(0.0, 1.0) * w.effort_scale.max(0.0))
            .unwrap_or(1.0);

        // Boost explicitly failed turns
        let failure_boost = if h.capsule.failure_mode != crate::types::FailureMode::None {
            1.5
        } else {
            1.0
        };

        let score = logic_recency * overlap_boost * emo_boost * effort_boost * failure_boost;
        candidates.push((
            score,
            HydrationNode {
                symbols: h.capsule.symbols.clone(),
                intent: h.capsule.intent.clone(),
                decision: h.capsule.decision.clone(),
                user_emotion: emo,
                tokens_total: tok,
                failure_mode: h.capsule.failure_mode.clone(),
            },
        ));
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut graveyard: Vec<HydrationNode> = Vec::new();
    let mut seen_fps: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_, node) in candidates {
        if graveyard.len() >= 5 {
            break;
        }
        let mut syms = node.symbols.clone();
        syms.sort();
        let fp = syms.join("|");
        if !seen_fps.insert(fp) {
            continue;
        }
        graveyard.push(node);
    }
    let map = build_symbol_map(recent, w);
    let north_star = north_star
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|mut s| {
            const MAX: usize = 200;
            if s.len() > MAX {
                s.truncate(MAX);
                s.push_str("...");
            }
            s
        });
    Some(HydrationPacket {
        north_star,
        graveyard,
        map,
    })
}

fn build_symbol_map(recent: &[&CapsuleHit], w: &FrictionWeights) -> Option<String> {
    let mut counts: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
    let mut edges: std::collections::HashMap<(&str, &str), i32> = std::collections::HashMap::new();
    for h in recent.iter().take(w.map_window) {
        let mut syms: Vec<&str> = h.capsule.symbols.iter().map(|s| s.as_str()).collect();
        syms.sort();
        syms.dedup();
        for s in &syms {
            *counts.entry(s).or_insert(0) += 1;
        }
        for i in 0..syms.len() {
            for j in (i + 1)..syms.len() {
                let a = syms[i];
                let b = syms[j];
                let key = if a <= b { (a, b) } else { (b, a) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
    }
    if counts.is_empty() {
        return None;
    }
    let mut top: Vec<(&str, i32)> = counts.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    top.truncate(8);
    let top_set: std::collections::HashSet<&str> = top.iter().map(|(s, _)| *s).collect();
    let mut edge_list: Vec<((&str, &str), i32)> = edges
        .into_iter()
        .filter(|((a, b), w)| *w >= 2 && top_set.contains(a) && top_set.contains(b))
        .collect();
    edge_list.sort_by(|a, b| b.1.cmp(&a.1));
    edge_list.truncate(6);
    let sym_part = top
        .into_iter()
        .map(|(s, n)| format!("{s}({n})"))
        .collect::<Vec<_>>()
        .join(", ");
    if edge_list.is_empty() {
        return Some(format!("Territory: {sym_part}"));
    }
    let edge_part = edge_list
        .into_iter()
        .map(|((a, b), w)| format!("{a}<->{b}({w})"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!("Territory: {sym_part}. Links: {edge_part}"))
}

fn render_hydration_warning(
    symbols_str: &str,
    amplifier: String,
    packet: Option<&HydrationPacket>,
) -> String {
    let mut out = String::new();
    out.push_str(
        "[SYSTEM NOTE: Friction detected. Stop this approach. Propose a different strategy.",
    );
    if !symbols_str.trim().is_empty() {
        out.push_str(&format!(" (Loop area: {symbols_str})"));
    }
    out.push_str("]\n");
    if let Some(p) = packet {
        if let Some(ns) = p.north_star.as_deref() {
            out.push_str("North Star: ");
            out.push_str(ns);
            out.push('\n');
        }
        if !p.graveyard.is_empty() {
            out.push_str("Graveyard (recent failed attempts):\n");
            for n in p.graveyard.iter().take(5) {
                let mut line = String::new();
                if !n.symbols.is_empty() {
                    let mut syms = n.symbols.clone();
                    syms.sort();
                    syms.truncate(6);
                    line.push_str(&format!("- {}", syms.join(", ")));
                } else {
                    line.push_str("- (no symbols)");
                }
                if let Some(lbl) = n.user_emotion.as_deref() {
                    if is_friction_label(lbl) {
                        line.push_str(&format!(" [user:{lbl}]"));
                    }
                }
                if let Some(t) = n.tokens_total {
                    if t > 0 {
                        line.push_str(&format!(" (tokens~{t})"));
                    }
                }
                if n.failure_mode != crate::types::FailureMode::None {
                    line.push_str(&format!(" [FAILURE:{:?}]", n.failure_mode));
                }
                if !n.decision.trim().is_empty() {
                    line.push_str(" -> ");
                    line.push_str(n.decision.trim());
                } else if !n.intent.trim().is_empty() {
                    line.push_str(" :: ");
                    line.push_str(n.intent.trim());
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        if let Some(map) = p.map.as_deref() {
            out.push_str("Map: ");
            out.push_str(map);
            out.push('\n');
        }
    }
    if !amplifier.trim().is_empty() {
        out.push_str(amplifier.trim());
        out.push('\n');
    }
    out.push('\n');
    out
}

fn render_resumption_brief(history: &[CapsuleHit]) -> Option<String> {
    if history.is_empty() {
        return None;
    }

    // Find the last intent and decision
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));
    let last = recent.first()?;

    let mut out = String::new();
    out.push_str("[SYSTEM NOTE: Welcome back. A long gap was detected since the last turn. To re-orient yourself:\n");
    if !last.capsule.intent.trim().is_empty() {
        out.push_str(&format!(
            "Last User Intent: \"{}\"\n",
            last.capsule.intent.trim()
        ));
    }
    if !last.capsule.decision.trim().is_empty() {
        out.push_str(&format!(
            "Last Agent Decision: \"{}\"\n",
            last.capsule.decision.trim()
        ));
    }
    out.push_str("Please summarize your current understanding of the task before proceeding with more code.]\n");

    Some(out)
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
                failure_mode: crate::types::FailureMode::None,
                failure_signals: None,
            },
            meta: ResponseMeta {
                source: "test".to_string(),
                upstream_host: "test".to_string(),
                request_path: "test".to_string(),
                http_status: 200,
                agent_session_id: None,
                usage: None,
            },
        }
    }

    #[test]
    fn test_stateless_friction_note_emitted_for_frustration() {
        let e = EmotionMeta {
            label: "frustration".to_string(),
            valence: -0.7,
            intensity: 0.6,
            confidence: 0.8,
        };
        let note = evaluate_stateless_friction(Some(&e));
        assert!(note.is_some());
        assert!(note.unwrap().contains("SYSTEM NOTE"));
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
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        assert!(evaluate_friction(&current, None, &[]).is_none());
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
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        let history = vec![
            make_hit(vec!["auth.ts"], Some("frustration")),
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], None),
        ];
        let result = evaluate_friction(&current, None, &history);
        assert!(result.is_some());
        let warning = result.unwrap();
        assert!(warning.contains("auth.ts"));
        assert!(warning.contains("SYSTEM NOTE"));
    }
}
