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
    /// Per-basin cooldown counters (turns remaining).
    pub basin_cooldowns: std::collections::HashMap<String, usize>,
    /// History of user-mentioned symbols with timestamps for decay.
    pub user_symbol_history: std::collections::VecDeque<(std::collections::HashSet<String>, i64)>,
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

pub struct TrajectoryUpdate {
    pub state: TrajectoryState,
    pub note: Option<String>,
    pub intensity: f32,
    pub cause: String,
}

impl TrajectoryController {
    pub fn update(
        &mut self,
        workspace_id: &str,
        current: &IntentCapsule,
        current_emotion: Option<&crate::emotion::EmotionMeta>,
        history: &[CapsuleHit],
        ts_ms: i64,
    ) -> TrajectoryUpdate {
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
        if !current.user_symbols.is_empty() {
            let user_paths: std::collections::HashSet<_> =
                current.user_symbols.iter().cloned().collect();
            self.user_symbol_history.push_back((user_paths, ts_ms));
            if self.user_symbol_history.len() > 10 {
                self.user_symbol_history.pop_front();
            }
        }

        // Grounding Stall: previous assistant ignored recently mentioned user paths (with decay)
        let has_stall = if let Some(last) = history.first() {
            let last_assistant_symbols: std::collections::HashSet<_> =
                last.capsule.symbols.iter().cloned().collect();

            // Check recent history for ignored paths, with exponential decay on importance
            let mut weighted_stall = 0.0;
            for (paths, mentioned_ts) in &self.user_symbol_history {
                let age_mins = (ts_ms - *mentioned_ts) as f32 / 60000.0;
                let weight = (-0.2 * age_mins).exp(); // Path importance decays over time

                let missing = paths
                    .iter()
                    .filter(|p| !last_assistant_symbols.contains(*p))
                    .count();
                if !paths.is_empty() && missing > 0 {
                    weighted_stall += (missing as f32 / paths.len() as f32) * weight;
                }
            }
            weighted_stall > 0.5
        } else {
            false
        };

        if has_stall {
            self.stall_streak += 1;
        } else {
            self.stall_streak = 0;
        }
        let s_stall = if self.stall_streak >= 2 { 1.0 } else { 0.0 };

        // Instruction Staticness: user repeats same long message
        let is_static = history.first().map_or(false, |h| {
            let cur_intent = current.intent.trim();
            let prev_intent = h.capsule.intent.trim();
            cur_intent.len() > 50
                && (cur_intent == prev_intent
                    || cur_intent.starts_with(prev_intent)
                    || prev_intent.starts_with(cur_intent))
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
        for count in self.basin_cooldowns.values_mut() {
            if *count > 0 {
                *count -= 1;
            }
        }

        // 7. Select Intervention
        let cause = if drift_intensity > spec_intensity && drift_intensity > loop_intensity {
            "drift"
        } else if spec_intensity > loop_intensity {
            "spec"
        } else {
            "loop"
        };

        let mut note = if self.state != prev_state || self.state == TrajectoryState::Intervene {
            // Check per-basin cooldown
            if self.basin_cooldowns.get(cause).map_or(0, |c| *c) > 0 {
                None
            } else {
                let intervention_type = format!("{}:{}", cause, self.state as u8);
                if self.last_intervention_type.as_ref() == Some(&intervention_type) {
                    // One-Shot Rule: Don't repeat the exact same intervention type within the same episode
                    None
                } else {
                    self.last_intervention_type = Some(intervention_type);

                    // Apply Refractory Period
                    let cooldown = match cause {
                        "loop" => 5,
                        _ => 2,
                    };
                    self.basin_cooldowns.insert(cause.to_string(), cooldown);

                    select_intervention_with_substance(
                        self.intensity,
                        self.state,
                        current_emotion,
                        cause,
                        workspace_id,
                        current,
                        history,
                    )
                }
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

        TrajectoryUpdate {
            state: self.state,
            note,
            intensity: self.intensity,
            cause: cause.to_string(),
        }
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
    intensity: f32,
    _state: TrajectoryState,
    emotion: Option<&crate::emotion::EmotionMeta>,
    cause: &str,
    workspace_id: &str,
    current: &IntentCapsule,
    history: &[CapsuleHit],
) -> Option<String> {
    let label = emotion.map(|e| e.label.as_str()).unwrap_or("neutral");
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    // Stratified Policy: Map intensity to structural severity
    // - I_t < 0.8: Ambient Note (Hints)
    // - I_t >= 0.8: Structural Note (Hydration/Fact Check)
    // - I_t > 0.95: Actionable Intervene (Hard Stop)
    let is_ambient = intensity < 0.8;
    let _is_structural = intensity >= 0.8 && intensity < 0.95;
    let is_emergency = intensity >= 0.95;

    match (cause, label) {
        // --- Specification Basin (Staff Engineer Voice) ---
        ("spec", "confused" | "doubt") => {
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
        ("spec", _) if is_ambient => {
            let intent = current.intent.trim();
            let decision = current.decision.trim();
            Some(format!(
                "[SYSTEM NOTE: To ensure we're aligned: my current understanding is \"{}\". Next I'll do \"{}\". Does that sound right?]",
                intent, decision
            ))
        }
        ("spec", _) => {
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
        ("drift", _) if is_ambient => {
            let (_, missing_count) = crate::workspace::validate_paths(workspace_id, &current.symbols);
            if missing_count > 0 {
                Some("[SYSTEM NOTE: Potential drift detected. Some mentioned paths do not exist. Verify the workspace state before proceeding.]".to_string())
            } else {
                Some("[SYSTEM NOTE: High assumption load or grounding mismatch detected. Verify your facts about the codebase.]".to_string())
            }
        }
        ("drift", _) => {
            Some("[SYSTEM NOTE: Factual drift is high. Stop. Re-read the relevant files and list 3 verified facts about the current code structure before continuing.]".to_string())
        }

        // --- Loop Basin (Hydration/Attempt Log) ---
        ("loop", "frustration") => Some(
            "[SYSTEM NOTE: User frustration detected in a potential loop. Pause to clarify the immediate blocker.]"
                .to_string(),
        ),
        ("loop", "anger") | ("loop", _) if is_emergency => Some(
            "[SYSTEM NOTE: CRITICAL: Repetitive stall detected. Stop all execution. Apologize and await explicit instructions.]"
                .to_string(),
        ),
        ("loop", _) if is_ambient => {
            let syms = current.symbols.join(", ");
            Some(format!(
                "[SYSTEM NOTE: A lot of repeat activity detected in [{}]. If this approach is stalling, consider proposing an alternative.]",
                syms
            ))
        }
        ("loop", _) => {
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
    pub(crate) emotion_scale: f32,
    pub(crate) effort_scale: f32,
}

impl Default for FrictionWeights {
    fn default() -> Self {
        Self {
            recent_window: 8,
            symbol_repeat_threshold: 2,
            emotion_scale: 1.5,
            effort_scale: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HydrationNode {
    pub(crate) intent: String,
    pub(crate) decision: String,
    pub(crate) user_emotion: Option<String>,
    pub(crate) tokens_total: Option<i64>,
    pub(crate) failure_mode: crate::types::FailureMode,
}

const FRICTION_EMOTIONS: &[&str] = &[
    "frustration",
    "annoyance",
    "anger",
    "disapproval",
    "disappointment",
];

fn build_hydration_packet(
    current: &IntentCapsule,
    _emotion: Option<&crate::emotion::EmotionMeta>,
    history: &[&CapsuleHit],
    w: &FrictionWeights,
) -> Vec<HydrationNode> {
    let mut candidates = Vec::new();
    let now = crate::now_ms();

    let norm = (current.symbols.len() * 100 + current.intent.len()) as f32;

    for h in history.iter().take(w.recent_window) {
        let age_ms = now - h.ts_ms;
        let logic_recency = 1.0 / (1.0 + (age_ms as f32 / 60000.0));

        let mut overlap = 0;
        for s in &h.capsule.symbols {
            if current.symbols.contains(s) {
                overlap += 1;
            }
        }
        let overlap_boost = 1.0 + (overlap as f32 / (current.symbols.len().max(1) as f32));

        let emo = h.user_emotion.as_ref().map(|e| e.label.clone());
        let emo_boost = if let Some(ref e) = emo {
            if FRICTION_EMOTIONS.contains(&e.as_str()) {
                w.emotion_scale
            } else {
                1.0
            }
        } else {
            1.0
        };

        let tok = h.meta.usage.as_ref().and_then(|u| u.tokens_total());
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
                intent: h.capsule.intent.clone(),
                decision: h.capsule.decision.clone(),
                user_emotion: emo,
                tokens_total: tok,
                failure_mode: h.capsule.failure_mode.clone(),
            },
        ));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    candidates.into_iter().take(3).map(|c| c.1).collect()
}

fn render_hydration_warning(
    symbols: &str,
    _history_summary: String,
    packet: &[HydrationNode],
) -> String {
    let mut out = format!("[SYSTEM NOTE: Possible loop detected in symbols [{}]. To help break out, here is a hydration packet of the most relevant recent context:\n\n", symbols);

    if packet.is_empty() {
        out.push_str("(No recent relevant history found)\n");
    } else {
        for (i, n) in packet.iter().enumerate() {
            out.push_str(&format!("{}. ", i + 1));
            if !n.intent.trim().is_empty() {
                out.push_str("User: ");
                out.push_str(n.intent.trim());
                if let Some(ref e) = n.user_emotion {
                    out.push_str(&format!(" ({})", e));
                }
                if let Some(t) = n.tokens_total {
                    out.push_str(&format!(" (tokens~{t})"));
                }
                if n.failure_mode != crate::types::FailureMode::None {
                    line_failure_mode(&mut out, &n.failure_mode);
                }
                if !n.decision.trim().is_empty() {
                    out.push_str(" -> ");
                    out.push_str(n.decision.trim());
                }
            } else {
                out.push_str("Agent: ");
                out.push_str(n.decision.trim());
            }
            out.push('\n');
        }
    }

    out.push_str(
        "\nPlease analyze why these previous attempts failed and propose a DIFFERENT approach.]",
    );
    out
}

fn line_failure_mode(out: &mut String, fm: &crate::types::FailureMode) {
    out.push_str(&format!(" [FAILURE:{:?}]", fm));
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

pub fn evaluate_friction(
    current: &IntentCapsule,
    emotion: Option<&crate::emotion::EmotionMeta>,
    history: &[CapsuleHit],
) -> Option<String> {
    let w = FrictionWeights::default();
    if history.len() < w.symbol_repeat_threshold {
        return None;
    }

    let mut repeats = 0;
    for s in &current.symbols {
        let mut count = 0;
        for h in history.iter().take(w.recent_window) {
            if h.capsule.symbols.contains(s) {
                count += 1;
            }
        }
        if count >= w.symbol_repeat_threshold {
            repeats += 1;
        }
    }

    let has_frustration = emotion.map_or(false, |e| FRICTION_EMOTIONS.contains(&e.label.as_str()));

    if repeats > 0 && has_frustration {
        let mut recent: Vec<&CapsuleHit> = history.iter().collect();
        recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));
        let packet = build_hydration_packet(current, emotion, &recent, &w);
        let symbols_str = current.symbols.join(", ");
        return Some(render_hydration_warning(
            &symbols_str,
            String::new(),
            &packet,
        ));
    }

    None
}

pub fn evaluate_stateless_friction(
    _text: &str,
    _emotion: Option<&crate::emotion::EmotionMeta>,
) -> Option<String> {
    None
}

pub fn evaluate_failure_modes(history: &[CapsuleHit]) -> Option<String> {
    // Check if the most recent capsules indicate a recurring failure mode
    if history.is_empty() {
        return None;
    }

    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    let last = recent.first()?;
    match last.capsule.failure_mode {
        crate::types::FailureMode::Drift => {
            Some("[SYSTEM NOTE: Factual drift was detected in the last turn. Before continuing, verify your assumptions about the codebase by reading the relevant files.]".to_string())
        }
        crate::types::FailureMode::Rediscovery => {
            Some("[SYSTEM NOTE: You seem to be rediscovering information or decisions that were already established. Review the conversation history to avoid redundant work.]".to_string())
        }
        crate::types::FailureMode::RetrySpiral => {
            Some("[SYSTEM NOTE: You are in a retry spiral. Stop and analyze why the previous approach failed before trying again.]".to_string())
        }
        _ => None,
    }
}

pub fn evaluate_decision_conflict(_history: &[CapsuleHit]) -> Option<String> {
    None
}

pub fn detect_failure_keywords(text: &str) -> Option<crate::types::FailureMode> {
    let lower = text.to_lowercase();
    if lower.contains("same error") || lower.contains("still not working") {
        return Some(crate::types::FailureMode::RetrySpiral);
    }
    if lower.contains("wrong file") || lower.contains("doesn't exist") {
        return Some(crate::types::FailureMode::Drift);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_correction() {
        assert!(detect_correction("no that's wrong") > 0.5);
        assert!(detect_correction("actually I meant") > 0.5);
        assert_eq!(detect_correction("looks good"), 0.0);
    }
}
