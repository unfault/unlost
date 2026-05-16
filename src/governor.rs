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
const WEIGHT_LOGIC_CHURN: f32 = 0.20;
const WEIGHT_FLUENCY: f32 = 0.15;

const THRESHOLD_WATCH: f32 = 0.5;
const THRESHOLD_INTERVENE: f32 = 0.8;
const THRESHOLD_STABLE_OFF: f32 = 0.4;

/// Resurfacing gate (see `internal/SOURCE_POINTERS.md` §Recurrence Channel).
/// `1.0 - distance >= REC_SIM_THRESHOLD` to qualify.
const REC_SIM_THRESHOLD: f32 = 0.78;
/// Standalone (no-other-basin) resurfacing fires only when `recurrence_signal`
/// crosses this bar. Identical to the candidate gate today; broken out for
/// future tuning without touching the gate.
const REC_FIRE_THRESHOLD: f32 = REC_SIM_THRESHOLD;

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
    pub watch_start_ts: Option<i64>,
    pub turns_since_intervention: usize,
    /// History of intensity values for persistence checking.
    pub intensity_history: std::collections::VecDeque<f32>,
    /// Tracks the last intervention type to avoid repetitive "nagging" oscillations.
    pub last_intervention_type: Option<String>,
    /// Persistence counter for grounding stall (consecutive turns).
    pub stall_streak: usize,
    /// Persistence counter for instruction staticness.
    pub static_streak: usize,
    /// Consecutive turns where the user expressed anger/frustration/disapproval.
    pub anger_streak: usize,
    /// Per-basin cooldown counters (turns remaining).
    pub basin_cooldowns: std::collections::HashMap<String, usize>,
    /// History of user-mentioned symbols with timestamps for decay.
    pub user_symbol_history: std::collections::VecDeque<(std::collections::HashSet<String>, i64)>,
    /// The last assistant decision for churn calculation.
    pub last_decision: Option<String>,
    /// The last agent session ID seen in this workspace.
    pub last_agent_session_id: Option<String>,
    /// Snapshot of (smoothed_channels, intensity, state) captured at the end of the most
    /// recent `update()` call. Read by `record_turn` so the background worker can persist
    /// them into the capsule without re-deriving EMA state.
    pub last_channels: Option<(SymptomChannels, f32, TrajectoryState)>,
    /// Resurfacing channel: the capsule that this turn matched (if any), kept so the
    /// intervention selector can build a SYSTEM NOTE without a second query.
    pub recurrence_match: Option<RecurrenceMatch>,
    /// In-memory cooldown map keyed by capsule id (loaded lazily from
    /// `resurfaced.jsonl`). `None` means "not yet loaded for this workspace".
    pub resurfaced_cache: Option<std::collections::HashMap<String, i64>>,
    /// Session id of the last standalone resurfacing intervention, to enforce
    /// "one standalone resurfacing per agent session".
    pub last_resurfacing_session_id: Option<String>,
}

/// A dormant capsule that the current movement matched, scored, and is a candidate
/// for resurfacing.
#[derive(Debug, Clone)]
pub struct RecurrenceMatch {
    pub capsule_id: String,
    pub ts_ms: i64,
    /// Similarity in [0, 1]. We derive this from LanceDB distance via `1 - dist`.
    pub similarity: f32,
    /// Composite recurrence score (similarity × structural_weight).
    pub score: f32,
    pub decision: String,
    pub rationale: String,
    pub source_pointer: Option<String>,
    /// Workspace where this capsule lives. `None` for matches inside the
    /// current workspace (the local case — caller already knows). `Some(id)`
    /// when the match came from a peer workspace via cross-workspace
    /// retrieval; rendered as a label in the SYSTEM NOTE.
    pub origin_workspace_id: Option<String>,
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
    "wait",
    "stop",
    "hold on",
    "not quite",
    "don't do",
    "never mind",
    "re-read",
    "false",
];

pub fn detect_correction(text: &str, emotion: Option<&crate::emotion::EmotionMeta>) -> f32 {
    let lower = text.to_lowercase();
    let mut score = 0.0;
    for p in CORRECTION_PATTERNS {
        if lower.contains(p) {
            if *p == "actually" || *p == "wait" || *p == "hold on" {
                score += 0.5;
            } else {
                score += 1.0;
            }
        }
    }
    let mut final_score = (score / 1.5_f32).min(1.0_f32);

    // Affective Boost: If the user is frustrated while correcting, it's high signal
    if let Some(e) = emotion {
        if FRICTION_EMOTIONS.contains(&e.label.as_str()) && final_score > 0.1 {
            final_score = (final_score + 0.3).min(1.0);
        }
    }

    final_score
}

const SUMMARY_CUES: &[&str] = &[
    "summary",
    "recap",
    "summarize",
    "consolidate",
    "overview",
    "in short",
    "to conclude",
    // NOTE: "sorry" / "apologize" / "my apologies" were intentionally removed.
    // Detecting them in the assistant's decision text was damping the trajectory score
    // precisely when the agent was looping on apologies — the opposite of what we want.
    // Apology detection for de-escalation is handled via anger_streak instead.
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
    pub channels: SymptomChannels,
    pub watch_start_ts: Option<i64>,
    /// When set, the recurrence channel matched this dormant capsule. Carried
    /// outward so `record_resurfacing_emitted` can log the surfacing.
    pub recurrence_match: Option<RecurrenceMatch>,
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
        self.update_with_candidates(
            workspace_id,
            current,
            current_emotion,
            history,
            &[],
            None,
            ts_ms,
        )
    }

    /// Like [`update`], but also considers a set of *dormant* candidates retrieved
    /// via ANN to drive the recurrence channel. `current_session_id` lets the
    /// controller enforce one-standalone-resurfacing-per-session. See
    /// `internal/SOURCE_POINTERS.md` §Recurrence Channel.
    pub fn update_with_candidates(
        &mut self,
        workspace_id: &str,
        current: &IntentCapsule,
        current_emotion: Option<&crate::emotion::EmotionMeta>,
        history: &[CapsuleHit],
        dormant_candidates: &[CapsuleHit],
        current_session_id: Option<&str>,
        ts_ms: i64,
    ) -> TrajectoryUpdate {
        let mut reset_note = None;

        // 1. Check for Coffee Pause (Soft Decay Reset)
        if self.last_ts_ms > 0 && (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS {
            self.state = TrajectoryState::Stable;
            self.intensity *= COFFEE_PAUSE_DECAY;
            self.watch_start_ts = None;
            reset_note = render_resumption_brief(history);
        }
        self.last_ts_ms = ts_ms;

        // 2. Calculate raw symptoms
        let s_rep = calculate_repetition(current, history);
        let s_nov = calculate_novelty_collapse(current, history);
        let s_sem = calculate_semantic_stall(current, history);
        let s_eff = calculate_effort_spike(current, history);
        let s_corr = detect_correction(&current.intent, current_emotion);
        let s_hallucination = calculate_drift_hallucination(workspace_id, current);
        let s_summary = detect_summary_intent(&current.decision);
        let s_churn = calculate_logic_churn(&current.decision, &self.last_decision);
        let s_fluency = calculate_fluency(history);
        self.last_decision = Some(current.decision.clone());

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

        // Anger escalation streak: track consecutive turns where the user is angry/frustrated.
        // Note: "disapproval" is excluded here — it maps to intellectual disagreement in go_emotions
        // and should not increment the streak. It still affects trajectory intensity via affective
        // modulation (valence contribution), so it is not ignored entirely.
        let is_angry = matches!(
            current_emotion.map(|e| e.label.as_str()),
            Some("anger" | "frustration")
        );
        if is_angry {
            self.anger_streak += 1;
        } else {
            self.anger_streak = 0;
        }

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
        self.smoothed_channels.path_hallucination = EMA_ALPHA * s_hallucination
            + (1.0 - EMA_ALPHA) * self.smoothed_channels.path_hallucination;
        self.smoothed_channels.grounding_stall =
            EMA_ALPHA * s_stall + (1.0 - EMA_ALPHA) * self.smoothed_channels.grounding_stall;
        self.smoothed_channels.instruction_staticness =
            EMA_ALPHA * s_stat + (1.0 - EMA_ALPHA) * self.smoothed_channels.instruction_staticness;
        self.smoothed_channels.logic_churn =
            EMA_ALPHA * s_churn + (1.0 - EMA_ALPHA) * self.smoothed_channels.logic_churn;
        self.smoothed_channels.fluency =
            EMA_ALPHA * s_fluency + (1.0 - EMA_ALPHA) * self.smoothed_channels.fluency;

        // ── Recurrence channel ───────────────────────────────────────────────────
        // Score dormant ANN candidates: similarity = 1 - distance (BGE cosine in
        // LanceDB). Weight by structural value (decision + rationale present).
        // Filter out capsules that are still cooling per `resurfaced.jsonl`.
        let recent_ids: std::collections::HashSet<&str> =
            history.iter().map(|h| h.id.as_str()).collect();
        if self.resurfaced_cache.is_none() {
            self.resurfaced_cache = Some(crate::resurfaced::load(workspace_id));
        }
        let cooldown_map = self.resurfaced_cache.as_ref().expect("just initialised");
        let mut best: Option<RecurrenceMatch> = None;
        for cand in dormant_candidates {
            if recent_ids.contains(cand.id.as_str()) {
                continue;
            }
            // Skip non-conversational sources — git/changelog rows are facts, not
            // unfinished threads. See doc §Recurrence Channel filters.
            let src = cand.meta.source.as_str();
            if matches!(src, "git" | "changelog" | "init") {
                continue;
            }
            let category = cand.capsule.category.as_str();
            if matches!(category, "GitCommit" | "GitTag" | "Version" | "Replay" | "replay") {
                continue;
            }
            if crate::resurfaced::is_cooling(cooldown_map, &cand.id, ts_ms) {
                continue;
            }
            let similarity = (1.0_f32 - cand.distance).clamp(0.0, 1.0);
            if similarity < REC_SIM_THRESHOLD {
                continue;
            }
            let has_decision = !cand.capsule.decision.trim().is_empty();
            let has_rationale = !cand.capsule.rationale.trim().is_empty();
            let structural_weight = if has_decision && has_rationale {
                1.0_f32
            } else {
                0.5_f32
            };
            let score = similarity * structural_weight;
            // Tag the match with its origin workspace only when it came from a
            // *different* workspace than the one we're currently scoring against.
            // Same-workspace matches leave this field as None so the SYSTEM NOTE
            // doesn't render a redundant "in <current project>" suffix.
            let origin_ws = cand
                .origin_workspace_id
                .as_deref()
                .filter(|id| *id != workspace_id)
                .map(str::to_string);
            let candidate_match = RecurrenceMatch {
                capsule_id: cand.id.clone(),
                ts_ms: cand.ts_ms,
                similarity,
                score,
                decision: cand.capsule.decision.clone(),
                rationale: cand.capsule.rationale.clone(),
                source_pointer: cand.meta.source_pointer.clone(),
                origin_workspace_id: origin_ws,
            };
            best = match best {
                None => Some(candidate_match),
                Some(prev) => {
                    // Tiebreak by dormancy: older capsule wins on equal score.
                    if score > prev.score
                        || (score == prev.score && candidate_match.ts_ms < prev.ts_ms)
                    {
                        Some(candidate_match)
                    } else {
                        Some(prev)
                    }
                }
            };
        }
        let s_recurrence = best.as_ref().map(|m| m.score).unwrap_or(0.0);
        // Recurrence is intentionally NOT EMA-smoothed: it's a per-turn match signal,
        // not a slow-burning channel. Store directly.
        self.smoothed_channels.recurrence_signal = s_recurrence;
        self.recurrence_match = best;

        // 4. Intensity Calculation
        let loop_intensity = WEIGHT_REPETITION * self.smoothed_channels.repetition
            + WEIGHT_NOVELTY * self.smoothed_channels.novelty_collapse
            + WEIGHT_SEMANTIC * self.smoothed_channels.semantic_stall
            + WEIGHT_EFFORT * self.smoothed_channels.effort_spike
            + WEIGHT_LOGIC_CHURN * self.smoothed_channels.logic_churn;

        let spec_intensity = WEIGHT_ALIGNMENT_DEBT * self.smoothed_channels.alignment_debt
            + WEIGHT_INSTRUCTION_STATICNESS * self.smoothed_channels.instruction_staticness
            + WEIGHT_FLUENCY * self.smoothed_channels.fluency;

        let drift_intensity = WEIGHT_PATH_HALLUCINATION * self.smoothed_channels.path_hallucination
            + WEIGHT_GROUNDING_STALL * self.smoothed_channels.grounding_stall;

        let mut raw_intensity = loop_intensity + spec_intensity + drift_intensity;

        // "Stubbornness" Boost: If logic churn is LOW but alignment debt is HIGH,
        // the agent is stubbornly repeating a failed approach.
        if self.smoothed_channels.alignment_debt > 0.5 && self.smoothed_channels.logic_churn < 0.2 {
            raw_intensity = (raw_intensity + 0.2).min(1.0);
        }

        // NEW: "Blind Acceptance" Risk (Feb 16, 2026)
        // High fluency in previous turn + short/passive current user input indicates
        // the user might be blindly accepting verbose output without structural verification.
        // Exclude clear short directives and confirmations — "yes", "ok", "sure", "do it",
        // "extend trace." — these are decisive, not passive, and must not inflate intensity.
        const CONFIRMATION_WORDS: &[&str] = &[
            "yes",
            "ok",
            "okay",
            "sure",
            "yep",
            "yup",
            "agreed",
            "correct",
            "right",
            "good",
            "great",
            "perfect",
            "fine",
            "sounds good",
            "go ahead",
            "do it",
            "proceed",
            "continue",
            "carry on",
        ];
        let intent_lower = current.intent.trim().to_lowercase();
        let is_confirmation = CONFIRMATION_WORDS
            .iter()
            .any(|w| intent_lower == *w || intent_lower == format!("{}.", w));
        let is_passive_user =
            !is_confirmation && current.intent.len() < 30 && current.user_symbols.is_empty();
        if self.smoothed_channels.fluency > 0.6 && is_passive_user {
            raw_intensity = (raw_intensity + 0.15).min(1.0);
        }

        // NEW: Summary Intent Damping (Prevents false positives during consolidation)
        // Also includes "Apology Damping" to filter out submissive noise.
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
                    self.watch_start_ts = Some(ts_ms);
                }
            }
            TrajectoryState::Watch => {
                if (self.intensity > THRESHOLD_INTERVENE && slope > 0.05) || is_persistent {
                    self.state = TrajectoryState::Intervene;
                } else if self.intensity < THRESHOLD_STABLE_OFF {
                    self.state = TrajectoryState::Stable;
                    self.watch_start_ts = None;
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

        // Resurfacing: emit a standalone SYSTEM NOTE when (a) the recurrence
        // channel exceeds the firing threshold, (b) no other basin is at Watch+,
        // and (c) we have not already surfaced a standalone in this session.
        let mut standalone_resurfacing = false;
        if self.smoothed_channels.recurrence_signal >= REC_FIRE_THRESHOLD
            && self.recurrence_match.is_some()
            && self.state == TrajectoryState::Stable
        {
            let already_fired = match (
                self.last_resurfacing_session_id.as_deref(),
                current_session_id,
            ) {
                (Some(prev), Some(cur)) => prev == cur,
                // Replay / unattributed turns: don't fire standalone — we have no
                // good way to throttle per session.
                _ => current_session_id.is_none(),
            };
            if !already_fired {
                standalone_resurfacing = true;
            }
        }

        // Anger escalation override: if the user has been angry/frustrated for 2+ consecutive
        // turns AND the trajectory intensity is already elevated (>= Watch threshold), fire a
        // de-escalation note. The trajectory gate prevents pure emotion-classification noise
        // (e.g. a fruitful back-and-forth discussion) from triggering this path — there must
        // be corroborating behavioral evidence before we short-circuit normal dispatch.
        let anger_override = if self.anger_streak >= 2 && self.intensity >= THRESHOLD_WATCH {
            let key = "anger_escalation";
            if self.basin_cooldowns.get(key).map_or(0, |c| *c) == 0 {
                self.basin_cooldowns.insert(key.to_string(), 3);
                Some(de_escalation_note(history))
            } else {
                None
            }
        } else {
            None
        };

        let mut effective_cause = cause.to_string();
        let mut note = if anger_override.is_some() {
            anger_override
        } else if self.state != prev_state || self.state == TrajectoryState::Intervene {
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

                    let primary = select_intervention_with_substance(
                        self.intensity,
                        self.state,
                        current_emotion,
                        cause,
                        workspace_id,
                        current,
                        history,
                    );
                    // Modifier mode: when another basin fires AND we have a strong
                    // recurrence match, append the resurfacing context. Modifier is
                    // unthrottled — we ride on the existing intervention.
                    match (primary, self.recurrence_match.as_ref()) {
                        (Some(p), Some(m))
                            if self.smoothed_channels.recurrence_signal >= REC_FIRE_THRESHOLD =>
                        {
                            // Record surfacing (modifier-mode counts too).
                            crate::resurfaced::record(workspace_id, &m.capsule_id, ts_ms);
                            if let Some(cache) = self.resurfaced_cache.as_mut() {
                                cache.insert(m.capsule_id.clone(), ts_ms);
                            }
                            Some(format!("{p}\n{}", format_resurfacing_addendum(m)))
                        }
                        (other, _) => other,
                    }
                }
            }
        } else if standalone_resurfacing {
            // No other basin fired — emit a standalone resurfacing SYSTEM NOTE.
            effective_cause = "resurfacing".to_string();
            // Mark this session so we don't fire standalone again.
            self.last_resurfacing_session_id = current_session_id.map(str::to_string);
            let m = self
                .recurrence_match
                .as_ref()
                .expect("standalone_resurfacing implies a match");
            crate::resurfaced::record(workspace_id, &m.capsule_id, ts_ms);
            if let Some(cache) = self.resurfaced_cache.as_mut() {
                cache.insert(m.capsule_id.clone(), ts_ms);
            }
            Some(format_resurfacing_standalone_note(m))
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

        // Stash channels so record_turn can copy them into ChunkInput without
        // re-deriving EMA state in the background worker.
        self.last_channels = Some((self.smoothed_channels.clone(), self.intensity, self.state));

        TrajectoryUpdate {
            state: self.state,
            note,
            intensity: self.intensity,
            cause: effective_cause,
            channels: self.smoothed_channels.clone(),
            watch_start_ts: self.watch_start_ts,
            recurrence_match: self.recurrence_match.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.state = TrajectoryState::Stable;
        self.intensity = 0.0;
        self.smoothed_channels = SymptomChannels::default();
        self.turns_since_intervention = 0;
        self.intensity_history.clear();
        self.last_intervention_type = None;
        self.last_decision = None;
        self.user_symbol_history.clear();
        self.watch_start_ts = None;
        self.anger_streak = 0;
    }
}

fn calculate_fluency(history: &[CapsuleHit]) -> f32 {
    if let Some(last) = history.first() {
        let user_toks = last
            .meta
            .usage
            .as_ref()
            .and_then(|u| u.tokens_input)
            .unwrap_or((last.capsule.intent.len() / 4) as i64)
            .max(1);
        let assistant_toks = last
            .meta
            .usage
            .as_ref()
            .and_then(|u| u.tokens_output)
            .unwrap_or((last.capsule.decision.len() / 4) as i64);

        let ratio = assistant_toks as f32 / user_toks as f32;
        // 10x verbosity is our "high fluency" signal baseline
        (ratio / 10.0).min(1.0)
    } else {
        0.0
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

fn calculate_drift_hallucination(workspace_id: &str, current: &IntentCapsule) -> f32 {
    let (paths_checked, paths_missing) =
        crate::workspace::validate_paths(workspace_id, &current.symbols);
    let (idents_checked, idents_missing) =
        crate::workspace::validate_identifiers(workspace_id, &current.symbols);

    let total_checked = paths_checked + idents_checked;
    let total_missing = paths_missing + idents_missing;

    if total_checked == 0 {
        return 0.0;
    }

    // Hallucination score is the ratio of missing symbols, but we floor it at 0.5
    // if ANY path is missing to prioritize grounding.
    let ratio = total_missing as f32 / total_checked as f32;
    if paths_missing > 0 {
        ratio.max(0.5)
    } else {
        ratio
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

fn calculate_logic_churn(current: &str, last: &Option<String>) -> f32 {
    if let Some(prev) = last {
        if current.is_empty() || prev.is_empty() {
            return 0.0;
        }
        let w1: std::collections::HashSet<_> = current
            .split_whitespace()
            .map(|s| s.to_lowercase().replace(|c: char| !c.is_alphanumeric(), ""))
            .filter(|s| !s.is_empty())
            .collect();
        let w2: std::collections::HashSet<_> = prev
            .split_whitespace()
            .map(|s| s.to_lowercase().replace(|c: char| !c.is_alphanumeric(), ""))
            .filter(|s| !s.is_empty())
            .collect();

        if w1.is_empty() || w2.is_empty() {
            return 0.0;
        }

        let intersection = w1.intersection(&w2).count();
        let union = w1.len().max(w2.len());
        1.0 - (intersection as f32 / union as f32)
    } else {
        0.0
    }
}

/// Truncate a string at a UTF-8-safe character boundary, appending an ellipsis
/// when truncation occurred.
fn truncate_for_note(s: &str, max_chars: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max_chars).collect();
    out.push('…');
    out
}

/// Render a calendar-date label for a UTC timestamp. Falls back to the raw
/// epoch ms if conversion fails.
fn render_date_label(ts_ms: i64) -> String {
    use chrono::TimeZone;
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("ts_ms={ts_ms}"))
}

/// Build the standalone resurfacing SYSTEM NOTE — fired when no other basin
/// fires AND the recurrence channel matched a dormant capsule. See
/// `internal/SOURCE_POINTERS.md` §Recurrence Channel.
pub(crate) fn format_resurfacing_standalone_note(m: &RecurrenceMatch) -> String {
    let date = render_date_label(m.ts_ms);
    let decision = truncate_for_note(&m.decision, 200);
    let rationale = truncate_for_note(&m.rationale, 200);
    let source_line = match m.source_pointer.as_deref() {
        Some(uri) => match crate::workspace::resolve_source_label(uri) {
            Some(label) => format!("\n Source: {uri}\n ({label})"),
            None => format!("\n Source: {uri}"),
        },
        None => String::new(),
    };
    let rationale_line = if rationale.is_empty() {
        String::new()
    } else {
        format!("\n Rationale: \"{rationale}\".")
    };
    let origin_clause = format_origin_clause(m);
    format!(
        "[SYSTEM NOTE: This appears to continue a prior thread{origin_clause}.\n \
         On {date}, the decision was: \"{decision}\".{rationale_line}\n \
         If relevant, the user may want to revisit this before proceeding.{source_line}]"
    )
}

/// Build the modifier-mode addendum — appended to another basin's intervention
/// when the recurrence channel also matched. Kept terse on purpose: the primary
/// note carries the main signal; this is supplementary.
pub(crate) fn format_resurfacing_addendum(m: &RecurrenceMatch) -> String {
    let date = render_date_label(m.ts_ms);
    let decision = truncate_for_note(&m.decision, 160);
    let origin_clause = format_origin_clause(m);
    let source = m
        .source_pointer
        .as_deref()
        .and_then(crate::workspace::resolve_source_label)
        .map(|label| format!(" Source: {label}."))
        .unwrap_or_default();
    format!(
        "[SYSTEM NOTE: Related prior thread from {date}{origin_clause}: \
         \"{decision}\".{source}]"
    )
}

/// Render the " in <project>" clause when the matched capsule lives in a
/// different workspace. Returns an empty string for same-workspace matches so
/// the note reads naturally in the local case.
fn format_origin_clause(m: &RecurrenceMatch) -> String {
    let Some(ws_id) = m.origin_workspace_id.as_deref() else {
        return String::new();
    };
    let label = crate::workspace::workspace_label_by_id(ws_id)
        .unwrap_or_else(|| ws_id.to_string());
    format!(" in {label}")
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
            // Only fire if there is enough substance to produce a coherent check-in.
            // A one-word intent (e.g. "yes") or an empty decision produces a note that
            // reads as nonsense and erodes trust in the intervention system.
            if intent.split_whitespace().count() < 4 || decision.is_empty() {
                None
            } else {
                Some(format!(
                    "[SYSTEM NOTE: To ensure we're aligned: my current understanding is \"{}\". Next I'll do \"{}\". Does that sound right?]",
                    intent, decision
                ))
            }
        }
        ("spec", _) => {
            let corrections: Vec<String> = recent
                .iter()
                .take(5)
                .filter(|h| detect_correction(&h.capsule.intent, h.user_emotion.as_ref()) > 0.5)
                .map(|h| h.capsule.intent.trim().to_string())
                .collect();

            // Only surface a goal that looks like substantive work: at least 6 words,
            // not a bare confirmation ("yes", "ok", "sure"), and not a meta phrase like
            // "casual check-in" or "continue".  Surfacing an ambient/meta intent as
            // "Original Goal" makes the note incoherent and erodes trust.
            let north_star = recent
                .iter()
                .rev()
                .find(|h| {
                    let t = h.capsule.intent.trim().to_lowercase();
                    let words = t.split_whitespace().count();
                    words >= 6
                        && !t.starts_with("casual")
                        && !t.starts_with("carry on")
                        && !t.starts_with("go ahead")
                        && !t.starts_with("continue")
                        && !t.starts_with("check-in")
                        && !t.starts_with("check in")
                })
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
            let (_, missing_paths) = crate::workspace::validate_paths(workspace_id, &current.symbols);
            let (_, missing_idents) = crate::workspace::validate_identifiers(workspace_id, &current.symbols);

            if missing_paths > 0 {
                Some("[SYSTEM NOTE: Potential drift detected. Some mentioned paths do not exist. Please re-read the relevant files and list 3 verified facts about the current codebase before proceeding.]".to_string())
            } else if missing_idents > 0 {
                Some("[SYSTEM NOTE: Potential symbol drift detected. Some mentioned functions or classes do not exist. Verify the codebase structure.]".to_string())
            } else {
                Some("[SYSTEM NOTE: High assumption load or grounding mismatch detected. Verify your facts about the codebase. List your core assumptions and confirm them against the source code.]".to_string())
            }
        }
        ("drift", _) => {
            Some("[SYSTEM NOTE: Factual drift is high. Stop. Re-read the relevant files and list 3 verified facts about the current code structure before continuing. You must explicitly cite the source files for these facts.]".to_string())
        }

        // --- Loop Basin (Hydration/Attempt Log) ---
        ("loop", "frustration") => Some(
            "[SYSTEM NOTE: User frustration detected in a potential loop. Pause to clarify the immediate blocker.]"
                .to_string(),
        ),
        ("loop", "anger") | ("loop", _) if is_emergency => Some(de_escalation_note(history)),
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

pub const FRICTION_EMOTIONS: &[&str] = &[
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
    let mut out = format!(
        "[SYSTEM NOTE: Possible loop detected in symbols [{}]. To help break out, here is a hydration packet of the most relevant recent context:\n\n",
        symbols
    );

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

/// Build a de-escalation note for when the user's anger is visibly escalating.
///
/// The goal is to stop the agent from tunnelling further into a broken state,
/// have it acknowledge what went wrong kindly, and propose a concrete recovery
/// plan — not to make it apologise in a loop or passively "await instructions".
fn de_escalation_note(history: &[CapsuleHit]) -> String {
    // Find the last stable goal: earliest non-meta intent in the recent window.
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    let last_goal = recent
        .iter()
        .rev() // oldest first
        .find(|h| {
            let intent = h.capsule.intent.trim();
            intent.len() > 20
                && !intent.to_lowercase().starts_with("carry on")
                && !intent.to_lowercase().starts_with("go ahead")
                && !intent.to_lowercase().starts_with("continue")
        })
        .map(|h| h.capsule.intent.trim().to_string());

    let mut note = String::from(
        "[SYSTEM NOTE: The user's frustration is escalating. \
STOP all processing and do not make any further changes to the codebase.\n\
Instead, do the following — in this order:\n\
1. Acknowledge the problem clearly and kindly: name specifically what went wrong, \
without deflecting or over-explaining.\n\
2. Briefly summarise the current broken state so the user can see you understand it.\n\
3. Propose a concrete recovery plan of no more than 3 steps to get back to a satisfying point.\n\
Wait for the user to confirm the plan before taking any action.",
    );

    if let Some(goal) = last_goal {
        note.push_str(&format!("\nLast known goal: \"{}\"", goal));
    }

    note.push(']');
    note
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

pub fn evaluate_failure_modes(history: &[CapsuleHit], session_id: Option<&str>) -> Option<String> {
    // Check if the most recent capsules indicate a recurring failure mode.
    // If a session ID is provided, restrict evaluation to capsules from that session only —
    // this prevents failure modes tagged in a previous session from firing at the start of a
    // new one.
    let session_history: Vec<&CapsuleHit> = if let Some(sid) = session_id {
        history
            .iter()
            .filter(|h| h.meta.agent_session_id.as_deref() == Some(sid))
            .collect()
    } else {
        history.iter().collect()
    };

    if session_history.is_empty() {
        return None;
    }

    let mut recent: Vec<&CapsuleHit> = session_history;
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
        crate::types::FailureMode::FalseProgress => {
            // Only fire when at least 2 of the last 3 capsules show false_progress,
            // so a single ambiguous turn doesn't trigger it.
            let fp_count = recent
                .iter()
                .take(3)
                .filter(|h| h.capsule.failure_mode == crate::types::FailureMode::FalseProgress)
                .count();
            if fp_count >= 2 {
                Some(
                    "[SYSTEM NOTE: Multiple consecutive attempts have produced no observable \
change in behaviour. Stop making changes. Read the actual error output or run the \
relevant command yourself, form a single testable hypothesis about the root cause, \
and state it explicitly — then wait for confirmation before writing any code.]"
                        .to_string(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn evaluate_decision_conflict(_history: &[CapsuleHit]) -> Option<String> {
    None
}

const FAILURE_REPORT_PATTERNS: &[&str] = &[
    "failed",
    "failing",
    "error",
    "doesn't work",
    "does not work",
    "build",
    "tests",
    "ci",
    "workflow",
    "exception",
    "crash",
    "panic",
    "exit code",
    "status 1",
];

pub fn detect_failure_report(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut count = 0;
    for p in FAILURE_REPORT_PATTERNS {
        if lower.contains(p) {
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (0.4 + (count as f32 * 0.2)).min(1.0)
    }
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

// ============================================================================
// TurnEval: Developer coaching score computation (all heuristic, zero LLM)
// ============================================================================

/// Inputs fed to the coach scorer from the background flush worker.
pub struct CoachInput<'a> {
    /// The freshly-extracted capsule for this turn.
    pub capsule: &'a IntentCapsule,
    /// Recent prior capsules in this session (newest first).
    pub history: &'a [CapsuleHit],
    /// Raw exchange text (user + assistant combined) for the turn.
    pub exchange_text: &'a str,
    /// Token usage for this turn (used for context_freshness cache ratio).
    pub usage: Option<&'a crate::types::UsageMeta>,
    /// Number of turns in this session so far (including this one).
    pub session_turn_count: usize,
}

/// Returned coach dimension scores (0.0–1.0 each, higher = healthier).
#[derive(Debug, Clone, Default)]
pub struct CoachScores {
    pub clarity: f32,
    pub context_freshness: f32,
    pub verification_rigor: f32,
    pub decision_progress: f32,
    pub scope_discipline: f32,
    /// Rolling slope of tokens_input over last 3 turns, normalised 0–1.
    /// High = token spend accelerating without corresponding progress.
    pub cost_acceleration: f32,
    /// Short auditable evidence strings.
    pub evidence: Vec<String>,
}

/// Patterns indicating a tool outcome was verified (build, test, or static analysis).
const VERIFICATION_PASS_PATTERNS: &[&str] = &[
    // Build / test
    "succeeded",
    "tests passed",
    "test passed",
    "all tests",
    "cargo test",
    "npm test",
    "pytest",
    "build ok",
    "✓",
    "✔",
    "passing",
    "0 failed",
    // Static analysis / type-checking (clean output)
    "no errors",
    "no warnings",
    "0 errors",
    "0 warnings",
    "type-checked",
    "type checked",
    "clippy",
    "mypy",
    "tsc",
    "pyright",
    "eslint",
    "ruff",
    "biome",
    "golangci",
];

const VERIFICATION_FAIL_PATTERNS: &[&str] = &[
    // Build / test failures
    "failed",
    "error",
    "exit code",
    "FAILED",
    "panicked",
    "assertion failed",
    "stderr",
    "status 1",
    // Static analysis / type-checking failures
    "type error",
    "type mismatch",
    "cannot find",
    "unresolved",
    "lint error",
    "lint warning",
    "E0", // Rust compiler error codes
    "TS", // TypeScript error prefix
];

const CODE_TOUCHING_CATEGORIES: &[&str] = &[
    "implementation",
    "refactoring",
    "debugging",
    "fix",
    "cli_feature_update",
];

/// Compute developer coaching scores for a single turn.
/// All heuristic — no LLM calls, no I/O.
pub fn compute_coach_scores(input: &CoachInput<'_>) -> CoachScores {
    let mut evidence: Vec<String> = Vec::new();

    // ── clarity ─────────────────────────────────────────────────────────────
    // High when: intent is long enough, symbols present, low alignment_debt in
    // the prior turn (meaning the previous response didn't need correction).
    let clarity = {
        let intent_len = input.capsule.intent.len();
        let sym_count = input.capsule.user_symbols.len() + input.capsule.symbols.len();

        // Normalise intent length: <20 chars = 0.0, >=150 chars = 1.0
        let len_score = (intent_len as f32 / 150.0).min(1.0);

        // Symbol specificity: 0 symbols = 0.0, 3+ = 1.0
        let sym_score = (sym_count as f32 / 3.0).min(1.0);

        // Correction penalty: if history[0] had high alignment_debt, the user
        // probably had to correct — reduce clarity of this new request.
        let correction_penalty = input
            .history
            .first()
            .and_then(|h| h.turn_eval.as_ref())
            .map(|te| te.alignment_debt * 0.4)
            .unwrap_or(0.0);

        let raw = (len_score * 0.45 + sym_score * 0.55) - correction_penalty;

        if intent_len < 20 {
            evidence.push("intent too short to be precise".to_string());
        }
        if sym_count == 0 {
            evidence.push("no symbols specified in request".to_string());
        }

        raw.clamp(0.0, 1.0)
    };

    // ── context_freshness ────────────────────────────────────────────────────
    // Decays when:
    // (a) cache_read/tokens_input ratio drops (context being squeezed / compaction)
    // (b) trajectory intensity has been rising for 2+ consecutive turns
    // (c) session turn count is very high (heavy session)
    let context_freshness = {
        // Cache ratio: high ratio = context being served from cache (healthy).
        // Low ratio = fresh tokens dominating (context rotating / compaction triggered).
        let cache_ratio = input.usage.and_then(|u| {
            let cache_read = u.tokens_cache_read.unwrap_or(0);
            let input_tokens = u.tokens_input.unwrap_or(0);
            if input_tokens > 0 {
                Some(cache_read as f32 / input_tokens as f32)
            } else {
                None
            }
        });

        let cache_score: f32 = match cache_ratio {
            Some(r)
                if r < 0.2 && input.usage.and_then(|u| u.tokens_input).unwrap_or(0) > 10_000 =>
            {
                // Low cache ratio with high input tokens = likely compaction
                evidence.push(format!(
                    "low cache hit ratio ({:.0}%) with high input tokens — possible context compaction",
                    r * 100.0
                ));
                0.2
            }
            Some(r) if r < 0.4 => 0.5,
            Some(_) => 1.0,
            None => 0.8, // No usage data — neutral
        };

        // Frustration slope: check if last 2+ turns had rising intensity
        let frustration_slope = if input.history.len() >= 2 {
            let intensities: Vec<f32> = input
                .history
                .iter()
                .take(3)
                .filter_map(|h| h.turn_eval.as_ref())
                .map(|te| te.trajectory_intensity)
                .collect();
            if intensities.len() >= 2 && intensities[0] > intensities[1] + 0.1 {
                evidence
                    .push("trajectory intensity rising — agent may be losing context".to_string());
                0.3 // penalise freshness when frustration is escalating
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Heavy session penalty: high turn count
        let session_penalty = if input.session_turn_count > 40 {
            evidence.push(format!(
                "session has {} turns — context may be stale",
                input.session_turn_count
            ));
            0.3
        } else if input.session_turn_count > 20 {
            0.1
        } else {
            0.0
        };

        (cache_score - frustration_slope - session_penalty).clamp(0.0_f32, 1.0_f32)
    };

    // ── verification_rigor ───────────────────────────────────────────────────
    // High when the exchange contains tool outcome strings (pass/fail).
    // Lower when a code-touching category produced no verification evidence.
    let verification_rigor = {
        let lower = input.exchange_text.to_lowercase();
        let has_pass = VERIFICATION_PASS_PATTERNS.iter().any(|p| lower.contains(p));
        let has_fail = VERIFICATION_FAIL_PATTERNS
            .iter()
            .any(|p| lower.contains(&p.to_lowercase()));
        let is_code_touching = CODE_TOUCHING_CATEGORIES
            .iter()
            .any(|c| input.capsule.category.to_lowercase().contains(c));

        if has_pass {
            1.0
        } else if has_fail {
            // Failure is still verification — the agent checked.
            0.8
        } else if is_code_touching {
            // Code was likely changed but no tool outcome found.
            evidence.push("code-touching turn with no build/test outcome found".to_string());
            0.2
        } else {
            // Non-code turn: not applicable, neutral.
            0.7
        }
    };

    // ── decision_progress ────────────────────────────────────────────────────
    // High when: new decision text (different from prior), new symbols appearing,
    // or category advancing (e.g. planning → implementation).
    let decision_progress = {
        let prior = input.history.first();
        match prior {
            None => 0.8, // First turn: assume progress
            Some(prev) => {
                // Decision text similarity (simple: shared token overlap)
                let new_dec = input.capsule.decision.to_lowercase();
                let old_dec = prev.capsule.decision.to_lowercase();
                let new_words: std::collections::HashSet<&str> =
                    new_dec.split_whitespace().collect();
                let old_words: std::collections::HashSet<&str> =
                    old_dec.split_whitespace().collect();
                let overlap = if old_words.is_empty() {
                    0.0
                } else {
                    new_words.intersection(&old_words).count() as f32 / old_words.len() as f32
                };
                // High overlap = stalled decision
                let decision_novelty = 1.0 - overlap.min(1.0);

                // New symbols compared to last turn
                let prior_syms: std::collections::HashSet<&str> =
                    prev.capsule.symbols.iter().map(|s| s.as_str()).collect();
                let new_syms: std::collections::HashSet<&str> =
                    input.capsule.symbols.iter().map(|s| s.as_str()).collect();
                let new_sym_count = new_syms.difference(&prior_syms).count();
                let sym_novelty = (new_sym_count as f32 / 3.0_f32).min(1.0);

                let progress = decision_novelty * 0.6 + sym_novelty * 0.4;

                if progress < 0.2 {
                    evidence.push(
                        "decision text and symbols unchanged from prior turn — possible stall"
                            .to_string(),
                    );
                }
                progress.clamp(0.0, 1.0)
            }
        }
    };

    // ── scope_discipline ─────────────────────────────────────────────────────
    // High when symbol set stays focused; drops when many new unresolved symbols
    // accumulate across recent turns without corresponding resolution.
    let scope_discipline = {
        if input.history.len() < 3 {
            1.0 // Not enough history to judge scope creep
        } else {
            // Count distinct symbols across last N turns
            let mut all_syms: std::collections::HashSet<&str> = std::collections::HashSet::new();
            let window = input.history.iter().take(5);
            for h in window {
                for s in &h.capsule.symbols {
                    all_syms.insert(s.as_str());
                }
            }
            let symbol_spread = all_syms.len();

            // Category diversity: distinct categories in recent history
            let distinct_cats: std::collections::HashSet<&str> = input
                .history
                .iter()
                .take(5)
                .map(|h| h.capsule.category.as_str())
                .collect();
            let cat_diversity = distinct_cats.len();

            // Symbol spread > 12 across 5 turns = scope concern
            let sym_score: f32 = if symbol_spread > 15 {
                evidence.push(format!(
                    "{} distinct symbols across last 5 turns — possible scope creep",
                    symbol_spread
                ));
                0.2
            } else if symbol_spread > 8 {
                0.5
            } else {
                1.0
            };

            // High category diversity is OK if progressive (planning→impl→test);
            // penalise only if it looks like thrashing (>3 distinct cats in 5 turns).
            let cat_score = if cat_diversity > 3 {
                evidence.push(format!(
                    "{} distinct categories in last 5 turns — possible topic thrashing",
                    cat_diversity
                ));
                0.4
            } else {
                1.0
            };

            (sym_score * 0.6 + cat_score * 0.4).clamp(0.0_f32, 1.0_f32)
        }
    };

    // ── cost_acceleration ────────────────────────────────────────────────────
    // Rolling slope of tokens_input over last 3 turns, normalised to [0, 1].
    // High = spend is growing without corresponding decision_progress.
    // Requires at least 2 prior turns with usage data; 0.0 otherwise.
    let cost_acceleration = {
        // Collect tokens_input from the 3 most recent history entries that have it.
        let token_history: Vec<i64> = input
            .history
            .iter()
            .take(3)
            .filter_map(|h| h.meta.usage.as_ref()?.tokens_input)
            .collect();

        let current_tokens = input.usage.and_then(|u| u.tokens_input).unwrap_or(0);

        if token_history.len() >= 2 && current_tokens > 0 {
            // Oldest of the window
            let oldest = token_history.last().copied().unwrap_or(0);
            if oldest > 0 {
                // Relative growth over the window
                let growth = (current_tokens - oldest) as f32 / oldest as f32;
                // Only flag when growth is meaningful AND progress is not keeping up.
                // growth > 0.5 = tokens grew >50% over 3 turns; penalise by lack of progress.
                let accel = if growth > 0.0 {
                    (growth * (1.0 - decision_progress)).clamp(0.0_f32, 1.0_f32)
                } else {
                    0.0
                };
                if accel > 0.5 {
                    evidence.push(format!(
                        "token spend grew {:.0}% over last {} turns without matching progress",
                        growth * 100.0,
                        token_history.len() + 1
                    ));
                }
                accel
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    CoachScores {
        clarity,
        context_freshness,
        verification_rigor,
        decision_progress,
        scope_discipline,
        cost_acceleration,
        evidence,
    }
}

/// Derive behavioral flags from combined tune + coach scores.
pub fn compute_flags(
    channels: &SymptomChannels,
    intensity: f32,
    coach: &CoachScores,
    session_turn_count: usize,
) -> Vec<String> {
    let mut flags: Vec<String> = Vec::new();

    // ── Agent tuning flags ───────────────────────────────────────────────────
    if channels.repetition > 0.55 {
        flags.push("retry_loop".to_string());
    }
    if channels.logic_churn > 0.50 {
        flags.push("high_churn".to_string());
    }
    if channels.path_hallucination > 0.50 {
        flags.push("hallucination_risk".to_string());
    }
    if channels.alignment_debt > 0.45 {
        flags.push("instruction_drift".to_string());
    }
    if channels.fluency > 0.60 {
        flags.push("blind_acceptance".to_string());
    }
    if intensity >= THRESHOLD_INTERVENE {
        flags.push("trajectory_critical".to_string());
    } else if intensity >= THRESHOLD_WATCH {
        flags.push("trajectory_watch".to_string());
    }

    // ── Developer coaching flags ─────────────────────────────────────────────
    if coach.clarity < 0.30 {
        flags.push("needs_clarification".to_string());
    }
    if coach.context_freshness < 0.40 {
        // Context is heavy — not a problem per se, worth noting
        flags.push("session_heavy".to_string());
    }
    if coach.verification_rigor < 0.25 {
        flags.push("unverified_claim".to_string());
    }
    if coach.scope_discipline < 0.35 {
        flags.push("scope_shift".to_string());
    }
    if coach.cost_acceleration > 0.50 {
        flags.push("cost_spike".to_string());
    }

    // ── Combined / meta flags ────────────────────────────────────────────────
    if session_turn_count > 40 && coach.context_freshness < 0.50 {
        flags.push("session_too_long".to_string());
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_correction() {
        assert!(detect_correction("no that's wrong", None) > 0.5);
        assert!(detect_correction("actually I meant", None) > 0.5);
        assert!(detect_correction("wait stop", None) > 0.5);
        assert_eq!(detect_correction("looks good", None), 0.0);
    }

    // ── Recurrence channel tests ─────────────────────────────────────────────
    //
    // These tests exercise the resurfacing path in isolation, without ANN. We
    // synthesise dormant candidates with controlled `distance` values, feed
    // them through the controller, and inspect the resulting TrajectoryUpdate.
    //
    // See `internal/SOURCE_POINTERS.md` §Recurrence Channel for the contract.

    fn empty_capsule(intent: &str) -> IntentCapsule {
        IntentCapsule {
            category: String::new(),
            intent: intent.to_string(),
            decision: String::new(),
            rationale: String::new(),
            next_steps: vec![],
            symbols: vec![],
            user_symbols: vec![],
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
            extraction_mode: crate::types::ExtractionMode::None,
            questions: vec![],
        }
    }

    /// Build a synthetic dormant capsule with a per-test unique id. Use
    /// `prefix` to scope the ids (passed to `WorkspaceCleanup` so the global
    /// resurfaced ledger is cleaned at test end).
    fn make_candidate(
        prefix: &str,
        suffix: &str,
        distance: f32,
        decision: &str,
        rationale: &str,
    ) -> CapsuleHit {
        let mut cap = empty_capsule("prior intent");
        cap.decision = decision.to_string();
        cap.rationale = rationale.to_string();
        CapsuleHit {
            id: format!("{prefix}-{suffix}"),
            ts_ms: 1_700_000_000_000, // older than 'now' in tests below
            conn_id: 0,
            exchange_seq: 0,
            distance,
            user_emotion: None,
            assistant_emotion: None,
            capsule: cap,
            meta: crate::ResponseMeta {
                source: "record".to_string(),
                upstream_host: String::new(),
                request_path: String::new(),
                http_status: 200,
                agent_session_id: Some("prior-session".to_string()),
                source_pointer: Some(
                    "claude+jsonl:///tmp/a.jsonl#turn=abc".to_string(),
                ),
                usage: None,
            },
            head_sha: None,
            commit_sha: None,
            turn_eval: None,
            origin_workspace_id: None,
        }
    }

    /// Return a `(workspace_id, capsule_id_prefix)` pair. Both share the same
    /// UUID so the cleanup guard can wipe both the per-workspace dir and the
    /// global resurfaced ledger entries with one identifier.
    fn test_ids() -> (String, String) {
        let uuid = uuid::Uuid::new_v4().simple().to_string();
        (format!("wks_test_{uuid}"), format!("cap_test_{uuid}"))
    }

    /// Drop-guard that removes a test workspace's data directory and best-effort
    /// removes any global-ledger entries that mention the given capsule-id
    /// prefix. Use `let _cleanup = WorkspaceCleanup::new(...);` at the top of
    /// any test that may cause `resurfaced::record` to be called.
    struct WorkspaceCleanup {
        ws: String,
        capsule_prefix: String,
    }
    impl WorkspaceCleanup {
        fn new(ws: &str, capsule_prefix: &str) -> Self {
            Self {
                ws: ws.to_string(),
                capsule_prefix: capsule_prefix.to_string(),
            }
        }
    }
    impl Drop for WorkspaceCleanup {
        fn drop(&mut self) {
            let dir = crate::workspace::unlost_workspace_dir(&self.ws);
            let _ = std::fs::remove_dir_all(dir);
            // Best-effort: strip lines from the global resurfaced ledger that
            // mention our test capsule prefix so the file doesn't grow with
            // every test run.
            let global = crate::resurfaced::global_resurfaced_path();
            if let Ok(content) = std::fs::read_to_string(&global) {
                let kept: String = content
                    .lines()
                    .filter(|l| !l.contains(&self.capsule_prefix))
                    .map(|l| format!("{l}\n"))
                    .collect();
                let _ = std::fs::write(&global, kept);
            }
        }
    }

    #[test]
    fn recurrence_scores_high_for_close_match() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        // distance 0.10 => similarity 0.90, structural=1.0 (decision+rationale)
        let cand = make_candidate(&prefix, "a", 0.10, "use opaque markdown", "no curation");
        let now = 1_730_000_000_000;
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("how should context resurface"),
            None,
            &[],
            &[cand],
            Some("ses_now"),
            now,
        );
        assert!(upd.channels.recurrence_signal > 0.85);
        assert!(upd.recurrence_match.is_some());
        let m = upd.recurrence_match.unwrap();
        assert!(m.capsule_id.ends_with("-a"));
        // Standalone fires when no other basin is firing.
        assert_eq!(upd.cause, "resurfacing");
        let note = upd.note.expect("expected a SYSTEM NOTE");
        assert!(note.contains("prior thread"));
        assert!(note.contains("opaque markdown"));
        assert!(note.contains("claude+jsonl"));
    }

    #[test]
    fn recurrence_rejects_below_threshold() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        // distance 0.5 => similarity 0.5, below 0.78 threshold
        let cand = make_candidate(&prefix, "b", 0.5, "unrelated decision", "unrelated rationale");
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("query"),
            None,
            &[],
            &[cand],
            Some("ses"),
            1_730_000_000_000,
        );
        assert_eq!(upd.channels.recurrence_signal, 0.0);
        assert!(upd.recurrence_match.is_none());
        assert!(upd.note.is_none());
    }

    #[test]
    fn recurrence_skips_capsules_in_recent_window() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        let in_recent = make_candidate(&prefix, "recent", 0.05, "fresh decision", "fresh rationale");
        let in_recent_hit = in_recent.clone();
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("query"),
            None,
            std::slice::from_ref(&in_recent_hit),
            &[in_recent],
            Some("ses"),
            1_730_000_000_000,
        );
        // The capsule appears in recent history, so it's filtered before scoring.
        assert_eq!(upd.channels.recurrence_signal, 0.0);
        assert!(upd.recurrence_match.is_none());
    }

    #[test]
    fn recurrence_skips_git_and_changelog_sources() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        let mut git_cand = make_candidate(&prefix, "git", 0.05, "merge feature", "ship");
        git_cand.meta.source = "git".to_string();
        let mut cl_cand = make_candidate(&prefix, "cl", 0.05, "release 1.0", "");
        cl_cand.meta.source = "changelog".to_string();
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("query"),
            None,
            &[],
            &[git_cand, cl_cand],
            Some("ses"),
            1_730_000_000_000,
        );
        assert_eq!(upd.channels.recurrence_signal, 0.0);
        assert!(upd.recurrence_match.is_none());
    }

    #[test]
    fn standalone_fires_once_per_session() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        let cand = make_candidate(&prefix, "s1", 0.08, "decision", "rationale");
        let now = 1_730_000_000_000;
        let upd1 = c.update_with_candidates(
            &ws,
            &empty_capsule("q1"),
            None,
            &[],
            std::slice::from_ref(&cand),
            Some("ses-1"),
            now,
        );
        assert_eq!(upd1.cause, "resurfacing");
        assert!(upd1.note.is_some());
        // Second turn in the same session with a different (still strong) match
        // must NOT fire standalone again.
        let cand2 = make_candidate(&prefix, "s2", 0.05, "decision2", "rationale2");
        let upd2 = c.update_with_candidates(
            &ws,
            &empty_capsule("q2"),
            None,
            &[],
            std::slice::from_ref(&cand2),
            Some("ses-1"),
            now + 60_000,
        );
        assert_ne!(upd2.cause, "resurfacing");
        assert!(upd2.note.is_none());

        // A different session may fire again.
        let cand3 = make_candidate(&prefix, "s3", 0.05, "decision3", "rationale3");
        let upd3 = c.update_with_candidates(
            &ws,
            &empty_capsule("q3"),
            None,
            &[],
            std::slice::from_ref(&cand3),
            Some("ses-2"),
            now + 120_000,
        );
        assert_eq!(upd3.cause, "resurfacing");
        assert!(upd3.note.is_some());
    }

    #[test]
    fn structural_weight_halves_score_when_rationale_missing() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        // distance 0.10 => similarity 0.90, but no rationale => weight 0.5 => score 0.45
        let cand = make_candidate(&prefix, "thin", 0.10, "decision only", "");
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("q"),
            None,
            &[],
            std::slice::from_ref(&cand),
            Some("ses"),
            1_730_000_000_000,
        );
        // Score is below the firing threshold, so no note despite high similarity.
        assert!(upd.channels.recurrence_signal < REC_FIRE_THRESHOLD);
        assert!(upd.note.is_none());
    }

    #[test]
    fn dormancy_tiebreaks_equal_score_to_older_capsule() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        let newer = {
            let mut h = make_candidate(&prefix, "newer", 0.10, "same dec", "same rat");
            h.ts_ms = 1_720_000_000_000;
            h
        };
        let older = {
            let mut h = make_candidate(&prefix, "older", 0.10, "same dec", "same rat");
            h.ts_ms = 1_650_000_000_000;
            h
        };
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("q"),
            None,
            &[],
            &[newer, older],
            Some("ses"),
            1_730_000_000_000,
        );
        let m = upd.recurrence_match.expect("expected a match");
        assert!(m.capsule_id.ends_with("-older"));
    }

    #[test]
    fn cross_workspace_match_tags_origin_and_renders_in_note() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        // Synthesise a candidate that was retrieved from a *different*
        // workspace (`origin_workspace_id` set to a peer id).
        let peer_ws = format!("wks_peer_{}", uuid::Uuid::new_v4().simple());
        let mut cand = make_candidate(&prefix, "peer", 0.08, "shared concern", "rationale");
        cand.origin_workspace_id = Some(peer_ws.clone());
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("question that echoes elsewhere"),
            None,
            &[],
            std::slice::from_ref(&cand),
            Some("ses-now"),
            1_730_000_000_000,
        );
        let m = upd.recurrence_match.expect("expected a match");
        assert_eq!(m.origin_workspace_id.as_deref(), Some(peer_ws.as_str()));
        let note = upd.note.expect("expected a SYSTEM NOTE");
        // No registry entry exists for `peer_ws`, so the renderer falls back
        // to the raw id — but the " in " clause must still appear.
        assert!(
            note.contains(" in "),
            "note should include 'in <project>' clause; got: {note}"
        );
        assert!(note.contains(&peer_ws));
    }

    #[test]
    fn same_workspace_match_omits_origin_clause() {
        let mut c = TrajectoryController::default();
        let (ws, prefix) = test_ids();
        let _cleanup = WorkspaceCleanup::new(&ws, &prefix);
        // origin_workspace_id == current workspace → no " in <project>" clause.
        let mut cand = make_candidate(&prefix, "local", 0.08, "local decision", "local rationale");
        cand.origin_workspace_id = Some(ws.clone());
        let upd = c.update_with_candidates(
            &ws,
            &empty_capsule("q"),
            None,
            &[],
            std::slice::from_ref(&cand),
            Some("ses-now"),
            1_730_000_000_000,
        );
        let m = upd.recurrence_match.expect("expected a match");
        // origin_workspace_id is dropped because it equals the current ws.
        assert!(m.origin_workspace_id.is_none());
        let note = upd.note.expect("expected a SYSTEM NOTE");
        assert!(
            !note.contains(" in "),
            "note should NOT include 'in <project>' clause; got: {note}"
        );
    }
}
