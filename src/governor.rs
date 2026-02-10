//! Friction-based loop detection using existing signals.
//!
//! No frameworks, no state machines - just cheap heuristics:
//! - User emotion (frustration/anger/annoyance/disapproval from ONNX model)
//! - Symbol repetition (same files touched repeatedly)
//!
//! Total overhead: <15ms (local emotion + LanceDB query + matching)

use crate::CapsuleHit;
use crate::IntentCapsule;

#[derive(Debug, Clone, Copy)]
pub(crate) struct FrictionWeights {
    /// How many recent capsules are treated as the "immediate" window.
    pub(crate) recent_window: usize,
    /// Symbol repetition threshold within `recent_window`.
    pub(crate) symbol_repeat_threshold: usize,
    /// How many recent capsules are considered when ranking Graveyard nodes.
    pub(crate) graveyard_window: usize,
    /// How many recent capsules are considered when building the Map.
    pub(crate) map_window: usize,

    /// Controls decay of the Logic Recency base. Larger => slower decay.
    pub(crate) logic_recency_tau: f32,

    /// Multiplier for Jaccard symbol overlap in Graveyard scoring.
    pub(crate) overlap_scale: f32,
    /// Multiplicative boost applied when friction emotion is present.
    pub(crate) emotion_boost: f32,

    /// Token count at which the effort boost saturates.
    pub(crate) effort_tokens_norm: f32,
    /// Max effort multiplier contribution (added on top of 1.0).
    pub(crate) effort_scale: f32,

    /// Token count at which we add the stronger warning copy.
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
    /// One short reminder of the top-level goal (the "North Star").
    pub(crate) north_star: Option<String>,
    /// 3-5 most recent attempts/failures in the same code area (the "Graveyard").
    pub(crate) graveyard: Vec<HydrationNode>,
    /// A compact symbolic map of the territory touched recently (the "Map").
    pub(crate) map: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct HydrationNode {
    pub(crate) symbols: Vec<String>,
    pub(crate) intent: String,
    pub(crate) decision: String,
    pub(crate) user_emotion: Option<String>,
    pub(crate) tokens_total: Option<i64>,
}

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

/// Stateless friction note for the very first user message (no history yet).
///
/// When history is empty, the main friction heuristics can't run (they rely on
/// repetition and past emotion). However, we still want to inject a lightweight
/// warning when the user is clearly upset so the assistant can acknowledge and
/// re-align immediately.
pub(crate) fn evaluate_stateless_friction(
    current_user_emotion: Option<&crate::emotion::EmotionMeta>,
) -> Option<String> {
    let e = current_user_emotion?;
    if !FRICTION_EMOTIONS.contains(&e.label.as_str()) {
        return None;
    }

    // Keep thresholds low-ish; we prefer a gentle nudge over missing obvious frustration.
    if e.confidence < 0.45 || e.intensity < 0.35 {
        return None;
    }

    Some(
        "[SYSTEM NOTE: User frustration detected. Pause to acknowledge the concern and ask what they want to achieve next before continuing.]"
            .to_string(),
    )
}

/// Check whether we should inject a *friction warning* into the next LLM request.
///
/// This is intentionally a small, cheap heuristic (no frameworks, no state machines).
/// The goal is to catch the common failure mode where an agent keeps hammering the same
/// files/symbols while the user is clearly not happy with progress.
///
/// What it looks at
/// - **Current context**: the symbols extracted from the request we are about to send.
/// - **Current user emotion** (optional): a local classifier output for *this* user message.
/// - **Recent history**: the last few recorded capsules (for the same workspace/session).
///
/// Signals
/// 1. **Friction emotion**: user emotion buckets that correlate with “we're stuck / this isn't working”.
///    - Current message counts (when provided).
///    - Recent history counts (best-effort; may be missing).
/// 2. **Symbol repetition**: if the same symbols appear repeatedly in recent capsules, we assume
///    we are stuck in the same code area.
/// 3. **Effort amplifier**: if recent repeated-symbol capsules show high token usage, we make the
///    warning more forceful (this is a proxy for “we already burned time/money here”).
///
/// Trigger
/// - We inject a warning when **(friction emotion)** AND **(symbol repetition >= 2)**.
///
/// Output
/// - Returns `Some(warning)` containing a small SYSTEM note intended to be *prepended* to the
///   next upstream request (see `crate::net::inject_warning`).
/// - Returns `None` when no warning should be injected.
///
/// Design notes / trade-offs
/// - We require symbols to be present; otherwise we cannot tell *where* the loop is.
/// - Emotion classification is noisy; we therefore combine it with concrete “same symbols again” evidence.
/// - This function must stay cheap: it only scans a small prefix of `history`.
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
    if history.is_empty() || current.symbols.is_empty() {
        return None;
    }

    // Treat the most recent items first even if storage returns an arbitrary order.
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    // Signal 1: User friction emotion (current message OR recent history)
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

    // Signal 2: Same symbols touched 2+ times recently
    let symbol_repeats = recent
        .iter()
        .take(w.recent_window)
        .filter(|h| {
            h.capsule
                .symbols
                .iter()
                .any(|s| current.symbols.contains(s))
        })
        .count();

    // Amplifier: if we're burning tokens in the same area, be more forceful.
    let effort_tokens: i64 = recent
        .iter()
        .take(w.recent_window)
        .filter(|h| {
            h.capsule
                .symbols
                .iter()
                .any(|s| current.symbols.contains(s))
        })
        .filter_map(|h| h.meta.usage.as_ref().and_then(|u| u.tokens_total()))
        .sum();

    // Trigger: User is frustrated AND agent is stuck in the same code area
    if frustrated && symbol_repeats >= w.symbol_repeat_threshold {
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

        // Log friction detection with metadata
        let user_emotion_label = current_user_emotion
            .map(|e| e.label.as_str())
            .unwrap_or("none");
        tracing::info!(
            friction = "detected",
            symbols = %symbols_str,
            symbol_repeats = symbol_repeats,
            effort_tokens = effort_tokens,
            user_emotion = user_emotion_label,
            current_friction = current_friction,
            recent_friction = recent_friction,
            has_hydration = packet.is_some(),
            "friction warning will be injected"
        );

        return Some(render_hydration_warning(
            &symbols_str,
            amplifier,
            packet.as_ref(),
        ));
    }

    // If symbol-based friction didn't trigger, check for conversational friction
    // (symbol-less frustration detection)
    evaluate_conversational_friction(history, w.recent_window)
}

/// Soft warning for 2 frustrated messages in window
const SOFT_FRICTION_WARNING: &str =
    "The user seems frustrated. Consider pausing to acknowledge their concern before continuing.";

/// Firm warning for 3+ frustrated messages in window  
const FIRM_FRICTION_WARNING: &str = "The user has expressed repeated frustration. Stop and ask what's wrong or how you can help differently.";

/// Evaluate conversational friction based on user emotion patterns alone.
///
/// This catches cases where the user is frustrated with the conversation
/// but no specific code symbols are involved.
///
/// Returns:
/// - None if fewer than 2 negative emotions in window
/// - Soft warning if exactly 2 negative emotions
/// - Firm warning if 3+ negative emotions
pub(crate) fn evaluate_conversational_friction(
    history: &[CapsuleHit],
    window: usize,
) -> Option<String> {
    if history.is_empty() {
        return None;
    }

    // Sort by recency
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    // Filter to capsules that have user emotion (proxy for "has user message")
    // and count negative emotions in the window
    let frustrated_count = recent
        .iter()
        .take(window)
        .filter(|h| h.user_emotion.is_some())
        .filter(|h| {
            h.user_emotion
                .as_ref()
                .map(|e| crate::emotion::is_negative_emotion(&e.label))
                .unwrap_or(false)
        })
        .count();

    if frustrated_count >= 3 {
        tracing::info!(
            friction = "conversational",
            frustrated_count = frustrated_count,
            window = window,
            severity = "firm",
            "conversational friction warning will be injected"
        );
        return Some(FIRM_FRICTION_WARNING.to_string());
    }

    if frustrated_count >= 2 {
        tracing::info!(
            friction = "conversational",
            frustrated_count = frustrated_count,
            window = window,
            severity = "soft",
            "conversational friction warning will be injected"
        );
        return Some(SOFT_FRICTION_WARNING.to_string());
    }

    None
}

/// Warning messages for LLM-detected failure modes
const DRIFT_WARNING: &str = "Previous context may be stale or incorrect. The last exchange showed signs of drift (wrong mental model). Verify your assumptions about the codebase before proceeding.";

const FALSE_PROGRESS_WARNING: &str = "The user disputed completion in a recent exchange. Verify that your changes actually work before claiming the task is done.";

const REDISCOVERY_WARNING: &str = "This topic was already discussed recently. Check the prior decision before re-exploring the same ground.";

const DECISION_CONFLICT_WARNING: &str = "Intervention: This approach conflicts with a prior project decision. Re-route to the compliant pattern.";

/// Evaluate recent capsules for LLM-detected failure modes.
///
/// Returns a warning if a relevant failure mode was detected in recent history.
/// Only looks at the most recent capsule to avoid stale warnings.
pub(crate) fn evaluate_failure_modes(history: &[CapsuleHit]) -> Option<String> {
    if history.is_empty() {
        return None;
    }

    // Sort by recency and look at the most recent capsule
    let mut recent: Vec<&CapsuleHit> = history.iter().collect();
    recent.sort_by_key(|h| std::cmp::Reverse(h.ts_ms));

    // Check the most recent capsule for failure modes
    // (Only look at the latest to avoid repeatedly warning about old issues)
    let latest = recent.first()?;

    let failure_mode = &latest.capsule.failure_mode;
    let failure_signals = latest.capsule.failure_signals.as_deref().unwrap_or("");
    let symbols = &latest.capsule.symbols;

    match failure_mode {
        crate::types::FailureMode::Drift => {
            let symbols_str = if symbols.is_empty() {
                String::new()
            } else {
                format!(" Relevant symbols: {}", symbols.join(", "))
            };

            tracing::info!(
                failure_mode = "drift",
                failure_signals = failure_signals,
                symbols = ?symbols,
                "injecting drift warning"
            );

            Some(format!("{}{}", DRIFT_WARNING, symbols_str))
        }
        crate::types::FailureMode::DecisionConflict => {
            // Prefer a concrete prior decision reminder if it exists in the capsule.
            let prior_decision = if !latest.capsule.decision.is_empty() {
                format!(" Prior decision: \"{}\"", latest.capsule.decision)
            } else {
                String::new()
            };

            tracing::info!(
                failure_mode = "decision_conflict",
                failure_signals = failure_signals,
                prior_decision = %latest.capsule.decision,
                "injecting decision conflict warning"
            );

            Some(format!("{}{}", DECISION_CONFLICT_WARNING, prior_decision))
        }
        crate::types::FailureMode::FalseProgress => {
            tracing::info!(
                failure_mode = "false_progress",
                failure_signals = failure_signals,
                "injecting false progress warning"
            );

            Some(FALSE_PROGRESS_WARNING.to_string())
        }
        crate::types::FailureMode::Rediscovery => {
            // For rediscovery, include the prior decision if available
            let prior_decision = if !latest.capsule.decision.is_empty() {
                format!(" Prior decision: \"{}\"", latest.capsule.decision)
            } else {
                String::new()
            };

            tracing::info!(
                failure_mode = "rediscovery",
                failure_signals = failure_signals,
                prior_decision = %latest.capsule.decision,
                "injecting rediscovery warning"
            );

            Some(format!("{}{}", REDISCOVERY_WARNING, prior_decision))
        }
        // RetrySpiral is already handled by symbol-based friction detection
        // UnboundedHorizon is out of scope for now
        _ => None,
    }
}

fn has_any_marker(s: &str, markers: &[&str]) -> bool {
    markers.iter().any(|m| s.contains(m))
}

fn tokenize_keywords(s: &str) -> std::collections::HashSet<String> {
    // ASCII-only, cheap tokenization; keep identifiers/paths-like tokens.
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
        // Very small stopword set; keep it conservative.
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
    // If user is explicitly saying to avoid/not-do the token, treat as not a conflict.
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
        // Special-case: "no <token>".
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

/// Detect a likely "I told you NOT to" moment: current request asks for something that
/// conflicts with an established decision capsule.
///
/// This is intentionally heuristic and cheap (no extra LLM call):
/// - semantic retrieval happens upstream (caller provides relevant hits)
/// - we look for strong prohibition language in `capsule.decision`
/// - we look for overlapping keywords between the prompt and the decision
pub(crate) fn evaluate_decision_conflict(prompt: &str, hits: &[CapsuleHit]) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    if prompt_lc.trim().is_empty() {
        return None;
    }
    if !prompt_requests_action(&prompt_lc) {
        return None;
    }

    // Pick the strongest nearby decision (first relevant hit with a non-empty decision).
    let best = hits
        .iter()
        .find(|h| !h.capsule.decision.trim().is_empty())?;
    let decision = best.capsule.decision.trim();
    let decision_lc = decision.to_ascii_lowercase();

    // Only trigger this mode when the stored decision reads like a constraint.
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

    // Keyword overlap is the guardrail against random matches.
    let p_tok = tokenize_keywords(&prompt_lc);
    let d_tok = tokenize_keywords(&decision_lc);
    let shared: Vec<String> = p_tok.intersection(&d_tok).cloned().collect();
    if shared.is_empty() {
        return None;
    }

    // Confirm at least one shared token is actually prohibited and not negated in the prompt.
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

    // --- North Star (goal reminder) ---
    let north_star = if !current.intent.trim().is_empty() {
        Some(current.intent.trim().to_string())
    } else {
        recent.iter().find_map(|h| {
            (!h.capsule.intent.trim().is_empty()).then(|| h.capsule.intent.trim().to_string())
        })
    };

    // --- Graveyard (recent failed/repeated nodes) ---
    // Rank by a lightweight "Logic Recency" score: most recent hit wins, then prefer same-symbol overlap,
    // then prefer explicit friction emotion, then tokens usage.
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
        let logic_recency = (-((i as f32) / tau)).exp(); // ~1.0, 0.72, 0.51, ...
        let emo = h.user_emotion.as_ref().map(|e| e.label.clone());
        let emo_is_friction = emo.as_deref().map(is_friction_label).unwrap_or(false)
            || current_user_emotion
                .map(|e| is_friction_label(&e.label))
                .unwrap_or(false);
        let emo_boost = if emo_is_friction {
            if w.emotion_boost < 1.0 {
                1.0
            } else {
                w.emotion_boost
            }
        } else {
            1.0
        };
        let overlap_scale = if w.overlap_scale < 0.0 {
            0.0
        } else {
            w.overlap_scale
        };
        let overlap_boost = 1.0 + (overlap * overlap_scale);
        let tok = h.meta.usage.as_ref().and_then(|u| u.tokens_total());
        let norm = if w.effort_tokens_norm <= 1.0 {
            1.0
        } else {
            w.effort_tokens_norm
        };
        let effort_scale = if w.effort_scale < 0.0 {
            0.0
        } else {
            w.effort_scale
        };
        let effort_boost = tok
            .map(|t| 1.0 + ((t as f32) / norm).clamp(0.0, 1.0) * effort_scale)
            .unwrap_or(1.0);

        let score = logic_recency * overlap_boost * emo_boost * effort_boost;
        candidates.push((
            score,
            HydrationNode {
                symbols: h.capsule.symbols.clone(),
                intent: h.capsule.intent.clone(),
                decision: h.capsule.decision.clone(),
                user_emotion: emo,
                tokens_total: tok,
            },
        ));
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Keep diversity: avoid returning 5 nodes that are effectively the same symbol set.
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
    if graveyard.len() > 5 {
        graveyard.truncate(5);
    }
    if graveyard.len() < 3 {
        // Not enough overlap candidates; still useful to return a packet with the Map.
        // (We keep whatever we found.)
    }

    // --- Map (symbolic territory graph) ---
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
        // Keep the note compact but structured.
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
    fn test_stateless_friction_note_not_emitted_for_neutral() {
        let e = EmotionMeta {
            label: "neutral".to_string(),
            valence: 0.0,
            intensity: 0.2,
            confidence: 0.9,
        };
        assert!(evaluate_stateless_friction(Some(&e)).is_none());
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
    fn test_no_friction_empty_symbols() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec![],
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        let history = vec![make_hit(vec!["auth.ts"], Some("frustration"))];
        assert!(evaluate_friction(&current, None, &history).is_none());
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
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        let history = vec![
            make_hit(vec!["auth.ts"], Some("joy")),
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], None),
        ];
        assert!(evaluate_friction(&current, None, &history).is_none());
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
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        let history = vec![
            // Negative emotions would trigger *conversational* friction even without symbol overlap.
            // This test is specifically about symbol-based loops.
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], Some("joy")),
        ];
        assert!(evaluate_friction(&current, None, &history).is_none());
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

    #[test]
    fn test_friction_with_annoyance() {
        let current = IntentCapsule {
            category: "".to_string(),
            intent: "".to_string(),
            decision: "".to_string(),
            rationale: "".to_string(),
            next_steps: vec![],
            symbols: vec!["db.rs".to_string()],
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
        };
        let history = vec![
            make_hit(vec!["db.rs"], Some("annoyance")),
            make_hit(vec!["db.rs", "models.rs"], None),
        ];
        let result = evaluate_friction(&current, None, &history);
        assert!(result.is_some());
    }

    #[test]
    fn test_friction_current_emotion_triggers() {
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
        // Not frustrated in history, but we are repeating symbols.
        let history = vec![
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], Some("joy")),
            make_hit(vec!["auth.ts"], None),
        ];

        let cur_emotion = EmotionMeta {
            label: "frustration".to_string(),
            valence: -0.7,
            intensity: 0.8,
            confidence: 0.9,
        };

        let result = evaluate_friction(&current, Some(&cur_emotion), &history);
        assert!(result.is_some());
    }

    #[test]
    fn test_no_friction_current_emotion_happy() {
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
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], Some("neutral")),
            make_hit(vec!["auth.ts"], None),
        ];

        let cur_emotion = EmotionMeta {
            label: "joy".to_string(),
            valence: 0.8,
            intensity: 0.4,
            confidence: 0.9,
        };
        let result = evaluate_friction(&current, Some(&cur_emotion), &history);
        assert!(result.is_none());
    }
}
