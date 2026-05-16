use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MetricsEvent {
    CapsuleSaved {
        ts_ms: i64,
        conn_id: u64,
        exchange_seq: u64,
        source: String,
        upstream_host: String,
        request_path: String,
        http_status: u16,
        agent_session_id: Option<String>,
        tokens_total: Option<i64>,
        tokens_input: Option<i64>,
        cost: Option<f64>,
        symbols_total: usize,
        paths_checked: usize,
        paths_missing: usize,
        user_emotion: Option<String>,
        assistant_emotion: Option<String>,
        /// Failure mode detected by LLM semantic analysis
        failure_mode: Option<String>,
        /// TurnEval: composite trajectory intensity (tune)
        #[serde(default)]
        te_intensity: Option<f32>,
        /// TurnEval: developer coaching clarity score
        #[serde(default)]
        te_clarity: Option<f32>,
        /// TurnEval: context freshness score
        #[serde(default)]
        te_context_freshness: Option<f32>,
        /// TurnEval: behavioral flags (comma-joined)
        #[serde(default)]
        te_flags: Option<String>,
        /// TurnEval: token spend acceleration (0–1)
        #[serde(default)]
        te_cost_acceleration: Option<f32>,
    },
    FrictionWarningInjected {
        ts_ms: i64,
        conn_id: u64,
        workspace_id: String,
        #[serde(default)]
        agent_session_id: Option<String>,
        symbols: Vec<String>,
        user_emotion: Option<String>,
        intensity: f32,
        cause: String,
        /// Top contributing symptom channels at trigger time
        #[serde(default)]
        top_channels: std::collections::HashMap<String, f32>,
        /// The user's intent that triggered the intervention
        #[serde(default)]
        topic: Option<String>,
        /// When the controller first entered the "Watch" state
        #[serde(default)]
        watch_start_ts: Option<i64>,
    },
    CommandQuery {
        ts_ms: i64,
        query_len: usize,
        limit: usize,
        has_symbol_filter: bool,
        has_emotion_filter: bool,
        has_provider_filter: bool,
    },
    CommandRecall {
        ts_ms: i64,
        target_len: usize,
        limit: usize,
        has_emotion_filter: bool,
        has_provider_filter: bool,
    },
    /// Emitted by the recurrence channel each time a dormant capsule is surfaced
    /// via the resurfacing basin (standalone or as a modifier on another basin).
    /// See `internal/SOURCE_POINTERS.md` §Recurrence Channel.
    ResurfacingEmitted {
        ts_ms: i64,
        workspace_id: String,
        agent_session_id: Option<String>,
        matched_capsule_id: String,
        similarity: f32,
        /// `"standalone"` (no other basin firing) or `"modifier"` (appended to
        /// another basin's intervention).
        mode: String,
        candidate_age_days: i64,
    },
}

fn append_event(path: &std::path::Path, ev: &MetricsEvent) -> anyhow::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let line = serde_json::to_string(ev)?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{}\n", line).as_bytes())?;
    Ok(())
}

pub(crate) fn record_capsule_saved(
    ws: &crate::WorkspacePaths,
    ts_ms: i64,
    conn_id: u64,
    exchange_seq: u64,
    meta: &crate::ResponseMeta,
    user_emotion: Option<&crate::emotion::EmotionMeta>,
    assistant_emotion: Option<&crate::emotion::EmotionMeta>,
    capsule: &crate::IntentCapsule,
    turn_eval: &crate::types::TurnEval,
) -> anyhow::Result<()> {
    let (paths_checked, paths_missing) = crate::workspace::validate_paths(&ws.id, &capsule.symbols);
    let (tokens_total, tokens_input, cost) = meta
        .usage
        .as_ref()
        .map(|u| (u.tokens_total(), u.tokens_input, u.cost))
        .unwrap_or((None, None, None));

    let failure_mode_str = match capsule.failure_mode {
        crate::types::FailureMode::None => None,
        crate::types::FailureMode::Drift => Some("drift".to_string()),
        crate::types::FailureMode::Rediscovery => Some("rediscovery".to_string()),
        crate::types::FailureMode::DecisionConflict => Some("decision_conflict".to_string()),
        crate::types::FailureMode::RetrySpiral => Some("retry_spiral".to_string()),
        crate::types::FailureMode::FalseProgress => Some("false_progress".to_string()),
        crate::types::FailureMode::UnboundedHorizon => Some("unbounded_horizon".to_string()),
    };

    let te_flags_str = if turn_eval.flags.is_empty() {
        None
    } else {
        Some(turn_eval.flags.join(","))
    };

    let ev = MetricsEvent::CapsuleSaved {
        ts_ms,
        conn_id,
        exchange_seq,
        source: meta.source.clone(),
        upstream_host: meta.upstream_host.clone(),
        request_path: meta.request_path.clone(),
        http_status: meta.http_status,
        agent_session_id: meta.agent_session_id.clone(),
        tokens_total,
        tokens_input,
        cost,
        symbols_total: capsule.symbols.len(),
        paths_checked,
        paths_missing,
        user_emotion: user_emotion.map(|e| e.label.clone()),
        assistant_emotion: assistant_emotion.map(|e| e.label.clone()),
        failure_mode: failure_mode_str,
        te_intensity: if turn_eval.trajectory_intensity > 0.0 {
            Some(turn_eval.trajectory_intensity)
        } else {
            None
        },
        te_clarity: if turn_eval.clarity > 0.0 {
            Some(turn_eval.clarity)
        } else {
            None
        },
        te_context_freshness: if turn_eval.context_freshness > 0.0 {
            Some(turn_eval.context_freshness)
        } else {
            None
        },
        te_flags: te_flags_str,
        te_cost_acceleration: if turn_eval.cost_acceleration > 0.0 {
            Some(turn_eval.cost_acceleration)
        } else {
            None
        },
    };
    append_event(&ws.metrics_jsonl, &ev)
}

pub(crate) fn record_friction_warning_injected(
    workspace_id: &str,
    conn_id: u64,
    agent_session_id: Option<String>,
    symbols: Vec<String>,
    user_emotion: Option<&crate::emotion::EmotionMeta>,
    intensity: f32,
    cause: String,
    channels: Option<&crate::types::SymptomChannels>,
    topic: Option<String>,
    watch_start_ts: Option<i64>,
) -> anyhow::Result<()> {
    let ws_dir = crate::unlost_workspace_dir(workspace_id);
    let path = ws_dir.join("metrics.jsonl");

    let mut top_channels = std::collections::HashMap::new();
    if let Some(c) = channels {
        // Only include channels with non-zero contribution
        if c.repetition > 0.01 {
            top_channels.insert("repetition".to_string(), c.repetition);
        }
        if c.novelty_collapse > 0.01 {
            top_channels.insert("novelty".to_string(), c.novelty_collapse);
        }
        if c.semantic_stall > 0.01 {
            top_channels.insert("semantic".to_string(), c.semantic_stall);
        }
        if c.effort_spike > 0.01 {
            top_channels.insert("effort".to_string(), c.effort_spike);
        }
        if c.alignment_debt > 0.01 {
            top_channels.insert("alignment".to_string(), c.alignment_debt);
        }
        if c.path_hallucination > 0.01 {
            top_channels.insert("hallucination".to_string(), c.path_hallucination);
        }
        if c.grounding_stall > 0.01 {
            top_channels.insert("stall".to_string(), c.grounding_stall);
        }
        if c.instruction_staticness > 0.01 {
            top_channels.insert("staticness".to_string(), c.instruction_staticness);
        }
        if c.logic_churn > 0.01 {
            top_channels.insert("churn".to_string(), c.logic_churn);
        }
        if c.fluency > 0.01 {
            top_channels.insert("fluency".to_string(), c.fluency);
        }
    }

    let ev = MetricsEvent::FrictionWarningInjected {
        ts_ms: crate::now_ms(),
        conn_id,
        workspace_id: workspace_id.to_string(),
        agent_session_id,
        symbols,
        user_emotion: user_emotion.map(|e| e.label.clone()),
        intensity,
        cause,
        top_channels,
        topic,
        watch_start_ts,
    };
    append_event(&path, &ev)
}

pub(crate) fn record_resurfacing_emitted(
    workspace_id: &str,
    agent_session_id: Option<String>,
    matched_capsule_id: &str,
    similarity: f32,
    mode: &str,
    candidate_age_days: i64,
) -> anyhow::Result<()> {
    let ws_dir = crate::unlost_workspace_dir(workspace_id);
    let path = ws_dir.join("metrics.jsonl");
    let ev = MetricsEvent::ResurfacingEmitted {
        ts_ms: crate::now_ms(),
        workspace_id: workspace_id.to_string(),
        agent_session_id,
        matched_capsule_id: matched_capsule_id.to_string(),
        similarity,
        mode: mode.to_string(),
        candidate_age_days,
    };
    append_event(&path, &ev)
}

pub(crate) fn record_command_query(
    ws: &crate::WorkspacePaths,
    query: &str,
    limit: usize,
    symbol: Option<&str>,
    emotion: Option<&str>,
    provider: Option<&str>,
) -> anyhow::Result<()> {
    let ev = MetricsEvent::CommandQuery {
        ts_ms: crate::now_ms(),
        query_len: query.len(),
        limit,
        has_symbol_filter: symbol.map(|s| !s.trim().is_empty()).unwrap_or(false),
        has_emotion_filter: emotion.map(|s| !s.trim().is_empty()).unwrap_or(false),
        has_provider_filter: provider.map(|s| !s.trim().is_empty()).unwrap_or(false),
    };
    append_event(&ws.metrics_jsonl, &ev)
}

pub(crate) fn record_command_recall(
    ws: &crate::WorkspacePaths,
    target: &str,
    limit: usize,
    emotion: Option<&str>,
    provider: Option<&str>,
) -> anyhow::Result<()> {
    let ev = MetricsEvent::CommandRecall {
        ts_ms: crate::now_ms(),
        target_len: target.len(),
        limit,
        has_emotion_filter: emotion.map(|s| !s.trim().is_empty()).unwrap_or(false),
        has_provider_filter: provider.map(|s| !s.trim().is_empty()).unwrap_or(false),
    };
    append_event(&ws.metrics_jsonl, &ev)
}

#[derive(Default, Debug, Clone)]
pub(crate) struct FailureModeCounts {
    pub(crate) drift: u64,
    pub(crate) rediscovery: u64,
    pub(crate) decision_conflict: u64,
    pub(crate) retry_spiral: u64,
    pub(crate) false_progress: u64,
    pub(crate) unbounded_horizon: u64,
}

impl FailureModeCounts {
    pub(crate) fn total(&self) -> u64 {
        self.drift
            + self.rediscovery
            + self.decision_conflict
            + self.retry_spiral
            + self.false_progress
            + self.unbounded_horizon
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExpensiveIntervention {
    pub(crate) ts_ms: i64,
    pub(crate) cause: String,
    pub(crate) intensity: f32,
    pub(crate) symbols: Vec<String>,
    pub(crate) cost_next_5: f64,
    #[allow(dead_code)]
    pub(crate) tokens_next_5: i64,
}

/// A friction warning intervention for display in recall
#[derive(Debug, Clone)]
pub(crate) struct Intervention {
    pub(crate) ts_ms: i64,
    pub(crate) cause: String,
    pub(crate) intensity: f32,
    pub(crate) symbols: Vec<String>,
    pub(crate) user_emotion: Option<String>,
    pub(crate) top_channels: std::collections::HashMap<String, f32>,
    pub(crate) topic: Option<String>,
    pub(crate) watch_start_ts: Option<i64>,
}

pub(crate) fn get_diagnosis(
    cause: &str,
    channels: &std::collections::HashMap<String, f32>,
) -> String {
    let mut best_chan = "";
    let mut best_score = -1.0;
    for (chan, score) in channels {
        if *score > best_score {
            best_score = *score;
            best_chan = chan;
        }
    }

    match best_chan {
        "hallucination" => "Factual drift (hallucinating paths)".to_string(),
        "stall" => "Grounding failure (ignoring user files)".to_string(),
        "repetition" => "Repetitive stall".to_string(),
        "churn" => "Logic churn (circular editing)".to_string(),
        "alignment" => "Alignment divergence".to_string(),
        "staticness" => "Instruction freeze".to_string(),
        "fluency" => "Blind acceptance (assistant verbosity)".to_string(),
        _ => match cause {
            "drift" => "Factual drift".to_string(),
            "loop" => "Repetitive loop".to_string(),
            "spec" => "Alignment divergence".to_string(),
            _ => format!("Trajectory friction ({})", cause),
        },
    }
}

pub(crate) fn get_severity_label(intensity: f32) -> &'static str {
    if intensity >= 0.95 {
        "Acute"
    } else if intensity >= 0.88 {
        "Strong"
    } else {
        "Significant"
    }
}

/// Read the last N friction warning interventions from metrics
pub(crate) fn get_recent_interventions(
    path: &std::path::Path,
    n: usize,
) -> anyhow::Result<Vec<Intervention>> {
    use std::io::BufRead;

    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(Vec::new()),
    };

    let reader = std::io::BufReader::new(f);
    let mut interventions = Vec::new();

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<MetricsEvent>(&line) else {
            continue;
        };

        if let MetricsEvent::FrictionWarningInjected {
            ts_ms,
            cause,
            intensity,
            symbols,
            user_emotion,
            top_channels,
            topic,
            watch_start_ts,
            ..
        } = ev
        {
            interventions.push(Intervention {
                ts_ms,
                cause,
                intensity,
                symbols,
                user_emotion,
                top_channels,
                topic,
                watch_start_ts,
            });
        }
    }

    // Sort by timestamp descending and take last N
    interventions.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
    interventions.truncate(n);

    Ok(interventions)
}

#[derive(Default, Debug, Clone)]
pub(crate) struct MetricsSummary {
    pub(crate) capsules: u64,
    pub(crate) tokens_total: i64,
    pub(crate) cost_total: f64,
    pub(crate) drift_paths_checked: u64,
    pub(crate) drift_paths_missing: u64,
    pub(crate) friction_warnings: u64,
    pub(crate) friction_intensity_total: f32,
    pub(crate) friction_by_cause: std::collections::HashMap<String, u64>,
    pub(crate) friction_by_symbol: std::collections::HashMap<String, u64>,
    pub(crate) query_commands: u64,
    pub(crate) recall_commands: u64,
    /// Failure modes detected via LLM semantic analysis
    pub(crate) failure_modes: FailureModeCounts,
    /// Average tokens between interventions across all sessions
    pub(crate) avg_tokens_between_interventions: f64,
    /// Breakdown of friction rate by input token buckets (context growth proxy)
    /// Key is lower bound of bucket (e.g. 0, 4000, 8000), value is (warnings, total_turns_in_bucket)
    pub(crate) friction_by_input_bucket: std::collections::BTreeMap<i64, (u64, i64)>,
    /// Top expensive interventions (by cost of next 5 turns)
    pub(crate) top_expensive_interventions: Vec<ExpensiveIntervention>,
    /// Stats for the last 24 hours
    pub(crate) recent_capsules: u64,
    pub(crate) recent_cost_total: f64,
    pub(crate) recent_friction_warnings: u64,
    /// Contributions of individual symptom channels to all warnings
    pub(crate) channel_contributions: std::collections::HashMap<String, f32>,
    /// Average verbosity ratio (assistant tokens / user tokens)
    pub(crate) avg_fluency: f32,
}

pub(crate) fn summarize_metrics(path: &std::path::Path) -> anyhow::Result<MetricsSummary> {
    use std::io::BufRead;

    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(MetricsSummary::default()),
    };
    let mut out = MetricsSummary::default();
    let reader = std::io::BufReader::new(f);

    let mut all_events = Vec::new();
    let blacklist = [
        "NOTE",
        "REASON",
        "DECISION",
        "INTENT",
        "NEXT_STEPS",
        "RATIONALE",
        "SYMBOLS",
        "USER",
        "ASSISTANT",
        "SYSTEM",
        "SUCCESS",
        "FAILURE",
        "ERROR",
        "WARNING",
    ];

    let now = crate::now_ms();
    let day_ms = 24 * 60 * 60 * 1000;
    let mut total_fluency = 0.0;
    let mut turns_with_fluency = 0;

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<MetricsEvent>(&line) else {
            continue;
        };

        match &ev {
            MetricsEvent::CapsuleSaved {
                ts_ms,
                tokens_total,
                tokens_input,
                cost,
                paths_checked,
                paths_missing,
                failure_mode,
                ..
            } => {
                out.capsules += 1;
                if now - *ts_ms < day_ms {
                    out.recent_capsules += 1;
                }
                if let Some(t) = tokens_total {
                    out.tokens_total = out.tokens_total.saturating_add(*t);
                    if let Some(input) = tokens_input {
                        let output = t - input;
                        if *input > 0 {
                            total_fluency += output as f32 / *input as f32;
                            turns_with_fluency += 1;
                        }
                    }
                }
                if let Some(c) = cost {
                    out.cost_total += c;
                    if now - *ts_ms < day_ms {
                        out.recent_cost_total += c;
                    }
                }
                out.drift_paths_checked += *paths_checked as u64;
                out.drift_paths_missing += *paths_missing as u64;

                if let Some(mode) = failure_mode {
                    match mode.as_str() {
                        "drift" => out.failure_modes.drift += 1,
                        "rediscovery" => out.failure_modes.rediscovery += 1,
                        "decision_conflict" => out.failure_modes.decision_conflict += 1,
                        "retry_spiral" => out.failure_modes.retry_spiral += 1,
                        "false_progress" => out.failure_modes.false_progress += 1,
                        "unbounded_horizon" => out.failure_modes.unbounded_horizon += 1,
                        _ => {}
                    }
                }
            }
            MetricsEvent::FrictionWarningInjected {
                ts_ms,
                intensity,
                cause,
                symbols,
                top_channels,
                ..
            } => {
                out.friction_warnings += 1;
                if now - *ts_ms < day_ms {
                    out.recent_friction_warnings += 1;
                }
                out.friction_intensity_total += *intensity;
                *out.friction_by_cause.entry(cause.clone()).or_insert(0) += 1;
                for s in symbols {
                    let cleaned = s.trim_matches(|c: char| c == ':' || c == '.').to_string();
                    if blacklist.iter().any(|&b| b == cleaned.to_uppercase()) {
                        continue;
                    }
                    if cleaned.contains('/') && !cleaned.contains('.') {
                        let parts: Vec<&str> = cleaned.split('/').collect();
                        if parts.iter().any(|&p| {
                            p == "read"
                                || p == "write"
                                || p == "inspect"
                                || p == "call"
                                || p == "tool"
                        }) {
                            continue;
                        }
                    }
                    *out.friction_by_symbol.entry(cleaned).or_insert(0) += 1;
                }
                for (chan, val) in top_channels {
                    *out.channel_contributions.entry(chan.clone()).or_insert(0.0) += *val;
                }
            }
            MetricsEvent::CommandQuery { .. } => {
                out.query_commands += 1;
            }
            MetricsEvent::CommandRecall { .. } => {
                out.recall_commands += 1;
            }
            MetricsEvent::ResurfacingEmitted { .. } => {
                // Counted only on demand by future `unlost metrics resurfacings`;
                // not folded into aggregate friction stats today.
            }
        }
        all_events.push(ev);
    }

    // Pass 2: Attributing Orphan Warnings & Grouping by Session
    // We sort everything by timestamp first.
    all_events.sort_by_key(|e| match e {
        MetricsEvent::CapsuleSaved { ts_ms, .. } => *ts_ms,
        MetricsEvent::FrictionWarningInjected { ts_ms, .. } => *ts_ms,
        MetricsEvent::CommandQuery { ts_ms, .. } => *ts_ms,
        MetricsEvent::CommandRecall { ts_ms, .. } => *ts_ms,
        MetricsEvent::ResurfacingEmitted { ts_ms, .. } => *ts_ms,
    });

    let mut session_events: std::collections::HashMap<String, Vec<MetricsEvent>> =
        std::collections::HashMap::new();
    let mut last_sid: Option<String> = None;
    let mut last_sid_ts: i64 = 0;

    for ev in all_events {
        let sid = match &ev {
            MetricsEvent::CapsuleSaved {
                agent_session_id,
                ts_ms,
                ..
            } => {
                let s = agent_session_id
                    .clone()
                    .unwrap_or_else(|| "__unknown__".to_string());
                last_sid = Some(s.clone());
                last_sid_ts = *ts_ms;
                s
            }
            MetricsEvent::FrictionWarningInjected {
                agent_session_id,
                ts_ms,
                ..
            } => {
                if let Some(s) = agent_session_id {
                    last_sid = Some(s.clone());
                    last_sid_ts = *ts_ms;
                    s.clone()
                } else if let Some(ref s) = last_sid {
                    // Orphan warning: attribute to last session if close (within 30s)
                    if (*ts_ms - last_sid_ts).abs() < 30000 {
                        s.clone()
                    } else {
                        "__unknown__".to_string()
                    }
                } else {
                    "__unknown__".to_string()
                }
            }
            _ => "__unknown__".to_string(),
        };
        session_events.entry(sid).or_default().push(ev);
    }

    // Pass 3: Session-level analysis
    let mut total_spacing_tokens = 0i64;
    let mut total_spacing_segments = 0u64;
    let mut all_expensive_interventions = Vec::new();

    for events in session_events.values() {
        let mut tokens_since_last_warning = 0i64;
        let mut had_warning = false;
        let mut last_capsule_bucket = 0i64;

        for (idx, ev) in events.iter().enumerate() {
            match ev {
                MetricsEvent::CapsuleSaved {
                    tokens_input,
                    tokens_total,
                    ..
                } => {
                    let input = tokens_input.unwrap_or(0);
                    let total = tokens_total.unwrap_or(0);
                    tokens_since_last_warning += total;

                    last_capsule_bucket = (input / 4000) * 4000;
                    let b = out
                        .friction_by_input_bucket
                        .entry(last_capsule_bucket)
                        .or_default();
                    b.1 += 1;
                }
                MetricsEvent::FrictionWarningInjected {
                    ts_ms,
                    cause,
                    intensity,
                    symbols,
                    ..
                } => {
                    if had_warning {
                        total_spacing_tokens += tokens_since_last_warning;
                        total_spacing_segments += 1;
                    }
                    had_warning = true;
                    tokens_since_last_warning = 0;

                    let b = out
                        .friction_by_input_bucket
                        .entry(last_capsule_bucket)
                        .or_default();
                    b.0 += 1;

                    // Calculate cost of next 5 turns
                    let mut cost_next_5 = 0.0;
                    let mut tokens_next_5 = 0i64;
                    let mut count = 0;
                    for next_ev in events.iter().skip(idx + 1) {
                        if let MetricsEvent::CapsuleSaved {
                            cost, tokens_total, ..
                        } = next_ev
                        {
                            if let Some(c) = cost {
                                cost_next_5 += c;
                            }
                            if let Some(t) = tokens_total {
                                tokens_next_5 += t;
                            }
                            count += 1;
                            if count >= 5 {
                                break;
                            }
                        }
                    }

                    let mut cleaned_symbols = Vec::new();
                    for s in symbols {
                        let cleaned = s.trim_matches(|c: char| c == ':' || c == '.').to_string();
                        if blacklist.iter().any(|&b| b == cleaned.to_uppercase()) {
                            continue;
                        }
                        if cleaned.contains('/') && !cleaned.contains('.') {
                            let parts: Vec<&str> = cleaned.split('/').collect();
                            if parts.iter().any(|&p| {
                                p == "read"
                                    || p == "write"
                                    || p == "inspect"
                                    || p == "call"
                                    || p == "tool"
                            }) {
                                continue;
                            }
                        }
                        cleaned_symbols.push(cleaned);
                    }

                    all_expensive_interventions.push(ExpensiveIntervention {
                        ts_ms: *ts_ms,
                        cause: cause.clone(),
                        intensity: *intensity,
                        symbols: cleaned_symbols,
                        cost_next_5,
                        tokens_next_5,
                    });
                }
                _ => {}
            }
        }
    }

    all_expensive_interventions.sort_by(|a, b| b.cost_next_5.partial_cmp(&a.cost_next_5).unwrap());
    out.top_expensive_interventions = all_expensive_interventions.into_iter().take(5).collect();

    if total_spacing_segments > 0 {
        out.avg_tokens_between_interventions =
            total_spacing_tokens as f64 / total_spacing_segments as f64;
    }

    if turns_with_fluency > 0 {
        out.avg_fluency = total_fluency / turns_with_fluency as f32;
    }

    Ok(out)
}
