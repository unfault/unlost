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
) -> anyhow::Result<()> {
    let ws_dir = crate::unlost_workspace_dir(workspace_id);
    let path = ws_dir.join("metrics.jsonl");
    let ev = MetricsEvent::FrictionWarningInjected {
        ts_ms: crate::now_ms(),
        conn_id,
        workspace_id: workspace_id.to_string(),
        agent_session_id,
        symbols,
        user_emotion: user_emotion.map(|e| e.label.clone()),
        intensity,
        cause,
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
}

pub(crate) fn summarize_metrics(path: &std::path::Path) -> anyhow::Result<MetricsSummary> {
    use std::io::BufRead;

    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(MetricsSummary::default()),
    };
    let mut out = MetricsSummary::default();
    let reader = std::io::BufReader::new(f);

    let mut session_events: std::collections::HashMap<String, Vec<MetricsEvent>> =
        std::collections::HashMap::new();

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<MetricsEvent>(&line) else {
            continue;
        };

        // Pass 1: Global totals and session grouping
        match &ev {
            MetricsEvent::CapsuleSaved {
                tokens_total,
                cost,
                paths_checked,
                paths_missing,
                failure_mode,
                agent_session_id,
                ..
            } => {
                out.capsules += 1;
                if let Some(t) = tokens_total {
                    out.tokens_total = out.tokens_total.saturating_add(*t);
                }
                if let Some(c) = cost {
                    out.cost_total += c;
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

                if let Some(sid) = agent_session_id {
                    session_events
                        .entry(sid.clone())
                        .or_default()
                        .push(ev.clone());
                }
            }
            MetricsEvent::FrictionWarningInjected {
                intensity,
                cause,
                symbols,
                agent_session_id,
                ..
            } => {
                out.friction_warnings += 1;
                out.friction_intensity_total += *intensity;
                *out.friction_by_cause.entry(cause.clone()).or_insert(0) += 1;
                for s in symbols {
                    *out.friction_by_symbol.entry(s.clone()).or_insert(0) += 1;
                }
                if let Some(sid) = agent_session_id {
                    session_events
                        .entry(sid.clone())
                        .or_default()
                        .push(ev.clone());
                }
            }
            MetricsEvent::CommandQuery { .. } => {
                out.query_commands += 1;
            }
            MetricsEvent::CommandRecall { .. } => {
                out.recall_commands += 1;
            }
        }
    }

    // Pass 2: Session-level spacing and bucket analysis
    let mut total_spacing_tokens = 0i64;
    let mut total_spacing_segments = 0u64;

    for events in session_events.values_mut() {
        events.sort_by_key(|e| match e {
            MetricsEvent::CapsuleSaved { ts_ms, .. } => *ts_ms,
            MetricsEvent::FrictionWarningInjected { ts_ms, .. } => *ts_ms,
            MetricsEvent::CommandQuery { ts_ms, .. } => *ts_ms,
            MetricsEvent::CommandRecall { ts_ms, .. } => *ts_ms,
        });

        let mut tokens_since_last_warning = 0i64;
        let mut had_warning = false;
        let mut last_capsule_bucket = 0i64;

        for ev in events {
            match ev {
                MetricsEvent::CapsuleSaved {
                    tokens_input,
                    tokens_total,
                    ..
                } => {
                    let input = tokens_input.unwrap_or(0);
                    let total = tokens_total.unwrap_or(0);
                    tokens_since_last_warning += total;

                    // Bucket analysis (using tokens_input as the real context proxy)
                    last_capsule_bucket = (input / 4000) * 4000;
                    let b = out
                        .friction_by_input_bucket
                        .entry(last_capsule_bucket)
                        .or_default();
                    b.1 += 1; // Increment turn count N
                }
                MetricsEvent::FrictionWarningInjected { .. } => {
                    if had_warning {
                        total_spacing_tokens += tokens_since_last_warning;
                        total_spacing_segments += 1;
                    }
                    had_warning = true;
                    tokens_since_last_warning = 0;

                    // Mark warning in the current bucket
                    let b = out
                        .friction_by_input_bucket
                        .entry(last_capsule_bucket)
                        .or_default();
                    b.0 += 1;
                }
                _ => {}
            }
        }
    }

    if total_spacing_segments > 0 {
        out.avg_tokens_between_interventions =
            total_spacing_tokens as f64 / total_spacing_segments as f64;
    }

    Ok(out)
}
