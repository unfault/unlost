//! MCP-facing data types for unlost tools.
//!
//! These are the structured input/output schemas exposed over MCP.
//! Kept separate from internal domain types so the MCP contract can evolve
//! independently without touching storage or governor internals.

use chrono::{SecondsFormat, TimeZone};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Shared output type ───────────────────────────────────────────────────────

/// A single memory capsule returned by any query tool.
///
/// `score` is present (0.0–1.0, lower = closer) on retrieval results and
/// absent (`null`) when fetched directly by id.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct McpCapsule {
    /// Stable opaque id. Use with `unlost_capsule_get` to fetch full details.
    pub id: String,
    /// RFC3339 timestamp (UTC).
    pub timestamp: String,
    /// Human-readable workspace name (git repo dir name).
    pub workspace_name: String,
    /// Absolute path to the workspace root.
    pub workspace_path: String,
    /// Agent session id, if available.
    pub session_id: Option<String>,
    /// Git commit SHA associated with this turn, if available.
    pub commit_sha: Option<String>,
    /// Capsule kind: agent_turn | git_commit | note | checkpoint | version
    pub kind: String,
    /// Short description of the user's intent.
    pub intent: Option<String>,
    /// The decision made or action taken.
    pub decision: Option<String>,
    /// Why this decision was made.
    pub rationale: Option<String>,
    /// Code symbols referenced (functions, structs, files, etc.)
    pub symbols: Vec<String>,
    /// Files referenced or modified.
    pub files: Vec<String>,
    /// Recorded failure mode, if any.
    pub failure_mode: Option<String>,
    /// Category label assigned during extraction.
    pub category: Option<String>,
    /// Retrieval score (distance, lower = more relevant). Null for direct fetches.
    pub score: Option<f32>,
}

/// Convert an internal `CapsuleHit` to an `McpCapsule`.
///
/// `workspace_name` must be supplied by the caller (derived from the workspace root).
pub fn capsule_hit_to_mcp(
    hit: &crate::CapsuleHit,
    workspace_name: &str,
    workspace_path: &str,
) -> McpCapsule {
    let ts = chrono::Utc
        .timestamp_millis_opt(hit.ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap_or_default();

    // Determine kind from source field.
    let kind = match hit.meta.source.as_str() {
        "git" => "git_commit",
        "changelog" => "version",
        "note" => "note",
        "checkpoint" => "checkpoint",
        _ => "agent_turn",
    }
    .to_string();

    let failure_mode = match hit.capsule.failure_mode {
        crate::types::FailureMode::None => None,
        crate::types::FailureMode::Drift => Some("drift".to_string()),
        crate::types::FailureMode::Rediscovery => Some("rediscovery".to_string()),
        crate::types::FailureMode::DecisionConflict => Some("decision_conflict".to_string()),
        crate::types::FailureMode::RetrySpiral => Some("retry_spiral".to_string()),
        crate::types::FailureMode::FalseProgress => Some("false_progress".to_string()),
        crate::types::FailureMode::UnboundedHorizon => Some("unbounded_horizon".to_string()),
    };

    // Separate file paths from symbol names in the symbols list.
    let (files, symbols): (Vec<String>, Vec<String>) = hit
        .capsule
        .symbols
        .iter()
        .cloned()
        .partition(|s| s.contains('/') || s.contains('.') && !s.starts_with('.'));

    McpCapsule {
        id: hit.id.clone(),
        timestamp: ts,
        workspace_name: workspace_name.to_string(),
        workspace_path: workspace_path.to_string(),
        session_id: hit.meta.agent_session_id.clone(),
        commit_sha: hit.commit_sha.clone(),
        kind,
        intent: nonempty(hit.capsule.intent.trim()),
        decision: nonempty(hit.capsule.decision.trim()),
        rationale: nonempty(hit.capsule.rationale.trim()),
        symbols,
        files,
        failure_mode,
        category: nonempty(&hit.capsule.category),
        score: Some(hit.distance),
    }
}

fn nonempty(s: &str) -> Option<String> {
    if s.is_empty() { None } else { Some(s.to_string()) }
}

// ── Tool input types ─────────────────────────────────────────────────────────

/// Optional scope filters for workspace-scoped queries.
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct QueryScope {
    /// Filter results to capsules touching this file path.
    pub file: Option<String>,
    /// Filter results to capsules mentioning this symbol.
    pub symbol: Option<String>,
    /// Only include capsules after this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y).
    pub since: Option<String>,
    /// Only include capsules before this time.
    pub until: Option<String>,
    /// Restrict to capsules from a specific agent session.
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// What to search for: a question, file path, symbol name, or concept.
    pub query: String,
    /// Optional scope filters.
    pub scope: Option<QueryScope>,
    /// Max results to return. Default 8, max 25.
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TraceDecisionInput {
    /// File path, symbol name, or free-text question.
    /// e.g. "src/governor.rs", "why is the timeout 30s?", "TrajectoryController"
    pub target: String,
    /// Max seed capsules from the initial semantic search. Default 5.
    pub seeds: Option<usize>,
    /// Max capsules per symbol fan-out. Default 8.
    pub fan_out: Option<usize>,
    /// Similarity distance threshold (0.0–1.0). Default 0.65.
    pub threshold: Option<f32>,
    /// Only include capsules after this time.
    pub since: Option<String>,
    /// Only include capsules before this time.
    pub until: Option<String>,
    /// Restrict to a specific agent session.
    pub session_id: Option<String>,
    /// Lower commit bound (git ref, e.g. "main").
    pub from_commit: Option<String>,
    /// Upper commit bound (git ref, e.g. "HEAD").
    pub to_commit: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChallengeInput {
    /// The proposal, decision, or technology to pressure-test.
    /// e.g. "should we move from lancedb to sqlite+fts?"
    pub proposal: String,
    /// Max evidence capsules to retrieve. Default 10, max 25.
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ThreadInput {
    /// The topic to trace across all workspaces.
    pub topic: String,
    /// Max results to return. Default 12, max 30.
    pub limit: Option<usize>,
    /// Only include capsules after this time.
    pub since: Option<String>,
    /// Restrict search to specific workspace names (default: all).
    pub workspaces: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct OrientInput {
    /// Files the agent is about to work on.
    pub files: Option<Vec<String>>,
    /// Symbols the agent is about to touch.
    pub symbols: Option<Vec<String>>,
    /// Optional description of what the agent is about to do.
    pub intent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NoteInput {
    /// The note text. Required.
    pub text: String,
    /// The specific decision that was made (optional, structured).
    pub decision: Option<String>,
    /// Why this decision was made.
    pub rationale: Option<String>,
    /// Code symbols this note relates to.
    pub symbols: Option<Vec<String>>,
    /// Files this note relates to.
    pub files: Option<Vec<String>>,
    /// Optional source label (e.g. "agent_subtask", "manual", "planning").
    pub source: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CapsuleGetInput {
    /// The capsule id to fetch. Returned by any other tool.
    pub id: String,
}

// ── Tool output types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecallOutput {
    pub hits: Vec<McpCapsule>,
    /// The query as interpreted (post-HyPE framing if applied).
    pub query_used: String,
    /// Total capsules considered before ranking cutoff.
    pub total_considered: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChainGap {
    /// Ids of the two capsules between which the gap exists.
    pub between: [String; 2],
    /// Reason for the gap: time_jump | session_change | low_continuity
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TraceDecisionOutput {
    /// The causal chain, chronological (oldest first).
    pub chain: Vec<McpCapsule>,
    /// Notable gaps in the chain.
    pub gaps: Vec<ChainGap>,
    /// Which capsule ids seeded the trace.
    pub seed_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChallengeSummaryFacts {
    pub earliest_mention: Option<String>,
    pub latest_mention: Option<String>,
    pub distinct_sessions: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChallengeOutput {
    /// Capsules explaining why the current state was chosen.
    pub prior_rationale: Vec<McpCapsule>,
    /// Capsules where alternatives were weighed.
    pub prior_alternatives: Vec<McpCapsule>,
    /// Capsules tagged with relevant failure modes.
    pub failure_modes_recorded: Vec<McpCapsule>,
    /// True if memory has no material signal about this proposal.
    pub silence: bool,
    pub summary_facts: ChallengeSummaryFacts,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ThreadOutput {
    pub hits: Vec<McpCapsule>,
    /// How many hits came from each workspace name.
    pub workspace_breakdown: std::collections::HashMap<String, usize>,
    pub span: ThreadSpan,
    /// Total number of workspaces searched.
    pub workspaces_searched: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ThreadSpan {
    pub earliest: Option<String>,
    pub latest: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DriftSignal {
    /// Current trajectory state: stable | watch | intervene
    pub state: String,
    /// Which basin: loop | spec | drift | null
    pub basin: Option<String>,
    /// Composite intensity 0.0–1.0
    pub intensity: f32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct OrientSuggestion {
    /// prior_attempt | rejected_alternative | open_question | related_failure
    pub kind: String,
    /// Short structured phrase explaining why this is relevant.
    pub why: String,
    /// Capsule ids backing this suggestion.
    pub ref_capsule_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct OrientOutput {
    /// Recent capsules touching the given files/symbols, newest first.
    pub recent_touches: Vec<McpCapsule>,
    /// Drift signal from last persisted governor state. Null if no sessions recorded.
    pub drift_signal: Option<DriftSignal>,
    /// Suggested areas to investigate before proceeding.
    pub suggestions: Vec<OrientSuggestion>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NoteOutput {
    pub id: String,
    pub persisted_at: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CapsuleGetOutput {
    /// The capsule, or null if not found.
    pub capsule: Option<McpCapsule>,
    /// Raw exchange text snippet, if stored.
    pub raw_text: Option<String>,
    /// Citation pointer (e.g. "session:xxx", "commit:sha", "version:v0.14").
    pub reference: Option<String>,
}

// ── Server capability advertisement ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ServerMeta {
    pub version: String,
    pub workspace_name: String,
    pub workspace_path: String,
    pub writes_enabled: bool,
    pub cross_workspace_enabled: bool,
    pub embed_model: String,
    pub capsule_count: usize,
}
