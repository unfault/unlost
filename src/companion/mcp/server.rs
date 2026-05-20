//! MCP server implementation for unlost.
//!
//! One server instance per invocation, workspace-bound.
//! Transport: stdio (via rmcp's `transport-io` feature).

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::schemars;
use rmcp::RoleServer;
use tracing::debug;

use crate::companion::mcp::types::*;

/// Configuration for the MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Workspace root (git toplevel).
    pub workspace_root: std::path::PathBuf,
    /// Workspace paths (LanceDB dir, etc.)
    pub ws: crate::WorkspacePaths,
    /// Whether `unlost_note` is available (opt-in).
    pub allow_writes: bool,
    /// Whether `unlost_thread` searches other workspaces.
    pub cross_workspace: bool,
    /// Embed model name (for capability advertisement).
    pub embed_model: String,
    /// Embed cache directory.
    pub embed_cache_dir: Option<std::path::PathBuf>,
}

/// The unlost MCP server.
///
/// Implements `ServerHandler` from `rmcp`. Each tool is a method called by the
/// dispatch layer; all are async and run on the tokio runtime.
#[derive(Clone)]
pub struct UnlostMcpServer {
    config: Arc<McpServerConfig>,
    /// Workspace name derived from the root directory name.
    workspace_name: String,
    workspace_path: String,
}

impl UnlostMcpServer {
    pub fn new(config: McpServerConfig) -> Self {
        let workspace_name = config
            .workspace_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("workspace")
            .to_string();
        let workspace_path = config.workspace_root.to_string_lossy().to_string();
        Self {
            config: Arc::new(config),
            workspace_name,
            workspace_path,
        }
    }

    /// Load the embedder (cold start is slow; warm is instant).
    async fn embedder(&self) -> anyhow::Result<crate::embed::Embedder> {
        crate::embed::load_embedder(
            &self.config.embed_model,
            self.config.embed_cache_dir.clone(),
            false,
        )
        .await
    }

    /// Whether write tools are enabled on this server instance.
    pub fn allow_writes(&self) -> bool {
        self.config.allow_writes
    }

    fn hit_to_mcp(&self, hit: &crate::CapsuleHit) -> McpCapsule {
        let ws_name = hit
            .origin_workspace_id
            .as_deref()
            .and_then(crate::workspace::workspace_label_by_id)
            .unwrap_or_else(|| self.workspace_name.clone());
        let ws_path = self.workspace_path.clone();
        capsule_hit_to_mcp(hit, &ws_name, &ws_path)
    }

    fn ok_json<T: serde::Serialize>(value: T) -> CallToolResult {
        let json = serde_json::to_value(&value).unwrap_or_default();
        CallToolResult::structured(json)
    }

    fn err_text(msg: impl std::fmt::Display) -> CallToolResult {
        CallToolResult::error(vec![Content::text(msg.to_string())])
    }
}

// ── Tool implementations ─────────────────────────────────────────────────────

impl UnlostMcpServer {
    async fn tool_recall(&self, input: RecallInput) -> anyhow::Result<RecallOutput> {
        let limit = input.limit.unwrap_or(8).min(25).max(1);
        let scope = input.scope.unwrap_or_default();

        let since_ms = scope
            .since
            .as_deref()
            .map(crate::util::parse_time_filter)
            .transpose()?
            .flatten();
        let until_ms = scope
            .until
            .as_deref()
            .map(crate::util::parse_time_filter)
            .transpose()?
            .flatten();

        let framed = crate::storage::frame_query_for_command(
            &input.query,
            crate::storage::QueryIntent::Recall,
        );

        let embedder = self.embedder().await?;

        let mut hits = crate::storage::query_capsules_lancedb(
            &framed,
            limit,
            scope.symbol.as_deref(),
            None,
            None,
            since_ms,
            until_ms,
            embedder,
            &self.config.ws,
        )
        .await
        .unwrap_or_default();

        // Apply session_id filter if requested
        if let Some(ref sid) = scope.session_id {
            hits.retain(|h| {
                h.meta
                    .agent_session_id
                    .as_deref()
                    .map(|s| s == sid.as_str())
                    .unwrap_or(false)
            });
        }

        // Apply file filter
        if let Some(ref file) = scope.file {
            let file_lower = file.to_lowercase();
            hits.retain(|h| {
                h.capsule
                    .symbols
                    .iter()
                    .any(|s| s.to_lowercase().contains(&file_lower))
                    || h.meta.request_path.to_lowercase().contains(&file_lower)
            });
        }

        let total_considered = hits.len();
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);

        Ok(RecallOutput {
            hits: hits.iter().map(|h| self.hit_to_mcp(h)).collect(),
            query_used: framed,
            total_considered,
        })
    }

    async fn tool_trace_decision(
        &self,
        input: TraceDecisionInput,
    ) -> anyhow::Result<TraceDecisionOutput> {
        let seeds = input.seeds.unwrap_or(5).min(12).max(1);
        let fan_out = input.fan_out.unwrap_or(8).min(16).max(1);
        let threshold = input.threshold.unwrap_or(0.65).clamp(0.0, 1.0);

        let since_ms = input
            .since
            .as_deref()
            .map(crate::util::parse_time_filter)
            .transpose()?
            .flatten();
        let until_ms = input
            .until
            .as_deref()
            .map(crate::util::parse_time_filter)
            .transpose()?
            .flatten();

        // Resolve commit range to timestamps if provided.
        let (since_ms, until_ms) = if input.from_commit.is_some() || input.to_commit.is_some() {
            let from_ts = input.from_commit.as_deref().and_then(|c| {
                crate::commands::trace::resolve_commit_timestamp(&self.config.workspace_root, c)
            });
            let to_ts = input.to_commit.as_deref().and_then(|c| {
                crate::commands::trace::resolve_commit_timestamp(&self.config.workspace_root, c)
            });
            (since_ms.or(from_ts), until_ms.or(to_ts))
        } else {
            (since_ms, until_ms)
        };

        let framed = crate::storage::frame_query_for_command(
            &input.target,
            crate::storage::QueryIntent::Trace,
        );

        let embedder = self.embedder().await?;

        let chain = crate::storage::trace_capsules_lancedb(
            &framed,
            seeds,
            fan_out,
            threshold,
            since_ms,
            until_ms,
            input.session_id.as_deref(),
            embedder,
            &self.config.ws,
        )
        .await?;

        let seed_ids: Vec<String> = chain.iter().take(seeds).map(|h| h.id.clone()).collect();
        let gaps = detect_chain_gaps(&chain);

        Ok(TraceDecisionOutput {
            chain: chain.iter().map(|h| self.hit_to_mcp(h)).collect(),
            gaps,
            seed_ids,
        })
    }

    async fn tool_challenge(&self, input: ChallengeInput) -> anyhow::Result<ChallengeOutput> {
        let limit = input.limit.unwrap_or(10).min(25).max(1);

        let framed = crate::storage::frame_query_for_command(
            &input.proposal,
            crate::storage::QueryIntent::Challenge,
        );
        let embedder = self.embedder().await?;

        let sem_hits = crate::storage::query_capsules_lancedb(
            &framed,
            40,
            None,
            None,
            None,
            None,
            None,
            embedder,
            &self.config.ws,
        )
        .await
        .unwrap_or_default();

        let broad_hits = crate::storage::scan_capsules_lancedb(
            &self.config.ws,
            200,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap_or_default();

        let all = merge_and_score_challenge(sem_hits, broad_hits, limit * 3);

        let mut prior_rationale = Vec::new();
        let mut prior_alternatives = Vec::new();
        let mut failure_modes_recorded = Vec::new();

        let proposal_lower = input.proposal.to_lowercase();
        let proposal_keywords: Vec<&str> = proposal_lower.split_whitespace().collect();

        for hit in &all {
            let cap = &hit.capsule;
            let has_failure = cap.failure_mode != crate::types::FailureMode::None;
            let has_rationale = !cap.rationale.trim().is_empty();
            let has_decision = !cap.decision.trim().is_empty();

            let text = format!("{} {}", cap.decision, cap.rationale).to_lowercase();
            let mentions_alternative = text.contains("instead")
                || text.contains("alternative")
                || text.contains("considered")
                || text.contains("rejected")
                || text.contains("vs ")
                || text.contains("versus")
                || text.contains("compared")
                || text.contains("option");

            let relevant = proposal_keywords
                .iter()
                .filter(|&&kw| kw.len() > 3)
                .any(|&kw| text.contains(kw));

            if has_failure && relevant {
                failure_modes_recorded.push(self.hit_to_mcp(hit));
            } else if mentions_alternative && relevant {
                prior_alternatives.push(self.hit_to_mcp(hit));
            } else if has_rationale && has_decision && relevant {
                prior_rationale.push(self.hit_to_mcp(hit));
            }
        }

        prior_rationale.truncate(limit);
        prior_alternatives.truncate(limit);
        failure_modes_recorded.truncate(limit);

        let silence = prior_rationale.is_empty()
            && prior_alternatives.is_empty()
            && failure_modes_recorded.is_empty();

        // Collect summary facts before moving the vecs into the output struct.
        let sessions: std::collections::HashSet<String> = prior_rationale
            .iter()
            .chain(prior_alternatives.iter())
            .chain(failure_modes_recorded.iter())
            .filter_map(|c| c.session_id.clone())
            .collect();

        let earliest = prior_rationale
            .iter()
            .chain(prior_alternatives.iter())
            .chain(failure_modes_recorded.iter())
            .map(|c| c.timestamp.clone())
            .min();
        let latest = prior_rationale
            .iter()
            .chain(prior_alternatives.iter())
            .chain(failure_modes_recorded.iter())
            .map(|c| c.timestamp.clone())
            .max();
        let distinct_sessions = sessions.len();

        Ok(ChallengeOutput {
            prior_rationale,
            prior_alternatives,
            failure_modes_recorded,
            silence,
            summary_facts: ChallengeSummaryFacts {
                earliest_mention: earliest,
                latest_mention: latest,
                distinct_sessions,
            },
        })
    }

    async fn tool_thread(&self, input: ThreadInput) -> anyhow::Result<ThreadOutput> {
        let limit = input.limit.unwrap_or(12).min(30).max(1);
        let per_ws = (limit / 2).max(5);

        let since_ms = input
            .since
            .as_deref()
            .map(crate::util::parse_time_filter)
            .transpose()?
            .flatten();

        let framed = crate::storage::frame_query_for_command(
            &input.topic,
            crate::storage::QueryIntent::Trace,
        );
        let embedder = self.embedder().await?;

        let mut hits = crate::storage::query_capsules_cross_workspace(
            &framed,
            embedder,
            &self.config.ws,
            per_ws,
            limit,
        )
        .await;

        if let Some(since_ms) = since_ms {
            hits.retain(|h| h.ts_ms >= since_ms);
        }

        if let Some(ref ws_filter) = input.workspaces {
            if !ws_filter.is_empty() {
                let ws_name_self = self.workspace_name.clone();
                hits.retain(|h| {
                    let ws_name = h
                        .origin_workspace_id
                        .as_deref()
                        .and_then(crate::workspace::workspace_label_by_id)
                        .unwrap_or_else(|| ws_name_self.clone());
                    ws_filter.contains(&ws_name)
                });
            }
        }

        hits.truncate(limit);

        let mut breakdown: HashMap<String, usize> = HashMap::new();
        let mut all_ws_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let ws_name_self = self.workspace_name.clone();
        for h in &hits {
            let ws_name = h
                .origin_workspace_id
                .as_deref()
                .and_then(crate::workspace::workspace_label_by_id)
                .unwrap_or_else(|| ws_name_self.clone());
            *breakdown.entry(ws_name).or_insert(0) += 1;
            if let Some(ref wid) = h.origin_workspace_id {
                all_ws_ids.insert(wid.clone());
            }
        }
        all_ws_ids.insert(self.config.ws.id.clone());
        let workspaces_searched = all_ws_ids.len();

        let mcp_hits: Vec<McpCapsule> = hits.iter().map(|h| self.hit_to_mcp(h)).collect();

        let timestamps: Vec<i64> = hits.iter().map(|h| h.ts_ms).collect();
        let earliest = timestamps.iter().copied().min().map(ts_to_rfc3339);
        let latest = timestamps.iter().copied().max().map(ts_to_rfc3339);

        Ok(ThreadOutput {
            hits: mcp_hits,
            workspace_breakdown: breakdown,
            span: ThreadSpan { earliest, latest },
            workspaces_searched,
        })
    }

    async fn tool_orient(&self, input: OrientInput) -> anyhow::Result<OrientOutput> {
        let mut query_parts: Vec<String> = Vec::new();
        if let Some(ref files) = input.files {
            query_parts.extend(files.iter().cloned());
        }
        if let Some(ref symbols) = input.symbols {
            query_parts.extend(symbols.iter().cloned());
        }
        if let Some(ref intent) = input.intent {
            query_parts.push(intent.clone());
        }

        let mut recent_touches: Vec<McpCapsule> = Vec::new();
        let drift_signal = read_persisted_drift_signal(&self.config.ws).await;

        if !query_parts.is_empty() {
            let query = query_parts.join(" ");
            let embedder = self.embedder().await?;
            let mut hits = crate::storage::query_capsules_lancedb(
                &query,
                12,
                None,
                None,
                None,
                None,
                None,
                embedder,
                &self.config.ws,
            )
            .await
            .unwrap_or_default();

            let file_syms: Vec<String> = input
                .files
                .iter()
                .flatten()
                .chain(input.symbols.iter().flatten())
                .map(|s| s.to_lowercase())
                .collect();

            if !file_syms.is_empty() {
                hits.retain(|h| {
                    let combined: Vec<String> = h
                        .capsule
                        .symbols
                        .iter()
                        .chain(std::iter::once(&h.meta.request_path))
                        .map(|s| s.to_lowercase())
                        .collect();
                    file_syms
                        .iter()
                        .any(|fs| combined.iter().any(|c| c.contains(fs.as_str())))
                });
            }

            hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
            hits.truncate(8);
            recent_touches = hits.iter().map(|h| self.hit_to_mcp(h)).collect();
        }

        let suggestions = build_orient_suggestions(&recent_touches);

        Ok(OrientOutput {
            recent_touches,
            drift_signal,
            suggestions,
        })
    }

    async fn tool_note(&self, input: NoteInput) -> anyhow::Result<NoteOutput> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut text_parts = vec![input.text.clone()];
        if let Some(ref d) = input.decision {
            if !d.trim().is_empty() {
                text_parts.push(format!("Decision: {d}"));
            }
        }
        if let Some(ref r) = input.rationale {
            if !r.trim().is_empty() {
                text_parts.push(format!("Rationale: {r}"));
            }
        }
        let full_text = text_parts.join("\n");

        let source_label = input.source.as_deref().unwrap_or("mcp");

        let mut symbols: Vec<String> = crate::net::extract_symbols_from_text(&full_text);
        if let Some(ref extra_syms) = input.symbols {
            for s in extra_syms {
                if !symbols.contains(s) {
                    symbols.push(s.clone());
                }
            }
        }
        if let Some(ref files) = input.files {
            for f in files {
                if !symbols.contains(f) {
                    symbols.push(f.clone());
                }
            }
        }

        let capsule = crate::IntentCapsule {
            category: format!("Note:{source_label}"),
            intent: "Agent note".to_string(),
            decision: input.decision.clone().unwrap_or_else(|| input.text.clone()),
            rationale: input.rationale.clone().unwrap_or_default(),
            next_steps: Vec::new(),
            symbols,
            user_symbols: Vec::new(),
            failure_mode: crate::types::FailureMode::None,
            failure_signals: None,
            extraction_mode: crate::types::ExtractionMode::None,
            questions: Vec::new(),
        };

        let pointer = self
            .config
            .ws
            .root
            .to_str()
            .map(|p| format!("note+local://{p}#{ts_ms}"));

        let meta = crate::ResponseMeta {
            source: "note".to_string(),
            upstream_host: "mcp".to_string(),
            request_path: source_label.to_string(),
            http_status: 0,
            agent_session_id: None,
            source_pointer: pointer,
            usage: None,
        };

        let embedder = self.embedder().await?;
        std::fs::create_dir_all(&self.config.ws.db_dir)?;
        let db = lancedb::connect(self.config.ws.db_dir.to_string_lossy().as_ref())
            .execute()
            .await?;
        let _ = crate::storage::ensure_capsules_table(&db).await?;

        let id = crate::storage::insert_capsule_row(
            &db,
            &embedder,
            0,
            0,
            ts_ms,
            &meta,
            None,
            None,
            &capsule,
            &crate::types::TurnEval::default(),
            Some(&full_text),
            None,
            None,
        )
        .await?;

        let _ = crate::recording::append_capsule_jsonl(
            &self.config.ws.capsules_jsonl,
            ts_ms,
            0,
            0,
            &meta,
            &capsule,
            None,
            None,
        );

        Ok(NoteOutput {
            id,
            persisted_at: ts_to_rfc3339(ts_ms),
        })
    }

    async fn tool_capsule_get(
        &self,
        input: CapsuleGetInput,
    ) -> anyhow::Result<CapsuleGetOutput> {
        let db = match lancedb::connect(self.config.ws.db_dir.to_string_lossy().as_ref())
            .execute()
            .await
        {
            Ok(db) => db,
            Err(_) => {
                return Ok(CapsuleGetOutput {
                    capsule: None,
                    raw_text: None,
                    reference: None,
                });
            }
        };

        match fetch_capsule_by_id(&db, &input.id, &self.config.ws).await {
            Some((hit, raw_text, reference)) => Ok(CapsuleGetOutput {
                capsule: {
                    let mut mcp = self.hit_to_mcp(&hit);
                    mcp.score = None;
                    Some(mcp)
                },
                raw_text,
                reference,
            }),
            None => Ok(CapsuleGetOutput {
                capsule: None,
                raw_text: None,
                reference: None,
            }),
        }
    }
}

// ── rmcp ServerHandler impl ──────────────────────────────────────────────────

impl ServerHandler for UnlostMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "unlost is a local-first code memory store. \
                Use unlost_recall before editing non-trivial code to check for prior decisions. \
                Use unlost_trace_decision to understand the causal chain behind current state. \
                Use unlost_challenge before reversing a past decision. \
                Use unlost_thread when a topic feels like it came up before, across projects. \
                Use unlost_orient to check for drift or prior attempts on the current files. \
                Use unlost_note to record a decision that wouldn't survive in the exchange transcript. \
                Use unlost_capsule_get to fetch the full text of a specific capsule by id.",
            )
            .with_server_info(rmcp::model::Implementation::new(
                "unlost",
                env!("CARGO_PKG_VERSION"),
            ))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::model::ErrorData> {
        let mut tools = vec![
            make_tool::<RecallInput>(
                "unlost_recall",
                "Search workspace memory for past decisions, context, or rationale. \
                Call this before editing non-trivial code to check if a relevant decision exists. \
                Returns structured capsule evidence — no LLM narration.",
            ),
            make_tool::<TraceDecisionInput>(
                "unlost_trace_decision",
                "Reconstruct the causal chain of decisions that led to the current state of a \
                file, symbol, or concept. Returns capsules in chronological order with detected \
                gaps. Use when you need to understand WHY code is the way it is.",
            ),
            make_tool::<ChallengeInput>(
                "unlost_challenge",
                "Pressure-test a proposal against workspace memory. Returns prior rationale, \
                alternatives that were considered, and recorded failure modes. \
                The 'silence' flag is true when memory has no material signal either way.",
            ),
            make_tool::<ThreadInput>(
                "unlost_thread",
                "Search for a topic ACROSS ALL workspaces. Returns where this idea has appeared \
                before, in any project. Use when a topic feels familiar but you don't have the \
                context from this session.",
            ),
            make_tool::<OrientInput>(
                "unlost_orient",
                "Check for prior touches on the files/symbols you're about to work on, \
                and read the last persisted drift signal. Use at the start of a non-trivial \
                subtask to surface prior attempts, rejected alternatives, and open questions.",
            ),
            make_tool::<CapsuleGetInput>(
                "unlost_capsule_get",
                "Fetch the full text and metadata of a specific capsule by id. \
                Use to cite or quote a specific past decision returned by another tool.",
            ),
        ];

        if self.config.allow_writes {
            tools.push(make_tool::<NoteInput>(
                "unlost_note",
                "Record a deliberate decision into workspace memory. \
                Use for decisions made in sub-agent fan-outs or tool-only turns that \
                won't be captured by the auto-extractor. Requires allow_writes=true.",
            ));
        }

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        debug!(tool = %request.name, "MCP tool call");

        let args = serde_json::Value::Object(request.arguments.clone().unwrap_or_default());

        macro_rules! dispatch {
            ($name:expr, $input_type:ty, $method:ident) => {
                if request.name == $name {
                    let input: $input_type = serde_json::from_value(args.clone())
                        .map_err(|e| rmcp::model::ErrorData::invalid_params(e.to_string(), None))?;
                    return match self.$method(input).await {
                        Ok(output) => Ok(Self::ok_json(output)),
                        Err(e) => Ok(Self::err_text(e)),
                    };
                }
            };
        }

        dispatch!("unlost_recall", RecallInput, tool_recall);
        dispatch!("unlost_trace_decision", TraceDecisionInput, tool_trace_decision);
        dispatch!("unlost_challenge", ChallengeInput, tool_challenge);
        dispatch!("unlost_thread", ThreadInput, tool_thread);
        dispatch!("unlost_orient", OrientInput, tool_orient);
        dispatch!("unlost_capsule_get", CapsuleGetInput, tool_capsule_get);

        if request.name == "unlost_note" {
            if !self.config.allow_writes {
                return Ok(Self::err_text(
                    "unlost_note is disabled. Start the MCP server with --allow-writes to enable.",
                ));
            }
            let input: NoteInput = serde_json::from_value(args)
                .map_err(|e| rmcp::model::ErrorData::invalid_params(e.to_string(), None))?;
            return match self.tool_note(input).await {
                Ok(output) => Ok(Self::ok_json(output)),
                Err(e) => Ok(Self::err_text(e)),
            };
        }

        Err(rmcp::model::ErrorData::method_not_found::<
            rmcp::model::CallToolRequestMethod,
        >())
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

/// Build a `Tool` definition with JSON Schema derived from the input type.
fn make_tool<T: schemars::JsonSchema>(name: impl Into<String>, description: impl Into<String>) -> Tool {
    let schema = schemars::schema_for!(T);
    let schema_value = serde_json::to_value(&schema).unwrap_or_default();
    let schema_obj: serde_json::Map<String, serde_json::Value> = schema_value
        .as_object()
        .cloned()
        .unwrap_or_default();
    Tool::new(name.into(), description.into(), schema_obj)
}

fn ts_to_rfc3339(ts_ms: i64) -> String {
    use chrono::{SecondsFormat, TimeZone};
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap_or_default()
}

/// Detect notable gaps between consecutive capsules in a causal chain.
fn detect_chain_gaps(chain: &[crate::CapsuleHit]) -> Vec<ChainGap> {
    const SESSION_GAP_MS: i64 = 30 * 60 * 1000; // 30 minutes
    const TIME_JUMP_MS: i64 = 24 * 60 * 60 * 1000; // 1 day

    let mut gaps = Vec::new();
    for window in chain.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        let delta = b.ts_ms - a.ts_ms;

        let reason = if delta >= TIME_JUMP_MS {
            "time_jump"
        } else if a.meta.agent_session_id != b.meta.agent_session_id && delta >= SESSION_GAP_MS {
            "session_change"
        } else if a.distance > 0.5 || b.distance > 0.5 {
            "low_continuity"
        } else {
            continue;
        };

        gaps.push(ChainGap {
            between: [a.id.clone(), b.id.clone()],
            reason: reason.to_string(),
        });
    }
    gaps
}

/// Read the last persisted drift signal from TurnEval columns in LanceDB.
async fn read_persisted_drift_signal(ws: &crate::WorkspacePaths) -> Option<DriftSignal> {
    let recent =
        crate::storage::scan_capsules_lancedb_recent(ws, 5, None, None, None, None, None)
            .await
            .ok()?;

    let hit = recent.iter().find(|h| h.turn_eval.is_some())?;
    let te = hit.turn_eval.as_ref()?;

    let state = match te.trajectory_state {
        crate::types::TrajectoryState::Stable => "stable",
        crate::types::TrajectoryState::Watch => "watch",
        crate::types::TrajectoryState::Intervene => "intervene",
    }
    .to_string();

    let channels = &[
        (
            "loop",
            te.repetition + te.novelty_collapse + te.semantic_stall,
        ),
        ("spec", te.alignment_debt + te.instruction_staticness),
        (
            "drift",
            te.path_hallucination + te.grounding_stall + te.logic_churn,
        ),
    ];
    let basin = if te.trajectory_intensity > 0.3 {
        channels
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|(_, score)| *score > 0.2)
            .map(|(name, _)| name.to_string())
    } else {
        None
    };

    Some(DriftSignal {
        state,
        basin,
        intensity: te.trajectory_intensity,
    })
}

/// Generate orient suggestions from recent touches.
fn build_orient_suggestions(touches: &[McpCapsule]) -> Vec<OrientSuggestion> {
    let mut suggestions = Vec::new();

    for cap in touches.iter().take(5) {
        if let Some(ref fm) = cap.failure_mode {
            suggestions.push(OrientSuggestion {
                kind: "related_failure".to_string(),
                why: format!("{fm} recorded on this area — review before proceeding"),
                ref_capsule_ids: vec![cap.id.clone()],
            });
        }

        let text = format!(
            "{} {}",
            cap.decision.as_deref().unwrap_or(""),
            cap.rationale.as_deref().unwrap_or("")
        )
        .to_lowercase();

        if text.contains("rejected")
            || text.contains("decided against")
            || text.contains("instead of")
        {
            suggestions.push(OrientSuggestion {
                kind: "rejected_alternative".to_string(),
                why: "a prior session recorded a rejected alternative in this area".to_string(),
                ref_capsule_ids: vec![cap.id.clone()],
            });
        }

        if text.contains("open")
            || text.contains("todo")
            || text.contains("revisit")
            || text.contains("later")
        {
            suggestions.push(OrientSuggestion {
                kind: "open_question".to_string(),
                why: "an open item was recorded in this area".to_string(),
                ref_capsule_ids: vec![cap.id.clone()],
            });
        }
    }

    // Prior attempts: same symbol in >1 capsule
    let mut symbol_count: HashMap<String, Vec<String>> = HashMap::new();
    for cap in touches {
        for sym in &cap.symbols {
            symbol_count.entry(sym.clone()).or_default().push(cap.id.clone());
        }
    }
    for (sym, ids) in &symbol_count {
        if ids.len() >= 2 {
            suggestions.push(OrientSuggestion {
                kind: "prior_attempt".to_string(),
                why: format!(
                    "`{sym}` has been touched in {count} prior capsules",
                    count = ids.len()
                ),
                ref_capsule_ids: ids.clone(),
            });
            break;
        }
    }

    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    suggestions.retain(|s| {
        let key = s.ref_capsule_ids.join(",");
        seen_ids.insert(key)
    });
    suggestions.truncate(5);
    suggestions
}

/// Score and merge challenge hits.
fn merge_and_score_challenge(
    sem_hits: Vec<crate::CapsuleHit>,
    broad_hits: Vec<crate::CapsuleHit>,
    limit: usize,
) -> Vec<crate::CapsuleHit> {
    use std::collections::{HashMap, HashSet};

    let mut by_id: HashMap<String, crate::CapsuleHit> = HashMap::new();
    for h in sem_hits.into_iter().chain(broad_hits) {
        match by_id.get(&h.id) {
            Some(existing) if existing.ts_ms >= h.ts_ms => {}
            _ => {
                by_id.insert(h.id.clone(), h);
            }
        }
    }
    let hits: Vec<crate::CapsuleHit> = by_id.into_values().collect();
    if hits.is_empty() {
        return hits;
    }

    let mut symbol_sessions: HashMap<String, HashSet<String>> = HashMap::new();
    for h in &hits {
        let session_key = h
            .meta
            .agent_session_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("ses:{s}"))
            .unwrap_or_else(|| format!("conn:{}", h.conn_id));
        for sym in &h.capsule.symbols {
            symbol_sessions
                .entry(sym.clone())
                .or_default()
                .insert(session_key.clone());
        }
    }
    let symbol_session_counts: HashMap<String, usize> = symbol_sessions
        .into_iter()
        .map(|(sym, sessions)| (sym, sessions.len()))
        .collect();

    let mut scored: Vec<(f32, crate::CapsuleHit)> = hits
        .into_iter()
        .map(|h| {
            let mut score = 0.0f32;
            if !h.capsule.decision.trim().is_empty() {
                score += 2.0;
            }
            if !h.capsule.rationale.trim().is_empty() {
                score += 2.0;
            }
            if h.capsule.failure_mode != crate::types::FailureMode::None {
                score += 3.0;
            }
            if matches!(
                h.capsule.failure_mode,
                crate::types::FailureMode::DecisionConflict
                    | crate::types::FailureMode::RetrySpiral
            ) {
                score += 2.0;
            }
            for sym in &h.capsule.symbols {
                if symbol_session_counts.get(sym).copied().unwrap_or(0) >= 2 {
                    score += 1.0;
                }
            }
            (score, h)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(limit).map(|(_, h)| h).collect()
}

/// Fetch a single capsule by id from LanceDB.
async fn fetch_capsule_by_id(
    db: &lancedb::connection::Connection,
    id: &str,
    ws: &crate::WorkspacePaths,
) -> Option<(crate::CapsuleHit, Option<String>, Option<String>)> {
    use futures_util::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let table = db
        .open_table(crate::storage::CAPSULES_TABLE)
        .execute()
        .await
        .ok()?;

    let safe_id = id.replace('\'', "''");
    let filter = format!("id = '{safe_id}'");
    let stream = table
        .query()
        .only_if(filter)
        .limit(1)
        .execute()
        .await
        .ok()?;

    let batches: Vec<arrow_array::RecordBatch> = stream
        .try_collect::<Vec<_>>()
        .await
        .unwrap_or_default();

    let hits = crate::storage::record_batches_to_hits(&batches, &ws.id).ok()?;
    let hit = hits.into_iter().next()?;

    let reference = match hit.meta.source.as_str() {
        "git" => hit
            .commit_sha
            .as_deref()
            .map(|sha| format!("commit:{sha}")),
        "changelog" => Some(format!("version:{}", hit.meta.request_path)),
        _ => hit
            .meta
            .agent_session_id
            .as_deref()
            .map(|sid| format!("session:{sid}")),
    };

    Some((hit, None, reference))
}
