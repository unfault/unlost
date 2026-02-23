use anyhow::Result;
use indicatif::ProgressDrawTarget;
use std::io::IsTerminal;

use chrono::{SecondsFormat, TimeZone};

use crate::cli::OutputFormat;

fn looks_like_semver(s: &str) -> bool {
    // Cheap check: `0.7.0` (optionally with a leading `v`).
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let s = s.strip_prefix('v').unwrap_or(s);
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() < 2 || parts.len() > 4 {
        return false;
    }
    parts
        .iter()
        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

fn capsule_ref_token(meta: &crate::types::ResponseMeta) -> Option<String> {
    let src = meta.source.trim();
    let rp = meta.request_path.trim();
    if rp.is_empty() {
        return None;
    }
    if src == "git" {
        return Some(format!("commit:{rp}"));
    }
    if src == "changelog" {
        if looks_like_semver(rp) {
            let v = rp.strip_prefix('v').unwrap_or(rp);
            return Some(format!("version:v{v}"));
        }
        return Some(format!("version:{rp}"));
    }
    None
}

pub(crate) async fn llm_query_narrative(
    llm_model_override: Option<&str>,
    query_text: &str,
    symbol: Option<&str>,
    workspace_root: &str,
    matches: &[crate::CapsuleHit],
) -> Result<String> {
    let fmt_ts_utc = |ts_ms: i64| -> Option<String> {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
    };

    // Build graph once for relationship grounding. Failure is non-fatal.
    let cg = if matches.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        match crate::workspace::build_graph_for_workspace(root) {
            Some(g) => Some(g),
            None => {
                tracing::warn!(
                    "llm_query_narrative: failed to build code graph for {workspace_root}, proceeding without relationship grounding"
                );
                None
            }
        }
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Query:\n");
    context.push_str(query_text);
    context.push('\n');
    if let Some(sym) = symbol {
        context.push_str("Symbol filter: ");
        context.push_str(sym);
        context.push('\n');
    }

    // Session IDs can be long; we bucket them into short tags so the LLM can avoid
    // accidentally printing raw identifiers.
    let mut session_tags: std::collections::HashMap<&str, String> =
        std::collections::HashMap::new();
    let mut sessions: Vec<&str> = matches
        .iter()
        .filter_map(|h| h.meta.agent_session_id.as_deref())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sessions.sort_unstable();
    for (i, s) in sessions.iter().enumerate() {
        session_tags.insert(*s, format!("s{}", i + 1));
    }
    if sessions.len() > 1 {
        context.push_str(&format!(
            "Distinct agent sessions in matches: {}\n",
            sessions.len()
        ));
    }
    context.push_str("Matches (lower distance = closer):\n");
    for (i, hit) in matches.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ts = fmt_ts_utc(hit.ts_ms)
            .map(|s| format!(" time_utc={s}"))
            .unwrap_or_default();
        let session_tag = meta
            .agent_session_id
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .and_then(|s| session_tags.get(s))
            .cloned();
        let session_tag = session_tag
            .map(|t| format!(" session={t}"))
            .unwrap_or_default();
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} distance={} source={} category={} upstream={} path={}{}{}{}\n",
            i + 1,
            hit.distance,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path,
            ref_tok,
            session_tag,
            ts
        ));
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }

        // Help the LLM distinguish high-signal capsules from "ghost"/fallback ones.
        // Many recent capsules may be recorded without a full LLM extraction, which can
        // otherwise distort the recency-weighted narrative.
        if cap.extraction_mode != crate::types::ExtractionMode::Hybrid {
            let mode = match cap.extraction_mode {
                crate::types::ExtractionMode::None => "none",
                crate::types::ExtractionMode::Hybrid => "hybrid",
                crate::types::ExtractionMode::Full => "full",
            };
            context.push_str(&format!("extraction_mode: {mode}\n"));
        }
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = match cap.failure_mode {
                crate::types::FailureMode::None => "none",
                crate::types::FailureMode::Drift => "drift",
                crate::types::FailureMode::Rediscovery => "rediscovery",
                crate::types::FailureMode::DecisionConflict => "decision_conflict",
                crate::types::FailureMode::RetrySpiral => "retry_spiral",
                crate::types::FailureMode::FalseProgress => "false_progress",
                crate::types::FailureMode::UnboundedHorizon => "unbounded_horizon",
            };
            context.push_str(&format!("failure_mode: {fm}\n"));
        }
        if let Some(sig) = cap.failure_signals.as_deref() {
            let sig = sig.trim();
            if !sig.is_empty() {
                context.push_str(&format!("failure_signals: {}\n", sig.replace('\n', " ")));
            }
        }
        if let Some(e) = hit.user_emotion.as_ref() {
            context.push_str(&format!(
                "user_mood: {} conf={:.2} val={:.2} int={:.2}\n",
                e.label, e.confidence, e.valence, e.intensity
            ));
        }
        if let Some(e) = hit.assistant_emotion.as_ref() {
            context.push_str(&format!(
                "asst_mood: {} conf={:.2} val={:.2} int={:.2}\n",
                e.label, e.confidence, e.valence, e.intensity
            ));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 9 {
            break;
        }
    }

    let preamble = r#"You are unlost. Talk like a teammate discussing the codebase with the user.

Grounding rules:
- Base your answer ONLY on the provided matches. Don't invent files, symbols, routes, frameworks, or auth mechanisms.
- When you make a claim, anchor it to concrete evidence by mentioning 1-3 specific backticked tokens pulled from the matches (paths, symbols, or routes).
- When a match includes `ref=version:...` or `ref=commit:...`, prefer using that ref as the citation anchor for any noteworthy fact. Do NOT guess or invent a mapping from commit refs to changelog versions.

Session rules:
- Matches may come from different agent sessions. If you see multiple distinct sessions, call that out briefly (e.g. "across multiple sessions") and avoid merging conflicting threads.
- Do NOT print session identifiers (even if shown as `session=s1` etc) unless the user explicitly asks.
- If timestamps are present, you may reference recency or ordering (keep it brief).

Clarity rules:
- Start with a direct answer (no forced "Yes/No" prelude).
- If the question is too broad to answer from evidence, say that plainly and state what narrower question would be answerable.

Emotion rules:
- Only mention emotional tone if explicit `user_mood` / `asst_mood` lines are present in the matches.
- If there are no mood lines, do NOT infer or guess emotion; leave it out entirely.
- If mood lines are present but weak/mixed, omit emotion.

Style rules:
- First person, conversational, concise, kind, constructive: 4-6 sentences.
- No headings, no bullets, no "report" language.
- Never output internal/system/tool boilerplate (e.g. anything like `<system-reminder>...</system-reminder>`).
- Wrap code identifiers in backticks (e.g. `proxy_request`), file paths in backticks (e.g. `src/main.rs`, `main.py`), and routes in backticks (e.g. `GET /inventory`).
- End with ONE actionable next step, phrased as a concrete `unlost query ...` suggestion (not grep/file search)."#;

    let out =
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative;
    Ok(crate::util::strip_llm_boilerplate(out))
}

pub(crate) fn colorize_backticks(input: &str) -> String {
    // Very small, dependency-free ANSI highlighting pass.
    // - `GET /foo` etc -> yellow
    // - `src/main.rs` or `main.py` -> green
    // - everything else -> cyan
    let methods = [
        "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "CONNECT", "TRACE",
    ];
    let exts = [
        ".rs", ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".toml", ".json", ".yaml",
        ".yml", ".md",
    ];

    let mut out = String::with_capacity(input.len() + 32);
    let mut in_tick = false;
    let mut buf = String::new();

    for ch in input.chars() {
        if ch == '`' {
            if in_tick {
                let t = buf.trim();
                let is_route = methods.iter().any(|m| t.starts_with(m) && t.contains(" /"));
                let is_path = t.contains('/') || exts.iter().any(|e| t.ends_with(e));

                let color = if is_route {
                    "\x1b[33m" // yellow
                } else if is_path {
                    "\x1b[32m" // green
                } else {
                    "\x1b[36m" // cyan
                };

                out.push('`');
                out.push_str(color);
                out.push_str(t);
                out.push_str("\x1b[0m");
                out.push('`');

                buf.clear();
                in_tick = false;
            } else {
                in_tick = true;
            }
            continue;
        }

        if in_tick {
            buf.push(ch);
        } else {
            out.push(ch);
        }
    }

    // Unbalanced backtick: just append it back.
    if in_tick {
        out.push('`');
        out.push_str(&buf);
    }

    out
}

fn protect_spaces_inside_backticks(s: &str) -> String {
    // Prevent wrapping from splitting code spans like `GET /foo`.
    // We replace spaces inside backticks with a sentinel, then restore after wrapping.
    const SENTINEL: char = '\x1f';
    let mut out = String::with_capacity(s.len());
    let mut in_tick = false;
    for ch in s.chars() {
        if ch == '`' {
            in_tick = !in_tick;
            out.push(ch);
            continue;
        }
        if in_tick && ch == ' ' {
            out.push(SENTINEL);
        } else {
            out.push(ch);
        }
    }
    out
}

fn restore_spaces_inside_backticks(s: &str) -> String {
    s.replace('\x1f', " ")
}

fn split_list_prefix(s: &str) -> Option<(&str, &str)> {
    // Returns (marker, rest) for markdown-ish list lines.
    // Assumes `s` is left-trimmed.
    if let Some(rest) = s.strip_prefix("- ") {
        return Some(("- ", rest));
    }
    if let Some(rest) = s.strip_prefix("* ") {
        return Some(("* ", rest));
    }
    if let Some(rest) = s.strip_prefix("+ ") {
        return Some(("+ ", rest));
    }
    if let Some(rest) = s.strip_prefix("> ") {
        return Some(("> ", rest));
    }

    // Numbered list: 1. foo  /  1) foo
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    if i + 1 < bytes.len() && (bytes[i] == b'.' || bytes[i] == b')') && bytes[i + 1] == b' ' {
        let (marker, rest) = s.split_at(i + 2);
        return Some((marker, rest));
    }
    None
}

fn wrap_line_preserving_backticks(line: &str, width: usize) -> Vec<String> {
    if width < 10 {
        return vec![line.trim_end().to_string()];
    }
    let line = line.trim_end();
    if line.trim().is_empty() {
        return vec![String::new()];
    }
    if line.len() <= width {
        return vec![line.to_string()];
    }

    let indent_len = line.chars().take_while(|c| c.is_ascii_whitespace()).count();
    let indent = " ".repeat(indent_len);

    let trimmed = line.trim_start();
    let protected = protect_spaces_inside_backticks(trimmed);
    let (marker, rest) = split_list_prefix(&protected).unwrap_or(("", protected.as_str()));
    let first_prefix = format!("{indent}{marker}");
    let hanging_prefix = " ".repeat(indent_len + marker.len());

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    cur.push_str(&first_prefix);
    let mut cur_len = first_prefix.len();
    let mut base_len = cur_len;

    for word in rest.split_whitespace() {
        let wlen = word.len();
        let needs_space = cur_len > base_len;
        let add_len = wlen + if needs_space { 1 } else { 0 };

        if cur_len + add_len > width && cur_len > base_len {
            out.push(restore_spaces_inside_backticks(cur.trim_end()));
            cur.clear();
            cur.push_str(&hanging_prefix);
            cur_len = hanging_prefix.len();
            base_len = cur_len;
        }

        if cur_len > base_len {
            cur.push(' ');
            cur_len += 1;
        }
        cur.push_str(word);
        cur_len += wlen;
    }

    out.push(restore_spaces_inside_backticks(cur.trim_end()));
    out
}

pub(crate) fn render_narrative(output: OutputFormat, s: &str) -> String {
    let output = if std::env::var_os("NO_COLOR").is_some() {
        OutputFormat::Plain
    } else {
        output
    };

    let s = crate::util::strip_llm_boilerplate(s.trim().to_string());

    match output {
        OutputFormat::Plain => s.trim().to_string(),
        OutputFormat::Ansi => {
            // Dim “tips” lines so they read as guidance, not facts.
            // We intentionally skip backtick-coloring inside dimmed lines, so dim stays consistent.
            let wrap_width = 80usize;
            let mut out = String::with_capacity(s.len() + 64);
            let mut first = true;
            for line in s.lines() {
                let l = line.trim_end();
                let lower = l.to_ascii_lowercase();
                let is_tip = lower.starts_with("evidence note:")
                    || lower.starts_with("follow-up query:")
                    || lower.starts_with("follow up query:")
                    || lower.starts_with("next step:");

                let wrapped = wrap_line_preserving_backticks(l, wrap_width);
                for wl in wrapped {
                    if !first {
                        out.push('\n');
                    }
                    first = false;

                    if is_tip {
                        out.push_str("\x1b[2m");
                        out.push_str(&wl);
                        out.push_str("\x1b[0m");
                    } else {
                        out.push_str(&colorize_backticks(&wl));
                    }
                }
            }
            out
        }
    }
}

pub(crate) async fn llm_trace_narrative(
    llm_model_override: Option<&str>,
    query_text: &str,
    workspace_root: &str,
    chain: &[crate::CapsuleHit],
) -> anyhow::Result<String> {
    use chrono::{SecondsFormat, TimeZone};

    let fmt_ts = |ts_ms: i64| -> String {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
            .unwrap_or_else(|| ts_ms.to_string())
    };

    // Build graph for relationship grounding (non-fatal)
    let cg = if chain.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        crate::workspace::build_graph_for_workspace(root)
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Trace query:\n");
    context.push_str(query_text);
    context.push_str("\n\n");
    context.push_str(&format!(
        "Chain: {} capsules (chronological, oldest first)\n\n",
        chain.len()
    ));

    for (i, hit) in chain.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} time={} source={} category={}{}\n",
            i + 1,
            fmt_ts(hit.ts_ms),
            meta.source,
            cap.category,
            ref_tok,
        ));
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
            let fm = fm.trim_matches('"');
            context.push_str(&format!("failure_mode: {fm}\n"));
        }
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 49 {
            break;
        }
    }

    let preamble = r#"You are unlost trace. Your job is to reconstruct the causal story: the path of decisions that led to the current state.

Rules:
- The capsules are in chronological order (oldest first). Read them as a timeline.
- Base your output ONLY on the provided capsules. Do not invent decisions or symbols.
- Identify the TURNING POINTS: moments where the direction changed, a constraint was established, or a failure was recorded.
- When failure modes are present (retry_spiral, drift, decision_conflict, etc.), name them — they explain WHY the path bent.
- Keep the narrative causal: "because of X, we ended up doing Y, which led to Z."
- Anchor every claim to 1-2 specific backticked tokens (file paths, symbols, or decisions).
- When a capsule includes `ref=version:...` or `ref=commit:...`, use that ref to ground noteworthy facts (prefer release versions when present, otherwise commit refs). Do NOT guess commit->version mappings.
- Do NOT mention timestamps, session IDs, or capsule numbers.

Output format:
- 1 sentence: the current state (what is true now, at the end of the chain).
- Then 3-6 bullets: the key steps in the causal path, in order. Each bullet is one turning point.
- Then 1 sentence: the single most important constraint or lesson that emerges from this path.
- End with ONE concrete follow-up: `unlost query "..."` or `unlost brief <scope>`.

Style: first person, teammate tone, concise. No headings. No "report" language."#;

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative,
    )
}

pub(crate) async fn llm_recall_narrative(
    llm_model_override: Option<&str>,
    scope: Option<&str>,
    workspace_id: &str,
    workspace_root: &str,
    hits: &[crate::CapsuleHit],
    interventions: &[crate::metrics::Intervention],
    interventions_printed: bool,
    interventions_in_context: bool,
    git_capsules_included: bool,
) -> Result<String> {
    // Build graph once for relationship grounding. Failure is non-fatal.
    let cg = if hits.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        match crate::workspace::build_graph_for_workspace(root) {
            Some(g) => Some(g),
            None => {
                tracing::warn!(
                    "llm_recall_narrative: failed to build code graph for {workspace_root}, proceeding without relationship grounding"
                );
                None
            }
        }
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Recall context\n\n");

    // Runtime settings are non-capsule evidence about how this recall run was configured.
    // The LLM may use these to avoid suggesting already-applied changes.
    context.push_str("Recall runtime settings (non-capsule evidence):\n");
    context.push_str(&format!("interventions_printed: {}\n", if interventions_printed { "true" } else { "false" }));
    context.push_str(&format!("interventions_in_context: {}\n", if interventions_in_context { "true" } else { "false" }));
    context.push_str(&format!("git_capsules_included: {}\n\n", if git_capsules_included { "true" } else { "false" }));
    context.push_str("Runtime controls (non-capsule evidence):\n");
    context.push_str("interventions_default: printed\n");
    context.push_str("hide_interventions_env: UNLOST_RECALL_HIDE_INTERVENTIONS\n");
    context.push_str("include_interventions_in_context_env: UNLOST_RECALL_INTERVENTIONS_IN_CONTEXT\n\n");
    if let Some(s) = scope {
        context.push_str("Scope:\n");
        context.push_str(s);
        context.push_str("\n\n");
    } else {
        context.push_str("Scope:\n");
        context.push_str("workspace: ");
        context.push_str(workspace_id);
        context.push('\n');
        context.push_str("root: ");
        context.push_str(workspace_root);
        context.push_str("\n\n");
    }
    // Determine the most-recent session key so we can suppress `next:` lines
    // for older sessions — stale next-steps from past sessions are almost never
    // still valid and mislead the LLM into suggesting resolved work.
    let latest_session_key: Option<String> = hits.first().map(|h| {
        if let Some(s) = h.meta.agent_session_id.as_deref() {
            if !s.trim().is_empty() {
                return format!("ses:{s}");
            }
        }
        format!("conn:{}", h.conn_id)
    });

    context.push_str("Capsules (most recent first):\n");
    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let this_session_key: String = {
            if let Some(s) = hit.meta.agent_session_id.as_deref() {
                if !s.trim().is_empty() {
                    format!("ses:{s}")
                } else {
                    format!("conn:{}", hit.conn_id)
                }
            } else {
                format!("conn:{}", hit.conn_id)
            }
        };
        let is_latest_session = latest_session_key.as_deref() == Some(&this_session_key);
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} ts_ms={} id={} conn_id={} exchange_seq={} http_status={} source={} category={} upstream={} path={}{}\n",
            i + 1,
            hit.ts_ms,
            hit.id,
            hit.conn_id,
            hit.exchange_seq,
            meta.http_status,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path,
            ref_tok,
        ));
        if let Some(e) = hit.user_emotion.as_ref() {
            context.push_str(&format!(
                "user_mood: {} conf={:.2} val={:.2} int={:.2}\n",
                e.label, e.confidence, e.valence, e.intensity
            ));
        }
        if let Some(e) = hit.assistant_emotion.as_ref() {
            context.push_str(&format!(
                "asst_mood: {} conf={:.2} val={:.2} int={:.2}\n",
                e.label, e.confidence, e.valence, e.intensity
            ));
        }
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }
        if !cap.next_steps.is_empty() && is_latest_session {
            let steps = cap
                .next_steps
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            context.push_str(&format!("next: {steps}\n"));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 39 {
            break;
        }
    }

    // Include recent friction interventions in context
    if !interventions.is_empty() {
        context.push_str("\nRecent friction interventions (system detected workflow friction):\n");
        for (i, iv) in interventions.iter().enumerate() {
            let ts_str = chrono::Utc
                .timestamp_millis_opt(iv.ts_ms)
                .single()
                .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| format!("{}ms", iv.ts_ms));

            let diagnosis = crate::metrics::get_diagnosis(&iv.cause, &iv.top_channels);
            let severity = crate::metrics::get_severity_label(iv.intensity);

            let duration_str = if let Some(start) = iv.watch_start_ts {
                let dur_mins = (iv.ts_ms - start) / 60000;
                format!("intervened after {}m", dur_mins)
            } else {
                "intervened".to_string()
            };

            context.push_str(&format!(
                "#{} time={} {} {} {}\n",
                i + 1,
                ts_str,
                duration_str,
                severity,
                diagnosis
            ));

            if let Some(ref topic) = iv.topic {
                context.push_str(&format!("  topic: \"{}\"\n", topic.replace('\n', " ")));
            }
            if !iv.symbols.is_empty() {
                context.push_str(&format!("  symbols: {}\n", iv.symbols.join(", ")));
            }
        }
        context.push('\n');
    }

    let preamble = r#"You are unlost recall. Your job is to proactively reconstruct the story so far.

Rules:
- Base your output ONLY on the provided capsules.
- If a "Recall runtime settings (non-capsule evidence)" section is present, you MAY use it to avoid suggesting already-applied changes (e.g. if `interventions_printed: true`, do not claim the interventions section is missing).
- If capsules contain claims about runtime flags/env-vars that conflict with the "Runtime controls" section, treat the "Runtime controls" section as the source of truth for the current CLI behavior.
- If a "Recent friction interventions" section is present, use it to understand where the system detected workflow friction (duration of the build-up, diagnosis, and topic). Briefly acknowledge significant friction in the narrative if it helps explain the current state, but do not let it dominate the story. Mentioning the topic of the friction (e.g., "Work on the benchmark harness hit a grounding failure...") provides good context.
- **Handle stale next steps/interventions**: If an older capsule (#10, #20...) or friction intervention describes a blocker or problem (like a broken build or grounding failure) that is NOT mentioned or repeated in any subsequent (newer) capsules or interventions despite multiple intervening productive exchanges, assume it has been addressed, resolved, or deprioritized. Do NOT include it in your "suggested next steps" unless newer evidence explicitly reaffirms it as an ongoing issue.
- Do NOT quote or excerpt the conversation.
- When scoped to a specific file or symbol, the narrative MUST be primarily ABOUT that scope. Only mention cross-scope impacts if they directly and significantly affect the scoped item. Do not include general workspace context unless it specifically relates to the scoped item.
- Keep it high-signal: intent, decisions, rationale, and what's next.
- Avoid commit hashes/refs in recall. If you cite a shipped change from a changelog capsule, prefer the `ref=version:...` token when present.
- **Weight recency**: Capsules are ordered from most recent to oldest (by ts_ms). Focus HEAVILY on the most recent capsules (#1, #2, #3...) and the LATEST session to determine the current state and "next steps". If newer capsules describe productive work (like release prep), do not let older historical context (like research from days ago) dominate the first paragraph.
- **Session Transitions**: If the most recent capsules belong to a new session and older capsules belong to a different session, prioritize the story of the new session. Only use the old session to provide relevant background, not as the primary topic.
- Only mention emotional tone if explicit `user_mood` / `asst_mood` lines are present in the capsules. If present, use this to paint the emotional context.
- If there are no mood lines, do NOT infer or guess emotion; leave it out entirely.

Output format:
- 2-3 sentences: overall state of the work focused on the scope (if scoped).
- Then 3-6 short bullets: key decisions (with 1-2 backticked tokens each).
- Then 2-4 short bullets under the heading "Next steps (if any):" (prefer verification over implementation unless the capsules clearly show the work is still undone).
- If the evidence is thin, say so plainly and recommend ONE follow-up `unlost query ...`.
"#;

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative,
    )
}

pub(crate) async fn llm_brief_narrative(
    llm_model_override: Option<&str>,
    scope: Option<&str>,
    workspace_id: &str,
    workspace_root: &str,
    hits: &[crate::CapsuleHit],
) -> anyhow::Result<String> {
    // Build graph once for relationship grounding. Failure is non-fatal.
    let cg = if hits.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        match crate::workspace::build_graph_for_workspace(root) {
            Some(g) => Some(g),
            None => {
                tracing::warn!(
                    "llm_brief_narrative: failed to build code graph for {workspace_root}, proceeding without relationship grounding"
                );
                None
            }
        }
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Brief context\n\n");

    if let Some(s) = scope {
        context.push_str("Scope:\n");
        context.push_str(s);
        context.push_str("\n\n");
    } else {
        context.push_str("Scope: full workspace\n");
        context.push_str("workspace: ");
        context.push_str(workspace_id);
        context.push('\n');
        context.push_str("root: ");
        context.push_str(workspace_root);
        context.push_str("\n\n");
    }

    context.push_str(
        "Capsules (scored by importance — failure modes, rationale, cross-session recurrence):\n",
    );
    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} ts_ms={} id={} source={} category={}{}\n",
            i + 1,
            hit.ts_ms,
            hit.id,
            meta.source,
            cap.category,
            ref_tok,
        ));
        // Include failure mode explicitly — it's the primary selection signal and the LLM
        // should know it was recorded as a trap or mistake.
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
            let fm = fm.trim_matches('"');
            context.push_str(&format!("failure_mode: {}", fm));
            if let Some(ref sig) = cap.failure_signals {
                context.push_str(&format!(" ({})", sig.replace('\n', " ")));
            }
            context.push('\n');
        }
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 39 {
            break;
        }
    }

    let scope_clause = if let Some(s) = scope {
        format!(
            "The brief is scoped to: `{s}`. Focus exclusively on what a collaborator needs \
             to know about that specific area. Only mention cross-cutting concerns if they \
             directly affect `{s}`.\n\n"
        )
    } else {
        String::new()
    };

    let preamble = format!(
        r#"You are a staff engineer giving a new collaborator a codebase brief.
{scope_clause}Your job: synthesize what this person MUST know to work here without getting surprised.
Do NOT narrate history. Do NOT summarize what happened. Produce a map, not a story.
Prioritize non-obvious choices, invariants, and traps. Capsules with `failure_mode` set
represent real recorded pain — those belong in THINGS THAT BITE.

Output EXACTLY these 5 section headers, each on its own line in ALL CAPS, followed by
their content. No other headers. No preamble. Start directly with the first header.

MENTAL MODEL
  2-3 sentences. What is this system, what is its core loop, the one invariant to internalize.
  Be concrete — name the key modules/files with backticks.

KEY DESIGN DECISIONS
  3-6 bullets starting with `• `. Each bullet: a non-obvious choice and the reason it was made.
  Anchor each bullet with 1-2 backticked symbols, paths, or concepts from the capsules.
  Only include decisions that have a recorded rationale or that recur across sessions.

THINGS THAT BITE
  2-5 bullets starting with `• `. Gotchas, footguns, and hard-learned lessons.
  Any capsule with failure_mode set (retry_spiral, decision_conflict, drift, etc.) MUST
  appear here. Make it concrete — what exactly breaks, what assumption is wrong.

ENTRY POINTS
  2-4 bullets starting with `• `. Where to start reading. Use backticked `file:line` format
  where possible. One sentence per bullet on why this is the right starting place.

GO DEEPER
  2-3 lines. Each line is a concrete `unlost` command the reader should run next to drill
  into the most important or unclear areas. Use `unlost brief <scope>` for area deep-dives
  and `unlost query "<question>"` for specific questions. No bullet markers on these lines.

Rules:
- Base output ONLY on the provided capsules.
- Do not invent symbols, paths, or decisions not present in the capsules.
- Do not mention timestamps, session IDs, or capsule IDs.
- When a capsule includes `ref=version:...` or `ref=commit:...`, use that ref to ground noteworthy facts (prefer `ref=version:...` when present). Do NOT guess commit->version mappings.
- Keep each bullet to one line. No sub-bullets.
"#
    );

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, &preamble, &context)
            .await?
            .narrative,
    )
}

pub(crate) async fn llm_explore_narrative(
    llm_model_override: Option<&str>,
    query: &str,
    workspace_root: &str,
    hits: &[crate::CapsuleHit],
) -> anyhow::Result<String> {
    let cg = if hits.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        match crate::workspace::build_graph_for_workspace(root) {
            Some(g) => Some(g),
            None => {
                tracing::warn!(
                    "llm_explore_narrative: failed to build code graph for {workspace_root}, proceeding without relationship grounding"
                );
                None
            }
        }
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Scenario to explore:\n");
    context.push_str(query);
    context.push_str("\n\nWorkspace root: ");
    context.push_str(workspace_root);
    context.push_str("\n\nCapsules (scored by importance — failure modes, rationale, cross-session recurrence):\n");

    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} ts_ms={} source={} category={}{}\n",
            i + 1,
            hit.ts_ms,
            meta.source,
            cap.category,
            ref_tok,
        ));
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
            let fm = fm.trim_matches('"');
            context.push_str(&format!("failure_mode: {}", fm));
            if let Some(ref sig) = cap.failure_signals {
                context.push_str(&format!(" ({})", sig.replace('\n', " ")));
            }
            context.push('\n');
        }
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 29 {
            break;
        }
    }

    let preamble = r#"You are unlost explore. Your job is forward-looking planning grounded strictly in workspace memory.

Output EXACTLY these 5 section headers, each on its own line in ALL CAPS, followed by their content.
No other headers. No preamble. Start directly with the first header.

SCENARIO
  1 sentence restatement of what is being explored, grounded in what the capsules reveal about the current state.

OPTIONS
  A table. Each row: one plausible path. Columns (pipe-separated, keep each row on ONE line):
  Option | Upside | Downside | Effort | Reversibility | Evidence
  Use 2-4 rows. "Evidence" must cite 1-2 backticked tokens from capsules (paths, symbols, ref=version:..., ref=commit:...).
  If you cannot cite evidence for an option, mark Evidence as "not in memory".

RECOMMENDATION
  1-2 sentences. What the capsule evidence points toward for THIS workspace specifically.
  Anchor with 1-2 backticked tokens. If evidence is too thin to recommend, say so plainly.

UNKNOWNS
  Bullet list starting with "• ". Explicit gaps in memory that would change the answer.
  Always include at least one bullet. If memory is thin, this section carries the weight.

PROBES
  2-4 lines. Each is a concrete unlost command the reader should run next.
  Use `unlost query "..."` for specific questions, `unlost trace ...` for causal chains.
  No bullet markers on these lines.

Rules:
- Base output ONLY on the provided capsules. Do not invent symbols, paths, decisions, or technologies not present.
- Every non-trivial claim in OPTIONS and RECOMMENDATION must cite 1-2 backticked tokens from capsules.
- When a capsule includes ref=version:... or ref=commit:..., prefer that ref as the citation anchor.
- Table rows must stay on ONE line. No sub-bullets inside table cells. Use short phrases.
- Do not mention session IDs, timestamps, or capsule IDs.
- If capsules don't mention the scenario at all, say so in SCENARIO and put everything in UNKNOWNS.
"#;

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative,
    )
}

pub(crate) async fn llm_challenge_narrative(
    llm_model_override: Option<&str>,
    target: &str,
    workspace_root: &str,
    hits: &[crate::CapsuleHit],
) -> anyhow::Result<String> {
    let cg = if hits.iter().any(|h| !h.capsule.symbols.is_empty()) {
        let root = std::path::Path::new(workspace_root);
        match crate::workspace::build_graph_for_workspace(root) {
            Some(g) => Some(g),
            None => {
                tracing::warn!(
                    "llm_challenge_narrative: failed to build code graph for {workspace_root}, proceeding without relationship grounding"
                );
                None
            }
        }
    } else {
        None
    };

    let mut context = String::new();
    context.push_str("Decision or technology to challenge:\n");
    context.push_str(target);
    context.push_str("\n\nWorkspace root: ");
    context.push_str(workspace_root);
    context.push_str("\n\nCapsules (scored by importance — decision/rationale, failure modes, cross-session recurrence):\n");

    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} ts_ms={} source={} category={}{}\n",
            i + 1,
            hit.ts_ms,
            meta.source,
            cap.category,
            ref_tok,
        ));
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
            let fm = fm.trim_matches('"');
            context.push_str(&format!("failure_mode: {}", fm));
            if let Some(ref sig) = cap.failure_signals {
                context.push_str(&format!(" ({})", sig.replace('\n', " ")));
            }
            context.push('\n');
        }
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
        }
        if !cap.symbols.is_empty() {
            let syms = cap
                .symbols
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            context.push_str(&format!("symbols: {syms}\n"));
            if let Some(ref g) = cg {
                if let Some(rel) = crate::workspace::relationships_for_symbols(g, &cap.symbols) {
                    context.push_str(&format!("relationships: {rel}\n"));
                }
            }
        }
        context.push('\n');
        if i >= 29 {
            break;
        }
    }

    let preamble = r#"You are unlost challenge. Your job is to pressure-test a past decision or technology choice, strictly grounded in workspace memory.

Output EXACTLY these 5 section headers, each on its own line in ALL CAPS, followed by their content.
No other headers. No preamble. Start directly with the first header.

THE DECISION
  1-2 sentences. What the decision actually was, per capsule evidence. Cite 1-2 backticked tokens.
  If the capsules don't clearly record this decision, say so plainly.

ALTERNATIVES
  A table. Each row: one realistic alternative. Columns (pipe-separated, keep each row on ONE line):
  Alternative | Upside | Downside | Migration cost | Evidence
  Use 2-4 rows. "Evidence" must cite 1-2 backticked tokens, OR state "not in memory".
  Only include alternatives that are plausible given the capsule context — do not invent generic options.

VERDICT
  Two lines:
  Keep if: <condition grounded in evidence>
  Change if: <condition grounded in evidence>

UNKNOWNS
  Bullet list starting with "• ". Explicit gaps in memory that would change the verdict.
  Always include at least one bullet. If memory is thin, this section carries the weight.

PROBES
  2-4 lines. Each is a concrete unlost command the reader should run next.
  Use `unlost query "..."` for specific questions, `unlost trace ...` for causal chains.
  No bullet markers on these lines.

Rules:
- Base output ONLY on the provided capsules. Do not invent symbols, paths, decisions, or technologies not present.
- Every non-trivial claim must cite 1-2 backticked tokens from capsules (paths, symbols, ref=version:..., ref=commit:...).
- Table rows must stay on ONE line. No sub-bullets inside table cells. Use short phrases.
- Capsules with failure_mode set (retry_spiral, decision_conflict, drift, etc.) are recorded pain — treat them as evidence against the current approach.
- Do not mention session IDs, timestamps, or capsule IDs.
- If the target decision isn't mentioned in the capsules at all, say so in THE DECISION and put everything in UNKNOWNS.
"#;

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative,
    )
}

/// Render the output of `brief` with ANSI styling.
///
/// Differs from `render_narrative` in that it bolds the 5 fixed section headers
/// so the structure is immediately scannable at a glance.
pub(crate) fn render_brief(output: OutputFormat, s: &str) -> String {
    let output = if std::env::var_os("NO_COLOR").is_some() {
        OutputFormat::Plain
    } else {
        output
    };

    let s = crate::util::strip_llm_boilerplate(s.trim().to_string());

    // Section headers the LLM is instructed to produce
    const SECTION_HEADERS: &[&str] = &[
        "MENTAL MODEL",
        "KEY DESIGN DECISIONS",
        "THINGS THAT BITE",
        "ENTRY POINTS",
        "GO DEEPER",
    ];

    match output {
        OutputFormat::Plain => s.trim().to_string(),
        OutputFormat::Ansi => {
            let wrap_width = 80usize;
            let mut out = String::with_capacity(s.len() + 128);
            let mut first = true;

            for line in s.lines() {
                let l = line.trim_end();
                let trimmed = l.trim();

                let is_header = SECTION_HEADERS
                    .iter()
                    .any(|&h| trimmed.eq_ignore_ascii_case(h));

                // GO DEEPER lines are the `unlost ...` commands — dim them like tips
                let is_go_deeper_cmd = trimmed.starts_with("unlost ");

                let wrapped = wrap_line_preserving_backticks(l, wrap_width);
                for wl in wrapped {
                    if !first {
                        out.push('\n');
                    }
                    // Add a blank line before each section header for visual breathing room,
                    // but not before the very first one.
                    if is_header && !first {
                        out.push('\n');
                    }
                    first = false;

                    if is_header {
                        // Bold + bright white for section headers
                        out.push_str("\x1b[1;97m");
                        out.push_str(&wl);
                        out.push_str("\x1b[0m");
                    } else if is_go_deeper_cmd {
                        // Dim cyan for suggested commands
                        out.push_str("\x1b[2;36m");
                        out.push_str(&wl);
                        out.push_str("\x1b[0m");
                    } else {
                        out.push_str(&colorize_backticks(&wl));
                    }
                }
            }
            out
        }
    }
}

/// Render the output of `explore` and `challenge` with ANSI styling.
///
/// Section headers (SCENARIO, OPTIONS, etc.) are bolded. `unlost ...` probe
/// lines are dimmed. Tables (lines containing `|`) are left as-is — we
/// intentionally skip `wrap_plain_text` here to preserve column alignment.
pub(crate) fn render_structured(output: OutputFormat, s: &str) -> String {
    let output = if std::env::var_os("NO_COLOR").is_some() {
        OutputFormat::Plain
    } else {
        output
    };

    let s = crate::util::strip_llm_boilerplate(s.trim().to_string());

    const SECTION_HEADERS: &[&str] = &[
        "SCENARIO",
        "OPTIONS",
        "RECOMMENDATION",
        "THE DECISION",
        "ALTERNATIVES",
        "VERDICT",
        "UNKNOWNS",
        "PROBES",
    ];

    match output {
        OutputFormat::Plain => s.trim().to_string(),
        OutputFormat::Ansi => {
            let mut out = String::with_capacity(s.len() + 256);
            let mut first = true;
            // Track whether the previous non-empty line was a table row so we
            // can detect the first row of each table block (= column headers).
            let mut prev_was_table = false;

            for line in s.lines() {
                let l = line.trim_end();
                let trimmed = l.trim();

                let is_header = SECTION_HEADERS
                    .iter()
                    .any(|&h| trimmed.eq_ignore_ascii_case(h));

                let is_probe_cmd = trimmed.starts_with("unlost ");
                let is_table_row = !is_header && trimmed.contains('|');
                // First row of a table block = column header row
                let is_table_header_row = is_table_row && !prev_was_table;

                if !first {
                    out.push('\n');
                }
                if is_header && !first {
                    out.push('\n');
                }
                first = false;

                if is_header {
                    out.push_str("\x1b[1;97m");
                    out.push_str(trimmed);
                    out.push_str("\x1b[0m");
                } else if is_probe_cmd {
                    out.push_str("\x1b[2;36m");
                    out.push_str(l);
                    out.push_str("\x1b[0m");
                } else if is_table_header_row {
                    // Bold column headers; dim the pipe separators
                    out.push_str(&render_table_row(l, true));
                    // Print a thin separator line under the header
                    out.push('\n');
                    out.push_str(&render_table_separator(l));
                } else if is_table_row {
                    // Data rows: normal text, dim pipes
                    out.push_str(&render_table_row(l, false));
                } else {
                    out.push_str(&colorize_backticks(l));
                }

                if !trimmed.is_empty() {
                    prev_was_table = is_table_row;
                }
            }
            out
        }
    }
}

/// Render a single table row: dim the `|` separators, colorize backticks in
/// cell content, and optionally bold the entire row (for column headers).
fn render_table_row(line: &str, bold_header: bool) -> String {
    // Split on `|`, colorize each cell, reassemble with dimmed pipes.
    let dim_pipe = "\x1b[2m|\x1b[0m";
    let cells: Vec<&str> = line.split('|').collect();
    let mut out = String::with_capacity(line.len() + 64);

    for (i, cell) in cells.iter().enumerate() {
        if i > 0 {
            out.push_str(dim_pipe);
        }
        let colored = colorize_backticks(cell);
        if bold_header {
            // Bold + slightly brighter for the header text
            out.push_str("\x1b[1m");
            out.push_str(colored.trim());
            // Pad with spaces to preserve rough alignment
            if cell.len() > colored.trim().len() {
                let pad = cell.len() - colored.trim().len();
                for _ in 0..pad / 2 {
                    out.push(' ');
                }
            }
            out.push_str("\x1b[0m");
        } else {
            out.push_str(&colored);
        }
    }
    out
}

/// Build a dim separator line that mirrors the column widths of the header row.
/// e.g. `Alternative | Upside | ...` → `───────────── + ────── + ...`
fn render_table_separator(header_line: &str) -> String {
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";
    let cells: Vec<&str> = header_line.split('|').collect();
    let mut out = String::new();
    for (i, cell) in cells.iter().enumerate() {
        if i > 0 {
            out.push_str(dim);
            out.push('+');
            out.push_str(reset);
        }
        let w = cell.len().max(1);
        out.push_str(dim);
        for _ in 0..w {
            out.push('─');
        }
        out.push_str(reset);
    }
    out
}

pub(crate) fn spinner_draw_target(output: OutputFormat) -> Option<ProgressDrawTarget> {
    if output != OutputFormat::Ansi {
        return None;
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return None;
    }

    // Prefer stdout so the user sees it even if stderr is hidden.
    if std::io::stdout().is_terminal() {
        return Some(ProgressDrawTarget::stdout());
    }
    if std::io::stderr().is_terminal() {
        return Some(ProgressDrawTarget::stderr());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::ENV_LOCK;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, val: &std::ffi::OsStr) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn test_colorize_backticks_plain_text() {
        let input = "hello world";
        let output = colorize_backticks(input);
        assert_eq!(output, "hello world");
    }

    #[test]
    fn test_colorize_backticks_code_identifier() {
        let input = "Use `proxy_request` to handle this";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[36m")); // cyan for identifiers
        assert!(output.contains("proxy_request"));
    }

    #[test]
    fn test_colorize_backticks_file_path() {
        let input = "Check `src/main.rs` for details";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for paths
        assert!(output.contains("src/main.rs"));
    }

    #[test]
    fn test_colorize_backticks_http_route() {
        let input = "The route is `GET /inventory`";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[33m")); // yellow for routes
        assert!(output.contains("GET /inventory"));
    }

    #[test]
    fn test_colorize_backticks_multiple_backticks() {
        let input = "`foo` and `bar` are different";
        let output = colorize_backticks(input);
        assert!(output.contains("foo"));
        assert!(output.contains("bar"));
    }

    #[test]
    fn test_colorize_backticks_unbalanced() {
        let input = "unbalanced `backtick";
        let output = colorize_backticks(input);
        assert_eq!(output, "unbalanced `backtick");
    }

    #[test]
    fn test_colorize_backticks_empty() {
        let input = "``";
        let output = colorize_backticks(input);
        // Empty backticks get colored (cyan) with escape sequences
        assert!(output.contains("\x1b[36m"));
        assert!(output.contains('`'));
    }

    #[test]
    fn test_render_narrative_plain() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input = "Yes, the code uses `auth_service` for authentication.\n\nNext step: Review permissions.";
        let output = render_narrative(OutputFormat::Plain, input);
        assert!(!output.contains("\x1b[")); // No ANSI codes
        assert!(output.contains("auth_service"));
    }

    #[test]
    fn test_render_narrative_ansi_tip_lines() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input = "Yes, this is correct.\nFollow-up query: Check the logs.\nDone.";
        let output = render_narrative(OutputFormat::Ansi, input);
        // Follow-up query line should be dimmed
        assert!(output.contains("\x1b[2m")); // dim
        assert!(output.contains("Follow-up query:"));
    }

    #[test]
    fn test_render_narrative_ansi_evidence_note() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input = "Yes.\nEvidence note: Found in config.\nNext.";
        let output = render_narrative(OutputFormat::Ansi, input);
        assert!(output.contains("\x1b[2m")); // dim
        assert!(output.contains("Evidence note:"));
    }

    #[test]
    fn test_render_narrative_no_color_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NO_COLOR", std::ffi::OsStr::new("1"));
        let input = "Test content";
        let output = render_narrative(OutputFormat::Ansi, input);
        assert!(!output.contains("\x1b["));
    }

    #[test]
    fn test_render_narrative_trims_content() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input = "  \n  Yes, this works.\n  \n  ";
        let output = render_narrative(OutputFormat::Plain, input);
        assert!(!output.starts_with("\n"));
        assert!(!output.ends_with("\n"));
    }

    #[test]
    fn test_render_narrative_next_step_tip() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input = "Yes.\nNext step: Run the tests.\nDone.";
        let output = render_narrative(OutputFormat::Ansi, input);
        assert!(output.contains("\x1b[2m")); // dim
        assert!(output.contains("Next step:"));
    }

    #[test]
    fn test_render_narrative_follow_up_query_variants() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("NO_COLOR");
        let input1 = "Answer.\nfollow up query: Check docs.";
        let input2 = "Answer.\nfollow-up query: Check docs.";
        let output1 = render_narrative(OutputFormat::Ansi, input1);
        let output2 = render_narrative(OutputFormat::Ansi, input2);
        assert!(output1.contains("\x1b[2m"));
        assert!(output2.contains("\x1b[2m"));
    }

    #[test]
    fn test_colorize_backticks_js_extension() {
        let input = "Check `app.js` for logic";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for .js files
        assert!(output.contains("app.js"));
    }

    #[test]
    fn test_colorize_backticks_python_file() {
        let input = "The file is `main.py`";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for .py files
        assert!(output.contains("main.py"));
    }

    #[test]
    fn test_colorize_backticks_go_file() {
        let input = "See `main.go` for the entry point";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for .go files
        assert!(output.contains("main.go"));
    }

    #[test]
    fn test_colorize_backticks_json_file() {
        let input = "Config is in `config.json`";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for .json files
        assert!(output.contains("config.json"));
    }

    #[test]
    fn test_colorize_backticks_yaml_file() {
        let input = "Settings in `settings.yaml`";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[32m")); // green for .yaml files
        assert!(output.contains("settings.yaml"));
    }

    #[test]
    fn test_colorize_backticks_post_route() {
        let input = "Use `POST /users` endpoint";
        let output = colorize_backticks(input);
        assert!(output.contains("\x1b[33m")); // yellow for routes
        assert!(output.contains("POST /users"));
    }
}
