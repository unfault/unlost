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

pub(crate) async fn llm_thread_narrative(
    llm_model_override: Option<&str>,
    topic: &str,
    view: &crate::commands::thread::ThreadView,
) -> anyhow::Result<String> {
    let mut context = String::new();
    context.push_str("Thread query:\n");
    context.push_str(topic);
    context.push_str("\n\n");
    context.push_str(&view.to_llm_context());

    let preamble = format!(
        r#"A colleague skimmed my notes about "{topic}" and I asked: "so what's the deal with this?"

Below are the notes, already clustered by time with gaps and echoes marked. Say the useful thing I can't see by scanning the notes myself.

Source data says "User" — that's me. Never write "the user" or "User".

Answer in exactly 2-3 sentences, like you're talking to me. Plain language. No metaphors, no poetry, no jargon. If you'd be embarrassed saying it out loud to a colleague, don't write it.

Sentence 1: What's the one concrete thing I keep trying to get right here? Name the actual problem or goal, not an abstraction of it.

Sentence 2: What changed between the oldest and newest notes? Be specific — name the shift. If there's a gap of weeks, say whether it looks like I dropped it or came back with a different angle.

Sentence 3 (optional): One practical thing this pattern suggests I should watch out for or lean into.

Hard rules:
- No words like "throughline", "visceral", "worldview", "scope jump", "consolidation", "incubation". Write like a normal person.
- Do not restate what any note says. If I can find your sentence in the notes, it's useless.
- Do not be encouraging or affirming. Just be accurate.
- Max 80 words total."#,
    );

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, &preamble, &context)
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

    let preamble = r#"You are unlost explore. You are a thinking partner, not an auditor.

Your job: use the workspace memory as context and constraint, then think freely beyond it.
The capsules tell you what this project is, what it has decided, and where it has hurt.
Use that to inform your thinking — but don't limit yourself to it. Bring in external knowledge,
patterns from other systems, approaches this team hasn't tried. Be genuinely generative.

Output EXACTLY these 5 section headers, each on its own line in ALL CAPS, followed by their content.
No other headers. No preamble. Start directly with the first header.

CONTEXT FROM MEMORY
  2-4 sentences. What the capsules reveal about the current state relevant to this scenario.
  Mention key symbols/paths with backticks. Be direct — this is what you're building on.
  If the capsules say very little about the scenario, say so and note what IS known.

PATHS WORTH CONSIDERING
  3-5 bullets starting with "• ". Each is a distinct direction worth exploring.
  Mix: some grounded in what memory shows is already possible, some that push beyond it.
  Label each bullet with [memory] if it follows directly from capsule evidence,
  or [outside] if it draws on external knowledge/patterns not in the capsules.
  Be concrete — name specific technologies, patterns, or architectural moves.

TENSIONS
  2-4 bullets starting with "• ". What the memory reveals as real constraints, risks,
  or recorded pain that any path forward must reckon with.
  Anchor each with 1-2 backticked symbols/paths from the capsules.
  If memory is thin, name the tensions you'd expect given what you know of this kind of system.

QUESTIONS TO SIT WITH
  3-5 questions. Not things to google — things to genuinely think through before deciding.
  Provoke. The best questions here will make the user realise something they hadn't considered.

IF YOU GO FURTHER
  2-4 lines. Concrete next steps: unlost commands to deepen memory, and/or one external resource
  or experiment worth running. Mix `unlost query "..."` / `unlost trace ...` with real suggestions.
  No bullet markers on these lines.

Rules:
- Use the capsules as context and grounding, not as a cage. You are allowed to reason beyond them.
- When referencing something from memory, anchor with a backticked token. When reasoning beyond memory, say so plainly — don't fabricate capsule evidence.
- Table rows must stay on ONE line. No sub-bullets inside cells.
- Do not mention session IDs, timestamps, or capsule IDs.
- Tone: curious, direct, a smart colleague who has read your codebase and is genuinely thinking with you.
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
    deep: bool,
) -> anyhow::Result<String> {
    let root = std::path::Path::new(workspace_root);

    // Build the full graph context (hotspots, deps, routes, file list, stats).
    // This is the primary source of ground truth for structural questions —
    // capsules alone often don't capture architectural intent.
    let graph_ctx = crate::workspace::build_graph_context_for_workspace(root);
    if graph_ctx.is_none() {
        tracing::warn!(
            "llm_challenge_narrative: failed to build code graph for {workspace_root}, proceeding without graph grounding"
        );
    }

    // Build per-symbol relationships using the same graph (for capsule annotation).
    let cg = graph_ctx
        .as_ref()
        .and_then(|_| crate::workspace::build_graph_for_workspace(root));

    // Separate changelog capsules — they carry version history and are a
    // distinct signal from conversational capsules.
    let (changelog_hits, conv_hits): (Vec<_>, Vec<_>) = hits
        .iter()
        .partition(|h| h.meta.source.trim() == "changelog");

    let mut context = String::new();
    context.push_str("Decision or technology to challenge:\n");
    context.push_str(target);
    context.push_str("\n\nWorkspace root: ");
    context.push_str(workspace_root);
    context.push('\n');

    // Inject graph context so the LLM can reason about actual code structure,
    // not just what happened to get recorded in capsules.
    if let Some(ref gc) = graph_ctx {
        context.push_str(&format!(
            "\nCode graph: files={}, functions={}, call_edges={}, import_edges={}, external_modules={}\n",
            gc.stats.file_count,
            gc.stats.function_count,
            gc.stats.calls_edge_count,
            gc.stats.import_edge_count,
            gc.stats.external_module_count,
        ));
        if !gc.hotspots.is_empty() {
            context.push_str("hotspots (most-imported files):\n");
            for (score, path) in gc.hotspots.iter().take(20) {
                context.push_str(&format!("  - {path} (score={score})\n"));
            }
        }
        if !gc.deps.is_empty() {
            context.push_str("top hub dependencies:\n");
            for dep in gc.deps.iter().take(20) {
                context.push_str(&format!("  - {dep}\n"));
            }
        }
        if !gc.routes.is_empty() {
            context.push_str("routes:\n");
            for (route, handler) in gc.routes.iter().take(20) {
                context.push_str(&format!("  - {route} -> {handler}\n"));
            }
        }
        if !gc.file_paths.is_empty() {
            context.push_str("source files:\n");
            for p in gc.file_paths.iter().take(100) {
                context.push_str(&format!("  - {p}\n"));
            }
        }
        context.push('\n');
    }

    // Changelog capsules: version history and evolution signals
    if !changelog_hits.is_empty() {
        context.push_str("Changelog (version history, ordered newest first):\n");
        for hit in changelog_hits.iter().take(15) {
            let cap = &hit.capsule;
            let meta = &hit.meta;
            let ref_tok = capsule_ref_token(meta)
                .map(|r| format!(" {r}"))
                .unwrap_or_default();
            context.push_str(&format!("  -{ref_tok}"));
            if !cap.intent.trim().is_empty() {
                context.push_str(&format!(" {}", cap.intent.replace('\n', " ")));
            }
            if !cap.decision.trim().is_empty() {
                context.push_str(&format!(" | {}", cap.decision.replace('\n', " ")));
            }
            context.push('\n');
        }
        context.push('\n');
    }

    // Conversational capsules: recorded decisions, rationale, failure modes
    context.push_str("Memory capsules (decisions, rationale, recorded pain — scored by importance):\n");
    for (i, hit) in conv_hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        let ref_tok = capsule_ref_token(meta)
            .map(|r| format!(" ref={r}"))
            .unwrap_or_default();
        context.push_str(&format!(
            "#{} source={} category={}{}\n",
            i + 1,
            meta.source,
            cap.category,
            ref_tok,
        ));
        if cap.failure_mode != crate::types::FailureMode::None {
            let fm = serde_json::to_string(&cap.failure_mode).unwrap_or_default();
            let fm = fm.trim_matches('"');
            context.push_str(&format!("failure_mode: {fm}"));
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

    let preamble = if deep {
        r#"You are unlost challenge. Your job is to pressure-test a past decision or technology choice.

You have three sources of evidence — use all of them:
1. Code graph (hotspots, file structure, dependency topology, routes) — ground truth about what actually exists
2. Changelog (version history) — what changed over time and why
3. Memory capsules (recorded decisions, rationale, failure modes) — what the team thought and intended

When capsules are thin, lean on the code graph and changelog. They don't lie.

Output EXACTLY these 5 section headers, each on its own line in ALL CAPS, followed by their content.
No other headers. No preamble. Start directly with the first header.

THE DECISION
  1-2 sentences. What the decision actually was, inferred from all three sources.
  If memory capsules don't record it explicitly, infer from the code graph and changelog.
  Cite 1-2 backticked tokens (file paths, symbols, ref=version:...).

ALTERNATIVES
  2-4 alternatives. Each alternative is a named card in this exact format (repeat for each):

  ① <Short name for the alternative>
    Upside:    <one sentence>
    Downside:  <one sentence>
    Cost:      <Low | Medium | High>
    Evidence:  <1-2 backticked tokens from any source, or "not in evidence">

  Use ①, ②, ③, ④ as the numbering. Each field on its own indented line.
  Alternatives must be concrete and specific to this codebase — not generic industry options.

VERDICT
  Two lines:
  Keep if: <condition grounded in code graph / changelog / capsule evidence>
  Change if: <condition grounded in code graph / changelog / capsule evidence>

UNKNOWNS
  Bullet list starting with "• ". Gaps that would change the verdict — things not visible in any source.
  Always include at least one bullet.

PROBES
  2-4 lines. Concrete next steps: unlost commands to dig deeper.
  Use `unlost query "..."` for specific questions, `unlost trace ...` for causal chains.
  No bullet markers on these lines.

Rules:
- Use the code graph as ground truth for what exists. If hotspots show a file is heavily depended on, that's a structural fact.
- Capsules with failure_mode set (retry_spiral, decision_conflict, drift, etc.) are recorded pain — treat them as evidence against the current approach.
- Do not use tables or pipe-separated rows anywhere. Use the card format above for ALTERNATIVES.
- Do not mention session IDs, timestamps, or capsule IDs.
- Do not invent symbols or paths not present in any of the three sources.
"#
    } else {
        r#"You are unlost challenge. Your job is to pressure-test a past decision or technology choice.

You have three sources of evidence — use all of them:
1. Code graph (hotspots, file structure, dependency topology, routes) — ground truth about what actually exists
2. Changelog (version history) — what changed over time and why
3. Memory capsules (recorded decisions, rationale, failure modes) — what the team thought and intended

When capsules are thin, lean on the code graph and changelog. They don't lie.

Output EXACTLY these 3 section headers, each on its own line in ALL CAPS, followed by their content.
No other headers. No preamble. No UNKNOWNS. No PROBES. Start directly with the first header.

THE DECISION
  1 sentence. What the decision was, grounded in the evidence.
  Cite 1-2 backticked tokens (file paths, symbols, or ref=version:...).

ALTERNATIVES
  2-3 alternatives. Each in this exact format:

  ① <Short name>
    Upside:   <one short clause>
    Downside: <one short clause>

  Use ①, ②, ③ as numbering. No Cost or Evidence fields. No extra lines.
  Concrete and specific to this codebase — not generic industry options.

VERDICT
  Two lines only:
  Keep if: <one condition>
  Change if: <one condition>

Rules:
- Be brief. Every sentence must earn its place.
- Do not add sections beyond THE DECISION, ALTERNATIVES, VERDICT.
- Do not use tables or pipe-separated rows.
- Do not mention session IDs, timestamps, or capsule IDs.
- Do not invent symbols or paths not present in the evidence.
"#
    };

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
        // explore
        "CONTEXT FROM MEMORY",
        "PATHS WORTH CONSIDERING",
        "TENSIONS",
        "QUESTIONS TO SIT WITH",
        "IF YOU GO FURTHER",
        // challenge
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

            // Card field labels used in ALTERNATIVES cards
            const CARD_FIELDS: &[&str] =
                &["Upside:", "Downside:", "Cost:", "Evidence:", "Keep if:", "Change if:"];

            for line in s.lines() {
                let l = line.trim_end();
                let trimmed = l.trim();

                let is_header = SECTION_HEADERS
                    .iter()
                    .any(|&h| trimmed.eq_ignore_ascii_case(h));

                let is_probe_cmd = trimmed.starts_with("unlost ");

                // Card title: lines starting with a circled number ①②③④⑤
                let is_card_title = trimmed.starts_with(['①', '②', '③', '④', '⑤']);

                // Card field: indented line starting with a known field label
                let card_field = CARD_FIELDS
                    .iter()
                    .find(|&&f| trimmed.starts_with(f))
                    .copied();

                if !first {
                    out.push('\n');
                }
                if is_header && !first {
                    out.push('\n');
                }
                // Add a blank line before each card title for breathing room
                if is_card_title && !first {
                    out.push('\n');
                }
                first = false;

                const WRAP: usize = 80;

                if is_header {
                    out.push_str("\x1b[1;97m");
                    out.push_str(trimmed);
                    out.push_str("\x1b[0m");
                } else if is_probe_cmd {
                    // Normal (not dim) cyan so it's readable on black backgrounds.
                    // Wrap long probe lines with a hanging indent matching the leading spaces.
                    let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                    let hang = indent.len() + 2; // extra indent for continuation lines
                    for (wi, wl) in wrap_ansi_line(l.trim_end(), WRAP, hang).iter().enumerate() {
                        if wi > 0 {
                            out.push('\n');
                            for _ in 0..hang {
                                out.push(' ');
                            }
                        }
                        out.push_str(&indent);
                        out.push_str("\x1b[36m");
                        out.push_str(wl.trim_start());
                        out.push_str("\x1b[0m");
                    }
                } else if is_card_title {
                    // Bold cyan; insert a space between the circled number and the title text.
                    let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                    // The circle is one unicode scalar; split it off.
                    let mut chars = trimmed.chars();
                    let circle = chars.next().unwrap_or('①');
                    let rest_title = chars.as_str().trim_start();
                    out.push_str(&indent);
                    out.push_str("\x1b[1;36m");
                    out.push(circle);
                    out.push(' ');
                    out.push_str(rest_title);
                    out.push_str("\x1b[0m");
                } else if let Some(field) = card_field {
                    // Dim the field label; wrap the value at 80 cols with a hanging indent.
                    let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                    let rest = trimmed.strip_prefix(field).unwrap_or("").trim();
                    // Hanging indent = indent + field width + 1 space
                    let hang = indent.len() + field.len() + 1;
                    let colored_rest = colorize_backticks(rest);
                    let wrapped = wrap_ansi_line(&colored_rest, WRAP.saturating_sub(hang), 0);
                    out.push_str(&indent);
                    out.push_str("\x1b[2m");
                    out.push_str(field);
                    out.push_str("\x1b[0m ");
                    for (wi, wl) in wrapped.iter().enumerate() {
                        if wi > 0 {
                            out.push('\n');
                            for _ in 0..hang {
                                out.push(' ');
                            }
                        }
                        out.push_str(wl);
                    }
                } else {
                    // Regular prose — wrap at 80, preserving leading indent.
                    let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                    let hang = indent.len();
                    let colored = colorize_backticks(l.trim_end());
                    let wrapped = wrap_ansi_line(&colored, WRAP, hang);
                    for (wi, wl) in wrapped.iter().enumerate() {
                        if wi > 0 {
                            out.push('\n');
                            for _ in 0..hang {
                                out.push(' ');
                            }
                        }
                        out.push_str(wl);
                    }
                }
            }
            out
        }
    }
}


/// Wrap a single line (which may already contain ANSI escape sequences) to
/// `max_visible_width` columns. Returns one or more segments; callers are
/// responsible for rejoining with newline + hanging indent.
///
/// We measure visible width by skipping `\x1b[...m` sequences (SGR only).
/// Words are split on ASCII spaces; we never break inside a word.
fn wrap_ansi_line(line: &str, max_visible_width: usize, _hang: usize) -> Vec<String> {
    if max_visible_width == 0 {
        return vec![line.to_string()];
    }

    /// Count visible (non-ANSI-escape) characters in a str.
    fn visible_len(s: &str) -> usize {
        let mut len = 0;
        let mut in_esc = false;
        for ch in s.chars() {
            if ch == '\x1b' {
                in_esc = true;
            } else if in_esc {
                if ch == 'm' {
                    in_esc = false;
                }
            } else {
                len += 1;
            }
        }
        len
    }

    // Split into space-delimited tokens preserving the spaces as part of tokens
    // so we can reconstruct faithfully.
    let words: Vec<&str> = line.split(' ').collect();
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for (i, word) in words.iter().enumerate() {
        let wlen = visible_len(word);
        if current.is_empty() {
            current.push_str(word);
            current_len = wlen;
        } else if current_len + 1 + wlen <= max_visible_width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + wlen;
        } else {
            lines.push(current.clone());
            current = word.to_string();
            current_len = wlen;
        }
        let _ = i; // suppress unused warning
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// ── Reflect ───────────────────────────────────────────────────────────────────

const REFLECT_COACH_PREAMBLE: &str = "\
You are a developer effectiveness coach reviewing a coding session.\n\
Given a structured turn-by-turn evaluation timeline, produce a concise, \
actionable reflection on how the human developer collaborated with the AI agent.\n\
\n\
Focus on DEVELOPER behaviour and habits — not the agent's. \
Never blame; identify patterns and offer concrete improvements.\n\
\n\
Format your response as:\n\
NEXT ACTIONS: (3-5 bullet points — ultra-short imperatives, max 10 words each, \
no evidence citations, pure \"do this next session\"; this section comes FIRST)\n\
SESSION QUALITY: (1-2 sentences overall)\n\
WHAT WORKED: (2-3 bullet points — patterns to repeat)\n\
FRICTION POINTS: (2-4 bullet points — where collaboration broke down and why)\n\
RECOMMENDATIONS: (2-3 specific, actionable suggestions for the next session)\n\
\n\
Rules:\n\
- Every claim in SESSION QUALITY / WHAT WORKED / FRICTION POINTS / RECOMMENDATIONS \
must reference a specific turn index or flag name as evidence.\n\
- NEXT ACTIONS must be scannable in 5 seconds — no evidence, no scores, just the action.\n\
- Keep score values for evidence only — do not expose raw numbers as headlines.\n\
- Confidence: mark any claim with (low confidence) if fewer than 2 turns support it.\n\
- Max 350 words total.";

const REFLECT_TUNE_PREAMBLE: &str = "\
You are an AI agent reliability analyst reviewing a coding session.\n\
Given a structured turn-by-turn evaluation timeline, identify where the agent \
drifted, looped, hallucinated, or failed to follow instructions.\n\
\n\
Focus on AGENT behaviour — not the developer's. \
Be precise and evidence-grounded.\n\
\n\
Format your response as:\n\
NEXT ACTIONS: (3-5 bullet points — ultra-short imperatives, max 10 words each, \
no evidence citations, pure tuning changes to make; this section comes FIRST)\n\
SKILL ASSESSMENT: (for each installed skill: one line — name: helped / hurt / neutral — \
one-sentence reason grounded in turn data; then a Look for skills that: list with \
2-4 bullet points describing desired agent behaviours from the skill gaps provided — \
no skill names, just the behaviour)\n\
AGENT HEALTH: (1-2 sentences overall — stable / degraded / poor)\n\
FAILURE PATTERNS: (2-4 bullet points — specific failure modes observed with turn index)\n\
STABILITY SIGNALS: (2-3 bullet points — where the agent performed well)\n\
TUNING RECOMMENDATIONS: (2-3 suggestions for system prompt, tool policy, or model choice)\n\
\n\
Rules:\n\
- NEXT ACTIONS must be scannable in 5 seconds — no evidence, no scores, just the change.\n\
- SKILL ASSESSMENT: only audit skills from the provided installed list — do NOT \
invent or assess skills not listed. Base helped/hurt/neutral on concrete turn-data \
evidence. If evidence is weak, say neutral. For the Look for skills that: list, \
use only the skill gap descriptions provided — do not name specific skills or tools.\n\
- Anchor every finding in FAILURE PATTERNS / STABILITY SIGNALS to specific channel values \
(e.g. alignment_debt=0.72 at turn 4).\n\
- Confidence: mark any claim with (low confidence) if fewer than 2 turns support it.\n\
- Max 400 words total.";

const REFLECT_BOTH_PREAMBLE: &str = "\
You are reviewing a coding session as both a developer effectiveness coach \
and an AI agent reliability analyst.\n\
Given a structured turn-by-turn evaluation timeline, produce a combined reflection.\n\
\n\
Format your response as:\n\
NEXT ACTIONS: (3-5 bullet points — ultra-short imperatives, max 10 words each, \
no evidence citations, mix of developer and agent changes; this section comes FIRST)\n\
SKILL ASSESSMENT: (for each installed skill: one line — name: helped / hurt / neutral — \
one-sentence reason grounded in turn data; then a Look for skills that: list with \
2-4 bullet points describing desired agent behaviours from the skill gaps provided — \
no skill names, just the behaviour)\n\
SESSION QUALITY: (1-2 sentences — overall impression)\n\
DEVELOPER PATTERNS: (2-3 bullet points — human collaboration habits, good or bad)\n\
AGENT PATTERNS: (2-3 bullet points — agent drift, loops, or reliability issues)\n\
SHARED FRICTION: (1-2 bullet points — where both sides contributed to a problem)\n\
NEXT SESSION: (2-3 concrete recommendations addressing both sides)\n\
\n\
Rules:\n\
- NEXT ACTIONS must be scannable in 5 seconds — no evidence, no scores, just the action.\n\
- SKILL ASSESSMENT: only audit skills from the provided installed list — do NOT \
invent or assess skills not listed. Base verdicts on concrete turn-data evidence. \
If evidence is weak, say neutral. For the Look for skills that: list, use only \
the skill gap descriptions provided — do not name specific skills or tools.\n\
- Every claim in the other sections must reference a specific turn index or flag name.\n\
- Confidence: mark any claim with (low confidence) if fewer than 2 turns support it.\n\
- Max 500 words total.";

/// Build the structured TurnEval timeline context fed to the LLM.
/// Format: turn index, timestamp, category, outcome hint, scores, flags — no raw text.
fn build_reflect_context(
    capsules: &[crate::CapsuleHit],
    session_id: Option<&str>,
    mode: crate::cli::ReflectMode,
) -> String {
    use chrono::{SecondsFormat, TimeZone};
    let fmt_ts = |ts_ms: i64| -> String {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
            .unwrap_or_else(|| ts_ms.to_string())
    };

    let mut ctx = String::new();

    if let Some(sid) = session_id {
        ctx.push_str(&format!("Session: {sid}\n"));
    }
    ctx.push_str(&format!("Turns with evaluation data: {}\n\n", capsules.len()));
    ctx.push_str("Turn-by-turn evaluation timeline:\n\n");

    for (i, hit) in capsules.iter().enumerate() {
        let te = match hit.turn_eval.as_ref() {
            Some(te) => te,
            None => continue,
        };

        ctx.push_str(&format!("Turn {} [{}]\n", i + 1, fmt_ts(hit.ts_ms)));
        ctx.push_str(&format!("  category: {}\n", hit.capsule.category));
        ctx.push_str(&format!("  outcome: {}\n", te.outcome_hint));

        match mode {
            crate::cli::ReflectMode::Coach | crate::cli::ReflectMode::Both => {
                ctx.push_str(&format!(
                    "  coach: clarity={:.2} freshness={:.2} verify={:.2} progress={:.2} scope={:.2}\n",
                    te.clarity,
                    te.context_freshness,
                    te.verification_rigor,
                    te.decision_progress,
                    te.scope_discipline,
                ));
            }
            crate::cli::ReflectMode::Tune => {}
        }

        match mode {
            crate::cli::ReflectMode::Tune | crate::cli::ReflectMode::Both => {
                ctx.push_str(&format!(
                    "  tune: intensity={:.2} state={:?} rep={:.2} align={:.2} \
                     hall={:.2} churn={:.2} fluency={:.2}\n",
                    te.trajectory_intensity,
                    te.trajectory_state,
                    te.repetition,
                    te.alignment_debt,
                    te.path_hallucination,
                    te.logic_churn,
                    te.fluency,
                ));
            }
            crate::cli::ReflectMode::Coach => {} // tune-only block above
        }

        if !te.flags.is_empty() {
            ctx.push_str(&format!("  flags: {}\n", te.flags.join(", ")));
        }

        if !te.evidence.is_empty() {
            for ev in te.evidence.iter().take(2) {
                ctx.push_str(&format!("  evidence: {ev}\n"));
            }
        }

        ctx.push('\n');
    }

    // Aggregate summary for the LLM
    if !capsules.is_empty() {
        let total = capsules.len() as f32;
        let eval_turns: Vec<_> = capsules.iter().filter_map(|h| h.turn_eval.as_ref()).collect();
        if !eval_turns.is_empty() {
            let n = eval_turns.len() as f32;
            let avg_clarity = eval_turns.iter().map(|te| te.clarity).sum::<f32>() / n;
            let avg_freshness = eval_turns.iter().map(|te| te.context_freshness).sum::<f32>() / n;
            let avg_intensity = eval_turns.iter().map(|te| te.trajectory_intensity).sum::<f32>() / n;
            let avg_progress = eval_turns.iter().map(|te| te.decision_progress).sum::<f32>() / n;
            let flag_counts = eval_turns
                .iter()
                .flat_map(|te| te.flags.iter())
                .fold(std::collections::HashMap::<&str, usize>::new(), |mut m, f| {
                    *m.entry(f.as_str()).or_default() += 1;
                    m
                });
            let outcome_dist = capsules
                .iter()
                .filter_map(|h| h.turn_eval.as_ref())
                .fold(std::collections::HashMap::<&str, usize>::new(), |mut m, te| {
                    *m.entry(te.outcome_hint.as_str()).or_default() += 1;
                    m
                });

            ctx.push_str("Session aggregates:\n");
            ctx.push_str(&format!("  total_turns: {}\n", total as usize));
            ctx.push_str(&format!("  turns_with_eval: {}\n", eval_turns.len()));
            ctx.push_str(&format!("  avg_clarity: {avg_clarity:.2}\n"));
            ctx.push_str(&format!("  avg_context_freshness: {avg_freshness:.2}\n"));
            ctx.push_str(&format!("  avg_trajectory_intensity: {avg_intensity:.2}\n"));
            ctx.push_str(&format!("  avg_decision_progress: {avg_progress:.2}\n"));

            let mut sorted_flags: Vec<_> = flag_counts.iter().collect();
            sorted_flags.sort_by(|a, b| b.1.cmp(a.1));
            if !sorted_flags.is_empty() {
                let flag_str = sorted_flags
                    .iter()
                    .take(6)
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                ctx.push_str(&format!("  top_flags: {flag_str}\n"));
            }

            if !outcome_dist.is_empty() {
                let out_str = outcome_dist
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                ctx.push_str(&format!("  outcomes: {out_str}\n"));
            }
        }
    }

    ctx
}

/// Render the reflect narrative with ANSI colour coding.
///
/// Section headers are bold white; bullets use a cyan dash; score evidence is
/// green/yellow/red depending on value; `(low confidence)` markers are yellow.
pub(crate) fn render_reflect(output: OutputFormat, mode: crate::cli::ReflectMode, s: &str) -> String {
    let output = if std::env::var_os("NO_COLOR").is_some() {
        OutputFormat::Plain
    } else {
        output
    };

    let s = crate::util::strip_llm_boilerplate(s.trim().to_string());

    if output == OutputFormat::Plain {
        return s;
    }

    // Section headers for each mode
    const COACH_HEADERS: &[&str] = &[
        "NEXT ACTIONS",
        "SESSION QUALITY",
        "WHAT WORKED",
        "FRICTION POINTS",
        "RECOMMENDATIONS",
    ];
    const TUNE_HEADERS: &[&str] = &[
        "NEXT ACTIONS",
        "SKILL ASSESSMENT",
        "AGENT HEALTH",
        "FAILURE PATTERNS",
        "STABILITY SIGNALS",
        "TUNING RECOMMENDATIONS",
    ];
    const BOTH_HEADERS: &[&str] = &[
        "NEXT ACTIONS",
        "SKILL ASSESSMENT",
        "SESSION QUALITY",
        "DEVELOPER PATTERNS",
        "AGENT PATTERNS",
        "SHARED FRICTION",
        "NEXT SESSION",
    ];

    let headers: &[&str] = match mode {
        crate::cli::ReflectMode::Coach => COACH_HEADERS,
        crate::cli::ReflectMode::Tune => TUNE_HEADERS,
        crate::cli::ReflectMode::Both => BOTH_HEADERS,
    };

    // Mode badge colour
    let (mode_label, mode_colour) = match mode {
        crate::cli::ReflectMode::Coach => ("COACH", "\x1b[1;34m"),    // bold blue
        crate::cli::ReflectMode::Tune  => ("TUNE",  "\x1b[1;35m"),    // bold magenta
        crate::cli::ReflectMode::Both  => ("BOTH",  "\x1b[1;96m"),    // bold cyan
    };

    let mut out = String::with_capacity(s.len() + 512);

    // Top badge
    out.push_str(mode_colour);
    out.push_str("unlost reflect");
    out.push_str("\x1b[0m");
    out.push_str("  \x1b[2m──\x1b[0m  ");
    out.push_str(mode_colour);
    out.push_str(mode_label);
    out.push_str("\x1b[0m\n\n");

    const WRAP: usize = 80;

    // Track which section we're currently inside so bullets can be styled differently
    let mut current_section: &str = "";

    for line in s.lines() {
        let l = line.trim_end();
        let trimmed = l.trim();

        // Detect section header: line that is one of the known headers (with optional colon)
        let header_match = headers.iter().find(|&&h| {
            let norm = trimmed.trim_end_matches(':').trim();
            norm.eq_ignore_ascii_case(h)
        });

        if let Some(&header) = header_match {
            // Update section tracker
            current_section = header;

            // Blank line before each section (except very first)
            if !out.ends_with("\n\n") && !out.ends_with("m\n\n") {
                out.push('\n');
            }

            if header.eq_ignore_ascii_case("NEXT ACTIONS") {
                // NEXT ACTIONS: bold white + underline — visually distinct from mode sections
                out.push_str("\x1b[1;4;97m");
                out.push_str(header);
                out.push_str("\x1b[0m");
            } else if header.eq_ignore_ascii_case("SKILL ASSESSMENT") {
                // SKILL ASSESSMENT: bold yellow — skills/tooling context distinct from analysis
                out.push_str("\x1b[1;33m");
                out.push_str(header);
                out.push_str("\x1b[0m");
            } else {
                // Regular sections: bold in mode colour
                out.push_str(mode_colour);
                out.push_str(header);
                out.push_str("\x1b[0m");
            }
            // Colon if original had one
            if trimmed.ends_with(':') {
                out.push(':');
            }
            out.push('\n');
            continue;
        }

        let in_next_actions = current_section.eq_ignore_ascii_case("NEXT ACTIONS");

        // Bullet point lines
        if trimmed.starts_with("- ") || trimmed.starts_with("• ") {
            let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
            let bullet_content = trimmed[2..].trim();
            let hang = indent.len() + 2; // hanging indent for wrapped continuation

            let in_skill_assessment = current_section.eq_ignore_ascii_case("SKILL ASSESSMENT");

            if in_next_actions {
                // NEXT ACTIONS bullets: bold white arrow
                out.push_str(&indent);
                out.push_str("\x1b[1;97m→\x1b[0m ");
            } else if in_skill_assessment {
                // SKILL ASSESSMENT bullets: yellow diamond
                out.push_str(&indent);
                out.push_str("\x1b[33m◆\x1b[0m ");
            } else {
                // Regular bullets: cyan dash
                out.push_str(&indent);
                out.push_str("\x1b[36m-\x1b[0m ");
            }

            // Render the content:
            // - NEXT ACTIONS: bold white imperative (no score colouring)
            // - SKILL ASSESSMENT: colour "helped"→green, "hurt"→red, "neutral"→dim
            // - everything else: full inline colouring
            let rendered_content: String = if in_next_actions {
                format!("\x1b[1m{bullet_content}\x1b[0m")
            } else if in_skill_assessment {
                colour_skill_assessment_line(bullet_content)
            } else {
                colour_reflect_inline(bullet_content)
            };
            // Word-wrap
            for (wi, wl) in wrap_ansi_line(&rendered_content, WRAP.saturating_sub(hang + 2), 0).iter().enumerate() {
                if wi > 0 {
                    out.push('\n');
                    for _ in 0..hang {
                        out.push(' ');
                    }
                }
                out.push_str(wl);
            }
            out.push('\n');
            continue;
        }

        // Plain line — inline colouring + word-wrap at WRAP columns
        if trimmed.is_empty() {
            out.push('\n');
        } else {
            let coloured = colour_reflect_inline(trimmed);
            for (wi, wl) in wrap_ansi_line(&coloured, WRAP, 0).iter().enumerate() {
                if wi > 0 {
                    out.push('\n');
                }
                out.push_str(wl);
            }
            out.push('\n');
        }
    }

    out
}

/// Colour a skill assessment bullet line.
/// Patterns coloured:
/// - `name: helped` → name in green
/// - `name: hurt` → name in red  
/// - `name: neutral` → name in dim
/// - `recommend:` / `add:` prefix hints → yellow
fn colour_skill_assessment_line(s: &str) -> String {
    // Check for verdict pattern: "skill-name: helped/hurt/neutral ..."
    if let Some(colon_pos) = s.find(':') {
        let skill_name = s[..colon_pos].trim();
        let rest = s[colon_pos + 1..].trim();
        let verdict_lower = rest.to_lowercase();

        if verdict_lower.starts_with("helped") || verdict_lower.starts_with("positive") {
            return format!(
                "\x1b[32m{skill_name}\x1b[0m: \x1b[32mhelped\x1b[0m{}",
                &rest[6..]
            );
        } else if verdict_lower.starts_with("hurt") || verdict_lower.starts_with("negative") {
            return format!(
                "\x1b[31m{skill_name}\x1b[0m: \x1b[31mhurt\x1b[0m{}",
                &rest[4..]
            );
        } else if verdict_lower.starts_with("neutral") {
            return format!(
                "\x1b[2m{skill_name}\x1b[0m: \x1b[2mneutral\x1b[0m{}",
                &rest[7..]
            );
        }
    }
    // Hypothetical skill suggestion lines (no verdict pattern) — yellow
    format!("\x1b[33m{s}\x1b[0m")
}

/// Apply inline ANSI colour to reflect narrative text:
/// - `(low confidence)` → yellow
/// - `key=0.NN` score patterns → green/yellow/red by value
/// - `turn N` references → dim cyan
/// - backtick code spans → green (reuse existing helper)
fn colour_reflect_inline(s: &str) -> String {
    // First pass: colorize backtick spans using existing helper
    let s = colorize_backticks(s);

    // Second pass: (low confidence) → yellow
    let s = s.replace(
        "(low confidence)",
        "\x1b[33m(low confidence)\x1b[0m",
    );

    // Third pass: score patterns like `clarity=0.72`, `align=0.31` → coloured
    // We do a simple scan for `word=0.NN` patterns
    let mut result = String::with_capacity(s.len() + 64);
    let mut remaining = s.as_str();
    while let Some(eq_pos) = remaining.find('=') {
        // Check that what follows is a float
        let after_eq = &remaining[eq_pos + 1..];
        if after_eq.starts_with("0.") || after_eq.starts_with("1.") {
            // Find the end of the number
            let num_end = after_eq
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .count();
            if num_end >= 3 {
                let num_str = &after_eq[..num_end];
                if let Ok(val) = num_str.parse::<f32>() {
                    // Push everything before the `=`
                    result.push_str(&remaining[..eq_pos + 1]);
                    // Colour by value
                    let colour = if val >= 0.65 {
                        "\x1b[32m" // green: healthy / high
                    } else if val >= 0.35 {
                        "\x1b[33m" // yellow: moderate
                    } else {
                        "\x1b[31m" // red: low / concern
                    };
                    result.push_str(colour);
                    result.push_str(num_str);
                    result.push_str("\x1b[0m");
                    remaining = &after_eq[num_end..];
                    continue;
                }
            }
        }
        // No match — push up to and including `=`
        result.push_str(&remaining[..eq_pos + 1]);
        remaining = &remaining[eq_pos + 1..];
    }
    result.push_str(remaining);

    // Fourth pass: turn references → dim (char-safe)
    // Matches: "Turn N", "Turns N", "turn N", "turns N"
    // followed by digits and optional range/list suffixes like "-24", ", 25-30"
    let mut final_out = String::with_capacity(result.len() + 32);
    let mut rest = result.as_str();
    while !rest.is_empty() {
        // Case-insensitive prefix match for "turn " or "turns "
        let lower6 = rest.chars().take(6).collect::<String>().to_lowercase();
        let (matched_prefix_len, prefix_str) = if lower6.starts_with("turns ") {
            (6, &rest[..6])
        } else if lower6.starts_with("turn ") {
            (5, &rest[..5])
        } else {
            let ch = rest.chars().next().unwrap();
            final_out.push(ch);
            rest = &rest[ch.len_utf8()..];
            continue;
        };

        // What follows must start with a digit to be a real turn reference
        let after = &rest[matched_prefix_len..];
        if !after.starts_with(|c: char| c.is_ascii_digit()) {
            // Not a turn reference — emit the prefix literally and move on
            let ch = rest.chars().next().unwrap();
            final_out.push(ch);
            rest = &rest[ch.len_utf8()..];
            continue;
        }

        // Consume the full turn reference including ranges (e.g. "1-2", "8-24")
        // and comma-separated lists (e.g. "1, 3-5, 7")
        let ref_body: String = after
            .chars()
            .take_while(|&c| c.is_ascii_digit() || c == '-' || c == ',' || c == ' ')
            .collect();
        // Trim trailing spaces/commas that aren't part of the reference
        let ref_body = ref_body.trim_end_matches(|c: char| c == ',' || c == ' ');

        final_out.push_str("\x1b[2m"); // dim only — no colour, just muted
        final_out.push_str(prefix_str);
        final_out.push_str(ref_body);
        final_out.push_str("\x1b[0m");
        rest = &rest[matched_prefix_len + ref_body.len()..];
    }

    final_out
}

// ── Skill gap guidance ────────────────────────────────────────────────────────

/// Maps an observed tune signal to a short behavioural description of what
/// kind of skill would address it. Used to generate "Look for a skill that..."
/// guidance without needing a skill registry.
struct SkillGap {
    /// Tune flags or channel names that trigger this gap.
    triggers: &'static [&'static str],
    /// Short imperative describing the desired agent behaviour.
    /// Written as a "reduces X" / "improves Y" phrase.
    guidance: &'static str,
}

const SKILL_GAPS: &[SkillGap] = &[
    // ── Clarity & scope ──────────────────────────────────────────────────────
    SkillGap {
        triggers: &["needs_clarification", "scope_shift"],
        guidance: "requires definition of done and acceptance criteria before any implementation starts",
    },
    SkillGap {
        triggers: &["scope_shift", "high_churn", "logic_churn"],
        guidance: "pauses and restates scope when new topics or symbols appear mid-task",
    },
    // ── Verification & code quality ──────────────────────────────────────────
    SkillGap {
        triggers: &["unverified_claim", "retry_loop"],
        guidance: "runs build/test after every code-touching turn and surfaces the outcome before continuing",
    },
    SkillGap {
        triggers: &["unverified_claim", "high_churn"],
        guidance: "runs static analysis or type-checking on generated code before presenting it as complete",
    },
    // ── Instruction alignment ────────────────────────────────────────────────
    SkillGap {
        triggers: &["alignment_debt", "instruction_drift", "blind_acceptance"],
        guidance: "requires explicit user confirmation before irreversible changes or when instructions conflict",
    },
    // ── Grounding & hallucination ────────────────────────────────────────────
    SkillGap {
        triggers: &["hallucination_risk", "path_hallucination"],
        guidance: "verifies file and symbol existence before referencing or editing paths",
    },
    // ── Loop / stall recovery ────────────────────────────────────────────────
    SkillGap {
        triggers: &["retry_loop", "semantic_stall", "novelty_collapse"],
        guidance: "detects repeated failed approaches and proposes an alternative strategy instead of retrying",
    },
    // ── Context & cost ───────────────────────────────────────────────────────
    SkillGap {
        triggers: &["session_heavy", "session_too_long"],
        guidance: "compresses or summarises prior context before it degrades, to reduce token spend and maintain coherence",
    },
    SkillGap {
        triggers: &["session_heavy", "context_freshness"],
        guidance: "signals a session boundary recommendation when compaction pressure is high, preventing wasted turns",
    },
    SkillGap {
        triggers: &["cost_spike", "session_heavy"],
        guidance: "detects when token spend is accelerating without matching progress and proposes a scope reduction or session split",
    },
    // ── Output quality ───────────────────────────────────────────────────────
    SkillGap {
        triggers: &["blind_acceptance", "fluency"],
        guidance: "challenges its own outputs and surfaces potential issues before presenting results as complete",
    },
];

/// Build the skill context block injected into the tune/both LLM prompt.
/// Includes: installed skills (for audit) + catalogue candidates (for recommendations).
fn build_skill_context(
    installed: &[crate::commands::reflect::InstalledSkill],
    observed_flags: &std::collections::HashSet<&str>,
) -> String {
    let mut ctx = String::new();

    // Installed skills eligible for audit.
    // Infrastructure/observer skills (unlost, git-workflow, graph tools, etc.)
    // have already been excluded by the discovery step — everything here is
    // a legitimate agent-behaviour skill that can be audited against turn data.
    if !installed.is_empty() {
        ctx.push_str(
            "Installed skills to audit (these are agent-behaviour skills — \
             assess whether each one helped or hurt based on what the turn data \
             shows about agent behaviour during turns; only verdict if there is \
             actual evidence, otherwise say neutral):\n",
        );
        for s in installed {
            ctx.push_str(&format!("  - {} : {}\n", s.name, s.description));
        }
        ctx.push_str(
            "\nIMPORTANT: do NOT assess skills that are missing from this list. \
             Infrastructure tools (memory recorders, graph explorers, workflow \
             enforcers) have been excluded — they cannot be evaluated against \
             per-turn metrics without circular reasoning.\n\n",
        );
    } else {
        ctx.push_str("Installed skills eligible for audit: none found (infrastructure skills are excluded).\n\n");
    }

    // Skill gap guidance — derived from observed signals, no registry needed.
    // Deduped: take the first matching gap per unique guidance string.
    let mut gaps: Vec<(&SkillGap, Vec<&str>)> = Vec::new();
    let mut seen_guidance: Vec<&str> = Vec::new();
    for g in SKILL_GAPS {
        let matched: Vec<&str> = g
            .triggers
            .iter()
            .copied()
            .filter(|t| observed_flags.contains(*t))
            .collect();
        if matched.is_empty() || seen_guidance.contains(&g.guidance) {
            continue;
        }
        seen_guidance.push(g.guidance);
        gaps.push((g, matched));
        if gaps.len() >= 5 {
            break;
        }
    }

    if !gaps.is_empty() {
        ctx.push_str("Look for skills that:\n");
        for (g, triggers) in &gaps {
            ctx.push_str(&format!(
                "  - {} [evidence: {}]\n",
                g.guidance,
                triggers.join(", ")
            ));
        }
    } else {
        ctx.push_str("Skill gaps: no strong behavioural gaps detected from observed patterns.\n");
    }

    ctx
}

pub(crate) async fn llm_reflect_narrative(
    llm_model_override: Option<&str>,
    mode: crate::cli::ReflectMode,
    capsules: &[crate::CapsuleHit],
    session_id: Option<&str>,
    installed_skills: &[crate::commands::reflect::InstalledSkill],
) -> anyhow::Result<String> {
    let preamble = match mode {
        crate::cli::ReflectMode::Coach => REFLECT_COACH_PREAMBLE,
        crate::cli::ReflectMode::Tune => REFLECT_TUNE_PREAMBLE,
        crate::cli::ReflectMode::Both => REFLECT_BOTH_PREAMBLE,
    };

    // Collect observed flags from all eval turns for skill matching
    let observed_flags: std::collections::HashSet<&str> = capsules
        .iter()
        .filter_map(|h| h.turn_eval.as_ref())
        .flat_map(|te| te.flags.iter().map(|f| f.as_str()))
        .collect();

    // Also include high-signal channel names as pseudo-flags
    let mut observed_flags = observed_flags;
    for h in capsules {
        if let Some(te) = &h.turn_eval {
            if te.alignment_debt > 0.45      { observed_flags.insert("alignment_debt"); }
            if te.path_hallucination > 0.45  { observed_flags.insert("path_hallucination"); }
            if te.novelty_collapse > 0.45    { observed_flags.insert("novelty_collapse"); }
            if te.semantic_stall > 0.45      { observed_flags.insert("semantic_stall"); }
            if te.logic_churn > 0.45         { observed_flags.insert("logic_churn"); }
            if te.fluency > 0.55             { observed_flags.insert("fluency"); }
            // context_freshness low = compaction pressure
            if te.context_freshness < 0.40   { observed_flags.insert("context_freshness"); }
            // verification_rigor low on a code-touching turn = unverified claim
            if te.verification_rigor < 0.30  { observed_flags.insert("unverified_claim"); }
            // cost_acceleration high = token spend spiking without progress
            if te.cost_acceleration > 0.50   { observed_flags.insert("cost_spike"); }
        }
    }

    let mut context = build_reflect_context(capsules, session_id, mode);

    // Append skill context for tune/both modes
    if matches!(mode, crate::cli::ReflectMode::Tune | crate::cli::ReflectMode::Both) {
        context.push_str("\n\n");
        context.push_str(&build_skill_context(installed_skills, &observed_flags));
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct ReflectOutput {
        /// The full formatted reflection narrative.
        narrative: String,
    }

    let result =
        crate::llm_extract::<ReflectOutput>(llm_model_override, preamble, &context).await?;

    Ok(result.narrative)
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
