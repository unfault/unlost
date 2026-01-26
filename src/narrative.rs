use anyhow::Result;
use indicatif::ProgressDrawTarget;
use std::io::IsTerminal;

use crate::cli::OutputFormat;

pub(crate) async fn llm_query_narrative(
    llm_model_override: Option<&str>,
    query_text: &str,
    symbol: Option<&str>,
    matches: &[crate::CapsuleHit],
) -> Result<String> {
    let mut context = String::new();
    context.push_str("Query:\n");
    context.push_str(query_text);
    context.push('\n');
    if let Some(sym) = symbol {
        context.push_str("Symbol filter: ");
        context.push_str(sym);
        context.push('\n');
    }
    context.push_str("Matches (lower distance = closer):\n");
    for (i, hit) in matches.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        context.push_str(&format!(
            "#{} distance={} source={} category={} upstream={} path={}\n",
            i + 1,
            hit.distance,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path
        ));
        if !cap.intent.trim().is_empty() {
            context.push_str(&format!("intent: {}\n", cap.intent.replace('\n', " ")));
        }
        if !cap.decision.trim().is_empty() {
            context.push_str(&format!("decision: {}\n", cap.decision.replace('\n', " ")));
        }
        if !cap.rationale.trim().is_empty() {
            context.push_str(&format!("rationale: {}\n", cap.rationale.replace('\n', " ")));
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

Clarity rules:
- The FIRST sentence must be an explicit verdict: "Yes", "No", or "I don't know yet".
- If you say "I don't know yet", immediately say what is missing in one sentence.

Style rules:
- First person, conversational, concise: 4-6 sentences.
- No headings, no bullets, no "report" language.
- Never output internal/system/tool boilerplate (e.g. anything like `<system-reminder>...</system-reminder>`).
- Wrap code identifiers in backticks (e.g. `proxy_request`), file paths in backticks (e.g. `src/main.rs`, `main.py`), and routes in backticks (e.g. `GET /inventory`).
- End with ONE actionable next step, phrased as a concrete `unlost query ...` suggestion (not grep/file search)."#;

    let out = crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
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
        ".rs", ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".toml", ".json",
        ".yaml", ".yml", ".md",
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
            let mut out = String::with_capacity(s.len() + 32);
            for (i, line) in s.lines().enumerate() {
                if i > 0 {
                    out.push('\n');
                }

                let l = line.trim_end();
                let lower = l.to_ascii_lowercase();
                let is_tip = lower.starts_with("evidence note:")
                    || lower.starts_with("follow-up query:")
                    || lower.starts_with("follow up query:")
                    || lower.starts_with("next step:");

                if is_tip {
                    out.push_str("\x1b[2m");
                    out.push_str(l);
                    out.push_str("\x1b[0m");
                } else {
                    out.push_str(&colorize_backticks(l));
                }
            }
            out
        }
    }
}

pub(crate) async fn llm_recall_narrative(
    llm_model_override: Option<&str>,
    scope: Option<&str>,
    hits: &[crate::CapsuleHit],
) -> Result<String> {
    let mut context = String::new();
    context.push_str("Recall context\n\n");
    if let Some(s) = scope {
        context.push_str("Scope:\n");
        context.push_str(s);
        context.push_str("\n\n");
    } else {
        context.push_str("Scope:\n<workspace>\n\n");
    }
    context.push_str("Capsules (most recent first):\n");
    for (i, hit) in hits.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        context.push_str(&format!(
            "#{} ts_ms={} id={} conn_id={} exchange_seq={} http_status={} source={} category={} upstream={} path={}\n",
            i + 1,
            hit.ts_ms,
            hit.id,
            hit.conn_id,
            hit.exchange_seq,
            meta.http_status,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path
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
            context.push_str(&format!("rationale: {}\n", cap.rationale.replace('\n', " ")));
        }
        if !cap.next_steps.is_empty() {
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
        }
        context.push('\n');
        if i >= 39 {
            break;
        }
    }

    let preamble = r#"You are unlost recall. Your job is to proactively reconstruct the story so far.

Rules:
- Base your output ONLY on the provided capsules.
- Do NOT quote or excerpt the conversation.
- If scoped (a file path or symbol), focus on that scope but explicitly call out cross-scope impacts: any important symbols or files outside the scope that appear connected.
- Keep it high-signal: intent, decisions, rationale, and what's next.

Output format:
- 2-3 sentences: overall state of the work.
- Then 3-6 short bullets: key decisions (with 1-2 backticked tokens each).
- Then 2-4 short bullets: suggested next steps (as actions).
- If the evidence is thin, say so plainly and recommend ONE follow-up `unlost query ...`.
"#;

    Ok(crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
        .await?
        .narrative)
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
