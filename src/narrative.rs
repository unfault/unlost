use anyhow::Result;
use indicatif::ProgressDrawTarget;
use std::io::IsTerminal;

use chrono::{SecondsFormat, TimeZone};

use crate::cli::OutputFormat;

pub(crate) async fn llm_query_narrative(
    llm_model_override: Option<&str>,
    query_text: &str,
    symbol: Option<&str>,
    matches: &[crate::CapsuleHit],
) -> Result<String> {
    let fmt_ts_utc = |ts_ms: i64| -> Option<String> {
        chrono::Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
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
        context.push_str(&format!(
            "#{} distance={} source={} category={} upstream={} path={}{}{}\n",
            i + 1,
            hit.distance,
            meta.source,
            cap.category,
            meta.upstream_host,
            meta.request_path,
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

pub(crate) async fn llm_recall_narrative(
    llm_model_override: Option<&str>,
    scope: Option<&str>,
    workspace_id: &str,
    workspace_root: &str,
    hits: &[crate::CapsuleHit],
) -> Result<String> {
    fn workspace_git_status_porcelain(workspace_root: &str) -> Option<String> {
        use std::process::Command;

        let root = std::path::Path::new(workspace_root);
        if !root.join(".git").exists() {
            return None;
        }

        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["status", "--porcelain=v1"]) // stable, easy to parse
            .output()
            .ok()?;

        if !out.status.success() {
            return None;
        }

        let s = String::from_utf8_lossy(&out.stdout);
        let mut lines = s.lines();

        let mut snap = String::new();
        let mut n = 0usize;
        while let Some(line) = lines.next() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if n == 0 {
                snap.push_str("git status --porcelain=v1:\n");
            }
            if n >= 40 {
                snap.push_str("... (truncated)\n");
                break;
            }
            snap.push_str(line);
            snap.push('\n');
            n += 1;
        }

        if snap.is_empty() {
            Some("git status: clean\n".to_string())
        } else {
            Some(snap)
        }
    }

    let mut context = String::new();
    context.push_str("Recall context\n\n");
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

        // Optional: allow callers to include a git snapshot to reflect uncommitted work.
        // Default is off to keep recall strictly capsule-driven.
        if std::env::var_os("UNLOST_RECALL_GIT_SNAPSHOT").is_some() {
            if let Some(snap) = workspace_git_status_porcelain(workspace_root) {
                context.push_str("Workspace snapshot (non-capsule evidence):\n");
                context.push_str(&snap);
                context.push('\n');
            }
        }
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
            context.push_str(&format!(
                "rationale: {}\n",
                cap.rationale.replace('\n', " ")
            ));
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
- If a "Workspace snapshot (non-capsule evidence)" section is present, you MAY use it only to describe current uncommitted work (e.g., which files are being edited). Do not treat it as decisions/intent; do not infer beyond what it shows.
- Do NOT quote or excerpt the conversation.
- When scoped to a specific file or symbol, the narrative MUST be primarily ABOUT that scope. Only mention cross-scope impacts if they directly and significantly affect the scoped item. Do not include general workspace context unless it specifically relates to the scoped item.
- Keep it high-signal: intent, decisions, rationale, and what's next.
- Only mention emotional tone if explicit `user_mood` / `asst_mood` lines are present in the capsules. If present, use this to paint the emotional context.
- If there are no mood lines, do NOT infer or guess emotion; leave it out entirely.

Output format:
- 2-3 sentences: overall state of the work focused on the scope (if scoped).
- Then 3-6 short bullets: key decisions (with 1-2 backticked tokens each).
- Then 2-4 short bullets: suggested next steps (as actions).
- If the evidence is thin, say so plainly and recommend ONE follow-up `unlost query ...`.
"#;

    Ok(
        crate::llm_extract::<crate::QueryNarrativeOutput>(llm_model_override, preamble, &context)
            .await?
            .narrative,
    )
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
