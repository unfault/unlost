use crate::cli::OutputFormat;
use chrono::{SecondsFormat, TimeZone};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tracing::warn;

fn fmt_ts_utc(ts_ms: i64) -> Option<String> {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
}

fn render_query_footer(output: OutputFormat, matches: &[crate::CapsuleHit]) -> String {
    if matches.is_empty() {
        return String::new();
    }

    let sessions = matches
        .iter()
        .filter_map(|h| h.meta.agent_session_id.as_deref())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let sources = matches
        .iter()
        .map(|h| h.meta.source.trim())
        .filter(|s| !s.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let providers = matches
        .iter()
        .map(|h| h.meta.upstream_host.trim())
        .filter(|s| !s.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let (min_ts, max_ts) = matches
        .iter()
        .fold((i64::MAX, i64::MIN), |(minv, maxv), h| {
            (minv.min(h.ts_ms), maxv.max(h.ts_ms))
        });

    let mut tokens_total: i64 = 0;
    let mut cost_total: f64 = 0.0;
    let mut usage_hits: usize = 0;
    for h in matches {
        if let Some(u) = h.meta.usage.as_ref() {
            usage_hits += 1;
            if let Some(t) = u.tokens_total() {
                tokens_total = tokens_total.saturating_add(t);
            }
            if let Some(c) = u.cost {
                cost_total += c;
            }
        }
    }

    let (time_start, time_end) = match (fmt_ts_utc(min_ts), fmt_ts_utc(max_ts)) {
        (Some(a), Some(b)) => (a, b),
        _ => (min_ts.to_string(), max_ts.to_string()),
    };

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "Meta: matches {} | sessions {} | sources {} | providers {}",
        matches.len(),
        sessions,
        sources,
        providers
    ));
    lines.push(format!("Time (UTC): {time_start} -> {time_end}"));

    if usage_hits > 0 {
        let mut parts: Vec<String> = Vec::new();
        parts.push(format!("Usage: {usage_hits}/{} hits", matches.len()));
        if tokens_total > 0 {
            parts.push(format!("tokens {tokens_total}"));
        }
        if cost_total > 0.0 {
            parts.push(format!("cost {cost_total:.6}"));
        }
        lines.push(parts.join(" | "));
    }

    let sep = "----------------------------------------";
    let body = lines.join("\n");

    // Only dim when ANSI output is active and NO_COLOR isn't set.
    let dim = output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none();
    if dim {
        format!("\n{sep}\n\x1b[2m{body}\x1b[0m")
    } else {
        format!("\n{sep}\n{body}")
    }
}

// plain wrapping lives in `crate::util::wrap_plain_text`

fn query_capsules_jsonl(path: &str, query: &str, limit: usize) -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    struct UsageTokensCache {
        read: Option<i64>,
        write: Option<i64>,
    }

    #[derive(serde::Deserialize)]
    struct UsageTokens {
        input: Option<i64>,
        output: Option<i64>,
        reasoning: Option<i64>,
        cache: Option<UsageTokensCache>,
    }

    #[derive(serde::Deserialize)]
    struct Usage {
        provider_id: Option<String>,
        model_id: Option<String>,
        cost: Option<f64>,
        tokens: Option<UsageTokens>,
    }

    #[derive(serde::Deserialize)]
    struct Row {
        ts_ms: Option<i64>,
        conn_id: Option<u64>,
        exchange_seq: Option<u64>,
        agent_session_id: Option<String>,
        usage: Option<Usage>,
        capsule: crate::IntentCapsule,
    }

    fn print_usage(usage: &Usage) {
        if let Some(provider_id) = usage.provider_id.as_deref() {
            println!("provider_id: {provider_id}");
        }
        if let Some(model_id) = usage.model_id.as_deref() {
            println!("model_id:    {model_id}");
        }
        if let Some(cost) = usage.cost {
            println!("cost:        {cost:.6}");
        }
        if let Some(tokens) = usage.tokens.as_ref() {
            let total = tokens.input.unwrap_or(0)
                + tokens.output.unwrap_or(0)
                + tokens.reasoning.unwrap_or(0);
            if total > 0 {
                println!(
                    "tokens:      total={total} in={} out={} reasoning={}",
                    tokens.input.unwrap_or(0),
                    tokens.output.unwrap_or(0),
                    tokens.reasoning.unwrap_or(0)
                );
            }
            if let Some(cache) = tokens.cache.as_ref() {
                let read = cache.read.unwrap_or(0);
                let write = cache.write.unwrap_or(0);
                if read > 0 || write > 0 {
                    println!("cache:       read={read} write={write}");
                }
            }
        }
    }

    let q = query.to_lowercase();
    let data = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("No capsules file found at: {path}");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    let mut matches: Vec<Row> = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let hay = format!(
            "{}\n{}\n{}",
            row.capsule.intent, row.capsule.decision, row.capsule.rationale
        )
        .to_lowercase();
        if hay.contains(&q) {
            matches.push(row);
        }
    }

    matches.reverse();
    let out = matches.into_iter().take(limit).collect::<Vec<_>>();
    if out.is_empty() {
        println!("No matches for: {query}");
        return Ok(());
    }

    println!("Found {} matches:\n", out.len());
    for row in out {
        println!("---");
        if let Some(ts_ms) = row.ts_ms {
            if let Some(ts) = fmt_ts_utc(ts_ms) {
                println!("time_utc: {ts}");
            }
            println!("ts_ms:    {ts_ms}");
        }
        if let Some(conn_id) = row.conn_id {
            println!("conn_id: {conn_id}");
        }
        if let Some(exchange_seq) = row.exchange_seq {
            println!("exchange: {exchange_seq}");
        }
        if let Some(session) = row.agent_session_id.as_deref() {
            if !session.trim().is_empty() {
                println!("session: {session}");
            }
        }
        if let Some(usage) = row.usage.as_ref() {
            print_usage(usage);
        }
        println!("category:  {}", row.capsule.category);
        if !row.capsule.intent.trim().is_empty() {
            println!("intent:    {}", row.capsule.intent);
        }
        if !row.capsule.decision.trim().is_empty() {
            println!("decision:  {}", row.capsule.decision);
        }
        if !row.capsule.rationale.trim().is_empty() {
            println!("rationale: {}", row.capsule.rationale);
        }
        if !row.capsule.next_steps.is_empty() {
            println!("next:      {:?}", row.capsule.next_steps);
        }
        println!("symbols:   {:?}\n", row.capsule.symbols);
    }

    Ok(())
}

pub async fn run(
    query: Vec<String>,
    limit: usize,
    symbol: Option<String>,
    emotion: Option<crate::cli::EmotionType>,
    provider: Option<crate::cli::ProviderType>,
    since: Option<String>,
    until: Option<String>,
    no_llm: bool,
    llm_model: Option<String>,
    facts: bool,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
    file: String,
) -> anyhow::Result<()> {
    let query = query.join(" ");
    let emotion_label = emotion.map(|e| match e {
        crate::cli::EmotionType::Joy => "joy",
        crate::cli::EmotionType::Anger => "anger",
        crate::cli::EmotionType::Frustration => "frustration",
        crate::cli::EmotionType::Sad => "sad",
        crate::cli::EmotionType::Confused => "confused",
        crate::cli::EmotionType::Neutral => "neutral",
    });
    let provider_label = provider.map(|p| match p {
        crate::cli::ProviderType::Openai => "openai",
        crate::cli::ProviderType::Anthropic => "anthropic",
        crate::cli::ProviderType::Opencode => "opencode",
    });
    let since_ms = match since {
        Some(ref s) => crate::util::parse_time_filter(s)?,
        None => None,
    };
    let until_ms = match until {
        Some(ref u) => crate::util::parse_time_filter(u)?,
        None => None,
    };
    let ws = crate::workspace::get_or_create_workspace_paths(&std::env::current_dir()?)?;

    let _ = crate::metrics::record_command_query(
        &ws,
        &query,
        limit,
        symbol.as_deref(),
        emotion_label,
        provider_label,
    );

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Let me check...");
        Some(pb)
    } else {
        None
    };

    match crate::storage::query_capsules_lancedb(
        &query,
        limit,
        symbol.as_deref(),
        emotion_label.as_deref(),
        provider_label.as_deref(),
        since_ms,
        until_ms,
        embedder.clone(),
        &ws,
    )
    .await
    {
        Ok(mut matches) if !matches.is_empty() => {
            matches.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if !no_llm {
                if let Some(pb) = spinner.as_ref() {
                    pb.set_message("Putting it together...");
                }
                match crate::narrative::llm_query_narrative(
                    llm_model.as_deref(),
                    &query,
                    symbol.as_deref(),
                    &matches,
                )
                .await
                {
                    Ok(n) => {
                        let rendered = crate::narrative::render_narrative(output, &n);
                        if let Some(pb) = spinner.as_ref() {
                            pb.finish_and_clear();
                        }
                        let footer = render_query_footer(output, &matches);
                        let mut full =
                            crate::util::strip_llm_boilerplate(format!("{rendered}{footer}"));
                        let wrap =
                            output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
                        if wrap {
                            full = crate::util::wrap_plain_text(&full, 80);
                        }
                        if !full.ends_with('\n') {
                            full.push('\n');
                        }
                        println!("{}", full);
                    }
                    Err(e) => {
                        warn!(error = ?e, "query narrative failed; printing raw matches");
                    }
                }
            }

            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }

            if no_llm || facts {
                let print_usage = |usage: &crate::types::UsageMeta| {
                    if let Some(provider_id) = usage.provider_id.as_deref() {
                        println!("provider_id: {provider_id}");
                    }
                    if let Some(model_id) = usage.model_id.as_deref() {
                        println!("model_id:    {model_id}");
                    }
                    if let Some(cost) = usage.cost {
                        println!("cost:        {cost:.6}");
                    }
                    if let Some(total) = usage.tokens_total() {
                        println!(
                            "tokens:      total={total} in={} out={} reasoning={}",
                            usage.tokens_input.unwrap_or(0),
                            usage.tokens_output.unwrap_or(0),
                            usage.tokens_reasoning.unwrap_or(0)
                        );
                    }
                    let cache_read = usage.tokens_cache_read.unwrap_or(0);
                    let cache_write = usage.tokens_cache_write.unwrap_or(0);
                    if cache_read > 0 || cache_write > 0 {
                        println!("cache:       read={cache_read} write={cache_write}");
                    }
                };

                for hit in matches {
                    let dist = hit.distance;
                    let cap = hit.capsule;
                    let meta = hit.meta;
                    println!("---");
                    println!("distance:  {dist}");
                    if let Some(ts) = fmt_ts_utc(hit.ts_ms) {
                        println!("time_utc:  {ts}");
                    }
                    println!("ts_ms:     {}", hit.ts_ms);
                    println!("source:    {}", meta.source);
                    println!("category:  {}", cap.category);
                    println!("upstream:  {}", meta.upstream_host);
                    if let Some(session) = meta.agent_session_id.as_deref() {
                        if !session.trim().is_empty() {
                            println!("session:   {session}");
                        }
                    }
                    if let Some(usage) = meta.usage.as_ref() {
                        print_usage(usage);
                    }
                    if let Some(e) = &hit.user_emotion {
                        println!(
                            "user_mood: {} (conf={:.2} val={:.2} int={:.2})",
                            e.label, e.confidence, e.valence, e.intensity
                        );
                    }
                    if let Some(e) = &hit.assistant_emotion {
                        println!(
                            "asst_mood: {} (conf={:.2} val={:.2} int={:.2})",
                            e.label, e.confidence, e.valence, e.intensity
                        );
                    }
                    println!("path:      {}", meta.request_path);
                    if !cap.intent.trim().is_empty() {
                        println!("intent:    {}", cap.intent);
                    }
                    if !cap.decision.trim().is_empty() {
                        println!("decision:  {}", cap.decision);
                    }
                    if !cap.rationale.trim().is_empty() {
                        println!("rationale: {}", cap.rationale);
                    }
                    if !cap.next_steps.is_empty() {
                        println!("next:      {:?}", cap.next_steps);
                    }
                    println!("symbols:   {:?}\n", cap.symbols);
                }
            }
        }
        Ok(_) => {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }
            println!("No matches for: {query}");
        }
        Err(e) => {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }

            let msg = e.to_string();
            if msg.contains("capsules table not found") {
                println!(
                    "No capsules indexed yet for this workspace. If you're using the companion, ensure it sends `record` calls; otherwise run `unlost init` to seed the workspace."
                );
            }
            warn!(error = ?e, "lancedb query failed; falling back to jsonl");
            let fallback = if file.trim().is_empty() {
                ws.capsules_jsonl.to_string_lossy().to_string()
            } else {
                file
            };
            query_capsules_jsonl(&fallback, &query, limit)?;
        }
    }

    Ok(())
}
