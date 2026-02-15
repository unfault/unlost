use chrono::{SecondsFormat, TimeZone};

fn fmt_ts_utc(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap_or_else(|| ts_ms.to_string())
}

pub async fn run(
    path: String,
    limit: usize,
    emotion: Option<crate::cli::EmotionType>,
    provider: Option<crate::cli::ProviderType>,
    since: Option<String>,
    until: Option<String>,
    filter: Option<String>,
) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;

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

    match crate::storage::scan_capsules_lancedb_recent(
        &ws,
        limit,
        filter.as_deref(),
        emotion_label.as_deref(),
        provider_label.as_deref(),
        since_ms,
        until_ms,
    )
    .await
    {
        Ok(rows) if !rows.is_empty() => {
            println!("workspace: {}", ws.id);
            for hit in rows {
                let cap = hit.capsule;
                let meta = hit.meta;
                println!("---");
                println!("chunked_at: {}", fmt_ts_utc(hit.ts_ms));
                if let Some(ref session) = meta.agent_session_id {
                    println!("session:   {}", session);
                }
                println!(
                    "conn:      {} seq: {} status: {}",
                    hit.conn_id, hit.exchange_seq, meta.http_status
                );
                if let Some(e) = hit.user_emotion {
                    println!(
                        "user_mood:  {} (conf={:.2} val={:.2} int={:.2})",
                        e.label, e.confidence, e.valence, e.intensity
                    );
                }
                if let Some(e) = hit.assistant_emotion {
                    println!(
                        "asst_mood:  {} (conf={:.2} val={:.2} int={:.2})",
                        e.label, e.confidence, e.valence, e.intensity
                    );
                }
                println!("source:    {}", meta.source);
                println!("category:  {}", cap.category);
                println!("upstream:  {}", meta.upstream_host);
                println!("path:      {}", meta.request_path);
                if let Some(ref u) = meta.usage {
                    let model = u.model_id.as_deref().unwrap_or("-");
                    let provider = u.provider_id.as_deref().unwrap_or("-");
                    let cost = u
                        .cost
                        .map(|c| format!("{c:.6}"))
                        .unwrap_or_else(|| "-".to_string());
                    let tokens = u
                        .tokens_total()
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let input = u
                        .tokens_input
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let output = u
                        .tokens_output
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let reasoning = u
                        .tokens_reasoning
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let cache_r = u
                        .tokens_cache_read
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let cache_w = u
                        .tokens_cache_write
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    println!("model:     {provider}/{model}");
                    println!("cost:      {cost}");
                    println!(
                        "tokens:    total={tokens} in={input} out={output} reason={reasoning} cache_r={cache_r} cache_w={cache_w}"
                    );
                }
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
            Ok(())
        }
        Ok(_) => {
            println!("workspace: {}", ws.id);
            println!("no rows found");
            Ok(())
        }
        Err(e) => {
            tracing::warn!(error = ?e, "inspect failed");
            println!("inspect failed: {e}");
            Ok(())
        }
    }
}
