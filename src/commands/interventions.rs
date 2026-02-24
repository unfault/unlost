use chrono::TimeZone;

pub fn run(
    path: String,
    limit: usize,
    since: Option<String>,
    until: Option<String>,
) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;

    let since_ms = match since {
        Some(ref s) => crate::util::parse_time_filter(s)?,
        None => None,
    };
    let until_ms = match until {
        Some(ref u) => crate::util::parse_time_filter(u)?,
        None => None,
    };

    let all_interventions = crate::metrics::get_recent_interventions(&ws.metrics_jsonl, 1000)?;

    let filtered: Vec<_> = all_interventions
        .into_iter()
        .filter(|iv| {
            let after = since_ms.map(|s| iv.ts_ms >= s).unwrap_or(true);
            let before = until_ms.map(|u| iv.ts_ms <= u).unwrap_or(true);
            after && before
        })
        .take(limit)
        .collect();

    if filtered.is_empty() {
        println!("No interventions found for workspace: {}", ws.id);
        return Ok(());
    }

    println!("workspace: {}", ws.id);
    println!();

    let now = crate::workspace::now_ms();
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

    for (i, iv) in filtered.iter().enumerate() {
        let ts_str = chrono::Utc
            .timestamp_millis_opt(iv.ts_ms)
            .single()
            .map(|dt| dt.format("%H:%M").to_string())
            .unwrap_or_else(|| "--:--".to_string());

        let ago_str = crate::util::format_elapsed_time(iv.ts_ms, now);

        let duration_str = if let Some(start) = iv.watch_start_ts {
            let dur_mins = (iv.ts_ms - start) / 60000;
            format!("Intervened after {}m", dur_mins)
        } else {
            "Intervened".to_string()
        };

        let diagnosis = crate::metrics::get_diagnosis(&iv.cause, &iv.top_channels);
        let severity = crate::metrics::get_severity_label(iv.intensity);

        let emotion_str = if let Some(ref e) = iv.user_emotion {
            if crate::governor::FRICTION_EMOTIONS.contains(&e.as_str()) {
                format!(" (user {})", e)
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        };

        let mut clean_symbols = Vec::new();
        for s in &iv.symbols {
            let trimmed = s.trim_matches(|c: char| c == ':' || c == '.' || c == ',');
            if blacklist.iter().any(|&b| b == trimmed.to_uppercase()) {
                continue;
            }
            if trimmed.contains('/') && !trimmed.contains('.') {
                let parts: Vec<&str> = trimmed.split('/').collect();
                if parts.iter().any(|&p| {
                    p == "read" || p == "write" || p == "inspect" || p == "call" || p == "tool"
                }) {
                    continue;
                }
            }
            if !trimmed.is_empty() {
                clean_symbols.push(trimmed.to_string());
            }
        }

        let symbols_str = if clean_symbols.is_empty() {
            "—".to_string()
        } else if clean_symbols.len() > 3 {
            format!(
                "{}, {} and {} others",
                clean_symbols[0],
                clean_symbols[1],
                clean_symbols.len() - 2
            )
        } else {
            clean_symbols.join(", ")
        };

        let topic = iv.topic.as_deref().unwrap_or("");
        let topic_line = if !topic.is_empty() {
            format!("\n     Topic: \"{}\"", truncate(topic, 80))
        } else {
            String::new()
        };

        println!(
            "  {}. {} ({}) | {}: {} - {}{}",
            i + 1,
            ts_str,
            ago_str,
            duration_str,
            severity,
            diagnosis,
            emotion_str
        );
        println!("     Symbols: {}", symbols_str);
        print!("{}", topic_line);
        if !topic_line.is_empty() {
            println!();
        }
        println!();
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
