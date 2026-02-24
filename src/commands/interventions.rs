use chrono::{SecondsFormat, TimeZone};

fn fmt_ts_utc(ts_ms: i64) -> String {
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap_or_else(|| ts_ms.to_string())
}

fn fmt_duration_ms(ms: i64) -> String {
    let secs = ms / 1000;
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h", secs / 3600)
    }
}

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
    println!("\n");

    for iv in filtered {
        println!("---");
        println!("at:       {}", fmt_ts_utc(iv.ts_ms));

        if let Some(watch_start) = iv.watch_start_ts {
            let duration = iv.ts_ms - watch_start;
            println!(
                "building: {} ({})",
                fmt_ts_utc(watch_start),
                fmt_duration_ms(duration)
            );
        }

        let diagnosis = crate::metrics::get_diagnosis(&iv.cause, &iv.top_channels);
        let severity = crate::metrics::get_severity_label(iv.intensity);
        println!("severity: {} ({:.2})", severity, iv.intensity);
        println!("cause:    {} ({})", iv.cause, diagnosis);

        let topic = iv.topic.as_deref().unwrap_or("");
        if !topic.is_empty() {
            println!("topic:    {}", topic);
        }

        if !iv.symbols.is_empty() {
            println!("symbols:  {}", iv.symbols.join(", "));
        }

        if let Some(ref emotion) = iv.user_emotion {
            println!("emotion:  {}", emotion);
        }

        if !iv.top_channels.is_empty() {
            let mut channels: Vec<_> = iv.top_channels.iter().collect();
            channels.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let channel_str = channels
                .iter()
                .take(5)
                .map(|(k, v)| format!("{}={:.2}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            println!("channels: {}", channel_str);
        }

        println!();
    }

    Ok(())
}
