use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tracing::warn;

fn query_capsules_jsonl(path: &str, query: &str, limit: usize) -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    struct Row {
        ts_ms: Option<i64>,
        conn_id: Option<u64>,
        exchange_seq: Option<u64>,
        capsule: crate::IntentCapsule,
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
            println!("ts_ms:   {ts_ms}");
        }
        if let Some(conn_id) = row.conn_id {
            println!("conn_id: {conn_id}");
        }
        if let Some(exchange_seq) = row.exchange_seq {
            println!("exchange: {exchange_seq}");
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

pub(crate) async fn run(
    query: Vec<String>,
    limit: usize,
    symbol: Option<String>,
    no_llm: bool,
    llm_model: Option<String>,
    facts: bool,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
    file: String,
) -> anyhow::Result<()> {
    let query = query.join(" ");
    let ws = crate::workspace::get_or_create_workspace_paths(&std::env::current_dir()?)?;
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
                        println!("{}\n", rendered);
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
                for hit in matches {
                    let dist = hit.distance;
                    let cap = hit.capsule;
                    let meta = hit.meta;
                    println!("---");
                    println!("distance:  {dist}");
                    println!("source:    {}", meta.source);
                    println!("category:  {}", cap.category);
                    println!("upstream:  {}", meta.upstream_host);
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
