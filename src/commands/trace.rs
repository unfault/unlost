use crate::cli::OutputFormat;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    target: Vec<String>,
    seeds: usize,
    fan_out: usize,
    threshold: f32,
    since: Option<String>,
    until: Option<String>,
    no_llm: bool,
    llm_model: Option<String>,
    output: OutputFormat,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let query = target.join(" ");
    let query = query.trim().to_string();
    if query.is_empty() {
        println!("Usage: unlost trace <file|symbol|question>");
        println!("  Examples:");
        println!("    unlost trace src/governor.rs");
        println!("    unlost trace \"why is the timeout 30s?\"");
        println!("    unlost trace proxy_request --since 3M");
        return Ok(());
    }

    let since_ms = match since {
        Some(ref s) => crate::util::parse_time_filter(s)?,
        None => None,
    };
    let until_ms = match until {
        Some(ref u) => crate::util::parse_time_filter(u)?,
        None => None,
    };

    let cwd = std::env::current_dir()?;
    let ws = crate::workspace::get_or_create_workspace_paths(&cwd)?;

    let spinner = if let Some(target) = crate::narrative::spinner_draw_target(output) {
        let pb = ProgressBar::new_spinner();
        pb.set_draw_target(target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg:.dim}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_message("Tracing the path...");
        Some(pb)
    } else {
        None
    };

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        false,
    )
    .await?;

    let chain = crate::storage::trace_capsules_lancedb(
        &query, seeds, fan_out, threshold, since_ms, until_ms, embedder, &ws,
    )
    .await?;

    if chain.is_empty() {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        println!("No causal chain found for: {query}");
        println!(
            "Try running `unlost replay opencode` or `unlost replay git` first to seed memory."
        );
        return Ok(());
    }

    if no_llm {
        if let Some(pb) = spinner.as_ref() {
            pb.finish_and_clear();
        }
        print_raw_chain(output, &chain);
        return Ok(());
    }

    if let Some(pb) = spinner.as_ref() {
        pb.set_message("Reconstructing the story...");
    }

    let workspace_root = crate::workspace::git_toplevel(&cwd)
        .unwrap_or_else(|| crate::workspace::canonicalize_dir(&cwd).unwrap_or(cwd.clone()));
    let workspace_root = workspace_root.to_string_lossy().to_string();

    match crate::narrative::llm_trace_narrative(
        llm_model.as_deref(),
        &query,
        &workspace_root,
        &chain,
    )
    .await
    {
        Ok(narrative) => {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }
            let rendered = crate::narrative::render_narrative(output, &narrative);
            let wrap = output != OutputFormat::Ansi || std::env::var_os("NO_COLOR").is_some();
            let out = if wrap {
                crate::util::wrap_plain_text(&rendered, 80)
            } else {
                rendered
            };
            println!("{}", out);
            // Print a compact chain footer so the user can see what was used
            if output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none() {
                println!(
                    "\x1b[2m─── chain: {} capsules | oldest: {} | newest: {}\x1b[0m",
                    chain.len(),
                    fmt_ts_short(chain.first().map(|h| h.ts_ms).unwrap_or(0)),
                    fmt_ts_short(chain.last().map(|h| h.ts_ms).unwrap_or(0)),
                );
            } else {
                println!(
                    "--- chain: {} capsules | oldest: {} | newest: {}",
                    chain.len(),
                    fmt_ts_short(chain.first().map(|h| h.ts_ms).unwrap_or(0)),
                    fmt_ts_short(chain.last().map(|h| h.ts_ms).unwrap_or(0)),
                );
            }
        }
        Err(e) => {
            if let Some(pb) = spinner.as_ref() {
                pb.finish_and_clear();
            }
            tracing::warn!(error = ?e, "trace narrative failed; printing raw chain");
            print_raw_chain(output, &chain);
        }
    }

    Ok(())
}

fn fmt_ts_short(ts_ms: i64) -> String {
    use chrono::{SecondsFormat, TimeZone};
    chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true))
        .unwrap_or_else(|| "-".to_string())
}

fn print_raw_chain(output: OutputFormat, chain: &[crate::CapsuleHit]) {
    println!("Causal chain: {} capsules (chronological)\n", chain.len());
    for (i, hit) in chain.iter().enumerate() {
        let cap = &hit.capsule;
        let meta = &hit.meta;
        if output == OutputFormat::Ansi && std::env::var_os("NO_COLOR").is_none() {
            println!("\x1b[2m#{} {}\x1b[0m", i + 1, fmt_ts_short(hit.ts_ms));
        } else {
            println!("#{} {}", i + 1, fmt_ts_short(hit.ts_ms));
        }
        if !meta.source.trim().is_empty() {
            println!("source:    {}", meta.source);
        }
        let rp = meta.request_path.trim();
        if !rp.is_empty() {
            let looks_like_semver = {
                let s = rp.strip_prefix('v').unwrap_or(rp);
                let parts: Vec<&str> = s.split('.').collect();
                parts.len() >= 2
                    && parts.len() <= 4
                    && parts
                        .iter()
                        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
            };
            if meta.source == "git" {
                println!("ref:       commit:{}", rp);
            } else if meta.source == "changelog" {
                if looks_like_semver {
                    let v = rp.strip_prefix('v').unwrap_or(rp);
                    println!("ref:       version:v{}", v);
                } else {
                    println!("ref:       version:{}", rp);
                }
            }
        }
        if !cap.category.is_empty() {
            println!("category:  {}", cap.category);
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
        if cap.failure_mode != crate::types::FailureMode::None {
            println!("failure:   {:?}", cap.failure_mode);
        }
        if !cap.symbols.is_empty() {
            println!("symbols:   {:?}", cap.symbols);
        }
        println!();
    }
}
