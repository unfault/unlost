use colored::Colorize;

pub fn run(path: String) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;
    let summary = crate::metrics::summarize_metrics(&ws.metrics_jsonl)?;

    println!("{} {}", "workspace:".dimmed(), ws.id.bold());
    println!(
        "{} {}",
        "metrics:  ".dimmed(),
        ws.metrics_jsonl.display().to_string().dimmed()
    );
    println!();

    println!("{}", "=== At a Glance (All Time) ===".cyan().bold());
    println!(
        "{:18} {}",
        "capsules:".dimmed(),
        summary.capsules.to_string().bold()
    );
    println!(
        "{:18} {}",
        "total cost:".dimmed(),
        format!("${:.4}", summary.cost_total).green().bold()
    );

    let friction_rate = if summary.tokens_total > 0 {
        (summary.friction_warnings as f64) / (summary.tokens_total as f64 / 1_000_000.0)
    } else {
        0.0
    };

    let friction_color = if friction_rate > 10.0 {
        "red"
    } else if friction_rate > 5.0 {
        "yellow"
    } else {
        "green"
    };

    println!(
        "{:18} {}",
        "friction rate:".dimmed(),
        format!("{:.1} warnings / 1M tokens", friction_rate)
            .color(friction_color)
            .bold()
    );
    println!(
        "{:18} {}",
        "avg interval:".dimmed(),
        format!(
            "{} tokens",
            summary.avg_tokens_between_interventions.round() as i64
        )
        .bold()
    );
    println!();

    if summary.recent_capsules > 0 {
        println!("{}", "=== Recent Trends (Last 24h) ===".cyan().bold());
        println!(
            "{:18} {}",
            "capsules:".dimmed(),
            summary.recent_capsules.to_string().bold()
        );
        println!(
            "{:18} {}",
            "recent cost:".dimmed(),
            format!("${:.4}", summary.recent_cost_total).green().bold()
        );
        println!(
            "{:18} {}",
            "warnings:".dimmed(),
            summary.recent_friction_warnings.to_string().yellow().bold()
        );
        println!();
    }

    println!("{}", "=== Failure Modes (LLM-detected) ===".cyan().bold());
    let fm = &summary.failure_modes;
    let total_failures = fm.total();
    if summary.capsules > 0 {
        let failure_rate = (total_failures as f64) / (summary.capsules as f64) * 100.0;
        let rate_color = if failure_rate > 10.0 {
            "red"
        } else if failure_rate > 5.0 {
            "yellow"
        } else {
            "green"
        };
        println!(
            "{} {}/{} capsules ({})",
            "total:".dimmed(),
            total_failures,
            summary.capsules,
            format!("{:.1}%", failure_rate).color(rate_color).bold()
        );
    }
    println!("  {:18} {}", "drift:".dimmed(), fm.drift);
    println!("  {:18} {}", "rediscovery:".dimmed(), fm.rediscovery);
    println!(
        "  {:18} {}",
        "retry_spiral:".dimmed(),
        fm.retry_spiral.to_string().yellow()
    );
    println!("  {:18} {}", "false_progress:".dimmed(), fm.false_progress);
    println!(
        "  {:18} {}",
        "unbounded_horizon:".dimmed(),
        fm.unbounded_horizon
    );
    println!();

    println!("{}", "=== Heuristic Signals ===".cyan().bold());
    if summary.drift_paths_checked > 0 {
        let miss = summary.drift_paths_missing;
        let chk = summary.drift_paths_checked;
        let rate = (miss as f64) / (chk as f64) * 100.0;
        let rate_color = if rate > 10.0 {
            "red"
        } else if rate > 5.0 {
            "yellow"
        } else {
            "green"
        };
        println!(
            "{} {}/{} missing ({})",
            "drift (paths):".dimmed(),
            miss,
            chk,
            format!("{:.1}%", rate).color(rate_color).bold()
        );
    } else {
        println!("{}", "drift (paths):    no path symbols checked".dimmed());
    }
    println!(
        "{} {}",
        "retry_spiral: ".dimmed(),
        format!("{} friction warnings injected", summary.friction_warnings)
            .yellow()
            .bold()
    );
    if summary.friction_warnings > 0 {
        println!(
            "  {:18} {}",
            "loop:".dimmed(),
            summary.friction_by_cause.get("loop").unwrap_or(&0)
        );
        println!(
            "  {:18} {}",
            "spec:".dimmed(),
            summary.friction_by_cause.get("spec").unwrap_or(&0)
        );
        println!(
            "  {:18} {}",
            "drift:".dimmed(),
            summary.friction_by_cause.get("drift").unwrap_or(&0)
        );
        println!(
            "  {:18} {}",
            "legacy:".dimmed(),
            summary.friction_by_cause.get("legacy").unwrap_or(&0)
        );
        println!(
            "  {:18} {:.2}",
            "avg intensity:".dimmed(),
            summary.friction_intensity_total / (summary.friction_warnings as f32)
        );

        if !summary.friction_by_input_bucket.is_empty() {
            println!(
                "\n{}",
                "=== Friction vs Context Size (Input Tokens) ==="
                    .cyan()
                    .bold()
            );
            println!(
                "  {} | {} | {} | {}",
                "Bucket".dimmed(),
                "Turns".dimmed(),
                "Warnings".dimmed(),
                "Rate (Warnings/100 Turns)".dimmed()
            );
            println!(
                "{}",
                "--------------|-------|----------|---------------------------".dimmed()
            );
            let mut stable_bucket = None;
            let mut inflection_bucket = None;
            let mut baseline_rate = 0.0;

            for (bucket, (warnings, turns)) in &summary.friction_by_input_bucket {
                let rate = if *turns > 0 {
                    (*warnings as f64 / *turns as f64) * 100.0
                } else {
                    0.0
                };

                if stable_bucket.is_none() && *turns >= 5 {
                    stable_bucket = Some(*bucket);
                    baseline_rate = rate;
                }

                if stable_bucket.is_some()
                    && inflection_bucket.is_none()
                    && *turns >= 5
                    && rate > baseline_rate * 2.0
                    && rate > 5.0
                {
                    inflection_bucket = Some(*bucket);
                }

                let rate_color = if rate > 20.0 {
                    "red"
                } else if rate > 10.0 {
                    "yellow"
                } else {
                    "white"
                };
                let low_n = if *turns < 5 {
                    " (low N)".dimmed()
                } else {
                    "".clear()
                };

                println!(
                    "  {:>5} - {:>5} | {:>5} | {:>8} | {}{}",
                    bucket,
                    bucket + 4000,
                    turns,
                    warnings,
                    format!("{:>5.1}%", rate).color(rate_color).bold(),
                    low_n
                );
            }

            if let Some(b) = inflection_bucket {
                println!("\n{}", format!("[DIAGNOSTIC: Context Inflection detected at {} tokens. Friction rate more than doubles past this point.]", b).red().bold());
            } else if let Some(b) = stable_bucket {
                println!("\n{}", format!("[DIAGNOSTIC: System is stable up to {} tokens. No clear friction inflection found yet.]", b).green());
            }
        }

        if !summary.top_expensive_interventions.is_empty() {
            println!("\n{}", "=== High-Cost Friction Windows ===".cyan().bold());
            println!(
                "  {} | {} | {} | {} | {}",
                "Time".dimmed(),
                "Cause".dimmed(),
                "Intensity".dimmed(),
                "Next 5 Cost".dimmed(),
                "Top Symbol".dimmed()
            );
            println!(
                "{}",
                "---------|--------|-----------|-------------|----------------".dimmed()
            );
            for intervention in &summary.top_expensive_interventions {
                let dt = chrono::DateTime::from_timestamp(intervention.ts_ms / 1000, 0)
                    .map(|d| d.format("%H:%M:%S").to_string())
                    .unwrap_or_else(|| "??:??:??".to_string());
                let sym = intervention
                    .symbols
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("-");
                println!(
                    "  {} | {:>6} | {:>9.2} | {} | {}",
                    dt.dimmed(),
                    intervention.cause.yellow(),
                    intervention.intensity,
                    format!("${:>10.4}", intervention.cost_next_5)
                        .green()
                        .bold(),
                    if sym.len() > 20 {
                        format!("{}...", &sym[..17])
                    } else {
                        sym.to_string()
                    }
                );
            }
        }

        if !summary.friction_by_symbol.is_empty() {
            println!("\n{}", "=== Top Friction Files ===".cyan().bold());
            let mut top: Vec<_> = summary.friction_by_symbol.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            for (sym, count) in top.iter().take(5) {
                println!(
                    "  {:32} {}",
                    sym.yellow(),
                    format!("{} warnings", count).bold()
                );
            }
        }

        if !summary.channel_contributions.is_empty() {
            println!(
                "\n{}",
                "=== Trigger Drivers (Symptom Breakdown) ===".cyan().bold()
            );
            let total_contrib: f32 = summary.channel_contributions.values().sum();
            let mut top: Vec<_> = summary.channel_contributions.iter().collect();
            top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (chan, val) in top.iter().take(5) {
                println!(
                    "  {:32} {}",
                    chan.dimmed(),
                    format!("{:.1}%", (*val / total_contrib) * 100.0).bold()
                );
            }
        }
    }
    println!();

    println!("{}", "=== User Engagement ===".cyan().bold());
    println!(
        "{:18} {}",
        "recall commands:".dimmed(),
        summary.recall_commands
    );
    println!(
        "{:18} {}",
        "query commands:".dimmed(),
        summary.query_commands
    );

    Ok(())
}
