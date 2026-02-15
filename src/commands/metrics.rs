pub fn run(path: String) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;
    let summary = crate::metrics::summarize_metrics(&ws.metrics_jsonl)?;

    println!("workspace: {}", ws.id);
    println!("metrics:   {}", ws.metrics_jsonl.display());
    println!();

    println!("=== At a Glance (All Time) ===");
    println!("capsules:         {}", summary.capsules);
    println!("total cost:       ${:.4}", summary.cost_total);
    println!(
        "friction rate:    {:.1} warnings / 1M tokens",
        if summary.tokens_total > 0 {
            (summary.friction_warnings as f64) / (summary.tokens_total as f64 / 1_000_000.0)
        } else {
            0.0
        }
    );
    println!(
        "avg interval:     {} tokens",
        summary.avg_tokens_between_interventions.round() as i64
    );
    println!();

    if summary.recent_capsules > 0 {
        println!("=== Recent Trends (Last 24h) ===");
        println!("capsules:         {}", summary.recent_capsules);
        println!("recent cost:      ${:.4}", summary.recent_cost_total);
        println!("warnings:         {}", summary.recent_friction_warnings);
        println!();
    }

    println!("=== Failure Modes (LLM-detected) ===");
    let fm = &summary.failure_modes;
    let total_failures = fm.total();
    if summary.capsules > 0 {
        let failure_rate = (total_failures as f64) / (summary.capsules as f64) * 100.0;
        println!(
            "total:            {}/{} capsules ({:.1}%)",
            total_failures, summary.capsules, failure_rate
        );
    }
    println!("  drift:            {}", fm.drift);
    println!("  rediscovery:      {}", fm.rediscovery);
    println!("  retry_spiral:     {}", fm.retry_spiral);
    println!("  false_progress:   {}", fm.false_progress);
    println!("  unbounded_horizon:{}", fm.unbounded_horizon);
    println!();

    println!("=== Heuristic Signals ===");
    if summary.drift_paths_checked > 0 {
        let miss = summary.drift_paths_missing;
        let chk = summary.drift_paths_checked;
        let rate = (miss as f64) / (chk as f64) * 100.0;
        println!("drift (paths):    {}/{} missing ({:.1}%)", miss, chk, rate);
    } else {
        println!("drift (paths):    no path symbols checked");
    }
    println!(
        "retry_spiral:     {} friction warnings injected",
        summary.friction_warnings
    );
    if summary.friction_warnings > 0 {
        println!(
            "  loop:             {}",
            summary.friction_by_cause.get("loop").unwrap_or(&0)
        );
        println!(
            "  spec:             {}",
            summary.friction_by_cause.get("spec").unwrap_or(&0)
        );
        println!(
            "  drift:            {}",
            summary.friction_by_cause.get("drift").unwrap_or(&0)
        );
        println!(
            "  legacy:           {}",
            summary.friction_by_cause.get("legacy").unwrap_or(&0)
        );
        println!(
            "  avg intensity:    {:.2}",
            summary.friction_intensity_total / (summary.friction_warnings as f32)
        );
        println!(
            "  avg interval:     {} tokens",
            summary.avg_tokens_between_interventions.round() as i64
        );

        if !summary.friction_by_input_bucket.is_empty() {
            println!("\n=== Friction vs Context Size (Input Tokens) ===");
            println!("  Bucket      | Turns | Warnings | Rate (Warnings/100 Turns)");
            println!("--------------|-------|----------|---------------------------");
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

                let low_n = if *turns < 5 { " (low N)" } else { "" };
                println!(
                    "  {:>5} - {:>5} | {:>5} | {:>8} | {:>5.1}%{}",
                    bucket,
                    bucket + 4000,
                    turns,
                    warnings,
                    rate,
                    low_n
                );
            }

            if let Some(b) = inflection_bucket {
                println!("\n[DIAGNOSTIC: Context Inflection detected at {} tokens. Friction rate more than doubles past this point.]", b);
            } else if let Some(b) = stable_bucket {
                println!("\n[DIAGNOSTIC: System is stable up to {} tokens. No clear friction inflection found yet.]", b);
            }
        }

        if !summary.top_expensive_interventions.is_empty() {
            println!("\n=== High-Cost Friction Windows ===");
            println!("  Cause  | Intensity | Next 5 Cost | Top Symbol");
            println!("---------|-----------|-------------|----------------");
            for intervention in &summary.top_expensive_interventions {
                let sym = intervention
                    .symbols
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("-");
                println!(
                    "  {:>6} | {:>9.2} | ${:>10.4} | {}",
                    intervention.cause,
                    intervention.intensity,
                    intervention.cost_next_5,
                    if sym.len() > 20 {
                        format!("{}...", &sym[..17])
                    } else {
                        sym.to_string()
                    }
                );
            }
        }

        if !summary.friction_by_symbol.is_empty() {
            println!("\n=== Top Friction Files ===");
            let mut top: Vec<_> = summary.friction_by_symbol.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            for (sym, count) in top.iter().take(5) {
                println!("  {:32}: {} warnings", sym, count);
            }
        }

        if !summary.channel_contributions.is_empty() {
            println!("\n=== Trigger Drivers (Symptom Breakdown) ===");
            let total_contrib: f32 = summary.channel_contributions.values().sum();
            let mut top: Vec<_> = summary.channel_contributions.iter().collect();
            top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (chan, val) in top.iter().take(5) {
                println!("  {:32}: {:.1}%", chan, (*val / total_contrib) * 100.0);
            }
        }
    }
    println!();

    println!("=== User Engagement ===");
    println!("recall commands:  {}", summary.recall_commands);
    println!("query commands:   {}", summary.query_commands);

    Ok(())
}
