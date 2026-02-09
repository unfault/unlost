pub(crate) fn run(path: String) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;
    let summary = crate::metrics::summarize_metrics(&ws.metrics_jsonl)?;

    println!("workspace: {}", ws.id);
    println!("metrics:   {}", ws.metrics_jsonl.display());
    println!();
    println!("=== Overview ===");
    println!("capsules:  {}", summary.capsules);
    if summary.tokens_total > 0 {
        println!("tokens:    {}", summary.tokens_total);
    }
    if summary.cost_total > 0.0 {
        println!("cost:      ${:.4}", summary.cost_total);
    }
    println!();

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
    println!();

    println!("=== User Engagement ===");
    println!("recall commands:  {}", summary.recall_commands);
    println!("query commands:   {}", summary.query_commands);

    Ok(())
}
