pub(crate) async fn run(
    path: String,
    limit: usize,
    filter: Option<String>,
) -> anyhow::Result<()> {
    let ws = crate::workspace::get_or_create_workspace_paths(std::path::Path::new(&path))?;
    match crate::storage::scan_capsules_lancedb(&ws, limit, filter.as_deref()).await {
        Ok(rows) if !rows.is_empty() => {
            println!("workspace: {}", ws.id);
            for hit in rows {
                let cap = hit.capsule;
                let meta = hit.meta;
                println!("---");
                println!("chunked_at: {}", hit.ts_ms);
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
