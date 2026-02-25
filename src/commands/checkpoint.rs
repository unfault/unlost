//! `unlost checkpoint` — manually create or list checkpoints.
//!
//! Examples:
//!   unlost checkpoint               # generate a checkpoint for the current workspace now
//!   unlost checkpoint --list        # list recent checkpoints
//!   unlost checkpoint --list --since 7d

use chrono::TimeZone;

pub async fn run(
    list: bool,
    session_id: Option<String>,
    since: Option<String>,
    llm_model: Option<String>,
) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()?;
    let ws = crate::workspace::get_or_create_workspace_paths(&cwd)?;

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    if list {
        // ── List mode ──────────────────────────────────────────────────────────
        let since_ms = match since {
            Some(ref s) => crate::util::parse_time_filter(s)?,
            None => None,
        };

        let checkpoints =
            crate::storage_checkpoint::get_recent_checkpoints(&db, &ws.id, 50).await?;

        let filtered: Vec<_> = checkpoints
            .into_iter()
            .filter(|c| {
                if let Some(since) = since_ms {
                    c.ts_ms >= since
                } else {
                    true
                }
            })
            .collect();

        if filtered.is_empty() {
            println!("No checkpoints found for this workspace.");
            println!("Run `unlost checkpoint` after a coding session to create one.");
            return Ok(());
        }

        println!("Checkpoints for workspace {} ({} found):\n", ws.id, filtered.len());
        for cp in &filtered {
            let ts_str = chrono::Utc
                .timestamp_millis_opt(cp.ts_ms)
                .single()
                .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| cp.ts_ms.to_string());
            let from_str = chrono::Utc
                .timestamp_millis_opt(cp.from_ts_ms)
                .single()
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| cp.from_ts_ms.to_string());
            let to_str = chrono::Utc
                .timestamp_millis_opt(cp.to_ts_ms)
                .single()
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| cp.to_ts_ms.to_string());

            println!("  [{}]  created: {}", &cp.id[..8], ts_str);
            println!(
                "    session:  {}",
                cp.session_id.as_deref().unwrap_or("(all)")
            );
            println!("    window:   {} → {}", from_str, to_str);
            println!(
                "    capsules: {}  |  model: {}  |  trigger: {}",
                cp.capsule_count, cp.model_used, cp.trigger
            );
            println!();
            // Preview first 200 chars of narrative
            let preview: String = cp.narrative.chars().take(200).collect();
            let preview = if cp.narrative.len() > 200 {
                format!("{}…", preview)
            } else {
                preview
            };
            println!("    {}", preview.replace('\n', "\n    "));
            println!();
        }
    } else {
        // ── Generate mode ──────────────────────────────────────────────────────
        eprintln!("unlost: generating checkpoint for workspace {}…", ws.id);

        match crate::storage_checkpoint::maybe_create_checkpoint(
            &ws,
            session_id.as_deref(),
            "manual",
            llm_model.as_deref(),
        )
        .await?
        {
            Ok(cp) => {
                let ts_str = chrono::Utc
                    .timestamp_millis_opt(cp.ts_ms)
                    .single()
                    .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                    .unwrap_or_else(|| cp.ts_ms.to_string());
                println!("Checkpoint created [{}] at {}", &cp.id[..8], ts_str);
                println!("  session:  {}", cp.session_id.as_deref().unwrap_or("(all)"));
                println!("  capsules: {}  |  model: {}", cp.capsule_count, cp.model_used);
                println!();
                println!(
                    "{}",
                    crate::narrative::render_narrative(crate::cli::OutputFormat::Ansi, &cp.narrative)
                );
            }
            Err(reason) => {
                use crate::storage_checkpoint::CheckpointSkipReason::*;
                match reason {
                    TooFewCapsules { found, minimum } => {
                        println!(
                            "Not enough capsules to checkpoint: found {found}, need at least {minimum}."
                        );
                        println!("Record more sessions and try again.");
                    }
                    AlreadyCurrent => {
                        // Manual trigger bypasses this guard, so this branch should
                        // not be reachable — but handle it gracefully anyway.
                        println!("A current checkpoint already covers this workspace.");
                        println!("Run `unlost checkpoint --list` to see it.");
                    }
                    NoConversationalCapsules => {
                        println!("No conversational capsules found — only git capsules present.");
                        println!("A checkpoint requires at least some recorded conversation turns.");
                    }
                }
            }
        }
    }

    Ok(())
}
