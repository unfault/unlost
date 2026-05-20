use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

pub async fn run(
    text: Vec<String>,
    source_label: Option<String>,
    global: bool,
    stdin: bool,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let note_text = if stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        let trimmed = buf.trim().to_string();
        if trimmed.is_empty() {
            anyhow::bail!("no input on stdin");
        }
        trimmed
    } else if !text.is_empty() {
        text.join(" ")
    } else {
        anyhow::bail!("no note text provided (pass text as argument or use --stdin)");
    };

    let note_text = wrap_plain_text(&note_text, 80);

    let label = source_label.as_deref().unwrap_or("note");

    let ws_root = resolve_workspace_root(global)?;
    let ws = crate::workspace::get_or_create_workspace_paths(&ws_root)?;

    let embedder = crate::embed::load_embedder(
        &embed_model,
        embed_cache_dir.as_deref().map(std::path::PathBuf::from),
        true,
    )
    .await?;

    std::fs::create_dir_all(&ws.db_dir)?;
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let _ = crate::storage::ensure_capsules_table(&db).await?;

    let symbols = crate::net::extract_symbols_from_text(&note_text);
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let capsule = crate::IntentCapsule {
        category: format!("Note:{label}"),
        intent: "Manual note".to_string(),
        decision: note_text.clone(),
        rationale: String::new(),
        next_steps: Vec::new(),
        symbols,
        user_symbols: vec![],
        failure_mode: crate::types::FailureMode::None,
        failure_signals: None,
        extraction_mode: crate::types::ExtractionMode::None,
        questions: vec![],
    };

    let pointer = ws.root.to_str().map(|p| format!(
        "note+local://{p}#{ts_ms}"
    ));
    let meta = crate::ResponseMeta {
        source: "note".to_string(),
        upstream_host: "note".to_string(),
        request_path: label.to_string(),
        http_status: 0,
        agent_session_id: None,
        source_pointer: pointer,
        usage: None,
    };

    crate::storage::insert_capsule_row(
        &db, &embedder, 0, 0, ts_ms, &meta, None, None, &capsule,
        &crate::types::TurnEval::default(), Some(&note_text), None, None,
    )
    .await?;

    let _ = crate::recording::append_capsule_jsonl(
        &ws.capsules_jsonl, ts_ms, 0, 0, &meta, &capsule, None, None,
    );

    let ws_label = crate::workspace::workspace_label_by_id(&ws.id)
        .unwrap_or_else(|| ws.id.clone());

    let symbol_count = capsule.symbols.len();
    let ts_fmt = chrono_now_ms(ts_ms);
    if symbol_count > 0 {
        eprintln!("\x1b[2mnote recorded \x1b[0m· \x1b[36m{ws_label}\x1b[0m \x1b[2m· {symbol_count} symbols \x1b[0m· \x1b[32mqueryable now\x1b[0m");
    } else {
        eprintln!("\x1b[2mnote recorded \x1b[0m· \x1b[36m{ws_label}\x1b[0m \x1b[2m· {ts_fmt}\x1b[0m");
    }

    Ok(())
}

fn resolve_workspace_root(global: bool) -> anyhow::Result<std::path::PathBuf> {
    if global {
        let root = crate::workspace::unlost_data_root().join("global");
        std::fs::create_dir_all(&root)?;
        return Ok(root);
    }

    let cwd = std::env::current_dir()?;
    if let Some(repo_root) = crate::workspace::git_toplevel(&cwd) {
        return Ok(repo_root);
    }

    let has_manifest = ["Cargo.toml", "pyproject.toml", "package.json", "go.mod"]
        .iter()
        .any(|name| cwd.join(name).exists());
    if has_manifest {
        return Ok(cwd);
    }

    let root = crate::workspace::unlost_data_root().join("global");
    std::fs::create_dir_all(&root)?;
    Ok(root)
}

fn chrono_now_ms(ts_ms: i64) -> String {
    let secs = ts_ms / 1000;
    let nsecs = ((ts_ms % 1000) * 1_000_000) as u32;
    match chrono::DateTime::from_timestamp(secs, nsecs) {
        Some(dt) => dt.format("%Y-%m-%d %H:%M").to_string(),
        None => "unknown".to_string(),
    }
}

fn wrap_plain_text(s: &str, max_width: usize) -> String {
    if max_width == 0 {
        return s.to_string();
    }
    let mut lines: Vec<String> = Vec::new();
    for line in s.lines() {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() <= max_width {
            lines.push(line.to_string());
            continue;
        }
        let mut start = 0;
        while start < chars.len() {
            if chars.len() - start <= max_width {
                lines.push(chars[start..].iter().collect());
                break;
            }
            let end = start + max_width;
            let split = chars[start..end]
                .iter()
                .rposition(|c| c.is_whitespace())
                .map(|pos| start + pos)
                .unwrap_or(end);
            lines.push(chars[start..split].iter().collect());
            start = split;
            while start < chars.len() && chars[start].is_whitespace() {
                start += 1;
            }
        }
    }
    if lines.len() == 1 {
        return s.to_string();
    }
    lines.join("\n")
}
