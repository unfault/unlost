//! `unlost ingest <file.md>` — chunk a markdown document into capsules.
//!
//! Each `##` / `###` heading becomes a capsule with:
//!   - `category: cartography` (or `cartography:<tag>` from frontmatter)
//!   - `intent`: the heading text
//!   - `decision`: the section body
//!   - `symbols`: code-fence identifiers + [[wiki-links]] + frontmatter symbols
//!
//! YAML frontmatter (delimited by `---`) is parsed for:
//!   - `target`: used as the intent if no heading is present
//!   - `target_path`: added to symbols
//!   - `symbols`: extra symbols list
//!   - `related`: added to symbols
//!   - `category`: overrides the default `cartography` category tag
//!
//! Capsules are inserted into the current workspace's LanceDB store and
//! appended to `capsules.jsonl`, exactly like `unlost note`.

use std::time::{SystemTime, UNIX_EPOCH};

/// A section extracted from a markdown document.
#[derive(Debug)]
struct Section {
    /// Heading text (empty string for preamble before first heading)
    heading: String,
    /// Section body text
    body: String,
    /// Heading level (2 for `##`, 3 for `###`, 0 for preamble)
    level: usize,
}

/// Parsed frontmatter from a markdown file.
#[derive(Debug, Default)]
struct Frontmatter {
    target: Option<String>,
    target_path: Option<String>,
    symbols: Vec<String>,
    related: Vec<String>,
    category_tag: Option<String>,
}

/// Parse YAML-like frontmatter from the top of a markdown string.
/// Returns (frontmatter, rest_of_content).
fn parse_frontmatter(content: &str) -> (Frontmatter, &str) {
    let content = content.trim_start();
    if !content.starts_with("---") {
        return (Frontmatter::default(), content);
    }
    let after_open = &content[3..];
    // Skip optional newline after opening ---
    let after_open = after_open.trim_start_matches('\n').trim_start_matches('\r');
    let close = match after_open.find("\n---") {
        Some(p) => p,
        None => return (Frontmatter::default(), content),
    };
    let frontmatter_text = &after_open[..close];
    let rest = &after_open[close + 4..]; // skip \n---

    let mut fm = Frontmatter::default();
    for line in frontmatter_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim().trim_matches('"').trim_matches('\'');
            match key {
                "target" => fm.target = Some(val.to_string()),
                "target_path" => fm.target_path = Some(val.to_string()),
                "category" => fm.category_tag = Some(val.to_string()),
                "symbols" | "related" => {
                    // Accept "key: [a, b, c]" or "key: a" (single value)
                    let vals = parse_yaml_list(val);
                    if key == "symbols" {
                        fm.symbols.extend(vals);
                    } else {
                        fm.related.extend(vals);
                    }
                }
                _ => {}
            }
        }
    }

    (fm, rest)
}

/// Parse a YAML inline list `[a, b, c]` or a bare value `a`.
fn parse_yaml_list(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        inner
            .split(',')
            .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|v| !v.is_empty())
            .collect()
    } else if !s.is_empty() {
        vec![s.trim_matches('"').trim_matches('\'').to_string()]
    } else {
        vec![]
    }
}

/// Split markdown content (with frontmatter already stripped) into sections.
/// Splits on `##` and `###` headings. Content before the first heading
/// is returned as a preamble section with an empty heading.
fn split_sections(content: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current_heading = String::new();
    let mut current_level = 0usize;
    let mut current_body = String::new();

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("### ") {
            if !current_body.trim().is_empty() || !current_heading.is_empty() {
                sections.push(Section {
                    heading: current_heading.clone(),
                    body: current_body.trim().to_string(),
                    level: current_level,
                });
            }
            current_heading = rest.trim().to_string();
            current_level = 3;
            current_body = String::new();
        } else if let Some(rest) = line.strip_prefix("## ") {
            if !current_body.trim().is_empty() || !current_heading.is_empty() {
                sections.push(Section {
                    heading: current_heading.clone(),
                    body: current_body.trim().to_string(),
                    level: current_level,
                });
            }
            current_heading = rest.trim().to_string();
            current_level = 2;
            current_body = String::new();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    // Flush final section
    if !current_body.trim().is_empty() || !current_heading.is_empty() {
        sections.push(Section {
            heading: current_heading,
            body: current_body.trim().to_string(),
            level: current_level,
        });
    }

    sections
}

/// Extract symbols from markdown text:
/// - Code-fence language identifiers: ```rust, ```python, etc.
/// - Inline backtick tokens that look like identifiers: `MyStruct`, `fn_name`
/// - [[wiki-links]]
fn extract_md_symbols(text: &str) -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();

    // [[wiki-links]]
    let mut rest = text;
    while let Some(start) = rest.find("[[") {
        let after = &rest[start + 2..];
        if let Some(end) = after.find("]]") {
            let link = after[..end].trim();
            if !link.is_empty() {
                symbols.push(link.to_string());
            }
            rest = &after[end + 2..];
        } else {
            break;
        }
    }

    // Code-fence language tags: ```rust\n or ```python\n
    for line in text.lines() {
        let l = line.trim();
        if let Some(lang) = l.strip_prefix("```") {
            let lang = lang.trim();
            if !lang.is_empty() && lang.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                symbols.push(lang.to_string());
            }
        }
    }

    // Inline backtick identifiers: `SomeName` or `some_fn`
    let mut s = text;
    while let Some(start) = s.find('`') {
        let after = &s[start + 1..];
        if after.starts_with('`') {
            // Skip `` double-backtick sequences
            s = &after[1..];
            continue;
        }
        if let Some(end) = after.find('`') {
            let token = &after[..end];
            // Only keep identifiers: CamelCase, snake_case, paths with ::
            let looks_like_ident = !token.is_empty()
                && token.len() <= 64
                && token.chars().all(|c| c.is_alphanumeric() || c == '_' || c == ':' || c == '.')
                && token.chars().next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false);
            if looks_like_ident {
                symbols.push(token.to_string());
            }
            s = &after[end + 1..];
        } else {
            break;
        }
    }

    // Dedup while preserving order
    let mut seen = std::collections::HashSet::new();
    symbols.retain(|s| seen.insert(s.clone()));

    symbols
}

pub async fn run(
    paths: Vec<String>,
    category_override: Option<String>,
    global: bool,
    embed_model: String,
    embed_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    if paths.is_empty() {
        println!("Ingest markdown documents into workspace memory.\n");
        println!("Usage: unlost ingest <file.md> [<file2.md> ...]");
        println!("\nOptions:");
        println!("  --category <tag>   Override the category tag (default: cartography)");
        println!("  --global           Ingest into the global workspace");
        println!("\nEach ## / ### heading becomes a capsule. Frontmatter fields:");
        println!("  target, target_path, symbols, related, category");
        return Ok(());
    }

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

    let ws_label = crate::workspace::workspace_label_by_id(&ws.id)
        .unwrap_or_else(|| ws.id.clone());

    let mut total_capsules = 0usize;

    for path in &paths {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("unlost ingest: cannot read {path}: {e}");
                continue;
            }
        };

        let file_name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());

        let (fm, body) = parse_frontmatter(&content);
        let sections = split_sections(body);

        // Base category: prefer --category flag, then frontmatter, then "cartography"
        let base_category = category_override
            .as_deref()
            .or(fm.category_tag.as_deref())
            .unwrap_or("cartography");

        // Symbols that apply to the whole document (from frontmatter)
        let mut doc_symbols: Vec<String> = Vec::new();
        if let Some(ref tp) = fm.target_path {
            doc_symbols.push(tp.clone());
        }
        doc_symbols.extend(fm.symbols.iter().cloned());
        doc_symbols.extend(fm.related.iter().cloned());

        // Also extract symbols from the full document text for better retrieval
        let doc_wide_syms = extract_md_symbols(&content);
        for s in &doc_wide_syms {
            if !doc_symbols.contains(s) {
                doc_symbols.push(s.clone());
            }
        }

        let doc_target = fm.target.as_deref().unwrap_or(&file_name);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let non_empty_sections: Vec<&Section> = sections
            .iter()
            .filter(|s| !s.body.trim().is_empty() || !s.heading.is_empty())
            .collect();

        if non_empty_sections.is_empty() {
            eprintln!("unlost ingest: {file_name}: no sections found, skipping");
            continue;
        }

        for (i, section) in non_empty_sections.iter().enumerate() {
            // Space capsule timestamps 1ms apart to preserve section order
            let ts_ms = now_ms + i as i64;

            let intent = if section.heading.is_empty() {
                // Preamble: use the document target as the intent
                format!("{doc_target} (overview)")
            } else {
                section.heading.clone()
            };

            // Combine doc-level symbols with section-local symbols
            let mut section_symbols = doc_symbols.clone();
            let local_syms = extract_md_symbols(&section.body);
            for s in &local_syms {
                if !section_symbols.contains(s) {
                    section_symbols.push(s.clone());
                }
            }
            // Also add the heading itself as a symbol for direct lookup
            if !section.heading.is_empty() {
                section_symbols.push(section.heading.clone());
            }

            // Category: cartography (or override), optionally with level suffix
            let category = if section.level == 3 {
                format!("{base_category}:sub")
            } else {
                base_category.to_string()
            };

            let capsule = crate::IntentCapsule {
                category,
                intent: intent.clone(),
                decision: section.body.clone(),
                rationale: String::new(),
                next_steps: Vec::new(),
                symbols: section_symbols,
                user_symbols: vec![],
                failure_mode: crate::types::FailureMode::None,
                failure_signals: None,
                extraction_mode: crate::types::ExtractionMode::None,
                questions: vec![],
            };

            let source_pointer = ws.root.to_str().map(|p| {
                format!("ingest+local://{p}/{file_name}#section={i}")
            });

            let meta = crate::ResponseMeta {
                source: "ingest".to_string(),
                upstream_host: "ingest".to_string(),
                request_path: file_name.clone(),
                http_status: 0,
                agent_session_id: None,
                source_pointer,
                usage: None,
            };

            let prior = if i > 0 {
                non_empty_sections.get(i - 1).map(|s| s.body.as_str())
            } else {
                None
            };

            crate::storage::insert_capsule_row(
                &db, &embedder, 0, i as u64, ts_ms, &meta,
                None, None, &capsule,
                &crate::types::TurnEval::default(),
                prior, None, None,
            )
            .await?;

            let _ = crate::recording::append_capsule_jsonl(
                &ws.capsules_jsonl, ts_ms, 0, i as u64, &meta, &capsule, None, None,
            );

            total_capsules += 1;
        }

        eprintln!(
            "\x1b[2mingest\x1b[0m · \x1b[36m{ws_label}\x1b[0m · \x1b[1m{file_name}\x1b[0m → {} capsules",
            non_empty_sections.len()
        );
    }

    eprintln!(
        "\x1b[2m{total_capsules} capsule{} ingested\x1b[0m · queryable now",
        if total_capsules == 1 { "" } else { "s" }
    );

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
