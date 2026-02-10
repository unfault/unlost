use crate::cli::{AgentCommand, ConfigCommand, LlmCommand};
use crate::config::LlmConfig;

fn ensure_object(v: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    if !v.is_object() {
        *v = serde_json::Value::Object(serde_json::Map::new());
    }
    v.as_object_mut().unwrap()
}

fn set_nested_string(root: &mut serde_json::Value, path: &[&str], value: String) {
    let mut cur = root;
    for (i, key) in path.iter().enumerate() {
        let is_last = i + 1 == path.len();
        let obj = ensure_object(cur);
        if is_last {
            obj.insert((*key).to_string(), serde_json::Value::String(value));
            return;
        }
        cur = obj
            .entry((*key).to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    }
}

fn ensure_top_level_string_array_contains(root: &mut serde_json::Value, key: &str, value: &str) {
    let obj = ensure_object(root);
    let entry = obj
        .entry(key.to_string())
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));

    if !entry.is_array() {
        *entry = serde_json::Value::Array(Vec::new());
    }
    let arr = entry.as_array_mut().unwrap();
    let already = arr.iter().any(|v| v.as_str() == Some(value));
    if !already {
        arr.push(serde_json::Value::String(value.to_string()));
    }
}

fn handle_llm_command(cmd: LlmCommand) -> anyhow::Result<()> {
    match cmd {
        LlmCommand::Openai {
            api_key,
            base_url,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Openai {
                api_key,
                base_url,
                model,
            }))?;
            println!("LLM provider set to OpenAI");
        }
        LlmCommand::Anthropic {
            api_key,
            base_url,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Anthropic {
                api_key,
                base_url,
                model,
            }))?;
            println!("LLM provider set to Anthropic");
        }
        LlmCommand::Ollama { base_url, model } => {
            crate::llm::set_llm_config(Some(LlmConfig::Ollama { base_url, model }))?;
            println!("LLM provider set to Ollama");
        }
        LlmCommand::Custom {
            base_url,
            api_key,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Custom {
                base_url,
                api_key,
                model,
            }))?;
            println!("LLM provider set to custom endpoint");
        }
        LlmCommand::Show => {
            crate::llm::show_llm_config();
        }
        LlmCommand::Remove => {
            crate::llm::set_llm_config(None)?;
            println!("LLM configuration removed");
        }
    }
    Ok(())
}

fn handle_agent_command(cmd: AgentCommand) -> anyhow::Result<()> {
    match cmd {
        AgentCommand::Opencode {
            path,
            plugin,
            global,
        } => {
            let cfg_path = if global {
                opencode_global_config_path()?
            } else {
                let root = crate::workspace::git_toplevel(std::path::Path::new(&path))
                    .unwrap_or_else(|| {
                        crate::workspace::canonicalize_dir(std::path::Path::new(&path))
                            .unwrap_or_else(|_| std::path::PathBuf::from(&path))
                    });
                let root = crate::workspace::canonicalize_dir(&root)?;
                root.join("opencode.json")
            };

            if let Some(parent) = cfg_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut json = match std::fs::read_to_string(&cfg_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => return Err(e.into()),
            };

            set_nested_string(
                &mut json,
                &["$schema"],
                "https://opencode.ai/config.json".to_string(),
            );
            ensure_top_level_string_array_contains(&mut json, "plugin", &plugin);

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;

            let scope = if global { "global" } else { "project" };
            println!("configured ({scope}): {}", cfg_path.display());
        }

        AgentCommand::Claude { path, global } => {
            configure_claude(&path, global)?;
        }
    }
    Ok(())
}

fn opencode_global_config_path() -> anyhow::Result<std::path::PathBuf> {
    let dir = if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        std::path::PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
            .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
        home.join(".config")
    };

    Ok(dir.join("opencode").join("opencode.json"))
}

fn configure_claude(path: &str, global: bool) -> anyhow::Result<()> {
    let cfg_path = if global {
        // Global: ~/.claude/settings.json
        let home = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
            .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
        home.join(".claude").join("settings.json")
    } else {
        // Per-project: <project>/.claude/settings.json
        let root =
            crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
                crate::workspace::canonicalize_dir(std::path::Path::new(path))
                    .unwrap_or_else(|_| std::path::PathBuf::from(path))
            });
        let root = crate::workspace::canonicalize_dir(&root)?;
        root.join(".claude").join("settings.json")
    };

    // Ensure parent directory exists
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load existing config or start fresh
    let mut json = match std::fs::read_to_string(&cfg_path) {
        Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            serde_json::Value::Object(serde_json::Map::new())
        }
        Err(e) => return Err(e.into()),
    };

    // Build the hooks config
    let unlost_hook = serde_json::json!({
        "type": "command",
        "command": "unlost shim claude"
    });

    let unlost_hook_async = serde_json::json!({
        "type": "command",
        "command": "unlost shim claude",
        "async": true
    });

    // Ensure hooks object exists
    let obj = ensure_object(&mut json);
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let hooks_obj = ensure_object(hooks);

    // Add UserPromptSubmit hook
    let user_prompt_submit = hooks_obj
        .entry("UserPromptSubmit")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    add_unlost_hook_if_missing(user_prompt_submit, unlost_hook);

    // Add Stop hook
    let stop = hooks_obj
        .entry("Stop")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    add_unlost_hook_if_missing(stop, unlost_hook_async);

    // Migrate any legacy hook command strings in-place.
    if let Some(v) = hooks_obj.get_mut("UserPromptSubmit") {
        rewrite_unlost_hook_commands(v);
    }
    if let Some(v) = hooks_obj.get_mut("Stop") {
        rewrite_unlost_hook_commands(v);
    }

    // Write back
    let rendered = serde_json::to_string_pretty(&json)?;
    std::fs::write(&cfg_path, rendered)?;

    let scope = if global { "global" } else { "project" };
    println!("configured ({scope}): {}", cfg_path.display());
    Ok(())
}

/// Add an unlost hook to a hook array if not already present
fn add_unlost_hook_if_missing(hook_array: &mut serde_json::Value, hook: serde_json::Value) {
    if !hook_array.is_array() {
        *hook_array = serde_json::Value::Array(Vec::new());
    }
    let arr = hook_array.as_array_mut().unwrap();

    // Check if unlost hook already exists
    let has_unlost = arr.iter().any(|entry| {
        if let Some(hooks) = entry.get("hooks").and_then(|h| h.as_array()) {
            hooks.iter().any(|h| {
                h.get("command")
                    .and_then(|c| c.as_str())
                    .map(|c| {
                        c.contains("unlost shim claude") || c.contains("unlost shim claudecode")
                    })
                    .unwrap_or(false)
            })
        } else {
            false
        }
    });

    if !has_unlost {
        arr.push(serde_json::json!({
            "hooks": [hook]
        }));
    }
}

fn rewrite_unlost_hook_commands(hook_array: &mut serde_json::Value) {
    let Some(arr) = hook_array.as_array_mut() else {
        return;
    };
    for entry in arr.iter_mut() {
        let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) else {
            continue;
        };
        for h in hooks.iter_mut() {
            let Some(cmd) = h.get_mut("command") else {
                continue;
            };
            let Some(s) = cmd.as_str() else {
                continue;
            };
            if s.trim() == "unlost shim claudecode" {
                *cmd = serde_json::Value::String("unlost shim claude".to_string());
            }
        }
    }
}

pub(crate) fn run(command: ConfigCommand) -> anyhow::Result<()> {
    match command {
        ConfigCommand::Llm { command } => handle_llm_command(command),
        ConfigCommand::Agent { command } => handle_agent_command(command),
    }
}
