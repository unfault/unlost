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
        AgentCommand::Opencode { path, server } => {
            let root =
                crate::workspace::git_toplevel(std::path::Path::new(&path)).unwrap_or_else(|| {
                    crate::workspace::canonicalize_dir(std::path::Path::new(&path))
                        .unwrap_or_else(|_| std::path::PathBuf::from(&path))
                });
            let root = crate::workspace::canonicalize_dir(&root)?;

            let (workspace_id, _src) = crate::workspace::compute_workspace_id(&root)
                .ok_or_else(|| anyhow::anyhow!("unable to compute workspace id"))?;

            let server = server.trim_end_matches('/');
            let base_url = format!("{server}/w/{workspace_id}/opencode/zen/v1");

            let cfg_path = root.join("opencode.json");
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
            set_nested_string(
                &mut json,
                &["provider", "opencode", "options", "baseURL"],
                base_url,
            );

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;
            println!("configured: {}", cfg_path.display());
        }
    }
    Ok(())
}

pub(crate) fn run(command: ConfigCommand) -> anyhow::Result<()> {
    match command {
        ConfigCommand::Llm { command } => handle_llm_command(command),
        ConfigCommand::Agent { command } => handle_agent_command(command),
    }
}
