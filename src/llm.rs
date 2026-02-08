use crate::config::LlmConfig;
use anyhow::Context;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::{anthropic, openai};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub(crate) fn get_llm_config() -> Option<LlmConfig> {
    crate::workspace::load_workspace_config().llm
}

pub(crate) fn set_llm_config(new_cfg: Option<LlmConfig>) -> anyhow::Result<()> {
    let mut cfg = crate::workspace::load_workspace_config();
    cfg.llm = new_cfg;
    crate::workspace::save_workspace_config(&cfg)
}

pub(crate) fn show_llm_config() {
    let cfg = get_llm_config();
    match cfg {
        None => {
            println!("LLM: not configured");
        }
        Some(LlmConfig::Openai { base_url, model, .. }) => {
            println!("LLM: openai");
            println!("model: {model}");
            if let Some(b) = base_url {
                println!("base_url: {b}");
            }
        }
        Some(LlmConfig::Anthropic { base_url, model, .. }) => {
            println!("LLM: anthropic");
            println!("model: {model}");
            if let Some(b) = base_url {
                println!("base_url: {b}");
            }
        }
        Some(LlmConfig::Ollama { base_url, model }) => {
            println!("LLM: ollama");
            println!("model: {model}");
            println!("base_url: {base_url}");
        }
        Some(LlmConfig::Custom { base_url, model, .. }) => {
            println!("LLM: custom");
            println!("model: {model}");
            println!("base_url: {base_url}");
        }
    }
}

pub(crate) async fn llm_extract<T>(
    model_override: Option<&str>,
    preamble: &str,
    input: &str,
) -> anyhow::Result<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    let cfg = get_llm_config();

    match cfg {
        Some(LlmConfig::Openai {
            api_key,
            base_url,
            model,
        }) => {
            let model = model_override.unwrap_or(&model);
            let mut builder: openai::ClientBuilder<reqwest::Client> =
                openai::Client::builder().api_key(&api_key);
            if let Some(base) = base_url.as_deref() {
                builder = builder.base_url(base);
            } else {
                // Avoid accidentally routing the extractor through unlost itself if the user set
                // OPENAI_BASE_URL/OPENAI_API_BASE in their shell for other tools.
                builder = builder.base_url("https://api.openai.com/v1");
            }
            let client = builder.build().context("failed to build OpenAI client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Anthropic {
            api_key,
            base_url,
            model,
        }) => {
            let model = model_override.unwrap_or(&model);
            let mut builder: anthropic::ClientBuilder<reqwest::Client> =
                anthropic::Client::builder().api_key(api_key);
            if let Some(base) = base_url.as_deref() {
                builder = builder.base_url(base);
            } else {
                // Same idea as OpenAI: keep extractor traffic off the local recorder.
                builder = builder.base_url("https://api.anthropic.com");
            }
            let client = builder.build().context("failed to build Anthropic client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Ollama { base_url, model }) => {
            // Ollama provides an OpenAI-compatible endpoint. Use a dummy key.
            let model = model_override.unwrap_or(&model);
            let mut builder: openai::ClientBuilder<reqwest::Client> =
                openai::Client::builder().api_key("ollama");
            builder = builder.base_url(&base_url);
            let client = builder
                .build()
                .context("failed to build Ollama (OpenAI-compatible) client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        Some(LlmConfig::Custom {
            base_url,
            api_key,
            model,
        }) => {
            let model = model_override.unwrap_or(&model);
            let key = api_key.as_deref().unwrap_or("custom");
            let mut builder: openai::ClientBuilder<reqwest::Client> = openai::Client::builder().api_key(key);
            builder = builder.base_url(&base_url);
            let client = builder
                .build()
                .context("failed to build custom OpenAI-compatible client")?;
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
        None => {
            // Default: OpenAI from env.
            let model = model_override.unwrap_or("gpt-4o-mini");
            let client = openai::Client::from_env();
            Ok(client
                .extractor::<T>(model)
                .preamble(preamble)
                .build()
                .extract(input)
                .await?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::test_support::ENV_LOCK;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, val: &std::ffi::OsStr) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn test_get_llm_config_none() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        let cfg = get_llm_config();
        assert!(cfg.is_none());
    }

    #[test]
    fn test_set_and_get_llm_config() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        let openai_cfg = LlmConfig::Openai {
            api_key: "sk-test-key".to_string(),
            base_url: Some("https://api.example.com".to_string()),
            model: "gpt-4".to_string(),
        };

        set_llm_config(Some(openai_cfg.clone())).unwrap();

        let retrieved = get_llm_config();
        match retrieved {
            Some(LlmConfig::Openai { api_key, base_url, model }) => {
                assert_eq!(api_key, "sk-test-key");
                assert_eq!(base_url, Some("https://api.example.com".to_string()));
                assert_eq!(model, "gpt-4");
            }
            _ => panic!("Expected OpenAI config"),
        }
    }

    #[test]
    fn test_set_llm_config_none() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        set_llm_config(None).unwrap();
        let cfg = get_llm_config();
        assert!(cfg.is_none());
    }

    #[test]
    fn test_anthropic_config_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        let anthropic_cfg = LlmConfig::Anthropic {
            api_key: "sk-ant-test".to_string(),
            base_url: None,
            model: "claude-3-sonnet".to_string(),
        };

        set_llm_config(Some(anthropic_cfg.clone())).unwrap();

        let retrieved = get_llm_config();
        match retrieved {
            Some(LlmConfig::Anthropic { api_key, base_url, model }) => {
                assert_eq!(api_key, "sk-ant-test");
                assert_eq!(base_url, None);
                assert_eq!(model, "claude-3-sonnet");
            }
            _ => panic!("Expected Anthropic config"),
        }
    }

    #[test]
    fn test_ollama_config_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        let ollama_cfg = LlmConfig::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
        };

        set_llm_config(Some(ollama_cfg.clone())).unwrap();

        let retrieved = get_llm_config();
        match retrieved {
            Some(LlmConfig::Ollama { base_url, model }) => {
                assert_eq!(base_url, "http://localhost:11434");
                assert_eq!(model, "llama3");
            }
            _ => panic!("Expected Ollama config"),
        }
    }

    #[test]
    fn test_custom_config_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        let custom_cfg = LlmConfig::Custom {
            base_url: "https://custom.api.com/v1".to_string(),
            api_key: Some("custom-key-123".to_string()),
            model: "custom-model".to_string(),
        };

        set_llm_config(Some(custom_cfg.clone())).unwrap();

        let retrieved = get_llm_config();
        match retrieved {
            Some(LlmConfig::Custom {
                base_url,
                api_key,
                model,
            }) => {
                assert_eq!(base_url, "https://custom.api.com/v1");
                assert_eq!(api_key, Some("custom-key-123".to_string()));
                assert_eq!(model, "custom-model");
            }
            _ => panic!("Expected Custom config"),
        }
    }
}
