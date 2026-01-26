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
