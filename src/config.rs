use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkspaceConfig {
    pub(crate) version: u32,
    // Map canonical workspace root -> workspace_id
    pub(crate) path_index: std::collections::BTreeMap<String, String>,
    // Map workspace_id -> info
    pub(crate) workspaces: std::collections::BTreeMap<String, WorkspaceInfo>,

    #[serde(default)]
    pub(crate) llm: Option<LlmConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub(crate) enum LlmConfig {
    Openai {
        api_key: String,
        #[serde(default)]
        base_url: Option<String>,
        model: String,
    },
    Anthropic {
        api_key: String,
        #[serde(default)]
        base_url: Option<String>,
        model: String,
    },
    Ollama {
        #[serde(default = "default_ollama_base_url")]
        base_url: String,
        model: String,
    },
    Custom {
        base_url: String,
        #[serde(default)]
        api_key: Option<String>,
        model: String,
    },
}

fn default_ollama_base_url() -> String {
    "http://127.0.0.1:11434/v1".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkspaceInfo {
    pub(crate) id: String,
    pub(crate) root: String,
    pub(crate) source: String,
    pub(crate) db_dir: String,
    pub(crate) capsules_jsonl: String,
    pub(crate) created_ts_ms: i64,
    pub(crate) updated_ts_ms: i64,
}
