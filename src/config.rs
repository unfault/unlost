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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ollama_base_url() {
        assert_eq!(default_ollama_base_url(), "http://127.0.0.1:11434/v1");
    }

    #[test]
    fn test_llm_config_serialization() {
        // Test OpenAI config
        let openai = LlmConfig::Openai {
            api_key: "sk-test123".to_string(),
            base_url: Some("https://api.openai.com".to_string()),
            model: "gpt-4".to_string(),
        };
        let json = serde_json::to_string(&openai).unwrap();
        let parsed: LlmConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            LlmConfig::Openai {
                api_key,
                base_url,
                model,
            } => {
                assert_eq!(api_key, "sk-test123");
                assert_eq!(base_url, Some("https://api.openai.com".to_string()));
                assert_eq!(model, "gpt-4");
            }
            _ => panic!("Expected OpenAI config"),
        }

        // Test Anthropic config
        let anthropic = LlmConfig::Anthropic {
            api_key: "sk-ant-test".to_string(),
            base_url: None,
            model: "claude-3-sonnet".to_string(),
        };
        let json = serde_json::to_string(&anthropic).unwrap();
        let parsed: LlmConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            LlmConfig::Anthropic {
                api_key,
                base_url,
                model,
            } => {
                assert_eq!(api_key, "sk-ant-test");
                assert_eq!(base_url, None);
                assert_eq!(model, "claude-3-sonnet");
            }
            _ => panic!("Expected Anthropic config"),
        }

        // Test Ollama config with default base URL
        let ollama = LlmConfig::Ollama {
            base_url: "http://localhost:11434/v1".to_string(),
            model: "llama3.2".to_string(),
        };
        let json = serde_json::to_string(&ollama).unwrap();
        let parsed: LlmConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            LlmConfig::Ollama { base_url, model } => {
                assert_eq!(base_url, "http://localhost:11434/v1");
                assert_eq!(model, "llama3.2");
            }
            _ => panic!("Expected Ollama config"),
        }

        // Test Custom config
        let custom = LlmConfig::Custom {
            base_url: "https://my-endpoint.com/v1".to_string(),
            api_key: Some("custom-key".to_string()),
            model: "custom-model".to_string(),
        };
        let json = serde_json::to_string(&custom).unwrap();
        let parsed: LlmConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            LlmConfig::Custom {
                base_url,
                api_key,
                model,
            } => {
                assert_eq!(base_url, "https://my-endpoint.com/v1");
                assert_eq!(api_key, Some("custom-key".to_string()));
                assert_eq!(model, "custom-model");
            }
            _ => panic!("Expected Custom config"),
        }
    }

    #[test]
    fn test_workspace_info_serialization() {
        let info = WorkspaceInfo {
            id: "test_workspace".to_string(),
            root: "/path/to/workspace".to_string(),
            source: "git".to_string(),
            db_dir: "/path/to/db".to_string(),
            capsules_jsonl: "/path/to/capsules.jsonl".to_string(),
            created_ts_ms: 1234567890,
            updated_ts_ms: 1234567891,
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: WorkspaceInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, info.id);
        assert_eq!(parsed.root, info.root);
        assert_eq!(parsed.source, info.source);
        assert_eq!(parsed.db_dir, info.db_dir);
        assert_eq!(parsed.capsules_jsonl, info.capsules_jsonl);
        assert_eq!(parsed.created_ts_ms, info.created_ts_ms);
        assert_eq!(parsed.updated_ts_ms, info.updated_ts_ms);
    }

    #[test]
    fn test_workspace_config_serialization() {
        let mut config = WorkspaceConfig {
            version: 1,
            path_index: std::collections::BTreeMap::new(),
            workspaces: std::collections::BTreeMap::new(),
            llm: None,
        };

        config
            .path_index
            .insert("/path/to/workspace1".to_string(), "workspace1".to_string());
        config
            .path_index
            .insert("/path/to/workspace2".to_string(), "workspace2".to_string());

        config.workspaces.insert(
            "workspace1".to_string(),
            WorkspaceInfo {
                id: "workspace1".to_string(),
                root: "/path/to/workspace1".to_string(),
                source: "git".to_string(),
                db_dir: "/path/to/db1".to_string(),
                capsules_jsonl: "/path/to/capsules1.jsonl".to_string(),
                created_ts_ms: 1000,
                updated_ts_ms: 2000,
            },
        );

        config.llm = Some(LlmConfig::Openai {
            api_key: "sk-test".to_string(),
            base_url: None,
            model: "gpt-4".to_string(),
        });

        let json = serde_json::to_string(&config).unwrap();
        let parsed: WorkspaceConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.path_index.len(), 2);
        assert_eq!(parsed.workspaces.len(), 1);
        assert!(parsed.llm.is_some());
    }

    #[test]
    fn test_workspace_config_default_fields() {
        let config = WorkspaceConfig {
            version: 1,
            path_index: std::collections::BTreeMap::new(),
            workspaces: std::collections::BTreeMap::new(),
            llm: None,
        };

        // Test that default fields work correctly
        assert_eq!(config.version, 1);
        assert!(config.path_index.is_empty());
        assert!(config.workspaces.is_empty());
        assert!(config.llm.is_none());
    }
}
