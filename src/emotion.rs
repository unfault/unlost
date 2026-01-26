use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer as HfTokenizer, TruncationParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EmotionMeta {
    /// One of: joy | neutral | confused | frustration | anger | sad
    pub(crate) label: String,
    /// -1..1 (negative..positive)
    pub(crate) valence: f32,
    /// 0..1 (calm..intense)
    pub(crate) intensity: f32,
    /// 0..1
    pub(crate) confidence: f32,
}

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn is_turn_marker(t: &str) -> bool {
    // Format we generate: "Turn <n>:" (e.g. "Turn 1:")
    let t = t.trim();
    if !t.starts_with("Turn ") || !t.ends_with(':') {
        return false;
    }
    let inner = &t[5..t.len() - 1];
    let inner = inner.trim();
    !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit())
}

pub(crate) fn extract_user_and_assistant_text(slice: &str) -> (String, String) {
    // slice format produced by build_flush_job(): Turn i: <exchange_text>
    // exchange_text itself starts with "User:" and/or "Assistant:".
    let mut user = String::new();
    let mut assistant = String::new();

    let mut cur: Option<&str> = None;
    for line in slice.lines() {
        let t = line.trim_end();

        if is_turn_marker(t) {
            // Reset state so markers don't get treated as content.
            cur = None;
            continue;
        }
        if t == "User:" {
            cur = Some("user");
            continue;
        }
        if t == "Assistant:" {
            cur = Some("assistant");
            continue;
        }

        match cur {
            Some("user") => {
                user.push_str(t);
                user.push('\n');
            }
            Some("assistant") => {
                assistant.push_str(t);
                assistant.push('\n');
            }
            _ => {}
        }
    }

    (user.trim().to_string(), assistant.trim().to_string())
}

#[derive(Debug)]
pub(crate) struct EmotionModel {
    tokenizer: HfTokenizer,
    session: Session,
    id2label: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct EmotionConfig {
    /// HuggingFace repo id
    repo: String,
    /// ONNX model path within repo
    model_path: String,
    /// Tokenizer path within repo
    tokenizer_path: String,
    /// Config path within repo (for id2label)
    config_path: String,
    /// Max tokens for classification
    max_len: usize,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            repo: "SamLowe/roberta-base-go_emotions-onnx".to_string(),
            model_path: "onnx/model_quantized.onnx".to_string(),
            tokenizer_path: "onnx/tokenizer.json".to_string(),
            config_path: "config.json".to_string(),
            max_len: 128,
        }
    }
}

fn emotion_cache_dir() -> std::path::PathBuf {
    if let Some(d) = std::env::var_os("UNLOST_EMOTION_CACHE_DIR") {
        return std::path::PathBuf::from(d);
    }
    // ~/.local/share/unlost/models/emotion
    crate::unlost_data_root().join("models").join("emotion")
}

fn sanitize_repo_dir_name(repo: &str) -> String {
    repo.replace('/', "_")
}

async fn download_hf_file(repo: &str, rfilename: &str, dest: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{rfilename}");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("failed to download {url} (status {status})");
    }
    let bytes = resp.bytes().await?;
    std::fs::write(dest, &bytes)?;
    Ok(())
}

async fn ensure_emotion_model_files(cfg: &EmotionConfig) -> anyhow::Result<std::path::PathBuf> {
    let base = emotion_cache_dir().join(sanitize_repo_dir_name(&cfg.repo));

    let model_path = base.join(&cfg.model_path);
    let tok_path = base.join(&cfg.tokenizer_path);
    let cfg_path = base.join(&cfg.config_path);

    download_hf_file(&cfg.repo, &cfg.model_path, &model_path).await?;
    download_hf_file(&cfg.repo, &cfg.tokenizer_path, &tok_path).await?;
    download_hf_file(&cfg.repo, &cfg.config_path, &cfg_path).await?;
    Ok(base)
}

fn parse_id2label(config_json: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(config_json).ok()?;
    let map = v.get("id2label")?.as_object()?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for (k, v) in map {
        let idx: usize = k.parse().ok()?;
        let label = v.as_str().unwrap_or("").to_string();
        if !label.is_empty() {
            pairs.push((idx, label));
        }
    }
    pairs.sort_by_key(|p| p.0);
    if pairs.is_empty() {
        return None;
    }
    Some(pairs.into_iter().map(|p| p.1).collect())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn map_go_emotions(label: &str, score: f32) -> EmotionMeta {
    // Map go_emotions labels to a small stable set.
    // ref: google-research-datasets/go_emotions
    let l = label.to_ascii_lowercase();

    let (bucket, valence, base_intensity) = match l.as_str() {
        // Positive cluster
        "admiration" | "amusement" | "approval" | "caring" | "desire" | "excitement" | "gratitude"
        | "joy" | "love" | "optimism" | "pride" | "relief" => ("joy", 0.8, 0.4),

        // Negative cluster
        "anger" => ("anger", -0.9, 0.75),
        "annoyance" => ("frustration", -0.7, 0.55),
        "disappointment" | "remorse" | "sadness" | "grief" => ("sad", -0.8, 0.45),
        "fear" | "nervousness" => ("frustration", -0.6, 0.55),
        "disgust" => ("frustration", -0.7, 0.6),
        "embarrassment" => ("confused", -0.3, 0.4),

        // Cognitive/uncertainty
        "confusion" | "curiosity" | "realization" | "surprise" => ("confused", 0.0, 0.35),

        // Neutral
        "neutral" => ("neutral", 0.0, 0.1),

        // Default
        _ => ("neutral", 0.0, 0.15),
    };

    let confidence = clamp01(score);
    let intensity = clamp01(base_intensity * 0.4 + confidence * 0.6);
    EmotionMeta {
        label: bucket.to_string(),
        valence,
        intensity,
        confidence,
    }
}

impl EmotionModel {
    pub(crate) async fn load(cfg: EmotionConfig) -> anyhow::Result<Self> {
        let base = ensure_emotion_model_files(&cfg).await?;
        let model_path = base.join(&cfg.model_path);
        let tok_path = base.join(&cfg.tokenizer_path);
        let cfg_path = base.join(&cfg.config_path);

        let config_json = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("failed to read {}", cfg_path.display()))?;
        let id2label = parse_id2label(&config_json).unwrap_or_else(|| vec!["neutral".to_string()]);

        let mut tokenizer = HfTokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: cfg.max_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("failed to set truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(cfg.max_len),
            ..Default::default()
        }));

        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("failed to load emotion ONNX model")?;

        Ok(Self {
            tokenizer,
            session,
            id2label,
        })
    }

    pub(crate) fn classify_one(&mut self, text: &str) -> anyhow::Result<(String, f32)> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids().iter().map(|&v| v as i64).collect::<Vec<_>>();
        let mask = enc
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>();

        let len = ids.len();
        if len == 0 {
            anyhow::bail!("empty tokenization");
        }

        let input_ids = Tensor::<i64>::from_array((vec![1_usize, len], ids))?;
        let attention_mask = Tensor::<i64>::from_array((vec![1_usize, len], mask))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])
            .context("emotion model run failed")?;

        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract logits")?;
        if logits.is_empty() {
            anyhow::bail!("empty logits");
        }

        // go_emotions is typically multi-label; treat as sigmoid probs and take the max.
        let mut best_i = 0usize;
        let mut best_p = 0.0f32;
        for (i, &x) in logits.iter().enumerate() {
            let p = sigmoid(x);
            if p > best_p {
                best_p = p;
                best_i = i;
            }
        }

        let label = self
            .id2label
            .get(best_i)
            .cloned()
            .unwrap_or_else(|| "neutral".to_string());
        Ok((label, best_p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_user_and_assistant_text() {
        let s = "Turn 1:\nUser:\nHello\nAssistant:\nHi there\nTurn 2:\nUser:\nOk\nAssistant:\nDone";
        let (u, a) = extract_user_and_assistant_text(s);
        assert_eq!(u, "Hello\nOk");
        assert_eq!(a, "Hi there\nDone");
    }

    #[test]
    fn test_sanitize_repo_dir_name() {
        assert_eq!(sanitize_repo_dir_name("SamLowe/roberta-base"), "SamLowe_roberta-base");
        assert_eq!(sanitize_repo_dir_name("a/b/c"), "a_b_c");
    }

    #[test]
    fn test_parse_id2label_sorts_by_numeric_key() {
        let json = r#"{"id2label":{"2":"joy","0":"neutral","1":"sad"}}"#;
        let labels = parse_id2label(json).expect("id2label parsed");
        assert_eq!(labels, vec!["neutral", "sad", "joy"]);
    }

    #[test]
    fn test_map_go_emotions_bucket_and_ranges() {
        let e = map_go_emotions("anger", 0.9);
        assert_eq!(e.label, "anger");
        assert!(e.valence < 0.0);
        assert!((0.0..=1.0).contains(&e.confidence));
        assert!((0.0..=1.0).contains(&e.intensity));

        let e2 = map_go_emotions("JOY", 0.7);
        assert_eq!(e2.label, "joy");
        assert!(e2.valence > 0.0);
    }
}
