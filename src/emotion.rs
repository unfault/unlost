use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer as HfTokenizer, TruncationParams};

use crate::workspace::unlost_data_root;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionMeta {
    /// One of: joy | neutral | confused | doubt | frustration | anger | sad | disapproval
    pub label: String,
    /// -1..1 (negative..positive)
    pub valence: f32,
    /// 0..1 (calm..intense)
    pub intensity: f32,
    /// 0..1
    pub confidence: f32,
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
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

pub fn extract_user_and_assistant_text(slice: &str) -> (String, String) {
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
pub struct EmotionModel {
    tokenizer: HfTokenizer,
    session: Session,
    id2label: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EmotionConfig {
    /// HuggingFace repo id
    pub repo: String,
    /// ONNX model path within repo
    pub model_path: String,
    /// Tokenizer path within repo
    pub tokenizer_path: String,
    /// Config path within repo (for id2label)
    pub config_path: String,
    /// Max tokens for classification
    pub max_len: usize,
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
    unlost_data_root().join("models").join("emotion")
}

fn sanitize_repo_dir_name(repo: &str) -> String {
    repo.replace('/', "_")
}

async fn download_hf_file(
    repo: &str,
    rfilename: &str,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
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

pub fn map_go_emotions(label: &str, score: f32) -> EmotionMeta {
    // Map go_emotions labels to a small stable set.
    // ref: google-research-datasets/go_emotions
    let l = label.to_ascii_lowercase();

    let (bucket, valence, base_intensity) = match l.as_str() {
        // Positive cluster
        "admiration" | "amusement" | "approval" | "caring" | "desire" | "excitement"
        | "gratitude" | "joy" | "love" | "optimism" | "pride" | "relief" => ("joy", 0.8, 0.4),

        // Negative cluster
        "anger" => ("anger", -0.9, 0.75),
        "annoyance" => ("frustration", -0.7, 0.55),
        "disappointment" | "remorse" | "sadness" | "grief" => ("sad", -0.8, 0.45),
        "fear" => ("frustration", -0.6, 0.55),
        "disgust" => ("frustration", -0.7, 0.6),
        "embarrassment" => ("confused", -0.3, 0.4),

        // Doubt - user is uncertain/skeptical about the direction
        "nervousness" => ("doubt", -0.3, 0.45),

        // Disapproval - user disagrees with or dislikes the approach (decided)
        "disapproval" => ("disapproval", -0.5, 0.5),

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

/// Heuristic patterns that indicate doubt/skepticism in agent conversations.
/// These override neutral/confused when the model misses the context.
const DOUBT_PATTERNS: &[&str] = &[
    "why was",
    "why is",
    "why did",
    "why do",
    "clarify why",
    "explain why",
    "are you sure",
    "not sure",
    "not convinced",
    "is this right",
    "is that right",
    "is this correct",
    "is that correct",
    "doesn't seem right",
    "doesn't look right",
    "seems wrong",
    "looks wrong",
    "really necessary",
    "was necessary",
    "is necessary",
    "do we need",
    "do you need",
    "should we",
    "shouldn't we",
];

/// Heuristic patterns that indicate frustration/negative emotions.
/// Used to boost negative emotion detection when the model misclassifies as positive/neutral.
const FRUSTRATION_SIGNALS: &[&str] = &[
    // Exasperation
    "sigh",
    "uh?",
    "ugh",
    "seriously",
    "come on",
    "again?",
    "again!",
    // Sarcasm/rhetorical
    "you kidding",
    "are you serious",
    "this is ridiculous",
    "that's ridiculous",
    // Repetition complaints
    "still not",
    "already tried",
    "same error",
    "same issue",
    "same problem",
    "we discussed this",
    "going in circles",
    "not working",
    "doesn't work",
    "broken",
    // Short dismissive
    "whatever",
    "never mind",
    "forget it",
    // Direct frustration
    "frustrated",
    "upset",
    "annoyed",
    "annoying",
    "irritating",
    "terrible",
    "awful",
    "useless",
    "pointless",
    "waste of time",
];

/// Emotions that are considered negative for friction detection purposes.
pub const NEGATIVE_EMOTIONS: &[&str] = &[
    "frustration",
    "anger",
    "sad",
    "confused",
    "doubt",
    "disapproval",
];

/// Check if an emotion label is considered negative.
pub fn is_negative_emotion(label: &str) -> bool {
    NEGATIVE_EMOTIONS.contains(&label)
}

/// Apply text-based heuristics to catch doubt patterns the model misses.
/// Only overrides neutral/confused, preserves stronger signals.
pub fn apply_context_heuristics(text: &str, meta: EmotionMeta) -> EmotionMeta {
    // First, try to boost negative emotions if the model misclassified
    let meta = boost_negative_emotions(text, meta);

    // Only apply doubt heuristics to weak signals (neutral/confused)
    if meta.label != "neutral" && meta.label != "confused" {
        return meta;
    }

    let lower = text.to_lowercase();
    let has_doubt_pattern = DOUBT_PATTERNS.iter().any(|p| lower.contains(p));

    if has_doubt_pattern {
        EmotionMeta {
            label: "doubt".to_string(),
            valence: -0.3,
            intensity: 0.5,
            confidence: meta.confidence,
        }
    } else {
        meta
    }
}

/// Boost negative emotion detection when text contains frustration signals
/// but the model classified as positive (joy) or neutral.
///
/// This helps catch sarcasm, exasperation ("sigh"), and other signals
/// that the model often misclassifies.
fn boost_negative_emotions(text: &str, meta: EmotionMeta) -> EmotionMeta {
    // If model already detected a negative emotion, trust it
    if is_negative_emotion(&meta.label) {
        return meta;
    }

    let lower = text.to_lowercase();

    // Check for frustration signal patterns
    let has_frustration_signal = FRUSTRATION_SIGNALS.iter().any(|p| lower.contains(p));

    // Check for rhetorical markers (e.g., "?!" or multiple "?")
    let has_rhetorical =
        text.contains("?!") || text.contains("!?") || text.matches('?').count() >= 2;

    // Check for short dismissive replies (< 15 words with certain keywords)
    let word_count = text.split_whitespace().count();
    let is_short_dismissive = word_count < 15
        && (lower.contains("fine")
            || lower.contains("ok then")
            || lower.contains("okay then")
            || lower.contains("i guess"));

    // Count how many signals we found
    let signal_count =
        has_frustration_signal as u8 + has_rhetorical as u8 + is_short_dismissive as u8;

    if signal_count == 0 {
        return meta;
    }

    // If model said joy but we found frustration signals, override to frustration
    if meta.label == "joy" && signal_count >= 1 {
        tracing::debug!(
            original_label = %meta.label,
            original_confidence = meta.confidence,
            has_frustration_signal,
            has_rhetorical,
            is_short_dismissive,
            "boosting emotion from joy to frustration based on text signals"
        );
        return EmotionMeta {
            label: "frustration".to_string(),
            valence: -0.6,
            intensity: 0.5,
            // Use a moderate confidence since we're overriding the model
            confidence: 0.6_f32.max(meta.confidence * 0.7),
        };
    }

    // If model said neutral and we have strong signals (2+), boost to frustration
    if meta.label == "neutral" && signal_count >= 2 {
        tracing::debug!(
            original_label = %meta.label,
            original_confidence = meta.confidence,
            signal_count,
            "boosting emotion from neutral to frustration based on multiple text signals"
        );
        return EmotionMeta {
            label: "frustration".to_string(),
            valence: -0.5,
            intensity: 0.45,
            confidence: 0.55,
        };
    }

    // If model said neutral and we have exactly 1 signal, do not override — one matched
    // keyword on an otherwise neutral message is too weak a signal (e.g. the word "broken"
    // in a technical description, or "seriously" as an intensifier in a positive context).
    // Two signals are required to conclude even mild disapproval (handled above at signal_count >= 2).

    meta
}

impl EmotionModel {
    pub async fn load(cfg: EmotionConfig) -> anyhow::Result<Self> {
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

    pub fn classify_one(&mut self, text: &str) -> anyhow::Result<(String, f32)> {
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
        assert_eq!(
            sanitize_repo_dir_name("SamLowe/roberta-base"),
            "SamLowe_roberta-base"
        );
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

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(0.0), 0.0);
        assert_eq!(clamp01(1.0), 1.0);
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn test_is_turn_marker() {
        assert!(is_turn_marker("Turn 1:"));
        assert!(is_turn_marker("Turn 123:"));
        assert!(is_turn_marker("  Turn 5:  "));
        assert!(!is_turn_marker("Turn:"));
        assert!(!is_turn_marker("Turn a:"));
        assert!(!is_turn_marker("Turn 1"));
        assert!(!is_turn_marker("turn 1:"));
    }

    #[test]
    fn test_sigmoid() {
        assert!(sigmoid(0.0) - 0.5 < 0.001);
        assert!(sigmoid(f32::INFINITY) - 1.0 < 0.001);
        assert!(sigmoid(f32::NEG_INFINITY) < 0.001);
        assert!(sigmoid(1.0) > 0.5);
        assert!(sigmoid(-1.0) < 0.5);
    }

    #[test]
    fn test_emotion_meta_serialization() {
        let meta = EmotionMeta {
            label: "joy".to_string(),
            valence: 0.8,
            intensity: 0.6,
            confidence: 0.9,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: EmotionMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.label, "joy");
        assert_eq!(parsed.valence, 0.8);
        assert_eq!(parsed.intensity, 0.6);
        assert_eq!(parsed.confidence, 0.9);
    }

    #[test]
    fn test_emotion_config_default() {
        let cfg = EmotionConfig::default();
        assert_eq!(cfg.repo, "SamLowe/roberta-base-go_emotions-onnx");
        assert_eq!(cfg.model_path, "onnx/model_quantized.onnx");
        assert_eq!(cfg.tokenizer_path, "onnx/tokenizer.json");
        assert_eq!(cfg.config_path, "config.json");
        assert_eq!(cfg.max_len, 128);
    }

    #[test]
    fn test_extract_user_and_assistant_text_edge_cases() {
        // Empty input
        let (u, a) = extract_user_and_assistant_text("");
        assert_eq!(u, "");
        assert_eq!(a, "");

        // Only user text
        let s = "Turn 1:\nUser:\nHello world";
        let (u, a) = extract_user_and_assistant_text(s);
        assert_eq!(u, "Hello world");
        assert_eq!(a, "");

        // Only assistant text
        let s = "Turn 1:\nAssistant:\nHello world";
        let (u, a) = extract_user_and_assistant_text(s);
        assert_eq!(u, "");
        assert_eq!(a, "Hello world");

        // Multiple turns with mixed content
        let s = r#"Turn 1:
User:
Hello
Assistant:
Hi there
Turn 2:
User:
How are you?
Assistant:
I'm doing great!
Turn 3:
User:
Bye
"#;
        let (u, a) = extract_user_and_assistant_text(s);
        assert_eq!(u, "Hello\nHow are you?\nBye");
        assert_eq!(a, "Hi there\nI'm doing great!");
    }

    #[test]
    fn test_map_go_emotions_comprehensive() {
        // Test positive emotions
        let test_cases = vec![
            ("admiration", "joy", 0.8),
            ("amusement", "joy", 0.8),
            ("approval", "joy", 0.8),
            ("caring", "joy", 0.8),
            ("desire", "joy", 0.8),
            ("excitement", "joy", 0.8),
            ("gratitude", "joy", 0.8),
            ("love", "joy", 0.8),
            ("optimism", "joy", 0.8),
            ("pride", "joy", 0.8),
            ("relief", "joy", 0.8),
        ];

        for (emotion, expected_label, expected_valence) in test_cases {
            let meta = map_go_emotions(emotion, 0.7);
            assert_eq!(meta.label, expected_label);
            assert_eq!(meta.valence, expected_valence);
            assert!(meta.confidence >= 0.0 && meta.confidence <= 1.0);
        }

        // Test negative emotions
        let meta = map_go_emotions("anger", 0.8);
        assert_eq!(meta.label, "anger");
        assert!(meta.valence < 0.0);

        let meta = map_go_emotions("sadness", 0.6);
        assert_eq!(meta.label, "sad");
        assert!(meta.valence < 0.0);

        // Test cognitive emotions
        let meta = map_go_emotions("confusion", 0.5);
        assert_eq!(meta.label, "confused");
        assert_eq!(meta.valence, 0.0);

        // Test neutral
        let meta = map_go_emotions("neutral", 0.9);
        assert_eq!(meta.label, "neutral");
        assert_eq!(meta.valence, 0.0);

        // Test unknown emotion
        let meta = map_go_emotions("unknown_emotion", 0.3);
        assert_eq!(meta.label, "neutral");
        assert_eq!(meta.valence, 0.0);
    }

    #[test]
    fn test_map_go_emotions_doubt_and_disapproval() {
        // Test doubt (from nervousness)
        let meta = map_go_emotions("nervousness", 0.7);
        assert_eq!(meta.label, "doubt");
        assert!(meta.valence < 0.0); // slightly negative
        assert!(meta.valence > -0.5); // but not strongly negative

        // Test disapproval
        let meta = map_go_emotions("disapproval", 0.8);
        assert_eq!(meta.label, "disapproval");
        assert!(meta.valence < 0.0);

        // Verify the progression: doubt is less negative than disapproval
        let doubt = map_go_emotions("nervousness", 0.7);
        let disapproval = map_go_emotions("disapproval", 0.7);
        let frustration = map_go_emotions("annoyance", 0.7);

        // doubt (-0.3) > disapproval (-0.5) > frustration (-0.7)
        assert!(doubt.valence > disapproval.valence);
        assert!(disapproval.valence > frustration.valence);
    }
}
