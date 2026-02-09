//! Test emotion detection on arbitrary text.
//!
//! This is a hidden developer tool for validating emotion classification.
//! Usage: unlost emotion "I'm not sure this is right"

use crate::emotion::{EmotionConfig, EmotionModel, apply_context_heuristics, map_go_emotions};

pub(crate) async fn run(text: String) -> anyhow::Result<()> {
    println!("Loading emotion model...");
    let mut model = EmotionModel::load(EmotionConfig::default()).await?;

    println!("Classifying: {:?}\n", text);

    let (raw_label, score) = model.classify_one(&text)?;
    let model_meta = map_go_emotions(&raw_label, score);
    let meta = apply_context_heuristics(&text, model_meta.clone());

    println!("Raw GoEmotions label: {}", raw_label);
    println!("Confidence:          {:.2}", score);
    println!();
    println!("Model bucket:        {}", model_meta.label);
    if meta.label != model_meta.label {
        println!("After heuristics:    {} (overridden)", meta.label);
    }
    println!("Final bucket:        {}", meta.label);
    println!(
        "Valence:             {:.2} (-1=negative, +1=positive)",
        meta.valence
    );
    println!(
        "Intensity:           {:.2} (0=calm, 1=intense)",
        meta.intensity
    );

    // Show if this would trigger friction detection
    // Note: "sad" in agent context usually means dissatisfaction, not actual sadness
    let friction_emotions = [
        "frustration",
        "anger",
        "annoyance",
        "disapproval",
        "doubt",
        "sad",
    ];
    let triggers_friction = friction_emotions.contains(&meta.label.as_str());
    println!();
    println!(
        "Triggers friction:   {} {}",
        if triggers_friction { "YES" } else { "no" },
        if triggers_friction {
            "(would inject warning if symbols repeat)"
        } else {
            ""
        }
    );

    Ok(())
}
