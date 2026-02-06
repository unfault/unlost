//! Integration tests for emotion detection.
//!
//! These test cases document the expected emotion mappings.
//! Run with: cargo test --test emotion_detection

/// Test cases: (input_text, expected_buckets, description)
///
/// Expected buckets are what we want the model to detect.
/// Multiple expected values means any of them is acceptable (ML isn't deterministic).
const TEST_CASES: &[(&str, &[&str], &str)] = &[
    // Joy/positive
    (
        "This is exactly what I needed, thank you so much!",
        &["joy"],
        "clear gratitude",
    ),
    (
        "Perfect! That's brilliant!",
        &["joy"],
        "excitement/approval",
    ),
    // Frustration
    (
        "This still doesn't work. I've tried this three times already.",
        &["frustration", "anger"],
        "repeated failure frustration",
    ),
    (
        "Why do you keep suggesting the same thing?",
        &["frustration", "anger", "disapproval"],
        "annoyance at repetition",
    ),
    // Disapproval
    (
        "No, that's not right. This approach is wrong.",
        &["disapproval", "frustration"],
        "clear disapproval",
    ),
    (
        "I don't think this is the correct solution.",
        &["disapproval", "doubt"],
        "polite disapproval",
    ),
    // Doubt/uncertainty (maps from nervousness in GoEmotions)
    (
        "Are you sure about this? I'm not convinced.",
        &["doubt", "disapproval", "confused"],
        "skepticism",
    ),
    (
        "I'm worried this might break something else.",
        &["doubt", "frustration"],
        "nervousness about approach",
    ),
    // Confusion
    (
        "I don't understand what you mean.",
        &["confused"],
        "genuine confusion",
    ),
    (
        "Wait, how does that work?",
        &["confused"],
        "seeking clarification",
    ),
    // Neutral
    (
        "Can you modify the auth.ts file?",
        &["neutral", "joy"],
        "neutral request",
    ),
    (
        "What's in the config?",
        &["neutral", "confused"],
        "neutral question",
    ),
    // Anger
    (
        "This is completely broken! Nothing works!",
        &["anger", "frustration"],
        "strong anger",
    ),
];

#[test]
fn test_cases_are_valid() {
    // Verify all test cases have valid structure
    for (input, expected, description) in TEST_CASES {
        assert!(!input.is_empty(), "Input should not be empty");
        assert!(!expected.is_empty(), "Expected buckets should not be empty");
        assert!(!description.is_empty(), "Description should not be empty");

        // Verify expected buckets are valid emotion labels
        let valid_buckets = [
            "joy",
            "neutral",
            "confused",
            "doubt",
            "frustration",
            "anger",
            "sad",
            "disapproval",
            "fear",
        ];
        for bucket in *expected {
            assert!(
                valid_buckets.contains(bucket),
                "Invalid bucket '{}' in test case '{}'",
                bucket,
                description
            );
        }
    }
}

#[test]
fn test_friction_emotions_are_covered() {
    // Verify that friction-triggering emotions appear in test cases
    let friction_emotions = ["frustration", "anger", "annoyance", "disapproval"];

    for emotion in friction_emotions {
        let covered = TEST_CASES
            .iter()
            .any(|(_, expected, _)| expected.contains(&emotion));
        // annoyance maps to frustration bucket, so we check for frustration
        if emotion != "annoyance" {
            assert!(
                covered,
                "Friction emotion '{}' should have test cases",
                emotion
            );
        }
    }
}

/// Print test cases for manual verification
/// Run with: cargo test --test emotion_detection -- --nocapture print_test_cases
#[test]
fn print_test_cases() {
    println!("\n{:=<80}", "");
    println!("EMOTION DETECTION TEST CASES");
    println!("{:=<80}\n", "");

    for (input, expected, description) in TEST_CASES {
        println!("Description: {}", description);
        println!("Input:       {:?}", input);
        println!("Expected:    {:?}", expected);
        println!("{:-<80}\n", "");
    }
}
