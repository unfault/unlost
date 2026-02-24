//! Integration tests for the OpenCode stdio shim deduplication behavior.
//!
//! These tests verify that the `turn_key` field prevents duplicate capsules
//! from being written to capsules.jsonl.
//!
//! Run with: cargo test --test shim_dedup

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use tempfile::TempDir;

/// Spawn `unlost shim opencode --no-extraction` in an isolated environment.
///
/// Both XDG_DATA_HOME and XDG_CONFIG_HOME are redirected into `data_dir` so
/// that no real workspace data is touched and the workspace config is fresh.
fn spawn_shim(workspace_dir: &PathBuf, data_dir: &PathBuf) -> Child {
    let bin_path = env!("CARGO_BIN_EXE_unlost");
    Command::new(bin_path)
        .args(["shim", "opencode", "--no-extraction"])
        .env("XDG_DATA_HOME", data_dir)
        .env("XDG_CONFIG_HOME", data_dir)
        .env("RUST_BACKTRACE", "1")
        .current_dir(workspace_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn shim")
}

fn wait_for_ready(stdout: &mut BufReader<std::process::ChildStdout>) {
    let mut line = String::new();
    stdout
        .read_line(&mut line)
        .expect("failed to read ready signal");
    assert!(
        line.contains("\"ready\":true"),
        "unexpected ready signal: {}",
        line
    );
}

fn send_record(
    stdin: &mut std::process::ChildStdin,
    directory: &str,
    user_text: &str,
    assistant_text: &str,
    turn_key: Option<&str>,
    agent_session_id: &str,
) {
    let request = serde_json::json!({
        "method": "record",
        "params": {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "directory": directory,
            "turn_key": turn_key,
            "agent_session_id": agent_session_id,
        }
    });
    stdin
        .write_all(format!("{}\n", request).as_bytes())
        .expect("failed to write request");
    stdin.flush().expect("failed to flush stdin");
}

fn read_response(stdout: &mut BufReader<std::process::ChildStdout>) {
    let mut line = String::new();
    stdout
        .read_line(&mut line)
        .expect("failed to read response");
}

/// Find capsules.jsonl anywhere under `root` and count its non-empty lines.
fn count_capsules(root: &PathBuf) -> usize {
    for entry in walkdir::WalkDir::new(root).into_iter().flatten() {
        if entry.file_name() == "capsules.jsonl" {
            return std::fs::read_to_string(entry.path())
                .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
                .unwrap_or(0);
        }
    }
    0
}

fn run_shim_with_records(
    workspace_dir: &PathBuf,
    data_dir: &PathBuf,
    records: &[(&str, &str, Option<&str>, &str)], // (user, assistant, turn_key, session_id)
) {
    let mut child = spawn_shim(workspace_dir, data_dir);
    let mut stdin = child.stdin.take().expect("failed to take stdin");
    let stdout = child.stdout.take().expect("failed to take stdout");
    let mut stdout = BufReader::new(stdout);
    let directory = workspace_dir.to_string_lossy().to_string();

    wait_for_ready(&mut stdout);

    for (user, assistant, turn_key, session_id) in records {
        send_record(
            &mut stdin, &directory, user, assistant, *turn_key, session_id,
        );
        read_response(&mut stdout);
    }

    // Close stdin — shim drains and exits
    drop(stdin);
    let status = child.wait().expect("failed to wait for shim");
    assert!(status.success(), "shim exited with: {}", status);
}

/// Sending the same turn twice WITH a turn_key must produce exactly 1 capsule.
/// This is the desired behavior — what turn_key in the plugin provides.
#[test]
fn test_turn_key_prevents_duplicate() {
    let temp = TempDir::new().expect("failed to create temp dir");
    let workspace_dir = temp.path().join("repo");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(workspace_dir.join(".git")).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    let turn_key = "user_msg_aaa:assistant_msg_bbb";
    let records = vec![
        (
            "how do I fix this?",
            "Try approach X.",
            Some(turn_key),
            "ses_test_001",
        ),
        // Same turn again — simulates plugin restart replaying the same exchange
        (
            "how do I fix this?",
            "Try approach X.",
            Some(turn_key),
            "ses_test_001",
        ),
    ];

    run_shim_with_records(&workspace_dir, &data_dir, &records);

    let count = count_capsules(&data_dir);
    assert_eq!(
        count, 1,
        "expected 1 capsule (dedup via turn_key), got {}",
        count
    );
}

/// Sending the same turn twice WITHOUT a turn_key produces 2 capsules.
/// This documents the current bug — without turn_key the shim always records.
#[test]
fn test_no_turn_key_writes_duplicate() {
    let temp = TempDir::new().expect("failed to create temp dir");
    let workspace_dir = temp.path().join("repo");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(workspace_dir.join(".git")).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    let records = vec![
        (
            "how do I fix this?",
            "Try approach X.",
            None,
            "ses_test_002",
        ),
        // Same turn again — no turn_key so no dedup possible
        (
            "how do I fix this?",
            "Try approach X.",
            None,
            "ses_test_002",
        ),
    ];

    run_shim_with_records(&workspace_dir, &data_dir, &records);

    let count = count_capsules(&data_dir);
    assert_eq!(
        count, 2,
        "expected 2 capsules (no dedup without turn_key), got {}",
        count
    );
}

/// Different turn_keys must each produce their own capsule — dedup is key-scoped.
#[test]
fn test_different_turn_keys_both_recorded() {
    let temp = TempDir::new().expect("failed to create temp dir");
    let workspace_dir = temp.path().join("repo");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(workspace_dir.join(".git")).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    let records = vec![
        (
            "first question",
            "first answer",
            Some("user_1:asst_1"),
            "ses_test_003",
        ),
        (
            "second question",
            "second answer",
            Some("user_2:asst_2"),
            "ses_test_003",
        ),
    ];

    run_shim_with_records(&workspace_dir, &data_dir, &records);

    let count = count_capsules(&data_dir);
    assert_eq!(
        count, 2,
        "expected 2 capsules for 2 distinct turn_keys, got {}",
        count
    );
}
