//! Per-capsule cooldown tracking for the recurrence channel.
//!
//! When the resurfacing basin surfaces a capsule via a SYSTEM NOTE, we record
//! `(capsule_id, ts_ms)` in `~/.local/share/unlost/workspaces/<wks>/resurfaced.jsonl`.
//! That capsule is then ineligible to be surfaced again for the next 30 days.
//!
//! The file is append-only. We do not rewrite or compact it; lookups load it
//! lazily into a `HashMap<CapsuleId, i64>` on the trajectory controller. See
//! `internal/SOURCE_POINTERS.md` §Recurrence Channel for the surrounding design.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// 30 days, in milliseconds. After this window a capsule may be resurfaced again.
pub const COOLDOWN_MS: i64 = 30 * 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    capsule_id: String,
    ts_ms: i64,
}

/// Path to a workspace's resurfaced.jsonl file. Kept for back-compat; new
/// code should prefer [`global_resurfaced_path`] so cross-workspace surfacings
/// share a single dedup ledger.
pub fn resurfaced_path(workspace_id: &str) -> PathBuf {
    crate::workspace::unlost_workspace_dir(workspace_id).join("resurfaced.jsonl")
}

/// Path to the global resurfaced ledger, shared across all workspaces.
/// Lives at `~/.local/share/unlost/resurfaced.jsonl`. Capsule ids are UUIDs
/// so collisions across workspaces are not a concern.
pub fn global_resurfaced_path() -> PathBuf {
    crate::workspace::unlost_data_root().join("resurfaced.jsonl")
}

/// Load all recorded surfacings into an in-memory map. Reads both the
/// workspace-scoped legacy file and the global shared ledger so cross-workspace
/// surfacings respect each other's cooldowns.
///
/// Best-effort: missing or malformed files yield an empty contribution; lines
/// that don't parse are skipped silently. Never panics, never errors back to
/// the caller — the recurrence channel must degrade gracefully.
pub fn load(workspace_id: &str) -> HashMap<String, i64> {
    let mut out: HashMap<String, i64> = HashMap::new();
    for path in [resurfaced_path(workspace_id), global_resurfaced_path()] {
        load_into(&path, &mut out);
    }
    out
}

fn load_into(path: &Path, out: &mut HashMap<String, i64>) {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let reader = std::io::BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<Entry>(line) {
            // Keep the most recent surfacing for any given capsule (append-only
            // file may legitimately contain multiple entries if the cooldown
            // elapsed).
            let cur = out.get(&entry.capsule_id).copied().unwrap_or(0);
            if entry.ts_ms > cur {
                out.insert(entry.capsule_id, entry.ts_ms);
            }
        }
    }
}

/// Append a surfacing record to the global ledger. Best-effort: I/O errors
/// are logged at debug level and swallowed so a write failure never blocks
/// the friction-check hot path.
///
/// The `workspace_id` is currently unused — kept in the signature so callers
/// don't need to know about the global-vs-local split, and to leave room for
/// a future per-workspace audit log without changing call sites.
pub fn record(_workspace_id: &str, capsule_id: &str, ts_ms: i64) {
    let path = global_resurfaced_path();
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        tracing::debug!(error = %e, "resurfaced.jsonl: failed to create parent dir");
        return;
    }
    let entry = Entry {
        capsule_id: capsule_id.to_string(),
        ts_ms,
    };
    let line = match serde_json::to_string(&entry) {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!(error = %e, "resurfaced.jsonl: serialize failed");
            return;
        }
    };
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(format!("{line}\n").as_bytes()));
    if let Err(e) = res {
        tracing::debug!(error = %e, path = %path.display(), "resurfaced.jsonl: append failed");
    }
}

/// True if `capsule_id` was surfaced within `COOLDOWN_MS` of `now_ms`.
pub fn is_cooling(map: &HashMap<String, i64>, capsule_id: &str, now_ms: i64) -> bool {
    map.get(capsule_id)
        .map(|prev| now_ms - prev < COOLDOWN_MS)
        .unwrap_or(false)
}

/// Test-only override of the data root so tests can use a tmpdir.
#[cfg(test)]
pub fn test_path_with_root(root: &Path, workspace_id: &str) -> PathBuf {
    root.join("workspaces").join(workspace_id).join("resurfaced.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooling_window() {
        let mut map = HashMap::new();
        map.insert("cap_1".to_string(), 1_000_000_000_000);
        // 1 day after — within 30 day window
        assert!(is_cooling(&map, "cap_1", 1_000_000_000_000 + 86_400_000));
        // 31 days after — past the window
        assert!(!is_cooling(
            &map,
            "cap_1",
            1_000_000_000_000 + 31 * 86_400_000
        ));
        // Different capsule — never seen
        assert!(!is_cooling(&map, "cap_2", 1_000_000_000_000 + 86_400_000));
    }

    #[test]
    fn roundtrip_load_and_record() {
        let tmp = tempfile::tempdir().unwrap();
        // Redirect via env XDG_DATA_HOME so the workspace dir resolves into tmp.
        // (The real `record` uses `unlost_workspace_dir` which honors XDG_DATA_HOME.)
        let prev = std::env::var_os("XDG_DATA_HOME");
        // SAFETY: tests are single-threaded with respect to env (we set
        // RUST_TEST_THREADS=1 in CI for the few tests that touch env); the
        // small risk is acceptable for this local round-trip.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", tmp.path());
        }
        let ws = "wks_test_roundtrip";
        record(ws, "cap_x", 1_700_000_000_000);
        record(ws, "cap_x", 1_700_000_000_500); // newer entry wins
        record(ws, "cap_y", 1_700_000_001_000);
        let map = load(ws);
        assert_eq!(map.get("cap_x"), Some(&1_700_000_000_500));
        assert_eq!(map.get("cap_y"), Some(&1_700_000_001_000));
        // Restore
        unsafe {
            match prev {
                Some(v) => std::env::set_var("XDG_DATA_HOME", v),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
        }
        // path helper sanity
        let _ = test_path_with_root(tmp.path(), ws);
    }
}
