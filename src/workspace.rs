use crate::config::{WorkspaceConfig, WorkspaceInfo};
use anyhow::Context;
use sha2::{Digest, Sha256};
use std::io::IsTerminal;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub(crate) struct WorkspacePaths {
    pub(crate) id: String,
    pub(crate) db_dir: std::path::PathBuf,
    pub(crate) capsules_jsonl: std::path::PathBuf,
    pub(crate) metrics_jsonl: std::path::PathBuf,
}

fn xdg_config_home() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        return std::path::PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".config");
    }
    std::path::PathBuf::from(".")
}

fn xdg_data_home() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        return std::path::PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join(".local").join("share");
    }
    std::path::PathBuf::from(".")
}

pub(crate) fn unlost_config_path() -> std::path::PathBuf {
    xdg_config_home().join("unlost").join("config.json")
}

pub(crate) fn unlost_data_root() -> std::path::PathBuf {
    xdg_data_home().join("unlost")
}

pub(crate) fn unlost_workspace_dir(workspace_id: &str) -> std::path::PathBuf {
    unlost_data_root().join("workspaces").join(workspace_id)
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

pub(crate) fn canonicalize_dir(path: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    std::fs::canonicalize(path).context("failed to canonicalize path")
}

pub(crate) fn git_toplevel(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        return None;
    }
    std::fs::canonicalize(s).ok()
}

fn normalize_git_remote(remote: &str) -> String {
    let mut remote = remote.trim().to_string();

    if remote.starts_with("git@") {
        remote = remote[4..].to_string();
        remote = remote.replacen(":", "/", 1);
    } else if remote.starts_with("ssh://") {
        remote = remote[6..].to_string();
        if remote.starts_with("git@") {
            remote = remote[4..].to_string();
        }
    } else if let Some(pos) = remote.find("://") {
        remote = remote[(pos + 3)..].to_string();
        if let Some(at_pos) = remote.find('@') {
            if at_pos < remote.find('/').unwrap_or(remote.len()) {
                remote = remote[(at_pos + 1)..].to_string();
            }
        }
    }

    if remote.ends_with(".git") {
        remote = remote[..remote.len() - 4].to_string();
    }

    remote = remote.trim_end_matches('/').to_string();
    remote.to_lowercase()
}

fn compute_hash16(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..8])
}

fn get_git_remote(workspace_root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if output.status.success() {
        let remote = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !remote.is_empty() {
            return Some(remote);
        }
    }
    None
}

fn read_meta_files(workspace_root: &std::path::Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let candidates = [
        ("pyproject", "pyproject.toml"),
        ("package_json", "package.json"),
        ("cargo_toml", "Cargo.toml"),
        ("go_mod", "go.mod"),
    ];
    for (kind, name) in candidates {
        let p = workspace_root.join(name);
        if let Ok(s) = std::fs::read_to_string(&p) {
            out.push((kind.to_string(), s));
        }
    }
    out
}

fn extract_project_name_from_meta_files(meta_files: &[(String, String)]) -> Option<String> {
    let re_pyproject_project =
        regex::Regex::new(r#"\[project\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_pyproject_poetry =
        regex::Regex::new(r#"\[tool\.poetry\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_cargo =
        regex::Regex::new(r#"\[package\]\s*\n[^\[]*?name\s*=\s*[\"']([^\"']+)[\"']"#).ok();
    let re_go_mod = regex::Regex::new(r#"^module\s+(\S+)"#).ok();

    for (kind, contents) in meta_files {
        match kind.as_str() {
            "package_json" => {
                let json: serde_json::Value = serde_json::from_str(contents).ok()?;
                if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
                    return Some(name.to_string());
                }
            }
            "pyproject" => {
                if let Some(re) = &re_pyproject_project {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
                if let Some(re) = &re_pyproject_poetry {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
            }
            "cargo_toml" => {
                if let Some(re) = &re_cargo {
                    if let Some(caps) = re.captures(contents) {
                        return Some(caps.get(1)?.as_str().to_string());
                    }
                }
            }
            "go_mod" => {
                if let Some(re) = &re_go_mod {
                    for line in contents.lines() {
                        if let Some(caps) = re.captures(line) {
                            return Some(caps.get(1)?.as_str().to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn compute_workspace_id(workspace_root: &std::path::Path) -> Option<(String, String)> {
    if let Some(remote) = get_git_remote(workspace_root) {
        let norm = normalize_git_remote(&remote);
        if !norm.is_empty() {
            return Some((
                format!("wks_{}", compute_hash16(&format!("git:{norm}"))),
                "git".to_string(),
            ));
        }
    }

    let meta = read_meta_files(workspace_root);
    if let Some(name) = extract_project_name_from_meta_files(&meta) {
        return Some((
            format!("wks_{}", compute_hash16(&format!("manifest:{name}"))),
            "manifest".to_string(),
        ));
    }

    let label = workspace_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace");
    Some((
        format!("wks_{}", compute_hash16(&format!("label:cli:{label}"))),
        "label".to_string(),
    ))
}

pub(crate) fn load_workspace_config() -> WorkspaceConfig {
    let p = unlost_config_path();
    if let Ok(s) = std::fs::read_to_string(&p) {
        if let Ok(cfg) = serde_json::from_str::<WorkspaceConfig>(&s) {
            return cfg;
        }
    }
    WorkspaceConfig {
        version: 1,
        path_index: Default::default(),
        workspaces: Default::default(),
        llm: None,
    }
}

pub(crate) fn save_workspace_config(cfg: &WorkspaceConfig) -> anyhow::Result<()> {
    let p = unlost_config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = serde_json::to_string_pretty(cfg)?;
    std::fs::write(&p, s)?;
    Ok(())
}

pub(crate) fn get_or_create_workspace_paths(
    workspace_root: &std::path::Path,
) -> anyhow::Result<WorkspacePaths> {
    let root = git_toplevel(workspace_root).unwrap_or_else(|| {
        canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf())
    });
    let root = canonicalize_dir(&root)?;
    let root_str = root.to_string_lossy().to_string();
    let mut cfg = load_workspace_config();

    if let Some(existing_id) = cfg.path_index.get(&root_str).cloned() {
        if let Some(info) = cfg.workspaces.get_mut(&existing_id) {
            info.updated_ts_ms = now_ms();
            let _ = save_workspace_config(&cfg);

            let ws_dir = unlost_workspace_dir(&existing_id);
            return Ok(WorkspacePaths {
                id: existing_id,
                db_dir: ws_dir.join("lancedb"),
                capsules_jsonl: ws_dir.join("capsules.jsonl"),
                metrics_jsonl: ws_dir.join("metrics.jsonl"),
            });
        }
    }

    let (id, source) = compute_workspace_id(&root)
        .ok_or_else(|| anyhow::anyhow!("unable to compute workspace id"))?;

    let ws_dir = unlost_workspace_dir(&id);
    let db_dir = ws_dir.join("lancedb");
    let capsules_jsonl = ws_dir.join("capsules.jsonl");

    let t = now_ms();
    cfg.path_index.insert(root_str.clone(), id.clone());
    cfg.workspaces.insert(
        id.clone(),
        WorkspaceInfo {
            id: id.clone(),
            root: root_str,
            source,
            db_dir: db_dir.to_string_lossy().to_string(),
            capsules_jsonl: capsules_jsonl.to_string_lossy().to_string(),
            created_ts_ms: t,
            updated_ts_ms: t,
        },
    );
    let _ = save_workspace_config(&cfg);

    Ok(WorkspacePaths {
        id,
        db_dir,
        capsules_jsonl,
        metrics_jsonl: ws_dir.join("metrics.jsonl"),
    })
}

pub(crate) fn clear_workspace(workspace_root: &std::path::Path, yes: bool) -> anyhow::Result<()> {
    let root = git_toplevel(workspace_root).unwrap_or_else(|| {
        canonicalize_dir(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf())
    });
    let root = canonicalize_dir(&root)?;
    let root_str = root.to_string_lossy().to_string();

    let mut cfg = load_workspace_config();

    let workspace_id = cfg
        .path_index
        .get(&root_str)
        .cloned()
        .or_else(|| compute_workspace_id(&root).map(|(id, _src)| id));

    let Some(workspace_id) = workspace_id else {
        println!("No workspace mapping found for: {root_str}");
        return Ok(());
    };

    let ws_dir = unlost_workspace_dir(&workspace_id);

    if !yes {
        if !std::io::stdin().is_terminal() {
            anyhow::bail!("refusing to clear without --yes in non-interactive mode");
        }

        let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let (warn_on, warn_off) = if use_color {
            ("\x1b[33;1m", "\x1b[0m")
        } else {
            ("", "")
        };
        let (dim_on, dim_off) = if use_color {
            ("\x1b[2m", "\x1b[0m")
        } else {
            ("", "")
        };

        println!("{warn_on}This will permanently delete unlost data{warn_off}");
        println!("workspace: {workspace_id}");
        println!("{dim_on}path:{dim_off} {root_str}");
        println!("{dim_on}data:{dim_off} {}", ws_dir.display());
        print!("{warn_on}Continue?{warn_off} [y/N]: ");
        std::io::stdout().flush().ok();

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        let ans = line.trim().to_ascii_lowercase();
        if ans != "y" && ans != "yes" {
            println!("aborted");
            return Ok(());
        }
    }

    if ws_dir.exists() {
        std::fs::remove_dir_all(&ws_dir)
            .with_context(|| format!("failed to delete {}", ws_dir.display()))?;
        println!("deleted: {}", ws_dir.display());
    } else {
        println!(
            "no data dir for workspace {workspace_id} (expected {})",
            ws_dir.display()
        );
    }

    // Remove config mappings pointing to this id.
    cfg.workspaces.remove(&workspace_id);
    cfg.path_index.retain(|_k, v| v != &workspace_id);
    save_workspace_config(&cfg)?;

    println!("cleared workspace: {workspace_id}");
    Ok(())
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
            // Rust 2024: modifying process env is `unsafe` (can race with other threads).
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
    fn test_xdg_config_home() {
        let _lock = ENV_LOCK.lock().unwrap();

        // Test with XDG_CONFIG_HOME set
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());
        assert_eq!(xdg_config_home(), temp_dir.path());

        // Test fallback to HOME/.config
        let home_dir = TempDir::new().unwrap();
        let _h = EnvVarGuard::set("HOME", home_dir.path().as_os_str());
        let _r = EnvVarGuard::remove("XDG_CONFIG_HOME");
        assert_eq!(xdg_config_home(), home_dir.path().join(".config"));
    }

    #[test]
    fn test_xdg_data_home() {
        let _lock = ENV_LOCK.lock().unwrap();

        // Test with XDG_DATA_HOME set
        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_DATA_HOME", temp_dir.path().as_os_str());
        assert_eq!(xdg_data_home(), temp_dir.path());

        // Test fallback to HOME/.local/share
        let home_dir = TempDir::new().unwrap();
        let _h = EnvVarGuard::set("HOME", home_dir.path().as_os_str());
        let _r = EnvVarGuard::remove("XDG_DATA_HOME");
        assert_eq!(
            xdg_data_home(),
            home_dir.path().join(".local").join("share")
        );
    }

    #[test]
    fn test_now_ms() {
        let t1 = now_ms();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t2 = now_ms();
        assert!(t2 >= t1);
    }

    #[test]
    fn test_normalize_git_remote() {
        assert_eq!(
            normalize_git_remote("git@github.com:user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("ssh://git@github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://user@github.com/user/repo.git"),
            "github.com/user/repo"
        );
        assert_eq!(
            normalize_git_remote("https://github.com/user/repo"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_compute_hash16() {
        let hash1 = compute_hash16("test");
        let hash2 = compute_hash16("test");
        let hash3 = compute_hash16("different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_extract_project_name_from_meta_files() {
        let package_json = r#"
        {
            "name": "my-project",
            "version": "1.0.0"
        }
        "#;
        let meta = vec![("package_json".to_string(), package_json.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-project".to_string())
        );

        let pyproject = r#"
        [project]
        name = "my-py-project"
        version = "0.1.0"
        "#;
        let meta = vec![("pyproject".to_string(), pyproject.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-py-project".to_string())
        );

        let cargo_toml = r#"
        [package]
        name = "my-rust-project"
        version = "0.1.0"
        "#;
        let meta = vec![("cargo_toml".to_string(), cargo_toml.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("my-rust-project".to_string())
        );

        let go_mod = r#"
module github.com/user/my-go-project
        "#;
        let meta = vec![("go_mod".to_string(), go_mod.to_string())];
        assert_eq!(
            extract_project_name_from_meta_files(&meta),
            Some("github.com/user/my-go-project".to_string())
        );
    }

    #[test]
    fn test_compute_workspace_id() {
        // Test with a mock directory that doesn't exist
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path();

        let (id, source) = compute_workspace_id(workspace_root).unwrap();
        assert!(id.starts_with("wks_"));
        assert_eq!(source, "label");

        // Test that the same path produces the same ID
        let (id2, source2) = compute_workspace_id(workspace_root).unwrap();
        assert_eq!(id, id2);
        assert_eq!(source, source2);
    }

    #[test]
    fn test_load_and_save_workspace_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let temp_dir = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());

        // Load non-existent config
        let config = load_workspace_config();
        assert_eq!(config.version, 1);
        assert!(config.path_index.is_empty());
        assert!(config.workspaces.is_empty());
        assert!(config.llm.is_none());

        // Save and load config
        let mut new_config = config.clone();
        new_config.version = 2;
        new_config
            .path_index
            .insert("/test/path".to_string(), "test_id".to_string());
        save_workspace_config(&new_config).unwrap();

        let loaded = load_workspace_config();
        assert_eq!(loaded.version, 2);
        assert!(loaded.path_index.contains_key("/test/path"));
    }

    #[test]
    fn test_unlost_paths() {
        let _lock = ENV_LOCK.lock().unwrap();

        let temp_dir = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("XDG_CONFIG_HOME", temp_dir.path().as_os_str());
        let _g2 = EnvVarGuard::set("XDG_DATA_HOME", temp_dir.path().as_os_str());

        let config_path = unlost_config_path();
        assert!(config_path.starts_with(temp_dir.path()));
        assert!(config_path.ends_with("unlost/config.json"));

        let data_root = unlost_data_root();
        assert!(data_root.starts_with(temp_dir.path()));
        assert!(data_root.ends_with("unlost"));

        let workspace_dir = unlost_workspace_dir("test_id");
        assert!(workspace_dir.starts_with(&data_root));
        assert!(workspace_dir.ends_with("test_id"));
    }
}
