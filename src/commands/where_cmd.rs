pub(crate) fn run(path: String) -> anyhow::Result<()> {
    let workspace_root = std::path::Path::new(&path);
    let root = crate::workspace::git_toplevel(workspace_root).unwrap_or_else(|| {
        crate::workspace::canonicalize_dir(workspace_root)
            .unwrap_or_else(|_| workspace_root.to_path_buf())
    });
    let root = crate::workspace::canonicalize_dir(&root)?;
    let root_str = root.to_string_lossy().to_string();

    let cfg = crate::workspace::load_workspace_config();

    let workspace_id = cfg.path_index.get(&root_str).cloned();

    let (id, source) = if let Some(ref existing_id) = workspace_id {
        // Use the existing ID from config (handles cases where manifest files were added/removed)
        let source = cfg
            .workspaces
            .get(existing_id)
            .map(|w| w.source.clone())
            .unwrap_or_else(|| "unknown".to_string());
        (existing_id.clone(), source)
    } else {
        // Compute fresh ID since workspace not yet registered
        let (id, source) = crate::workspace::compute_workspace_id(&root)
            .ok_or_else(|| anyhow::anyhow!("unable to compute workspace id"))?;
        (id, source)
    };

    let ws_dir = crate::workspace::unlost_workspace_dir(&id);
    let is_registered = workspace_id.is_some();

    println!("Workspace ID: {id}");
    println!("Source: {source}");
    println!("Root: {root_str}");
    println!("Storage: {}", ws_dir.display());

    if !is_registered {
        println!("\nNote: workspace is not yet registered in config (run any command first)");
    }

    Ok(())
}
