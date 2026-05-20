use crate::cli::{AgentCommand, ConfigCommand, LlmCommand};

use crate::config::LlmConfig;

fn ensure_object(v: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    if !v.is_object() {
        *v = serde_json::Value::Object(serde_json::Map::new());
    }
    v.as_object_mut().unwrap()
}

fn set_nested_string(root: &mut serde_json::Value, path: &[&str], value: String) {
    let mut cur = root;
    for (i, key) in path.iter().enumerate() {
        let is_last = i + 1 == path.len();
        let obj = ensure_object(cur);
        if is_last {
            obj.insert((*key).to_string(), serde_json::Value::String(value));
            return;
        }
        cur = obj
            .entry((*key).to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    }
}

fn ensure_top_level_string_array_contains(root: &mut serde_json::Value, key: &str, value: &str) {
    let obj = ensure_object(root);
    let entry = obj
        .entry(key.to_string())
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));

    if !entry.is_array() {
        *entry = serde_json::Value::Array(Vec::new());
    }
    let arr = entry.as_array_mut().unwrap();
    let already = arr.iter().any(|v| v.as_str() == Some(value));
    if !already {
        arr.push(serde_json::Value::String(value.to_string()));
    }
}

fn handle_llm_command(cmd: LlmCommand) -> anyhow::Result<()> {
    match cmd {
        LlmCommand::Openai {
            api_key,
            base_url,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Openai {
                api_key,
                base_url,
                model,
            }))?;
            println!("LLM provider set to OpenAI");
        }
        LlmCommand::Anthropic {
            api_key,
            base_url,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Anthropic {
                api_key,
                base_url,
                model,
            }))?;
            println!("LLM provider set to Anthropic");
        }
        LlmCommand::Ollama { base_url, model } => {
            crate::llm::set_llm_config(Some(LlmConfig::Ollama { base_url, model }))?;
            println!("LLM provider set to Ollama");
        }
        LlmCommand::Custom {
            base_url,
            api_key,
            model,
        } => {
            crate::llm::set_llm_config(Some(LlmConfig::Custom {
                base_url,
                api_key,
                model,
            }))?;
            println!("LLM provider set to custom endpoint");
        }
        LlmCommand::Show => {
            crate::llm::show_llm_config();
        }
        LlmCommand::Remove => {
            crate::llm::set_llm_config(None)?;
            println!("LLM configuration removed");
        }
    }
    Ok(())
}

fn handle_agent_command(cmd: AgentCommand) -> anyhow::Result<()> {
    match cmd {
        AgentCommand::Opencode {
            path,
            plugin,
            global,
        } => {
            let cfg_path = if global {
                opencode_global_config_path()?
            } else {
                let root = crate::workspace::git_toplevel(std::path::Path::new(&path))
                    .unwrap_or_else(|| {
                        crate::workspace::canonicalize_dir(std::path::Path::new(&path))
                            .unwrap_or_else(|_| std::path::PathBuf::from(&path))
                    });
                let root = crate::workspace::canonicalize_dir(&root)?;
                root.join("opencode.json")
            };

            if let Some(parent) = cfg_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut json = match std::fs::read_to_string(&cfg_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => return Err(e.into()),
            };

            set_nested_string(
                &mut json,
                &["$schema"],
                "https://opencode.ai/config.json".to_string(),
            );
            ensure_top_level_string_array_contains(&mut json, "plugin", &plugin);

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;

            let scope = if global { "global" } else { "project" };
            println!("configured ({scope}): {}", cfg_path.display());

            write_opencode_skill(&cfg_path, global)?;
            write_opencode_walkthrough_skill(&cfg_path, global)?;
            write_opencode_pr_comment_command(&cfg_path, global)?;
        }

        AgentCommand::Claude { path, global } => {
            configure_claude(&path, global)?;
        }

        AgentCommand::Copilot { path } => {
            configure_copilot(&path)?;
        }

        AgentCommand::Cowork { path, global } => {
            configure_cowork(&path, global)?;
        }

        AgentCommand::Mcp {
            target,
            path,
            global,
            allow_writes,
        } => {
            configure_mcp(&target, &path, global, allow_writes)?;
        }
    }
    Ok(())
}

fn opencode_global_config_path() -> anyhow::Result<std::path::PathBuf> {
    let dir = if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        std::path::PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
            .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
        home.join(".config")
    };

    Ok(dir.join("opencode").join("opencode.json"))
}

fn configure_claude(path: &str, global: bool) -> anyhow::Result<()> {
    let cfg_path = if global {
        // Global: ~/.claude/settings.json
        let home = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
            .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
        home.join(".claude").join("settings.json")
    } else {
        // Per-project: <project>/.claude/settings.json
        let root =
            crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
                crate::workspace::canonicalize_dir(std::path::Path::new(path))
                    .unwrap_or_else(|_| std::path::PathBuf::from(path))
            });
        let root = crate::workspace::canonicalize_dir(&root)?;
        root.join(".claude").join("settings.json")
    };

    // Ensure parent directory exists
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load existing config or start fresh
    let mut json = match std::fs::read_to_string(&cfg_path) {
        Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            serde_json::Value::Object(serde_json::Map::new())
        }
        Err(e) => return Err(e.into()),
    };

    // Build the hooks config
    let unlost_hook = serde_json::json!({
        "type": "command",
        "command": "unlost shim claude"
    });

    let unlost_hook_async = serde_json::json!({
        "type": "command",
        "command": "unlost shim claude",
        "async": true
    });

    // Ensure hooks object exists
    let obj = ensure_object(&mut json);
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let hooks_obj = ensure_object(hooks);

    // Add UserPromptSubmit hook
    let user_prompt_submit = hooks_obj
        .entry("UserPromptSubmit")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    add_unlost_hook_if_missing(user_prompt_submit, unlost_hook);

    // Add Stop hook
    let stop = hooks_obj
        .entry("Stop")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    add_unlost_hook_if_missing(stop, unlost_hook_async);

    // Migrate any legacy hook command strings in-place.
    if let Some(v) = hooks_obj.get_mut("UserPromptSubmit") {
        rewrite_unlost_hook_commands(v);
    }
    if let Some(v) = hooks_obj.get_mut("Stop") {
        rewrite_unlost_hook_commands(v);
    }

    // Write back
    let rendered = serde_json::to_string_pretty(&json)?;
    std::fs::write(&cfg_path, rendered)?;

    let scope = if global { "global" } else { "project" };
    println!("configured ({scope}): {}", cfg_path.display());

    write_claude_walkthrough_skill(&cfg_path)?;
    Ok(())
}

/// Write the unlost plugin package for Claude Cowork.
///
/// Cowork uses a plugin directory (not a settings JSON like Claude Code).
/// The plugin is written to:
///   global:      ~/.config/claude/plugins/unlost/
///   per-project: <project>/.claude/plugins/unlost/
///
/// The user loads it via Customize → Plugins → install from file in the Cowork UI.
fn configure_cowork(path: &str, global: bool) -> anyhow::Result<()> {
    let plugin_dir = if global {
        let home = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
            .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
        home.join(".config").join("claude").join("plugins").join("unlost")
    } else {
        let root =
            crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
                crate::workspace::canonicalize_dir(std::path::Path::new(path))
                    .unwrap_or_else(|_| std::path::PathBuf::from(path))
            });
        let root = crate::workspace::canonicalize_dir(&root)?;
        root.join(".claude").join("plugins").join("unlost")
    };

    std::fs::create_dir_all(plugin_dir.join("hooks"))?;

    // plugin.json — manifest
    let plugin_json = serde_json::json!({
        "name": "unlost",
        "version": "1.0.0",
        "description": "Unlost memory for Claude Cowork: friction detection and automatic session recording.",
        "author": "unfault"
    });
    std::fs::write(
        plugin_dir.join("plugin.json"),
        serde_json::to_string_pretty(&plugin_json)?,
    )?;

    // hooks/hooks.json — UserPromptSubmit (friction) + Stop (record)
    let hooks_json = serde_json::json!({
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "unlost shim cowork",
                            "timeout": 10
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "unlost shim cowork",
                            "async": true,
                            "timeout": 60
                        }
                    ]
                }
            ]
        }
    });
    std::fs::write(
        plugin_dir.join("hooks").join("hooks.json"),
        serde_json::to_string_pretty(&hooks_json)?,
    )?;

    // .mcp.json — registers unlost MCP server as a connector
    let mcp_json = serde_json::json!({
        "mcpServers": {
            "unlost": {
                "type": "stdio",
                "command": "unlost",
                "args": ["mcp", "serve", "--allow-writes"]
            }
        }
    });
    std::fs::write(
        plugin_dir.join(".mcp.json"),
        serde_json::to_string_pretty(&mcp_json)?,
    )?;

    let scope = if global { "global" } else { "project" };
    println!("configured ({scope}): {}", plugin_dir.display());
    println!();
    println!("Next: open Cowork → Customize → Plugins → install from file");
    println!("      and select: {}", plugin_dir.display());

    Ok(())
}

/// Add an unlost hook to a hook array if not already present
fn add_unlost_hook_if_missing(hook_array: &mut serde_json::Value, hook: serde_json::Value) {
    if !hook_array.is_array() {
        *hook_array = serde_json::Value::Array(Vec::new());
    }
    let arr = hook_array.as_array_mut().unwrap();

    // Check if unlost hook already exists
    let has_unlost = arr.iter().any(|entry| {
        if let Some(hooks) = entry.get("hooks").and_then(|h| h.as_array()) {
            hooks.iter().any(|h| {
                h.get("command")
                    .and_then(|c| c.as_str())
                    .map(|c| {
                        c.contains("unlost shim claude") || c.contains("unlost shim claudecode")
                    })
                    .unwrap_or(false)
            })
        } else {
            false
        }
    });

    if !has_unlost {
        arr.push(serde_json::json!({
            "hooks": [hook]
        }));
    }
}

fn rewrite_unlost_hook_commands(hook_array: &mut serde_json::Value) {
    let Some(arr) = hook_array.as_array_mut() else {
        return;
    };
    for entry in arr.iter_mut() {
        let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) else {
            continue;
        };
        for h in hooks.iter_mut() {
            let Some(cmd) = h.get_mut("command") else {
                continue;
            };
            let Some(s) = cmd.as_str() else {
                continue;
            };
            if s.trim() == "unlost shim claudecode" {
                *cmd = serde_json::Value::String("unlost shim claude".to_string());
            }
        }
    }
}

const OPENCODE_SKILL_CONTENT: &str = r#"---
name: unlost
description: Query project memory, retrieve past decisions, and check agent orientation via the unlost capsule store
compatibility: opencode
---

## What unlost does

Unlost runs silently as an OpenCode plugin. Before each of your prompts, it checks
for friction patterns (drift, retry spirals, false progress, unbounded scope) and
injects a warning when something is off. After each exchange, it extracts a small
capsule — intent, decision, rationale, key symbols — and stores it locally.

You do not need to invoke unlost for friction detection. It runs automatically.

## Querying project memory

There are two tiers. The fast path you run proactively. The LLM path you run only
when the user explicitly asks for a deep recall or narrative summary.

### Fast path — run proactively (no LLM, instant)

These commands are safe to run at any time without asking the user. They are
entirely local: no LLM call, no network, no meaningful latency.

| Command | What it returns |
|---------|-----------------|
| `unlost query --no-llm "<question>"` | Raw capsule hits from vector search, ranked by relevance |
| `unlost metrics` | Friction hotspots, loop patterns, verbosity trends from local metrics log |

Run these when:
- You are about to work on something and want to check if a relevant decision exists
- You want to orient yourself after a context gap
- You need to know *if* something was recorded, not a full explanation

### LLM path — on demand only (narrative, slower)

These commands call the configured LLM and should only be run when the user
explicitly asks for a recall, summary, or brief. Do not run them proactively.

| Command | What it returns |
|---------|-----------------|
| `unlost query "<question>"` | Narrative answer grounded in matching capsules |
| `unlost recall` | Chronological decision trail with LLM-written summaries |
| `unlost brief` | Structured brief: what happened, key decisions, what's next |

Run these when the user says things like:
- "catch me up", "what did we decide about X", "summarize what happened"
- "give me a brief before we continue"
- "why did we do Y"

## Examples

```bash
# Proactive: check if a topic was ever decided before starting work
unlost query --no-llm "error handling strategy"

# Proactive: inspect friction and loop patterns
unlost metrics

# On demand: explain a past decision (user asked)
unlost query "why did we switch to lancedb"

# On demand: full decision trail (user asked to be caught up)
unlost recall

# On demand: structured brief before resuming (user asked)
unlost brief
```

## Notes

- All data is stored locally. No transcripts leave the machine.
- Capsules are workspace-scoped (git toplevel).
- If unlost is not configured for this project, run: `unlost config agent opencode`
"#;

const OPENCODE_WALKTHROUGH_SKILL_CONTENT: &str = r#"---
name: unlost-walkthrough
description: Guide the user through recent changes step-by-step (open files, explain why)
compatibility: opencode
---

## Trigger

User invokes: `/unlost-walkthrough`

## What to do

Walk the user through what you did in this session (or on this branch) so they can follow the changes more intimately.

Guidelines:

- Prefer a step-by-step flow. After each step, pause and wait for the user to say `next`.
- For each significant change, open the relevant file in VSCode at the most relevant line using:
  - `code --goto <file>:<line>`
- Within a file, progress through the change in order (top-to-bottom): jump to the next relevant hunk/region and explain it before moving on. If you can't programmatically highlight lines, explicitly tell the user which line range to highlight.
- Explain **what changed** and **why** (intent + implications). Keep it concrete.
- Prioritize files the user is least likely to know (entry points, core modules, config) before low-impact or mechanical edits.
- If `code` is not available, still provide a precise navigation hint like `path/to/file:line`.
- Avoid dumping huge diffs; show only small excerpts when helpful.
- Do not reveal secrets (tokens, credentials, `.env` contents). Redact if needed.

## Suggested interaction

Ask the user where to start if ambiguous ("start with the high-level flow or the riskiest change?"). Otherwise, start with the most impactful change.
"#;

fn write_opencode_skill(cfg_path: &std::path::Path, global: bool) -> anyhow::Result<()> {
    let skills_dir = if global {
        // cfg_path is ~/.config/opencode/opencode.json
        // skill goes to ~/.config/opencode/skills/unlost/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine opencode config directory"))?
            .join("skills")
            .join("unlost")
    } else {
        // cfg_path is <git-root>/opencode.json
        // skill goes to <git-root>/.opencode/skills/unlost/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine project root"))?
            .join(".opencode")
            .join("skills")
            .join("unlost")
    };

    let skill_path = skills_dir.join("SKILL.md");

    if skill_path.exists() {
        print!(
            "SKILL.md already exists at {}. Overwrite? [y/N] ",
            skill_path.display()
        );
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("skill: skipped");
            return Ok(());
        }
    }

    std::fs::create_dir_all(&skills_dir)?;
    std::fs::write(&skill_path, OPENCODE_SKILL_CONTENT)?;
    println!("skill: {}", skill_path.display());
    Ok(())
}

fn write_opencode_walkthrough_skill(
    cfg_path: &std::path::Path,
    global: bool,
) -> anyhow::Result<()> {
    let skills_dir = if global {
        // cfg_path is ~/.config/opencode/opencode.json
        // skill goes to ~/.config/opencode/skills/unlost-walkthrough/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine opencode config directory"))?
            .join("skills")
            .join("unlost-walkthrough")
    } else {
        // cfg_path is <git-root>/opencode.json
        // skill goes to <git-root>/.opencode/skills/unlost-walkthrough/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine project root"))?
            .join(".opencode")
            .join("skills")
            .join("unlost-walkthrough")
    };

    let skill_path = skills_dir.join("SKILL.md");

    if skill_path.exists() {
        print!(
            "SKILL.md already exists at {}. Overwrite? [y/N] ",
            skill_path.display()
        );
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("skill: skipped");
            return Ok(());
        }
    }

    std::fs::create_dir_all(&skills_dir)?;
    std::fs::write(&skill_path, OPENCODE_WALKTHROUGH_SKILL_CONTENT)?;
    println!("skill: {}", skill_path.display());
    Ok(())
}

const OPENCODE_PR_COMMENT_COMMAND_CONTENT: &str = r#"---
description: Run unlost pr-comment on a PR
---
Please run the `unlost pr-comment` command for the provided PR:

unlost pr-comment $ARGUMENTS
"#;

fn write_opencode_pr_comment_command(
    cfg_path: &std::path::Path,
    global: bool,
) -> anyhow::Result<()> {
    let commands_dir = if global {
        // cfg_path is ~/.config/opencode/opencode.json
        // command goes to ~/.config/opencode/commands/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine opencode config directory"))?
            .join("commands")
    } else {
        // cfg_path is <git-root>/opencode.json
        // command goes to <git-root>/.opencode/commands/
        cfg_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("could not determine project root"))?
            .join(".opencode")
            .join("commands")
    };

    let command_path = commands_dir.join("unlost-pr-comment.md");

    if command_path.exists() {
        print!(
            "unlost-pr-comment.md already exists at {}. Overwrite? [y/N] ",
            command_path.display()
        );
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("command: skipped");
            return Ok(());
        }
    }

    std::fs::create_dir_all(&commands_dir)?;
    std::fs::write(&command_path, OPENCODE_PR_COMMENT_COMMAND_CONTENT)?;
    println!("command: {}", command_path.display());
    Ok(())
}

const CLAUDE_WALKTHROUGH_SKILL_CONTENT: &str = r#"---
name: unlost-walkthrough
description: Guide the user through recent changes step-by-step (open files, explain why)
compatibility: claude
---

## Trigger

User invokes: `/unlost-walkthrough`

## What to do

Walk the user through what you did in this session (or on this branch) so they can follow the changes more intimately.

Guidelines:

- Prefer a step-by-step flow. After each step, pause and wait for the user to say `next`.
- For each significant change, open the relevant file in VSCode at the most relevant line using:
  - `code --goto <file>:<line>`
- Within a file, progress through the change in order (top-to-bottom): jump to the next relevant hunk/region and explain it before moving on. If you can't programmatically highlight lines, explicitly tell the user which line range to highlight.
- Explain **what changed** and **why** (intent + implications). Keep it concrete.
- Prioritize files the user is least likely to know (entry points, core modules, config) before low-impact or mechanical edits.
- If `code` is not available, still provide a precise navigation hint like `path/to/file:line`.
- Avoid dumping huge diffs; show only small excerpts when helpful.
- Do not reveal secrets (tokens, credentials, `.env` contents). Redact if needed.

## Suggested interaction

Ask the user where to start if ambiguous ("start with the high-level flow or the riskiest change?"). Otherwise, start with the most impactful change.
"#;

fn write_claude_walkthrough_skill(cfg_path: &std::path::Path) -> anyhow::Result<()> {
    let claude_dir = cfg_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("could not determine claude config directory"))?;
    let skills_dir = claude_dir.join("skills").join("unlost-walkthrough");
    let skill_path = skills_dir.join("SKILL.md");

    if skill_path.exists() {
        print!(
            "SKILL.md already exists at {}. Overwrite? [y/N] ",
            skill_path.display()
        );
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("skill: skipped");
            return Ok(());
        }
    }

    std::fs::create_dir_all(&skills_dir)?;
    std::fs::write(&skill_path, CLAUDE_WALKTHROUGH_SKILL_CONTENT)?;
    println!("skill: {}", skill_path.display());
    Ok(())
}

// ============================================================================
// GitHub Copilot CLI configuration
// ============================================================================

const COPILOT_SKILL_CONTENT: &str = r#"---
name: unlost
description: Query project memory, retrieve past decisions, and check agent orientation via the unlost capsule store
compatibility: copilot
---

## What unlost does

Unlost runs silently as a Copilot CLI hook. At session start it checks for
friction patterns (drift, retry spirals, false progress, unbounded scope) and
logs a warning when something is off. At session end it extracts the full
exchange from the session transcript and stores capsules locally.

You do not need to invoke unlost for friction detection. It runs automatically.

## Querying project memory

There are two tiers. The fast path you run proactively. The LLM path you run only
when the user explicitly asks for a deep recall or narrative summary.

### Fast path — run proactively (no LLM, instant)

These commands are safe to run at any time without asking the user. They are
entirely local: no LLM call, no network, no meaningful latency.

| Command | What it returns |
|---------|-----------------|
| `unlost query --no-llm "<question>"` | Raw capsule hits from vector search, ranked by relevance |
| `unlost metrics` | Friction hotspots, loop patterns, verbosity trends from local metrics log |

Run these when:
- You are about to work on something and want to check if a relevant decision exists
- You want to orient yourself after a context gap
- You need to know *if* something was recorded, not a full explanation

### LLM path — on demand only (narrative, slower)

These commands call the configured LLM and should only be run when the user
explicitly asks for a recall, summary, or brief. Do not run them proactively.

| Command | What it returns |
|---------|-----------------|
| `unlost query "<question>"` | Narrative answer grounded in matching capsules |
| `unlost recall` | Chronological decision trail with LLM-written summaries |
| `unlost brief` | Structured brief: what happened, key decisions, what's next |

Run these when the user says things like:
- "catch me up", "what did we decide about X", "summarize what happened"
- "give me a brief before we continue"
- "why did we do Y"

## Examples

```bash
# Proactive: check if a topic was ever decided before starting work
unlost query --no-llm "error handling strategy"

# Proactive: inspect friction and loop patterns
unlost metrics

# On demand: explain a past decision (user asked)
unlost query "why did we switch to lancedb"

# On demand: full decision trail (user asked to be caught up)
unlost recall

# On demand: structured brief before resuming (user asked)
unlost brief
```

## Notes

- All data is stored locally. No transcripts leave the machine.
- Capsules are workspace-scoped (git toplevel).
- If unlost is not configured for this project, run: `unlost config agent copilot`
- Friction detection output is currently logged for metrics only; Copilot CLI
  does not surface `userPromptSubmitted` hook output to the conversation.
"#;

fn configure_copilot(path: &str) -> anyhow::Result<()> {
    let root = crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
        crate::workspace::canonicalize_dir(std::path::Path::new(path))
            .unwrap_or_else(|_| std::path::PathBuf::from(path))
    });
    let root = crate::workspace::canonicalize_dir(&root)?;

    // Write .github/hooks/unlost.json
    let hooks_dir = root.join(".github").join("hooks");
    std::fs::create_dir_all(&hooks_dir)?;
    let hooks_path = hooks_dir.join("unlost.json");

    let hooks_json = serde_json::json!({
        "version": 1,
        "hooks": {
            "sessionStart": [
                {
                    "type": "command",
                    "bash": "unlost shim copilot",
                    "timeoutSec": 30
                }
            ],
            "userPromptSubmitted": [
                {
                    "type": "command",
                    "bash": "unlost shim copilot",
                    "timeoutSec": 15
                }
            ],
            "sessionEnd": [
                {
                    "type": "command",
                    "bash": "unlost shim copilot",
                    "timeoutSec": 60
                }
            ]
        }
    });

    let rendered = serde_json::to_string_pretty(&hooks_json)?;
    std::fs::write(&hooks_path, rendered)?;
    println!("hooks: {}", hooks_path.display());

    // Write skill
    write_copilot_skill(&root)?;

    Ok(())
}

fn write_copilot_skill(project_root: &std::path::Path) -> anyhow::Result<()> {
    // Copilot CLI loads skills from .github/copilot/skills/<name>/
    let skills_dir = project_root
        .join(".github")
        .join("copilot")
        .join("skills")
        .join("unlost");
    let skill_path = skills_dir.join("SKILL.md");

    if skill_path.exists() {
        print!(
            "SKILL.md already exists at {}. Overwrite? [y/N] ",
            skill_path.display()
        );
        use std::io::Write as _;
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("skill: skipped");
            return Ok(());
        }
    }

    std::fs::create_dir_all(&skills_dir)?;
    std::fs::write(&skill_path, COPILOT_SKILL_CONTENT)?;
    println!("skill: {}", skill_path.display());
    Ok(())
}

fn configure_mcp(target: &str, path: &str, global: bool, allow_writes: bool) -> anyhow::Result<()> {
    let writes_flag = if allow_writes { " --allow-writes" } else { "" };
    let serve_cmd = format!("unlost mcp serve{writes_flag}");

    match target {
        "opencode" => {
            let cfg_path = if global {
                opencode_global_config_path()?
            } else {
                let root = crate::workspace::git_toplevel(std::path::Path::new(path))
                    .unwrap_or_else(|| {
                        crate::workspace::canonicalize_dir(std::path::Path::new(path))
                            .unwrap_or_else(|_| std::path::PathBuf::from(path))
                    });
                let root = crate::workspace::canonicalize_dir(&root)?;
                root.join("opencode.json")
            };

            if let Some(parent) = cfg_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut json = match std::fs::read_to_string(&cfg_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => return Err(e.into()),
            };

            let obj = ensure_object(&mut json);
            let mcp = obj
                .entry("mcp")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            let mcp_obj = ensure_object(mcp);
            let mut cmd_args = vec!["unlost", "mcp", "serve"];
            if allow_writes {
                cmd_args.push("--allow-writes");
            }
            mcp_obj.insert(
                "unlost".to_string(),
                serde_json::json!({
                    "type": "local",
                    "command": cmd_args
                }),
            );

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;
            let scope = if global { "global" } else { "project" };
            println!("configured ({scope}): {}", cfg_path.display());
        }

        "claude" => {
            let cfg_path = if global {
                let home = std::env::var("HOME")
                    .map(std::path::PathBuf::from)
                    .or_else(|_| std::env::var("USERPROFILE").map(std::path::PathBuf::from))
                    .map_err(|_| anyhow::anyhow!("could not determine home directory"))?;
                home.join(".claude.json")
            } else {
                let root =
                    crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
                        crate::workspace::canonicalize_dir(std::path::Path::new(path))
                            .unwrap_or_else(|_| std::path::PathBuf::from(path))
                    });
                let root = crate::workspace::canonicalize_dir(&root)?;
                root.join(".claude").join("settings.local.json")
            };

            if let Some(parent) = cfg_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut json = match std::fs::read_to_string(&cfg_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => return Err(e.into()),
            };

            let obj = ensure_object(&mut json);
            let mcp = obj
                .entry("mcpServers")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            let mcp_obj = ensure_object(mcp);
            let mut args = vec!["mcp", "serve"];
            if allow_writes {
                args.push("--allow-writes");
            }
            mcp_obj.insert(
                "unlost".to_string(),
                serde_json::json!({
                    "command": "unlost",
                    "args": args
                }),
            );

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&cfg_path, rendered)?;
            let scope = if global { "global" } else { "project" };
            println!("configured ({scope}): {}", cfg_path.display());
        }

        "copilot" => {
            // Copilot MCP: .copilot/mcp.json in the workspace root
            let root = crate::workspace::git_toplevel(std::path::Path::new(path)).unwrap_or_else(|| {
                crate::workspace::canonicalize_dir(std::path::Path::new(path))
                    .unwrap_or_else(|_| std::path::PathBuf::from(path))
            });
            let root = crate::workspace::canonicalize_dir(&root)?;
            let mcp_dir = root.join(".copilot");
            std::fs::create_dir_all(&mcp_dir)?;
            let mcp_path = mcp_dir.join("mcp.json");

            let mut json = match std::fs::read_to_string(&mcp_path) {
                Ok(s) => serde_json::from_str::<serde_json::Value>(&s)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => return Err(e.into()),
            };

            let obj = ensure_object(&mut json);
            let servers = obj
                .entry("servers")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            let servers_obj = ensure_object(servers);
            let mut args = vec!["mcp", "serve"];
            if allow_writes {
                args.push("--allow-writes");
            }
            servers_obj.insert(
                "unlost".to_string(),
                serde_json::json!({
                    "command": "unlost",
                    "args": args
                }),
            );

            let rendered = serde_json::to_string_pretty(&json)?;
            std::fs::write(&mcp_path, rendered)?;
            println!("configured (project): {}", mcp_path.display());
        }

        _ => {
            // Generic: print the snippet to stdout
            println!("# Add to your MCP host's config:");
            println!();
            println!("## Claude Code (~/.claude.json or .claude/settings.local.json):");
            println!();
            println!(r#"  "mcpServers": {{"#);
            println!(r#"    "unlost": {{"#);
            println!(r#"      "command": "unlost","#);
            let args_str = if allow_writes {
                r#"      "args": ["mcp", "serve", "--allow-writes"]"#
            } else {
                r#"      "args": ["mcp", "serve"]"#
            };
            println!("{args_str}");
            println!(r#"    }}"#);
            println!(r#"  }}"#);
            println!();
            println!("## OpenCode (opencode.json or ~/.config/opencode/opencode.json):");
            println!();
            println!(r#"  "mcp": {{"#);
            println!(r#"    "unlost": {{"#);
            println!(r#"      "type": "local","#);
            println!(r#"      "command": {serve_cmd:?}"#);
            println!(r#"    }}"#);
            println!(r#"  }}"#);
        }
    }

    Ok(())
}

pub fn run(command: ConfigCommand) -> anyhow::Result<()> {
    match command {
        ConfigCommand::Llm { command } => handle_llm_command(command),
        ConfigCommand::Agent { command } => handle_agent_command(command),
    }
}
