# Claude Cowork Integration

Unlost integrates with Claude Cowork via hooks and an MCP connector.

## What it does

- **`UserPromptSubmit`** — friction check before each prompt; injects a warning via `additionalContext` if a contradicting past decision is detected.
- **`Stop`** — reads the session transcript after each turn and records user ↔ assistant exchanges as capsules in the local workspace store.
- **MCP connector** — registers `unlost mcp serve` so Cowork (and the AI inside it) can query memory via `unlost_recall`, `unlost_orient`, `unlost_challenge`, etc.

## Setup

```bash
# Per-project (writes to <project>/.claude/plugins/unlost/)
unlost config agent cowork --path .

# Global (writes to ~/.config/claude/plugins/unlost/)
unlost config agent cowork --global
```

Then in Cowork: **Customize → Plugins → install from file** and select the printed directory.

Alternatively, point Cowork's marketplace at this repository and install the `unlost` plugin directly.

## Backfill

To replay an existing Cowork transcript:

```bash
unlost shim replay cowork --path . --transcript-path /path/to/session.jsonl
```

## Cursor state

Progress is tracked per session:

```
~/.local/share/unlost/workspaces/<ws_id>/cowork/<session_id>.cursor
```
