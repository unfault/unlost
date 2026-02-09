# Claude Code Integration

Unlost integrates with Claude Code via hooks. Two hooks are configured:

- `UserPromptSubmit`: Fires before the agent sends a prompt. Unlost checks for friction and returns guidance to inject if needed.
- `Stop`: Fires after each exchange completes. Unlost parses the transcript, extracts capsules, and records them locally.

## Setup

```bash
# Global (all projects, forever)
unlost config agent claudecode --global

# Per-project
unlost config agent claudecode --path .
```

This writes to `~/.claude/settings.json` (global) or `<project>/.claude/settings.json` (per-project).

## What gets recorded

Unlost parses Claude Code's transcript JSONL format and extracts user → assistant turn pairs, ignoring:

- Tool use blocks (`tool_use`)
- Tool results (`tool_result`)
- Thinking blocks (`thinking`)
- Sidechain messages (prompt suggestions, background agents)

Each turn becomes a capsule containing:

- `user_text`: The user's prompt
- `assistant_text`: The assistant's response (text content only)
- `usage`: Token usage metadata (provider, model, token counts)

## Cursor state

Unlost tracks progress per session via cursor files:

```
~/.local/share/unlost/workspaces/<ws_id>/claudecode/<session_id>.cursor
```

This allows incremental ingestion — only new transcript lines are processed on each `Stop` hook invocation.

## Friction detection

Before each `UserPromptSubmit`, unlost runs a friction check against recent capsules in the workspace. If friction is detected (e.g., contradicting a previous decision), a guidance note is returned and injected into the prompt via Claude Code's `additionalContext` hook mechanism.
