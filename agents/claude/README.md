# Claude Integration

Unlost integrates with Claude via hooks. Two hooks are configured:

- `UserPromptSubmit`: Fires before the agent sends a prompt. Unlost checks for friction and returns guidance to inject if needed.
- `Stop`: Fires after each exchange completes. Unlost parses the transcript, extracts capsules, and records them locally.

## Setup

```bash
# Global (all projects, forever)
unlost config agent claude --global

# Per-project
unlost config agent claude --path .
```

This writes to `~/.claude/settings.json` (global) or `<project>/.claude/settings.json` (per-project).

## What gets recorded

Unlost parses Claude's transcript JSONL format and extracts user → assistant turn pairs.

- Tool use blocks (`tool_use`)
- Tool results (`tool_result`) are preserved as part of the recorded context (bounded/truncated)
- Thinking blocks (`thinking`) are ignored
- Sidechain messages (prompt suggestions, background agents)

Each turn becomes a capsule containing:

- `user_text`: The user's prompt
- `assistant_text`: The assistant's response (text content only)
- `usage`: Token usage metadata (provider, model, token counts)

## Cursor state

Unlost tracks progress per session via cursor files:

```
~/.local/share/unlost/workspaces/<ws_id>/claude/<session_id>.cursor
```

This allows incremental ingestion - only new transcript lines are processed on each `Stop` hook invocation.

## Backfill / replay

If you need to re-ingest a transcript file (e.g. after upgrading unlost), use:

```bash
unlost shim replay claude --path . --transcript-path ~/.claude/projects/<project>/<session_id>.jsonl
```

## Friction detection

Before each `UserPromptSubmit`, unlost runs a friction check against recent capsules in the workspace. If friction is detected (e.g., contradicting a previous decision), a guidance note is returned and injected into the prompt via Claude Code's `additionalContext` hook mechanism.
