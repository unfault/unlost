# unlost MCP server

Exposes unlost workspace memory as a Model Context Protocol (MCP) server, giving any MCP-aware agent direct structured access to past decisions — no LLM narration, no ANSI prose, just data.

## Quick start

```bash
unlost config agent mcp --target opencode   # OpenCode (per-project)
unlost config agent mcp --target claude     # Claude Code (per-project)
unlost config agent mcp --target copilot   # GitHub Copilot
unlost config agent mcp --target generic   # Print snippet for manual paste
```

To allow the agent to write notes into memory:

```bash
unlost config agent mcp --target claude --allow-writes
```

## Tools

| Tool | Purpose | Writes? |
|---|---|---|
| `unlost_recall` | Search workspace memory by query, file, or symbol | No |
| `unlost_trace_decision` | Causal chain of decisions leading to current state | No |
| `unlost_challenge` | Prior rationale + alternatives against a proposal | No |
| `unlost_thread` | Cross-workspace topic history (all projects) | No |
| `unlost_orient` | Recent touches + drift signal for current files/symbols | No |
| `unlost_capsule_get` | Fetch full capsule by id for citation | No |
| `unlost_note` | Record a deliberate decision (opt-in) | **Yes** |

All read tools run on the no-LLM fast path. Target: < 250ms per call on a warm workspace.

## Manual configuration

### Claude Code

**Global** (`~/.claude.json`):

```json
{
  "mcpServers": {
    "unlost": {
      "command": "unlost",
      "args": ["mcp", "serve"]
    }
  }
}
```

**Per-project** (`.claude/settings.local.json`):

```json
{
  "mcpServers": {
    "unlost": {
      "command": "unlost",
      "args": ["mcp", "serve"]
    }
  }
}
```

### OpenCode

**Global** (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "unlost": {
      "type": "local",
      "command": ["unlost", "mcp", "serve"]
    }
  }
}
```

**Per-project** (`opencode.json`):

```json
{
  "mcp": {
    "unlost": {
      "type": "local",
      "command": ["unlost", "mcp", "serve"]
    }
  }
}
```

### GitHub Copilot

**Per-project** (`.copilot/mcp.json`):

```json
{
  "servers": {
    "unlost": {
      "command": "unlost",
      "args": ["mcp", "serve"]
    }
  }
}
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--allow-writes` | off | Enable `unlost_note`. Agents can write decisions into memory. |
| `--no-cross-workspace` | off | Disable cross-workspace queries in `unlost_thread`. |
| `--workspace <path>` | `.` | Override workspace root (default: git toplevel of cwd). |

## Design notes

- **Workspace-scoped**: the server is bound to the git toplevel of the directory where it's launched. All tools except `unlost_thread` query only that workspace.
- **`unlost_thread` is cross-workspace by default**: it searches all locally indexed workspaces. Use `workspaces` input to restrict.
- **No LLM inside tools**: the calling agent has a model. Tools return structured evidence; the agent synthesizes.
- **Writes are opt-in**: `unlost_note` only appears in `tools/list` when `--allow-writes` is set. This prevents misbehaving agents from polluting memory unprompted.
- **Capsule ids**: every tool that returns capsule data includes a stable `id` field. Pass it to `unlost_capsule_get` to fetch the full capsule for citation.
