# agents/

Vendor-specific integrations that make unlost "just work" without manual setup.

This directory contains agent-specific shims that translate between each agent's protocol and unlost's core flow.

## Current targets

### Claude (`agents/claude/`)

Claude hooks integration. Configures `UserPromptSubmit` and `Stop` hooks that invoke `unlost shim claude` for friction detection and transcript ingestion.

### OpenCode (`agents/opencode/`)

OpenCode plugin integration. Configures `opencode.json` to load the unlost plugin, which runs `unlost shim opencode` over stdio.

### Cowork (`agents/cowork/`)

Claude Cowork plugin integration. A plugin package (hooks + MCP connector) installed via the Cowork plugin UI. Uses `UserPromptSubmit` and `Stop` hooks — the same transcript format as Claude Code — so the implementation shares the same parsing pipeline.
