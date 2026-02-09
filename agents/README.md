# agents/

Vendor-specific integrations that make unlost "just work" without manual setup.

This directory contains agent-specific shims that translate between each agent's protocol and unlost's core flow.

## Current targets

### Claude Code (`agents/claudecode/`)

Claude Code hooks integration. Configures `UserPromptSubmit` and `Stop` hooks that invoke `unlost shim claudecode` for friction detection and transcript ingestion.

### OpenCode (`agents/opencode/`)

OpenCode plugin integration. Configures `opencode.json` to load the unlost plugin, which runs `unlost shim opencode` over stdio.
