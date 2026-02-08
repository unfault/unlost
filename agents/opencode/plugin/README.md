# @unfault/opencode-unlost

OpenCode plugin for unlost/unloop.

## What it does (v0)

- Uses OpenCode's `experimental.chat.messages.transform` hook to perform pre-flight prompt injection.
- Detects basic "thrash + negative cue" loops using only the in-memory message history.

This avoids the UX footguns of:
- starting a proxy server manually
- changing provider endpoints

## Install

Add this plugin to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@unfault/opencode-unlost"]
}
```

Or via unlost:

```bash
unlost config agent opencode-plugin --path .
```

Restart OpenCode after changes.

## Notes

This initial version is intentionally dependency-free and runs locally.
The plugin spawns a local unlost shim for conversation recording and friction detection.

## Shim Integration

The plugin spawns `unlost shim opencode` and communicates via stdio JSON-RPC:

```
check  → {"method": "check", "params": {"text": "...", "directory": "..."}}
       ← {"note": "warning to inject"} or {"note": null}

record → {"method": "record", "params": {"user_text": "...", "assistant_text": "...", "directory": "..."}}
       ← {"ok": true}

record (optional usage) → {"method": "record", "params": {"user_text": "...", "assistant_text": "...", "directory": "...", "usage": {"provider_id": "...", "model_id": "...", "cost": 0.0, "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}}}}}
                     ← {"ok": true}
```

**Performance guarantee**: The `record` call returns immediately (~0ms). All heavy processing
(LLM extraction, embedding, LanceDB insert) happens asynchronously in a background task.
This ensures the agent is never blocked waiting for unlost.

**Trade-off**: Queries have eventual consistency—a capsule may not appear in search results
for a few seconds after the `record` call returns.
