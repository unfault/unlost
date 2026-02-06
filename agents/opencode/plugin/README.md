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
Future versions can optionally forward events to a local unlost companion for richer analysis.
