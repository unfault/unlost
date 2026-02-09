# Unlost

Intercepts loops, injects guidance, and quietly records what actually happened.

## Why this exists

You know the drill:

- You ask it to rename a function. It renames it, then renames it again. Then again. You intervene: "stop renaming, just add a wrapper."
- You come back after the weekend. The agent made decisions you don't understand. Nobody remembers why.
- It says it finished. It didn't. It created a file that was never committed.

That's the babysitting tax.

## What it does

It intercepts before your agent goes off the rails:

- Checks prompts for signs of thrashing before sending them
- Injects a warning into the conversation when it detects friction
- Records user/assistant exchanges for later recall
- Runs locally, talks to `unlost shim opencode` over stdio

No proxy server, no config juggling. Just add the plugin and go.

## Install

Add to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@unfault/unlost-opencode"]
}
```

Or via the CLI:

```bash
unlost config agent opencode --path .
```

Restart OpenCode. That's it.

## How it works

The plugin speaks JSON-RPC to a companion process:

```
check    → {"method": "check", "params": {"text": "...", "directory": "..."}}
         ← {"note": "warning to inject"} or {"note": null}

record   → {"method": "record", "params": {"user_text": "...", "assistant_text": "...", "directory": "..."}}
         ← {"ok": true}
```

`check` runs before each prompt. `record` runs after each exchange.

## Performance

`record` returns immediately. Everything heavy (LLM extraction, embeddings, storage) happens in the background. Your agent never waits.

Trade-off: capsules won't show up in queries for a few seconds after recording.
