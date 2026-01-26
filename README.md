# unlost

Local-first code memory for a workspace.

`unlost` records LLM/API exchanges via a local HTTP reverse proxy, extracts small structured "capsules" (intent/decision/rationale/next steps/symbols), and stores them in a local LanceDB with local embeddings.

It does not store full transcripts; only the capsule fields and minimal request metadata.

## Install

```bash
cargo install --path .
```

## Quickstart

Run the multiplexed proxy:

```bash
unlost serve --bind 127.0.0.1:3000
```

Configure an agent/client to use the proxy base URL (example: opencode):

```bash
unlost configure agent opencode --path . --server http://127.0.0.1:3000
```

Then use your agent normally; `unlost` will store capsules as requests flow through.

## Recall and Query

```bash
unlost recall
unlost recall src/main.rs

unlost query "what did we change about chunking?"
unlost query --scope scan_capsules_lancedb "where is this used?"
```

## Inspect Stored Capsules

```bash
unlost inspect --limit 10
```

## Data Storage

- Workspace data lives under XDG data dirs (default: `~/.local/share/unlost/workspaces/<id>/`).
- Embedding and emotion model artifacts are cached under `~/.local/share/unlost/models/`.

Useful env vars:

- `UNLOST_EMBED_CACHE_DIR` (fastembed cache override)
- `UNLOST_EMOTION_CACHE_DIR` (emotion model cache override)

## Dev

```bash
cargo test
cargo build
```
