# unlost

Local-first code memory for a workspace.

I built this the way I like to build tools: steady, quiet, and useful for the long haul. It lives on your machine, keeps the signal, and tries hard not to keep the noise. Open source is the give-back.

`unlost` records LLM/API exchanges via a local HTTP reverse proxy, extracts small structured "capsules" (intent/decision/rationale/next steps/symbols), and stores them in a local LanceDB with local embeddings.

It does not store full transcripts; only capsule fields and a bit of request metadata.

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

## When To Use unlost

If you’ve ever stared at a codebase after a few days away and thought “I know we made a decision here, but where did we land?”, this is for that moment.

Good fits:

- Long-running refactors where intent changes week to week
- Multi-agent or multi-person work where decisions drift between threads
- Onboarding yourself onto a new repo (or coming back after time off)
- Keeping a lightweight trail of why a PR ended up the way it did

Less useful:

- If you want full chat transcripts (unlost intentionally doesn’t store them)
- If you need strict compliance logging (this is a developer tool, local-first)

## Architecture (A Calm Walk Through It)

At a high level, the system is a recorder plus a small memory store:

1. Recorder / proxy
   - `unlost serve` runs a single HTTP server and multiplexes workspaces by path:
     - `/w/<workspace_id>/<provider>/...`
   - It forwards requests to the upstream provider (`openai`, `anthropic`, `opencode`) and watches the request/response stream.

2. Chunking
   - Exchanges are buffered per workspace and flushed into slices.
   - Flush triggers include: short idle gaps, size bounds, turn count, and “milestone” mentions (commit/PR).

3. Capsule extraction (no transcript storage)
   - Each flushed slice is summarized into a single capsule:
     - `category`, `intent`, `decision`, `rationale`, `next_steps[]`, `symbols[]`
   - The full raw transcript is not stored.

4. Local embeddings + LanceDB
   - Capsule text is embedded locally (fastembed) and stored in a LanceDB table.
   - `query` uses nearest-neighbor search over embeddings.
   - `inspect` and `recall` read capsules back out.

5. Mood metadata (optional, local)
   - A local ONNX classifier tags a coarse mood for user/assistant turns.
   - This is stored as metadata alongside capsules and can be used to enrich recall.

## Recall and Query

```bash
unlost recall
unlost recall src/main.rs

unlost query "what did we change about chunking?"
unlost query --symbol scan_capsules_lancedb "where is this used?"
```

## Examples

Configure an LLM for narratives (query/recall), while keeping capsule extraction local-first:

```bash
unlost config llm anthropic --model claude-3-5-sonnet-20241022
```

Start the recorder (single server, many workspaces):

```bash
unlost serve --bind 127.0.0.1:3000
```

Ask for a “story so far” after a day away:

```bash
unlost recall
unlost recall src/http_proxy.rs
```

Search for a decision you half-remember:

```bash
unlost query "why did we rename the capsules table?"
```

Focus a query on one symbol you care about:

```bash
unlost query --symbol proxy_request "how does the upstream routing work?"
```

See raw stored capsules (including mood, when present):

```bash
unlost inspect --limit 10
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
