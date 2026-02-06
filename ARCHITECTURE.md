# unlost - Architecture and Design Documentation

**unlost** is a local-first code memory system that captures, summarizes, and stores development decisions from LLM conversations. It acts as an intelligent proxy that records API exchanges, extracts structured "capsules" of intent and rationale, and provides semantic search and recall capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Data Flow](#data-flow)
5. [Module Reference](#module-reference)
6. [Data Models](#data-models)
7. [Storage Layer](#storage-layer)
8. [CLI Commands](#cli-commands)
9. [Configuration](#configuration)
10. [ML/AI Components](#mlai-components)
11. [Technology Stack](#technology-stack)

---

## Overview

### What is unlost?

unlost is a developer tool that solves the "where did we land on that decision?" problem. When working with AI coding assistants (like OpenCode, Cursor, or direct API usage), conversations happen, decisions get made, and then... they're lost in chat history.

unlost intercepts these LLM API conversations via a transparent HTTP reverse proxy, extracts the high-signal content (intent, decisions, rationale, next steps), and stores them in a local vector database for later retrieval.

### Key Principles

1. **Local-first**: All data stays on your machine. Full transcripts are never stored—only structured summaries.
2. **Privacy-preserving**: Only capsule fields and minimal metadata are persisted.
3. **Low-friction**: Works as a transparent proxy; minimal setup required.
4. **Workspace-aware**: Each repository/project gets isolated storage.

### Use Cases

- Recovering context after time away from a codebase
- Tracking decisions across long-running refactors
- Multi-agent or multi-person work where decisions drift
- Keeping a lightweight trail of why a PR ended up the way it did

---

## Core Concepts

### Capsules

The fundamental unit of stored knowledge is an **IntentCapsule**:

```rust
pub struct IntentCapsule {
    pub category: String,       // e.g., "refactor", "bugfix", "feature"
    pub intent: String,         // What was being attempted
    pub decision: String,       // What was decided
    pub rationale: String,      // Why this decision
    pub next_steps: Vec<String>, // Suggested follow-ups (max 3)
    pub symbols: Vec<String>,   // Referenced identifiers, paths, routes
}
```

Capsules are extracted from conversation slices using an LLM with structured output, then embedded locally and stored in LanceDB.

### Workspaces

A **workspace** represents a project/repository. Workspaces are identified by:
1. Git remote origin URL (normalized, hashed) - preferred
2. Project manifest name (package.json, Cargo.toml, etc.) - fallback
3. Directory name - last resort

This ensures the same workspace ID across machines for the same repository.

### Exchanges

An **exchange** is a single request-response pair through the proxy. Multiple exchanges are chunked together into conversation slices before capsule extraction.

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           unlost serve                                   │
│                    (Multiplexed HTTP Reverse Proxy)                      │
│                                                                          │
│   Coding Agent                                                           │
│   (opencode,       ──────►  /w/<workspace_id>/<provider>/...            │
│    cursor, etc.)                                                         │
│                       ┌──────────────────────────────────────────────┐   │
│                       │           Provider Routing                    │   │
│                       │  ┌──────┐  ┌──────────┐  ┌──────────┐       │   │
│                       │  │OpenAI│  │Anthropic │  │ OpenCode │       │   │
│                       │  └──┬───┘  └────┬─────┘  └────┬─────┘       │   │
│                       └─────┼───────────┼─────────────┼──────────────┘   │
└─────────────────────────────┼───────────┼─────────────┼──────────────────┘
                              │           │             │
                              ▼           ▼             ▼
                     api.openai.com  api.anthropic.com  opencode.ai
                              │           │             │
                              └───────────┼─────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────┐
                    │               Analysis Worker                        │
                    │  • Extract user/assistant text from request/response │
                    │  • Parse SSE streams for streaming responses         │
                    │  • Detect commit/PR milestone mentions               │
                    │  • Build exchange_text: "User:\n...\nAssistant:\n...│
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │             WorkspaceChunker                         │
                    │  • Buffer turns per workspace                        │
                    │  • Flush on: idle timeout, size, turn count,         │
                    │    milestone detection                               │
                    │  • Generate FlushJob with conversation slice         │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │              Flush Worker                            │
                    │  • Classify user/assistant emotion (local ONNX)     │
                    │  • LLM extraction → IntentCapsule                   │
                    │  • Embed capsule text (local fastembed)             │
                    │  • Write to JSONL backup                            │
                    │  • Insert into LanceDB with vector                  │
                    └──────────────────────┬──────────────────────────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            ▼                              ▼                              ▼
     ┌──────────────┐           ┌──────────────────┐           ┌──────────────┐
     │   LanceDB    │           │  capsules.jsonl  │           │ query/recall │
     │   (vectors)  │           │    (backup)      │           │  (retrieval) │
     └──────────────┘           └──────────────────┘           └──────────────┘
```

### Component Interaction

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HTTP Proxy    │────▶│ Analysis Worker │────▶│  Flush Worker   │
│  (http_proxy)   │     │   (recording)   │     │   (recording)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       ├──▶ LLM Client (llm.rs)
        │                       │                       ├──▶ Embedder (embed.rs)
        │                       │                       ├──▶ Emotion Model (emotion.rs)
        │                       │                       └──▶ Storage (storage.rs)
        │                       │
        │                       └──▶ Net Utils (net.rs)
        │                            • SSE parsing
        │                            • Body extraction
        │                            • Header sanitization
        │
        └──▶ Upstream Providers (HTTPS)
```

---

## Data Flow

### Recording Flow (serve/record commands)

```
1. HTTP Request arrives at proxy
   │
   ├─► Parse multiplexed URI: /w/<workspace_id>/<provider>/<path>
   ├─► Read and buffer request body (max 2MB)
   ├─► Sanitize headers (remove hop-by-hop, set Host)
   ├─► Forward to upstream provider (HTTPS)
   │
2. Response streaming begins
   │
   ├─► Send AnalysisMsg::ExchangeStart { meta, request_body }
   ├─► Stream response chunks to client AND to analysis channel
   ├─► Send AnalysisMsg::ResponseEnd when complete
   │
3. Analysis Worker processes exchange
   │
   ├─► Extract user text from request body (OpenAI/Anthropic format)
   ├─► If SSE: parse stream, extract assistant text deltas
   ├─► If JSON: parse response, extract assistant content
   ├─► Build exchange_text: "User:\n{user}\n\nAssistant:\n{assistant}"
   ├─► Detect commit/PR mentions (milestone)
   ├─► Create ChunkInput and send to WorkspaceChunker
   │
4. WorkspaceChunker buffers and flushes
   │
   ├─► Buffer turns per workspace
   ├─► Flush triggers:
   │   • IDLE_FLUSH_AFTER: 2 seconds of inactivity
   │   • MAX_TOTAL_CHARS: 16KB accumulated
   │   • MAX_TURNS: 8 turns
   │   • Milestone: commit/PR mention detected
   ├─► Build FlushJob with conversation slice
   │
5. Flush Worker processes job
   │
   ├─► Extract user/assistant text for emotion classification
   ├─► Run emotion model (spawn_blocking, ONNX)
   ├─► Call LLM for capsule extraction (structured output)
   ├─► Append to JSONL backup file
   ├─► Generate embedding for capsule (local fastembed)
   └─► Insert into LanceDB with all metadata
```

### Query Flow

```
1. User runs: unlost query "why did we rename X?"
   │
2. Load embedding model
   │
3. Embed query text (local)
   │
4. Search LanceDB with nearest-neighbor query
   ├─► Optional: filter by symbol
   ├─► Return top K matches with distance scores
   │
5. If LLM enabled (default):
   │
   ├─► Build context from matches
   ├─► Send to LLM with query narrative prompt
   └─► Format and display narrative with ANSI colors
   │
6. If --no-llm or --facts:
   │
   └─► Display raw capsule hits
```

### Recall Flow

```
1. User runs: unlost recall [scope]
   │
2. Load embedding model
   │
3. If scope provided:
   │   Embed scope text and search for related capsules
   │ Else:
   │   Scan recent capsules (by timestamp)
   │
4. Build recall context with capsules + emotion metadata
   │
5. Send to LLM with recall narrative prompt
   │
6. Format and display "story so far" narrative
```

---

## Module Reference

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **main** | `src/main.rs` | Entry point, CLI dispatch, runtime setup |
| **cli** | `src/cli.rs` | Command-line argument definitions (clap) |
| **http_proxy** | `src/http_proxy.rs` | TCP listener, request forwarding, response streaming |
| **recording** | `src/recording.rs` | Conversation chunking, analysis worker, flush processing |
| **analysis** | `src/analysis.rs` | Message types for async communication between workers |
| **storage** | `src/storage.rs` | LanceDB schema, insert, query, and scan operations |
| **embed** | `src/embed.rs` | Local embedding model (fastembed) loading and inference |
| **emotion** | `src/emotion.rs` | ONNX emotion classifier, go_emotions mapping |
| **llm** | `src/llm.rs` | Multi-provider LLM client abstraction |
| **narrative** | `src/narrative.rs` | LLM prompt templates for query/recall narratives |
| **net** | `src/net.rs` | HTTP utilities, SSE parsing, body extraction |
| **workspace** | `src/workspace.rs` | Workspace ID computation, path management |
| **config** | `src/config.rs` | Configuration data structures |
| **types** | `src/types.rs` | Core data types (IntentCapsule, CapsuleHit, etc.) |
| **util** | `src/util.rs` | String utilities, SQL escaping |
| **constants** | `src/constants.rs` | Default embedding model constants |

### Command Modules

| Module | File | Purpose |
|--------|------|---------|
| **serve** | `src/commands/serve.rs` | Multi-workspace proxy server |
| **record** | `src/commands/record.rs` | Single-workspace proxy mode |
| **query** | `src/commands/query.rs` | Semantic search over capsules |
| **recall** | `src/commands/recall.rs` | Proactive "story so far" narrative |
| **init** | `src/commands/init.rs` | Seed database from code graph |
| **inspect** | `src/commands/inspect.rs` | View raw stored capsules |
| **model** | `src/commands/model.rs` | Download embedding models |
| **config** | `src/commands/config.rs` | Configure LLM/agent settings |
| **clear** | `src/commands/clear.rs` | Delete workspace data |

---

## Data Models

### IntentCapsule

The primary knowledge unit extracted from conversations:

```rust
pub struct IntentCapsule {
    pub category: String,       // Classification: "refactor", "bugfix", "feature", etc.
    pub intent: String,         // What was the user trying to accomplish
    pub decision: String,       // What was actually decided/done
    pub rationale: String,      // Why this approach was chosen
    pub next_steps: Vec<String>, // Follow-up actions (max 3)
    pub symbols: Vec<String>,   // Code identifiers, file paths, routes mentioned
}
```

### ResponseMeta

Metadata about the HTTP exchange:

```rust
pub struct ResponseMeta {
    pub source: String,        // "record" | "init"
    pub upstream_host: String, // e.g., "api.openai.com"
    pub request_path: String,  // e.g., "/v1/chat/completions"
    pub http_status: u16,      // HTTP response status code
}
```

### CapsuleHit

Full result from a database query:

```rust
pub struct CapsuleHit {
    pub id: String,                              // UUID
    pub ts_ms: i64,                              // Unix timestamp (milliseconds)
    pub conn_id: i64,                            // Connection ID
    pub exchange_seq: i64,                       // Exchange sequence within connection
    pub distance: f32,                           // Vector similarity distance
    pub user_emotion: Option<EmotionMeta>,       // User's emotional state
    pub assistant_emotion: Option<EmotionMeta>,  // Assistant's emotional state
    pub capsule: IntentCapsule,                  // The actual capsule
    pub meta: ResponseMeta,                      // Request metadata
}
```

### EmotionMeta

Emotion classification result:

```rust
pub struct EmotionMeta {
    pub label: String,     // "joy" | "neutral" | "confused" | "frustration" | "anger" | "sad"
    pub valence: f32,      // -1..1 (negative to positive)
    pub intensity: f32,    // 0..1 (calm to intense)
    pub confidence: f32,   // 0..1 (model confidence)
}
```

### AnalysisMsg

Messages passed through the analysis channel:

```rust
pub enum AnalysisMsg {
    ExchangeStart {
        meta: AnalysisMeta,
        request_body: Bytes,
    },
    ResponseChunk(Bytes),
    ResponseEnd,
}

pub struct AnalysisMeta {
    pub workspace_id: String,
    pub upstream_host: String,
    pub request_path: String,
    pub http_status: u16,
    pub content_type: Option<String>,
}
```

### FlushJob

Job sent to the flush worker:

```rust
pub struct FlushJob {
    pub workspace_id: String,
    pub conn_id: u64,
    pub exchange_seq: u64,
    pub ts_ms: i64,
    pub meta: ResponseMeta,
    pub input: String,  // Formatted conversation slice
}
```

---

## Storage Layer

### LanceDB Schema

The `capsules_v2` table schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Utf8 | UUID primary key |
| `ts_ms` | Int64 | Unix timestamp in milliseconds |
| `source` | Utf8 | "record" or "init" |
| `upstream_host` | Utf8 | API provider hostname |
| `request_path` | Utf8 | API endpoint path |
| `http_status` | Int32 | HTTP response status |
| `conn_id` | Int64 | Connection identifier |
| `exchange_seq` | Int64 | Exchange sequence number |
| `user_emotion` | Utf8 (nullable) | User emotion label |
| `user_emotion_conf` | Float32 (nullable) | User emotion confidence |
| `user_valence` | Float32 (nullable) | User emotion valence |
| `user_intensity` | Float32 (nullable) | User emotion intensity |
| `assistant_emotion` | Utf8 (nullable) | Assistant emotion label |
| `assistant_emotion_conf` | Float32 (nullable) | Assistant emotion confidence |
| `assistant_valence` | Float32 (nullable) | Assistant emotion valence |
| `assistant_intensity` | Float32 (nullable) | Assistant emotion intensity |
| `category` | Utf8 | Capsule category |
| `intent` | Utf8 | Capsule intent |
| `decision` | Utf8 | Capsule decision |
| `rationale` | Utf8 | Capsule rationale |
| `next_steps` | List<Utf8> | Suggested next steps |
| `symbols` | List<Utf8> | Referenced symbols |
| `embedding` | FixedSizeList<Float32, 384> | Vector embedding |

### Indexes

- **Vector index** on `embedding` column (Auto/IVF_PQ)
- **LabelList index** on `symbols` column for symbol filtering

### Storage Locations

```
~/.local/share/unlost/
├── workspaces/
│   └── wks_<hash>/
│       ├── lancedb/           # LanceDB directory
│       │   └── capsules_v2/   # Vector table
│       └── capsules.jsonl     # JSONL backup
└── models/
    ├── fastembed/             # Embedding model cache
    │   └── BAAI_bge-small-en-v1.5/
    └── emotion/               # Emotion model cache
        └── SamLowe_roberta-base-go_emotions-onnx/

~/.config/unlost/
└── config.json                # Global configuration
```

### JSONL Backup Format

Each line is a JSON object:

```json
{
  "ts_ms": 1700000000000,
  "conn_id": 1,
  "exchange_seq": 1,
  "source": "record",
  "upstream_host": "api.openai.com",
  "request_path": "/v1/chat/completions",
  "http_status": 200,
  "capsule": {
    "category": "refactor",
    "intent": "...",
    "decision": "...",
    "rationale": "...",
    "next_steps": ["..."],
    "symbols": ["..."]
  }
}
```

---

## CLI Commands

### `unlost serve`

Multi-workspace proxy server with URL-based routing.

```bash
unlost serve --bind 127.0.0.1:3000
```

URL pattern: `http://127.0.0.1:3000/w/<workspace_id>/<provider>/<path>`

Providers: `openai`, `anthropic`, `opencode`

### `unlost record`

Single-workspace proxy mode (legacy).

```bash
unlost record --bind 3000 --upstream-host api.openai.com
```

### `unlost query`

Semantic search over stored capsules.

```bash
unlost query "why did we rename the function?"
unlost query --symbol scan_capsules "how is this used?"
unlost query --no-llm "authentication flow"
unlost query --facts "routing logic"
```

Options:
- `--limit N`: Max results (default: 5)
- `--symbol SYM`: Filter by symbol
- `--no-llm`: Skip narrative, show raw matches
- `--facts`: Show raw matches after narrative
- `--llm-model MODEL`: Override LLM model

### `unlost recall`

Generate a "story so far" narrative.

```bash
unlost recall
unlost recall src/http_proxy.rs
unlost recall WorkspaceChunker
```

Options:
- `--limit N`: Max capsules to use (default: 24)
- `--llm-model MODEL`: Override LLM model

### `unlost inspect`

View raw stored capsules.

```bash
unlost inspect --limit 10
unlost inspect --filter "category = 'refactor'"
```

### `unlost init`

Seed the database from codebase analysis.

```bash
unlost init --path .
unlost init --git-history --git-commits 100
```

Options:
- `--max-capsules N`: Max capsules to create
- `--no-llm`: Skip LLM summarization
- `--git-history`: Include git commit history
- `--git-commits N`: Number of commits to analyze

### `unlost config`

Manage configuration.

```bash
# Configure LLM provider
unlost config llm openai --api-key $OPENAI_API_KEY --model gpt-4o-mini
unlost config llm anthropic --api-key $ANTHROPIC_API_KEY
unlost config llm ollama --model llama3.2:3b
unlost config llm show
unlost config llm remove

# Configure agent
unlost configure agent opencode --path . --server http://127.0.0.1:3000
```

### `unlost model`

Manage local models.

```bash
unlost model download
unlost model download --force
```

### `unlost clear`

Delete workspace data.

```bash
unlost clear --path .
unlost clear --yes  # Skip confirmation
```

---

## Configuration

### Global Config File

Location: `~/.config/unlost/config.json`

```json
{
  "version": 1,
  "path_index": {
    "/path/to/repo": "wks_abc123def456"
  },
  "workspaces": {
    "wks_abc123def456": {
      "id": "wks_abc123def456",
      "root": "/path/to/repo",
      "source": "git",
      "db_dir": "/home/user/.local/share/unlost/workspaces/wks_abc123def456/lancedb",
      "capsules_jsonl": "/home/user/.local/share/unlost/workspaces/wks_abc123def456/capsules.jsonl",
      "created_ts_ms": 1700000000000,
      "updated_ts_ms": 1700000000000
    }
  },
  "llm": {
    "provider": "anthropic",
    "api_key": "sk-ant-...",
    "model": "claude-3-5-sonnet-20241022"
  }
}
```

### LLM Configuration Options

```json
// OpenAI
{
  "provider": "openai",
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1",  // optional
  "model": "gpt-4o-mini"
}

// Anthropic
{
  "provider": "anthropic",
  "api_key": "sk-ant-...",
  "base_url": "https://api.anthropic.com",  // optional
  "model": "claude-3-5-sonnet-20241022"
}

// Ollama (local)
{
  "provider": "ollama",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "llama3.2:3b"
}

// Custom (OpenAI-compatible)
{
  "provider": "custom",
  "base_url": "https://my-endpoint/v1",
  "api_key": "optional-key",
  "model": "custom-model"
}
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `UNLOST_EMBED_CACHE_DIR` | Override embedding model cache location |
| `UNLOST_EMOTION_CACHE_DIR` | Override emotion model cache location |
| `OPENAI_API_KEY` | OpenAI API key (fallback if not configured) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `RUST_LOG` | Logging filter (e.g., `unlost=debug`) |
| `NO_COLOR` | Disable ANSI colors |

---

## ML/AI Components

### Local Embedding (fastembed)

- **Model**: BAAI/bge-small-en-v1.5
- **Dimension**: 384
- **Library**: fastembed 5.8
- **Purpose**: Generate semantic embeddings for capsule content and queries

The embedding is generated from:
```
intent: {capsule.intent}
decision: {capsule.decision}
rationale: {capsule.rationale}
```

### Local Emotion Classification (ONNX)

- **Model**: SamLowe/roberta-base-go_emotions-onnx
- **Labels**: 28 go_emotions labels mapped to 6 buckets
- **Library**: ort 2.0 (ONNX Runtime) + tokenizers

Label mapping:
- **joy**: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
- **anger**: anger
- **frustration**: annoyance, fear, nervousness, disgust
- **sad**: disappointment, remorse, sadness, grief
- **confused**: confusion, curiosity, realization, surprise, embarrassment
- **neutral**: neutral, unknown

### Remote LLM (Capsule Extraction)

Used for:
1. **Capsule extraction**: Converting conversation slices to structured IntentCapsule
2. **Query narratives**: Generating conversational answers from search results
3. **Recall narratives**: Generating "story so far" summaries

Supported providers:
- OpenAI (gpt-4o-mini default)
- Anthropic (claude-3-5-sonnet-20241022)
- Ollama (local, OpenAI-compatible)
- Custom OpenAI-compatible endpoints

---

## Technology Stack

### Language & Runtime

- **Rust 2024 Edition** - Systems programming language
- **Tokio 1.49** - Async runtime with full features

### Networking

| Library | Version | Purpose |
|---------|---------|---------|
| hyper | 1.8 | HTTP client/server |
| hyper-rustls | 0.27 | TLS with native certs |
| hyper-util | 0.1 | Legacy client support |
| reqwest | 0.12 | High-level HTTP client |
| rustls | 0.23 | TLS (ring backend) |

### Storage & Data

| Library | Version | Purpose |
|---------|---------|---------|
| lancedb | 0.23 | Vector database |
| arrow-array/schema | 56.2 | Apache Arrow data |
| serde/serde_json | 1.0 | Serialization |
| uuid | 1.18 | Unique IDs |

### ML/AI

| Library | Version | Purpose |
|---------|---------|---------|
| fastembed | 5.8 | Local embeddings |
| ort | 2.0.0-rc.11 | ONNX Runtime |
| tokenizers | 0.20 | HuggingFace tokenizers |
| rig-core | 0.29 | LLM client abstraction |
| schemars | 1.2 | JSON Schema for structured extraction |

### CLI & Async

| Library | Version | Purpose |
|---------|---------|---------|
| clap | 4.5 | CLI argument parsing |
| kanal | 0.1 | MPMC channels |
| indicatif | 0.17 | Progress spinners |
| tracing | 0.1 | Structured logging |

### Code Analysis

| Library | Version | Purpose |
|---------|---------|---------|
| unfault-core | 0.1.7 | Code graph analysis |
| petgraph | 0.8 | Graph data structures |
| ignore | 0.4 | Gitignore-aware file walking |
| regex | 1.11 | Regular expressions |

### Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| anyhow | 1.0 | Error handling |
| sha2/hex | 0.10/0.4 | Hashing for workspace IDs |
| bytes | 1.10 | Binary data handling |
| flate2 | 1.1 | Gzip compression |

---

## Design Decisions

### Why No Full Transcript Storage?

1. **Privacy**: Conversations may contain sensitive information
2. **Signal vs Noise**: Most transcript content is low-value filler
3. **Storage**: Embeddings + metadata are much more compact
4. **Retrieval**: Structured fields enable precise filtering

### Why Local Embeddings?

1. **Privacy**: Text never leaves your machine
2. **Speed**: No network latency for embedding
3. **Cost**: No API charges for embedding operations
4. **Reliability**: Works offline

### Why LanceDB?

1. **Local-first**: Embedded database, no server needed
2. **Vector-native**: Built for similarity search
3. **Arrow-based**: Efficient columnar storage
4. **SQL-like filters**: DataFusion for filtering

### Why Chunking Strategy?

The chunker balances:
- **Latency**: Short idle timeout (2s) for responsive extraction
- **Context**: Multiple turns per chunk for coherent capsules
- **Cost**: Bounded chunk size reduces LLM token usage
- **Signals**: Milestone detection (commits/PRs) triggers immediate flush

---

## Future Considerations

### Potential Enhancements

1. **Multi-model support**: Different embedding models per workspace
2. **Incremental sync**: Sync capsules across machines via git
3. **IDE integration**: VS Code extension for inline recall
4. **Team features**: Shared capsule repositories
5. **Compliance mode**: Optional full transcript retention

### Known Limitations

1. **Single-threaded flush**: One LLM extraction at a time per workspace
2. **No streaming capsules**: Must wait for full response before extraction
3. **English-only emotion**: go_emotions model trained on English
4. **Fixed embedding dimension**: 384-dim embeddings only

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT - See [LICENSE](LICENSE) for details.
