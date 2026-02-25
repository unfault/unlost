<h2 align="center">
  <br>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png">
    <img alt="Unfault" src="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png" >
  </picture>
</h2>

<h4 align="center">Unlost - Agent orientation that prevents the babysitting tax</h4>

---

## Mission

Keep your agents oriented.

You know the drill:

- You ask an agent to rename a function. It renames it, then renames it again. Then again. You finally intervene: "stop renaming, just add a wrapper."
- An agent spent forty minutes "implementing" a feature. Turns out it only modified the README.
- You come back after the weekend. The agent made decisions you don't understand. Nobody remembers why.
- An agent says it finished. It didn't. It created a file that was never committed.

That's the babysitting tax. It adds up fast.

Unlost intercepts before you pay. It detects these failure modes and guides agents back on track:

| Failure Mode | What It Looks Like | What Unlost Does |
|--------------|-------------------|------------------|
| **Drift** | Agent thinks the system works one way. The code says otherwise. | Surfaces the contradiction before the agent compounds the error. |
| **Rediscovery** | You explain the same thing you explained last week. | Reminds the agent of what was already decided. |
| **Decision Conflict** | Agent starts implementing something that contradicts a project decision. | Flags the conflict and reminds the agent of the constraint. |
| **Retry Spiral** | Agent tries the same failed approach. Again. And again. | Catches the loop before another hour burns. |
| **False Progress** | Agent claims done. Verification would fail. | Detects the claim and flags it for review. |
| **Unbounded Horizon** | Agent wanders into unrelated side-quests. | Nudges back toward the original goal. |

## Install

```bash
curl -fsSL https://unlost.unfault.dev/install.sh | bash
```

Or download the binary manually from [releases](https://github.com/unfault/unlost/releases).

## Quick Start

### 1. Hook into your agent

**Claude Code** (Global, zero per-repo config):
```bash
unlost config agent claude --global
```

**OpenCode** (Global):
```bash
unlost config agent opencode --global
```
Or per-project: `unlost config agent opencode --path .`

### 2. (Optional) Configure extraction LLM
By default, unlost uses whatever LLM your agent is configured with. You can override this for better results (e.g., using a smaller/faster model for extraction):

```bash
# Use Claude
unlost config llm anthropic --model claude-3-5-sonnet-20241022

# Or OpenAI
unlost config llm openai --model gpt-4o-mini
```

## How it works

1. **Check:** Your agent is about to send a prompt → unlost checks for friction (drift, loops, etc.).
2. **Guide:** If something feels off → injects guidance before the agent goes off-track.
3. **Capture:** After each exchange → extracts a structured capsule (intent, decision, rationale).
4. **Recall:** Capsules stay local → query anytime: "why did we do X?"

## Features

### Memory Commands

Your agents build a memory trail. Here's how to use it:

```bash
# Get a staff engineer's debrief — what matters, what bites, where to start
unlost brief

# Drill into a specific area
unlost brief src/governor.rs

# What happened recently in this file?
unlost recall src/http_proxy.rs

# Reconstruct the causal chain that led to the current state
unlost trace src/governor.rs
unlost trace "why is the connection timeout 30 seconds?"

# Pressure-test a past technology choice
unlost challenge "was using fastembed the right call?"

# Explore future paths grounded in workspace memory
unlost explore "should we keep lancedb or move to sqlite+fts?"
```

#### Key Commands Explained:
- **`unlost brief`**: A one-command orientation for any codebase. Answers "Here's what this system is, the non-obvious choices, and where to start reading." Scans all recorded memory and git commits.
- **`unlost trace`**: Reconstructs the **causal chain** of decisions. Unlike `recall` (which narrates recent history), `trace` asks: *why is the code the way it is?* It builds a chronological chain seeded by semantic similarity.

### The Cognitive Mirror (Metrics)

Unlost tracks the **trajectory** of your collaboration. Use `metrics` to see friction points:

```bash
unlost metrics
```

This reveals:
- **Friction vs Context Size**: When does the agent get lost?
- **Average Verbosity**: A leading indicator for over-trust.
- **Top Friction Files**: Codebase "hotspots" causing stalls.

### Replay & Git History

Seed your memory from past sessions or git history:

```bash
# Replay OpenCode sessions (ingests git history automatically)
unlost replay opencode

# Replay a Claude transcript
unlost replay claude --transcript-path history.json

# Ingest only git history
unlost replay git --max-commits 200
```

## Privacy First

Everything unlost stores stays on your machine:

- **Capsules** — Stored locally in `~/.local/share/unlost/workspaces/`
- **Embeddings** — Generated locally with fastembed
- **Query history** — Never leaves your disk

The only network call unlost makes is to the LLM provider you configure for extraction. That LLM sees only the exchange text (no tool outputs), and it produces a capsule that never goes back upstream.

<details>
<summary><h2>Under the Hood (Technical Details)</h2></summary>

### Trajectory Sensing

- **Three-state FSM** — Stable → Watch → Intervene controller with hysteresis, per-basin cooldowns, and a one-shot rule preventing repeat intervention types
- **Weighted multi-channel basin scoring** — Loop, Spec, and Drift intensities computed as calibrated weighted sums of 10 independent symptom channels
- **EMA smoothing** — All 10 channels smoothed with exponential moving average (α=0.3) to suppress single-turn noise spikes
- **Sliding window persistence** — State only escalates after 3 consecutive turns above the 0.75 intensity threshold
- **Coffee Pause soft decay** — >30-minute gaps decay intensity to 0.3× and reset state; injects a resumption brief on return
- **Grounding stall detection** — User-mentioned file paths tracked with exponential time decay; stall streak increments when the agent ignores them
- **Jaccard-like logic churn** — Word-set distance between consecutive agent decisions; detects rapid plan changes without progress
- **Symbol repetition / novelty collapse** — Fraction of current capsule symbols seen in the last 8 capsules; complement is novelty score
- **Stubbornness boost** — Extra intensity when alignment debt is high but decision churn is low (agent acknowledges errors but keeps the same plan)
- **Blind Acceptance risk** — Detects fluent long responses followed by passive short user replies; flags over-trust risk
- **Summary intent damping** — Multiplies intensity by 0.6 on turns the agent is legitimately consolidating, preventing false positives
- **Stratified intervention policy** — Ambient hint / Structural note / Emergency hard-stop tiered by intensity level
- **Hydration packet** — For Loop interventions, injects the 3 most relevant recent capsules scored by recency, symbol overlap, emotion, effort, and failure mode

### Emotion & NLP

- **Multi-label emotion classification** — RoBERTa-base fine-tuned on GoEmotions (28 labels → 8 buckets), running locally via ONNX Runtime
- **Heuristic emotion override** — Pattern-based frustration and doubt detection that corrects misclassifications from the neural model
- **Affective modulation** — Joy halves trajectory intensity; persistent anger triggers a de-escalation override regardless of basin state

### Retrieval & Memory

- **HyPE** — At indexing time, the LLM generates 2–3 questions each capsule answers; at retrieval time, each command frames your query to match those questions — question-to-question match, not keyword-to-document. ([Ma et al., 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335))
- **Trajectory-encoded embeddings** — Each capsule is embedded with its category, failure mode, symbols, and the prior decision from the same work thread; causally related capsules cluster together across sessions
- **BGE-small-en-v1.5 dense embeddings** — 384-dimensional vectors, generated fully locally via fastembed + ONNX Runtime
- **ANN vector search** — LanceDB `nearest_to` with an auto-tuned approximate nearest-neighbour index on the embedding column
- **LabelList index** — Scalar index on the symbols array column enabling fast `array_contains` fan-out queries
- **Causal chain algorithm** — ANN seed → symbol fan-out via LabelList index → similarity threshold pruning → chronological sort; powers `trace`
- **Cross-session recurrence scoring** — Capsules scored for `brief` by failure mode, explicit rationale/decision, and symbols recurring across multiple sessions (no recency bias)
- **Recency-weighted fingerprint deduplication** — `recall` collapses near-duplicates by content fingerprint and caps older sessions at 3 results, with a 30-minute recency bypass
- **Checkpoint summarization** — Background process compresses windows of capsules into narrative checkpoints; `recall` and `brief` use a fast path when the delta since last checkpoint is small

### Storage & Infrastructure

- **Apache Arrow / LanceDB columnar store** — Capsules stored as Arrow RecordBatches with three indexes (ANN, LabelList, scalar timestamp); append-only with schema evolution
- **Code graph analysis** — `unfault-core` + petgraph builds a live static graph for centrality scoring, dependency/impact traversal, and symbol validation backing Drift detection
- **LLM structured extraction** — JSON Schema extraction via rig-core + schemars; produces typed `IntentCapsule` structs from raw agent exchanges
- **Hybrid extraction mode** — Heuristics identify "pivotal" turns before invoking the LLM, reducing extraction cost by skipping routine turns
- **SHA-256 job deduplication** — Flush jobs hashed by content; identical jobs within a 45-second window are suppressed
- **Git grounding & SHA provenance** — Git HEAD and commit SHAs stored on every capsule; git commits ingested as first-class capsules, deduplicated by hash
- **Changelog ingestion** — CHANGELOG.md versions parsed and stored as versioned capsules, surfaced with `ref=version:vX.Y.Z` citations in LLM prompts

</details>

## Dev

```bash
cargo test
cargo build
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Docs

- `agents/README.md` - Agent integrations
