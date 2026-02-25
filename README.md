<h2 align="center">
  <br>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png">
    <img alt="Unfault" src="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png" >
  </picture>
</h2>

<h4 align="center">Ownership isn't authorship anymore</h4>

---

It used to be. You wrote the code, so you understood it. The tradeoffs,
the dead ends, the reason the timeout is 30 seconds. That understanding
*was* ownership.

Agents have changed the writing part. They haven't changed the rest.
You're still the one accountable when something breaks. Still the one
who has to explain it, extend it, or defend it. Ownership without
authorship only works if the context stays close to you. Right now it
lives in the chat, and the chat doesn't last.

Unlost keeps it close.

---

## When it matters

**A colleague asks why it's built this way.**
You know it works. You're less sure you can explain it. The reasoning
was in the chat. `unlost trace` or `unlost brief` give you the answer,
not from memory, from the record.

**Six months later, someone needs to change it.**
Maybe it's you. The agent doesn't remember. The chat is gone. The
commit message says "feat: add retry logic." `unlost trace` gives you
the decision chain. `unlost challenge` tells you whether the original
call still holds before you undo it.

**Production is down.**
You're reading code under pressure that you didn't write. You don't
know if the retry logic was intentional or a guess. `unlost trace`
reconstructs the chain. `unlost brief` tells you what bites.

**The PR is the handoff.**
To your team, and to your future self. The diff shows what changed.
It doesn't show what was tried and rejected, what constraint you were
navigating, what's still open. `unlost pr-comment` posts that note
from session memory, not the diff. Ownership transfers with the code.

---

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

---

## Commands

### Understanding what was built

```bash
# Staff engineer's debrief on any file or module
unlost brief
unlost brief src/governor.rs

# What happened recently in this file?
unlost recall src/http_proxy.rs

# Reconstruct the decision chain that led to the current state
unlost trace src/governor.rs
unlost trace "why is the connection timeout 30 seconds?"
```

- **`unlost brief`**: Scans all recorded memory and git commits. Scores by importance, not recency. Answers: *what is this, what are the non-obvious choices, where do I start.*
- **`unlost trace`**: Builds a chronological causal chain seeded by semantic similarity. Answers: *why is the code the way it is*, not just what happened recently.
- **`unlost recall`**: Narrates the recent story for a file or concept. Useful for catching up after time away.

### Before you commit to a direction

```bash
# Argue with a past decision before you reverse it
unlost challenge "was using fastembed the right call?"

# Think through options using what this repo already knows
unlost explore "should we keep lancedb or move to sqlite+fts?"
```

- **`unlost challenge`**: Surfaces recorded rationale, failure modes, and alternatives. Gives you a verdict grounded in history before you make a call.
- **`unlost explore`**: Forward-looking planning. Labels what comes from memory `[memory]` vs. external knowledge `[outside]` so you know what you're actually standing on.

### Handing it off

```bash
# Post a PR comment from session history: intent, tradeoffs, risks
unlost pr-comment 42
unlost pr-comment https://github.com/owner/repo/pull/42
```

- **`unlost pr-comment`**: Posts a "staff engineer" style note on the PR. Not a diff summary. A note from someone who was in the room: what changed functionally, what we were navigating, what's left open, what to re-read in three months.

The `/unlost-walkthrough` agent skill does the same thing interactively: step through what changed, in order, with reasons, so you can review with confidence rather than just approve.

---

## How it works

1. **Capture:** After each agent exchange → extracts a structured capsule (intent, decision, rationale, symbols).
2. **Store:** Capsules stay local, embedded with fastembed, indexed in LanceDB. Nothing leaves your machine.
3. **Guide:** Before each prompt → checks for friction (loops, drift, misalignment). Injects a correction if something is off.
4. **Recall:** Capsules are queryable anytime, by file, symbol, question, or concept.

---

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

---

## Dev

```bash
cargo test
cargo build
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Docs

- `agents/README.md` - Agent integrations
