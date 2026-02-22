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

## Use unlost with your favourite agent

### Claude

All your Claude Code projects, forever, with zero per-repo config:

```bash
unlost config agent claude --global
```

This installs the Unlost agent skill and hooks unlost into every Claude Code session. Unlost checks for friction before each prompt, injects guidance when something feels off, and quietly records what actually happened (intent, decision, rationale, next steps) into local capsules you can query anytime.

### OpenCode

All your OpenCode projects (global config):

```bash
unlost config agent opencode --global
```

Or one project at a time:

```bash
unlost config agent opencode --path .
```

This installs the Unlost agent skill and hooks unlost into your OpenCode sessions. The skill teaches OpenCode:

- **Friction detection** runs automatically — unlost checks for drift, retry spirals, and false progress before each prompt
- **Two tiers of memory commands**:
  - Fast path (no LLM): `unlost query --no-llm`, `unlost metrics` — safe to run proactively
  - LLM path (on demand): `unlost query`, `unlost recall`, `unlost brief` — only when user explicitly asks

Unlost spots drift, catches false progress, and builds a local trail of decisions — without storing full transcripts.

## How it works

```
1. Your agent is about to send a prompt → unlost checks for friction
2. If something feels off → injects guidance before the agent goes off-track
3. After each exchange → extract a capsule (what we tried, why, next steps)
4. Capsules stay local → query anytime: "why did we do X?"
```

No transcripts. No external storage. Just small, queryable capsules that remember what your agents decided and why.

## Configure your LLM (optional, for better capsules)

Unlost uses a small LLM to extract structured capsules from agent exchanges. By default, it uses whatever LLM your agent is already configured with. You can override this for better results:

```bash
# Use Claude for extraction
unlost config llm anthropic --model claude-sonnet-4-5-20250929

# Or OpenAI
unlost config llm openai --model gpt-4o-mini
```

**What the LLM is fed:**
- The raw user → assistant exchange (just the text, not tool outputs)
- Recent capsule history from the workspace (to detect contradictions)

**What it produces:**
- A structured capsule with: category, intent, decision, rationale, next_steps, symbols, failure_mode

**Where it runs:**
- The LLM call goes through your configured provider (Anthropic, OpenAI, etc.)
- Capsule storage is entirely local — embeddings, the capsules themselves, and query history never leave your machine

## Recall, Brief, Query & Trace

Your agents built a memory trail. Here's how to use it:

```bash
# Get a staff engineer's debrief — what matters, what bites, where to start
unlost brief

# Drill into a specific area
unlost brief src/governor.rs

# What happened recently in this file?
unlost recall src/http_proxy.rs

# Why did we rename the capsules table?
unlost query "why did we rename the capsules table?"

# Find everything about the proxy routing
unlost query --symbol proxy_request

# Reconstruct the causal chain that led to the current state of a file
unlost trace src/governor.rs

# Ask a question about a decision's history
unlost trace "why is the connection timeout 30 seconds?"

# Investigate what happened in a specific time window
unlost trace "auth flow" --since 4M --until 3M
```

### `unlost brief`

`brief` is a one-command orientation for any codebase you didn't write (or haven't touched in a while). It answers the question a staff engineer would answer on day one: *"Here's what this system is, the non-obvious choices, the things that bite, and where to start reading."*

Unlike `recall` (which narrates recent history), `brief` scans all recorded memory — conversations and git commits — and scores each capsule by importance: failure modes that were recorded, decisions with rationale, knowledge that surfaced repeatedly across sessions. The output is structured into four sections:

```
MENTAL MODEL          — what this system is and its core invariant
KEY DESIGN DECISIONS  — non-obvious choices and why they were made
THINGS THAT BITE      — gotchas and hard-learned lessons
ENTRY POINTS          — where to start reading, with file:line references
GO DEEPER             — suggested unlost commands to drill down further
```

Git commit messages are first-class capsules in `brief`. A commit like `"fix: workspace ID must come from git remote, not path"` with an explanatory body will surface as a KEY DESIGN DECISION — even if it predates when you started using unlost.

### `unlost trace`

`trace` reconstructs the **causal chain** of decisions that explain the current state of a file, symbol, or concept. Where `recall` narrates what happened recently, `trace` asks: *why is the code the way it is?*

Pass a file path, a function name, or a natural-language question. Unlost builds a chronological chain — seeded by semantic similarity, expanded by symbol overlap — and narrates the path: the turning points, the recorded failures, the constraint that became an invariant.

```bash
# The path that led to this file's current state
unlost trace src/http_proxy.rs

# Why a specific decision was made
unlost trace "why did we switch to HTTP/2?"

# Incident investigation: what happened in a specific window?
unlost trace "connection timeout" --since 4M --until 3M
```

This works across session boundaries and across months of history. Engineers don't slice their work neatly per session — `trace` doesn't require them to.

## Long Memory & the Story Arc

Capsules are not just a log. They encode a *causal history* — the sequence of decisions, constraints, and failures that explain why the code looks the way it does today.

### How it works

**Richer embeddings.** Every capsule is embedded with its category, failure mode, top symbols, and the prior decision from the same work thread. This encodes trajectory into the vector — capsules from the same causal chain cluster together in embedding space, even across different sessions.

**HyPE questions.** When a capsule is extracted, the LLM also generates 2–3 questions the capsule answers — *"Why is the timeout 30 seconds?"*, *"How does the proxy route upstream requests?"*. At retrieval time, your query is matched against these pre-generated questions. Question-to-question matching is significantly more precise than query-to-capsule. Based on [Ma et al., "HyPE: Hypothetical Prompt Embeddings" (2024)](https://arxiv.org/abs/2412.12630).

**Causal chain (trace).** `unlost trace` seeds from a vector search, fans out to capsules sharing symbols, applies a similarity threshold, and sorts the survivors chronologically. The LLM narrates the causal path: turning points, recorded failures, the constraint that became an invariant.

```
2026-01-14  switched to HTTP/2 for upstream
2026-01-21  retry spiral on keepalive — increased timeout to 30s
2026-02-03  timeout 30s hardcoded in proxy_request
2026-02-18  ← you are here
```

### Already have capsules?

New capsules automatically get richer embeddings and HyPE questions. To apply these improvements to your existing history, run `unlost reindex`.

## The Cognitive Mirror (Metrics)

Unlost doesn't just store memory; it tracks the **trajectory** of your collaboration. As context grows, agents often begin to stall or drift. Unlost monitors these patterns across three "Basins of Friction":

| Basin | What it senses | Leading Indicator |
|-------|----------------|-------------------|
| **Loop** | Repetitive stalls | Symbol repetition, novelty collapse, logic churn. |
| **Spec** | Alignment debt | VERBATIM instruction repeats, corrective keywords. |
| **Drift** | Grounding failure | Hallucinated paths, ignoring user-mentioned files. |

Use the `metrics` command to see the "Cognitive Mirror" of your workspace:

```bash
unlost metrics
```

This reveals:
- **Friction vs Context Size**: See how the warning rate spikes as your input tokens grow (the "Lost in Context" threshold).
- **Average Verbosity**: Measures "Fluency" — how much the assistant is dominating the token share (a leading indicator for over-trust and blind acceptance).
- **Average Interval**: How many tokens of productive work you get between trajectory breakdowns.
- **Top Friction Files**: Codebase "hotspots" that are consistently causing the agent to stall or drift.

## Replay & Git History

Did an agent session happen while Unlost was off? Or do you want to seed a codebase with its git history?

```bash
# Replay OpenCode sessions — also ingests git history automatically
unlost replay opencode

# Replay a Claude transcript — also ingests git history automatically
unlost replay claude --transcript-path history.json

# Ingest only git history (no agent transcript needed)
unlost replay git
unlost replay git --max-commits 200
```

Git commit history is ingested automatically whenever you run `replay opencode` or `replay claude` — no extra step. Each commit becomes a capsule: subject as the decision, body as the rationale, touched files as symbols. Deduplication by hash means re-running replay never double-counts commits.

The `--git-grounding` flag (on `replay opencode` and `replay claude`) cross-references agent-touched files against actual commits in your history, marking capsules as "Verified via Git" when a match is found.

Git capsules are used by `brief` and `query` but deliberately excluded from `recall` (which stays focused on the conversational story) and from the trajectory controller (which operates on live turns only).

## Install

```bash
curl -fsSL https://unlost.unfault.dev/install.sh | bash
```

Or download the binary manually from [releases](https://github.com/unfault/unlost/releases).

## Dev

```bash
cargo test
cargo build
```

## Privacy First

Everything unlost stores stays on your machine:

- **Capsules** — Stored locally in `~/.local/share/unlost/workspaces/`
- **Embeddings** — Generated locally with fastembed
- **Query history** — Never leaves your disk

The only network call unlost makes is to the LLM provider you configure for extraction. That LLM sees only the exchange text (no tool outputs), and it produces a capsule that never goes back upstream.

## License

MIT. See [LICENSE](LICENSE) for details.

## Docs

- `agents/README.md` - Agent integrations
