# Research: Optimizing Historical Conversation Replay

## The Problem Space

### Context

AI coding agents (Claude Code, OpenCode, Cursor, etc.) generate conversation histories stored as JSONL files. These histories contain valuable signal:
- What the user intended
- What decisions were made and why
- What files/symbols were touched
- Where the agent went wrong (failure modes)

**Unlost** extracts this signal into structured "IntentCapsules" that enable semantic search and proactive friction detection. However, users often want to **replay existing history** from before they installed unlost, or from sessions recorded by other tools.

### The Core Tension

Extracting a high-quality IntentCapsule from a conversation turn requires understanding:
- Intent (what the user wanted)
- Decision (what was chosen)
- Rationale (why)
- Symbols (files, functions, endpoints touched)
- Failure mode (drift, retry spiral, false progress, etc.)

This is fundamentally a **comprehension task** - it requires "reading" the conversation and summarizing it. LLMs excel at this, but LLM calls are:
1. **Slow** (~500-2000ms per call)
2. **Expensive** (~$0.001-0.01 per turn depending on model)
3. **Rate-limited** (can't parallelize infinitely)

For a user with 1000 conversation turns, replaying with current approach takes **15-30 minutes** and costs **$1-10**.

### The Research Question

> **How can we extract structured knowledge from conversation histories with acceptable quality while minimizing latency and cost?**

Sub-questions:
1. What is the **minimum viable extraction** that still enables useful search and friction detection?
2. Where on the **Pareto frontier** (quality vs speed vs cost) do different approaches land?
3. Can we **decompose** the extraction task into parts with different cost/quality tradeoffs?
4. Is there an **information-theoretic lower bound** on what we can extract without comprehension?

---

## Hypothesis

### H1: The extraction task is decomposable

Not all capsule fields require the same level of comprehension:

| Field | Comprehension Required | Can Extract Locally? |
|-------|----------------------|---------------------|
| `symbols` | Low - pattern matching | Yes (regex for paths, identifiers) |
| `category` | Medium - keyword classification | Partially (heuristics for "fix", "add", "refactor") |
| `intent` | High - understanding user goal | No - needs LLM |
| `decision` | High - understanding what was chosen | No - needs LLM |
| `rationale` | High - understanding why | No - needs LLM |
| `failure_mode` | Very High - multi-turn reasoning | No - needs good LLM |

**Hypothesis:** We can extract `symbols` and approximate `category` locally, deferring `intent/decision/rationale/failure_mode` to LLM or skipping them entirely for "fast mode".

### H2: Most turns are low-signal

In a typical coding session:
- 10-20% of turns are "pivotal" (key decisions, direction changes, failures)
- 80-90% are incremental (small edits, clarifications, routine work)

**Hypothesis:** We can identify pivotal turns cheaply (via heuristics like emotion, symbol churn, length) and only run full LLM extraction on those. Incremental turns get lightweight extraction.

### H3: Batching provides superlinear gains

LLM calls have fixed overhead (network, prompt parsing). Batching multiple turns:
- Amortizes fixed costs
- May improve extraction quality (more context)
- But risks: longer prompts = more tokens = higher cost; may hit context limits

**Hypothesis:** There's an optimal batch size (likely 5-15 turns) that minimizes total cost while maintaining quality.

### H4: Local small models can approximate cloud quality

Models like Qwen2.5-1.5B, Phi-3-mini, Llama-3.2-1B run locally in ~100ms on CPU. They may:
- Handle simple extraction well
- Fail on nuanced failure mode detection
- Be "good enough" for search/recall use cases

**Hypothesis:** A local small model can achieve 70-80% of cloud model quality at 0 marginal cost and 10x speed.

### H5: Embedding-only mode enables deferred extraction

If we store raw turn text + embedding, we can:
- Enable semantic search immediately (no LLM needed)
- Defer capsule extraction to query time (pay only for what you use)
- Re-process later with better models

**Hypothesis:** For many users, "search works, details extracted on-demand" is acceptable.

---

## The Idea Space

### Approach 1: Tiered Extraction

```
Tier 0 (instant, local):
  - symbols: regex extraction
  - category: keyword heuristics
  - embedding: local fastembed
  → Enables: basic search, symbol tracking

Tier 1 (fast, local LLM):
  - intent: local small model (qwen2.5:1.5b)
  - decision: local small model
  → Enables: better search, basic recall

Tier 2 (quality, cloud LLM):
  - Full extraction including failure_mode
  - Run async/background or on-demand
  → Enables: friction detection, rich narratives
```

User chooses tier based on needs. Default to Tier 0 for replay, Tier 2 for live.

### Approach 2: Pivotal Turn Detection

```
For each turn:
  1. Compute "pivot score" locally:
     - User emotion (frustration, confusion) → +weight
     - Symbol churn (many new files) → +weight  
     - Turn length (very long) → +weight
     - Keywords ("decided", "instead", "actually") → +weight
  
  2. If pivot_score > threshold:
     → Full LLM extraction
     Else:
     → Lightweight local extraction only
```

Hypothesis: 20% of turns get full extraction, 80% get lightweight → 5x speedup.

### Approach 3: Batched Extraction with Clustering

```
1. Embed all turns locally (fast, parallel)
2. Cluster turns by semantic similarity
3. For each cluster:
   - Pick representative turn
   - Batch similar turns together
   - Single LLM call extracts capsules for batch
4. Distribute extracted info across cluster members
```

Benefit: Fewer LLM calls + potentially better extraction (more context).

### Approach 4: Progressive Enhancement

```
Pass 1 (instant): 
  - Local extraction only
  - Store raw text + embedding
  - User can search immediately

Pass 2 (background, async):
  - LLM extraction runs in background
  - Updates capsules as results arrive
  - User sees "enhancing..." indicator

Pass 3 (on-demand):
  - When user queries specific turn, extract if not done
  - Cache result
```

User gets instant value, quality improves over time.

### Approach 5: Compression via Summarization

Instead of extracting every turn, summarize sessions:

```
Session with 50 turns → LLM summarizes into 3-5 "mega-capsules"
  - "Implemented auth flow (turns 1-15)"
  - "Debugged token refresh bug (turns 16-30)"
  - "Added tests and documentation (turns 31-50)"
```

Trade granularity for massive speedup. Still enables search and recall.

---

## Experimental Design

### Independent Variables

1. **Extraction method**: baseline, parallel, batched, local-small, tiered, pivotal
2. **Model**: gpt-4o-mini, claude-3-5-haiku, qwen2.5:1.5b, phi3:mini
3. **Batch size** (for batching): 1, 5, 10, 20
4. **Pivot threshold** (for pivotal detection): 0.3, 0.5, 0.7

### Dependent Variables

1. **Latency**: wall-clock time, time per turn
2. **Cost**: API cost in dollars, tokens used
3. **Quality**: 
   - Symbol extraction accuracy (F1 vs ground truth)
   - Category accuracy
   - Semantic similarity of intent/decision to reference
   - Failure mode detection recall

### Dataset

Need a representative sample with:
- Variety of turn types (questions, debugging, refactoring, etc.)
- Known "pivotal" turns (manually labeled)
- Ground truth capsules (manually written for ~50 turns)

### Controls

- Same hardware for all tests
- Same network conditions (or account for variance)
- Multiple runs to reduce noise

---

## Success Criteria

A successful approach should achieve:

| Metric | Target |
|--------|--------|
| Latency (100 turns) | < 60 seconds |
| Cost (100 turns) | < $0.10 |
| Symbol extraction F1 | > 0.8 |
| Category accuracy | > 0.7 |
| Failure mode recall | > 0.5 (if extracting) |

The "winner" is the approach with best position on the Pareto frontier for the user's priorities.

---

## Open Questions

1. **How much quality degradation is acceptable?** 
   - For search: probably a lot (embeddings dominate)
   - For friction detection: probably not much (needs failure_mode)

2. **What's the distribution of turn complexity?**
   - Need to analyze real data to validate H2 (most turns are low-signal)

3. **Can we fine-tune a small model for this specific task?**
   - Would a 1B model fine-tuned on capsule extraction beat a 70B general model?

4. **Is there a way to detect extraction confidence?**
   - If we could flag "low confidence" extractions, we could re-run with better model

5. **How do users actually use the data?**
   - If 90% of queries are symbol-based, optimize for that
   - If recall/narrative is primary, need quality extraction

---

## Next Steps

1. **Data analysis**: Characterize the turn distribution in real OpenCode data
2. **Ground truth creation**: Manually write capsules for 50 turns
3. **Implement approaches**: Start with simplest (parallel, batched)
4. **Run experiments**: Measure latency, cost, quality
5. **Analyze results**: Find Pareto frontier, make recommendations
6. **Write blog post**: Share findings with community
