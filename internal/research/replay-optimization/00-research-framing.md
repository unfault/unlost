# Research Framing: The Deeper Question

## Stepping Back

The surface problem is "replay is slow." But the deeper question is:

> **What is the minimal representation of a conversation turn that preserves its utility for downstream tasks?**

This is fundamentally an **information compression** problem. We have:
- **Input**: Raw conversation turn (~1-10KB of text)
- **Output**: IntentCapsule (~500 bytes of structured data)
- **Compression ratio**: ~10-20x

The LLM is acting as a **lossy compressor** that preserves semantic information while discarding syntactic noise.

## The Real Questions

### Q1: What information do downstream tasks actually need?

Current downstream tasks in unlost:
1. **Semantic search** (`query`) - needs: embedding of intent/decision
2. **Friction detection** (`check`) - needs: symbols, failure_mode, emotion
3. **Narrative generation** (`recall`) - needs: intent, decision, rationale
4. **Symbol tracking** - needs: symbols array

Different tasks have different information requirements:

```
Task              | symbols | category | intent | decision | rationale | failure_mode
------------------|---------|----------|--------|----------|-----------|-------------
Semantic search   |    -    |    -     |   H    |    H     |     M     |      -
Friction detect   |    H    |    L     |   M    |    M     |     L     |      H
Narrative gen     |    M    |    M     |   H    |    H     |     H     |      M
Symbol tracking   |    H    |    -     |   -    |    -     |     -     |      -

H = High importance, M = Medium, L = Low, - = Not needed
```

**Insight**: If we could predict the user's primary task, we could optimize extraction accordingly.

### Q2: How much information is in a conversation turn?

A conversation turn has:
- **Explicit information**: What was literally said
- **Implicit information**: Context, references to prior turns, domain knowledge
- **Derived information**: Intent (why), decision (what), rationale (why that)

The derived information requires **inference** - you can't regex it out. This is why LLMs are needed.

But how much inference is really needed?

**Hypothesis**: Most turns have low information entropy. The derived information is largely predictable from explicit information + simple heuristics.

### Q3: Is extraction order-dependent?

Current approach: Extract each turn independently.

But conversation is sequential. Turn N's meaning depends on turns 1..N-1.

**Options**:
- **Independent extraction**: Simpler, parallelizable, but loses context
- **Sequential extraction**: Include prior turns in prompt, but slower
- **Hierarchical extraction**: Extract session-level summary, then turn-level within that context

### Q4: What's the information-theoretic minimum?

For search to work, we need the **embedding** of the turn's semantic content.
- Embedding dimension: 384 (BGE-small)
- Bits: 384 * 32 = 12,288 bits = 1.5KB

So the theoretical minimum representation for search is ~1.5KB per turn.

For friction detection, we need:
- Symbols: Variable, but often <10 paths = ~500 bytes
- Failure mode: 1 of 7 categories = 3 bits + optional explanation

For narrative, we need natural language - harder to compress.

**Insight**: Different use cases have fundamentally different compression limits.

---

## Reframing the Problem

Instead of "how do we speed up extraction?", the question is:

> **How do we design an extraction pipeline that produces representations optimized for each downstream task, with the right quality/cost tradeoff?**

This suggests a **multi-representation** approach:

```
Raw Turn
    │
    ├──[instant]──→ Embedding (for search)
    │
    ├──[instant]──→ Symbols (for tracking, friction)
    │
    ├──[fast]────→ Category + lightweight intent (for filtering, basic recall)
    │
    └──[quality]──→ Full capsule (for narrative, failure detection)
```

Each representation has its own extraction path with different costs.

---

## The Hypothesis, Refined

**Main Hypothesis**: 

The utility of extracted data follows a **power law distribution** - a small amount of extracted information (symbols + embedding) provides most of the utility, while full extraction provides diminishing returns for most use cases.

```
Utility
  ^
  │        ╭────────────────── Full capsule
  │       ╱
  │      ╱
  │     ╱
  │    ╱
  │   ╱
  │  ╱
  │ ╱
  │╱__________________________ Embedding + symbols only
  └─────────────────────────────────────→ Extraction cost
```

**Corollary**: For bulk replay, we should optimize for the "knee" of this curve - maximum utility per unit cost.

---

## Experimental Approach

To validate this hypothesis, we need to measure **utility** not just **quality**.

### Utility Metrics

1. **Search utility**: Given a query, does the representation enable finding relevant turns?
   - Metric: Recall@K for semantic search
   - Test: Run queries, measure if correct turns are returned

2. **Friction utility**: Does the representation enable detecting friction?
   - Metric: Friction detection accuracy
   - Test: Replay history, check if friction warnings fire appropriately

3. **Narrative utility**: Does the representation enable generating useful recall?
   - Metric: Human judgment (or LLM-as-judge)
   - Test: Generate narratives, rate usefulness

### Experiment Design

```
For each extraction approach:
  1. Process test dataset (100 turns)
  2. Measure: latency, cost
  3. Run utility benchmarks:
     a. 20 search queries → measure recall@5
     b. Simulate friction detection on 20 "friction moments" → measure detection rate
     c. Generate 5 recall narratives → rate quality (1-5)
  4. Plot: utility vs cost
  5. Identify Pareto-optimal approaches
```

---

## What We're Really Trying to Learn

1. **Is full LLM extraction necessary for useful search?**
   - If embedding alone is sufficient, we can skip extraction entirely for search use case

2. **Can we detect friction with local-only extraction?**
   - If symbols + emotion is enough, friction detection becomes instant

3. **What's the minimum extraction for acceptable recall narratives?**
   - Maybe category + intent is enough, don't need rationale

4. **Is there a "universal good enough" representation?**
   - Or must we always choose based on use case?

---

## Blog Angle

The blog post isn't about "we made it faster" - it's about:

> **"We discovered that 80% of the value comes from 20% of the extraction work. Here's how we measured it and what it means for building local-first AI tools."**

This is a story about:
- Information theory applied to LLM pipelines
- Measuring utility, not just quality
- Finding the Pareto frontier
- Designing systems that let users choose their tradeoff

This is interesting to:
- AI tooling developers (practical guidance)
- Researchers (methodology for measuring extraction utility)
- Product thinkers (how to design quality/cost tradeoffs)
