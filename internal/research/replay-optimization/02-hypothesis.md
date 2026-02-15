# Research Hypotheses

## Core Thesis

**The utility of extracted data follows a power law distribution**: a small amount of extracted information provides most of the value, while full extraction provides diminishing returns for most use cases.

```
Utility
   ^
100│                    ╭─────────── Full LLM extraction
   │                   ╱
 80│               ╱
   │            ╱      ← "Knee": Max utility per unit cost
 60│         ╱
   │       ╱
 40│     ╱
   │   ╱
 20│ ╱
   │╱
  0└──────────────────────────────────────→ Extraction Cost
        $0      $0.001    $0.005    $0.01
```

If true, we can dramatically reduce cost by targeting the "knee" of this curve.

---

## Hypothesis 1: Task-Specific Information Requirements

**H1**: Different downstream tasks have different minimum information requirements.

| Task | Hypothesized Minimum | Full Extraction Adds |
|------|---------------------|---------------------|
| **Search** | Embedding only | Marginal improvement |
| **Friction** | Symbols + emotion | Failure mode context |
| **Recall** | Intent + decision | Rationale + next steps |

**Testable prediction**: Search recall@5 with embedding-only will be within 10% of search recall@5 with full capsules.

**Null hypothesis**: Full extraction provides significant utility gains (>20%) for all tasks.

---

## Hypothesis 2: Most Turns Are Low-Signal

**H2**: The distribution of "information density" across turns is highly skewed.

In a typical coding session:
- ~10-20% of turns are "pivotal" (key decisions, failures, direction changes)
- ~80-90% are "incremental" (small edits, clarifications, routine work)

**Testable prediction**: Pivotal turns can be identified by local signals:
- User emotion score > 0.5 (frustration, confusion)
- Symbol churn > 3 new files
- Turn length in top 20%
- Keywords: "instead", "actually", "wait", "no", "decided"

**Null hypothesis**: Information density is uniformly distributed; all turns require equal extraction effort.

---

## Hypothesis 3: Extraction Is Decomposable

**H3**: The capsule fields can be grouped by extraction difficulty:

| Tier | Fields | Method | Latency | Cost |
|------|--------|--------|---------|------|
| **Tier 0** | symbols | Regex/pattern matching | <10ms | $0 |
| **Tier 1** | category, embedding | Local model + fastembed | <100ms | $0 |
| **Tier 2** | intent, decision | Small cloud LLM (gpt-4o-mini) | ~500ms | ~$0.001 |
| **Tier 3** | rationale, failure_mode | Quality cloud LLM | ~1000ms | ~$0.005 |

**Testable prediction**: Tier 0+1 alone provides >60% of search utility and >40% of friction utility.

**Null hypothesis**: Fields are interdependent; partial extraction produces incoherent/unusable capsules.

---

## Hypothesis 4: Batching Provides Superlinear Efficiency

**H4**: Batching multiple turns into a single LLM call reduces cost more than linearly.

**Reasoning**:
- Fixed overhead (network, prompt parsing) is amortized
- LLM may extract better with more context
- Fewer API calls = less rate limiting

**Testable prediction**: Batch size of 10 reduces per-turn cost by >50% with <10% quality degradation.

**Null hypothesis**: Batching provides only linear cost reduction, or quality degrades significantly.

---

## Hypothesis 5: Local Small Models Are "Good Enough"

**H5**: A local small model (1-3B parameters) can achieve acceptable quality for Tier 1-2 extraction.

**Candidates**:
- Qwen2.5-1.5B
- Phi-3-mini (3.8B)
- Llama-3.2-1B

**Testable prediction**: Local model achieves >70% of gpt-4o-mini quality for intent/decision extraction.

**Null hypothesis**: Small models produce significantly worse (<50%) quality, making them unsuitable.

---

## Hypothesis 6: Embedding Captures Most Semantic Value

**H6**: For search use cases, the embedding of raw turn text captures most of the retrievable information.

**Reasoning**:
- Embeddings are trained to capture semantic similarity
- The "cleaning" done by LLM extraction may not add much for retrieval
- Users search with natural language, which matches raw turn text well

**Testable prediction**: Search recall@5 using raw turn embeddings is within 5% of embeddings of extracted intent+decision.

**Null hypothesis**: LLM-extracted text produces significantly better embeddings for search.

---

## Summary of Predictions

| Hypothesis | Key Prediction | Validation Method |
|------------|----------------|-------------------|
| H1 | Tasks have different info requirements | Compare utility across tasks |
| H2 | 80% of turns are low-signal | Analyze turn distribution |
| H3 | Extraction is decomposable into tiers | Test tier combinations |
| H4 | Batching is superlinear efficient | Compare batch sizes |
| H5 | Local models are good enough | Compare local vs cloud |
| H6 | Embedding captures semantic value | Compare embedding sources |

---

## What Changes If Hypotheses Are Wrong

| If wrong... | Implication |
|-------------|-------------|
| H1 rejected | Must use full extraction for all tasks |
| H2 rejected | Cannot use pivotal-turn filtering |
| H3 rejected | Cannot do tiered extraction |
| H4 rejected | Batching not worthwhile |
| H5 rejected | Local models not viable |
| H6 rejected | Must extract before embedding |

Even if some hypotheses are rejected, the research will clarify the actual utility-cost relationship and inform product decisions.
