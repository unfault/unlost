# Experimental Results

## Strategy Benchmarks

Data collected on Sprint Set (50 turns from real OpenCode history).

| Strategy | Total Latency | Latency/Turn | Search Recall@5 | Friction F1 | Cost/Turn |
|----------|---------------|--------------|-----------------|-------------|-----------|
| **Raw Text** | 15.5s | 310ms | **87.0%** | 0.0 | $0 |
| **Level 0** | 9.8s | 196ms | 73.9% | 0.0 | $0 |

---

## Detailed Analysis

### 1. Dimension: Semantic Search (Query)

- **Finding**: Embedding the raw user + assistant text outperforms a basic heuristic extraction (first line of user/assistant).
- **Rationale**: Agents often have boilerplate or thought processes in the first lines. The core semantic context might be buried in the middle or end of the turn. Raw text preserves this context for the embedder.
- **Pareto Note**: Level 0 is faster but significantly less useful for retrieval. Raw Text is the current winner for zero-cost search.

### 2. Dimension: Friction Detection (Check)

- **Finding**: Zero-LLM strategies currently achieve **0.0 F1 score** on friction detection.
- **Rationale**:
    1. Level 0/Raw strategies do not populate the `failure_mode` field.
    2. The `unlost::governor` logic relies on **strong emotion** or **symbol repetition**.
    3. Many friction turns in our dataset (e.g. "let's try again") are classified as neutral by the local ONNX model and don't touch files immediately, so they fly under the radar of symbol-based heuristics.
- **Pareto Note**: High-fidelity friction detection likely *requires* LLM-based multi-turn reasoning or a significantly more aggressive emotion/pattern heuristic.

### 3. Latency Observations

- Latency is dominated by **local embedding generation** (fastembed).
- Embedding raw text (longer strings) takes ~1.5x longer than embedding extracted intents (shorter strings).
- On this machine, throughput is ~3-5 turns per second for embedding.

---

## Validating Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| **H1** | Tasks have different info requirements | **Supported** | Search works well with raw text; Friction doesn't. |
| **H6** | Embedding raw text is sufficient for search | **Supported** | 87% Recall@5 with zero extraction. |
| **H3** | Extraction is decomposable | **In Progress** | Need to test LLM-enhanced tiers. |
| **H4** | Batching is superlinear efficient | **Pending** | Need LLM bench. |
| **H5** | Local models are "good enough" | **Pending** | Need Ollama/SLM bench. |
