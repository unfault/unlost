# Methodology: Measuring Utility-Cost Tradeoffs

## Experimental Framework

To validate our hypotheses, we need a framework that measures both **cost** (latency, tokens, money) and **utility** (how well the extracted data performs in real-world tasks).

### 1. Cost Metrics (Independent of Utility)

| Metric | Measurement |
|--------|-------------|
| **Latency** | Wall-clock time (seconds) to process the entire test set |
| **Throughput** | Turns per second |
| **Cost ($)** | Actual API cost based on token usage and model pricing |
| **Token Efficiency** | Total tokens (input + output) per turn |

---

### 2. Utility Metrics (Task-Specific)

We define "Utility" as the performance of a downstream task when using a specific extraction strategy.

#### A. Semantic Search Utility (Query)
- **Goal**: Measure how well the representation supports semantic retrieval.
- **Benchmark**: A set of 20 human-written natural language queries mapped to specific "Ground Truth" turns.
- **Metric**: **Recall@5** (is the correct turn in the top 5 results?).
- **Baseline**: Raw turn text embedding.

#### B. Friction Detection Utility (Check)
- **Goal**: Measure the ability to detect agent failure modes (loops, drift, etc.).
- **Benchmark**: 10 known "friction moments" from the dataset where an agent gets stuck or drifts.
- **Metric**: **F1-Score** for failure mode detection (Precision/Recall).
- **Metric**: **Friction Delay** (how many turns after the failure starts is it detected?).

#### C. Narrative Recall Utility (Recall)
- **Goal**: Measure the quality of generated summaries/narratives.
- **Benchmark**: 5 sessions summarized using the extracted capsules.
- **Metric**: **LLM-as-a-Judge Score (1-5)**. A "Teacher" model (GPT-4o) rates the narrative based on accuracy, conciseness, and clarity against a ground truth session overview.

---

### 3. Extraction Strategies to Benchmark

We will test the following strategies (The "Treatment Groups"):

| Strategy | Level | Description |
|----------|-------|-------------|
| **Baseline** | 4 | Sequential `gpt-4o-mini` calls per turn (Current) |
| **Parallel** | 4b | 8 concurrent `gpt-4o-mini` calls per turn |
| **Batched** | 3 | Batch 10 turns per `gpt-4o-mini` call |
| **Tiered-Fast**| 1 | Local SLM (Qwen2.5-1.5B) for intent + Local regex for symbols |
| **Tiered-Hybrid**| 2 | SLM for all turns + `gpt-4o-mini` for "Pivotal" turns only |
| **Heuristic** | 0 | Local regex for symbols + Raw text embedding only |

---

### 4. Experimental Procedure

1. **Step 1: Dataset Preparation**
   - Select 200 turns from OpenCode history across 3 sessions.
   - Manually label symbols and write "Ideal Capsules" for 50 turns (Ground Truth).
   - Identify 10 "Friction Moments" for the Friction benchmark.

2. **Step 2: Execution**
   - Run each Strategy against the 200 turns.
   - Record all Cost Metrics.
   - Store the resulting capsules/embeddings in separate LanceDB tables.

3. **Step 3: Utility Measurement**
   - Run the Search Benchmark (Recall@5) against each table.
   - Run the Friction Benchmark (F1-Score) against each table.
   - Generate Narratives for the 5 sessions and have the Teacher LLM score them.

4. **Step 4: Pareto Analysis**
   - Plot Utility (Y) vs Cost (X) for each task.
   - Identify which strategies are on the Pareto frontier.

---

### 5. Success Criteria for Methodology

- **Reproducibility**: The benchmark harness should be runnable with a single command.
- **Sensitivity**: The utility metrics should show clear differentiation between strategies (e.g., Heuristic should score low on Recall Narrative).
- **Validity**: The Teacher LLM scoring must align with human intuition (verified via spot-check).
