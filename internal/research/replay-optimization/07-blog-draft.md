# Blog Post: The 80/20 of AI Agent Memory

## Hook
Replaying 1,000 turns of AI agent history shouldn't cost as much as a sandwich or take as long as a coffee break. But in current memory systems, it often does. In our latest research for Unlost, we discovered that 80% of the value of agent memory comes from 20% of the extraction effort.

## The Problem: The "Babysitting Tax"
AI coding agents (Claude Code, OpenCode, Aider) generate massive JSONL logs. To make this history searchable and useful for friction detection, we typically "reconstruct" it into structured capsules using LLMs. 

Each turn requires an LLM call to extract intent, decisions, and failure modes. This is slow (~2s/turn), expensive ($0.01/turn), and hits rate limits quickly. For a single active project, this is the "Babysitting Tax" we all pay to keep our agents oriented.

## The Research Question
What is the minimal granularity of data we need to still provide value for **Search**, **Friction Detection**, and **Narrative Recall**?

## Methodology
We built a benchmark harness (`unlost-bench`) and tested multiple strategies against a "Golden Set" of real-world agent conversations. We measured wall-clock latency, API costs, and "Utility" (Recall@5 for search, F1 for friction).

## Key Discoveries

### 1. The "Search Knee": Raw Text is the Silver Bullet
We found that embedding the **raw User + Assistant text** (no extraction) yielded an **88.5% Search Recall@5**. 
In our "Search Ablation" experiment, we discovered that the **Assistant's response** is the primary semantic anchor (81% recall on its own), while the User's intent provides the final ~8% boost.

**Insight**: "Cleaning" the text into a structured capsule actually *risks* discarding semantic context that the embedder needs. For retrieval, raw is better.

### 2. The "Friction Gap": Keywords are the Bridge
Friction detection (catching an agent going in circles) is hard for local heuristics. Baseline heuristics achieved 0% accuracy. However, we improved this to **57%** simply by adding a "Keyword Boost" (regex for "try again", "stuck", "retry"). 

**Insight**: You don't always need a 70B model to know when a user is frustrated.

### 3. The "Stateful Governor" Multiplier
We discovered that the `Governor` (the part of Unlost that triggers warnings) was "Memory-Blind." Even when we correctly identified a failure during extraction, the Governor would ignore it and try to re-derive it from raw signals. 

By switching to a **Stateful Governor** that "trusts" the capsule's metadata, our Friction Utility jumped from **50% to 75% F1**.

### 4. The 80/20 of Memory: Pivotal Filtering
Not every turn is a milestone. Most turns are incremental code edits. By using local heuristics (message length, symbol churn, emotion) to identify **"Pivotal Turns,"** we identified that only **42% of history** actually requires deep LLM extraction. 

**Insight**: We can maintain high-fidelity narratives while **cutting LLM costs by 60%**.

## The Result: The Tiered Memory Model
Based on these findings, we're introducing a tiered approach to memory reconstruction in Unlost:

1.  **Ghost Mode (--fast)**: Zero-cost, near-instant. Uses raw text embeddings + keyword friction. Perfect for replaying months of history in seconds.
2.  **Pivotal Mode**: Hybrid strategy. Ghost for the noise, Pro for the signals. High-fidelity narratives at a fraction of the cost.
3.  **Deep Mode**: Full sequential LLM extraction for maximum reasoning.

## Conclusion: Engineering for Information Density
Information Theory tells us that most data is noise. By identifying the "semantic signals" that actually drive search and friction detection, we can build agent memory systems that are both fast and affordable. 

The "Babysitting Tax" just got its first major cut.

---
*Unlost is an open-source orientation system for AI agents. Check out the benchmark harness and research notes in our [GitHub repository].*
