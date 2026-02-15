# Drift Regulation: Sensing Knowledge Misalignment

This document formalizes the **Drift Basin**, where friction arises not from repetition (Loop), but from a mismatch between the agent's mental model and the workspace reality.

## 1. Defining Drift

Drift is a **Grounding Failure**. It occurs when:
1. **Hallucination**: The agent references files, APIs, or behaviors that do not exist.
2. **Knowledge Stale-ness**: The agent believes a previous edit succeeded when it failed (or wasn't applied).
3. **Instruction Drift**: The agent's interpretation of the goal diverges from the user's explicit corrections.

---

## 2. Leading Symptoms for Drift

Unlike Loop symptoms (which are repetitive), Drift symptoms are **Divergent** or **Stalled**.

| Metric | Symbol | Definition |
| :--- | :--- | :--- |
| **Grounding Stall** | $s_{stall}$ | User mentions paths $P_u$, but agent touches disjoint set $S_a$ ($P_u \cap S_a = \varnothing$). |
| **Instruction Erosion** | $s_{eros}$ | Agent stops mentioning North Star symbols while production effort ($s_{eff}$) remains high. |
| **Semantic Staticness** | $s_{stat}$ | User repeats long structural instructions verbatim (detected via Jaccard on user turns). |
| **Path Hallucination** | $s_{path}$ | Rate of mentioned paths that do not exist in the workspace. |
| **Assumption Load** | $s_{asmp}$ | Frequency of hedging verbs ("assume", "likely", "probably") in the rationale. |

---

## 3. The Drift Intensity Channel ($I_{drift}$)

We add a third intensity to our controller:
$$I_{drift} = w_{path} s_{path} + w_{stall} s_{stall} + w_{stat} s_{stat}$$

### 3.1 State Transitions (Drift)
- **Stable → Watch(Drift)**: $I_{drift} > 0.4$.
- **Watch → Intervene(Drift)**: $I_{drift} > 0.7$ OR Grounding Stall persists for 3 turns.

---

## 4. Discovery-Aware Alignment

In discovery-based workflows, "Done When" is unknown. Stability is maintained via **Micro-Agreements**.

### 4.1 Indicators of Discovery Drift
- **Oscillation**: User toggles between two instructions (A -> B -> A).
- **Instruction Shadowing**: Agent acknowledges a correction ("Actually no X") but continues to use symbols related to X in the next turn.
- **Spirit Divergence**: If a "Style/Spirit" anchor was provided (e.g., a URL or a specific project reference), and the agent's proposed symbols diverge semantically from that anchor.

### 4.2 "Staff Engineer" Interventions
Instead of asking "What is the goal?", the system provides **Low-Ego Checkpoints**:
- **"Just to be sure"**: "My current read is X. If that's not it, tell me and I'll pivot."
- **"A vs B"**: "Are we exploring [Concept A] or [Concept B]?"

---

## 5. Implementation Path

1. **Path Validator**: Enhance `src/metrics.rs` logic to provide a real-time $s_{path}$ signal to the Governor.
2. **Rationale Analyzer**: A cheap regex-based scanner for "Assumption Load" ($s_{asmp}$).
3. **Grounding Checker**: Compare the `symbols` in the `IntentCapsule` against the symbols found in the raw tool outputs of the same turn.

---

## 6. Closing the Coverage Gap

By sensing Drift, we target the ~40-60% of user disputes that are not preceded by a Loop. 
- **Hypothesis**: $I_{drift}$ will provide a head-start for "You're looking at the wrong file" or "That's not how the API works" moments.
