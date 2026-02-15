# Research Memo: Alignment with EASE'25 Paper (Martinez Montes & Khojah)
*Date: Feb 15, 2026*
*Paper Reference: [arXiv:2504.10050](https://arxiv.org/abs/2504.10050) - "Emotional Strain and Frustration in LLM Interactions in Software Engineering"*

## Executive Summary
The EASE'25 paper independently validates the **Three-Basin Architecture** of Unlost. By surveying 62 software engineers, the authors identified four primary "Frustration Triggers" that map directly onto our current trajectory sensors. Crucially, they confirm that **context window limitations** and **adaptability to context** are major drivers of emotional strain, supporting our recent "Context Load" findings.

---

## 1. Trigger Mapping (Sanity Check)

| Paper Trigger Category | Unlost Basin/Sensor Alignment | Operational Match |
| :--- | :--- | :--- |
| **Repeated Inaccuracies / Hallucinations** | **Drift Basin** ($s_{path}$, $s_{stall}$) | "Stubbornly insisting on an incorrect answer" is detected via Grounding Stall and Path Hallucination sensors. |
| **Intent Not Understood** | **Spec Basin** ($s_{stat}$, $s_{corr}$) | "Receiving irrelevant responses" or "talking to LLM like a child" maps to Instruction Staticness and Alignment Debt (corrections). |
| **Personal Preferences Unmet** | **Spec Basin** (Affective Weighting) | Frustration with "ramblings" or "apologies" is captured by our high-confidence emotional modulation (Affective Boost). |
| **LLM Limitations (Context/Memory)** | **Loop Basin** ($s_{churn}$, $s_{rep}$) | Explicitly mentions "context window size" as a trigger. Matches our finding that friction rate doubles at 12k-20k tokens. |

---

## 2. Key Insights for Unlost Strategy

### A. The "Context Inflection" Validation
The paper identifies **context window size** and **memory utilisation** as critical friction points. 
- **Action**: Our newly implemented `unlost metrics` context-size diagnostic is the correct tool for surfacing this. It provides the "Cognitive Mirror" that warns the user when they are entering the high-instability 12k+ token zone identified in both the paper and our field tests.

### B. "Towards a Less-Frustrating UX"
The authors recommend several system improvements. Here is how Unlost already addresses them:
- **Transparency (Confidence/Source)**: Our "Ambient Notes" ($I_t < 0.8$) provide soft nudges rather than hard blocks, increasing transparency about trajectory health without being intrusive.
- **Clarification Questions**: Our Spec-basin interventions (e.g., "Compass Notes") explicitly prompt the agent to *summarize its understanding* before writing more code.
- **Adaptability in Communication**: Our **Logic Churn** sensor ($s_{churn}$) detects the "re-arguing" behavior the paper highlights as a major annoyance.

### C. Waste-Weighted Value
The paper notes that software engineers are "resilient" and "motivation is not necessarily affected," but **long-term strain leads to burnout**.
- **Alignment**: This justifies our focus on **High-Cost Friction Windows**. We don't need to catch every nit; we need to catch the *expensive trajectory breakdowns* that contribute to long-term cumulative strain (the "Babysitting Tax").

---

## 3. Recommended Refinements (Gap Analysis)

Based on the paper's findings, we should consider these subtle sensor/policy adjustments:

1.  **"Stubbornness" Persistence**: The paper highlights frustration when the LLM "insists on an incorrect answer." 
    - *Refinement*: Ensure our **Loop Basin** persistence gating (3-turns) is particularly sensitive to cases where `logic_churn` is LOW but `alignment_debt` (user corrections) is HIGH (the "Repetitive Wrongness" signature).
2.  **Anti-Apology Damping**: Users reported frustration with LLMs constantly apologizing.
    - *Refinement*: We should add "apology" lexical cues to our **Intent Damping** logic (similar to how we damp "summary" intents). If the agent is being overly submissive/apologetic, it can mask real progress and increase user annoyance.
3.  **Source Grounding**: The paper emphasizes "Transparency (source)."
    - *Refinement*: Our **Drift Basin** interventions should explicitly ask the agent to "list 3 verified facts" or "re-read specific files" to force a grounding reset.

---

## Conclusion
The Unlost Three-Basin regulator is empirically aligned with the latest academic research on SE-specific friction. Our decision to focus on **proactive trajectory regulation** rather than just "emotion sensing" is the right strategic move—it solves the root causes (drift, spec, loop) identified by practitioners.
