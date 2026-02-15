# Research Progress: Extraction Granularity for Agent Memory

This file serves as the "living memory" of our research to ensure continuity and accuracy as we progress.

## Current State: Baseline Results
*Status as of Feb 14, 2026*

| Strategy | Latency (50 turns) | Search Recall@5 | Extr. Friction F1 | Gov. Friction F1 | Symbol F1 | LLM Cost (Sim) |
|----------|-------------------|-----------------|-------------------|------------------|-----------|----------------|
| **Raw Text (U+A)**     | 13.4s | **88.5%** | 0.00 | 0.50 | 0.11 | $0.00 |
| **Assistant-only**     | 12.4s | 80.8% | 0.00 | 0.50 | 0.11 | $0.00 |
| **User-only**          | 12.6s | 65.4% | 0.00 | 0.50 | 0.11 | $0.00 |
| **Capsule-Trusted**    | 12.7s | 76.9% | 0.57 | **0.75** | 0.08 | $0.00 |
| **Pivotal Hybrid**     | 14.0s | 88.5%* | **0.90** | **0.85** | **0.80** | **~$0.04** |

*\*Note: Hybrid uses Raw Text for its search index.*

### Key Findings
1. **Search Ablation: The Assistant carries the Signal.** Assistant-only embedding (81%) significantly outperforms User-only (65%). However, combining both provides the best results (88.5%), confirming that user intent and assistant execution are synergistic for retrieval.
2. **Pivotal Filtering: The 80/20 of Memory.** We identified 42% of turns as "Pivotal" using local heuristics (length, symbols, emotion). By only running full extraction on these, we can maintain high-fidelity narratives while cutting LLM spend by 58%.
3. **Governor Utility: Trusting Memory is the multiplier.** By allowing the Governor to trigger warnings based on the capsule's `failure_mode` (Stateful Governor), we improved Friction Utility from 50% to 75% F1.
4. **Latency Floor**: With LLMs removed, latency is dominated by local embedding generation (~200-300ms/turn via fastembed).

## Hypotheses Status

| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| H1 | Task-specific requirements | **Supported** | Search works with raw text; Friction needs more. |
| H2 | Low-signal turns (80/20) | **Supported** | 42% of turns identified as "Pivotal" via local signals. |
| H3 | Extraction is decomposable | **Supported** | Tiers 0-2 provide distinct utility levels. |
| H4 | Batching is superlinear | *Pending* | Need real LLM bench to confirm network amortization. |
| H5 | Local SLM is "good enough" | **Rejected (Low-end)**| Struggle with JSON schema on low-end hardware. |
| H6 | Raw text is enough for search| **Supported** | 88.5% Recall@5 with zero extraction. |

## Work Log & Decisions

### Feb 14, 2026
*   **Search Ablation**: Confirmed Assistant-first signal for retrieval.
*   **Capsule-Trusted Governor**: Validated that the Governor becomes significantly more useful when it "trusts" the extracted memory.
*   **Pivotal Hybrid**: Simulated a tiered strategy where only 40% of turns get full LLM treatment.
*   **Symbol decision**: Acknowledged that high-fidelity symbol extraction (discussed symbols) is hard for regex. However, since Raw Text captures them for search, we can accept low Symbol F1 for --fast mode.
*   **Decision**: For historical replay, the default should be **Ghost Replay** (Raw Text + Keyword Friction). Users can opt-in to **Hybrid Replay** for high-quality narratives.

## Next Steps
1. [ ] **Terminal UX Design**: Define non-intrusive rendering for "Ambient" vs "Structural" notes.
2. [ ] **Logic Churn Sensing**: Research if multi-turn rationale divergence improves coverage.
3. [ ] **Simulate/Test Batching**: Measure efficiency of processing 10 turns in one Cloud LLM call.

## Deep Dive Notes (Search + Friction + Emotion)

### 1) Query/Search: “The Search Knee” (Raw Embeddings)
**Observation**
- Embedding raw turn text (`User + Assistant`) performs best among zero-LLM strategies (≈88.5% Recall@5).
- Assistant text (81%) is a much stronger retrieval driver than User text (65%).

**Interpretation (why raw text can win)**
- Raw text preserves “topical breadth” (many discriminative tokens) that naïve compression can destroy.
- Result is more accurately described as: **bad compression hurts retrieval** more than raw text noise does.

### 2) Friction: two different problems
We should keep these distinct:

**A) Friction labeling (replay / history reconstruction)**
- “Can we tag turns with retry_spiral/drift/etc cheaply?”
- Observation: keyword heuristics can reach meaningful extraction-level friction accuracy (Extr. F1 ≈ 0.57).

**B) Friction triggering (real-time Governor behavior)**
- “Does Unlost inject a warning at the right time given symbols/emotion/history?”
- Observation: Governor F1 stayed flat until we "trusted" the capsule labels, jumping to 0.75 F1.

### 3) Emotion & Recall: The Context/Zone Multiplier
**Insight**
- Recall should be factual but "tainted" by the general emotion/mood of the session. This helps put the user back in the "context/zone" when they re-read history.
- Using **Raw Text** for emotion sensing (Option B) allows for finer detection than summarized snippets.
- Emotions are lagging signals for the user, but for the system, sensing them early helps avoid increasing tension (preventing the "Babysitting Tax").

**Architectural Shift (Option B)**:
- **Index**: Embed raw text for maximum retrieval utility.
- **Sense**: Classify emotion from raw text for high-fidelity "mood" tracking.
- **Store**: Metadata + Vector in DB; keep raw text in JSONL logs for deep reconstruction if needed.

## Trajectory Research (Marathon Set - 390 turns)

### Precursor Hypothesis: "Pivotal Bursts as Early Warnings"
**Test**: Does high density of pivotal turns (local signal) precede a Governor-detected friction event?

**Result**: **REVISED (Feb 15)**.
- Previous test (predicting heuristic friction) failed.
- **New Test**: Predicting **User Dispute** from **Pivotal Density** achieved **80% Precision@5** with **3.1 turn lead**.
- High pivotal density **does** precede explicit user frustration.

**New Learning: The Instability Signature**
- Friction episodes (e.g., Turn 22-36) show an average pivotal density of **0.92**.
- Non-friction episodes show a baseline density of **<0.20**.
- **Insight**: Pivotal signals (length, churn, negative valence) are the **phenomenology of friction**. They are what friction *looks like* before it is officially classified as a `FailureMode`.

### Status of User-Provided Hypotheses

- **H1 (Early signals)**: **Supported**. Pivotal density leads user dispute by ~3 turns.
- **H2 (Spiral Signature)**: **Confirmed**. Spirals have a distinct density profile (0.9+).
- **H5 (Intervention works)**: **Confirmed**. Trusting state improved F1 to 0.75.
- **H6 (Instability correlates with waste)**: **Support**. Long sessions show clear high-density bursts corresponding to loops.

## Trajectory & Regulation Research (Goal 2: Proactive Warnings)

### Current Model: Trajectory Control (Feb 15, 2026)
We have shifted from "static labeling" to a **Trajectory Model** for proactive friction regulation. This transition represents an evolution from *detecting failure* to *modeling interaction dynamics*.

- **Primary Goal**: **2) Proactive Warnings** (Intervene before user frustration).
- **Secondary Goals**: **1) Zone Reconstruction** and **3) Post-hoc Diagnosis**.
    - *Decision*: We are focusing on Goal 2 for now to maximize immediate product utility. However, Goal 1 (analytics) and Goal 3 (debugging) remain in scope as potential extensions of the same trajectory data.
    - *Note*: This hierarchy of goals is a key strategic question and should be revisited as the model matures.
- **Formalized Doc**: `internal/research/replay-optimization/08-trajectory-model.md`

### From Observations to a "Proper Model"
To move beyond a collection of research findings into a robust engineering model, we have identified the following requirements:

1. **State-Space Formalism**: Defining the "Loop Basin" not just as a heuristic, but as a region in a multi-dimensional symptom space (repetition, novelty, effort).
2. **Controller Logic**: Implementing the **Stable → Watch → Intervene** transition rules with mathematically grounded thresholds and burst-gating.
3. **Policy: "Surfing the Emotional Wave"**:
    - **Detection**: Uses leading symptom indicators to identify a "loop trajectory" before the user gets frustrated.
    - **Intervention**: Uses lagging affect to decide *how* to intervene.
        - **Confusion/Doubt**: Inject "Compass Notes" (rationales) to provide transparency.
        - **Frustration**: Inject "Hard Resets" to force planning.
        - **Anger (Surprise/Escalation)**: Trigger "Emergency Brake" (stop all execution).
    - **Affective Resets**:
        - **Joy**: High-confidence positive feedback acts as a **Damping Factor** on instability intensity $I_t$.
        - **Reframe**: Explicit user intent to restart resets the controller immediately.
4. **Temporal Boundaries**: Explicitly modeling human state resets (e.g., the 30m "Coffee Pause") to avoid stale state pollution.

### Validation Status (Marathon Set)
- **Precision@5**: **72.2%** (at $\theta_I=0.8$)
- **Coverage@5**: **7.6%** (at $\theta_I=0.8$) to **27.3%** (at $\theta_I=0.4$)
- **Lead Time**: **1.2 - 1.8 turns** (head-start before explicit dispute).
- **Weights Derived**: $w_{eff}=0.34$, $w_{rep}=0.24$, $w_{nov}=0.24$, $w_{sem}=0.18$, $w_{corr}=0.45$.
- **New Features**: 
    - **Persistence Gating** (3-turn high intensity).
    - **Soft Coffee Pause** (30% decay).
    - **Alignment Debt Basin**: Added detection for user-side corrections ("Actually", "Not that") to address instruction misunderstanding.

### The Coverage Gap & Specification Basin
Our current model has high precision but low coverage (~8-27%). We have addressed the **Instruction Misunderstanding** gap by adding the **Specification Basin** layer (Alignment Debt). 

### New Research: Drift & Specification Regulation (Closing the Gap)
*Status as of Feb 15, 2026*

We have completed the Coverage Expansion phase, successfully filling the gaps in "Instruction Misunderstanding" and "Factual Drift" by implementing deeper sensors and persistence gating.

#### 1. Implementation: The Three-Basin Architecture
The trajectory regulator now operates across three distinct "Basins of Friction," implemented in `src/governor.rs`:

| Basin | Goal | Leading Sensors | Gating |
| :--- | :--- | :--- | :--- |
| **Loop** | Catch stalls | Repetition, Novelty Collapse, Effort Spike | Hysteresis |
| **Spec** | Catch misalignment | **Alignment Debt** (corrections), **Instruction Staticness** | Persistence |
| **Drift** | Catch grounding failure | **Grounding Stall**, Path Hallucination | Persistence |

- **Grounding Stall ($s_{stall}$)**: Detects when the agent ignores specific file paths mentioned by the user in the previous turn.
- **Instruction Staticness ($s_{stat}$)**: Detects when the user repeats long structural instructions verbatim (breakdown in understanding).
- **User Symbols**: Added a dedicated `user_symbols` channel to `IntentCapsule` to track what the user actually asked for vs. what the agent touched.

#### 2. Statistical Validation (Field Simulation)
We ran a comprehensive robustness report using both curated datasets and a 1,224-turn local history replay.

| Metrics (Marathon 390) | Committed (`dfb7591`) | Worktree (Final) | Coverage Delta |
| :--- | :--- | :--- | :--- |
| **Precision@5** | 83.9% | 67.7% | -16.2% |
| **Coverage@5** | 15.1% | **24.4%** | **+61%** |
| **Triggers** | 31 | 62 | +100% |

**Key Findings:**
- **Coverage Expansion**: Coverage of user disputes increased by **61%** on Marathon and **100%** on Sprint sets.
- **Precision Floor**: Precision remained high enough (~68%) for proactive "system notes."
- **Healthy Stress**: The model remains silent during deep refactors and exploratory debugging (Max Intensity 0.62 < 0.80 threshold).

#### 3. Enhanced Analytics: The Cognitive Mirror
We have upgraded `unlost metrics` to provide a complete picture of workspace health:
- **Friction Breakdown**: Separates triggers by cause (loop, spec, drift, legacy).
- **Average Intensity**: Tracks the severity of friction episodes.
- **Top Friction Files**: Identifies specific codebase "hotspots" where the agent consistently stalls or drifts.

### Stability Hardening & Stratified Policy
*Status as of Feb 15, 2026 (Post-v0.3.0)*

We have moved beyond pure detection into **Control-System Rigor**, addressing oscillation risks and policy confidence.

#### 1. Implementation: Control Constraints
- **Per-Basin Refractory Periods (Cooldowns)**:
    - **Loop**: 5-turn cooldown (prevent nagging on same repetition).
    - **Spec/Drift**: 2-turn cooldown (allow quicker re-alignment).
- **Stratified Intervention Policy**:
    - **Ambient Note ($I_t < 0.8$)**: Soft hints (non-blocking).
    - **Structural Note ($0.8 \le I_t < 0.95$)**: Heavy context injection (Hydration/Facts).
    - **Actionable Intervene ($I_t \ge 0.95$)**: CRITICAL emergency brake.
- **Grounding Decay**: Implemented exponential decay on user-mentioned path importance ($weight = e^{-0.2 \cdot age\_mins}$) to prevent "stale" grounding penalties.

#### 2. Statistical Validation (H-Curve Sweep)
We analyzed the **Coverage@H** curve (H=1..15) to determine if our window was too narrow.

| H (Lookahead) | Worktree Precision | Worktree Coverage |
| :--- | :--- | :--- |
| 1 | 62.9% | 22.7% |
| 5 | **67.7%** | **24.4%** |
| 10 | 77.4% | 25.6% |
| 15 | 83.9% | 26.7% |

**Insight**: The curve is remarkably flat. Increasing H from 5 to 15 only adds ~2% coverage. This confirms that missed disputes are **not preceded by measurable tension** in the current symptom space, rather than us being "too early."

### The "Model Gap" (Next Steps for Formalization)
- [x] **Mathematical Mapping**: Formal score function $I_t = \sum w_k d_t$ and trajectory slope $T_t = I_t - I_{t-\ell}$.
- [x] **Calibration Protocol**: Systematic threshold tuning using percentiles (derived $\theta_I=0.8$ for high precision).
- [x] **Intervention Taxonomy**: Mapping `(Trajectory, Affect)` states to specific "Control Actions."
- [x] **Intervention Substance**: Dynamic payload generation (Hydration Packets, Resumption Briefs).
- [x] **Stability Hardening**: Refractory periods and Stratified Policy (v0.3.0).
- [x] **Cross-Dataset Validation**: Generalization confirmed across Marathon and Sprint sets.
- [x] **Enhanced Metrics**: Detailed friction breakdown in `unlost metrics`.

## Remaining Tasks (Pending Phase)
1. [ ] **Terminal UX Design**: Render stratified notes in the CLI.
2. [ ] **Logic Churn Sensing (Rationale)**: Research rationale text divergence.
3. [ ] **Live Trial**: Enable regulator in interactive session.
4. [ ] **Analytics Dashboard**: Leverage symbol-specific metrics for workspace health.

## Questions to Revisit Later (Parking Lot)
- **Rationale Depth**: Should we store full rationales or just a semantic hash for churn detection?
- **VSCode Integration**: Can these trajectory warnings be surfaced as "Ambient Awareness" in the IDE?
- **Embedding engine choice**: Is `fastembed` the right speed/quality tradeoff?

