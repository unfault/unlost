# Trajectory Model: Proactive Friction Regulation

This document formalizes the shift from **per-turn friction detection** to **multi-turn trajectory control**.

## 1. Core Philosophy

Instead of asking "Is this turn failing?", we ask "**What is the momentum of this interaction, and is it heading into a loop basin?**"

- **Retrieval is about Compression**: Preserving semantic signal for search.
- **Regulation is about Trajectory**: Sensing and bending the path of the collaboration.

---

## 2. The State Space

We model the interaction as moving through a latent state space defined by **Leading Symptoms** and **Lagging Affect**.

### 2.1 Leading Symptoms (Observables)
These are computed from turn text, symbols, and usage metadata without using LLMs.

| Metric | Symbol | Definition |
| :--- | :--- | :--- |
| **Symbol Repetition** | $s_{rep}$ | Jaccard overlap of current symbols $U_t$ with recent window $U_{t-w}$. |
| **Novelty Collapse** | $s_{nov}$ | $1 - (\text{new symbols} / \text{total symbols})$. |
| **Semantic Stall** | $s_{sem}$ | Max cosine similarity of turn embedding $R_t$ to recent turns $R_{t-w}$. |
| **Effort Spike** | $s_{eff}$ | Normalized token burn or time-per-turn. |
| **Stall Composite** | $s_{stall}$ | Combined indicator: $s_{eff} \cdot s_{rep} \cdot s_{sem}$. |

### 2.2 Lagging Affect (Sentiment)
- **User Emotion**: Negative valence (frustration, confusion).
- **Explicit Dispute**: User keywords like "no", "wrong", "still broken".
- **Role**: Affect is a **lagging sensor**. We use it to calibrate the *urgency* and *style* of intervention, not necessarily to trigger the first warning.

---

## 3. The Controller: Stable → Watch → Intervene

We use a 3-state state machine to regulate the interaction.

### 3.1 Instability Metrics
We formalize the symptom channels into two primary intensities: **Loop Intensity** ($I_{loop}$) and **Alignment Debt** ($I_{spec}$).

For each symptom channel $k$, we compute a smoothed density $d^{(k)}_t$:
$$d^{(k)}_t = \alpha s^{(k)}_t + (1-\alpha)d^{(k)}_{t-1}$$

The **Total Instability Intensity** is a weighted sum:
$$I_t = \sum_k w_k d^{(k)}_t$$

**Initial Calibrated Weights ($w_k$):**
- $s_{eff}$ (Effort Spike): **0.34**
- $s_{rep}$ (Symbol Repetition): **0.24**
- $s_{nov}$ (Novelty Collapse): **0.24**
- $s_{sem}$ (Semantic Stall): **0.18**
- $s_{corr}$ (Alignment Debt / Corrections): **0.45**

The **Trajectory Slope** captures escalation:
$$T_t = I_t - I_{t-\ell}$$

---

## 4. The Specification Basin (Instruction Misunderstanding)

While the **Loop Basin** is defined by structural repetition, the **Specification Basin** is defined by **Alignment Debt**—observable volatility in the user's intent.

### 4.1 Leading Indicators for Misalignment
- **Correction Events**: User-side negated/correction markers ("no", "not that", "actually").
- **Grounding Mismatch**: Assistant proposing high-impact actions without explicit grounding in user intent.

### 4.2 Policy: Option 1 (Precision-First)
We only intervene on misalignment when we have **observable evidence** (repeated corrections). This respects discovery-based workflows where early vagueness is normal exploration.
- **Trigger**: 2+ correction events in a short window.
- **Intervention Style**: "Staff Engineer" voice—low-ego micro-contracts and one-at-a-time assumptions.

### 3.2 State Transitions
Based on Marathon set calibration:
- **Stable → Watch**: $I_t > 0.5 \wedge T_t > 0$
- **Watch → Intervene**: Trigger if $(I_t > 0.8 \wedge T_t > 0.05) \vee \text{Persistence}$
  - **Persistence**: $I_{t-\tau:t} > 0.75$ for $\tau=3$ turns (catches flat high loops).
- **Watch → Stable**: $I_t < 0.4$ for $r=2$ turns

### 3.3 Gating & Safety
- **The Burst Gate (G)**: To prevent false positives from slow, steady iteration, we only allow transitions when symptoms occur in bursts.
- **Cooldown**: Mandatory $C$ turns of `Stable` after an intervention.

---

## 4. Boundary Conditions: The Coffee Pause

Real-world usage includes long idle gaps. We treat these as **Human State Transitions**.

- **Rule**: If idle time $\Delta t > 30\text{min}$, force reset to `Stable` and decay intensity.
- **Logic**: A break allows for human "cool-down" and reframing. However, we preserve some memory of the prior state to avoid missing a resumed loop.
- **Intensity Decay**: $I_t := \gamma I_t$ where $\gamma = 0.3$.
- **Reacclimation**: Disable `Intervene` for the first $N=2$ turns after a break to allow the new trajectory to establish itself.

---

## 5. Intervention Calculus

The system applies a **Control Input** ($u_t$) to bend the trajectory.

| Affect | Trajectory: Watch | Trajectory: Intervene |
| :--- | :--- | :--- |
| **Neutral** | No action (or internal log). | **Soft Nudge**: "A lot of repeat activity here. Need a re-alignment?" |
| **Frustrated** | **Acknowledge**: "I see this is becoming difficult..." | **Hard Reset**: "Stop. Let's restate the goal and try a new approach." |

---

## 6. Evaluation Framework (The Report Card)

We measure the model's performance using **Precision within Horizon (H)**.

1.  **Precision@H**: Fraction of interventions followed by an explicit user dispute within $H$ turns.
2.  **Coverage@H**: Fraction of user disputes that were preceded by a warning.
3.  **Lead Time**: How many turns *before* the dispute did we warn? (Target: >0).
4.  **Alarm Rate**: How often do we interrupt? (Target: < 1 per 50 turns).

---

## 7. Next Steps for Formalization

- [ ] **Define Exact Formulas**: Standardize Jaccard vs. Cosine thresholds.
- [ ] **Calibrate θ**: Run the `analyze_trajectory_v2.py` script over multiple datasets to find the "knee" of the Precision/Alarm-Rate curve.
- [x] **State Machine Implementation**: Prototype the `Stable/Watch/Intervene` logic in a standalone simulator.

---

## 8. Summary: The Trajectory Theory of Agentic Collaboration

The core thesis of this model is that **human-agent friction is not a state, but a momentum.**

1. **Instability is Structural**: Friction episodes are preceded by measurable collapses in novelty and increases in repetitive effort (The Loop Basin).
2. **Symptoms Lead Affect**: We can sense the trajectory toward a loop using low-cost structural indicators (symbols, tokens, embeddings) 3-5 turns before the human expresses frustration.
3. **Regulation is Control**: By applying high-precision interventions (Hydration Packets, Context Anchors) at the point of "Watch" or "Intervene", we can bend the trajectory back toward Stable iteration.
4. **Affect is the Damper**: Lagging emotional signals don't define the trajectory, but they dictate the "gain" and "style" of the control input.
5. **Context is Segmented**: Collaboration is piecewise-continuous; time-based boundaries (The Coffee Pause) are critical for resetting the controller to match the human's affective recovery.

By treating the collaboration as a dynamical system, we move from "reacting to anger" to **proactively managing cognitive momentum.**
