# Intervention Taxonomy: Bending the Trajectory

This document defines the **Control Inputs** ($u_t$) used to bend the collaboration trajectory when the controller enters the `Watch` or `Intervene` states.

## 1. Dimensionality of Intervention

An intervention is not just a text message. It has three dimensions:
1. **Signal** (What we tell the Agent/User)
2. **Substance** (What context we provide to break the loop)
3. **Structure** (What constraints we impose on the next turn)

---

## 2. Taxonomy by State and Affect

### 2.1 State: Watch (Escalating Symptoms)
*Goal: Passive alignment. Low intrusiveness.*

| Affect State | Intervention | Substance | Structure |
| :--- | :--- | :--- | :--- |
| **Neutral** | **Shadow Note** | List of touched symbols in last 5 turns. | None (Hidden system note). |
| **Confusion / Doubt** | **Compass Note** | Summary of the *rationale* for the last 2 decisions. | "Briefly explain the 'why' behind the current approach." |
| **Frustrated** | **Acknowledge** | "The user's last message showed signs of frustration." | "Pause to clarify the immediate blocker." |
| **Alignment Debt** | **Micro-Contract** | Current goal + next step summary. | "Ask user if this interpretation is correct before proceeding." |

### 2.2 State: Intervene (Loop Detected)
*Goal: Active loop breaking. High head-start.*

| Affect State | Intervention | Substance | Structure |
| :--- | :--- | :--- | :--- |
| **Neutral** | **Hydration Nudge** | **Hydration Packet**: Summary of last 3 failed attempts. | "Propose a *different* approach than the last 3." |
| **Confusion** | **Context Anchor** | **Intent Anchor**: Restatement of original session goal. | "Clarify how current steps lead to the original goal." |
| **Frustrated** | **Hard Reset** | **Attempt Log**: Explicit list of what has already been tried. | "Stop. Do not write code. Ask user to verify current goal." |
| **Anger** | **Emergency Brake** | **State Snapshot**: Current symbols and pending diffs. | "Critical: Stop all execution. Apologize and await instructions." |
| **Alignment Debt** | **Clarify Reset** | List of contradictory user corrections. | "Stop. Restate the objective and ask for confirmation/pivot." |

### 2.3 Reset Operators (Trajectory Bending)
These events force the controller back to `Stable` or dampen the intensity $I_t$.

| Event | Type | Logic | Effect |
| :--- | :--- | :--- | :--- |
| **Joy** | Affective Reset | Confidence > 0.7 + Positive Valence. | $I_t := I_t \cdot 0.5$ (Damping). |
| **Reframe** | Intentional Reset | Keywords: "let's try again", "new plan", "my bad". | $q_t := Stable$, $I_t := 0$. |
| **Coffee Pause** | Temporal Reset | Gap $\Delta t > 30\text{min}$. | $q_t := Stable$, $I_t := 0$. |

---

## 3. Intervention Components

### 3.1 The Hydration Packet
When $s_{rep}$ (Symbol Repetition) is high, the agent likely has "tunnel vision".
- **Payload**: A list of `(Turn Index, Intent, Outcome)` for the last 3 turns involving the same symbols.
- **Logic**: Forces the agent's attention out of the immediate token-window and into the trajectory history.

### 3.2 The Compass Note (for Confusion)
Confusion often stems from a lack of transparency in the agent's reasoning.
- **Payload**: The `rationale` field from recent capsules.
- **Logic**: Provides the "why" to the user, allowing them to correct faulty logic early.

### 3.3 The Emergency Brake (for Anger)
Anger is often led by **Surprise** (e.g., destructive edits without consent) or **De-escalation Failure**.
- **Action**: Immediate cessation of tool use.
- **Substance**: Revert suggestions or provide a "revert command" if possible.
- **Structure**: Force the agent into a submissive, clarify-only mode.

### 3.3 The Constraint Injection
Interventions can modify the **Prompt Structure** to enforce better behavior:
- **"No-Code" Constraint**: Forbids code generation for one turn to force planning.
- **"Evidence" Requirement**: Requires the agent to provide a specific file path or test result before claiming progress.

---

## 4. Policy Rules

1. **Precision First**: Never trigger a `Hard Reset` without a `Frustrated` affect signal unless `I_t > 0.95`.
2. **One-Shot Rule**: Do not repeat the same intervention type within the same `Intervene` episode. If the first nudge fails, escalate or stay silent to avoid "Nagging Friction".
3. **Ghost Post-Break**: The `Resumption Brief` is always injected after a >30m gap, regardless of prior trajectory state.

---

## 5. Evaluation of Interventions

We measure intervention success via **Trajectory Bending**:
- **Recovery Rate**: Frequency of `Intervene → Stable` transitions within 2 turns of $u_t$.
- **Sentiment Delta**: Change in user emotion valence after $u_t$.
- **Effective Progress**: Increase in `s_nov` (Novelty) or `v_t` (Verification) following $u_t$.
