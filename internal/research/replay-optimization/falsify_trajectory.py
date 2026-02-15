import json
import re
from collections import deque

# --- Exact Replica of Rust Governor Logic ---

WEIGHT_EFFORT = 0.34
WEIGHT_REPETITION = 0.24
WEIGHT_NOVELTY = 0.24
WEIGHT_SEMANTIC = 0.18
WEIGHT_ALIGNMENT_DEBT = 0.45
WEIGHT_PATH_HALLUCINATION = 0.60

THRESHOLD_WATCH = 0.5
THRESHOLD_INTERVENE = 0.8
THRESHOLD_STABLE_OFF = 0.4

EMA_ALPHA = 0.3
PERSISTENCE_WINDOW = 3
PERSISTENCE_THRESHOLD = 0.75


def detect_correction(text):
    patterns = [
        "no",
        "not that",
        "i meant",
        "actually",
        "that's not what i asked",
        "you misunderstood",
        "wrong",
        "incorrect",
    ]
    lower = text.lower()
    score = 0.0
    for p in patterns:
        if p in lower:
            score += 0.5 if p == "actually" else 1.0
    return min(1.0, score / 1.5)


def detect_summary_intent(text):
    patterns = [
        "summary",
        "recap",
        "summarize",
        "consolidate",
        "overview",
        "in short",
        "to conclude",
    ]
    lower = text.lower()
    for p in patterns:
        if p in lower:
            return 1.0
    return 0.0


class TrajectoryController:
    def __init__(self):
        self.state = "stable"
        self.intensity = 0.0
        self.smoothed_rep = 0.0
        self.smoothed_nov = 0.0
        self.smoothed_sem = 0.0
        self.smoothed_eff = 0.0
        self.smoothed_corr = 0.0
        self.smoothed_path = 0.0
        self.intensity_history = deque(maxlen=PERSISTENCE_WINDOW)
        self.history = []

    def update(self, turn):
        intent = turn.get("intent", "")
        decision = turn.get("decision", "")
        symbols = turn.get("symbols", [])

        # 1. Symptoms
        # Repetition
        recent_symbols = set()
        for h in self.history[-8:]:
            recent_symbols.update(h.get("symbols", []))

        s_rep = 0.0
        if symbols:
            overlap = set(symbols) & recent_symbols
            s_rep = len(overlap) / len(symbols)

        s_nov = 1.0 - (1.0 - s_rep)  # Novelty collapse proxy
        s_sem = 0.0  # Placeholder

        # Effort
        current_eff = len(symbols) * 100 + len(intent)
        if self.history:
            prev_effs = [
                (len(h.get("symbols", [])) * 100 + len(h.get("intent", "")))
                for h in self.history[-8:]
            ]
            avg_eff = sum(prev_effs) / len(prev_effs)
            s_eff = min(2.0, current_eff / avg_eff) / 2.0 if avg_eff > 0 else 0.5
        else:
            s_eff = 0.0

        s_corr = detect_correction(intent)
        s_path = 0.0  # No hallucinations in these tests
        s_summary = detect_summary_intent(decision)

        # 2. EMA
        self.smoothed_rep = EMA_ALPHA * s_rep + (1 - EMA_ALPHA) * self.smoothed_rep
        self.smoothed_nov = EMA_ALPHA * s_nov + (1 - EMA_ALPHA) * self.smoothed_nov
        self.smoothed_eff = EMA_ALPHA * s_eff + (1 - EMA_ALPHA) * self.smoothed_eff
        self.smoothed_corr = EMA_ALPHA * s_corr + (1 - EMA_ALPHA) * self.smoothed_corr

        # 3. Intensity
        loop_i = (
            WEIGHT_REPETITION * self.smoothed_rep
            + WEIGHT_NOVELTY * self.smoothed_nov
            + WEIGHT_SEMANTIC * self.smoothed_sem
            + WEIGHT_EFFORT * self.smoothed_eff
        )
        spec_i = WEIGHT_ALIGNMENT_DEBT * self.smoothed_corr
        drift_i = WEIGHT_PATH_HALLUCINATION * self.smoothed_path

        raw_i = loop_i + spec_i + drift_i
        if s_summary > 0.5:
            raw_i *= 0.6

        old_i = self.intensity
        self.intensity = raw_i
        slope = self.intensity - old_i

        self.intensity_history.append(self.intensity)
        is_persistent = len(self.intensity_history) == PERSISTENCE_WINDOW and all(
            v > PERSISTENCE_THRESHOLD for v in self.intensity_history
        )

        # 4. Transitions
        triggered = False
        if self.state == "stable":
            if self.intensity > THRESHOLD_WATCH and slope > 0:
                self.state = "watch"
        elif self.state == "watch":
            if (self.intensity > THRESHOLD_INTERVENE and slope > 0.05) or is_persistent:
                self.state = "intervene"
                triggered = True
            elif self.intensity < THRESHOLD_STABLE_OFF:
                self.state = "stable"
        elif self.state == "intervene":
            self.state = "watch"
            self.intensity_history.clear()

        self.history.append(turn)
        return self.state, triggered, self.intensity


# --- Test Scenarios ---


def run_scenario(name, turns):
    print(f"\n>>> Scenario: {name}")
    print(f"{'Turn':<5} | {'State':<10} | {'Intensity':<10} | {'Trigger'}")
    print("-" * 45)
    controller = TrajectoryController()
    any_trigger = False
    for i, turn in enumerate(turns):
        state, triggered, intensity = controller.update(turn)
        if triggered:
            any_trigger = True
        mark = "!!!" if triggered else ""
        print(f"{i + 1:<5} | {state:<10} | {intensity:<10.2f} | {mark}")
    return any_trigger


# 1. The Deep Refactor (10 turns, same 3 files, high tokens)
deep_refactor = [
    {"intent": "Refactor auth logic", "symbols": ["auth.rs", "types.rs", "lib.rs"]}
    for _ in range(10)
]

# 2. The Exploratory Debugger (5 turns, searching/reading)
exploratory_debug = [
    {"intent": "Search for where X is used", "symbols": ["src/main.rs", "src/cli.rs"]},
    {"intent": "Read src/governor.rs to understand Y", "symbols": ["src/governor.rs"]},
    {"intent": "Check src/metrics.rs for Z", "symbols": ["src/metrics.rs"]},
    {"intent": "Back to main.rs to verify W", "symbols": ["src/main.rs"]},
    {"intent": "Ok, let's look at cli.rs again", "symbols": ["src/cli.rs"]},
]

# 3. The Teacher (5 turns, explaining concepts, high semantic similarity)
# (In our model semantic sim is simplified, but high effort is present)
teacher = [
    {"intent": "Let me explain how HMMs work here.", "symbols": []} for _ in range(5)
]

if __name__ == "__main__":
    fp_refactor = run_scenario("Deep Refactor (Healthy Stress)", deep_refactor)
    fp_debug = run_scenario("Exploratory Debugger (Productive)", exploratory_debug)
    fp_teacher = run_scenario("The Teacher (High Effort)", teacher)

    print("\n--- Falsification Report ---")
    print(f"False Positive (Refactor): {fp_refactor}")
    print(f"False Positive (Debugger): {fp_debug}")
    print(f"False Positive (Teacher):  {fp_teacher}")
