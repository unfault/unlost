import json
import re
from collections import Counter, deque
import sys
import os

# Constants from src/governor.rs
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
COFFEE_PAUSE_MS = 30 * 60 * 1000
PERSISTENCE_WINDOW = 3
PERSISTENCE_THRESHOLD = 0.75

CORRECTION_PATTERNS = [
    "no",
    "not that",
    "i meant",
    "actually",
    "that's not what i asked",
    "you misunderstood",
    "wrong",
    "incorrect",
]


def detect_correction(text):
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0
    for p in CORRECTION_PATTERNS:
        if p in lower:
            if p == "actually":
                score += 0.5
            else:
                score += 1.0
    return min(1.0, score / 1.5)


class TrajectoryController:
    def __init__(self):
        self.state = "stable"
        self.intensity = 0.0
        self.last_ts_ms = 0
        self.history = []
        self.intensity_history = deque(maxlen=PERSISTENCE_WINDOW)
        self.smoothed = {
            "rep": 0.0,
            "nov": 0.0,
            "sem": 0.0,
            "eff": 0.0,
            "corr": 0.0,
            "path": 0.0,
        }

    def update(self, turn):
        ts_ms = turn["ts_ms"]
        capsule = turn["capsule"]

        if self.last_ts_ms > 0 and (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS:
            self.state = "stable"
            self.intensity *= 0.3
        self.last_ts_ms = ts_ms

        current_symbols = capsule.get("symbols", [])
        history_symbols = []
        for h in self.history[-8:]:
            history_symbols.extend(h.get("symbols", []))

        s_rep = 0.0
        if current_symbols and history_symbols:
            overlap = set(current_symbols) & set(history_symbols)
            s_rep = len(overlap) / len(current_symbols)

        s_nov = 1.0 - (1.0 - s_rep)
        s_corr = detect_correction(capsule.get("intent", ""))
        s_path = 1.0 if capsule.get("failure_mode") == "drift" else 0.0

        current_eff = len(capsule.get("intent", "")) + len(capsule.get("decision", ""))
        if self.history:
            prev_effs = [
                len(h.get("intent", "")) + len(h.get("decision", ""))
                for h in self.history[-8:]
            ]
            avg_eff = sum(prev_effs) / len(prev_effs)
            s_eff = min(2.0, current_eff / avg_eff) / 2.0 if avg_eff > 0 else 0.5
        else:
            s_eff = 0.0

        self.smoothed["rep"] = (
            EMA_ALPHA * s_rep + (1 - EMA_ALPHA) * self.smoothed["rep"]
        )
        self.smoothed["nov"] = (
            EMA_ALPHA * s_nov + (1 - EMA_ALPHA) * self.smoothed["nov"]
        )
        self.smoothed["corr"] = (
            EMA_ALPHA * s_corr + (1 - EMA_ALPHA) * self.smoothed["corr"]
        )
        self.smoothed["eff"] = (
            EMA_ALPHA * s_eff + (1 - EMA_ALPHA) * self.smoothed["eff"]
        )
        self.smoothed["path"] = (
            EMA_ALPHA * s_path + (1 - EMA_ALPHA) * self.smoothed["path"]
        )

        loop_i = (
            WEIGHT_REPETITION * self.smoothed["rep"]
            + WEIGHT_NOVELTY * self.smoothed["nov"]
            + WEIGHT_EFFORT * self.smoothed["eff"]
        )
        spec_i = WEIGHT_ALIGNMENT_DEBT * self.smoothed["corr"]
        drift_i = WEIGHT_PATH_HALLUCINATION * self.smoothed["path"]

        raw_i = loop_i + spec_i + drift_i
        old_i = self.intensity
        self.intensity = raw_i
        slope = self.intensity - old_i

        self.intensity_history.append(self.intensity)
        is_persistent = len(self.intensity_history) == PERSISTENCE_WINDOW and all(
            v > PERSISTENCE_THRESHOLD for v in self.intensity_history
        )

        prev_state = self.state
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

        self.history.append(capsule)
        return triggered, self.intensity, (loop_i, spec_i, drift_i)


def main():
    workspace_id = "wks_83ae697ac0204117"
    path = f"/home/sylvain/.local/share/unlost/workspaces/{workspace_id}/capsules.jsonl"

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    data.sort(key=lambda x: x["ts_ms"])

    DISPUTE_KEYWORDS = [
        "still",
        "wrong",
        "no",
        "broken",
        "didn't",
        "fail",
        "not working",
        "frustrat",
        "confus",
    ]

    controller = TrajectoryController()

    print(f"{'Turn':<5} | {'Inten':<6} | {'Target (Next 5 turns)'}")
    print("-" * 60)

    for i, turn in enumerate(data):
        triggered, intensity, components = controller.update(turn)

        if triggered:
            # Show the next 5 turns
            targets = []
            for j in range(i + 1, min(i + 6, len(data))):
                user_text = data[j].get("capsule", {}).get("intent", "").lower()
                is_dispute = any(kw in user_text for kw in DISPUTE_KEYWORDS)
                if is_dispute:
                    targets.append(f"T+{j - i}: '{user_text[:30]}...'")

            print(f"{i + 1:<5} | {intensity:<6.2f} | {' | '.join(targets)}")


if __name__ == "__main__":
    main()
