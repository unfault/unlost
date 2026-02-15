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
    "don't like",
    "don't want",
]


def detect_correction(text):
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0
    for p in CORRECTION_PATTERNS:
        if p in lower:
            score += 0.5 if p == "actually" else 1.0
    return min(1.0, score / 1.5)


class TrajectoryController:
    def __init__(self, use_staticness=False):
        self.state = "stable"
        self.intensity = 0.0
        self.last_ts_ms = 0
        self.history = []
        self.intensity_history = deque(maxlen=PERSISTENCE_WINDOW)
        self.use_staticness = use_staticness
        self.smoothed = {
            "rep": 0.0,
            "nov": 0.0,
            "sem": 0.0,
            "eff": 0.0,
            "corr": 0.0,
            "path": 0.0,
        }

    def update(self, turn):
        ts_ms = turn.get("ts_ms", 0)
        if self.last_ts_ms > 0 and (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS:
            self.state = "stable"
            self.intensity *= 0.3
        self.last_ts_ms = ts_ms

        user_text = turn.get("user", "")
        combined_text = user_text + " " + turn.get("assistant", "")
        current_symbols = set(re.findall(r"[\w\-\./]+\.[a-z]{2,5}", combined_text))

        history_symbols = set()
        for h in self.history[-8:]:
            history_symbols.update(h["symbols"])

        s_rep = (
            len(current_symbols & history_symbols) / len(current_symbols)
            if current_symbols
            else 0.0
        )

        # New: User Staticness (Repeating same message)
        s_static = 0.0
        if self.history:
            if self.history[-1]["user"].strip() == user_text.strip():
                s_static = 1.0

        s_nov = 1.0 - (1.0 - s_rep)
        s_corr = detect_correction(user_text)
        s_path = 0.0

        current_eff = len(combined_text)
        if self.history:
            prev_effs = [len(h["text"]) for h in self.history[-8:]]
            avg_eff = sum(prev_effs) / len(prev_effs)
            s_eff = min(2.0, current_eff / avg_eff) / 2.0 if avg_eff > 0 else 0.5
        else:
            s_eff = 0.0

        self.smoothed["rep"] = (
            EMA_ALPHA * (max(s_rep, s_static) if self.use_staticness else s_rep)
            + (1 - EMA_ALPHA) * self.smoothed["rep"]
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

        loop_i = (
            WEIGHT_REPETITION * self.smoothed["rep"]
            + WEIGHT_NOVELTY * self.smoothed["nov"]
            + WEIGHT_EFFORT * self.smoothed["eff"]
        )
        spec_i = WEIGHT_ALIGNMENT_DEBT * self.smoothed["corr"]

        raw_i = loop_i + spec_i

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

        self.history.append(
            {"symbols": current_symbols, "text": combined_text, "user": user_text}
        )
        return triggered


def main():
    path = "internal/research/replay-optimization/marathon_set_raw.json"
    with open(path, "r") as f:
        data = json.load(f)

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
        "don't like",
    ]
    H = 5

    for mode_name, staticness in [("Current", False), ("With Staticness Sensor", True)]:
        total_triggers = 0
        total_hits = 0
        total_disputes = 0
        covered_disputes = set()

        controller = TrajectoryController(use_staticness=staticness)
        turn_results = []

        for i, turn in enumerate(data):
            triggered = controller.update(turn)
            user_text = turn.get("user", "").lower()
            is_dispute = any(kw in user_text for kw in DISPUTE_KEYWORDS)
            if is_dispute:
                total_disputes += 1
            turn_results.append({"triggered": triggered, "is_dispute": is_dispute})

        for i in range(len(turn_results)):
            if turn_results[i]["triggered"]:
                total_triggers += 1
                hit = False
                for j in range(i + 1, min(i + 1 + H, len(turn_results))):
                    if turn_results[j]["is_dispute"]:
                        hit = True
                        covered_disputes.add(j)
                        break
                if hit:
                    total_hits += 1

        precision = total_hits / total_triggers if total_triggers > 0 else 0
        coverage = len(covered_disputes) / total_disputes if total_disputes > 0 else 0

        print(f"\n--- Robustness: {mode_name} ---")
        print(f"Total Triggers: {total_triggers}")
        print(f"Precision@5:    {precision:.1%}")
        print(f"Coverage@5:     {coverage:.1%}")


if __name__ == "__main__":
    main()
