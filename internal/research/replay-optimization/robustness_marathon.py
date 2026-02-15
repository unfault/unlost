import json
import re
from collections import deque

# --- Model Constants (Mirrors src/governor.rs Worktree) ---
WEIGHT_EFFORT = 0.34
WEIGHT_REPETITION = 0.24
WEIGHT_NOVELTY = 0.24
WEIGHT_SEMANTIC = 0.18
WEIGHT_ALIGNMENT_DEBT = 0.45
WEIGHT_PATH_HALLUCINATION = 0.60
WEIGHT_GROUNDING_STALL = 0.30
WEIGHT_INSTRUCTION_STATICNESS = 0.25

THRESHOLD_WATCH = 0.5
THRESHOLD_INTERVENE = 0.8
THRESHOLD_STABLE_OFF = 0.4

EMA_ALPHA = 0.3
COFFEE_PAUSE_MS = 30 * 60 * 1000
PERSISTENCE_WINDOW = 3
PERSISTENCE_THRESHOLD = 0.75
COFFEE_PAUSE_DECAY = 0.3

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

SUMMARY_CUES = [
    "summary",
    "recap",
    "summarize",
    "consolidate",
    "overview",
    "in short",
    "to conclude",
]


def extract_paths(text):
    if not text:
        return set()
    return set(re.findall(r"[\w\-\./]+\.[a-z]{2,5}", text))


def extract_symbols(text):
    if not text:
        return set()
    files = set(re.findall(r"[\w\-\./]+\.[a-z]{2,5}", text))
    idents = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{6,}", text))
    return files | idents


def detect_correction(text):
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0
    for p in CORRECTION_PATTERNS:
        if p in lower:
            score += 0.5 if p == "actually" else 1.0
    return min(1.0, score / 1.5)


def detect_summary_intent(text):
    if not text:
        return 0.0
    lower = text.lower()
    return 1.0 if any(cue in lower for cue in SUMMARY_CUES) else 0.0


class TrajectoryController:
    def __init__(self, mode="worktree"):
        self.mode = mode
        self.state = "stable"
        self.intensity = 0.0
        self.last_ts_ms = 0
        self.history = []
        self.intensity_history = deque(maxlen=PERSISTENCE_WINDOW)
        self.stall_streak = 0
        self.static_streak = 0
        self.smoothed = {
            k: 0.0
            for k in ["rep", "nov", "sem", "eff", "corr", "path", "stall", "stat"]
        }

    def update(self, turn):
        ts_ms = turn.get("ts_ms", 0)
        # Soft Coffee Pause
        if self.last_ts_ms > 0 and (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS:
            self.state = "stable"
            self.intensity *= COFFEE_PAUSE_DECAY
        self.last_ts_ms = ts_ms

        user_text = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        combined_text = user_text + " " + assistant_text

        current_symbols = extract_symbols(combined_text)
        user_paths = extract_paths(user_text)
        # Proxy assistant symbols (since we don't have tool calls in this JSON)
        assistant_symbols = extract_symbols(assistant_text)

        # 1. Calculate Raw Symptoms
        history_symbols = set()
        for h in self.history[-8:]:
            history_symbols.update(h["symbols"])

        s_rep = (
            len(current_symbols & history_symbols) / len(current_symbols)
            if current_symbols
            else 0.0
        )
        s_nov = 1.0 - (1.0 - s_rep)
        s_corr = detect_correction(user_text)
        s_summary = detect_summary_intent(assistant_text)
        s_path = 0.0  # Not computable from raw JSON without workspace

        # New Sensors (Worktree Only)
        s_stall = 0.0
        s_stat = 0.0
        if self.mode == "worktree":
            has_stall = user_paths and not (user_paths & assistant_symbols)
            if has_stall:
                self.stall_streak += 1
            else:
                self.stall_streak = 0
            s_stall = 1.0 if self.stall_streak >= 2 else 0.0

            is_static = (
                self.history
                and len(user_text) > 50
                and user_text.strip() == self.history[-1]["user"].strip()
            )
            if is_static:
                self.static_streak += 1
            else:
                self.static_streak = 0
            s_stat = 1.0 if self.static_streak >= 2 else 0.0

        current_eff = len(combined_text)
        if self.history:
            prev_effs = [len(h["text"]) for h in self.history[-8:]]
            avg_eff = sum(prev_effs) / len(prev_effs)
            s_eff = min(2.0, current_eff / avg_eff) / 2.0 if avg_eff > 0 else 0.5
        else:
            s_eff = 0.0

        # 2. EMA
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
        self.smoothed["stall"] = (
            EMA_ALPHA * s_stall + (1 - EMA_ALPHA) * self.smoothed["stall"]
        )
        self.smoothed["stat"] = (
            EMA_ALPHA * s_stat + (1 - EMA_ALPHA) * self.smoothed["stat"]
        )

        # 3. Intensity
        loop_i = (
            WEIGHT_REPETITION * self.smoothed["rep"]
            + WEIGHT_NOVELTY * self.smoothed["nov"]
            + WEIGHT_EFFORT * self.smoothed["eff"]
        )
        spec_i = WEIGHT_ALIGNMENT_DEBT * self.smoothed["corr"]
        drift_i = 0.0

        if self.mode == "worktree":
            spec_i += WEIGHT_INSTRUCTION_STATICNESS * self.smoothed["stat"]
            drift_i += WEIGHT_GROUNDING_STALL * self.smoothed["stall"]

        raw_i = loop_i + spec_i + drift_i
        if self.mode == "worktree" and s_summary > 0.5:
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

        self.history.append(
            {"symbols": current_symbols, "text": combined_text, "user": user_text}
        )
        return triggered, self.intensity


def run_eval(data, mode, label):
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
    H = 5
    controller = TrajectoryController(mode=mode)
    results = []
    total_disputes = 0
    covered_disputes = set()

    for i, turn in enumerate(data):
        triggered, intensity = controller.update(turn)
        is_dispute = any(kw in turn.get("user", "").lower() for kw in DISPUTE_KEYWORDS)
        if is_dispute:
            total_disputes += 1
        results.append({"triggered": triggered, "is_dispute": is_dispute})

    total_triggers = sum(1 for r in results if r["triggered"])
    total_hits = 0
    for i in range(len(results)):
        if results[i]["triggered"]:
            for j in range(i + 1, min(i + 1 + H, len(results))):
                if results[j]["is_dispute"]:
                    total_hits += 1
                    covered_disputes.add(j)
                    break

    precision = total_hits / total_triggers if total_triggers > 0 else 0
    coverage = len(covered_disputes) / total_disputes if total_disputes > 0 else 0
    return precision, coverage, total_triggers, total_disputes


if __name__ == "__main__":
    with open("internal/research/replay-optimization/marathon_set_raw.json", "r") as f:
        marathon = json.load(f)
    with open("internal/research/replay-optimization/sprint_set_raw.json", "r") as f:
        sprint = json.load(f)

    for set_name, data in [
        ("Marathon (390 turns)", marathon),
        ("Sprint (50 turns)", sprint),
    ]:
        print(f"\n--- {set_name} ---")
        for mode in ["committed", "worktree"]:
            p, c, t, d = run_eval(data, mode, mode)
            print(
                f"[{mode:10}] Precision@5: {p:>5.1%} | Coverage@5: {c:>5.1%} | Triggers: {t:>2}"
            )
