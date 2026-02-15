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
WEIGHT_LOGIC_CHURN = 0.20

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
    "wait",
    "stop",
    "hold on",
    "not quite",
    "don't do",
    "never mind",
    "re-read",
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


def detect_correction(text, emotion=None):
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0
    for p in CORRECTION_PATTERNS:
        if p in lower:
            score += 0.5 if p in ["actually", "wait", "hold on"] else 1.0

    final_score = min(1.0, score / 1.5)
    return final_score


def detect_summary_intent(text):
    if not text:
        return 0.0
    lower = text.lower()
    return 1.0 if any(cue in lower for cue in SUMMARY_CUES) else 0.0


def word_overlap_churn(s1, s2):
    if not s1 or not s2:
        return 0.0
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))
    if not w1 or not w2:
        return 0.0
    intersection = w1 & w2
    return 1.0 - (len(intersection) / max(len(w1), len(w2)))


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
            for k in [
                "rep",
                "nov",
                "sem",
                "eff",
                "corr",
                "path",
                "stall",
                "stat",
                "churn",
            ]
        }

    def update(self, turn):
        ts_ms = turn.get("ts_ms", 0)
        if self.last_ts_ms > 0 and (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS:
            self.state = "stable"
            self.intensity *= COFFEE_PAUSE_DECAY
        self.last_ts_ms = ts_ms

        user_text = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        combined_text = user_text + " " + assistant_text

        current_symbols = extract_symbols(combined_text)
        user_paths = extract_paths(user_text)
        assistant_symbols = extract_symbols(assistant_text)

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
        s_churn = (
            word_overlap_churn(assistant_text, self.history[-1]["assistant"])
            if self.history
            else 0.0
        )

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
        self.smoothed["churn"] = (
            EMA_ALPHA * s_churn + (1 - EMA_ALPHA) * self.smoothed["churn"]
        )

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
            loop_i += WEIGHT_LOGIC_CHURN * self.smoothed["churn"]

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
            {
                "symbols": current_symbols,
                "text": combined_text,
                "user": user_text,
                "assistant": assistant_text,
            }
        )

        cause = "loop"
        if spec_i > loop_i and spec_i > drift_i:
            cause = "spec"
        elif drift_i > loop_i:
            cause = "drift"

        return triggered, self.intensity, cause


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
    H_RANGE = [1, 3, 5, 8, 10, 15]
    controller = TrajectoryController(mode=mode)
    results = []
    total_disputes = 0

    for i, turn in enumerate(data):
        triggered, intensity, cause = controller.update(turn)
        is_dispute = any(kw in turn.get("user", "").lower() for kw in DISPUTE_KEYWORDS)
        if is_dispute:
            total_disputes += 1
        results.append(
            {"triggered": triggered, "is_dispute": is_dispute, "cause": cause}
        )

    basin_stats = {b: {"hits": 0, "triggers": 0} for b in ["loop", "spec", "drift"]}
    for i, r in enumerate(results):
        if r["triggered"]:
            basin_stats[r["cause"]]["triggers"] += 1
            for j in range(i + 1, min(i + 1 + 5, len(results))):
                if results[j]["is_dispute"]:
                    basin_stats[r["cause"]]["hits"] += 1
                    break

    h_curve = []
    for h in H_RANGE:
        covered_disputes = set()
        total_triggers = sum(1 for r in results if r["triggered"])
        total_hits = 0
        for i, r in enumerate(results):
            if r["triggered"]:
                total_triggers += 1
                for j in range(i + 1, min(i + 1 + h, len(results))):
                    if results[j]["is_dispute"]:
                        total_hits += 1
                        covered_disputes.add(j)
                        break
        coverage = len(covered_disputes) / total_disputes if total_disputes > 0 else 0
        precision = total_hits / total_triggers if total_triggers > 0 else 0
        h_curve.append((h, precision, coverage))

    return h_curve, basin_stats, total_disputes


if __name__ == "__main__":
    with open("internal/research/replay-optimization/marathon_set_raw.json", "r") as f:
        marathon = json.load(f)
    print("=== Robustness Deep Dive (Marathon Set) ===")
    for mode in ["committed", "worktree"]:
        h_curve, basin_stats, total_d = run_eval(marathon, mode, mode)
        print(f"\n[{mode.upper()}]")
        print(f"Total Disputes: {total_d}")
        print("\nBasin Performance (H=5):")
        for b, stats in basin_stats.items():
            if stats["triggers"] > 0:
                p = stats["hits"] / stats["triggers"]
                print(f"  {b:6}: Precision {p:>5.1%} | Triggers {stats['triggers']:>2}")
            else:
                print(f"  {b:6}: No triggers")
        print("\nCoverage@H Curve:")
        print("  H  | Prec  | Cov")
        print("-----|-------|-----")
        for h, p, c in h_curve:
            print(f"  {h:2} | {p:>5.1%} | {c:>5.1%}")
