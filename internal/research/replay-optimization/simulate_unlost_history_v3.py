import json
import re
import math
from collections import Counter, deque
import sys
import os

# --- Model Constants (Mirrors src/governor.rs) ---
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


def detect_correction(text):
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0
    for p in CORRECTION_PATTERNS:
        if p in lower:
            score += 0.5 if p in ["actually", "wait", "hold on"] else 1.0
    return min(1.0, score / 1.5)


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
    def __init__(self, use_intent_damping=True):
        self.state = "stable"
        self.intensity = 0.0
        self.last_ts_ms = 0
        self.history = []
        self.intensity_history = deque(maxlen=PERSISTENCE_WINDOW)
        self.use_intent_damping = use_intent_damping
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
        ts_ms = turn["ts_ms"]
        capsule = turn["capsule"]

        if self.last_ts_ms > 0 and (ts_ms - self.last_ts_ms) > COFFEE_PAUSE_MS:
            self.state = "stable"
            self.intensity *= COFFEE_PAUSE_DECAY
        self.last_ts_ms = ts_ms

        current_symbols = set(capsule.get("symbols", []))
        history_symbols = set()
        for h in self.history[-8:]:
            history_symbols.update(h.get("symbols", []))

        s_rep = (
            len(current_symbols & history_symbols) / len(current_symbols)
            if current_symbols
            else 0.0
        )
        s_nov = 1.0 - (1.0 - s_rep)
        s_corr = detect_correction(capsule.get("intent", ""))
        s_summary = detect_summary_intent(capsule.get("decision", ""))
        s_path = 1.0 if capsule.get("failure_mode") == "drift" else 0.0
        s_churn = (
            word_overlap_churn(
                capsule.get("decision", ""), self.history[-1].get("decision", "")
            )
            if self.history
            else 0.0
        )

        user_paths = extract_paths(capsule.get("intent", ""))
        assistant_symbols = current_symbols

        has_stall = user_paths and not (user_paths & assistant_symbols)
        if has_stall:
            self.stall_streak += 1
        else:
            self.stall_streak = 0
        s_stall = 1.0 if self.stall_streak >= 2 else 0.0

        is_static = (
            self.history
            and len(capsule.get("intent", "")) > 50
            and capsule.get("intent", "").strip()
            == self.history[-1].get("intent", "").strip()
        )
        if is_static:
            self.static_streak += 1
        else:
            self.static_streak = 0
        s_stat = 1.0 if self.static_streak >= 2 else 0.0

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
            + WEIGHT_LOGIC_CHURN * self.smoothed["churn"]
        )
        spec_i = (
            WEIGHT_ALIGNMENT_DEBT * self.smoothed["corr"]
            + WEIGHT_INSTRUCTION_STATICNESS * self.smoothed["stat"]
        )
        drift_i = (
            WEIGHT_PATH_HALLUCINATION * self.smoothed["path"]
            + WEIGHT_GROUNDING_STALL * self.smoothed["stall"]
        )

        raw_i = loop_i + spec_i + drift_i
        if self.use_intent_damping and s_summary > 0.5:
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

        self.history.append(capsule)

        cause = "loop"
        if spec_i > loop_i and spec_i > drift_i:
            cause = "spec"
        elif drift_i > loop_i:
            cause = "drift"

        return triggered, self.intensity, cause


def main():
    workspace_id = "wks_83ae697ac0204117"
    path = f"/home/sylvain/.local/share/unlost/workspaces/{workspace_id}/capsules.jsonl"

    if not os.path.exists(path):
        print(f"Capsules file not found: {path}")
        return

    sessions = {}
    with open(path, "r") as f:
        for line in f:
            turn = json.loads(line)
            sid = turn.get("agent_session_id", "default")
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(turn)

    for sid in sessions:
        sessions[sid].sort(key=lambda x: x["ts_ms"])

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

    total_triggers = 0
    total_hits = 0
    total_disputes = 0
    covered_disputes = set()
    trigger_counts = Counter()
    cause_counts = Counter()

    total_spacing_tokens = 0
    total_spacing_segments = 0

    buckets = {}

    for sid, turns in sessions.items():
        if len(turns) < 3:
            continue

        controller = TrajectoryController(use_intent_damping=True)
        turn_results = []
        tokens_since_last_warning = 0
        had_warning = False

        for i, turn in enumerate(turns):
            triggered, intensity, cause = controller.update(turn)
            user_text = turn.get("capsule", {}).get("intent", "").lower()
            is_dispute = any(kw in user_text for kw in DISPUTE_KEYWORDS)

            tokens = turn.get("usage", {}).get("tokens", {}).get("input", 0)
            tokens_total = (
                turn.get("usage", {}).get("tokens", {}).get("total", tokens + 200)
            )  # Proxy total

            b = (tokens // 4000) * 4000
            if b not in buckets:
                buckets[b] = {"turns": 0, "triggers": 0}
            buckets[b]["turns"] += 1

            tokens_since_last_warning += tokens_total

            if is_dispute:
                total_disputes += 1
            if triggered:
                trigger_counts[sid] += 1
                cause_counts[cause] += 1
                buckets[b]["triggers"] += 1

                if had_warning:
                    total_spacing_tokens += tokens_since_last_warning
                    total_spacing_segments += 1
                had_warning = True
                tokens_since_last_warning = 0

            turn_results.append(
                {
                    "triggered": triggered,
                    "is_dispute": is_dispute,
                    "sid": sid,
                    "idx": i,
                    "intensity": intensity,
                    "cause": cause,
                }
            )

        for i in range(len(turn_results)):
            if turn_results[i]["triggered"]:
                total_triggers += 1
                hit = False
                for j in range(i + 1, min(i + 1 + H, len(turn_results))):
                    if turn_results[j]["is_dispute"]:
                        hit = True
                        covered_disputes.add((sid, j))
                        break
                if hit:
                    total_hits += 1

    precision = total_hits / total_triggers if total_triggers > 0 else 0
    coverage = len(covered_disputes) / total_disputes if total_disputes > 0 else 0

    print(
        f"\n--- Field Simulation Report ({len(sessions)} sessions, {sum(len(s) for s in sessions.values())} turns) ---"
    )
    print(f"Total Triggers:   {total_triggers}")
    print(f"Total Disputes:   {total_disputes}")
    print(f"Precision@5:      {precision:.1%}")
    print(f"Coverage@5:       {coverage:.1%}")

    if total_spacing_segments > 0:
        print(
            f"Avg Interval:     {total_spacing_tokens / total_spacing_segments:.0f} tokens between warnings"
        )

    print("\nTrigger Causes:")
    for cause, count in cause_counts.items():
        print(f"  {cause:8}: {count:>3} ({count / total_triggers:.1%})")

    print("\n=== Friction vs Context Size (Input Tokens) ===")
    print("  Bucket      | Turns | Warnings | Rate (Warnings/100 Turns)")
    print("--------------|-------|----------|---------------------------")
    for b in sorted(buckets.keys()):
        stats = buckets[b]
        rate = (stats["triggers"] / stats["turns"] * 100) if stats["turns"] > 0 else 0
        print(
            f"  {b:>5} - {b + 4000:>5} | {stats['turns']:>5} | {stats['triggers']:>8} | {rate:>5.1f}%"
        )


if __name__ == "__main__":
    main()
