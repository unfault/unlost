import json
import re
import math
from collections import Counter
import sys


def extract_symbols(text):
    files = set(re.findall(r"[\w\-\./]+\.[a-z]{2,5}", text))
    idents = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{6,}", text))
    return files | idents


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def get_cosine_sim(c1, c2):
    all_keys = set(c1.keys()) | set(c2.keys())
    if not all_keys:
        return 1.0
    v1 = [c1.get(k, 0) for k in all_keys]
    v2 = [c2.get(k, 0) for k in all_keys]
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot(v1, v2) / (n1 * n2)


def mean(data):
    if not data:
        return 0.0
    return sum(data) / len(data)


def std(data):
    if not data:
        return 0.0
    mu = mean(data)
    return math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))


def main():
    filename = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "internal/research/replay-optimization/marathon_set_raw.json"
    )
    with open(filename, "r") as f:
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
    ]

    turns = []
    all_word_counts = []
    for i, turn in enumerate(data):
        user_text = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        combined = user_text + " " + assistant_text
        current_symbols = extract_symbols(combined)
        current_symbols.update(turn.get("paths", []))
        word_counts = Counter(re.findall(r"\w+", combined.lower()))
        all_word_counts.append(word_counts)
        is_dispute = any(kw in user_text.lower() for kw in DISPUTE_KEYWORDS)
        turns.append(
            {
                "index": i,
                "symbols": current_symbols,
                "eff": len(combined),
                "is_dispute": is_dispute,
                "ts_ms": turn.get("ts_ms", 0),
            }
        )

    window = 8
    alpha = 0.3
    COFFEE_PAUSE_MS = 30 * 60 * 1000

    results = []
    ema_rep = 0
    ema_nov = 0
    ema_sem = 0
    ema_eff = 0

    for i in range(len(turns)):
        current = turns[i]
        if i > 0 and (current["ts_ms"] - turns[i - 1]["ts_ms"]) > COFFEE_PAUSE_MS:
            ema_rep = 0
            ema_nov = 0
            ema_sem = 0
            ema_eff = 0

        recent_symbols = set()
        for j in range(max(0, i - window), i):
            recent_symbols.update(turns[j]["symbols"])

        rep = 0
        nov = 1
        if current["symbols"]:
            overlap = current["symbols"] & recent_symbols
            rep = len(overlap) / len(current["symbols"])
            nov = (len(current["symbols"]) - len(overlap)) / len(current["symbols"])

        sem = 0
        if i > 0:
            sem_vals = [
                get_cosine_sim(all_word_counts[i], all_word_counts[j])
                for j in range(max(0, i - window), i)
            ]
            sem = max(sem_vals) if sem_vals else 0

        eff_raw = current["eff"]
        eff_baseline = mean([t["eff"] for t in turns[max(0, i - window) : i + 1]])
        eff_score = eff_raw / eff_baseline if eff_baseline > 0 else 1.0

        ema_rep = alpha * rep + (1 - alpha) * ema_rep
        ema_nov = alpha * (1 - nov) + (1 - alpha) * ema_nov
        ema_sem = alpha * sem + (1 - alpha) * ema_sem
        ema_eff = alpha * min(2.0, eff_score) / 2.0 + (1 - alpha) * ema_eff

        results.append(
            {
                "index": i,
                "d_rep": ema_rep,
                "d_nov_collapse": ema_nov,
                "d_sem": ema_sem,
                "d_eff": ema_eff,
                "is_dispute": current["is_dispute"],
            }
        )

    channels = ["d_rep", "d_nov_collapse", "d_sem", "d_eff"]
    # Derived weights from previous run
    weights = {"d_rep": 0.239, "d_nov_collapse": 0.239, "d_sem": 0.183, "d_eff": 0.339}

    print(
        f"{'Threshold':<10} | {'Triggers':<10} | {'Precision@5':<12} | {'Coverage@5':<12} | {'Lead Time':<10}"
    )
    print("-" * 75)
    H = 5
    COFFEE_PAUSE_DECAY = 0.3
    PERSISTENCE_WINDOW = 3
    PERSISTENCE_THRESHOLD = 0.75

    dispute_indices = [i for i, r in enumerate(results) if r["is_dispute"]]

    for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
        intensity = []
        for i in range(len(turns)):
            val = sum(results[i][ch] * weights[ch] for ch in channels)
            # Coffee Pause soft decay would be applied here in real controller,
            # but results[i] already has resets applied above.
            intensity.append(val)

        triggers = []
        last_trigger = -100
        for i, val in enumerate(intensity):
            is_persistent = False
            if i >= PERSISTENCE_WINDOW - 1:
                window_vals = intensity[i - PERSISTENCE_WINDOW + 1 : i + 1]
                if all(v > PERSISTENCE_THRESHOLD for v in window_vals):
                    is_persistent = True

            if val >= thresh or is_persistent:
                if i - last_trigger >= 5:
                    if (
                        i > 0
                        and (turns[i]["ts_ms"] - turns[i - 1]["ts_ms"])
                        > COFFEE_PAUSE_MS
                    ):
                        pass
                    else:
                        triggers.append(i)
                        last_trigger = i

        hits = 0
        lead_times = []
        covered_disputes = set()

        for t in triggers:
            has_hit = False
            for j in range(t + 1, min(t + 1 + H, len(results))):
                if results[j]["is_dispute"]:
                    has_hit = True
                    lead_times.append(j - t)
                    covered_disputes.add(j)
                    break
            if has_hit:
                hits += 1

        prec = hits / len(triggers) if triggers else 0
        coverage = (
            len(covered_disputes) / len(dispute_indices) if dispute_indices else 0
        )
        avg_lead = mean(lead_times) if lead_times else 0
        print(
            f"{thresh:<10.2f} | {len(triggers):<10} | {prec:<12.1%} | {coverage:<12.1%} | {avg_lead:<10.1f}"
        )


if __name__ == "__main__":
    main()
