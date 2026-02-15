import json
import re

CORRECTION_PATTERNS = [
    "no",
    "not that",
    "i meant",
    "actually",
    "wrong",
    "incorrect",
    "misunderstood",
]


def main():
    with open("internal/research/replay-optimization/marathon_set_raw.json", "r") as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} turns for real Drift...")

    drift_moments = []

    for i in range(1, len(data)):
        curr = data[i]
        prev = data[i - 1]

        user_text = curr.get("user", "").lower()
        assistant_text = prev.get("assistant", "").lower()

        # Look for explicit corrections about files/paths
        drift_signals = [
            "wrong file",
            "incorrect file",
            "wrong path",
            "doesn't exist",
            "that's not in",
            "can't find",
        ]

        has_correction = any(kw in user_text for kw in CORRECTION_PATTERNS)
        has_drift_signal = any(kw in user_text for kw in drift_signals)

        if has_correction and has_drift_signal:
            drift_moments.append(
                {
                    "index": i,
                    "user": curr.get("user"),
                    "assistant_prev": prev.get("assistant"),
                }
            )

    print(f"Found {len(drift_moments)} explicit file/path drift moments.")
    for m in drift_moments:
        print(f"\n--- Turn {m['index']} ---")
        print(f"Assistant said: {m['assistant_prev'][:300]}...")
        print(f"User replied: {m['user']}")


if __name__ == "__main__":
    main()
