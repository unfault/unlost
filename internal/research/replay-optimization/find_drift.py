import json
import re
from collections import Counter

# Standard correction patterns
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

    print(f"Analyzing {len(data)} turns for Drift...")

    drift_moments = []

    for i in range(1, len(data)):
        prev = data[i - 1]
        curr = data[i]

        user_text = curr.get("user", "").lower()
        assistant_text = prev.get("assistant", "").lower()

        # 1. Look for user corrections
        has_correction = any(kw in user_text for kw in CORRECTION_PATTERNS)

        # 2. Look for "hallucination" markers (referencing things not in paths or previous context)
        # For simplicity, we'll look for turns where the user says "that doesn't exist" or similar.
        hallucination_signals = [
            "doesn't exist",
            "can't find",
            "not there",
            "wrong file",
            "missing",
        ]
        has_hallucination_complaint = any(
            kw in user_text for kw in hallucination_signals
        )

        if has_correction and has_hallucination_complaint:
            drift_moments.append(
                {
                    "index": i,
                    "user": curr.get("user"),
                    "assistant_prev": prev.get("assistant"),
                    "type": "Hallucination/Drift",
                }
            )
        elif has_correction and i > 2:
            # Look for 3-turn patterns: A proposes X, U says no, A proposes Y, U says no.
            pass

    print(f"Found {len(drift_moments)} explicit drift/hallucination moments.")
    for m in drift_moments:
        print(f"\n--- Turn {m['index']} ---")
        print(f"Assistant said: {m['assistant_prev'][:200]}...")
        print(f"User replied: {m['user']}")


if __name__ == "__main__":
    main()
