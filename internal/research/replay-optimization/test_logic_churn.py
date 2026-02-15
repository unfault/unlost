import json
import re
from collections import deque


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_churn(s1, s2):
    if not s1 or not s2:
        return 0.0
    dist = levenshtein(s1, s2)
    return dist / max(len(s1), len(s2))


def run_churn_analysis(data):
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

    history = deque(maxlen=5)
    results = []

    for i, turn in enumerate(data):
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")

        churn = 0.0
        if history:
            prev_assistant = history[-1]["assistant"]
            churn = calculate_churn(assistant, prev_assistant)

        is_dispute = any(kw in user.lower() for kw in DISPUTE_KEYWORDS)
        results.append(
            {"idx": i, "churn": churn, "is_dispute": is_dispute, "user": user}
        )
        history.append({"user": user, "assistant": assistant})

    # Find high churn bursts that precede disputes
    print("--- High Churn Preceding Disputes ---")
    for i in range(len(results)):
        if results[i]["churn"] > 0.7:  # High divergence in plan
            # Check if followed by dispute
            hit = False
            for j in range(i + 1, min(i + 1 + H, len(results))):
                if results[j]["is_dispute"]:
                    print(
                        f"Turn {i}: Churn {results[i]['churn']:.2f} -> Dispute at {j}: '{results[j]['user'][:50]}...'"
                    )
                    hit = True
                    break


if __name__ == "__main__":
    with open("internal/research/replay-optimization/marathon_set_raw.json", "r") as f:
        marathon = json.load(f)
    run_churn_analysis(marathon)
