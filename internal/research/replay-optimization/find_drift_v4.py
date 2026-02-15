import json
import re


def main():
    with open("internal/research/replay-optimization/marathon_set_raw.json", "r") as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} turns for negations ('no', 'not')...")

    moments = []

    for i, turn in enumerate(data):
        user_text = turn.get("user", "").lower()
        # Look for "no" as a word, not part of "nothing", etc.
        if re.search(r"\bno\b", user_text) or re.search(r"\bnot that\b", user_text):
            moments.append(
                {
                    "index": i,
                    "user": turn.get("user"),
                    "assistant_prev": data[i - 1].get("assistant") if i > 0 else "",
                }
            )

    print(f"Found {len(moments)} negation moments.")
    for m in moments[:10]:
        print(f"\n--- Turn {m['index']} ---")
        print(f"Assistant said: {m['assistant_prev'][:200]}...")
        print(f"User replied: {m['user']}")


if __name__ == "__main__":
    main()
