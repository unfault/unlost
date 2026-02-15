import json
import csv
import os

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


def main():
    json_path = "internal/research/replay-optimization/marathon_set_raw.json"
    csv_path = "internal/research/replay-optimization/marathon_trajectory.csv"

    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        print("Files missing.")
        return

    with open(json_path, "r") as f:
        json_data = json.load(f)

    csv_rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            csv_rows.append(
                {
                    "index": int(row[0]),
                    "pivotal": row[1] == "true",
                    "friction": row[2] == "true",
                }
            )

    print(f"Loaded {len(json_data)} turns and {len(csv_rows)} trajectory rows.")

    results = []
    for i in range(min(len(json_data), len(csv_rows))):
        user_text = json_data[i].get("user", "").lower()
        is_dispute = any(kw in user_text for kw in DISPUTE_KEYWORDS)

        results.append(
            {
                "index": i,
                "pivotal": csv_rows[i]["pivotal"],
                "friction": csv_rows[i]["friction"],
                "dispute": is_dispute,
            }
        )

    # Density of pivotal turns
    window = 10
    densities = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        slice = [1 if r["pivotal"] else 0 for r in results[start : i + 1]]
        densities.append(sum(slice) / len(slice))

    # Evaluation
    H = 5
    threshold = 0.5
    filtered_triggers = []
    last_trigger = -100

    for i, d in enumerate(densities):
        # Trigger on high pivotal density
        if d >= threshold:
            if i - last_trigger >= 5:
                filtered_triggers.append(i)
                last_trigger = i

    hits = 0
    total_triggers = 0

    for t in filtered_triggers:
        total_triggers += 1
        has_hit = False
        # Does a DISPUTE happen soon?
        for j in range(t + 1, min(t + 1 + H, len(results))):
            if results[j]["dispute"]:
                has_hit = True
                break
        if has_hit:
            hits += 1

    print(f"\n--- Trajectory Evaluation (H={H}, threshold={threshold}) ---")
    print(f"Total Triggers (High Pivotal Density): {total_triggers}")
    print(f"Hits (followed by user dispute): {hits}")
    if total_triggers > 0:
        print(f"Precision@H: {hits / total_triggers:.1%}")

    # Lead Time
    lead_times = []
    for t in filtered_triggers:
        for j in range(t + 1, len(results)):
            if results[j]["dispute"]:
                lead_times.append(j - t)
                break

    if lead_times:
        avg_lead = sum(lead_times) / len(lead_times)
        print(f"Average Lead Time: {avg_lead:.1f} turns")


if __name__ == "__main__":
    main()
