import json
import csv


def rolling_mean(data, window):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        slice = data[start : i + 1]
        result.append(sum(slice) / len(slice))
    return result


def main():
    rows = []
    with open(
        "internal/research/replay-optimization/marathon_trajectory.csv", "r"
    ) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(
                {
                    "index": int(row[0]),
                    "pivotal": row[1] == "true",
                    "friction": row[2] == "true",
                }
            )

    pivotal_vals = [1 if r["pivotal"] else 0 for r in rows]
    friction_vals = [r["friction"] for r in rows]

    window_size = 5
    densities = rolling_mean(pivotal_vals, window_size)

    # Track episodes of medium instability
    threshold = 0.6
    episodes = []
    current_episode = None

    for i, d in enumerate(densities):
        if d >= threshold:
            if current_episode is None:
                current_episode = {"start": i, "peaks": [d], "friction_count": 0}
            else:
                current_episode["peaks"].append(d)
            if friction_vals[i]:
                current_episode["friction_count"] += 1
        else:
            if current_episode is not None:
                current_episode["end"] = i - 1
                current_episode["len"] = i - current_episode["start"]
                episodes.append(current_episode)
                current_episode = None

    total_friction = sum(1 for f in friction_vals if f)
    friction_in_episodes = sum(e["friction_count"] for e in episodes)

    print(f"--- Instability Episodes (Density > {threshold}) ---")
    print(
        f"Friction coverage: {friction_in_episodes}/{total_friction} ({friction_in_episodes / total_friction:.1%})"
    )

    # Pre-friction lookahead
    # For every turn that is NOT friction but has high density, does friction follow?
    lookahead = 5
    false_alarms = 0
    valid_alarms = 0

    for i, d in enumerate(densities):
        if d >= threshold and not friction_vals[i]:
            # It's an alarm!
            if any(friction_vals[i + 1 : i + 1 + lookahead]):
                valid_alarms += 1
            else:
                false_alarms += 1

    print(f"\nAlarm Reliability (Lookahead={lookahead}):")
    print(f"Valid Alarms: {valid_alarms}")
    print(f"False Alarms: {false_alarms}")
    print(f"Precision: {valid_alarms / (valid_alarms + false_alarms):.1%}")


if __name__ == "__main__":
    main()
