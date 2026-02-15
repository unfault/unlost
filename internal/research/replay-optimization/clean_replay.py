import json
import os


def main():
    workspace_id = "wks_83ae697ac0204117"
    path = f"/home/sylvain/.local/share/unlost/workspaces/{workspace_id}/capsules.jsonl"
    out_path = f"/home/sylvain/.local/share/unlost/workspaces/{workspace_id}/capsules_clean.jsonl"

    seen = set()
    unique_turns = []

    with open(path, "r") as f:
        for line in f:
            turn = json.loads(line)
            # Use a fingerprint of the exchange
            fp = (
                turn.get("agent_session_id"),
                turn.get("capsule", {}).get("intent", ""),
                turn.get("capsule", {}).get("decision", ""),
                # turn.get("exchange_seq") # exchange_seq might be same for different turns in replay
            )
            if fp not in seen:
                seen.add(fp)
                unique_turns.append(turn)

    with open(out_path, "w") as f:
        for turn in unique_turns:
            f.write(json.dumps(turn) + "\n")

    print(f"Original: {len(seen)} unique turns found from file.")
    print(f"Cleaned file written to {out_path}")


if __name__ == "__main__":
    main()
