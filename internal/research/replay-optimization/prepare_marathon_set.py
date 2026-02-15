import os
import json
import glob
from pathlib import Path


def get_opencode_storage():
    return Path.home() / ".local" / "share" / "opencode" / "storage"


def get_message_text(message_id):
    part_dir = get_opencode_storage() / "part" / message_id
    if not part_dir.exists():
        return ""
    parts = []
    for part_file in glob.glob(str(part_dir / "*.json")):
        try:
            with open(part_file, "r") as f:
                parts.append(json.load(f))
        except:
            continue
    parts.sort(key=lambda x: x.get("id", ""))
    return "\n".join(
        [p.get("text", "") for p in parts if p.get("type") == "text"]
    ).strip()


def load_session_turns(session_id):
    msg_dir = get_opencode_storage() / "message" / session_id
    if not msg_dir.exists():
        return []
    messages = []
    for msg_file in glob.glob(str(msg_dir / "*.json")):
        try:
            with open(msg_file, "r") as f:
                m = json.load(f)
                if isinstance(m, dict):
                    messages.append(m)
        except:
            continue
    messages.sort(key=lambda x: x.get("time", {}).get("created", 0))
    turns = []
    msg_map = {m["id"]: m for m in messages if "id" in m}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("parentID"):
            parent = msg_map.get(msg["parentID"])
            if parent and parent.get("role") == "user":
                paths = set()
                summary = msg.get("summary")
                if isinstance(summary, dict):
                    diffs = summary.get("diffs")
                    if isinstance(diffs, list):
                        for d in diffs:
                            if isinstance(d, dict) and d.get("file"):
                                paths.add(d["file"])
                turns.append(
                    {
                        "id": f"{parent['id']}:{msg['id']}",
                        "user": get_message_text(parent["id"]),
                        "assistant": get_message_text(msg["id"]),
                        "user_summary": (parent.get("summary") or {}).get("title", "")
                        if isinstance(parent.get("summary"), dict)
                        else "",
                        "paths": list(paths),
                        "session": session_id,
                        "ts_ms": msg.get("time", {}).get("created", 0),
                    }
                )
    return turns


def main():
    # Full session for Marathon set
    session_id = "ses_47f5beb34ffeLlwbwysX2dQVSW"
    marathon_set = load_session_turns(session_id)

    output_path = Path("internal/research/replay-optimization/marathon_set_raw.json")
    with open(output_path, "w") as f:
        json.dump(marathon_set, f, indent=2)

    print(f"Exported {len(marathon_set)} turns to {output_path}")


if __name__ == "__main__":
    main()
