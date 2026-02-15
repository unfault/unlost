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
                    }
                )
    return turns


def main():
    sessions = [
        "ses_3a2ae1953ffekU6ydgG3lCxNat",
        "ses_47f5beb34ffeLlwbwysX2dQVSW",
        "ses_4867cc9e5ffe8Thtw9frHujUeZ",
    ]

    all_turns = []
    for sid in sessions:
        all_turns.extend(load_session_turns(sid))

    # 1. Pivotal/High Signal (based on length and paths)
    pivotal = [t for t in all_turns if len(t["paths"]) > 2 or len(t["user"]) > 500]

    # 2. Friction (based on keywords)
    friction_kw = ["retry", "error", "again", "still not", "stuck", "frustrating"]
    friction = [
        t
        for t in all_turns
        if any(kw in (t["user"] + t["user_summary"]).lower() for kw in friction_kw)
    ]

    # 3. Informational (questions)
    questions = [
        t
        for t in all_turns
        if t["user"].strip().endswith("?") or "how" in t["user"].lower()[:20]
    ]

    # Mix them up
    sprint_set = []
    seen_ids = set()

    def add_turns(source, limit):
        added = 0
        for t in source:
            if t["id"] not in seen_ids:
                sprint_set.append(t)
                seen_ids.add(t["id"])
                added += 1
                if added >= limit:
                    break

    add_turns(friction, 10)
    add_turns(pivotal, 20)
    add_turns(questions, 10)
    add_turns([t for t in all_turns if len(t["user"]) > 20], 10)  # Rest

    output_path = Path("internal/research/replay-optimization/sprint_set_raw.json")
    with open(output_path, "w") as f:
        json.dump(sprint_set, f, indent=2)

    print(f"Exported {len(sprint_set)} diverse turns to {output_path}")


if __name__ == "__main__":
    main()
