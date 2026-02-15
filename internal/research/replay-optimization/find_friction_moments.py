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
                        "user_text": get_message_text(parent["id"]),
                        "assistant_text": get_message_text(msg["id"]),
                        "user_summary": (parent.get("summary") or {}).get("title", "")
                        if isinstance(parent.get("summary"), dict)
                        else "",
                        "paths": list(paths),
                    }
                )
    return turns


def find_friction(turns):
    friction_candidates = []
    for i in range(1, len(turns)):
        # Repetitive paths (touches same file in consecutive turns)
        if turns[i]["paths"] and turns[i - 1]["paths"]:
            if set(turns[i]["paths"]) & set(turns[i - 1]["paths"]):
                friction_candidates.append(
                    {"type": "repetitive_paths", "turn": turns[i], "index": i}
                )

        # Frustrated language
        frustration_keywords = [
            "error",
            "fail",
            "wrong",
            "again",
            "still not",
            "stuck",
            "frustrating",
        ]
        text_to_check = (turns[i]["user_text"] + " " + turns[i]["user_summary"]).lower()
        if any(kw in text_to_check for kw in frustration_keywords):
            friction_candidates.append(
                {"type": "frustration_keywords", "turn": turns[i], "index": i}
            )
    return friction_candidates


def main():
    session_id = "ses_47f5beb34ffeLlwbwysX2dQVSW"
    turns = load_session_turns(session_id)
    friction = find_friction(turns)

    # Sort friction by "intensity" - many repetitive paths or many keywords
    for f in friction[:15]:
        print(f"Type: {f['type']}, Index: {f['index']}, ID: {f['turn']['id']}")
        print(f"User: {f['turn']['user_summary']}")
        print(f"Paths: {f['turn']['paths']}")
        print("-" * 20)


if __name__ == "__main__":
    main()
