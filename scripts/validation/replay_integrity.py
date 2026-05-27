"""Replay file integrity check (Phase 0b item 7).

Read 5 random replay JSONLs. Confirm:
(a) action IDs are consistent (numeric vs string) within one file
(b) frame events are post-action (response frames) vs pre-action
(c) replay-end game state matches the `state` field

Outputs to scripts/validation/.replay_integrity.json.
"""

import json
import random
from pathlib import Path

REPLAYS_ROOT = Path(
    r"C:\Users\adars\Downloads\ARC-AGI-3\data\human_replays"
    r"\arc_agi_3_public_demo_human_testing\public_games-dataset"
)
OUT = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.replay_integrity.json")


def main():
    all_replays = []
    for env_dir in REPLAYS_ROOT.iterdir():
        if env_dir.is_dir():
            all_replays.extend(env_dir.glob("*.recording.jsonl"))
    rng = random.Random(42)
    sample = rng.sample(all_replays, min(5, len(all_replays)))

    out = []
    for replay in sample:
        info = {"path": str(replay.relative_to(REPLAYS_ROOT)), "env": replay.parent.name}
        action_id_types = set()  # "int" or "str"
        n_records = 0
        n_with_action_input = 0
        n_with_frame = 0
        first_states = []
        last_state = None
        last_action = None
        win_levels = None
        levels_completed = None
        all_states = []
        with replay.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_records += 1
                data = rec.get("data", {})
                if "frame" in data:
                    n_with_frame += 1
                ai = data.get("action_input")
                if ai is not None:
                    n_with_action_input += 1
                    aid = ai.get("id")
                    if isinstance(aid, int):
                        action_id_types.add("int")
                    elif isinstance(aid, str):
                        action_id_types.add("str")
                    last_action = aid
                state = data.get("state")
                if state:
                    all_states.append(state)
                    last_state = state
                if win_levels is None and "win_levels" in data:
                    win_levels = data["win_levels"]
                if "levels_completed" in data:
                    levels_completed = data["levels_completed"]
                if len(first_states) < 3:
                    first_states.append(state)

        info["n_records"] = n_records
        info["n_with_frame"] = n_with_frame
        info["n_with_action_input"] = n_with_action_input
        info["action_id_types"] = sorted(action_id_types)
        info["mixed_types"] = len(action_id_types) > 1
        info["first_states_3"] = first_states
        info["last_state"] = last_state
        info["last_action_id"] = last_action
        info["win_levels"] = win_levels
        info["final_levels_completed"] = levels_completed
        info["unique_states"] = sorted(set(all_states))
        out.append(info)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
