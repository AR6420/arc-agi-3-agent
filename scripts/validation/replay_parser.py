"""Replay parser for ARC-AGI-3 human-replay JSONL files.

Emits (state, action, next_state, terminal) tuples ready for BC training.
Pure data preprocessing — no model, no architecture.

Replay JSONL record schema (from agents/recorder.py + observed records):
    {"timestamp": "...", "data": {
        "game_id": "sp80-589a99af",
        "frame": [[[...]]],                # (T, 64, 64) int list-of-list-of-list
        "state": "NOT_FINISHED" | "WIN" | "GAME_OVER",
        "score" | "levels_completed": int,   # field renamed in v0.9.3
        "win_levels": int,
        "guid": "...",
        "full_reset": bool,
        "available_actions": [int, ...],
        "action_input": {                     # may be missing for first frame
            "id": int | "ACTION1..7" | "RESET",   # two recorder versions co-exist
            "data": {"x": int, "y": int, "game_id": str, ...},
            "reasoning": dict | None,
        },
    }}

Each (state, action, next_state) transition pairs a frame F_t (= the *response*
frame after action A_{t-1}) with the action A_t taken at step t (recorded on
record t+1) and the resulting frame F_{t+1} (recorded on record t+1 next-cycle).
We treat each record as carrying:
  - frame_t (the post-action observation at time t)
  - action_t (the action the human chose, given frame_t)  -- BC target
  - the *next* record carries the response frame F_{t+1}

This is the standard "observation -> action -> next-observation" layout for BC.

Action ID normalization:
    "RESET" -> 0
    "ACTION1".."ACTION7" -> 1..7
    int already in 0..7 -> kept as int

Output schema:
    np.savez_compressed(
        out_path,
        env_ids=np.array of int (sequential per env, see env_to_idx),
        replay_ids=np.array of int (sequential per replay),
        step_in_replay=np.array of int,
        state=np.array shape (N, 64, 64) int8,    # last frame of frame stack
        action_id=np.array shape (N,) int8,        # 0..7
        action_x=np.array shape (N,) int8,         # 0..63 or -1 if non-ACTION6
        action_y=np.array shape (N,) int8,
        next_state=np.array shape (N, 64, 64) int8,
        terminal=np.array shape (N,) bool,
        win=np.array shape (N,) bool,
        levels_completed=np.array shape (N,) int8,
        available_actions_mask=np.array shape (N, 8) bool,  # idx 0..7
    )
    env_to_idx, idx_to_env saved to a JSON sidecar.

Notes:
  - We keep only the *last* frame of each (T, 64, 64) stack as the "state" tensor.
    Animation transitions are discarded for now — Phase 0c may revisit this.
  - We skip records with no `action_input` (these are the initial reset frames).
  - We pair frame at record i with frame at record i+1 as next_state. If record i
    is the last in a replay, next_state = state and terminal = True.
"""

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np

REPLAYS_ROOT = Path(
    r"C:\Users\adars\Downloads\ARC-AGI-3\data\human_replays"
    r"\arc_agi_3_public_demo_human_testing\public_games-dataset"
)
OUT_NPZ = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions.npz")
OUT_META = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions_meta.json")

ACTION_NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}


def normalize_action_id(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        if 0 <= raw <= 7:
            return raw
        return None
    if isinstance(raw, str):
        return ACTION_NAME_TO_ID.get(raw.upper())
    return None


def iter_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def last_frame_as_array(frame: list) -> np.ndarray | None:
    """Take last entry of (T, 64, 64) list, return (64, 64) int8 array. None on bad shape."""
    if not isinstance(frame, list) or len(frame) == 0:
        return None
    last = frame[-1]
    if not isinstance(last, list) or len(last) != 64:
        return None
    if not isinstance(last[0], list) or len(last[0]) != 64:
        return None
    try:
        arr = np.array(last, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if arr.shape != (64, 64):
        return None
    return arr


def main():
    env_dirs = sorted([p for p in REPLAYS_ROOT.iterdir() if p.is_dir()])
    env_to_idx = {p.name: i for i, p in enumerate(env_dirs)}
    idx_to_env = {i: p.name for i, p in enumerate(env_dirs)}

    states, next_states = [], []
    env_ids, replay_ids, step_idx = [], [], []
    action_ids, action_xs, action_ys = [], [], []
    terminals, wins, levels = [], [], []
    avail_masks = []

    replay_counter = 0
    skipped_records = 0
    skipped_replays_empty = 0

    for env_dir in env_dirs:
        env_id = env_to_idx[env_dir.name]
        for replay in sorted(env_dir.glob("*.recording.jsonl")):
            records = list(iter_records(replay))
            if not records:
                skipped_replays_empty += 1
                continue
            replay_counter += 1
            parsed_steps = []
            for rec in records:
                data = rec.get("data", {})
                frame = data.get("frame")
                arr = last_frame_as_array(frame) if frame is not None else None
                if arr is None:
                    skipped_records += 1
                    continue
                ai = data.get("action_input")
                if ai is None:
                    continue
                aid = normalize_action_id(ai.get("id"))
                if aid is None:
                    skipped_records += 1
                    continue
                ax, ay = -1, -1
                if aid == 6:
                    adata = ai.get("data") or {}
                    ax = int(adata.get("x", -1)) if isinstance(adata.get("x"), (int, float)) else -1
                    ay = int(adata.get("y", -1)) if isinstance(adata.get("y"), (int, float)) else -1
                    if not (0 <= ax <= 63):
                        ax = -1
                    if not (0 <= ay <= 63):
                        ay = -1
                state_str = data.get("state", "")
                is_win = state_str == "WIN"
                is_game_over = state_str == "GAME_OVER"
                lvl_completed = data.get("levels_completed", data.get("score", 0))
                avail = data.get("available_actions", []) or []
                mask = np.zeros(8, dtype=bool)
                for a in avail:
                    aid_norm = normalize_action_id(a)
                    if aid_norm is not None:
                        mask[aid_norm] = True
                parsed_steps.append((arr, aid, ax, ay, is_win, is_game_over, lvl_completed, mask))

            for t, (arr, aid, ax, ay, is_win, is_game_over, lvl, mask) in enumerate(parsed_steps):
                if t + 1 < len(parsed_steps):
                    next_arr = parsed_steps[t + 1][0]
                    is_terminal = False
                else:
                    next_arr = arr  # final step: next = self
                    is_terminal = True
                states.append(arr)
                next_states.append(next_arr)
                env_ids.append(env_id)
                replay_ids.append(replay_counter - 1)
                step_idx.append(t)
                action_ids.append(aid)
                action_xs.append(ax)
                action_ys.append(ay)
                terminals.append(is_terminal)
                wins.append(is_win and is_terminal)
                levels.append(int(lvl) if isinstance(lvl, (int, float)) else 0)
                avail_masks.append(mask)

    N = len(states)
    print(f"Parsed transitions: N={N}")
    print(f"Replays processed: {replay_counter} (empty skipped: {skipped_replays_empty})")
    print(f"Records skipped (bad frame or action): {skipped_records}")

    if N == 0:
        print("No transitions parsed. Exiting.")
        return

    np.savez_compressed(
        OUT_NPZ,
        env_ids=np.array(env_ids, dtype=np.int8),
        replay_ids=np.array(replay_ids, dtype=np.int32),
        step_in_replay=np.array(step_idx, dtype=np.int32),
        state=np.stack(states).astype(np.int8),
        next_state=np.stack(next_states).astype(np.int8),
        action_id=np.array(action_ids, dtype=np.int8),
        action_x=np.array(action_xs, dtype=np.int8),
        action_y=np.array(action_ys, dtype=np.int8),
        terminal=np.array(terminals, dtype=bool),
        win=np.array(wins, dtype=bool),
        levels_completed=np.array(levels, dtype=np.int8),
        available_actions_mask=np.stack(avail_masks),
    )
    print(f"Wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1024 / 1024:.1f} MB)")

    meta = {
        "n_transitions": N,
        "n_replays": replay_counter,
        "n_envs": len(env_dirs),
        "env_to_idx": env_to_idx,
        "idx_to_env": {int(k): v for k, v in idx_to_env.items()},
        "action_hist_post_parse": {
            int(k): int(v)
            for k, v in zip(*np.unique(np.array(action_ids), return_counts=True))
        },
        "skipped_records": skipped_records,
        "skipped_replays_empty": skipped_replays_empty,
        "win_terminals": int(sum(wins)),
        "total_terminals": int(sum(terminals)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_META}")
    print(f"Action hist post-parse: {meta['action_hist_post_parse']}")
    print(f"Win terminals: {meta['win_terminals']} / {meta['total_terminals']} total terminals")


if __name__ == "__main__":
    main()
