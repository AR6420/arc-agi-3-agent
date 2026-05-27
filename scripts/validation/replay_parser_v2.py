"""Replay parser v2 — CORRECTED forward-BC (state, action, next_state) pairing.

Phase 0b §6.3 finding: in each JSONL record, `frame` is the POST-action frame
paired with the `action_input.id` that PRODUCED it. v1 (replay_parser.py) used
(state_t = record[t].frame, action_t = record[t].action_input.id) which is the
inverse-model pairing ("given the result, what action got us here").

v2 shifts the action by +1 so (s, a, s') is a proper Markov transition:

  state_t      = record[t].frame
  action_t     = record[t+1].action_input.id    # action taken FROM state_t
  next_state   = record[t+1].frame              # state after action_t
  per-action fields (x, y, terminal, win, levels_completed, avail_mask)
                  = derived from record[t+1] (they describe the action and what it produced)
  env_id / replay_id / step_in_replay
                  = aligned with state_t (= record[t])

Last record per replay has no t+1 → dropped from forward dataset (number of
dropped tail-records reported in meta).

Output:
- data/bc_transitions_v2.npz
- data/bc_transitions_v2_meta.json

v1 stays at data/bc_transitions.npz (kept for inverse-model auxiliary task).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPLAYS_ROOT = Path(
    r"C:\Users\adars\Downloads\ARC-AGI-3\data\human_replays"
    r"\arc_agi_3_public_demo_human_testing\public_games-dataset"
)
OUT_NPZ = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions_v2.npz")
OUT_META = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions_v2_meta.json")

ACTION_NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}


def normalize_action_id(raw):
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= 7 else None
    if isinstance(raw, str):
        return ACTION_NAME_TO_ID.get(raw.upper())
    return None


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def last_frame_as_array(frame):
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
    return arr if arr.shape == (64, 64) else None


def parse_record(rec):
    """Returns (frame_arr, action_id, ax, ay, is_win, is_game_over, lvl, mask)
    OR None if record is incomplete / unparseable.
    """
    data = rec.get("data", {})
    frame = data.get("frame")
    arr = last_frame_as_array(frame) if frame is not None else None
    if arr is None:
        return None
    ai = data.get("action_input")
    aid = None
    ax, ay = -1, -1
    if ai is not None:
        aid = normalize_action_id(ai.get("id"))
        if aid == 6:
            adata = ai.get("data") or {}
            xv, yv = adata.get("x"), adata.get("y")
            if isinstance(xv, (int, float)) and 0 <= xv <= 63:
                ax = int(xv)
            if isinstance(yv, (int, float)) and 0 <= yv <= 63:
                ay = int(yv)
    state_str = data.get("state", "")
    is_win = state_str == "WIN"
    is_game_over = state_str == "GAME_OVER"
    lvl = data.get("levels_completed", data.get("score", 0))
    if not isinstance(lvl, (int, float)):
        lvl = 0
    avail = data.get("available_actions", []) or []
    mask = np.zeros(8, dtype=bool)
    for a in avail:
        n = normalize_action_id(a)
        if n is not None:
            mask[n] = True
    return arr, aid, ax, ay, is_win, is_game_over, int(lvl), mask


def main():
    env_dirs = sorted([p for p in REPLAYS_ROOT.iterdir() if p.is_dir()])
    env_to_idx = {p.name: i for i, p in enumerate(env_dirs)}

    states, next_states = [], []
    env_ids, replay_ids, step_idx = [], [], []
    action_ids, action_xs, action_ys = [], [], []
    terminals, wins, levels = [], [], []
    avail_masks = []

    n_replays = 0
    n_records_total = 0
    n_records_dropped_bad = 0
    n_tail_records_dropped = 0
    n_transitions = 0

    for env_dir in env_dirs:
        eidx = env_to_idx[env_dir.name]
        for replay_path in sorted(env_dir.glob("*.recording.jsonl")):
            parsed = []
            for rec in iter_records(replay_path):
                n_records_total += 1
                p = parse_record(rec)
                if p is None:
                    n_records_dropped_bad += 1
                    continue
                parsed.append(p)
            if len(parsed) < 2:
                n_tail_records_dropped += len(parsed)
                continue

            # Forward-BC pairing: (parsed[t].frame, parsed[t+1].action, ..., parsed[t+1].frame)
            # The action_input on parsed[0] (if any) is dropped — it describes the action
            # that produced parsed[0].frame from the prior reset state we never see.
            n_replays += 1
            for t in range(len(parsed) - 1):
                cur = parsed[t]
                nxt = parsed[t + 1]
                # nxt.action_id is the action taken FROM cur.frame producing nxt.frame
                if nxt[1] is None:
                    # No action_input on next record (rare — typically reset/open). Skip.
                    continue
                states.append(cur[0])
                next_states.append(nxt[0])
                env_ids.append(eidx)
                replay_ids.append(n_replays - 1)
                step_idx.append(t)
                action_ids.append(nxt[1])
                action_xs.append(nxt[2])
                action_ys.append(nxt[3])
                # Terminal = the transition led to WIN or GAME_OVER state (state of nxt)
                terminals.append(nxt[4] or nxt[5])
                wins.append(nxt[4])
                levels.append(nxt[6])
                avail_masks.append(nxt[7])
                n_transitions += 1
            # Tail: parsed[-1].frame has no successor → dropped from forward set
            n_tail_records_dropped += 1

    N = len(states)
    print(f"replays kept: {n_replays}")
    print(f"records read total: {n_records_total}")
    print(f"records dropped (bad frame/action): {n_records_dropped_bad}")
    print(f"tail records dropped (no t+1): {n_tail_records_dropped}")
    print(f"forward transitions emitted: {N}")

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
    size_mb = OUT_NPZ.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_NPZ} ({size_mb:.1f} MB)")

    action_hist = Counter(action_ids)
    meta = {
        "version": "v2_forward_bc",
        "pairing": "(state_t = record[t].frame, action_t = record[t+1].action_input.id, next_state = record[t+1].frame)",
        "n_transitions": N,
        "n_replays": n_replays,
        "n_envs": len(env_dirs),
        "env_to_idx": env_to_idx,
        "idx_to_env": {i: p.name for i, p in enumerate(env_dirs)},
        "records_read_total": n_records_total,
        "records_dropped_bad": n_records_dropped_bad,
        "tail_records_dropped": n_tail_records_dropped,
        "action_hist": {int(k): int(v) for k, v in sorted(action_hist.items())},
        "win_terminals": int(sum(wins)),
        "total_terminals": int(sum(terminals)),
        "vs_v1_note": "v1 (bc_transitions.npz) keeps the inverse-model pairing (state_t=record[t].frame, action=record[t].action_input.id). v2 is the correct forward-BC pairing. v1 retained for inverse-model auxiliary task.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_META}")
    print(f"Action hist: {meta['action_hist']}")
    print(f"Wins: {meta['win_terminals']} / {meta['total_terminals']} terminals")


if __name__ == "__main__":
    main()
