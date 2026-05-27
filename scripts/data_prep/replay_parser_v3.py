"""Replay parser v3 — OQ7-correct perception encoding.

Re-parses raw human-replay JSONL files into `data/bc_transitions_v3.npz`.

Differences from v2 (Phase 0b addendum):
- v2 stored `state` and `next_state` as single (64, 64) int8 arrays (took frame[-1]).
- v3 stores `perception_input` as (N, 3, 64, 64) int8 = (first, last, max-abs-diff)
  per Phase 0c OQ7. Animation motion is now preserved in channel 2.
- v3 drops `state` and `next_state` from output (regenerable from raw JSONL if needed).
- All other v2 fields are preserved unchanged.

Pairing is identical to v2 (forward-BC):
    perception_input[t] = reduce(record[t].frame_stack)
    action_id[t]        = record[t+1].action_input.id     (action taken FROM record[t])
    per-action fields (x, y, terminal, win, levels_completed, available_actions_mask)
                        = derived from record[t+1]
    env_id / replay_id / step_in_replay = aligned with record[t]

Last record of each replay has no t+1 successor → dropped.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Make src/ importable when running as a top-level script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arc_agi_3_agent.data.perception_input import (  # noqa: E402
    reduce_frame_stack,
    t_distribution_stats,
)

REPLAYS_ROOT = Path(
    r"C:\Users\adars\Downloads\ARC-AGI-3\data\human_replays"
    r"\arc_agi_3_public_demo_human_testing\public_games-dataset"
)
OUT_NPZ = ROOT / "data" / "bc_transitions_v3.npz"
OUT_META = ROOT / "data" / "bc_transitions_v3_meta.json"

ACTION_NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}


def normalize_action_id(raw) -> int | None:
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


def parse_record(rec):
    """Returns (perception_input (3,64,64) int8, raw_T, action_id, ax, ay,
                 is_win, is_game_over, lvl, avail_mask (8,) bool)
    or None if record is unparseable."""
    data = rec.get("data", {})
    frame = data.get("frame")
    if frame is None:
        return None
    try:
        arr = np.asarray(frame, dtype=np.int16)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 3 or arr.shape[1:] != (64, 64) or arr.shape[0] == 0:
        return None
    raw_T = arr.shape[0]
    perception = reduce_frame_stack(arr)

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

    return perception, raw_T, aid, ax, ay, is_win, is_game_over, int(lvl), mask


def main():
    env_dirs = sorted([p for p in REPLAYS_ROOT.iterdir() if p.is_dir()])
    env_to_idx = {p.name: i for i, p in enumerate(env_dirs)}

    perceptions: list[np.ndarray] = []
    env_ids, replay_ids, step_idx = [], [], []
    action_ids, action_xs, action_ys = [], [], []
    terminals, wins, levels = [], [], []
    avail_masks: list[np.ndarray] = []
    raw_T_per_state: list[int] = []

    n_replays = 0
    n_records_total = 0
    n_records_dropped_bad = 0
    n_tail_records_dropped = 0

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

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
            n_replays += 1
            for t in range(len(parsed) - 1):
                cur = parsed[t]
                nxt = parsed[t + 1]
                # nxt[2] is action_id taken FROM cur.perception producing nxt.perception
                if nxt[2] is None:
                    continue
                perceptions.append(cur[0])
                raw_T_per_state.append(cur[1])
                env_ids.append(eidx)
                replay_ids.append(n_replays - 1)
                step_idx.append(t)
                action_ids.append(nxt[2])
                action_xs.append(nxt[3])
                action_ys.append(nxt[4])
                terminals.append(nxt[5] or nxt[6])
                wins.append(nxt[5])
                levels.append(nxt[7])
                avail_masks.append(nxt[8])
            n_tail_records_dropped += 1

    N = len(perceptions)
    print(f"replays kept: {n_replays}")
    print(f"records read total: {n_records_total}")
    print(f"records dropped (bad frame/action): {n_records_dropped_bad}")
    print(f"tail records dropped (no t+1): {n_tail_records_dropped}")
    print(f"forward transitions emitted: {N}")
    if N == 0:
        print("No transitions emitted. Exiting.")
        return

    perception_arr = np.stack(perceptions).astype(np.int8)
    print(f"perception_input array shape: {perception_arr.shape} dtype={perception_arr.dtype}")

    np.savez_compressed(
        OUT_NPZ,
        perception_input=perception_arr,
        env_ids=np.array(env_ids, dtype=np.int8),
        replay_ids=np.array(replay_ids, dtype=np.int32),
        step_in_replay=np.array(step_idx, dtype=np.int32),
        action_id=np.array(action_ids, dtype=np.int8),
        action_x=np.array(action_xs, dtype=np.int8),
        action_y=np.array(action_ys, dtype=np.int8),
        terminal=np.array(terminals, dtype=bool),
        win=np.array(wins, dtype=bool),
        levels_completed=np.array(levels, dtype=np.int8),
        available_actions_mask=np.stack(avail_masks),
        raw_T_per_state=np.array(raw_T_per_state, dtype=np.int32),
    )
    size_mb = OUT_NPZ.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_NPZ} ({size_mb:.1f} MB)")

    # Spot-check: t=1 frequency, motion-channel non-zero rate.
    t_stats = t_distribution_stats(raw_T_per_state)
    motion_channel = perception_arr[:, 2, :, :]
    motion_nonzero_rate = float((motion_channel.reshape(N, -1).sum(axis=1) > 0).sum() / N)
    ch0_eq_ch1 = perception_arr[:, 0, :, :] == perception_arr[:, 1, :, :]
    ch0_ch1_identity_rate = float(ch0_eq_ch1.reshape(N, -1).all(axis=1).sum() / N)

    action_hist = Counter(action_ids)
    meta = {
        "version": "v3_perception_oq7",
        "pairing": "(perception_input = reduce(record[t].frame_stack), "
                   "action = record[t+1].action_input.id, "
                   "post-action fields = record[t+1])",
        "perception_channels": "(channel 0 = frame[0]; channel 1 = frame[-1]; "
                               "channel 2 = max-abs-diff over T-1 transitions)",
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
        "spot_check": {
            "t1_frac": t_stats["t1_frac"],
            "t_max": t_stats["t_max"],
            "t_mean": t_stats["t_mean"],
            "motion_channel_nonzero_rate": motion_nonzero_rate,
            "ch0_eq_ch1_identity_rate": ch0_ch1_identity_rate,
            "note": "Expected: t1_frac ≈ 0.71 + motion_nonzero ≈ 0.29 (Phase 0a §4.5).",
        },
        "vs_v2_diff": "v3 adds perception_input (3,64,64); drops state and next_state. "
                      "v2 (bc_transitions_v2.npz) retained for inverse-model aux.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_META}")
    print(f"\n=== SPOT-CHECK ===")
    print(f"T=1 frac:                 {t_stats['t1_frac']:.4f}  (expected ~0.71)")
    print(f"T mean / max:             {t_stats['t_mean']:.2f} / {t_stats['t_max']}")
    print(f"Motion-channel non-zero:  {motion_nonzero_rate:.4f}  (expected ~0.29)")
    print(f"ch0==ch1 identity rate:   {ch0_ch1_identity_rate:.4f}  (expected ~0.71)")
    print(f"Action hist:              {dict(sorted(action_hist.items()))}")
    print(f"Wins / total terminals:   {sum(wins)} / {sum(terminals)}")


if __name__ == "__main__":
    main()
