"""Synthetic rollout generator — OFFLINE-mode biased-random over 20 train envs.

Produces `data/bc_synth.npz` with v3-compatible schema plus an explicit
`framechange` label (computed during rollout from pre/post state diff).

Rollout policy:
- Per env: hash(env_id, "synth_v0") seed → BiasedRandomAgent.
- Run until `n_steps_per_env` transitions recorded. Reset on GAME_OVER/WIN.
- Each transition stores: pre-state OQ7 perception, action taken, (x,y) for
  ACTION6, observed framechange, env_id index.

NOT used: holdout envs (vc33, tu93, sk48, lp85, dc22). They never appear here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Workaround for arc_agi base.py:172 — must be set BEFORE importing Arcade.
os.environ.setdefault("ARC_API_KEY", "noop")

import numpy as np

from ..agents.random_agent import BiasedRandomAgent
from ..data.perception_input import reduce_frame_stack
from ..eval.harness import DEFAULT_ENV_DIR
from ..eval.splits import HOLDOUT_ENV_IDS, TRAIN_ENV_IDS


REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH_NPZ = REPO_ROOT / "data" / "bc_synth.npz"
SYNTH_META = REPO_ROOT / "data" / "bc_synth_meta.json"

SYNTH_VERSION = "synth_v0"


def _synth_seed(env_id: str) -> int:
    return hash((env_id, SYNTH_VERSION)) & 0xFFFFFFFF


def _frame_stack_to_array(frame_payload) -> np.ndarray:
    """Normalize SDK frame payload to (T, 64, 64) int8."""
    if frame_payload is None:
        return np.zeros((1, 64, 64), dtype=np.int8)
    if isinstance(frame_payload, list):
        if not frame_payload:
            return np.zeros((1, 64, 64), dtype=np.int8)
        arr = np.stack([np.asarray(f, dtype=np.int8) for f in frame_payload], axis=0)
        return arr
    arr = np.asarray(frame_payload, dtype=np.int8)
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


def _run_one_env(arc, env_id_full: str, n_steps: int, env_idx: int) -> dict:
    from arcengine import GameAction, GameState

    base = env_id_full.split("-")[0]
    seed = _synth_seed(base)
    agent = BiasedRandomAgent()

    card = arc.open_scorecard(tags=[f"synth_{base}"])
    env = arc.make(env_id_full, scorecard_id=card)
    obs = env.reset()

    avail = list(obs.available_actions or [])
    agent.reset_for_env(base, avail, run_id=0)

    perceptions: list[np.ndarray] = []
    action_ids: list[int] = []
    action_xs: list[int] = []
    action_ys: list[int] = []
    framechanges: list[int] = []

    pre_stack = _frame_stack_to_array(obs.frame)
    pre_perception = reduce_frame_stack(pre_stack)  # (3, 64, 64) int8
    pre_last = pre_stack[-1]

    steps = 0
    t0 = time.perf_counter()
    while steps < n_steps:
        state = str(obs.state)
        if state.endswith("WIN") or state.endswith("GAME_OVER") or state.endswith("NOT_PLAYED"):
            obs = env.step(GameAction.RESET, data={})
            if obs is None:
                break
            pre_stack = _frame_stack_to_array(obs.frame)
            pre_perception = reduce_frame_stack(pre_stack)
            pre_last = pre_stack[-1]
            continue

        action_id, action_data = agent.choose_action(pre_last)
        ga = GameAction.from_id(action_id)
        obs = env.step(ga, data=action_data)
        if obs is None:
            break

        post_stack = _frame_stack_to_array(obs.frame)
        framechange = int(not np.array_equal(post_stack[-1], pre_last))

        perceptions.append(pre_perception)
        action_ids.append(int(action_id))
        action_xs.append(int(action_data.get("x", -1)))
        action_ys.append(int(action_data.get("y", -1)))
        framechanges.append(framechange)

        pre_stack = post_stack
        pre_perception = reduce_frame_stack(pre_stack)
        pre_last = pre_stack[-1]
        steps += 1

    arc.close_scorecard(card)

    wall = time.perf_counter() - t0
    return {
        "env_base": base,
        "env_idx": env_idx,
        "steps": steps,
        "wall_seconds": round(wall, 2),
        "perceptions": np.stack(perceptions, axis=0) if perceptions else np.empty((0, 3, 64, 64), dtype=np.int8),
        "action_ids": np.array(action_ids, dtype=np.int8),
        "action_xs": np.array(action_xs, dtype=np.int8),
        "action_ys": np.array(action_ys, dtype=np.int8),
        "framechanges": np.array(framechanges, dtype=np.int8),
    }


def generate(
    n_steps_per_env: int = 10_000,
    env_dir: str = DEFAULT_ENV_DIR,
    out_path: Path = SYNTH_NPZ,
) -> dict:
    """Run synth gen across 20 train envs. Returns meta dict and writes NPZ."""
    from arc_agi import Arcade, OperationMode

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    envs_info = list(arc.get_environments())
    by_base = {e.game_id.split("-")[0]: e for e in envs_info}

    env_to_idx: dict[str, int] = {e: i for i, e in enumerate(TRAIN_ENV_IDS)}
    per_env_meta: list[dict] = []

    all_perceptions: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_xs: list[np.ndarray] = []
    all_ys: list[np.ndarray] = []
    all_fc: list[np.ndarray] = []
    all_env_idx: list[np.ndarray] = []

    t_all = time.perf_counter()
    for base in TRAIN_ENV_IDS:
        if base in HOLDOUT_ENV_IDS:
            raise RuntimeError(f"holdout env {base} in TRAIN_ENV_IDS — split corruption")
        if base not in by_base:
            print(f"WARN: env {base} not in bundle; skipping.", flush=True)
            continue
        full_id = by_base[base].game_id
        env_idx = env_to_idx[base]
        print(f"--- synth {base} (target {n_steps_per_env}) ---", flush=True)
        r = _run_one_env(arc, full_id, n_steps_per_env, env_idx)
        per_env_meta.append({
            "env_base": r["env_base"], "env_idx": r["env_idx"],
            "steps": r["steps"], "wall_seconds": r["wall_seconds"],
        })
        print(f"  steps={r['steps']} wall={r['wall_seconds']}s", flush=True)
        if r["steps"] > 0:
            all_perceptions.append(r["perceptions"])
            all_actions.append(r["action_ids"])
            all_xs.append(r["action_xs"])
            all_ys.append(r["action_ys"])
            all_fc.append(r["framechanges"])
            all_env_idx.append(np.full(r["steps"], r["env_idx"], dtype=np.int8))

    total_wall = time.perf_counter() - t_all

    perception = np.concatenate(all_perceptions, axis=0) if all_perceptions else np.empty((0, 3, 64, 64), dtype=np.int8)
    action_id = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0,), dtype=np.int8)
    action_x = np.concatenate(all_xs, axis=0) if all_xs else np.empty((0,), dtype=np.int8)
    action_y = np.concatenate(all_ys, axis=0) if all_ys else np.empty((0,), dtype=np.int8)
    framechange = np.concatenate(all_fc, axis=0) if all_fc else np.empty((0,), dtype=np.int8)
    env_ids = np.concatenate(all_env_idx, axis=0) if all_env_idx else np.empty((0,), dtype=np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        perception_input=perception,
        action_id=action_id,
        action_x=action_x,
        action_y=action_y,
        framechange=framechange,
        env_ids=env_ids,
    )

    meta = {
        "version": SYNTH_VERSION,
        "n_transitions": int(perception.shape[0]),
        "n_envs": len(per_env_meta),
        "n_steps_per_env_target": n_steps_per_env,
        "env_to_idx": env_to_idx,
        "per_env": per_env_meta,
        "total_wall_seconds": round(total_wall, 1),
        "policy": "BiasedRandomAgent + seed=hash((env_id, 'synth_v0'))",
        "encoding": "perception_input = OQ7 reduce_frame_stack(pre-action T-stack)",
        "framechange_def": "1 if post_frame[-1] != pre_frame[-1] else 0",
    }
    SYNTH_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10_000)
    ap.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    ap.add_argument("--out", default=str(SYNTH_NPZ))
    args = ap.parse_args()
    meta = generate(args.n_steps, args.env_dir, Path(args.out))
    print(f"\n=== SYNTH GEN COMPLETE ===")
    print(f"n_transitions: {meta['n_transitions']}")
    print(f"total_wall:    {meta['total_wall_seconds']}s")
    print(f"wrote:         {SYNTH_NPZ}")
