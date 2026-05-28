"""Part 4.0 — click-tolerance check.

Pick a real human click from a replay; click at the exact pixel and at offsets
(±1, ±2, ±3). Compare resulting post-frame to determine if the env accepts
approximate clicks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "noop")

import numpy as np

from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, _get_arcade, _frame_last


def _find_a_click(transition_npz: Path, env_idx: int) -> tuple[int, int, int]:
    """Find a (step, x, y) of an ACTION6 in the v3 NPZ for the given env."""
    import numpy as np
    z = np.load(transition_npz)
    env_ids = z["env_ids"].astype(np.int32)
    aids = z["action_id"]
    xs = z["action_x"]
    ys = z["action_y"]
    steps = z["step_in_replay"]
    mask = (env_ids == env_idx) & (aids == 6) & (xs >= 0) & (ys >= 0)
    idxs = np.nonzero(mask)[0]
    if len(idxs) == 0:
        return -1, -1, -1
    pick = int(idxs[len(idxs) // 2])  # mid replay
    return int(steps[pick]), int(xs[pick]), int(ys[pick])


def probe_env(arc, base_id: str, env_idx: int) -> dict:
    from arcengine import GameAction
    repo = Path(__file__).resolve().parents[1]
    v3 = repo / "data" / "bc_transitions_v3.npz"
    step_target, x0, y0 = _find_a_click(v3, env_idx)
    if x0 < 0:
        return {"env": base_id, "error": "no click in v3 npz"}

    envs_info = list(arc.get_environments())
    by_base = {e.game_id.split("-")[0]: e for e in envs_info}
    full = by_base[base_id].game_id

    # Probe exact + offsets independently — fresh env per probe.
    results: dict[tuple[int, int], dict] = {}
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (3, 0), (-3, 0)]:
        x = max(0, min(63, x0 + dx))
        y = max(0, min(63, y0 + dy))
        card = arc.open_scorecard(tags=[f"click_probe_{base_id}_{dx}_{dy}"])
        env = arc.make(full, scorecard_id=card)
        obs = env.reset()
        pre = _frame_last(obs.frame)
        obs2 = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        post = _frame_last(obs2.frame) if obs2 else pre
        arc.close_scorecard(card)
        changed = not np.array_equal(pre, post)
        # diff magnitude
        diff = int(np.sum(pre != post)) if changed else 0
        results[(dx, dy)] = {"x": x, "y": y, "changed": changed, "diff_pixels": diff}

    return {"env": base_id, "target_click": (x0, y0), "step": step_target, "probes": results}


def main() -> None:
    arc = _get_arcade(DEFAULT_ENV_DIR)
    # vc33 = pure_click holdout; idx 23. tn36 = pure_click train; idx 20.
    for base, idx in [("vc33", 23), ("tn36", 20)]:
        out = probe_env(arc, base, idx)
        print(f"\n=== {base} ===")
        if "error" in out:
            print(f"  ERROR: {out['error']}")
            continue
        print(f"target click: {out['target_click']} (replay step {out['step']})")
        print(f"{'offset':>10} {'x':>4} {'y':>4} {'changed':>9} {'diff':>6}")
        for k, v in out["probes"].items():
            print(f"{str(k):>10} {v['x']:>4} {v['y']:>4} {str(v['changed']):>9} {v['diff_pixels']:>6}")


if __name__ == "__main__":
    main()
