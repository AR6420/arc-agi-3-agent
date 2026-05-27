"""Engine stress test (Phase 0b Section 5) + bias-random baseline (item 9).

For each of 25 public envs:
- Reset, take 50000 biased-random actions (StochasticGoose-style: prefer
  actions that produced a frame change last time).
- Track: steps/sec, level-1 completion rate, max levels_completed, crashes.

Outputs results to scripts/validation/.engine_stress_results.json.

Single-threaded. ~30 min total estimated at engine FPS of ~5000.
"""

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["RECORDINGS_DIR"] = r"C:\Users\adars\Downloads\ARC-AGI-3\.tmp_recordings"
os.makedirs(os.environ["RECORDINGS_DIR"], exist_ok=True)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

ENV_DIR = r"C:\Users\adars\Downloads\ARC-AGI-3\arc-prize-2026-arc-agi-3\environment_files"
OUT = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.engine_stress_results.json")
N_STEPS = 50000


def frames_equal(f1, f2) -> bool:
    """Fast frame-eq check on last frame of stack. Handles list-of-ndarray and ndarray."""
    if f1 is None or f2 is None:
        return False
    try:
        import numpy as np
        a = f1[-1] if isinstance(f1, list) else f1
        b = f2[-1] if isinstance(f2, list) else f2
        if hasattr(a, "shape") and hasattr(b, "shape"):
            return bool(np.array_equal(a, b))
        return a == b
    except Exception:
        return False


def run_env(arc: "Arcade", env_id: str, n_steps: int = N_STEPS) -> dict:
    card = arc.open_scorecard(tags=[f"stress_{env_id}"])
    env = arc.make(env_id, scorecard_id=card)
    obs = env.reset()
    legal = list(obs.available_actions) if obs.available_actions else [1, 2, 3, 4, 5, 6]
    rng = random.Random(0)

    # Per-action effectiveness counter (0..7) — start uniform
    action_change_count = Counter({a: 1 for a in legal})
    action_total_count = Counter({a: 1 for a in legal})

    last_frame = obs.frame
    max_levels = obs.levels_completed
    n_wins = 0
    n_game_over = 0
    n_resets = 0
    n_level_advance = 0
    n_step_calls = 0
    last_levels_completed = obs.levels_completed
    crash_msg = None

    t0 = time.time()
    try:
        for i in range(n_steps):
            # Biased selection: weight by change-rate
            weights = []
            for a in legal:
                rate = action_change_count[a] / max(action_total_count[a], 1)
                weights.append(rate + 0.05)  # smoothing
            choice = rng.choices(legal, weights=weights, k=1)[0]
            ga = GameAction.from_id(choice)
            if choice == 6:
                data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
            else:
                data = {}
            obs = env.step(ga, data=data)
            n_step_calls += 1
            if obs is None:
                crash_msg = f"None response at step {i}"
                break
            action_total_count[choice] += 1
            if not frames_equal(last_frame, obs.frame):
                action_change_count[choice] += 1
            last_frame = obs.frame
            if obs.levels_completed > last_levels_completed:
                n_level_advance += 1
                last_levels_completed = obs.levels_completed
            if obs.levels_completed > max_levels:
                max_levels = obs.levels_completed
            state = str(obs.state)
            if state.endswith("WIN"):
                n_wins += 1
                obs = env.reset()
                n_resets += 1
                last_frame = obs.frame
                last_levels_completed = obs.levels_completed
                legal = list(obs.available_actions) if obs.available_actions else legal
            elif state.endswith("GAME_OVER"):
                n_game_over += 1
                obs = env.reset()
                n_resets += 1
                last_frame = obs.frame
                last_levels_completed = obs.levels_completed
                legal = list(obs.available_actions) if obs.available_actions else legal
            else:
                if obs.available_actions:
                    legal = list(obs.available_actions)
    except Exception as e:
        crash_msg = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    sc = arc.close_scorecard(card)

    return {
        "env": env_id,
        "n_steps": n_step_calls,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(n_step_calls / elapsed, 1) if elapsed > 0 else None,
        "max_levels_completed": max_levels,
        "n_wins": n_wins,
        "n_game_over": n_game_over,
        "n_resets": n_resets,
        "n_level_advance_events": n_level_advance,
        "crash": crash_msg,
        "action_change_rate": {a: action_change_count[a] / max(action_total_count[a], 1) for a in legal},
    }


def main():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    envs = sorted([p.name for p in Path(ENV_DIR).iterdir() if p.is_dir()])
    results = []
    for env_id in envs:
        print(f"\n--- {env_id} ---", flush=True)
        r = run_env(arc, env_id, N_STEPS)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
    OUT.write_text(json.dumps({"n_steps_each": N_STEPS, "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
