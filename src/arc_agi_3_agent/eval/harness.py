"""Local eval harness — Phase 0c §2.

Runs an agent against the 25 public envs in OFFLINE mode, computes RHAE per env,
reports `harness_score_train` (20 envs) and `harness_score_holdout` (5 envs).

Output written to `harness_runs/<timestamp>/`:
  - summary.json             — overall scores + per-env breakdown
  - per_env/<env_id>.jsonl   — per-step records (action, state, scorecard counters)
  - action_histograms.json   — per-env action distribution
  - latency_p50_p95_p99.json — model inference latency percentiles
  - gate_status.txt          — PASS / FAIL line for `harness_score_holdout`

Usage:
    python -m arc_agi_3_agent.eval.harness --agent random --output harness_runs/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Workaround for arc_agi base.py:172 — must be set BEFORE importing Arcade.
os.environ.setdefault("ARC_API_KEY", "noop")

import numpy as np

from arc_agi_3_agent.agents.random_agent import (
    BiasedRandomAgent,
    max_actions_for_env,
    per_env_seed,
)
from arc_agi_3_agent.eval.scoring import env_score_from_actions, total_score
from arc_agi_3_agent.eval.splits import (
    ALL_PUBLIC_ENV_IDS,
    HOLDOUT_ENV_IDS,
    TRAIN_ENV_IDS,
    is_holdout,
)


GATE_HOLDOUT_THRESHOLD = 10.0  # CLAUDE.md §1.2

# Path to bundled Kaggle env files (mirrored locally for OFFLINE dev).
DEFAULT_ENV_DIR = r"C:\Users\adars\Downloads\ARC-AGI-3\arc-prize-2026-arc-agi-3\environment_files"

EARLY_EXIT_LEVEL1_ACTION_BUDGET = 50  # Phase 0c §2.5 — legacy; replaced by per-env threshold (Phase 2.5)
EARLY_EXIT_LEVEL1_FLOOR = 150         # Phase 2.5 — minimum action budget before declaring level-1 unreached
EARLY_EXIT_PROGRESS_WINDOW = 40       # Phase 2.5 — exit if no new frame-hash within this window AND no level advance


def _get_arcade(env_dir: str):
    """Lazy import to keep harness importable without arc_agi installed (e.g. for tests)."""
    from arc_agi import Arcade, OperationMode

    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=env_dir,
    )


def _frame_last(frame_payload: Any) -> np.ndarray:
    """Extract the last (64, 64) frame from whatever the SDK returns."""
    if frame_payload is None:
        return np.zeros((64, 64), dtype=np.int8)
    if isinstance(frame_payload, list):
        if not frame_payload:
            return np.zeros((64, 64), dtype=np.int8)
        last = frame_payload[-1]
        if hasattr(last, "shape"):
            return np.asarray(last, dtype=np.int8)
        return np.asarray(last, dtype=np.int8)
    # ndarray-like
    arr = np.asarray(frame_payload)
    if arr.ndim == 3:
        return arr[-1].astype(np.int8)
    return arr.astype(np.int8)


def run_one_env(
    arc,
    env_id: str,
    env_info,
    agent: BiasedRandomAgent,
    out_per_env_path: Path,
    run_id: int,
) -> dict:
    """Run agent on one env. Returns per-env summary dict."""
    from arcengine import GameAction, GameState

    baseline_actions = list(env_info.baseline_actions or [])
    n_levels = len(baseline_actions)
    max_actions = max_actions_for_env(baseline_actions) if baseline_actions else 250

    out_per_env_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = out_per_env_path.open("w", encoding="utf-8")

    card = arc.open_scorecard(tags=[f"harness_{env_id}", f"run_{run_id}"])
    env = arc.make(env_id, scorecard_id=card)
    obs = env.reset()
    if obs is None:
        log_fh.close()
        arc.close_scorecard(card)
        return {
            "env_id": env_id,
            "error": "reset returned None",
            "score": 0.0,
            "levels_completed": 0,
            "actions": 0,
            "level_actions": [0] * max(n_levels, 1),
            "baseline_actions": baseline_actions,
        }

    avail = list(obs.available_actions or [])
    agent.reset_for_env(env_id, avail, run_id=run_id)

    # Phase 2.5 — per-env early-exit budget: max(floor, 3 * level1 baseline)
    level1_baseline = baseline_actions[0] if baseline_actions else 50
    level1_exit_budget = max(EARLY_EXIT_LEVEL1_FLOOR, 3 * level1_baseline)

    action_hist: Counter[int] = Counter()
    latencies: list[float] = []
    level_actions = [0] * n_levels
    levels_completed_seen = 0
    actions_taken = 0
    resets_taken = 0
    early_exit_reason: str | None = None
    # Phase 2.5 progress-based exit: track distinct frame hashes; if no new state
    # AND no level advance for EARLY_EXIT_PROGRESS_WINDOW actions, exit early.
    seen_hashes: set[int] = set()
    last_new_state_step = 0
    last_level_advance_step = 0

    t_start = time.perf_counter()
    while actions_taken < max_actions:
        state = str(obs.state)
        if state.endswith("WIN") or state.endswith("GAME_OVER"):
            break  # natural terminal — stop without issuing extra resets.

        last_frame = _frame_last(obs.frame)

        t0 = time.perf_counter()
        # Pass raw frame payload to non-random agents (they need T-stack for OQ7 reduce).
        if isinstance(agent, BiasedRandomAgent):
            action_id, action_data = agent.choose_action(last_frame)
        else:
            action_id, action_data = agent.choose_action(obs.frame)
        latencies.append(time.perf_counter() - t0)

        ga = GameAction.from_id(action_id)
        obs = env.step(ga, data=action_data)
        if obs is None:
            early_exit_reason = "step returned None"
            break
        actions_taken += 1
        action_hist[action_id] += 1

        cur_level = int(getattr(obs, "levels_completed", 0))
        # Attribute the action to the *current* level (the one the agent was trying to complete).
        if cur_level < n_levels:
            level_actions[cur_level] += 1
        if cur_level > levels_completed_seen:
            # Just completed a level — record the transition.
            levels_completed_seen = cur_level
            last_level_advance_step = actions_taken

        # Phase 2.5 — progress tracking via post-action frame[-1] hash.
        post_frame = _frame_last(obs.frame)
        h = hash(post_frame.tobytes())
        if h not in seen_hashes:
            seen_hashes.add(h)
            last_new_state_step = actions_taken

        # Per-step log line (Phase 1 Issues 2 + 3 — but OFFLINE counters are zero per S5)
        log_fh.write(json.dumps({
            "step": actions_taken,
            "action_id": action_id,
            "action_data": action_data,
            "state": state,
            "levels_completed": cur_level,
            "frame_T": len(obs.frame) if isinstance(obs.frame, list) else 1,
        }) + "\n")

        # Phase 2.5 — per-env level-1 budget (replaces old 50-action global cap).
        if levels_completed_seen == 0 and actions_taken >= level1_exit_budget:
            early_exit_reason = f"level 1 not reached in {level1_exit_budget} actions"
            break

        # Phase 2.5 — progress-based exit: stuck if no new state AND no level advance.
        steps_since_progress = actions_taken - max(last_new_state_step, last_level_advance_step)
        if steps_since_progress >= EARLY_EXIT_PROGRESS_WINDOW and actions_taken >= EARLY_EXIT_PROGRESS_WINDOW:
            early_exit_reason = (
                f"no progress (no new state + no level advance) for "
                f"{EARLY_EXIT_PROGRESS_WINDOW} actions at step {actions_taken}"
            )
            break

    wall = time.perf_counter() - t_start
    sc = arc.close_scorecard(card)
    log_fh.close()

    score = env_score_from_actions(level_actions, baseline_actions, levels_completed_seen)

    # Cross-check vs OFFLINE scorecard (expected zeros per Phase 0b S5).
    sc_actions = sc_resets = 0
    if sc and sc.environments:
        run = sc.environments[0].runs[0]
        sc_actions = int(run.actions or 0)
        sc_resets = int(run.resets or 0)

    return {
        "env_id": env_id,
        "score": score,
        "levels_completed": levels_completed_seen,
        "actions": actions_taken,
        "level_actions": level_actions,
        "baseline_actions": baseline_actions,
        "n_levels": n_levels,
        "action_hist": dict(action_hist),
        "wall_seconds": round(wall, 3),
        "latency_p50_ms": round(1000 * statistics.median(latencies), 3) if latencies else None,
        "latency_p95_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 3) if len(latencies) >= 20 else None,
        "early_exit_reason": early_exit_reason,
        "scorecard_actions_offline": sc_actions,
        "scorecard_resets_offline": sc_resets,
    }


def run_harness(env_dir: str, agent_name: str, out_dir: Path, run_id: int = 0) -> dict:
    """Run the full harness across 25 envs.

    Returns:
        Summary dict containing harness_score_train, harness_score_holdout, per-env breakdown.
    """
    arc = _get_arcade(env_dir)
    if agent_name == "random":
        agent = BiasedRandomAgent()
    elif agent_name == "elite_v0":
        from arc_agi_3_agent.agent.elite_v0 import EliteV0
        agent = EliteV0()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_env").mkdir(exist_ok=True)

    envs_info = list(arc.get_environments())
    # Map by base env id (strip hash suffix).
    by_base = {e.game_id.split("-")[0]: e for e in envs_info}

    per_env_results: list[dict] = []
    t_total_start = time.perf_counter()

    for base_id in ALL_PUBLIC_ENV_IDS:
        if base_id not in by_base:
            print(f"WARN: env {base_id} not found in bundle; skipping.", flush=True)
            continue
        env_info = by_base[base_id]
        full_id = env_info.game_id
        log_path = out_dir / "per_env" / f"{base_id}.jsonl"
        print(f"--- {base_id} ({full_id}) ---", flush=True)
        result = run_one_env(arc, full_id, env_info, agent, log_path, run_id)
        result["holdout"] = is_holdout(base_id)
        per_env_results.append(result)
        print(
            f"  score={result['score']:.2f} "
            f"lvls={result['levels_completed']}/{result['n_levels']} "
            f"actions={result['actions']} "
            f"wall={result['wall_seconds']}s "
            f"{('[holdout]' if result['holdout'] else '[train]')}",
            flush=True,
        )

    total_wall = time.perf_counter() - t_total_start

    train_scores = [r["score"] for r in per_env_results if not r["holdout"]]
    holdout_scores = [r["score"] for r in per_env_results if r["holdout"]]
    harness_score_train = total_score(train_scores)
    harness_score_holdout = total_score(holdout_scores)
    gate_pass = harness_score_holdout >= GATE_HOLDOUT_THRESHOLD

    # Aggregate action histogram + latency percentiles across all envs.
    agg_hist: Counter[int] = Counter()
    all_lats: list[float] = []
    for r in per_env_results:
        agg_hist.update(r.get("action_hist", {}))
        # latencies aren't preserved per-env beyond p50/p95; aggregate from per-env summaries
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "run_id": run_id,
        "env_dir": env_dir,
        "harness_score_train": harness_score_train,
        "harness_score_holdout": harness_score_holdout,
        "gate_threshold": GATE_HOLDOUT_THRESHOLD,
        "gate_pass": gate_pass,
        "n_train": len(train_scores),
        "n_holdout": len(holdout_scores),
        "total_wall_seconds": round(total_wall, 1),
        "per_env": per_env_results,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "action_histograms.json").write_text(
        json.dumps({"aggregate": dict(agg_hist),
                    "per_env": {r["env_id"]: r.get("action_hist", {}) for r in per_env_results}},
                   indent=2),
        encoding="utf-8",
    )
    (out_dir / "gate_status.txt").write_text(
        f"{'PASS' if gate_pass else 'FAIL'}: harness_score_holdout = {harness_score_holdout:.4f} "
        f"(threshold {GATE_HOLDOUT_THRESHOLD})\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC-AGI-3 local eval harness")
    ap.add_argument("--agent", default="random", help="agent name (default: random)")
    ap.add_argument("--env-dir", default=DEFAULT_ENV_DIR, help="path to environment_files/")
    ap.add_argument("--output", default=None, help="output dir (default: harness_runs/<ts>/)")
    ap.add_argument("--run-id", type=int, default=0, help="seed component for per-env determinism")
    args = ap.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[3] / "harness_runs" / ts
    else:
        out_dir = Path(args.output)

    print(f"Output dir: {out_dir}", flush=True)
    summary = run_harness(args.env_dir, args.agent, out_dir, run_id=args.run_id)

    print(f"\n=== HARNESS COMPLETE ===")
    print(f"harness_score_train   = {summary['harness_score_train']:.4f}")
    print(f"harness_score_holdout = {summary['harness_score_holdout']:.4f}")
    print(f"gate_pass             = {summary['gate_pass']}")
    print(f"total_wall_seconds    = {summary['total_wall_seconds']}")
    print(f"output_dir            = {out_dir}")


if __name__ == "__main__":
    main()
