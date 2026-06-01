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
from arc_agi_3_agent.agent.discovery.agent import DiscoveryAgent
from arc_agi_3_agent.agent.discovery.novelty import frame_hash
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

RESET_ACTION_ID = 0  # GameAction.RESET — revives a GAME_OVER via level_reset (death-model.md Verdict C)

# Phase 3 v2 (Task B1): the Phase 2.5 early-exit heuristics (level-1 budget cap,
# progress-window cap) are REMOVED. Death is non-terminal (death-model.md Verdict C):
# the loop now RESETs-and-continues on GAME_OVER and terminates only on WIN-all-levels
# or when the action budget (5×Σbaseline) is exhausted — faithful to the canonical
# ARC-AGI-3-Agents loop (`while not is_done() and action_counter <= MAX_ACTIONS`).


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


def dispatch_choose(agent, obs, last_frame):
    """Route choose_action by agent type.

    - BiasedRandomAgent: single (64,64) last frame.
    - DiscoveryAgent: FULL obs (needs levels_completed / state / available_actions).
    - others (EliteV0): raw obs.frame T-stack.
    """
    if isinstance(agent, BiasedRandomAgent):
        return agent.choose_action(last_frame)
    if isinstance(agent, DiscoveryAgent):
        return agent.choose_action(obs)
    return agent.choose_action(obs.frame)


def _run_episode(
    env,
    agent,
    initial_obs,
    baseline_actions: list[int],
    max_actions: int,
    game_action_cls,
    log_fh=None,
) -> dict:
    """Drive one episode under RESET-and-continue semantics (death-model.md Verdict C).

    Death is NON-terminal: on GAME_OVER the only action that makes progress is RESET
    (level_reset — revives play on the current level, preserves completed levels; any
    non-RESET action returns a frozen empty frame). We force RESET on GAME_OVER and
    continue; the episode terminates ONLY on WIN-all-levels or budget exhaustion.
    This mirrors the canonical ARC-AGI-3-Agents loop exactly.

    RESET costs 1 action against the budget/score (scorecard.inc_reset_count bumps both
    `resets` and `actions`; CLAUDE.md §2.4). We count it in `actions_taken` and attribute
    it to the current level's `level_actions` like any other action.

    Pure of arc/scorecard plumbing so it can be unit-tested with a fake env. `env` needs
    `.step(action, data=...)`; `game_action_cls` needs `.from_id(int)`.
    """
    n_levels = len(baseline_actions)
    obs = initial_obs

    action_hist: Counter[int] = Counter()
    latencies: list[float] = []
    level_actions = [0] * n_levels
    levels_completed_seen = 0
    actions_taken = 0
    resets_taken = 0
    n_deaths = 0
    first_death_step: int | None = None
    seen_hashes: set[int] = set()
    termination = "budget_exhausted"

    was_game_over = False
    while actions_taken < max_actions:
        state = str(obs.state)
        if state.endswith("WIN"):
            termination = "win"
            break  # env fully won (all levels) — the only non-death terminal.

        is_game_over = state.endswith("GAME_OVER")
        if is_game_over:
            if not was_game_over:
                n_deaths += 1
                if first_death_step is None:
                    first_death_step = actions_taken
        was_game_over = is_game_over

        last_frame = _frame_last(obs.frame)

        # Always let the agent observe (incl. the GAME_OVER death frame — that is the
        # learning signal for the life/lethality model). Its chosen action is honoured
        # unless we're in GAME_OVER, where only RESET un-freezes the env.
        t0 = time.perf_counter()
        action_id, action_data = dispatch_choose(agent, obs, last_frame)
        latencies.append(time.perf_counter() - t0)

        if is_game_over and action_id != RESET_ACTION_ID:
            action_id, action_data = RESET_ACTION_ID, {}

        ga = game_action_cls.from_id(action_id)
        obs = env.step(ga, data=action_data)
        if obs is None:
            termination = "step_none"
            break
        actions_taken += 1
        action_hist[action_id] += 1
        if action_id == RESET_ACTION_ID:
            resets_taken += 1

        cur_level = int(getattr(obs, "levels_completed", 0))
        # Attribute the action to the *current* level (the one being attempted). All the
        # death+retry actions on level k pile into level_actions[k] — the honest cost of
        # retries against that level's RHAE.
        if cur_level < n_levels:
            level_actions[cur_level] += 1
        if cur_level > levels_completed_seen:
            levels_completed_seen = cur_level

        post_frame = _frame_last(obs.frame)
        seen_hashes.add(frame_hash(post_frame))  # blake2b — process-stable distinct-state count

        if log_fh is not None:
            log_fh.write(json.dumps({
                "step": actions_taken,
                "action_id": action_id,
                "action_data": action_data,
                "state": state,
                "levels_completed": cur_level,
                "frame_T": len(obs.frame) if isinstance(obs.frame, list) else 1,
            }) + "\n")

    return {
        "obs": obs,
        "action_hist": action_hist,
        "latencies": latencies,
        "level_actions": level_actions,
        "levels_completed_seen": levels_completed_seen,
        "actions_taken": actions_taken,
        "resets_taken": resets_taken,
        "n_deaths": n_deaths,
        "first_death_step": first_death_step,
        "distinct_states": len(seen_hashes),
        "termination": termination,
    }


def run_one_env(
    arc,
    env_id: str,
    env_info,
    agent: BiasedRandomAgent,
    out_per_env_path: Path,
    run_id: int,
) -> dict:
    """Run agent on one env. Returns per-env summary dict."""
    from arcengine import GameAction

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

    t_start = time.perf_counter()
    ep = _run_episode(env, agent, obs, baseline_actions, max_actions, GameAction, log_fh=log_fh)
    wall = time.perf_counter() - t_start

    sc = arc.close_scorecard(card)
    log_fh.close()

    level_actions = ep["level_actions"]
    levels_completed_seen = ep["levels_completed_seen"]
    actions_taken = ep["actions_taken"]
    latencies = ep["latencies"]
    action_hist = ep["action_hist"]
    score = env_score_from_actions(level_actions, baseline_actions, levels_completed_seen)

    archetype_detected = None
    if isinstance(agent, DiscoveryAgent):
        archetype_detected = agent.wm.confirmed_archetype()

    # Cross-check vs OFFLINE scorecard (expected zeros per Phase 0b S5).
    sc_actions = sc_resets = 0
    if sc and sc.environments:
        run = sc.environments[0].runs[0]
        sc_actions = int(run.actions or 0)
        sc_resets = int(run.resets or 0)

    # level-1 budget context for B5 reporting (actions-to-first-death vs level-1 baseline).
    level1_baseline = baseline_actions[0] if baseline_actions else None

    return {
        "env_id": env_id,
        "score": score,
        "levels_completed": levels_completed_seen,
        "actions": actions_taken,
        "level_actions": level_actions,
        "baseline_actions": baseline_actions,
        "n_levels": n_levels,
        "max_actions": max_actions,
        "action_hist": dict(action_hist),
        "wall_seconds": round(wall, 3),
        "latency_p50_ms": round(1000 * statistics.median(latencies), 3) if latencies else None,
        "latency_p95_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 3) if len(latencies) >= 20 else None,
        "termination": ep["termination"],
        "n_deaths": ep["n_deaths"],
        "resets": ep["resets_taken"],
        "first_death_step": ep["first_death_step"],
        "level1_baseline": level1_baseline,
        "distinct_states": ep["distinct_states"],
        "early_exit_reason": None,  # retained key; no early-exit in the fixed harness
        "scorecard_actions_offline": sc_actions,
        "scorecard_resets_offline": sc_resets,
        "archetype_detected": archetype_detected,
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
    elif agent_name == "discovery_explore_only":
        agent = DiscoveryAgent(explore_only=True)
    elif agent_name == "discovery_goalprobe":
        agent = DiscoveryAgent(goal_by_interaction=True)   # Task C — aggressive ablation (GoalProbe strategy)
    elif agent_name == "discovery_relexplore":
        agent = DiscoveryAgent(relational_explore=True)    # Task C — relational explore-bias (best-of-both)
    elif agent_name == "discovery" or agent_name.startswith("discovery:"):
        # "discovery" = all strategies; "discovery:resource,movement" = staged subset.
        enabled = None
        if ":" in agent_name:
            enabled = [s for s in agent_name.split(":", 1)[1].split(",") if s]
        agent = DiscoveryAgent(enabled_strategies=enabled)
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
            f"actions={result['actions']}/{result.get('max_actions','?')} "
            f"deaths={result.get('n_deaths',0)} resets={result.get('resets',0)} "
            f"1st_death@{result.get('first_death_step')} "
            f"term={result.get('termination')} "
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
