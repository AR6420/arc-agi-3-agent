"""Stage 5 — ls20 rule-recovery grader (SEPARATE from the agent).

The ls20 answer key lives ONLY in this file and is NEVER seen by the agent.
Runs the discovery agent on ls20, dumps its learned world model, and checks
whether the env-agnostic machinery independently re-derived ls20's mechanics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "noop")

from arc_agi_3_agent.agent.discovery.agent import DiscoveryAgent
from arc_agi_3_agent.agents.random_agent import max_actions_for_env
from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, _get_arcade, _run_episode

REPO_ROOT = Path(__file__).resolve().parents[3]

# --- ANSWER KEY (held outside the agent) --------------------------------------
# ls20 (per user hand-decode): a carried object's shape+color must match a goal;
# white cells transform it; one resource (yellow bar) depletes per action and
# refills at yellow squares; red segments are lives -> game over at zero.
LS20_KEY = {
    "transformer_changes_attr": "an interaction changes a tracked object's shape/color",
    "goal_has_target_attrs": "the goal is defined by target attributes (not a cell)",
    "reward_on_match": "level advances when attributes match the target",
    "resource_depletes": "a monotone resource depletes with actions",
    "resource_or_lives_terminal": "a resource/lives counter ends the episode at zero",
}


def run_ls20(budget: int | None = None, run_id: int = 0, goal_by_interaction: bool = True):
    """Run the discovery agent on ls20 under the FIXED RESET-and-continue harness loop
    (death is non-terminal — death-model.md Verdict C) so the agent gets its full
    5×Σbaseline budget of retries to re-derive ls20's mechanics. Returns the world model."""
    from arcengine import GameAction
    arc = _get_arcade(DEFAULT_ENV_DIR)
    by_base = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    info = by_base["ls20"]
    baseline = list(info.baseline_actions or [])
    max_actions = budget if budget is not None else (max_actions_for_env(baseline) if baseline else 1500)
    card = arc.open_scorecard(tags=["ls20_recovery"])
    env = arc.make(info.game_id, scorecard_id=card)
    obs = env.reset()
    agent = DiscoveryAgent(goal_by_interaction=goal_by_interaction)
    agent.reset_for_env(info.game_id, list(obs.available_actions or []), run_id=run_id)
    _run_episode(env, agent, obs, baseline, max_actions, GameAction)
    arc.close_scorecard(card)
    return agent.wm


def grade(wm) -> dict:
    dump = wm.dump_learned_model()
    checks = {
        "transformer_changes_attr": dump["n_transformer_events"] > 0,
        "goal_has_target_attrs": dump["goal"] is not None and dump["goal"].get("target_attr") is not None,
        "reward_on_match": dump["reward_events"] > 0,
        "resource_depletes": any(r["monotone_per_action"] for r in dump["resources"]),
        "resource_or_lives_terminal": any(r["depletes_to_terminal"] for r in dump["resources"]),
    }
    recovered = sum(1 for v in checks.values() if v)
    return {
        "checks": {k: {"recovered": checks[k], "key": LS20_KEY[k]} for k in checks},
        "recovered": recovered,
        "total": len(checks),
        "learned_dump": dump,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None, help="default: full 5×Σbaseline")
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--output", default=str(REPO_ROOT / "harness_runs" / "p3_ls20_recovery"))
    args = ap.parse_args()
    wm = run_ls20(args.budget, args.run_id)
    result = grade(wm)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery_scorecard.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ls20 rule-recovery: {result['recovered']}/{result['total']}")
    for k, v in result["checks"].items():
        print(f"  [{'X' if v['recovered'] else ' '}] {k}: {v['key']}")
    print(f"learned: {json.dumps(result['learned_dump'], indent=0)}")


if __name__ == "__main__":
    main()
