"""Substrate-progress probe: can the discovery agent complete a level at all,
given a large action budget and no early-exit? Isolates substrate-can-progress
from the local 5x-baseline scoring budget.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ARC_API_KEY", "noop")

from arc_agi_3_agent.agent.discovery.agent import DiscoveryAgent
from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, _get_arcade, dispatch_choose, _frame_last


def probe(base_id: str, budget: int, explore_only: bool) -> None:
    from arcengine import GameAction
    arc = _get_arcade(DEFAULT_ENV_DIR)
    by_base = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    info = by_base[base_id]
    card = arc.open_scorecard(tags=[f"probe_{base_id}"])
    env = arc.make(info.game_id, scorecard_id=card)
    obs = env.reset()
    agent = DiscoveryAgent(explore_only=explore_only)
    agent.reset_for_env(info.game_id, list(obs.available_actions or []), run_id=0)

    seen = set()
    max_level = 0
    for step in range(budget):
        st = str(obs.state)
        if st.endswith("WIN") or st.endswith("GAME_OVER"):
            print(f"  terminal {st} at step {step}")
            break
        a, d = dispatch_choose(agent, obs, _frame_last(obs.frame))
        obs = env.step(GameAction.from_id(a), data=d)
        if obs is None:
            print("  step None"); break
        seen.add(hash(_frame_last(obs.frame).tobytes()))
        lc = int(getattr(obs, "levels_completed", 0))
        if lc > max_level:
            max_level = lc
            print(f"  LEVEL {lc} reached at step {step+1}")
    arc.close_scorecard(card)
    print(f"[{base_id}] explore_only={explore_only} budget={budget} "
          f"max_level={max_level} distinct_states={len(seen)} "
          f"ctrl={agent.wm.controllable_sig is not None} "
          f"reward_clicks={len(agent.wm.rewarding_click_classes())} "
          f"archetype={agent.wm.confirmed_archetype()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="r11l,sk48,tn36,vc33,lp85")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--explore-only", action="store_true")
    args = ap.parse_args()
    for e in args.envs.split(","):
        probe(e.strip(), args.budget, args.explore_only)


if __name__ == "__main__":
    main()
