"""Task A — empirical death-model probe.

Drives an env to GAME_OVER, then issues RESET and verifies whether play
continues (Verdict C) or the env is truly terminal (Verdict T). Also checks
that a non-RESET action while in GAME_OVER returns an empty (frozen) frame, and
whether levels_completed is preserved across a reset.
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("ARC_API_KEY", "noop")

from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, _get_arcade


def frame_len(obs) -> int:
    f = getattr(obs, "frame", None)
    if f is None:
        return -1
    return len(f) if isinstance(f, list) else 1


def probe(base_id: str, budget: int, seed: int) -> None:
    from arcengine import GameAction
    rng = random.Random(seed)
    arc = _get_arcade(DEFAULT_ENV_DIR)
    by_base = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    info = by_base[base_id]
    avail = None

    card = arc.open_scorecard(tags=[f"death_probe_{base_id}"])
    env = arc.make(info.game_id, scorecard_id=card)
    obs = env.reset()
    avail = [a for a in (obs.available_actions or []) if a != 0] or [1]

    print(f"\n=== {base_id} ===  initial state={obs.state} lvls={obs.levels_completed} avail={obs.available_actions}")
    gameovers = 0
    first_go_step = None
    revived = False
    max_lvl = 0
    states_seq = []

    step = 0
    while step < budget:
        st = str(obs.state)
        states_seq.append(st.split(".")[-1])
        if st.endswith("WIN"):
            print(f"  WIN at step {step}, lvls={obs.levels_completed}")
            break
        if st.endswith("GAME_OVER"):
            gameovers += 1
            if first_go_step is None:
                first_go_step = step
                lvls_at_death = int(obs.levels_completed)
                # (a) try a NON-reset action while in GAME_OVER -> expect frozen/empty frame
                test_a = rng.choice([a for a in avail if a != 6] or avail)
                obs_frozen = env.step(GameAction.from_id(test_a), data={})
                print(f"  [death@{step}] lvls={lvls_at_death}; non-reset ACTION{test_a} while GAME_OVER "
                      f"-> state={obs_frozen.state} frame_len={frame_len(obs_frozen)}")
                # (b) now RESET -> expect revival
                obs = env.step(GameAction.from_id(0), data={})
                print(f"  [reset after death] -> state={obs.state} lvls={obs.levels_completed} "
                      f"full_reset={getattr(obs,'full_reset',None)} frame_len={frame_len(obs)}")
                revived = str(obs.state).endswith("NOT_FINISHED") or str(obs.state).endswith("NOT_PLAYED")
                step += 2
                continue
            else:
                # subsequent deaths: reset and continue
                obs = env.step(GameAction.from_id(0), data={})
                step += 1
                continue
        a = rng.choice(avail)
        data = {}
        if a == 6:
            data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
        obs = env.step(GameAction.from_id(a), data=data)
        if obs is None:
            print(f"  step None at {step}"); break
        max_lvl = max(max_lvl, int(getattr(obs, "levels_completed", 0)))
        step += 1

    arc.close_scorecard(card)
    # post-revival sanity: did distinct states appear after the first reset?
    post = states_seq[first_go_step:] if first_go_step is not None else []
    print(f"  SUMMARY: first_GAME_OVER@{first_go_step} revived_after_reset={revived} "
          f"total_gameovers={gameovers} max_levels={max_lvl} "
          f"states_after_first_death={len(set(post))}_distinct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="r11l,ls20")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for e in args.envs.split(","):
        probe(e.strip(), args.budget, args.seed)


if __name__ == "__main__":
    main()
