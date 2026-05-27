"""OFFLINE-mode smoke tests resolving VALIDATE-0b items 2, 3, 4, 5, 10.

Item 2: ACTION7 (undo) action accounting on bp35
Item 3: RESET cost in COMPETITION-mode-equivalent (OFFLINE proxy)
Item 4: Frame-stack semantics on env.step()
Item 5: available_actions per-turn variability
Item 10: Animation-frame handling (multi-stack-per-step vs forced-step)

Writes structured results to scripts/validation/.smoke_offline_results.json
"""

import json
import os
import sys
from pathlib import Path

os.environ["RECORDINGS_DIR"] = r"C:\Users\adars\Downloads\ARC-AGI-3\.tmp_recordings"
os.makedirs(os.environ["RECORDINGS_DIR"], exist_ok=True)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

ENV_DIR = r"C:\Users\adars\Downloads\ARC-AGI-3\arc-prize-2026-arc-agi-3\environment_files"
OUT = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.smoke_offline_results.json")

results = {}


def frame_shape(frame):
    if frame is None:
        return None
    if isinstance(frame, list):
        if not frame:
            return (0,)
        first = frame[0]
        if hasattr(first, "shape"):
            return (len(frame),) + tuple(first.shape)
        return (len(frame), len(first), len(first[0]) if first else 0)
    return getattr(frame, "shape", None)


def run():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)

    # ===== Item 5 + 4: available_actions variability + frame shape on sp80 =====
    print("\n=== Item 5+4: available_actions / frame-shape trace on sp80 ===")
    card = arc.open_scorecard(tags=["smoke_item5"])
    env = arc.make("sp80", scorecard_id=card)
    trace_step = []

    obs = env.reset()
    trace_step.append({
        "step": "reset",
        "available_actions": list(obs.available_actions),
        "frame_shape": frame_shape(obs.frame),
        "state": str(obs.state),
        "levels_completed": obs.levels_completed,
    })
    actions_seq = [
        ("ACTION1", GameAction.ACTION1, {}),
        ("ACTION6_32_32", GameAction.ACTION6, {"x": 32, "y": 32}),
        ("ACTION1", GameAction.ACTION1, {}),
        ("ACTION5", GameAction.ACTION5, {}),
        ("ACTION1", GameAction.ACTION1, {}),
    ]
    for name, ga, data in actions_seq:
        obs = env.step(ga, data=data)
        trace_step.append({
            "step": name,
            "available_actions": list(obs.available_actions) if obs else None,
            "frame_shape": frame_shape(obs.frame) if obs else None,
            "state": str(obs.state) if obs else None,
            "levels_completed": obs.levels_completed if obs else None,
        })
    sc = arc.close_scorecard(card)
    results["item_5_4_sp80_trace"] = {
        "trace": trace_step,
        "scorecard_actions": sc.environments[0].runs[0].actions if sc and sc.environments else None,
    }

    # ===== Item 4 + 10: frame-stack distribution over 500 random steps =====
    print("\n=== Item 4+10: 500-step frame-shape distribution on sp80 and lp85 ===")
    from collections import Counter
    import random
    rng = random.Random(0)
    for game in ["sp80", "lp85"]:
        card = arc.open_scorecard(tags=[f"smoke_item4_{game}"])
        env = arc.make(game, scorecard_id=card)
        obs = env.reset()
        legal = list(obs.available_actions) if obs.available_actions else [1, 2, 3, 4, 5, 6]
        shape_hist = Counter()
        state_hist = Counter()
        for i in range(500):
            a = rng.choice(legal)
            ga = GameAction.from_id(a)
            if a == 6:
                data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
            else:
                data = {}
            obs = env.step(ga, data=data)
            if obs is None:
                state_hist["NONE_RESPONSE"] += 1
                break
            fs = frame_shape(obs.frame)
            shape_hist[str(fs)] += 1
            state_hist[str(obs.state)] += 1
            if obs.available_actions:
                legal = list(obs.available_actions)
            if str(obs.state) in ("GameState.WIN", "GameState.GAME_OVER"):
                # reset for next iter (still counts toward sample size)
                obs = env.reset()
                legal = list(obs.available_actions) if obs.available_actions else [1, 2, 3, 4, 5, 6]
        sc = arc.close_scorecard(card)
        results[f"item_4_10_{game}_500step"] = {
            "frame_shape_hist": dict(shape_hist),
            "state_hist": dict(state_hist),
            "scorecard_total_actions": sc.environments[0].runs[0].actions if sc and sc.environments else None,
        }

    # ===== Item 2: ACTION7 (undo) accounting on bp35 =====
    print("\n=== Item 2: ACTION7 accounting on bp35 ===")
    card = arc.open_scorecard(tags=["smoke_item2"])
    env = arc.make("bp35", scorecard_id=card)
    obs = env.reset()
    initial_avail = list(obs.available_actions) if obs.available_actions else []
    # Take some ACTION6 clicks first (bp35 is pure click), then try ACTION7
    trace2 = [{"step": "reset", "available_actions": initial_avail, "state": str(obs.state)}]
    if 6 in initial_avail:
        obs = env.step(GameAction.ACTION6, data={"x": 30, "y": 30})
        trace2.append({"step": "ACTION6_30_30", "available_actions": list(obs.available_actions) if obs else None, "state": str(obs.state) if obs else None})
        obs = env.step(GameAction.ACTION6, data={"x": 40, "y": 40})
        trace2.append({"step": "ACTION6_40_40", "available_actions": list(obs.available_actions) if obs else None, "state": str(obs.state) if obs else None})
    # Now try undo
    if 7 in (list(obs.available_actions) if obs and obs.available_actions else []):
        obs = env.step(GameAction.ACTION7, data={})
        trace2.append({"step": "ACTION7_undo_1", "available_actions": list(obs.available_actions) if obs else None, "state": str(obs.state) if obs else None})
        obs = env.step(GameAction.ACTION7, data={})
        trace2.append({"step": "ACTION7_undo_2", "available_actions": list(obs.available_actions) if obs else None, "state": str(obs.state) if obs else None})
    sc = arc.close_scorecard(card)
    sc_run = sc.environments[0].runs[0] if sc and sc.environments else None
    results["item_2_action7_bp35"] = {
        "trace": trace2,
        "scorecard_actions": sc_run.actions if sc_run else None,
        "scorecard_level_actions": sc_run.level_actions if sc_run else None,
        "scorecard_resets": sc_run.resets if sc_run else None,
    }

    # ===== Item 3: RESET cost on sp80 =====
    print("\n=== Item 3: RESET cost on sp80 ===")
    card = arc.open_scorecard(tags=["smoke_item3"])
    env = arc.make("sp80", scorecard_id=card)
    obs = env.reset()
    # Take 5 ACTION1, then RESET
    for _ in range(5):
        obs = env.step(GameAction.ACTION1, data={})
    pre_reset_state = {"state": str(obs.state), "levels_completed": obs.levels_completed}
    obs = env.step(GameAction.RESET, data={})
    post_reset_state = {"state": str(obs.state) if obs else None, "levels_completed": obs.levels_completed if obs else None}
    # Take 2 more ACTION1
    obs = env.step(GameAction.ACTION1, data={})
    obs = env.step(GameAction.ACTION1, data={})
    sc = arc.close_scorecard(card)
    sc_run = sc.environments[0].runs[0] if sc and sc.environments else None
    results["item_3_reset_cost_sp80"] = {
        "pre_reset": pre_reset_state,
        "post_reset": post_reset_state,
        "scorecard_actions": sc_run.actions if sc_run else None,
        "scorecard_resets": sc_run.resets if sc_run else None,
        "scorecard_level_actions": sc_run.level_actions if sc_run else None,
    }

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    run()
