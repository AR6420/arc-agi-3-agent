"""Phase 3 v3 — per-env episode-memory boundary + unified anti-fixation.

(1) reset_for_env must RE-INSTANTIATE the world model / strategies / decider so NO learned
    episode state crosses env boundaries (only the env-agnostic procedure does).
(2) the unified decaying relational bonus must NOT fixate on the top class (the documented
    r11l failure): it spreads clicks across classes far more than the constant-bonus ablation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arc_agi_3_agent.agent.discovery.agent import DiscoveryAgent
from arc_agi_3_agent.agent.discovery.decision import DecisionRule
from arc_agi_3_agent.agent.discovery.options import ClickTarget, Options
from arc_agi_3_agent.agent.discovery.types import Object


# ---------- (1) episode-memory boundary -------------------------------------

def test_episode_memory_does_not_cross_envs():
    ag = DiscoveryAgent()
    ag.reset_for_env("envA", [1, 2, 6], run_id=0)
    wmA, decA = ag.wm, ag.decider

    # Pollute env-A episode memory with learned state of every kind.
    wmA.n_deaths = 5
    wmA.controllable_sig = 777
    wmA.move_vectors[1] = (0, 1)
    wmA.lethal.add((3, None))
    wmA._lethal_cells.add((10, 10))
    wmA.reward_click_classes.add((1, 2))
    wmA.effects[1] = object()
    wmA.goal = object()
    decA._class_clicks[(1, 2)] = 9

    ag.reset_for_env("envB", [1, 2, 6], run_id=0)
    wmB, decB = ag.wm, ag.decider

    assert wmB is not wmA and decB is not decA          # re-instantiated, not shared
    assert wmB.n_deaths == 0
    assert wmB.controllable_sig is None
    assert wmB.move_vectors == {}
    assert wmB.lethal == set() and wmB.lethal_cells() == set()
    assert wmB.reward_click_classes == set()
    assert wmB.effects == {}
    assert wmB.goal is None
    assert decB._class_clicks == {}


# ---------- (2) unified anti-fixation ---------------------------------------

def _obj(oid, color, r, c, h=1, w=1):
    mask = np.ones((h, w), dtype=bool)
    return Object(id=oid, color=color, size=int(mask.sum()), bbox=(r, c, r + h - 1, c + w - 1),
                  centroid=(r + (h - 1) / 2, c + (w - 1) / 2), mask=mask,
                  shape_sig=1000 + oid, pose_sig=0, conn=4)


class _StubWM:
    """Two click classes, no reward yet, action 1 + ACTION6 available, no controllable."""
    def __init__(self):
        g = np.zeros((64, 64), dtype=np.int8)
        g[20, 20] = 7        # rare/central -> rank 0
        g[5, 5] = 4
        self._g = g
        self.cur_af = SimpleNamespace(grid=g, active_region=(0, 0, 63, 63))
        self._objs = [_obj(1, 7, 20, 20), _obj(2, 4, 5, 5)]
        self._cls = [o.class_key for o in self._objs]
        self._cts = [ClickTarget(xy=(20, 20), class_key=self._objs[0].class_key, obj_id=1),
                     ClickTarget(xy=(5, 5), class_key=self._objs[1].class_key, obj_id=2)]

    def options(self):
        return Options(action_ids=[1, 6], click_targets=list(self._cts))
    def unknown_actions(self): return []
    def known_noop_actions(self): return set()
    def is_lethal(self, a, ck=None): return False
    def rewarding_click_classes(self): return set()
    def objects(self): return list(self._objs)
    def controllable_obj(self): return None
    def step_index(self): return 0
    def actions_remaining_estimate(self): return None


def _click_class_share(unified: bool, constant: bool, n=300):
    dr = DecisionRule([], explore_only=False, relational_explore=constant, unified=unified)
    dr.reset(np.random.default_rng(0))
    wm = _StubWM()
    top_cls = wm._objs[0].class_key  # rank-0 (rare + central)
    top_hits = 0
    click_hits = 0
    for _ in range(n):
        aid, data = dr._explore(wm)
        if aid == 6:
            click_hits += 1
            if (data["x"], data["y"]) == (20, 20):
                top_hits += 1
    return top_hits / max(click_hits, 1)


def test_unified_spreads_clicks_constant_fixates():
    unified_share = _click_class_share(unified=True, constant=False)
    constant_share = _click_class_share(unified=False, constant=True)
    # the constant bonus concentrates on the top class; the decaying unified bonus spreads.
    assert unified_share < constant_share
    # and unified still PROBES the top class early (not ignored).
    assert unified_share > 0.1
