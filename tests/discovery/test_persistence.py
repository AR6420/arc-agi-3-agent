"""Task 5 — cross-level rule persistence: the learned model survives a level advance."""

from __future__ import annotations

import numpy as np

from arc_agi_3_agent.agent.discovery.world_model import WorldModel


class FakeObs:
    def __init__(self, g, aa=(1, 2, 3, 4), lc=0, st="NOT_FINISHED"):
        self.frame = [np.asarray(g, dtype=np.int8)]
        self.available_actions = list(aa)
        self.levels_completed = lc
        self.state = st


def grid():
    return np.zeros((64, 64), dtype=np.int8)


def block(col):
    g = grid(); g[30:32, col:col + 2] = 5
    return g


def test_effects_and_controllable_survive_level_advance():
    wm = WorldModel()
    wm.reset("t", (1, 2, 3, 4), 0)
    col = 30
    wm.observe(FakeObs(block(col)))
    for _ in range(3):
        wm.record_decision(3, {}); col -= 1
        wm.observe(FakeObs(block(col)))
    assert wm.controllable_sig is not None
    sig_before = wm.controllable_sig
    mv_before = dict(wm.move_action_vectors())
    n_effects_before = len(wm.all_effects())

    # level advance
    wm.record_decision(3, {}); col -= 1
    wm.observe(FakeObs(block(col), lc=1))

    # learned action->effect + controllable persist across the level boundary
    assert wm.controllable_sig == sig_before
    assert wm.move_action_vectors() == mv_before
    assert len(wm.all_effects()) >= n_effects_before
    # layout-specific state was cleared (fresh candidate goals for the new level)
    assert wm.tried_goal_keys == set()
