"""Isolation unit tests for the discovery substrate (Stage 1 + Stage 2).

Hand-built numpy frames, no SDK. Must all pass before any harness run.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc_agi_3_agent.agent.discovery.analyser import analyse
from arc_agi_3_agent.agent.discovery.change import diff_frames
from arc_agi_3_agent.agent.discovery.matching import stable_object_ids
from arc_agi_3_agent.agent.discovery.novelty import coarse_signature, frame_hash
from arc_agi_3_agent.agent.discovery.options import generate_options
from arc_agi_3_agent.agent.discovery.segmentation import (
    canonical_shape_signature, flood_fill_components, pose_signature,
)
from arc_agi_3_agent.agent.discovery.world_model import WorldModel


def grid() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.int8)


class FakeObs:
    def __init__(self, g, available_actions=(1, 2, 3, 4), levels_completed=0, state="NOT_FINISHED"):
        self.frame = [np.asarray(g, dtype=np.int8)]
        self.available_actions = list(available_actions)
        self.levels_completed = levels_completed
        self.state = state


# ----------------------------- segmentation -----------------------------
class TestSegmentation:
    def test_single_square_one_object(self):
        g = grid(); g[5:9, 5:9] = 7
        objs = flood_fill_components(g, conn=4)
        assert len(objs) == 1
        o = objs[0]
        assert o.color == 7 and o.size == 16
        assert o.bbox == (5, 5, 8, 8)
        assert o.centroid == (6.5, 6.5)

    def test_background_never_emitted(self):
        assert flood_fill_components(grid(), conn=4) == []

    def test_diagonal_touch_4conn_vs_8conn(self):
        g = grid(); g[5, 5] = 3; g[6, 6] = 3
        assert len(flood_fill_components(g, conn=4)) == 2
        assert len(flood_fill_components(g, conn=8)) == 1

    def test_translation_same_shape_sig(self):
        a = grid(); a[2:4, 2:4] = 5
        b = grid(); b[10:12, 20:22] = 5
        oa = flood_fill_components(a)[0]
        ob = flood_fill_components(b)[0]
        assert oa.shape_sig == ob.shape_sig
        assert oa.pose_sig == ob.pose_sig   # same orientation

    def test_rotation_same_shapesig_diff_posesig(self):
        # L-tromino vs its 90-deg rotation
        m = np.array([[1, 0], [1, 1]], dtype=bool)
        r = np.rot90(m)
        assert canonical_shape_signature(m) == canonical_shape_signature(r)
        assert pose_signature(m) != pose_signature(r)


# ------------------------------- matching -------------------------------
class TestMatching:
    def test_translate_keeps_id_moved(self):
        a = grid(); a[2:4, 2:4] = 5
        b = grid(); b[2:4, 1:3] = 5          # moved left 1
        oa = flood_fill_components(a)
        oa, _ = stable_object_ids([], oa, 0)
        ob = flood_fill_components(b)
        ob, _ = stable_object_ids(oa, ob, 1)
        assert ob[0].id == oa[0].id

    def test_appear_disappear(self):
        a = flood_fill_components((lambda g: (g.__setitem__((slice(2, 4), slice(2, 4)), 5), g)[1])(grid()))
        a, nid = stable_object_ids([], a, 0)
        # next frame empty -> object disappears, no cur objects
        b, nid = stable_object_ids(a, [], nid)
        assert b == []


# ------------------------------- change ---------------------------------
class TestChange:
    def test_noop(self):
        g = grid(); g[3:5, 3:5] = 4
        objs, nid = stable_object_ids([], flood_fill_components(g), 0)
        d = diff_frames(objs, objs, g, g, 0)
        assert d.is_noop and d.changed_cells == 0

    def test_cosmetic_recolor(self):
        a = grid(); a[10, 10] = 5
        b = grid(); b[10, 10] = 6
        oa, nid = stable_object_ids([], flood_fill_components(a), 0)
        ob, nid = stable_object_ids(oa, flood_fill_components(b), nid)
        d = diff_frames(oa, ob, a, b, 0)
        assert d.cosmetic and not d.is_noop
        assert any(x.kind == "recolored" for x in d.object_deltas)

    def test_moved_translation(self):
        a = grid(); a[2:4, 5:7] = 8
        b = grid(); b[2:4, 4:6] = 8           # left 1
        oa, nid = stable_object_ids([], flood_fill_components(a), 0)
        ob, nid = stable_object_ids(oa, flood_fill_components(b), nid)
        d = diff_frames(oa, ob, a, b, 0)
        mv = d.moved()
        assert len(mv) == 1 and mv[0].translation == (0, -1)


# ------------------------------- options --------------------------------
class TestOptions:
    def test_one_target_per_class(self):
        g = grid()
        g[2:4, 2:4] = 5      # class A
        g[2:4, 10:12] = 5    # class A again (same shape+color)
        g[20:22, 20:22] = 6  # class B
        af = analyse([g], available_actions=(6,), levels_completed=0, state="NOT_FINISHED")
        opts = generate_options(af)
        assert len(opts.click_targets) == 2   # two classes, not three objects

    def test_no_clicks_when_action6_absent(self):
        g = grid(); g[2:4, 2:4] = 5
        af = analyse([g], available_actions=(1, 2, 3, 4), levels_completed=0, state="NOT_FINISHED")
        assert generate_options(af).click_targets == []


# ------------------------------- novelty --------------------------------
class TestNovelty:
    def test_frame_hash_stable_and_distinct(self):
        a = grid(); a[1, 1] = 1
        b = grid(); b[1, 2] = 1
        assert frame_hash(a) == frame_hash(a.copy())
        assert frame_hash(a) != frame_hash(b)

    def test_coarse_signature_stable(self):
        a = grid(); a[1, 1] = 1
        assert coarse_signature(a) == coarse_signature(a.copy())


# ----------------------------- world model ------------------------------
class TestWorldModel:
    def _move_block_left(self, start_col):
        g = grid(); g[30:32, start_col:start_col + 2] = 5
        return g

    def test_controllable_and_move_vector(self):
        wm = WorldModel()
        wm.reset("test", (1, 2, 3, 4), 0)
        col = 30
        wm.observe(FakeObs(self._move_block_left(col)))
        for _ in range(3):
            wm.record_decision(3, {})           # ACTION3 = left
            col -= 1
            wm.observe(FakeObs(self._move_block_left(col)))
        assert wm.controllable_sig is not None
        assert wm.move_action_vectors().get(3) == (0, -1)
        assert wm.confirmed_archetype() == "movement"

    def test_noop_action_detected(self):
        wm = WorldModel()
        wm.reset("test", (1, 2, 3, 4), 0)
        g = self._move_block_left(30)
        wm.observe(FakeObs(g))
        for _ in range(3):
            wm.record_decision(2, {})           # ACTION2 does nothing (same frame)
            wm.observe(FakeObs(g))
        assert 2 in wm.known_noop_actions()

    def test_budget_resource_detected(self):
        wm = WorldModel()
        wm.reset("test", (1,), 0)
        # a depleting bar in row 63: 12 cells shrinking by 1 each step
        def bar(n):
            g = grid(); g[63, 0:n] = 4; return g
        wm.observe(FakeObs(bar(12), available_actions=(1,)))
        for n in range(11, 5, -1):
            wm.record_decision(1, {})
            wm.observe(FakeObs(bar(n), available_actions=(1,)))
        kinds = {r.kind for r in wm.resources()}
        assert "budget" in kinds

    def test_goal_on_level_advance(self):
        wm = WorldModel()
        wm.reset("test", (1, 2, 3, 4), 0)
        col = 30
        wm.observe(FakeObs(self._move_block_left(col)))
        # build controllable first
        for _ in range(2):
            wm.record_decision(3, {}); col -= 1
            wm.observe(FakeObs(self._move_block_left(col)))
        # level advance
        wm.record_decision(3, {}); col -= 1
        wm.observe(FakeObs(self._move_block_left(col), levels_completed=1))
        assert wm.candidate_goal() is not None
        assert wm.candidate_goal().kind == "match_attr"

    def test_effect_filled_next_step(self):
        wm = WorldModel()
        wm.reset("test", (1, 2, 3, 4), 0)
        wm.observe(FakeObs(self._move_block_left(30)))
        wm.record_decision(3, {})
        assert wm.memory[-1].observed_delta is None     # not yet observed
        wm.observe(FakeObs(self._move_block_left(29)))
        assert wm.memory[-1].observed_delta is not None  # filled on next observe


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
