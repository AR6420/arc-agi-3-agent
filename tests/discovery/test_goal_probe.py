"""Task C — relational ranking + GoalProbe interaction-discovery."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arc_agi_3_agent.agent.discovery.interaction import rank_candidates_relational
from arc_agi_3_agent.agent.discovery.strategies.goal_probe import GoalProbe
from arc_agi_3_agent.agent.discovery.types import Object


def _obj(oid, color, r, c, h=1, w=1, shape_sig=None):
    mask = np.ones((h, w), dtype=bool)
    return Object(id=oid, color=color, size=int(mask.sum()),
                  bbox=(r, c, r + h - 1, c + w - 1),
                  centroid=(r + (h - 1) / 2, c + (w - 1) / 2),
                  mask=mask, shape_sig=shape_sig if shape_sig is not None else (1000 + oid),
                  pose_sig=0, conn=4)


def test_relational_ranks_distinct_over_common():
    # three common blue blocks + one rare red central block -> red ranks first.
    objs = [_obj(1, 4, 2, 2), _obj(2, 4, 2, 40), _obj(3, 4, 40, 2),
            _obj(4, 7, 20, 20)]  # rare color, central
    region = (0, 0, 63, 63)
    ranked = rank_candidates_relational(objs, controllable=None, active_region=region)
    assert ranked[0].obj_id == 4


def test_relational_match_potential_bumps_same_color():
    ctrl = _obj(99, 5, 1, 1, shape_sig=555)
    objs = [_obj(1, 8, 10, 10), _obj(2, 5, 30, 30)]  # obj2 shares the controllable's color
    ranked = rank_candidates_relational(objs, controllable=ctrl, active_region=(0, 0, 63, 63))
    top = ranked[0]
    assert top.obj_id == 2 and top.match is True


class _StubWM:
    def __init__(self, objs, avail_click=True):
        self.tried_goal_keys = set()
        self._objs = objs
        self._rewarding = set()
        self.cur_af = SimpleNamespace(active_region=(0, 0, 63, 63))
        self._click = avail_click

    def rewarding_click_classes(self):
        return set(self._rewarding)

    def last_levels_completed(self):
        return 0

    def controllable_obj(self):
        return None

    def objects(self):
        return list(self._objs)

    def mark_goal_tried(self, ck):
        self.tried_goal_keys.add(ck)

    def is_lethal(self, aid, ck=None):
        return False


def test_goal_probe_clicks_then_demotes_then_stops_on_reward():
    objs = [_obj(1, 7, 20, 20), _obj(2, 4, 5, 5)]
    wm = _StubWM(objs)
    gp = GoalProbe()
    gp.reset("env", [1, 6], 0)

    assert gp.applicable(wm)
    # K_PER_CLASS clicks per class, then it demotes that class.
    first = gp.propose(wm)
    assert first is not None and first[0] == 6           # clicks a candidate
    seen_classes = set()
    for _ in range(10):
        p = gp.propose(wm)
        if p is None:
            break
        assert p[0] == 6
        seen_classes.add((p[1]["x"], p[1]["y"]))
    # after exhausting all candidates (K each), every class is marked tried.
    assert len(wm.tried_goal_keys) == 2
    assert gp.propose(wm) is None

    # once a rewarding click class is known, GoalProbe yields to ClickToEffect.
    wm._rewarding.add(objs[0].class_key)
    assert not gp.applicable(wm)


def test_goal_probe_inactive_without_click():
    gp = GoalProbe()
    gp.reset("env", [1, 2, 3], 0)        # no ACTION6
    wm = _StubWM([_obj(1, 7, 20, 20)], avail_click=False)
    assert not gp.applicable(wm)
