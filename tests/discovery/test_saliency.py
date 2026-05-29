"""Task 1 unit tests — active region + relational saliency (candidates, not labels)."""

from __future__ import annotations

import numpy as np

from arc_agi_3_agent.agent.discovery.saliency import active_region, rank_saliency
from arc_agi_3_agent.agent.discovery.segmentation import flood_fill_components


def grid():
    return np.zeros((64, 64), dtype=np.int8)


def test_active_region_bbox():
    g = grid()
    g[10:14, 20:26] = 5
    assert active_region(g) == (10, 20, 13, 25)


def test_active_region_empty_is_full():
    assert active_region(grid()) == (0, 0, 63, 63)


def test_rare_color_ranks_higher():
    g = grid()
    # three common (color 5) objects + one rare (color 9)
    g[2:4, 2:4] = 5
    g[2:4, 10:12] = 5
    g[2:4, 20:22] = 5
    g[40:42, 40:42] = 9      # rare, isolated
    objs = flood_fill_components(g)
    ranked = rank_saliency(objs)
    assert ranked[0].color == 9          # the unique, isolated object is most salient


def test_exclude_controllable_sig():
    g = grid()
    g[2:4, 2:4] = 5
    g[40:42, 40:42] = 9
    objs = flood_fill_components(g)
    ctrl_sig = objs[0].shape_sig         # both are 2x2 squares -> same shape_sig!
    ranked = rank_saliency(objs, exclude_sig=ctrl_sig)
    # both squares share shape_sig, so excluding it removes all -> empty (correct:
    # nothing is distinguishable from the controllable shape)
    assert ranked == []


def test_distinct_shape_not_excluded():
    g = grid()
    g[2:4, 2:4] = 5                       # 2x2 controllable
    g[40, 40:45] = 9                      # 1x5 bar, different shape
    objs = flood_fill_components(g)
    ctrl = next(o for o in objs if o.size == 4)
    ranked = rank_saliency(objs, exclude_sig=ctrl.shape_sig)
    assert len(ranked) == 1 and ranked[0].color == 9
