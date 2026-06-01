"""B3/B4 — life-model learning + revival-discontinuity handling.

Asserts that:
  - an observed GAME_OVER records a death + learns lethality (never per-env hardcoded);
  - a RESET-revival is treated as a layout discontinuity (NOT attributed as an effect of
    action 0) so the learned model survives deaths intact (retry-with-knowledge);
  - a directional death records a positional lethal CELL (the cell moved into), not a
    blanket-lethal direction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arc_agi_3_agent.agent.discovery.types import Object
from arc_agi_3_agent.agent.discovery.world_model import WorldModel


def _obs(grid, state="NOT_FINISHED", levels=0, avail=(1, 2, 3, 5)):
    return SimpleNamespace(
        frame=[np.asarray(grid, dtype=np.int8)],
        state=f"GameState.{state}",
        levels_completed=levels,
        available_actions=list(avail),
    )


def _grid_block(r, c, color=3):
    g = np.zeros((64, 64), dtype=np.int8)
    g[r:r + 2, c:c + 2] = color
    return g


def test_death_revival_does_not_pollute_model():
    wm = WorldModel()
    wm.reset("envX", [1, 2, 3, 5], 0)

    wm.observe(_obs(_grid_block(10, 10)));         wm.record_decision(1, {})
    wm.observe(_obs(_grid_block(10, 12)));         wm.record_decision(5, {})   # 5 = fatal
    wm.observe(_obs(_grid_block(10, 12), state="GAME_OVER"))                   # death
    wm.record_decision(0, {})                                                  # agent RESETs
    wm.observe(_obs(_grid_block(10, 10)));         wm.record_decision(1, {})   # revived layout
    wm.observe(_obs(_grid_block(12, 10)));         wm.record_decision(1, {})

    assert wm.n_deaths == 1
    assert wm.n_revives == 1
    assert (5, None) in wm.lethal                  # learned the fatal action by observation
    # RESET (action 0) must NOT have an effect entry — the layout jump was not attributed.
    assert 0 not in wm.effects
    assert all(ev.trigger_action != 0 for ev in wm.transformer_events())
    # pre-death learning for action 1 survived the death (retry keeps knowledge).
    assert 1 in wm.effects


def test_directional_death_records_lethal_cell_not_whole_direction():
    wm = WorldModel()
    wm.reset("envY", [1, 2, 3, 4], 0)
    # Pretend the controllable + its action-3 vector were already learned.
    wm.controllable_sig = 999
    wm.move_vectors[3] = (0, 1)                     # action 3 moves +1 col
    ctrl = Object(id=7, color=4, size=1, bbox=(5, 5, 5, 5), centroid=(5.0, 5.0),
                  mask=np.array([[True]]), shape_sig=999, pose_sig=0, conn=4)
    wm.prev_objs = [ctrl]
    wm.prev_af = SimpleNamespace(strip_extents={}, state="GameState.NOT_FINISHED")
    wm.last_action = 3
    wm.last_click = None

    wm._record_death(None)

    assert wm.n_deaths == 1
    # cell moved into = (5, 6); blamed positionally, NOT as a blanket-lethal action.
    assert (5, 6) in wm.lethal_cells()
    assert (3, None) not in wm.lethal               # direction not frozen


def test_lethal_cells_cleared_on_new_level_kept_across_death():
    wm = WorldModel()
    wm.reset("envZ", [1, 2, 3, 4], 0)
    wm._lethal_cells.add((5, 6))
    wm._on_revive()                                 # same level, same hazard layout
    assert (5, 6) in wm.lethal_cells()              # kept across a death/revive
    wm._partial_reset_on_level()                    # new level -> stale
    assert wm.lethal_cells() == set()
