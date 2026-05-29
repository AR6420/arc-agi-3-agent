"""MovementPathfinding — pathfind to a saliency-hypothesized goal (Task 2).

Without waiting for a first level completion, treat the top untried salient
static object as a CANDIDATE goal and route the controllable object to a free
cell adjacent to it. Reaching it without reward -> demote (mark tried) and try
the next candidate. Reward (level advance) is observed by the world model. Falls
back to novelty-frontier exploration when no candidate remains.
"""

from __future__ import annotations

import numpy as np

from ..constants import GRID
from .pathfind import first_action_toward


class MovementPathfinding:
    name = "movement_pathfinding"
    archetype = "movement"
    priority = 80

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self._pursuing: tuple[int, int] | None = None    # class_key of the candidate in pursuit

    def applicable(self, wm) -> bool:
        return wm.controllable_obj_id() is not None and len(wm.move_action_vectors()) >= 1

    def propose(self, wm) -> tuple[int, dict] | None:
        ctrl = wm.controllable_obj()
        mv = wm.move_action_vectors()
        if ctrl is None or not mv:
            return None
        start = (int(round(ctrl.centroid[0])), int(round(ctrl.centroid[1])))
        occ = wm.occupancy()

        goal = wm.candidate_goal()
        if goal is not None and goal.kind == "reach_cell" and goal.cells:
            goals = {(int(r), int(c)) for r, c in goal.cells}
            return self._step_toward(start, goals, occ, mv)

        # Saliency goal-hypothesis: pursue the top untried salient candidate.
        for cand in wm.salient_candidates():
            if cand.class_key in wm.tried_goal_keys:
                continue
            obj = wm.object_by_id(cand.obj_id)
            if obj is None:
                continue
            goals = self._free_neighbors(obj, occ)
            if not goals:
                wm.mark_goal_tried(cand.class_key)
                continue
            # reached it? (adjacent / on it) -> demote and move on
            if start in goals or self._adjacent(start, obj):
                wm.mark_goal_tried(cand.class_key)
                self._pursuing = None
                continue
            self._pursuing = cand.class_key
            act = self._step_toward(start, goals, occ, mv)
            if act is not None:
                return act
            wm.mark_goal_tried(cand.class_key)     # unreachable -> demote

        # No candidate actionable -> frontier exploration.
        goals = self._frontier_goals(occ, start)
        if not goals:
            return None
        return self._step_toward(start, goals, occ, mv)

    # ---- helpers ----------------------------------------------------------
    def _step_toward(self, start, goals, occ, mv):
        aid = first_action_toward(start, goals, occ, mv)
        return (aid, {}) if aid is not None else None

    def _free_neighbors(self, obj, occ: np.ndarray) -> set[tuple[int, int]]:
        r0, c0, r1, c1 = obj.bbox
        cells: set[tuple[int, int]] = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if 0 <= r < GRID and 0 <= c < GRID and not occ[r, c]:
                    cells.add((r, c))
        return cells

    def _adjacent(self, start, obj) -> bool:
        r0, c0, r1, c1 = obj.bbox
        return (r0 - 1 <= start[0] <= r1 + 1) and (c0 - 1 <= start[1] <= c1 + 1)

    def _frontier_goals(self, occ: np.ndarray, start) -> set[tuple[int, int]]:
        free = ~occ
        ys, xs = np.nonzero(free)
        if len(ys) == 0:
            return set()
        d = np.abs(ys - start[0]) + np.abs(xs - start[1])
        order = np.argsort(-d)[:8]
        return {(int(ys[i]), int(xs[i])) for i in order}
