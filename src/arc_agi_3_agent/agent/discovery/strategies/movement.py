"""MovementPathfinding — BFS/A* over learned move vectors to a goal/frontier cell."""

from __future__ import annotations

import numpy as np

from ..constants import GRID
from .pathfind import first_action_toward


class MovementPathfinding:
    name = "movement_pathfinding"
    archetype = "movement"
    priority = 80

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        pass

    def applicable(self, wm) -> bool:
        return wm.controllable_obj_id() is not None and len(wm.move_action_vectors()) >= 1

    def propose(self, wm) -> tuple[int, dict] | None:
        ctrl = wm.controllable_obj()
        if ctrl is None:
            return None
        mv = wm.move_action_vectors()
        if not mv:
            return None
        start = (int(round(ctrl.centroid[0])), int(round(ctrl.centroid[1])))
        occ = wm.occupancy()

        goal = wm.candidate_goal()
        goals: set[tuple[int, int]] = set()
        if goal is not None and goal.kind == "reach_cell" and goal.cells:
            goals = {(int(r), int(c)) for r, c in goal.cells}
        else:
            # frontier: free cells adjacent to unexplored / highest-novelty region.
            goals = self._frontier_goals(wm, occ, start)
        if not goals:
            return None
        aid = first_action_toward(start, goals, occ, mv)
        if aid is None:
            return None
        return aid, {}

    def _frontier_goals(self, wm, occ: np.ndarray, start) -> set[tuple[int, int]]:
        """Pick reachable free cells far from the avatar as exploration targets."""
        free = ~occ
        ys, xs = np.nonzero(free)
        if len(ys) == 0:
            return set()
        # candidate goals = the K free cells farthest (Manhattan) from start
        d = np.abs(ys - start[0]) + np.abs(xs - start[1])
        order = np.argsort(-d)[:8]
        return {(int(ys[i]), int(xs[i])) for i in order}
