"""ResourceTracker — cross-cutting budget/lives guard (priority 100).

Always "applicable" as a guard, but only PROPOSES a non-None action when a
depletes-to-terminal resource is near its floor and a refill route exists. It
otherwise centralizes the budget-pressure signal the decision rule reads.
"""

from __future__ import annotations

from .pathfind import first_action_toward


class ResourceTracker:
    name = "resource_tracker"
    archetype = "resource"
    priority = 100

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        pass

    def applicable(self, wm) -> bool:
        return True

    def propose(self, wm) -> tuple[int, dict] | None:
        # Only act when we have a learned budget near floor AND refill cells known.
        rem = wm.actions_remaining_estimate()
        if rem is None or rem > 3:
            return None
        ctrl = wm.controllable_obj()
        if ctrl is None:
            return None
        refill_cells: set[tuple[int, int]] = set()
        for r in wm.resources():
            for cell in r.refill_cells:
                refill_cells.add(cell)
        if not refill_cells:
            return None
        start = (int(round(ctrl.centroid[0])), int(round(ctrl.centroid[1])))
        aid = first_action_toward(start, refill_cells, wm.occupancy(), wm.move_action_vectors())
        if aid is None:
            return None
        return aid, {}

    # Pressure helper the decision rule consults (0 relaxed .. 1 critical).
    @staticmethod
    def pressure(wm) -> float:
        rem = wm.actions_remaining_estimate()
        if rem is None:
            return 0.0
        if rem <= 1:
            return 1.0
        if rem >= 50:
            return 0.0
        return max(0.0, 1.0 - rem / 50.0)
