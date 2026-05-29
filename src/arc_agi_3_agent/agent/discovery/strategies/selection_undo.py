"""SelectionUndoTool — cross-cutting helper (priority 60).

Exposes select/undo helpers for click/attribute. Standalone it rarely proposes;
it mostly characterizes which action selects and whether ACTION7 undoes.
"""

from __future__ import annotations

from ..constants import ACTION7


class SelectionUndoTool:
    name = "selection_undo"
    archetype = "selection_undo"
    priority = 60

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self.available = tuple(int(a) for a in available_actions)

    def applicable(self, wm) -> bool:
        # Only standalone-applicable if undo is confirmed and could revert a bad probe;
        # in v0 we let click/attribute drive, so decline by default.
        return False

    def propose(self, wm) -> tuple[int, dict] | None:
        return None

    @staticmethod
    def undo() -> tuple[int, dict]:
        return ACTION7, {}
