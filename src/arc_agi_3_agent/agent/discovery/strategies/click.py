"""ClickToEffect — repeat clicks on object classes whose clicks earned reward."""

from __future__ import annotations

from ..constants import ACTION6


class ClickToEffect:
    name = "click_to_effect"
    archetype = "click"
    priority = 60

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self.has_click = ACTION6 in {int(a) for a in available_actions}

    def applicable(self, wm) -> bool:
        return self.has_click and len(wm.rewarding_click_classes()) > 0

    def propose(self, wm) -> tuple[int, dict] | None:
        rewarding = wm.rewarding_click_classes()
        if not rewarding:
            return None
        # Prefer the largest current object whose class previously earned reward.
        cands = [o for o in wm.objects() if o.class_key in rewarding]
        if not cands:
            return None
        obj = max(cands, key=lambda o: o.size)
        x, y = obj.rep_cell()
        return ACTION6, {"x": int(x), "y": int(y)}
