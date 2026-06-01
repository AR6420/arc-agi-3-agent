"""GoalProbe (Task C) — goal inference by interaction, not arrival.

When the reward is not yet known, systematically INTERACT with the top
relational-ranked candidates (click them) and let the world model watch for any
reward/level signal. Candidates are TESTED, never labelled. Once a click earns
reward, ClickToEffect (higher priority) takes over and exploits it.

This attacks the v1 blocker directly: random clicking is budget-inefficient; ranking
candidates by relational goal-likeness (distinctness / centrality / containment /
match-potential) and probing the best first finds the rewarding class far sooner.

Move-onto and transformer interactions are handled by MovementPathfinding (priority 80)
and AttributeMatching (40); GoalProbe fills the pure-click discovery gap (priority 55).
"""

from __future__ import annotations

from ..constants import ACTION6
from ..interaction import rank_candidates_relational


class GoalProbe:
    name = "goal_probe"
    archetype = "interaction"
    priority = 55
    K_PER_CLASS = 2          # clicks per candidate class before demoting (budget-bounded)

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self._has_click = ACTION6 in {int(a) for a in available_actions}
        self._inter: dict[tuple[int, tuple[int, int]], int] = {}   # (level, class_key) -> clicks

    def applicable(self, wm) -> bool:
        if not self._has_click:
            return False
        # Stop probing once a rewarding click class is known — ClickToEffect exploits it.
        if wm.rewarding_click_classes():
            return False
        return wm.cur_af is not None

    def propose(self, wm) -> tuple[int, dict] | None:
        if wm.cur_af is None:
            return None
        lvl = wm.last_levels_completed()
        ctrl = wm.controllable_obj()
        cands = rank_candidates_relational(wm.objects(), ctrl, wm.cur_af.active_region)
        for c in cands:
            if c.class_key in wm.tried_goal_keys:
                continue
            key = (lvl, c.class_key)
            n = self._inter.get(key, 0)
            if n >= self.K_PER_CLASS:
                wm.mark_goal_tried(c.class_key)
                continue
            if wm.is_lethal(ACTION6, c.class_key):
                wm.mark_goal_tried(c.class_key)
                continue
            self._inter[key] = n + 1
            return ACTION6, {"x": int(c.rep_xy[0]), "y": int(c.rep_xy[1])}
        return None
