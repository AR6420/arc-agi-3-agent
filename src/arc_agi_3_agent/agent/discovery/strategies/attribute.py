"""AttributeMatching — drive a controlled entity's attributes toward a target.

v0 greedy: if a target attribute tuple is known and the carried entity does not
match it, repeat a transformer action that is known to mutate attributes. A full
modular-cycle search is deferred; this nudges toward the target and is measured
behind the lower priority so it never overrides confirmed movement/click exploits.
"""

from __future__ import annotations


class AttributeMatching:
    name = "attribute_matching"
    archetype = "attribute"
    priority = 40

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self.available = tuple(int(a) for a in available_actions)

    def applicable(self, wm) -> bool:
        goal = wm.candidate_goal()
        if goal is None or goal.kind != "match_attr" or goal.target_attr is None:
            return False
        cur = wm.carried_attr()
        if cur is None:
            return False
        return cur != goal.target_attr and len(wm.transformer_events()) > 0

    def propose(self, wm) -> tuple[int, dict] | None:
        goal = wm.candidate_goal()
        cur = wm.carried_attr()
        if goal is None or cur is None or cur == goal.target_attr:
            return None
        events = wm.transformer_events()
        if not events:
            return None
        # Greedy: repeat the most recently observed transformer trigger action.
        ev = events[-1]
        aid = ev.trigger_action
        if aid not in self.available or aid == 0:
            return None
        data = {}
        if aid == 6 and ev.trigger_xy is not None:
            data = {"x": int(ev.trigger_xy[0]), "y": int(ev.trigger_xy[1])}
        elif aid == 6:
            return None
        return aid, data
