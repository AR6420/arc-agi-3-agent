"""Strategy protocol. Each exploit is applicable()+propose() against the world model."""

from __future__ import annotations

from typing import Protocol


class Strategy(Protocol):
    name: str
    archetype: str
    priority: int

    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None: ...

    def applicable(self, wm) -> bool:
        """Fast, side-effect-free trigger predicate (flips explore->exploit)."""
        ...

    def propose(self, wm) -> tuple[int, dict] | None:
        """Return (action_id, data) or None to decline this step."""
        ...
