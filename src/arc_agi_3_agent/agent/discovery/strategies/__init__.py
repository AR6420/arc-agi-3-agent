"""Pluggable exploit strategies, consulted by the decision rule in priority order."""

from __future__ import annotations

from .attribute import AttributeMatching
from .click import ClickToEffect
from .goal_probe import GoalProbe
from .movement import MovementPathfinding
from .resource_tracker import ResourceTracker
from .selection_undo import SelectionUndoTool

# Build-order / priority registry (higher priority consulted first).
ALL_STRATEGIES = {
    "resource": ResourceTracker,
    "movement": MovementPathfinding,
    "selection_undo": SelectionUndoTool,
    "click": ClickToEffect,
    "goal_probe": GoalProbe,          # Task C — additive (off by default for B-vs-C attribution)
    "attribute": AttributeMatching,
}

# Staged build order (Phase 3 Stage 4): enable cumulatively. goal_probe is NOT in the
# default order — it is added only by the Task-C agent variant (goal_by_interaction=True).
BUILD_ORDER = ["resource", "movement", "selection_undo", "click", "attribute"]


def build_strategies(enabled: list[str] | None = None):
    """Instantiate strategies for the given enabled keys (default: all), priority-sorted."""
    keys = enabled if enabled is not None else BUILD_ORDER
    strs = [ALL_STRATEGIES[k]() for k in keys]
    return sorted(strs, key=lambda s: -s.priority)
