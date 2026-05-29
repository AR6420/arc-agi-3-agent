"""Plain data containers for the discovery agent (no behavior beyond small helpers)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Object:
    """A connected same-color component on the board."""
    id: int                                  # stable id assigned by matching (-1 if unassigned)
    color: int                               # 1..15 (0=bg never emitted)
    size: int
    bbox: tuple[int, int, int, int]          # (r0, c0, r1, c1) inclusive
    centroid: tuple[float, float]            # (row, col)
    mask: np.ndarray                         # bool (h, w) local to bbox
    shape_sig: int                           # rotation+translation invariant
    pose_sig: int                            # translation invariant, orientation-sensitive
    conn: int                                # 4 or 8

    def rep_cell(self) -> tuple[int, int]:
        """A representative (x, y) ON the sprite for a click target (col=x, row=y)."""
        r0, c0, _, _ = self.bbox
        ys, xs = np.nonzero(self.mask)
        # pick the mask cell nearest the centroid for stability
        cy, cx = self.centroid
        j = int(np.argmin((ys + r0 - cy) ** 2 + (xs + c0 - cx) ** 2))
        return int(xs[j] + c0), int(ys[j] + r0)

    @property
    def class_key(self) -> tuple[int, int]:
        return (self.shape_sig, self.color)


@dataclass(frozen=True)
class ObjectDelta:
    kind: str                                # moved|appeared|disappeared|recolored|rotated|reshaped|static
    obj_id: int
    translation: tuple[int, int] | None = None   # (dr, dc)
    old_color: int | None = None
    new_color: int | None = None
    old_shape_sig: int | None = None
    new_shape_sig: int | None = None


@dataclass
class Delta:
    changed_cells: int
    changed_mask: np.ndarray
    object_deltas: list[ObjectDelta]
    levels_delta: int
    is_noop: bool
    cosmetic: bool
    multi_frame: bool

    def moved(self) -> list[ObjectDelta]:
        return [d for d in self.object_deltas if d.kind == "moved"]

    def attr_changes(self) -> list[ObjectDelta]:
        return [d for d in self.object_deltas if d.kind in ("recolored", "rotated", "reshaped")]


@dataclass
class TransStat:
    """Distribution of observed translation vectors for one (action, shape) pair."""
    vectors: Counter = field(default_factory=Counter)

    def add(self, v: tuple[int, int]) -> None:
        self.vectors[v] += 1

    @property
    def n(self) -> int:
        return sum(self.vectors.values())

    def mode(self) -> tuple[int, int] | None:
        if not self.vectors:
            return None
        return self.vectors.most_common(1)[0][0]

    def is_deterministic(self, frac: float) -> bool:
        if not self.vectors:
            return False
        top = self.vectors.most_common(1)[0][1]
        return top / self.n >= frac


@dataclass
class ActionEffect:
    action_id: int
    n_obs: int = 0
    n_noop: int = 0
    translations: dict[int, TransStat] = field(default_factory=dict)   # shape_sig -> stats
    appeared_colors: Counter = field(default_factory=Counter)
    disappeared_colors: Counter = field(default_factory=Counter)
    attr_change_count: int = 0
    levels_delta_count: int = 0
    is_cascade_trigger: bool = False
    is_undo: bool = False

    @property
    def noop_rate(self) -> float:
        return self.n_noop / self.n_obs if self.n_obs else 0.0


@dataclass
class TransformerEvent:
    obj_id: int
    axis: str                                # color|shape|rotation
    old: int
    new: int
    trigger_action: int
    trigger_xy: tuple[int, int] | None
    trigger_cell_color: int | None           # color of cell the object occupied (spatial trigger hint)


@dataclass
class GoalCandidate:
    kind: str                                # reach_cell | match_attr | satisfy_constraint
    cells: tuple[tuple[int, int], ...] = ()  # (row, col) targets
    target_attr: tuple[int, int] | None = None   # (shape_sig, color)
    confidence: float = 0.0


@dataclass
class ResourceSignal:
    region: tuple[int, int, int, int]
    kind: str = "candidate"                  # candidate|budget|refilling_budget|lives|rejected
    extent_history: list[int] = field(default_factory=list)
    per_action_cost: dict[int, float] = field(default_factory=dict)
    refill_events: int = 0
    refill_cells: list[tuple[int, int]] = field(default_factory=list)
    drop_only_on_failure: bool = False
    monotone_per_action: bool = False
    depletes_to_terminal: bool = False


@dataclass
class MemoryStep:
    step: int
    situation_sig: int
    options: list[int]
    n_click_targets: int
    decision: int
    click_xy: tuple[int, int] | None
    levels_completed: int
    novelty_count: int
    observed_delta: Delta | None = None      # filled in on the NEXT observe()
