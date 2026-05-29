"""Stage 1 — Analyser. Pure per-frame perception (no accumulated state)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.perception_input import reduce_frame_stack
from .novelty import coarse_signature, frame_hash
from .saliency import active_region
from .segmentation import flood_fill_components
from .structure import strip_extents, symmetry_flags
from .types import Object


@dataclass
class AnalysedFrame:
    grid: np.ndarray                 # (64,64) int8 — settled last frame
    motion: np.ndarray               # (64,64) int8 — max-abs-diff over the T-stack
    objects4: list[Object]
    objects8: list[Object]
    frame_hash: int
    coarse_sig: int
    available_actions: tuple[int, ...]
    levels_completed: int
    state: str
    strip_extents: dict[str, int]
    symmetry: dict[str, bool]
    active_region: tuple[int, int, int, int]

    @property
    def motion_cells(self) -> int:
        return int((self.motion != 0).sum())


def analyse(
    frame_stack,
    available_actions,
    levels_completed: int,
    state: str,
) -> AnalysedFrame:
    """Reduce the raw frame stack (OQ7) and segment the settled last frame."""
    reduced = reduce_frame_stack(frame_stack)        # (3,64,64): first,last,max-abs-diff
    grid = np.asarray(reduced[1], dtype=np.int8)
    motion = np.asarray(reduced[2], dtype=np.int8)
    return AnalysedFrame(
        grid=grid,
        motion=motion,
        objects4=flood_fill_components(grid, conn=4),
        objects8=flood_fill_components(grid, conn=8),
        frame_hash=frame_hash(grid),
        coarse_sig=coarse_signature(grid),
        available_actions=tuple(int(a) for a in (available_actions or ())),
        levels_completed=int(levels_completed),
        state=str(state),
        strip_extents=strip_extents(grid),
        symmetry=symmetry_flags(grid),
        active_region=active_region(grid),
    )
