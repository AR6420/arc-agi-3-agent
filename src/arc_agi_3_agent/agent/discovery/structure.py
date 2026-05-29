"""Candidate structure detection — monotone indicators (budget/lives bars) + symmetry.

Everything here is flagged as a CANDIDATE; the world model confirms by observing
deltas over time. Nothing is interpreted from appearance alone.
"""

from __future__ import annotations

import numpy as np

from .constants import BG


def strip_extents(grid: np.ndarray) -> dict[str, list[int]]:
    """Per-row and per-column non-background cell counts (candidate indicator extents).

    Returns {"row_i": count, "col_j": count} flattened into a dict keyed by a
    stable string id so the world model can track each strip's history.
    """
    nonbg = (grid != BG)
    out: dict[str, list[int]] = {}
    row_counts = nonbg.sum(axis=1)
    col_counts = nonbg.sum(axis=0)
    for i, c in enumerate(row_counts):
        out[f"row_{i}"] = int(c)
    for j, c in enumerate(col_counts):
        out[f"col_{j}"] = int(c)
    return out


def symmetry_flags(grid: np.ndarray) -> dict[str, bool]:
    """Whole-board symmetry candidates (for constraint-template style envs)."""
    return {
        "h_flip": bool(np.array_equal(grid, grid[:, ::-1])),
        "v_flip": bool(np.array_equal(grid, grid[::-1, :])),
        "rot180": bool(np.array_equal(grid, grid[::-1, ::-1])),
    }
