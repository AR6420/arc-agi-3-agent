"""Shared lattice BFS over learned move vectors (used by movement + resource detours)."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..constants import GRID


def first_action_toward(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    occupancy: np.ndarray,
    move_vectors: dict[int, tuple[int, int]],
    max_nodes: int = 4096,
) -> int | None:
    """BFS from `start` (row,col) over edges = learned move vectors. Returns the
    first action_id on a shortest path to any goal cell, or None if unreachable.

    A move is blocked if the destination cell is occupied (True in `occupancy`).
    """
    if not goals or not move_vectors:
        return None
    if start in goals:
        return None
    seen = {start}
    # queue holds (cell, first_action)
    q: deque[tuple[tuple[int, int], int]] = deque()
    for aid, (dy, dx) in move_vectors.items():
        nxt = (start[0] + dy, start[1] + dx)
        if _free(nxt, occupancy):
            q.append((nxt, aid))
            seen.add(nxt)
            if nxt in goals:
                return aid
    nodes = 0
    while q and nodes < max_nodes:
        cell, first = q.popleft()
        nodes += 1
        if cell in goals:
            return first
        for dy, dx in move_vectors.values():
            nxt = (cell[0] + dy, cell[1] + dx)
            if nxt not in seen and _free(nxt, occupancy):
                seen.add(nxt)
                if nxt in goals:
                    return first
                q.append((nxt, first))
    return None


def _free(cell: tuple[int, int], occ: np.ndarray) -> bool:
    r, c = cell
    if r < 0 or r >= GRID or c < 0 or c >= GRID:
        return False
    return not bool(occ[r, c])
