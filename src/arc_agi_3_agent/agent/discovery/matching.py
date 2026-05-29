"""Stable object-id assignment across consecutive frames.

Greedy 3-tier nearest-centroid matching prev->cur so motion (a moved object that
keeps shape+color) is tracked rather than read as disappear+appear.
"""

from __future__ import annotations

import dataclasses

from .constants import MATCH_MAX_DIST
from .types import Object


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def stable_object_ids(
    prev: list[Object],
    cur: list[Object],
    next_id_start: int,
    max_dist: float = MATCH_MAX_DIST,
) -> tuple[list[Object], int]:
    """Return cur objects with ids carried over from prev where matched.

    Tiers (greedy, each prev/cur used once):
      1. same shape_sig AND same color
      2. same color, shape_sig differs (reshape)
      3. same shape_sig, color differs (recolor)
    Unmatched cur objects get fresh ids from next_id_start.
    """
    max_d2 = max_dist * max_dist
    used_prev: set[int] = set()
    assigned: dict[int, int] = {}   # index in cur -> id

    def pass_match(pred) -> None:
        # Build all candidate pairs satisfying pred within radius, sort by distance, greedily assign.
        pairs = []
        for ci, co in enumerate(cur):
            if ci in assigned:
                continue
            for pi, po in enumerate(prev):
                if pi in used_prev:
                    continue
                if not pred(po, co):
                    continue
                d2 = _dist2(po.centroid, co.centroid)
                if d2 <= max_d2:
                    pairs.append((d2, ci, pi))
        pairs.sort()
        for _, ci, pi in pairs:
            if ci in assigned or pi in used_prev:
                continue
            assigned[ci] = prev[pi].id
            used_prev.add(pi)

    pass_match(lambda p, c: p.shape_sig == c.shape_sig and p.color == c.color)
    pass_match(lambda p, c: p.color == c.color and p.shape_sig != c.shape_sig)
    pass_match(lambda p, c: p.shape_sig == c.shape_sig and p.color != c.color)

    out: list[Object] = []
    nxt = next_id_start
    for ci, co in enumerate(cur):
        if ci in assigned:
            out.append(dataclasses.replace(co, id=assigned[ci]))
        else:
            out.append(dataclasses.replace(co, id=nxt))
            nxt += 1
    return out, nxt
