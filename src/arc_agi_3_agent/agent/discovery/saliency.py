"""Relational saliency scoring — ranks objects as CANDIDATE hypotheses.

This assigns NO semantic labels (no a-priori "this is the goal" — that is the
Approach-A trap). It produces a ranked list of candidates for the goal-hypothesis
and interactive-probing strategies to TEST. Saliency = relational distinctness:
rare color, atypical size, spatial isolation, and dissimilarity from the
controllable object once known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Object


@dataclass(frozen=True)
class SalientCandidate:
    obj_id: int
    score: float
    color: int
    centroid: tuple[float, float]
    class_key: tuple[int, int]


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def rank_saliency(
    objects: list[Object],
    exclude_sig: int | None = None,
) -> list[SalientCandidate]:
    """Rank objects by relational saliency (descending).

    exclude_sig: shape_sig of the controllable object — excluded from candidates
    (we don't want the avatar as its own goal) and used as a dissimilarity anchor.
    """
    cands = [o for o in objects if exclude_sig is None or o.shape_sig != exclude_sig]
    if not cands:
        return []

    color_counts: dict[int, int] = {}
    for o in objects:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
    sizes = [float(o.size) for o in objects]
    med_size = _median(sizes) or 1.0

    ctrl_centroids = [o.centroid for o in objects if exclude_sig is not None and o.shape_sig == exclude_sig]

    out: list[SalientCandidate] = []
    for o in cands:
        # color rarity: rarer color -> higher
        rarity = 1.0 / color_counts.get(o.color, 1)
        # size atypicality: deviation from median (normalized)
        size_atyp = abs(o.size - med_size) / (med_size + 1.0)
        # isolation: distance to nearest OTHER object centroid
        others = [p for p in objects if p.id != o.id]
        if others:
            iso = min(((o.centroid[0] - p.centroid[0]) ** 2
                       + (o.centroid[1] - p.centroid[1]) ** 2) ** 0.5 for p in others)
        else:
            iso = 64.0
        iso_n = min(iso / 64.0, 1.0)
        # dissimilarity from controllable (encourage goal != avatar)
        if ctrl_centroids:
            dctrl = min(((o.centroid[0] - c[0]) ** 2 + (o.centroid[1] - c[1]) ** 2) ** 0.5
                        for c in ctrl_centroids)
            dctrl_n = min(dctrl / 64.0, 1.0)
        else:
            dctrl_n = 0.0
        score = 1.5 * rarity + 0.5 * size_atyp + 0.7 * iso_n + 0.5 * dctrl_n
        out.append(SalientCandidate(
            obj_id=o.id, score=round(float(score), 4), color=o.color,
            centroid=o.centroid, class_key=o.class_key,
        ))
    out.sort(key=lambda c: -c.score)
    return out


def active_region(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of non-background content (r0, c0, r1, c1), inclusive.

    Restricts option-generation/search to the meaningful play area instead of all
    4096 cells. Empty grid -> full grid.
    """
    nz = np.argwhere(grid != 0)
    if nz.size == 0:
        return (0, 0, grid.shape[0] - 1, grid.shape[1] - 1)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1))
