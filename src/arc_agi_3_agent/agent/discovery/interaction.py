"""Task C — relational candidate ranking for goal-inference-by-interaction.

Ranks objects as CANDIDATE goals to TEST by interaction (click / move-onto / transform),
NOT by pure saliency. The features are relational priors a human reads off the board
before knowing the rules — and crucially none of them is a semantic label:

  - distinctness : rare color + atypical size (a goal is usually visually special)
  - centrality   : closeness to the active-region centre (goal zones tend to be framed)
  - containment  : "hollow / frame-like" (bbox much larger than filled cells — a target slot)
  - match-potential: same color/shape as the controllable (a deliver-to-match target)

The strategy interacts with the top candidate and watches for ANY reward/attribute/level
signal — arrival is not assumed to be the goal. Nothing about any specific env is encoded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Object


@dataclass(frozen=True)
class RelCandidate:
    obj_id: int
    class_key: tuple[int, int]
    score: float
    rep_xy: tuple[int, int]                # (x=col, y=row) — a cell ON the object
    centroid: tuple[float, float]          # (row, col)
    match: bool                            # shares color/shape with the controllable


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def rank_candidates_relational(
    objects: list[Object],
    controllable: Object | None,
    active_region: tuple[int, int, int, int],
) -> list[RelCandidate]:
    """Rank objects by relational goal-likeness (descending). Controllable excluded."""
    ctrl_sig = controllable.shape_sig if controllable is not None else None
    cands = [o for o in objects if ctrl_sig is None or o.shape_sig != ctrl_sig]
    if not cands:
        return []

    color_counts: dict[int, int] = {}
    for o in objects:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
    med_size = _median([float(o.size) for o in objects]) or 1.0

    r0, c0, r1, c1 = active_region
    cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    half_diag = max(1.0, ((r1 - r0) ** 2 + (c1 - c0) ** 2) ** 0.5 / 2.0)

    out: list[RelCandidate] = []
    for o in cands:
        rarity = 1.0 / color_counts.get(o.color, 1)
        size_atyp = abs(o.size - med_size) / (med_size + 1.0)
        distinct = rarity + size_atyp

        br0, bc0, br1, bc1 = o.bbox
        bbox_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
        hollow = max(0.0, 1.0 - o.size / bbox_area) if bbox_area > 0 else 0.0

        dist = ((o.centroid[0] - cy) ** 2 + (o.centroid[1] - cx) ** 2) ** 0.5
        central = max(0.0, 1.0 - dist / half_diag)

        match = False
        match_score = 0.0
        if controllable is not None:
            if o.color == controllable.color:
                match_score += 0.5
                match = True
            if o.shape_sig == controllable.shape_sig:
                match_score += 0.5
                match = True

        score = 1.0 * distinct + 0.4 * central + 0.6 * hollow + 1.0 * match_score
        out.append(RelCandidate(
            obj_id=o.id, class_key=o.class_key, score=round(float(score), 4),
            rep_xy=o.rep_cell(), centroid=o.centroid, match=match,
        ))
    out.sort(key=lambda c: -c.score)
    return out
