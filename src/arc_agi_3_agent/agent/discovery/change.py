"""Frame-to-frame change detection -> Delta.

Consumes prev/cur object lists that already carry stable ids (from matching) plus
the two last-frame grids and the reward delta. Produces a structured Delta the
world model interprets.
"""

from __future__ import annotations

import numpy as np

from .types import Delta, Object, ObjectDelta

COSMETIC_MAX_CELLS = 4    # a lone cursor-highlight recolor is at most this many cells


def _translation(prev: Object, cur: Object) -> tuple[int, int]:
    # rounded centroid delta, cross-checked against bbox top-left delta
    dr_c = cur.centroid[0] - prev.centroid[0]
    dc_c = cur.centroid[1] - prev.centroid[1]
    dr_b = cur.bbox[0] - prev.bbox[0]
    dc_b = cur.bbox[1] - prev.bbox[1]
    # prefer bbox delta for rigid translation when they disagree by rounding
    dr = dr_b if abs(round(dr_c) - dr_b) <= 1 else round(dr_c)
    dc = dc_b if abs(round(dc_c) - dc_b) <= 1 else round(dc_c)
    return int(dr), int(dc)


def diff_frames(
    prev_objs: list[Object],
    cur_objs: list[Object],
    prev_grid: np.ndarray,
    cur_grid: np.ndarray,
    levels_delta: int,
    motion_cells: int | None = None,
) -> Delta:
    changed_mask = prev_grid != cur_grid
    changed_cells = int(changed_mask.sum())

    prev_by_id = {o.id: o for o in prev_objs}
    cur_ids = {o.id for o in cur_objs}
    deltas: list[ObjectDelta] = []

    for co in cur_objs:
        po = prev_by_id.get(co.id)
        if po is None:
            deltas.append(ObjectDelta("appeared", co.id, new_color=co.color, new_shape_sig=co.shape_sig))
            continue
        moved = (abs(co.centroid[0] - po.centroid[0]) >= 0.5
                 or abs(co.centroid[1] - po.centroid[1]) >= 0.5
                 or po.bbox[0] != co.bbox[0] or po.bbox[1] != co.bbox[1])
        recolored = po.color != co.color
        reshaped = po.shape_sig != co.shape_sig
        rotated = (not reshaped) and (po.pose_sig != co.pose_sig)
        if moved and not reshaped and not recolored:
            deltas.append(ObjectDelta("moved", co.id, translation=_translation(po, co)))
        elif recolored and not reshaped and not moved:
            deltas.append(ObjectDelta("recolored", co.id, old_color=po.color, new_color=co.color))
        elif rotated and not moved:
            deltas.append(ObjectDelta("rotated", co.id,
                                      old_shape_sig=po.shape_sig, new_shape_sig=co.shape_sig))
        elif reshaped and not moved:
            deltas.append(ObjectDelta("reshaped", co.id,
                                      old_shape_sig=po.shape_sig, new_shape_sig=co.shape_sig))
        elif moved and (recolored or reshaped):
            # moved-while-transforming: emit both
            deltas.append(ObjectDelta("moved", co.id, translation=_translation(po, co)))
            if recolored:
                deltas.append(ObjectDelta("recolored", co.id, old_color=po.color, new_color=co.color))
            if reshaped:
                deltas.append(ObjectDelta("reshaped", co.id,
                                          old_shape_sig=po.shape_sig, new_shape_sig=co.shape_sig))
        else:
            deltas.append(ObjectDelta("static", co.id))

    for po in prev_objs:
        if po.id not in cur_ids:
            deltas.append(ObjectDelta("disappeared", po.id, old_color=po.color, old_shape_sig=po.shape_sig))

    kinds = {d.kind for d in deltas}
    structural = bool(kinds & {"moved", "appeared", "disappeared", "rotated", "reshaped"})
    is_noop = changed_cells == 0 and levels_delta == 0
    cosmetic = (
        changed_cells > 0
        and not structural
        and levels_delta == 0
        and changed_cells <= COSMETIC_MAX_CELLS
    )
    multi_frame = motion_cells is not None and motion_cells > changed_cells

    return Delta(
        changed_cells=changed_cells,
        changed_mask=changed_mask,
        object_deltas=deltas,
        levels_delta=levels_delta,
        is_noop=is_noop,
        cosmetic=cosmetic,
        multi_frame=multi_frame,
    )
