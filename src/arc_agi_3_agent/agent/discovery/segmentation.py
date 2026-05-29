"""Connected-components segmentation (numpy only, no scipy).

Iterative BFS flood-fill over a label array. 4- or 8-connected, same-color,
background (0) never emitted. Produces rotation-invariant shape signatures.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .constants import BG
from .types import Object

_NBR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NBR8 = _NBR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _hash_bytes(*parts: bytes | tuple) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, tuple):
            h.update(repr(p).encode())
        else:
            h.update(p)
    return int.from_bytes(h.digest(), "big")


def pose_signature(mask: np.ndarray) -> int:
    """Translation-invariant, orientation-SENSITIVE signature of a local bool mask."""
    return _hash_bytes(mask.shape, np.ascontiguousarray(mask).tobytes())


def canonical_shape_signature(mask: np.ndarray) -> int:
    """Rotation+translation invariant signature (min over 4 rotations).

    Chirality preserved (no mirror) so ar25-style L-pieces stay distinct from
    their mirror images.
    """
    best: tuple | None = None
    m = mask
    for _ in range(4):
        m = np.rot90(m)
        key = (m.shape, np.ascontiguousarray(m).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return _hash_bytes(best[0], best[1])


def flood_fill_components(grid: np.ndarray, conn: int = 4) -> list[Object]:
    """Segment `grid` (H,W int) into same-color connected components.

    Background (0) is never emitted. Returns Objects with id=-1 (assigned later
    by matching). `conn` is 4 or 8.
    """
    h, w = grid.shape
    nbrs = _NBR4 if conn == 4 else _NBR8
    labels = np.full((h, w), -1, dtype=np.int32)
    objects: list[Object] = []
    nxt = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or labels[r, c] != -1:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            labels[r, c] = nxt
            pix: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == -1 and grid[ny, nx] == color:
                        labels[ny, nx] = nxt
                        stack.append((ny, nx))
            ys = np.array([p[0] for p in pix])
            xs = np.array([p[1] for p in pix])
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[ys - r0, xs - c0] = True
            centroid = (float(ys.mean()), float(xs.mean()))
            objects.append(Object(
                id=-1,
                color=color,
                size=len(pix),
                bbox=(r0, c0, r1, c1),
                centroid=centroid,
                mask=mask,
                shape_sig=canonical_shape_signature(mask),
                pose_sig=pose_signature(mask),
                conn=conn,
            ))
            nxt += 1
    return objects
