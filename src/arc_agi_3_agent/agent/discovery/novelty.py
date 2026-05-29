"""Count-based novelty over board states (cross-process stable hashes)."""

from __future__ import annotations

import hashlib

import numpy as np


def frame_hash(grid: np.ndarray) -> int:
    """Exact-state hash of the (64,64) last frame. blake2b => stable across processes."""
    h = hashlib.blake2b(np.ascontiguousarray(grid, dtype=np.int8).tobytes(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def coarse_signature(grid: np.ndarray, block: int = 4) -> int:
    """Downsample 64x64 -> 16x16 dominant-color signature for coarse novelty."""
    h, w = grid.shape
    bh, bw = h // block, w // block
    g = grid[: bh * block, : bw * block].reshape(bh, block, bw, block)
    # dominant (max) color per block — cheap, order-invariant enough for coarse novelty
    coarse = g.max(axis=(1, 3)).astype(np.int8)
    hh = hashlib.blake2b(np.ascontiguousarray(coarse).tobytes(), digest_size=8)
    return int.from_bytes(hh.digest(), "big")


def novelty_score(visit_count: int) -> float:
    """1/sqrt(1+count): unseen states score 1.0, decays with visits."""
    return 1.0 / np.sqrt(1.0 + visit_count)
