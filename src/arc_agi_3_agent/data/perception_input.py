"""Perception input encoding — (T, 64, 64) frame stack → (3, 64, 64) fixed tensor.

Resolves Phase 0c OQ7: variable-length animation stacks must be reduced to a
fixed-shape input for batched training. Reduction:

    channel 0 = frame[0]                       — pre-animation state
    channel 1 = frame[-1]                      — post-animation state
    channel 2 = max over t of |frame[t+1] - frame[t]|  — motion mask

For T=1 (70.9% of replay records per Phase 0a §4.5): channels 0 and 1 are
identical, channel 2 is all zeros. Model learns to treat zero motion mask as
"no animation".

Single source of truth — used by v3 parser, harness, and inference agent.
"""

from __future__ import annotations

import numpy as np


def reduce_frame_stack(frame: list | np.ndarray) -> np.ndarray:
    """Reduce a (T, 64, 64) frame stack to (3, 64, 64) int8.

    Args:
        frame: list of 64x64 grids, or a (T, 64, 64) ndarray. Cells are ints in [0, 15].

    Returns:
        ndarray of shape (3, 64, 64), dtype int8.

    Notes:
        - Channels are (first, last, max-abs-diff).
        - max-abs-diff is computed in int16 to avoid int8 overflow on |diff| up to 15,
          then cast back to int8 (max value 15 fits).
        - Empty frame stack → all-zero (3, 64, 64) (defensive; shouldn't happen in practice).
    """
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim != 3 or arr.shape[1:] != (64, 64) or arr.shape[0] == 0:
        # Defensive fallback for malformed input.
        return np.zeros((3, 64, 64), dtype=np.int8)

    first = arr[0].astype(np.int8)
    last = arr[-1].astype(np.int8)

    if arr.shape[0] == 1:
        diff = np.zeros((64, 64), dtype=np.int8)
    else:
        # Pairwise absolute differences, then per-cell max across time.
        # diffs shape: (T-1, 64, 64)
        diffs = np.abs(arr[1:] - arr[:-1])
        diff = diffs.max(axis=0).astype(np.int8)

    return np.stack([first, last, diff], axis=0)


def t_distribution_stats(frames_meta: list[int]) -> dict:
    """Compute T-distribution stats over a list of frame-stack lengths.

    Useful for verifying that the v3 parser output matches the Phase 0a §4.5 distribution
    (T=1 frequency ≈ 70.9%, T>1 ≈ 29.1%, max T up to 404).
    """
    arr = np.asarray(frames_meta, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "t1_count": int((arr == 1).sum()),
        "t1_frac": float((arr == 1).sum() / n),
        "t_gt1_count": int((arr > 1).sum()),
        "t_max": int(arr.max()),
        "t_mean": float(arr.mean()),
        "t_median": float(np.median(arr)),
    }
