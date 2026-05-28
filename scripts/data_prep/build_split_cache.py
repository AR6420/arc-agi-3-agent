"""Deterministic 90/10 train/val split over real BC data (train envs only).

Stratified by env_id_idx so every train env appears in both splits.
Holdout envs are excluded upstream (drop_holdout=True in BCDataset).

Output: data/splits_cache.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from arc_agi_3_agent.training.data import BCDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CACHE = REPO_ROOT / "data" / "splits_cache.json"
SEED = 0


def main() -> None:
    ds = BCDataset(drop_holdout=True)
    n = len(ds)
    env_idx = ds._env_ids[ds._idx]  # filtered env indices

    rng = np.random.default_rng(SEED)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for e in np.unique(env_idx):
        rows_in_env = np.nonzero(env_idx == e)[0]
        perm = rng.permutation(rows_in_env)
        n_val = max(1, int(0.10 * len(perm)))
        val_indices.extend(perm[:n_val].tolist())
        train_indices.extend(perm[n_val:].tolist())

    train_arr = np.sort(np.array(train_indices, dtype=np.int64))
    val_arr = np.sort(np.array(val_indices, dtype=np.int64))

    h = hashlib.sha256()
    h.update(train_arr.tobytes())
    h.update(val_arr.tobytes())
    digest = h.hexdigest()

    payload = {
        "seed": SEED,
        "n_total": n,
        "n_train": len(train_arr),
        "n_val": len(val_arr),
        "fraction_val": round(len(val_arr) / n, 6),
        "sha256": digest,
        "train_indices": train_arr.tolist(),
        "val_indices": val_arr.tolist(),
    }
    SPLITS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {SPLITS_CACHE}")
    print(f"n_train={len(train_arr)}  n_val={len(val_arr)}  sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
