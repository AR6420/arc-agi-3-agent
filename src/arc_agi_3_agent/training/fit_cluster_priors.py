"""Stage 3 — empirical action prior per cluster signature.

Signature = derived from each env's modal available_actions mask:
    pure_click       — ACTION6 enabled, no movement actions (1..5)
    pure_movement    — at least one of 1..5 enabled, no ACTION6
    mixed            — both ACTION6 and at least one movement enabled

For each cluster, compute action_id histogram over real replays (train envs only).
Save normalized 8-dim categorical to weights/cluster_priors.json with KL divergences
between clusters for sanity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..training.data import BCDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "weights" / "cluster_priors.json"


def classify_env(mask: np.ndarray) -> str:
    """mask: (8,) bool, indices 0..7. ACTION0 = RESET ignored."""
    has_click = bool(mask[6])
    has_move = bool(mask[1] or mask[2] or mask[3] or mask[4] or mask[5])
    if has_click and not has_move:
        return "pure_click"
    if has_move and not has_click:
        return "pure_movement"
    return "mixed"


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    ds = BCDataset(drop_holdout=True)
    env_ids = ds._env_ids[ds._idx]
    actions = ds._action_id[ds._idx]
    masks = ds._avail[ds._idx]

    # Modal mask per env
    env_to_cluster: dict[int, str] = {}
    for e in np.unique(env_ids):
        rows = np.nonzero(env_ids == e)[0]
        # Modal mask = majority over rows (per column)
        modal = masks[rows].mean(axis=0) > 0.5
        env_to_cluster[int(e)] = classify_env(modal)

    cluster_hist: dict[str, np.ndarray] = {
        "pure_click": np.zeros(8, dtype=np.float64),
        "pure_movement": np.zeros(8, dtype=np.float64),
        "mixed": np.zeros(8, dtype=np.float64),
    }
    cluster_envs: dict[str, list[str]] = {k: [] for k in cluster_hist}

    for e, cl in env_to_cluster.items():
        cluster_envs[cl].append(ds.idx_to_env[e])
        rows = np.nonzero(env_ids == e)[0]
        for a in actions[rows]:
            cluster_hist[cl][int(a)] += 1

    priors: dict[str, list[float]] = {}
    for cl, h in cluster_hist.items():
        s = h.sum()
        if s > 0:
            priors[cl] = (h / s).tolist()
        else:
            priors[cl] = (np.ones(8) / 8).tolist()

    p = np.array(priors["pure_click"])
    m = np.array(priors["pure_movement"])
    x = np.array(priors["mixed"])

    payload = {
        "version": "cluster_priors_v0",
        "n_actions": 8,
        "priors": priors,
        "cluster_envs": cluster_envs,
        "diagnostics": {
            "kl_click_vs_movement": kl(p, m),
            "kl_movement_vs_click": kl(m, p),
            "kl_mixed_vs_click": kl(x, p),
            "kl_mixed_vs_movement": kl(x, m),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")

    kl_pm = payload["diagnostics"]["kl_click_vs_movement"]
    if kl_pm < 1.0:
        print(f"WARNING: KL(click||movement) = {kl_pm:.3f} < 1.0 sanity threshold")


if __name__ == "__main__":
    main()
