"""Stage 0 composition gate — print real/synth makeup before Stage 1 launches."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from arc_agi_3_agent.training.data import BCDataset
from arc_agi_3_agent.eval.splits import TRAIN_ENV_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_NPZ = REPO_ROOT / "data" / "bc_synth.npz"
SPLITS = REPO_ROOT / "data" / "splits_cache.json"
OUT_JSON = REPO_ROOT / "runs" / "stage0" / "composition.json"


def main() -> None:
    real_ds = BCDataset(drop_holdout=True)
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    n_real_train = splits["n_train"]
    n_real_val = splits["n_val"]

    real_action = real_ds._action_id[real_ds._idx]
    real_env = real_ds._env_ids[real_ds._idx]

    synth = np.load(SYNTH_NPZ)
    synth_action = synth["action_id"]
    synth_env = synth["env_ids"]
    synth_fc = synth["framechange"]

    n_real_total = len(real_action)
    n_synth = len(synth_action)

    real_act_hist = Counter(int(a) for a in real_action)
    synth_act_hist = Counter(int(a) for a in synth_action)

    real_env_counts = {TRAIN_ENV_IDS[i]: 0 for i in range(len(TRAIN_ENV_IDS))}
    for e, c in zip(*np.unique(real_env, return_counts=True)):
        name = real_ds.idx_to_env[int(e)]
        if name in real_env_counts:
            real_env_counts[name] = int(c)

    synth_env_to_name = {i: TRAIN_ENV_IDS[i] for i in range(len(TRAIN_ENV_IDS))}
    synth_env_counts = {n: 0 for n in TRAIN_ENV_IDS}
    for e, c in zip(*np.unique(synth_env, return_counts=True)):
        synth_env_counts[synth_env_to_name[int(e)]] = int(c)

    synth_fc_rate = float(synth_fc.mean()) if n_synth else 0.0

    # Simulated 3:1 minibatch: per batch of 128 = 96 real + 32 synth
    mix = {"real_per_batch": 96, "synth_per_batch": 32, "real_share": 0.75}

    report = {
        "n_real_total": int(n_real_total),
        "n_real_train": n_real_train,
        "n_real_val": n_real_val,
        "n_synth": int(n_synth),
        "real_to_synth_ratio": round(n_real_total / max(n_synth, 1), 3),
        "minibatch_mix_target": mix,
        "real_action_histogram": dict(sorted(real_act_hist.items())),
        "synth_action_histogram": dict(sorted(synth_act_hist.items())),
        "real_env_counts": real_env_counts,
        "synth_env_counts": synth_env_counts,
        "synth_framechange_rate": round(synth_fc_rate, 4),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
