"""Diagnostic — teacher-forced BC prediction accuracy, train vs holdout.

Loads v1 weights (560K-param backbone, Phase 2 stage1 best). For every human-replay
transition, feeds the perception (OQ7 3-channel) to the model and compares the
predicted action-type (top-1 / top-3, masked to available actions) and — for
ACTION6 rows — the predicted click pixel (top-5 exact, top-5 within Chebyshev
radius 3) to the human's actual action.

This is teacher-forced prediction on the human-replay distribution, NOT in-env
rollout. Goal: isolate "can the model predict the right action in unseen-env
states" (generalization) from "can it execute a full trajectory" (compounding).

No training. No env interaction. No submission.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arc_agi_3_agent.training.data import BCDataset, collate
from arc_agi_3_agent.training.models_v1 import EliteModel  # v1 128-ch arch
from arc_agi_3_agent.eval.splits import HOLDOUT_ENV_IDS, TRAIN_ENV_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_STAGE1_BEST = REPO_ROOT / "weights" / "stage1_v1_archive" / "best.pt"
SPATIAL_RADIUS = 3   # Chebyshev: max(|dx|,|dy|) <= 3 counts as a hit (click-tolerance insight)
SPATIAL_TOPK = 5

# Kaggle S1 biased-random baseline (2000-action COMPETITION-mode scorecard), holdout.
RANDOM_BASELINE = {
    "vc33": {"score": 0.003, "lvls": 1},
    "tu93": {"score": 0.000, "lvls": 0},
    "sk48": {"score": 0.283, "lvls": 1},
    "lp85": {"score": 0.030, "lvls": 2},
    "dc22": {"score": 0.000, "lvls": 0},
}


class EnvStats:
    __slots__ = ("n", "c1", "c3", "n6", "sp5", "sp5r")

    def __init__(self) -> None:
        self.n = 0       # transitions
        self.c1 = 0      # top-1 action-type correct
        self.c3 = 0      # top-3 action-type correct
        self.n6 = 0      # ACTION6 transitions
        self.sp5 = 0     # spatial top-5 exact hit
        self.sp5r = 0    # spatial top-5 within radius-3 hit


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EliteModel().to(device)
    sd = torch.load(V1_STAGE1_BEST, map_location=device, weights_only=True)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"loaded v1 stage1 best (val_acc={sd.get('val_acc'):.4f}) on {device}", flush=True)

    # drop_holdout=False → all 25 envs, including the 5 holdout. step>0 still filtered.
    ds = BCDataset(drop_holdout=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"transitions (step>0, all 25 envs): {len(ds)}", flush=True)

    per_env: dict[str, EnvStats] = defaultdict(EnvStats)

    with torch.no_grad():
        for batch in loader:
            x = batch["perception"].to(device)
            true_a = batch["action_id"].to(device)               # (B,)
            avail = batch["avail_mask"].to(device)                # (B, 8) bool
            is6 = batch["is_action6"].to(device)                  # (B,) bool
            axy = batch["action_xy"].to(device)                   # (B, 2) -> (x, y)
            env_strs = batch["env_id_str"]

            out = model(x)
            logits = out["action_logits"].clone()                 # (B, 8)
            logits[~avail] = float("-inf")                        # mask to available

            top3 = logits.topk(3, dim=1).indices                  # (B, 3)
            top1 = top3[:, 0]
            c1 = (top1 == true_a)
            c3 = (top3 == true_a.unsqueeze(1)).any(dim=1)

            # Spatial — only ACTION6 rows.
            sp = out["spatial_logits"]                            # (B, 64, 64)
            B = sp.shape[0]
            sp_flat = sp.reshape(B, -1)                            # (B, 4096)
            sp_top = sp_flat.topk(SPATIAL_TOPK, dim=1).indices     # (B, k)
            ty = axy[:, 1]                                         # true y
            tx = axy[:, 0]                                         # true x
            true_idx = (ty * 64 + tx)                              # (B,)
            sp_hit = (sp_top == true_idx.unsqueeze(1)).any(dim=1)  # (B,) exact

            # within radius-3 (Chebyshev) of true pixel
            pk_y = sp_top // 64                                    # (B, k)
            pk_x = sp_top % 64
            dy = (pk_y - ty.unsqueeze(1)).abs()
            dx = (pk_x - tx.unsqueeze(1)).abs()
            within = ((dy <= SPATIAL_RADIUS) & (dx <= SPATIAL_RADIUS)).any(dim=1)  # (B,)

            c1 = c1.cpu().numpy(); c3 = c3.cpu().numpy()
            is6 = is6.cpu().numpy()
            sp_hit = sp_hit.cpu().numpy(); within = within.cpu().numpy()

            for j, e in enumerate(env_strs):
                st = per_env[e]
                st.n += 1
                st.c1 += int(c1[j])
                st.c3 += int(c3[j])
                if is6[j]:
                    st.n6 += 1
                    st.sp5 += int(sp_hit[j])
                    st.sp5r += int(within[j])

    # Aggregate
    def agg(env_list: tuple[str, ...]) -> dict:
        n = c1 = c3 = n6 = sp5 = sp5r = 0
        for e in env_list:
            st = per_env.get(e)
            if not st:
                continue
            n += st.n; c1 += st.c1; c3 += st.c3
            n6 += st.n6; sp5 += st.sp5; sp5r += st.sp5r
        return {
            "n": n, "top1": c1 / n if n else 0.0, "top3": c3 / n if n else 0.0,
            "n6": n6, "sp5": sp5 / n6 if n6 else float("nan"),
            "sp5r": sp5r / n6 if n6 else float("nan"),
        }

    train_agg = agg(TRAIN_ENV_IDS)
    hold_agg = agg(HOLDOUT_ENV_IDS)

    result = {
        "v1_val_acc": float(sd.get("val_acc")),
        "spatial_radius": SPATIAL_RADIUS,
        "spatial_topk": SPATIAL_TOPK,
        "train_agg": train_agg,
        "holdout_agg": hold_agg,
        "per_env": {
            e: {
                "n": st.n, "top1": st.c1 / st.n if st.n else 0.0,
                "top3": st.c3 / st.n if st.n else 0.0, "n6": st.n6,
                "sp5": st.sp5 / st.n6 if st.n6 else None,
                "sp5r": st.sp5r / st.n6 if st.n6 else None,
                "holdout": e in HOLDOUT_ENV_IDS,
            }
            for e, st in per_env.items()
        },
        "gap_top1": train_agg["top1"] - hold_agg["top1"],
        "gap_top3": train_agg["top3"] - hold_agg["top3"],
    }
    out_path = REPO_ROOT / "diagnostic_bc_generalization_raw.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Console summary
    print("\n=== TEACHER-FORCED BC PREDICTION ACCURACY ===")
    print(f"{'split':>8} {'n':>8} {'top1':>7} {'top3':>7} {'n6':>7} {'sp5':>7} {'sp5_r3':>7}")
    for name, a in [("TRAIN", train_agg), ("HOLDOUT", hold_agg)]:
        sp5 = f"{a['sp5']:.3f}" if a["sp5"] == a["sp5"] else "—"
        sp5r = f"{a['sp5r']:.3f}" if a["sp5r"] == a["sp5r"] else "—"
        print(f"{name:>8} {a['n']:>8} {a['top1']:>7.3f} {a['top3']:>7.3f} {a['n6']:>7} {sp5:>7} {sp5r:>7}")
    print(f"\ngap top1 = {result['gap_top1']:+.3f}   gap top3 = {result['gap_top3']:+.3f}")
    gap = result["gap_top1"]
    if abs(gap) <= 0.05:
        print("VERDICT A: prediction GENERALIZES (gap within 5 pts) -> failure is EXECUTION.")
    elif gap > 0.10:
        print("VERDICT B: prediction does NOT generalize (gap > 10 pts) -> representation train-specific.")
    else:
        print(f"VERDICT borderline: gap {gap:+.3f} in 5-10 pt grey zone.")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
