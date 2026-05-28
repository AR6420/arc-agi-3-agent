"""Stage 2 — framechange head fine-tune. Freeze backbone + action/spatial heads.

Phase A: 10 epochs on synth-only.
Phase B: 5 epochs on combined (3:1 real:synth).
Reports ROC-AUC on real val.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import BCDataset, collate
from .models import EliteModel
from .train import (
    MixedDataset, RealValSubset, SynthDataset, SPLITS, set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO_ROOT / "weights" / "stage2"
RUNS_DIR = REPO_ROOT / "runs" / "stage2"
STAGE1_BEST = REPO_ROOT / "weights" / "stage1" / "best.pt"


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    sum_pos = ranks[:len(pos)].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def eval_auc(model: EliteModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[float] = []
    for batch in loader:
        x = batch["perception"].to(device)
        aid = batch["action_id"].to(device)
        fc = batch["framechange"].to(device)
        out = model(x, action_id=aid)
        all_scores.extend(out["framechange_logits"].cpu().numpy().tolist())
        all_labels.extend(fc.cpu().numpy().tolist())
    return _auc(np.array(all_scores), np.array(all_labels))


def train_one_pass(
    model: EliteModel,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    log_prefix: str,
    metrics_fh,
) -> None:
    opt = torch.optim.AdamW(
        [p for p in model.framechange_head.parameters() if p.requires_grad],
        lr=lr,
    )
    for ep in range(1, epochs + 1):
        model.train()
        model.backbone.eval()  # freeze BN
        t0 = time.perf_counter()
        n_total = 0
        n_correct = 0
        loss_sum = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["perception"].to(device)
            aid = batch["action_id"].to(device)
            fc = batch["framechange"].to(device)
            opt.zero_grad()
            out = model(x, action_id=aid)
            loss = F.binary_cross_entropy_with_logits(out["framechange_logits"], fc)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1
            pred = (out["framechange_logits"].detach() > 0).float()
            n_correct += int((pred == fc).sum())
            n_total += int(fc.shape[0])
        auc = eval_auc(model, val_loader, device)
        log = {
            "phase": log_prefix,
            "epoch": ep,
            "train_loss": round(loss_sum / max(n_batches, 1), 4),
            "train_acc": round(n_correct / max(n_total, 1), 4),
            "val_auc": round(auc, 4),
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        print(json.dumps(log), flush=True)
        metrics_fh.write(json.dumps(log) + "\n")
        metrics_fh.flush()


def train_stage2(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    real_ds = BCDataset(drop_holdout=True)
    synth_ds = SynthDataset()
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train_idx = np.array(splits["train_indices"], dtype=np.int64)
    val_idx = np.array(splits["val_indices"], dtype=np.int64)

    bs = int(cfg["batch_size"])
    val_loader = DataLoader(RealValSubset(real_ds, val_idx), batch_size=bs, shuffle=False, collate_fn=collate)
    synth_loader = DataLoader(synth_ds, batch_size=bs, shuffle=True, collate_fn=collate, drop_last=True)
    mixed_loader = DataLoader(
        MixedDataset(real_ds, synth_ds, train_idx),
        batch_size=bs, shuffle=True, collate_fn=collate, drop_last=True,
    )

    model = EliteModel().to(device)
    sd = torch.load(STAGE1_BEST, map_location=device, weights_only=True)
    model.load_state_dict(sd["model"])
    print(f"loaded Stage 1 best (val_acc={sd.get('val_acc'):.4f})", flush=True)

    # Freeze backbone + action_head + spatial_head; only framechange_head trains.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.action_head.parameters():
        p.requires_grad = False
    for p in model.spatial_head.parameters():
        p.requires_grad = False

    # Baseline AUC before tuning
    init_auc = eval_auc(model, val_loader, device)
    print(f"baseline framechange val AUC (post Stage 1): {init_auc:.4f}", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics_fh = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")
    metrics_fh.write(json.dumps({"phase": "baseline", "val_auc": init_auc}) + "\n")

    lr = float(cfg["optimizer"]["lr"])
    train_one_pass(model, synth_loader, val_loader, device,
                   int(cfg["phase_a"]["epochs"]), lr, "synth_only", metrics_fh)
    train_one_pass(model, mixed_loader, val_loader, device,
                   int(cfg["phase_b"]["epochs"]), lr, "combined", metrics_fh)

    final_auc = eval_auc(model, val_loader, device)
    target = float(cfg["target_auc"])
    halt = float(cfg["halt_below_auc"])

    out = WEIGHTS_DIR / "framechange_finetuned.pt"
    torch.save({"model": model.state_dict(), "final_auc": final_auc, "cfg": cfg}, out)

    summary = {
        "run_dir": str(run_dir),
        "baseline_auc": init_auc,
        "final_auc": final_auc,
        "target_auc": target,
        "passed": final_auc >= target,
        "halt_flag": final_auc < halt,
        "weights": str(out),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_fh.close()
    print(f"\n=== STAGE 2 DONE ===\n{json.dumps(summary, indent=2)}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_stage2(Path(args.config))


if __name__ == "__main__":
    main()
