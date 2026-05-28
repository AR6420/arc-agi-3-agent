"""Stage 1 BC pretrain entry point.

Combined real (75%) + synth (25%) sampling. AdamW + cosine. Per-epoch val
on real validation indices. Sanity gates:
    epoch 1: val action acc >= 30%
    epoch 10: val action acc >= 50%
    plateau 10 epochs: early stop
Checkpoints every 5 epochs + best-by-val to weights/stage1/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from .data import BCDataset, BCExample, collate
from .losses import combined_loss
from .models import EliteModel, param_count

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH_NPZ = REPO_ROOT / "data" / "bc_synth.npz"
SPLITS = REPO_ROOT / "data" / "splits_cache.json"
WEIGHTS_DIR = REPO_ROOT / "weights" / "stage1"
RUNS_DIR = REPO_ROOT / "runs" / "stage1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SynthDataset(Dataset):
    """Reads bc_synth.npz; returns BCExample-shaped dicts (framechange from saved label)."""

    def __init__(self, npz_path: Path = SYNTH_NPZ) -> None:
        z = np.load(npz_path)
        self.perception = z["perception_input"]
        self.action_id = z["action_id"].astype(np.int64)
        self.action_x = z["action_x"].astype(np.int64)
        self.action_y = z["action_y"].astype(np.int64)
        self.framechange = z["framechange"].astype(np.float32)
        self.env_ids = z["env_ids"].astype(np.int64)

    def __len__(self) -> int:
        return int(self.perception.shape[0])

    def __getitem__(self, i: int) -> BCExample:
        cur = self.perception[i].astype(np.float32) / 15.0
        aid = int(self.action_id[i])
        ax = int(self.action_x[i])
        ay = int(self.action_y[i])
        is_a6 = aid == 6
        avail = np.ones(8, dtype=bool)  # synth uses all actions; agent strips RESET upstream
        return BCExample(
            perception=torch.from_numpy(cur),
            action_id=torch.tensor(aid, dtype=torch.long),
            action_xy=torch.tensor([ax, ay], dtype=torch.long),
            is_action6=torch.tensor(is_a6, dtype=torch.bool),
            framechange=torch.tensor(float(self.framechange[i]), dtype=torch.float32),
            avail_mask=torch.from_numpy(avail),
            env_id_idx=torch.tensor(int(self.env_ids[i]), dtype=torch.long),
            env_id_str=f"synth_{int(self.env_ids[i])}",
        )


class MixedDataset(Dataset):
    """3:1 real:synth interleave. Index i % 4 == 3 -> synth; else real (train split)."""

    def __init__(
        self,
        real_ds: BCDataset,
        synth_ds: SynthDataset,
        real_train_indices: np.ndarray,
        real_share: float = 0.75,
    ) -> None:
        self.real = real_ds
        self.synth = synth_ds
        self.real_idx = real_train_indices
        self.real_share = real_share
        # virtual length = real / share so a full pass over the mix covers all reals
        self._n_real = len(real_train_indices)
        self._n = int(self._n_real / real_share)
        rng = np.random.default_rng(0)
        self._synth_perm = rng.permutation(len(synth_ds))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> BCExample:
        if (i % 4) == 3:
            j = self._synth_perm[i % len(self._synth_perm)]
            return self.synth[int(j)]
        # else real
        k = self.real_idx[i % self._n_real]
        return self.real[int(k)]


class RealValSubset(Dataset):
    def __init__(self, real_ds: BCDataset, val_indices: np.ndarray) -> None:
        self.real = real_ds
        self.val_idx = val_indices

    def __len__(self) -> int:
        return int(self.val_idx.shape[0])

    def __getitem__(self, i: int) -> BCExample:
        return self.real[int(self.val_idx[i])]


@torch.no_grad()
def evaluate(model: EliteModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    n_total = 0
    n_correct = 0
    a6_iou_num = 0.0
    a6_iou_den = 0
    fc_correct = 0
    fc_total = 0
    losses_sum = 0.0
    n_batches = 0
    for batch in loader:
        perception = batch["perception"].to(device, non_blocking=True)
        action_id = batch["action_id"].to(device, non_blocking=True)
        action_xy = batch["action_xy"].to(device, non_blocking=True)
        is_action6 = batch["is_action6"].to(device, non_blocking=True)
        framechange = batch["framechange"].to(device, non_blocking=True)

        out = model(perception, action_id=action_id)
        inputs = {
            "action_id": action_id, "action_xy": action_xy,
            "is_action6": is_action6, "framechange": framechange,
        }
        L = combined_loss(out, inputs)
        losses_sum += float(L["loss"])
        n_batches += 1

        pred = out["action_logits"].argmax(-1)
        n_correct += int((pred == action_id).sum())
        n_total += int(action_id.shape[0])

        # Spatial argmax IoU vs true xy (point-IoU = exact-pixel hit rate among ACTION6)
        if is_action6.any():
            sp = out["spatial_logits"][is_action6]  # (M, 64, 64)
            xy = action_xy[is_action6]
            flat = sp.flatten(1)
            am = flat.argmax(-1)
            true_flat = xy[:, 1] * 64 + xy[:, 0]  # y*64+x
            a6_iou_num += int((am == true_flat).sum())
            a6_iou_den += int(sp.shape[0])

        fc_pred = (out["framechange_logits"] > 0).float()
        fc_correct += int((fc_pred == framechange).sum())
        fc_total += int(framechange.shape[0])

    model.train()
    return {
        "val_loss": losses_sum / max(n_batches, 1),
        "val_action_acc": n_correct / max(n_total, 1),
        "val_action6_hit_rate": a6_iou_num / max(a6_iou_den, 1) if a6_iou_den else 0.0,
        "val_framechange_acc": fc_correct / max(fc_total, 1),
        "n_eval": n_total,
    }


def train_stage1(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    print("loading datasets ...", flush=True)
    real_ds = BCDataset(drop_holdout=True)
    synth_ds = SynthDataset()
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train_idx = np.array(splits["train_indices"], dtype=np.int64)
    val_idx = np.array(splits["val_indices"], dtype=np.int64)

    train_ds = MixedDataset(real_ds, synth_ds, train_idx)
    val_ds = RealValSubset(real_ds, val_idx)
    print(f"train: {len(train_ds):,}  val: {len(val_ds):,}  synth: {len(synth_ds):,}", flush=True)

    bs = int(cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)

    model = EliteModel().to(device)
    print(f"params: {param_count(model):,}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    n_epochs = int(cfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = n_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics_fh = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")

    w_sp = float(cfg["loss_weights"]["w_spatial"])
    w_fc = float(cfg["loss_weights"]["w_framechange"])
    ckpt_every = int(cfg["checkpoint_every"])

    best_val_acc = 0.0
    best_path: Path | None = None
    plateau = 0
    last_val_acc = 0.0

    t_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss_sum = 0.0
        ep_n = 0
        n_correct = 0
        n_total = 0
        t_ep = time.perf_counter()

        for step, batch in enumerate(train_loader):
            perception = batch["perception"].to(device, non_blocking=True)
            action_id = batch["action_id"].to(device, non_blocking=True)
            action_xy = batch["action_xy"].to(device, non_blocking=True)
            is_action6 = batch["is_action6"].to(device, non_blocking=True)
            framechange = batch["framechange"].to(device, non_blocking=True)

            opt.zero_grad()
            out = model(perception, action_id=action_id)
            inputs = {
                "action_id": action_id, "action_xy": action_xy,
                "is_action6": is_action6, "framechange": framechange,
            }
            L = combined_loss(out, inputs, w_spatial=w_sp, w_framechange=w_fc)
            L["loss"].backward()
            opt.step()
            sched.step()

            ep_loss_sum += float(L["loss"])
            ep_n += 1
            pred = out["action_logits"].argmax(-1)
            n_correct += int((pred == action_id).sum())
            n_total += int(action_id.shape[0])

        train_acc = n_correct / max(n_total, 1)
        train_loss = ep_loss_sum / max(ep_n, 1)
        val_metrics = evaluate(model, val_loader, device)
        ep_wall = time.perf_counter() - t_ep

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_action_acc": round(train_acc, 4),
            "wall_seconds": round(ep_wall, 1),
            "lr": opt.param_groups[0]["lr"],
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()},
        }
        print(json.dumps(log), flush=True)
        metrics_fh.write(json.dumps(log) + "\n")
        metrics_fh.flush()

        # Sanity gates
        if epoch == 1 and val_metrics["val_action_acc"] < 0.30:
            print(f"SANITY GATE FAIL: epoch 1 val_action_acc={val_metrics['val_action_acc']:.4f} < 0.30", flush=True)
        if epoch == 10 and val_metrics["val_action_acc"] < 0.50:
            print(f"SANITY GATE FAIL: epoch 10 val_action_acc={val_metrics['val_action_acc']:.4f} < 0.50", flush=True)

        # Checkpoint
        if epoch % ckpt_every == 0 or epoch == n_epochs:
            ckpt = WEIGHTS_DIR / f"epoch_{epoch:02d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, ckpt)
            print(f"  saved {ckpt}", flush=True)

        if val_metrics["val_action_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_action_acc"]
            best_path = WEIGHTS_DIR / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val_acc": best_val_acc}, best_path)
            plateau = 0
        else:
            plateau += 1
        last_val_acc = val_metrics["val_action_acc"]

        if plateau >= 10:
            print(f"early stop: val_action_acc plateau >=10 epochs (last {last_val_acc:.4f})", flush=True)
            break

    metrics_fh.close()
    total_wall = time.perf_counter() - t_start

    summary = {
        "run_dir": str(run_dir),
        "best_val_acc": best_val_acc,
        "best_ckpt": str(best_path) if best_path else None,
        "total_wall_seconds": round(total_wall, 1),
        "epochs_run": epoch,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== STAGE 1 DONE ===\n{json.dumps(summary, indent=2)}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_stage1(Path(args.config))


if __name__ == "__main__":
    main()
