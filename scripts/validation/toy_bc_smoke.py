"""Toy BC smoke test (Phase 0b addendum Task 2).

Verifies that the v2 forward-BC parser produces a learnable signal.

Trains tiny CNN: 2 conv (16→32) + global pool + 7-way action head.
300 steps, batch 64, Adam lr 1e-3. ~3-5 min CPU.

Also trains the same model on v1 (inverse-model pairing) as a control. Both
should learn (loss decreases). The point of comparison is the predicted action
distribution on a held-out batch — v1 and v2 must differ measurably (KL > 0.05).
If they're identical, the shift had no effect. If v2 learns and v1 doesn't,
shift is correct. If v1 learns and v2 doesn't, shift went wrong direction.

Smoke only — no weight saving, no further training, no architecture design.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

V1_NPZ = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions.npz")
V2_NPZ = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\data\bc_transitions_v2.npz")
OUT_LOG = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.toy_bc_smoke.json")

torch.manual_seed(0)
np.random.seed(0)


class TinyCNN(nn.Module):
    def __init__(self, n_actions=8):
        super().__init__()
        # One-hot input over 16 colors → 16 input channels
        self.c1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.head = nn.Linear(32, n_actions)

    def forward(self, x_int):
        # x_int: (B, 64, 64) int8 in [0..15]. Convert to (B, 16, 64, 64) one-hot.
        x = F.one_hot(x_int.long().clamp(min=0, max=15), num_classes=16).permute(0, 3, 1, 2).float()
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = h.mean(dim=[2, 3])
        return self.head(h)


def load_sample(npz_path: Path, n: int = 5000, seed: int = 0):
    data = np.load(npz_path, allow_pickle=False)
    state = data["state"]
    action_id = data["action_id"]
    rng = np.random.default_rng(seed)
    # Filter to action_id in [0..7] (always true, defensive)
    mask = (action_id >= 0) & (action_id <= 7)
    idx_all = np.where(mask)[0]
    sel = rng.choice(idx_all, size=min(n, len(idx_all)), replace=False)
    return state[sel].copy(), action_id[sel].astype(np.int64).copy()


def train_and_eval(npz_path: Path, tag: str, n_steps: int = 300, batch: int = 64, lr: float = 1e-3):
    print(f"\n=== {tag}: {npz_path.name} ===")
    state_np, action_np = load_sample(npz_path, n=5000, seed=0)
    print(f"sampled {len(state_np)} (state, action) pairs")
    print(f"action dist in sample: {np.bincount(action_np, minlength=8).tolist()}")
    state = torch.from_numpy(state_np)
    action = torch.from_numpy(action_np)
    # Hold out last 500 for eval
    train_state, eval_state = state[:-500], state[-500:]
    train_action, eval_action = action[:-500], action[-500:]
    ds = TensorDataset(train_state, train_action)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    iterator = iter(loader)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        try:
            sb, ab = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sb, ab = next(iterator)
        logits = model(sb)
        loss = F.cross_entropy(logits, ab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 50 == 0 or step == n_steps - 1:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"trained {n_steps} steps in {elapsed:.1f}s")

    # Held-out predicted-action probs (mean over batch)
    model.eval()
    with torch.no_grad():
        eval_logits = model(eval_state)
        eval_probs = F.softmax(eval_logits, dim=1).mean(dim=0).numpy()
    print(f"mean predicted action prob (held-out): {eval_probs.tolist()}")

    return {
        "tag": tag,
        "npz": str(npz_path.name),
        "n_train": len(train_state),
        "n_eval": len(eval_state),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_last10": float(np.mean(losses[-10:])),
        "loss_drop": losses[0] - losses[-1],
        "elapsed_sec": elapsed,
        "mean_eval_action_probs": eval_probs.tolist(),
        "losses": losses,
    }


def kl_div(p, q, eps=1e-9):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p /= p.sum()
    q /= q.sum()
    return float((p * np.log(p / q)).sum())


def main():
    print(f"BASELINE: ln(7) = {np.log(7):.4f}  ln(8) = {np.log(8):.4f}")
    v2 = train_and_eval(V2_NPZ, "v2_forward_bc")
    v1 = train_and_eval(V1_NPZ, "v1_inverse_model")
    kl_v2_vs_v1 = kl_div(v2["mean_eval_action_probs"], v1["mean_eval_action_probs"])
    kl_v1_vs_v2 = kl_div(v1["mean_eval_action_probs"], v2["mean_eval_action_probs"])

    out = {
        "v1_summary": {k: v for k, v in v1.items() if k != "losses"},
        "v2_summary": {k: v for k, v in v2.items() if k != "losses"},
        "kl_v2_vs_v1": kl_v2_vs_v1,
        "kl_v1_vs_v2": kl_v1_vs_v2,
        "verdict": {
            "v2_loss_dropped": v2["loss_first"] - v2["loss_mean_last10"] > 0.1,
            "v1_loss_dropped": v1["loss_first"] - v1["loss_mean_last10"] > 0.1,
            "distributions_differ": (kl_v2_vs_v1 + kl_v1_vs_v2) / 2 > 0.05,
        },
        "v1_losses": v1["losses"],
        "v2_losses": v2["losses"],
    }
    OUT_LOG.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== VERDICT ===")
    print(json.dumps(out["verdict"], indent=2))
    print(f"KL(v2 || v1) = {kl_v2_vs_v1:.4f}")
    print(f"KL(v1 || v2) = {kl_v1_vs_v2:.4f}")
    print(f"v2 loss_first={v2['loss_first']:.3f} loss_last10_mean={v2['loss_mean_last10']:.3f}")
    print(f"v1 loss_first={v1['loss_first']:.3f} loss_last10_mean={v1['loss_mean_last10']:.3f}")
    print(f"\nWrote {OUT_LOG}")


if __name__ == "__main__":
    main()
