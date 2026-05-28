"""5 forward+backward steps at batch 128, report peak VRAM."""

from __future__ import annotations

import torch

from arc_agi_3_agent.training.losses import combined_loss
from arc_agi_3_agent.training.models import EliteModel, param_count


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — VRAM probe skipped. (Local torch is CPU-only.)")
        m = EliteModel()
        print(f"model param count: {param_count(m):,}")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = EliteModel().to(device)
    print(f"model param count: {param_count(model):,}")

    B = 128
    x = torch.rand(B, 3, 64, 64, device=device)
    action_id = torch.randint(0, 8, (B,), device=device)
    action_xy = torch.randint(0, 64, (B, 2), device=device)
    is_action6 = action_id == 6
    framechange = torch.randint(0, 2, (B,), device=device).float()
    batch = {
        "action_id": action_id,
        "action_xy": action_xy,
        "is_action6": is_action6,
        "framechange": framechange,
    }
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        out = model(x, action_id=action_id)
        loss = combined_loss(out, batch)["loss"]
        loss.backward()
        opt.step()
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"peak VRAM @ batch={B}: {peak:.1f} MB")
    assert peak < 6 * 1024, f"VRAM budget exceeded: {peak:.1f} MB"


if __name__ == "__main__":
    main()
