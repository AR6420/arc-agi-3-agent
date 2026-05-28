"""Combined Stage 1 loss: CE(action) + 0.3*focal(spatial|ACTION6) + 0.1*BCE(framechange).

Focal loss for the spatial head because positive class density is 1 pixel of 4096.
Masked: only ACTION6 transitions contribute to spatial loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_focal_loss(
    logits: torch.Tensor,         # (B, 64, 64)
    target_xy: torch.Tensor,      # (B, 2) long, sentinel (-1, -1) for non-ACTION6
    is_action6: torch.Tensor,     # (B,) bool
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Per-pixel focal BCE with one-hot target at (x, y); mean over ACTION6 batch.

    Returns 0-d zero tensor if no ACTION6 examples in batch.
    """
    B, H, W = logits.shape
    if not is_action6.any():
        return logits.new_zeros(())

    mask = is_action6
    lg = logits[mask]                         # (M, 64, 64)
    xy = target_xy[mask]                      # (M, 2)
    M = lg.shape[0]
    # One-hot target
    target = lg.new_zeros((M, H, W))
    xs = xy[:, 0].clamp(0, W - 1)
    ys = xy[:, 1].clamp(0, H - 1)
    target[torch.arange(M, device=lg.device), ys, xs] = 1.0

    p = torch.sigmoid(lg)
    pt = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    bce = F.binary_cross_entropy_with_logits(lg, target, reduction="none")
    loss = alpha_t * (1 - pt).pow(gamma) * bce
    return loss.mean()


def combined_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_spatial: float = 0.3,
    w_framechange: float = 0.1,
) -> dict[str, torch.Tensor]:
    ce = F.cross_entropy(outputs["action_logits"], batch["action_id"])

    sp = spatial_focal_loss(
        outputs["spatial_logits"],
        batch["action_xy"],
        batch["is_action6"],
    )

    if "framechange_logits" in outputs:
        fc = F.binary_cross_entropy_with_logits(
            outputs["framechange_logits"],
            batch["framechange"],
        )
    else:
        fc = outputs["action_logits"].new_zeros(())

    total = ce + w_spatial * sp + w_framechange * fc
    return {
        "loss": total,
        "loss_action": ce.detach(),
        "loss_spatial": sp.detach(),
        "loss_framechange": fc.detach(),
    }
