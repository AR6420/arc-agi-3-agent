"""Sanity gate: model must overfit a single batch of 64 in 200 steps.

If this fails, architecture or loss has a bug — do NOT launch Stage 1 training.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from arc_agi_3_agent.training.data import BCDataset, collate
from arc_agi_3_agent.training.losses import combined_loss
from arc_agi_3_agent.training.models import EliteModel


def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_overfit_single_batch() -> None:
    _set_seed(0)
    ds = BCDataset(drop_holdout=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EliteModel().to(device)
    model.train()

    perception = batch["perception"].to(device)
    action_id = batch["action_id"].to(device)
    action_xy = batch["action_xy"].to(device)
    is_action6 = batch["is_action6"].to(device)
    framechange = batch["framechange"].to(device)

    inputs = {
        "action_id": action_id,
        "action_xy": action_xy,
        "is_action6": is_action6,
        "framechange": framechange,
    }

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    last: dict[str, torch.Tensor] = {}
    for step in range(200):
        opt.zero_grad()
        out = model(perception, action_id=action_id)
        losses = combined_loss(out, inputs)
        losses["loss"].backward()
        opt.step()
        last = losses

    L_action = float(last["loss_action"])
    L_spatial = float(last["loss_spatial"])
    L_fc = float(last["loss_framechange"])

    print(
        f"\nfinal: action={L_action:.4f} spatial={L_spatial:.4f} "
        f"framechange={L_fc:.4f}"
    )
    assert L_action < 0.1, f"action CE did not overfit: {L_action}"
    # Spatial focal can be tiny because most batches have few ACTION6 examples;
    # require <0.05 (focal is naturally small).
    assert L_spatial < 0.05, f"spatial focal did not overfit: {L_spatial}"
    assert L_fc < 0.1, f"framechange BCE did not overfit: {L_fc}"
