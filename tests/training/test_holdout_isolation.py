"""Hard gate: no holdout env may appear in training dataset rows.

Holdout envs (vc33, tu93, sk48, lp85, dc22) are NEVER trained on — neither
in real BC data nor in synthetic rollouts. A single leak fails this test.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from arc_agi_3_agent.eval.splits import HOLDOUT_ENV_IDS
from arc_agi_3_agent.training.data import BCDataset, collate


def test_no_holdout_envs_in_batches() -> None:
    ds = BCDataset(drop_holdout=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate)

    seen_envs: set[str] = set()
    holdout = set(HOLDOUT_ENV_IDS)

    for i, batch in enumerate(loader):
        envs_in_batch = set(batch["env_id_str"])
        seen_envs |= envs_in_batch
        leak = envs_in_batch & holdout
        assert not leak, f"HOLDOUT LEAK in batch {i}: {leak}"
        if i >= 100:
            break

    # Sanity: we should have seen multiple train envs by 100 batches
    assert len(seen_envs) >= 5, f"too few envs sampled: {seen_envs}"
    assert not (seen_envs & holdout)


def test_dataset_length_excludes_holdout() -> None:
    """Without drop_holdout, len > with drop_holdout."""
    ds_full = BCDataset(drop_holdout=False)
    ds_train = BCDataset(drop_holdout=True)
    assert len(ds_full) > len(ds_train)
    # Holdout envs in v3 NPZ should be a non-trivial fraction (~20% of 25 envs).
    delta = len(ds_full) - len(ds_train)
    assert delta > 1000, f"holdout drop too small: {delta}"
