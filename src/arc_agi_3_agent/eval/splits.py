"""Train / holdout env splits — frozen for the entire project lifetime.

Phase 0c §2.1: 5 envs held out for the "≥ 10" gate metric, 20 for training.
Selected to span action signatures, complexity, and dynamics:

    H1 vc33  — pure_click (simplest pure-click test)
    H2 tu93  — pure_movement (9 levels, mid-depth)
    H3 sk48  — mixed + undo (exercises full action head + ACTION7)
    H4 lp85  — max sprite tags (86, perception stress test)
    H5 dc22  — max lose calls (5, complex failure conditions)

The remaining 20 are the training set for any replay-trained component.
"""

from __future__ import annotations

# Phase 0c §2.1 — frozen selection. Do NOT change without a phase doc revision.
HOLDOUT_ENV_IDS: tuple[str, ...] = ("vc33", "tu93", "sk48", "lp85", "dc22")

ALL_PUBLIC_ENV_IDS: tuple[str, ...] = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
)

TRAIN_ENV_IDS: tuple[str, ...] = tuple(
    env_id for env_id in ALL_PUBLIC_ENV_IDS if env_id not in HOLDOUT_ENV_IDS
)


def is_holdout(env_id: str) -> bool:
    """Strip any hash suffix (e.g. 'sp80-589a99af') before lookup."""
    base = env_id.split("-")[0]
    return base in HOLDOUT_ENV_IDS


def is_train(env_id: str) -> bool:
    base = env_id.split("-")[0]
    return base in TRAIN_ENV_IDS


# Sanity check — exact counts at import time
assert len(HOLDOUT_ENV_IDS) == 5
assert len(TRAIN_ENV_IDS) == 20
assert len(set(HOLDOUT_ENV_IDS) | set(TRAIN_ENV_IDS)) == 25
assert set(HOLDOUT_ENV_IDS).isdisjoint(set(TRAIN_ENV_IDS))
