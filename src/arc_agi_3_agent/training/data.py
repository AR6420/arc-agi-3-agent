"""BC training dataset over `data/bc_transitions_v3.npz`.

Filters out holdout envs per `splits.HOLDOUT_ENV_IDS`. Computes frame-change
labels on-the-fly by diffing consecutive `perception_input` rows within a replay.

Schema (v3 NPZ):
    perception_input: (N, 3, 64, 64) int8         — (first, last, max-abs-diff)
    env_ids:          (N,) int8                    — index into env_to_idx
    replay_ids:       (N,) int32
    step_in_replay:   (N,) int32
    action_id:        (N,) int8 in [0, 7]
    action_x:         (N,) int8 in [-1, 63]        — -1 for non-ACTION6
    action_y:         (N,) int8 in [-1, 63]
    terminal:         (N,) bool
    levels_completed: (N,) int8
    available_actions_mask: (N, 8) bool
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eval.splits import HOLDOUT_ENV_IDS


REPO_ROOT = Path(__file__).resolve().parents[3]
NPZ_PATH = REPO_ROOT / "data" / "bc_transitions_v3.npz"
META_PATH = REPO_ROOT / "data" / "bc_transitions_v3_meta.json"


class BCExample(NamedTuple):
    perception: torch.Tensor       # (3, 64, 64) float32, /15.0
    action_id: torch.Tensor        # () long
    action_xy: torch.Tensor        # (2,) long; (-1, -1) for non-ACTION6
    is_action6: torch.Tensor       # () bool
    framechange: torch.Tensor      # () float32 in {0., 1.}
    avail_mask: torch.Tensor       # (8,) bool
    env_id_idx: torch.Tensor       # () long
    env_id_str: str                # holdout-isolation test reads this


def _load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _holdout_idx_set(env_to_idx: dict[str, int]) -> set[int]:
    return {env_to_idx[e] for e in HOLDOUT_ENV_IDS if e in env_to_idx}


class BCDataset(Dataset[BCExample]):
    """Reads v3 NPZ, drops holdout envs, computes frame-change labels.

    Frame-change at index `i` is defined as:
        1.0  if perception_input[i] differs from perception_input[i-1]
              AND step_in_replay[i] == step_in_replay[i-1] + 1
        0.0  otherwise (boundary or unchanged)

    Boundaries (first step of a replay) are LABEL=0 with mask=0; we drop them
    from the training population entirely to keep the loss clean.
    """

    def __init__(self, npz_path: Path = NPZ_PATH, drop_holdout: bool = True) -> None:
        z = np.load(npz_path)
        meta = _load_meta()
        env_to_idx = meta["env_to_idx"]
        self.idx_to_env: dict[int, str] = {v: k for k, v in env_to_idx.items()}

        env_ids = z["env_ids"].astype(np.int32)
        replay_ids = z["replay_ids"].astype(np.int32)
        step = z["step_in_replay"].astype(np.int32)

        # Frame-change requires same-replay predecessor → drop step==0 rows
        # AND drop any holdout-env rows.
        keep = step > 0
        if drop_holdout:
            holdout = _holdout_idx_set(env_to_idx)
            for h in holdout:
                keep &= env_ids != h

        idx = np.nonzero(keep)[0]
        prev_idx = idx - 1
        # Verify predecessor is same replay (always true since step>0 ensures
        # within-replay, but guard against parser bugs).
        same_replay = replay_ids[idx] == replay_ids[prev_idx]
        idx = idx[same_replay]
        prev_idx = prev_idx[same_replay]

        self._idx = idx
        self._prev_idx = prev_idx
        # Hold refs to underlying arrays (mmap if NPZ supports; here it loads).
        self._perception = z["perception_input"]
        self._env_ids = env_ids
        self._action_id = z["action_id"].astype(np.int64)
        self._action_x = z["action_x"].astype(np.int64)
        self._action_y = z["action_y"].astype(np.int64)
        self._avail = z["available_actions_mask"].astype(bool)

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def __getitem__(self, i: int) -> BCExample:
        row = int(self._idx[i])
        prev = int(self._prev_idx[i])

        cur = self._perception[row].astype(np.float32) / 15.0
        pre = self._perception[prev]
        framechange = float(not np.array_equal(self._perception[row], pre))

        ax = int(self._action_x[row])
        ay = int(self._action_y[row])
        aid = int(self._action_id[row])
        is_a6 = aid == 6

        env_idx = int(self._env_ids[row])
        env_str = self.idx_to_env.get(env_idx, "?")

        return BCExample(
            perception=torch.from_numpy(cur),
            action_id=torch.tensor(aid, dtype=torch.long),
            action_xy=torch.tensor([ax, ay], dtype=torch.long),
            is_action6=torch.tensor(is_a6, dtype=torch.bool),
            framechange=torch.tensor(framechange, dtype=torch.float32),
            avail_mask=torch.from_numpy(self._avail[row]),
            env_id_idx=torch.tensor(env_idx, dtype=torch.long),
            env_id_str=env_str,
        )


def collate(examples: list[BCExample]) -> dict[str, torch.Tensor | list[str]]:
    out: dict[str, torch.Tensor | list[str]] = {
        "perception": torch.stack([e.perception for e in examples]),
        "action_id": torch.stack([e.action_id for e in examples]),
        "action_xy": torch.stack([e.action_xy for e in examples]),
        "is_action6": torch.stack([e.is_action6 for e in examples]),
        "framechange": torch.stack([e.framechange for e in examples]),
        "avail_mask": torch.stack([e.avail_mask for e in examples]),
        "env_id_idx": torch.stack([e.env_id_idx for e in examples]),
        "env_id_str": [e.env_id_str for e in examples],
    }
    return out
