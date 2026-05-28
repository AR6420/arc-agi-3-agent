"""E-lite v0 inference agent.

- Backbone + 3 heads from Stage 1 (best.pt) optionally overlayed with Stage 2
  framechange weights.
- Cluster priors from Stage 3 (cluster_priors.json) as fallback distribution when
  action-type softmax is low-confidence.
- Encodes incoming frame stack via OQ7 reduce_frame_stack(frame).

Phase 0c §3.2 seed: per-env seed = hash((env_id, run_id, "elite_v0")) & 0xFFFFFFFF.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.perception_input import reduce_frame_stack
from ..training.fit_cluster_priors import classify_env
from ..training.models import EliteModel

AGENT_VERSION = "elite_v0"

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_BEST = REPO_ROOT / "weights" / "stage1" / "best.pt"
STAGE2_BEST = REPO_ROOT / "weights" / "stage2" / "framechange_finetuned.pt"
CLUSTER_PRIORS = REPO_ROOT / "weights" / "cluster_priors.json"


def per_env_seed(env_id: str, run_id: int = 0) -> int:
    return hash((env_id, run_id, AGENT_VERSION)) & 0xFFFFFFFF


class EliteV0:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold: float = 0.4,
        use_stage2: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = EliteModel().to(self.device)
        sd = torch.load(STAGE1_BEST, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd["model"])
        if use_stage2 and STAGE2_BEST.exists():
            sd2 = torch.load(STAGE2_BEST, map_location=self.device, weights_only=True)
            self.model.load_state_dict(sd2["model"])
        self.model.eval()

        priors_payload = json.loads(CLUSTER_PRIORS.read_text(encoding="utf-8"))
        self.priors: dict[str, np.ndarray] = {
            k: np.array(v, dtype=np.float32) for k, v in priors_payload["priors"].items()
        }

        self.confidence_threshold = float(confidence_threshold)
        self.env_id: str | None = None
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.cluster: str = "mixed"
        self.available_actions: list[int] = []

    def reset_for_env(
        self, env_id: str, available_actions: list[int], run_id: int = 0
    ) -> None:
        seed = per_env_seed(env_id, run_id)
        self.env_id = env_id
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.available_actions = [a for a in available_actions if a != 0]
        if not self.available_actions:
            self.available_actions = [1]

        mask = np.zeros(8, dtype=bool)
        for a in available_actions:
            if 0 <= a < 8:
                mask[a] = True
        self.cluster = classify_env(mask)

    @torch.no_grad()
    def choose_action(self, frame_stack: Any) -> tuple[int, dict[str, int]]:
        """frame_stack: list of (64,64) ndarrays or (T,64,64) ndarray."""
        assert self.rng is not None and self.np_rng is not None
        if isinstance(frame_stack, list):
            arr = np.stack([np.asarray(f, dtype=np.int8) for f in frame_stack], axis=0)
        else:
            arr = np.asarray(frame_stack, dtype=np.int8)
            if arr.ndim == 2:
                arr = arr[None, ...]
        perception = reduce_frame_stack(arr).astype(np.float32) / 15.0
        x = torch.from_numpy(perception).unsqueeze(0).to(self.device)
        out = self.model(x)
        action_logits = out["action_logits"][0].cpu().numpy()

        # Mask to available actions
        avail_set = set(self.available_actions)
        masked = np.full(8, -np.inf, dtype=np.float32)
        for a in self.available_actions:
            masked[a] = action_logits[a]
        probs = _softmax(masked)
        top = float(probs.max())

        if top < self.confidence_threshold:
            prior = self.priors.get(self.cluster, self.priors["mixed"]).copy()
            mask_arr = np.zeros(8, dtype=np.float32)
            for a in self.available_actions:
                mask_arr[a] = 1.0
            prior = prior * mask_arr
            s = prior.sum()
            if s > 0:
                prior = prior / s
                probs = prior

        action_id = int(probs.argmax())
        data: dict[str, int] = {}
        if action_id == 6:
            sp = out["spatial_logits"][0].cpu().numpy()  # (64, 64)
            flat = sp.reshape(-1)
            idx = int(flat.argmax())
            y, x_pix = divmod(idx, 64)
            data = {"x": int(x_pix), "y": int(y)}
        return action_id, data


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
