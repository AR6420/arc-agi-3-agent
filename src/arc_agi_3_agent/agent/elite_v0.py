"""E-lite v0 inference agent — Phase 2.5 parameterized.

Modes (toggleable via constructor):
    temperature        — action-type softmax temperature (None = argmax)
    spatial_topk       — top-k pixel sampling for ACTION6 (None = argmax)
    framechange_filter — multiply action probs by P(frame changes | s, a)
    stuck_detector_K   — if last K frame-hashes identical -> uniform escape

Per-env RNG seeded via `hash((env_id, run_id, "elite_v0")) & 0xFFFFFFFF`.
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


def _softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = (z - z.max()) / max(T, 1e-6)
    e = np.exp(z)
    return e / e.sum()


class EliteV0:
    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.4,
        use_stage2: bool = True,
        temperature: float | None = None,
        spatial_topk: int | None = None,
        framechange_filter: bool = False,
        framechange_min: float = 0.05,
        stuck_detector_K: int | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
        self.temperature = temperature
        self.spatial_topk = spatial_topk
        self.framechange_filter = framechange_filter
        self.framechange_min = float(framechange_min)
        self.stuck_detector_K = stuck_detector_K

        self.env_id: str | None = None
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.cluster: str = "mixed"
        self.available_actions: list[int] = []
        self._recent_hashes: list[int] = []
        self._last_action: int = -1

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
        self._recent_hashes = []
        self._last_action = -1

    def _frame_to_perception(self, frame_stack: Any) -> tuple[torch.Tensor, np.ndarray]:
        if isinstance(frame_stack, list):
            arr = np.stack([np.asarray(f, dtype=np.int8) for f in frame_stack], axis=0)
        else:
            arr = np.asarray(frame_stack, dtype=np.int8)
            if arr.ndim == 2:
                arr = arr[None, ...]
        perception = reduce_frame_stack(arr).astype(np.float32) / 15.0
        x = torch.from_numpy(perception).unsqueeze(0).to(self.device)
        return x, arr

    @torch.no_grad()
    def _framechange_probs(self, x: torch.Tensor) -> np.ndarray:
        """Returns P(frame changes | s, a) for each a in 0..7."""
        feat = self.model.backbone(x)
        feat_pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (1, 128)
        probs = np.zeros(8, dtype=np.float32)
        for a in range(8):
            onehot = F.one_hot(torch.tensor([a], device=self.device), num_classes=8).float()
            logit = self.model.framechange_head(feat_pool, onehot)
            probs[a] = float(torch.sigmoid(logit).item())
        return probs

    @torch.no_grad()
    def choose_action(self, frame_stack: Any) -> tuple[int, dict[str, int]]:
        assert self.rng is not None and self.np_rng is not None
        x, raw_stack = self._frame_to_perception(frame_stack)
        last_frame = raw_stack[-1]
        h = hash(last_frame.tobytes())

        # Track recent hashes for stuck detector
        if self.stuck_detector_K:
            self._recent_hashes.append(h)
            if len(self._recent_hashes) > self.stuck_detector_K:
                self._recent_hashes.pop(0)

        out = self.model(x)
        action_logits = out["action_logits"][0].cpu().numpy()

        # Mask to available
        masked = np.full(8, -np.inf, dtype=np.float32)
        for a in self.available_actions:
            masked[a] = action_logits[a]

        if self.temperature is not None:
            probs = _softmax(masked, T=self.temperature)
        else:
            probs = _softmax(masked, T=1.0)

        # Confidence fallback to cluster prior
        if float(probs.max()) < self.confidence_threshold:
            prior = self.priors.get(self.cluster, self.priors["mixed"]).copy()
            mask_arr = np.zeros(8, dtype=np.float32)
            for a in self.available_actions:
                mask_arr[a] = 1.0
            prior = prior * mask_arr
            s = prior.sum()
            if s > 0:
                probs = prior / s

        # L1 lookahead: down-weight low-framechange actions
        if self.framechange_filter:
            fc = self._framechange_probs(x)
            fc_masked = np.zeros(8, dtype=np.float32)
            for a in self.available_actions:
                fc_masked[a] = max(fc[a], self.framechange_min)
            probs = probs * fc_masked
            s = probs.sum()
            if s > 0:
                probs = probs / s

        # Stuck detector escape: force uniform over available\{last}
        stuck = False
        if (
            self.stuck_detector_K
            and len(self._recent_hashes) == self.stuck_detector_K
            and len(set(self._recent_hashes)) == 1
        ):
            stuck = True
            choices = [a for a in self.available_actions if a != self._last_action]
            if not choices:
                choices = list(self.available_actions)
            probs = np.zeros(8, dtype=np.float32)
            for a in choices:
                probs[a] = 1.0 / len(choices)

        # Sample action
        if self.temperature is not None or stuck or self.framechange_filter:
            p = probs / probs.sum()
            action_id = int(self.np_rng.choice(8, p=p))
        else:
            action_id = int(probs.argmax())

        data: dict[str, int] = {}
        if action_id == 6:
            sp = out["spatial_logits"][0].cpu().numpy().reshape(-1)  # (4096,)
            if self.spatial_topk is not None and self.spatial_topk > 1:
                k = min(self.spatial_topk, sp.size)
                top_idx = np.argpartition(-sp, k - 1)[:k]
                top_logits = sp[top_idx]
                top_probs = _softmax(top_logits, T=1.0)
                pick = int(self.np_rng.choice(k, p=top_probs))
                idx = int(top_idx[pick])
            else:
                idx = int(sp.argmax())
            y, x_pix = divmod(idx, 64)
            data = {"x": int(x_pix), "y": int(y)}

        self._last_action = action_id
        return action_id, data
