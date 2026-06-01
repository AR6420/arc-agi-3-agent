"""S1 plumbing-validation random agent.

Phase 0c §3.1 S1 spec:
- Biased-random over each env's `available_actions` (constant per env — Phase 0b Item 5).
- ACTION6 click → uniform over non-background cells (background = color 0); fall back to
  uniform 64×64 if no non-background cells exist.
- RESET avoided in voluntary action choice (Phase 0b S3: each non-first RESET costs 1 action).
- Seed-deterministic per (env_id, run_id, version).

Phase 1 Issues 2 + 3:
- Logs scorecard.actions + scorecard.resets after every step so we can verify
  RESET counter behavior (Issue 2) and ACTION7 cost (Issue 3) post-submission.
- In OFFLINE mode these stay zero (Phase 0b S5); in COMPETITION on Kaggle they increment.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

AGENT_VERSION = "random_agent_v1"


def per_env_seed(env_id: str, run_id: int = 0) -> int:
    """Deterministic per-env seed per Phase 0c §3.2 spec.

    Phase 3 v2 fix: builtin hash() is per-process salted, so the old
    `hash((env_id, run_id, AGENT_VERSION)) & 0xFFFFFFFF` drew a different seed every
    process launch (measurements were not reproducible). blake2b is process-stable.
    """
    from arc_agi_3_agent.seeding import stable_seed
    return stable_seed(env_id, run_id, AGENT_VERSION)


class BiasedRandomAgent:
    """Stand-alone agent (no subclass of the bundled `Agent` ABC — see notes).

    Notes on the harness contract:
    - The local eval harness drives this agent directly via `choose_action(obs, env_info)`.
    - For Kaggle submission, the same `choose_action` body is inlined into the notebook
      template (Phase 0c §5.1) because importing the bundled `Agent` ABC adds overhead and
      we don't need its lifecycle hooks (recorder, AgentOps tracing, etc.).
    - The agent is stateless across actions except for the seeded RNG. `reset_for_env`
      is called once per env to bind the seed.
    """

    def __init__(self) -> None:
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.env_id: str | None = None
        self.available_actions: list[int] | None = None
        self.reset_count: int = 0

    def reset_for_env(self, env_id: str, available_actions: list[int], run_id: int = 0) -> None:
        """Bind seed and cache the per-env action set."""
        seed = per_env_seed(env_id, run_id)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.env_id = env_id
        # Strip RESET (id 0) from voluntary choice — let the harness emit RESET only
        # on natural terminal states.
        self.available_actions = [a for a in available_actions if a != 0]
        self.reset_count = 0
        if not self.available_actions:
            # Defensive: if env exposes only RESET, fall back to ACTION1.
            self.available_actions = [1]

    def choose_action(self, latest_frame_last: np.ndarray) -> tuple[int, dict[str, int]]:
        """Pick an action.

        Args:
            latest_frame_last: (64, 64) int array of the most recent post-action frame
                (channel-1 of the perception input — sufficient for non-background lookup).

        Returns:
            (action_id, data) where data is {} for non-click actions and {"x", "y"} for ACTION6.
        """
        assert self.rng is not None and self.available_actions is not None
        action_id = self.rng.choice(self.available_actions)
        data: dict[str, int] = {}
        if action_id == 6:
            data = self._sample_click_xy(latest_frame_last)
        return action_id, data

    def _sample_click_xy(self, frame_last: np.ndarray) -> dict[str, int]:
        """Uniform over non-background cells (background = color 0). Fall back to full uniform."""
        assert self.np_rng is not None
        try:
            non_bg = np.argwhere(frame_last != 0)
            if len(non_bg) > 0:
                # argwhere returns (row, col) which is (y, x).
                idx = int(self.np_rng.integers(0, len(non_bg)))
                y, x = int(non_bg[idx, 0]), int(non_bg[idx, 1])
                return {"x": x, "y": y}
        except Exception:
            pass
        # Fall back: full uniform over 64×64.
        x = int(self.np_rng.integers(0, 64))
        y = int(self.np_rng.integers(0, 64))
        return {"x": x, "y": y}


def max_actions_for_env(baseline_actions_per_level: list[int]) -> int:
    """Phase 0c §3.1: MAX_ACTIONS = 5 * sum(baseline) + 50 safety margin."""
    return 5 * sum(baseline_actions_per_level) + 50
