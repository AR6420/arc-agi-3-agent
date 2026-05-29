"""DiscoveryAgent — wires Analyser -> WorldModel -> DecisionRule -> Strategies.

Standalone harness agent (mirrors BiasedRandomAgent / EliteV0 interface) but its
choose_action receives the FULL obs (it needs levels_completed / state /
available_actions, not just the frame).
"""

from __future__ import annotations

import random

import numpy as np

from .decision import DecisionRule
from .strategies import BUILD_ORDER, build_strategies
from .world_model import WorldModel

AGENT_VERSION = "discovery_v0"


def per_env_seed(env_id: str, run_id: int = 0) -> int:
    return hash((env_id, run_id, AGENT_VERSION)) & 0xFFFFFFFF


class DiscoveryAgent:
    def __init__(self, *, explore_only: bool = False, enabled_strategies: list[str] | None = None) -> None:
        self.explore_only = explore_only
        self.enabled_strategies = enabled_strategies if enabled_strategies is not None else list(BUILD_ORDER)
        self.wm = WorldModel()
        self.strategies = build_strategies(self.enabled_strategies)
        self.decider = DecisionRule(self.strategies, explore_only=explore_only)
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.env_id: str | None = None

    def reset_for_env(self, env_id: str, available_actions, run_id: int = 0) -> None:
        seed = per_env_seed(env_id, run_id)
        self.env_id = env_id
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.wm.reset(env_id, available_actions, run_id)
        for s in self.strategies:
            s.reset(env_id, available_actions, run_id)
        self.decider.reset(self.np_rng)

    def choose_action(self, obs) -> tuple[int, dict]:
        """obs = full FrameDataRaw (.frame, .state, .levels_completed, .available_actions)."""
        self.wm.observe(obs)
        action_id, data = self.decider.step(self.wm)
        self.wm.record_decision(action_id, data)
        return int(action_id), data
