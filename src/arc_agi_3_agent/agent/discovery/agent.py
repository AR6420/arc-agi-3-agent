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
    # Phase 3 v2 fix: process-stable seed (builtin hash() is per-process salted).
    from arc_agi_3_agent.seeding import stable_seed
    return stable_seed(env_id, run_id, AGENT_VERSION)


class DiscoveryAgent:
    def __init__(self, *, explore_only: bool = False, enabled_strategies: list[str] | None = None,
                 goal_by_interaction: bool = False, relational_explore: bool = False,
                 unified_goal: bool = False) -> None:
        self.explore_only = explore_only
        self.goal_by_interaction = goal_by_interaction
        self.relational_explore = relational_explore
        # v3 unified goal inference: both signals live, reward-arbitrated, anti-fixation
        # (decaying relational bonus on click exploration). The single submission candidate.
        self.unified_goal = unified_goal
        if enabled_strategies is not None:
            self.enabled_strategies = enabled_strategies
        else:
            self.enabled_strategies = list(BUILD_ORDER)
            # Task C (aggressive ablation): GoalProbe as a dedicated probing strategy.
            # NOT used by the unified agent (it monopolises the decision pre-reward).
            if goal_by_interaction and "goal_probe" not in self.enabled_strategies:
                self.enabled_strategies.append("goal_probe")
        self.wm = WorldModel()
        self.strategies = build_strategies(self.enabled_strategies)
        self.decider = DecisionRule(self.strategies, explore_only=explore_only,
                                    relational_explore=relational_explore, unified=unified_goal)
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.env_id: str | None = None

    def reset_for_env(self, env_id: str, available_actions, run_id: int = 0) -> None:
        seed = per_env_seed(env_id, run_id)
        self.env_id = env_id
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # v3 boundary: episode memory is per-env and DISPOSABLE. Re-instantiate the world
        # model / strategies / decider so NO learned state (effects, controllable, goal,
        # reward classes, lethal, resource) can leak across envs — only the env-agnostic
        # PROCEDURE crosses env boundaries. Carrying episode memory across envs would be the
        # BC overfitting failure. (Strongest no-leak guarantee: a fresh object graph per env.)
        self.wm = WorldModel()
        self.strategies = build_strategies(self.enabled_strategies)
        self.decider = DecisionRule(
            self.strategies, explore_only=self.explore_only,
            relational_explore=self.relational_explore, unified=self.unified_goal,
        )
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
