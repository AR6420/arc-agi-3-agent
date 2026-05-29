"""Guards the harness dispatch contract: DiscoveryAgent gets the full obs."""

from __future__ import annotations

import numpy as np

from arc_agi_3_agent.agents.random_agent import BiasedRandomAgent
from arc_agi_3_agent.agent.discovery.agent import DiscoveryAgent
from arc_agi_3_agent.eval.harness import dispatch_choose


class FakeObs:
    def __init__(self):
        self.frame = [np.zeros((64, 64), dtype=np.int8)]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.available_actions = [1, 2, 3, 4, 6]


def test_discovery_receives_full_obs():
    captured = {}

    class StubDiscovery(DiscoveryAgent):
        def choose_action(self, obs):
            captured["arg"] = obs
            return 1, {}

    obs = FakeObs()
    last = np.zeros((64, 64), dtype=np.int8)
    dispatch_choose(StubDiscovery(), obs, last)
    arg = captured["arg"]
    assert hasattr(arg, "levels_completed") and hasattr(arg, "state")
    assert hasattr(arg, "available_actions")


def test_random_receives_last_frame_only():
    captured = {}

    class StubRandom(BiasedRandomAgent):
        def choose_action(self, frame):
            captured["arg"] = frame
            return 1, {}

    obs = FakeObs()
    last = np.zeros((64, 64), dtype=np.int8)
    dispatch_choose(StubRandom(), obs, last)
    arg = captured["arg"]
    assert isinstance(arg, np.ndarray) and arg.shape == (64, 64)
