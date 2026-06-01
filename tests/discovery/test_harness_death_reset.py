"""B1 isolation tests — RESET-and-continue death handling + action accounting.

Drives `_run_episode` against a fake engine that mimics the Verdict-C semantics
proven in death-model.md:
  - a non-RESET action while GAME_OVER returns a frozen empty frame (state unchanged);
  - RESET while GAME_OVER does a level_reset -> revives (state NOT_FINISHED), preserves
    completed-level score, and does NOT cost an engine `_action_count` increment
    (but the harness/scorecard counts it as 1 action — CLAUDE.md §2.4);
  - WIN (all levels) is the only non-death terminal.

No SDK import: the engine, GameAction and the agent are all fakes.
"""

from __future__ import annotations

import numpy as np

from arc_agi_3_agent.eval.harness import _run_episode


class FakeGameAction:
    def __init__(self, aid: int) -> None:
        self.id = aid

    @classmethod
    def from_id(cls, aid: int) -> "FakeGameAction":
        return cls(int(aid))


class FakeObs:
    def __init__(self, frame, state: str, levels_completed: int) -> None:
        self.frame = frame
        self.state = state
        self.levels_completed = levels_completed
        self.available_actions = [1, 2, 9]


class FakeEnv:
    """Minimal engine faithful to Verdict C.

    Action semantics:
      1 ("advance") -> complete the current level (score += 1); WIN if all levels done.
      9 ("die")     -> lose() -> GAME_OVER.
      0 (RESET)     -> level_reset: revive current level, preserve score.
      other         -> no-op.
    """

    def __init__(self, n_levels: int) -> None:
        self.n_levels = n_levels
        self.score = 0
        self.state = "GameState.NOT_FINISHED"
        self._tick = 0

    def _frame(self):
        self._tick += 1
        g = np.zeros((64, 64), dtype=np.int8)
        g[0, 0] = self._tick % 16  # vary so distinct_states tracking sees movement
        return [g]

    def reset(self):
        self.score = 0
        self.state = "GameState.NOT_FINISHED"
        return FakeObs(self._frame(), self.state, self.score)

    def step(self, ga, data=None):
        aid = ga.id
        if aid == 0:  # RESET
            if self.state.endswith("GAME_OVER") or self.state.endswith("WIN"):
                # level_reset (or full_reset on WIN — irrelevant here): revive, keep score.
                self.state = "GameState.NOT_FINISHED"
            return FakeObs(self._frame(), self.state, self.score)
        if self.state.endswith("GAME_OVER") or self.state.endswith("WIN"):
            # frozen: non-RESET returns empty frame, state unchanged.
            return FakeObs([], self.state, self.score)
        if aid == 1:  # advance / complete current level
            self.score += 1
            self.state = "GameState.WIN" if self.score >= self.n_levels else "GameState.NOT_FINISHED"
        elif aid == 9:  # die
            self.state = "GameState.GAME_OVER"
        else:
            self.state = "GameState.NOT_FINISHED"
        return FakeObs(self._frame(), self.state, self.score)


class ScriptedAgent:
    """Returns a fixed action sequence; default action once exhausted. Neither a
    BiasedRandomAgent nor DiscoveryAgent, so dispatch routes obs.frame to it (ignored)."""

    def __init__(self, seq, default=2) -> None:
        self.seq = list(seq)
        self.default = default
        self.i = 0

    def choose_action(self, _frame):
        a = self.seq[self.i] if self.i < len(self.seq) else self.default
        self.i += 1
        return int(a), {}


def _drive(env, agent, baseline_actions, max_actions):
    obs = env.reset()
    return _run_episode(env, agent, obs, baseline_actions, max_actions, FakeGameAction)


# --------------------------------------------------------------------------- #

def test_win_after_death_continues_and_preserves_levels():
    # advance->lvl1, die, (forced RESET), advance->lvl2->WIN
    env = FakeEnv(n_levels=2)
    agent = ScriptedAgent(seq=[1, 9, 2, 1])
    r = _drive(env, agent, baseline_actions=[10, 10], max_actions=5 * 20 + 50)

    assert r["termination"] == "win"
    assert r["levels_completed_seen"] == 2          # level-1 survived the death
    assert r["actions_taken"] == 4                  # advance, die, reset, advance
    assert r["resets_taken"] == 1
    assert r["n_deaths"] == 1
    assert r["first_death_step"] == 2               # GAME_OVER seen after 2 actions


def test_death_is_not_terminal_runs_to_budget():
    # always die -> die/reset/die/reset... never breaks on first death.
    env = FakeEnv(n_levels=1)
    agent = ScriptedAgent(seq=[], default=9)        # every choice is "die"
    max_actions = 5 * 10 + 50                        # 100
    r = _drive(env, agent, baseline_actions=[10], max_actions=max_actions)

    assert r["termination"] == "budget_exhausted"
    assert r["actions_taken"] == max_actions         # ran the FULL budget, not 1
    assert r["n_deaths"] == 50                        # die on odd actions
    assert r["resets_taken"] == 50                    # forced RESET on even actions
    assert r["levels_completed_seen"] == 0
    assert r["first_death_step"] == 1
    # RESET is counted against the budget (scorecard.inc_reset_count bumps actions too).
    assert r["actions_taken"] == r["n_deaths"] + r["resets_taken"]


def test_normal_win_without_death_regression():
    env = FakeEnv(n_levels=2)
    agent = ScriptedAgent(seq=[1, 1])
    r = _drive(env, agent, baseline_actions=[10, 10], max_actions=5 * 20 + 50)

    assert r["termination"] == "win"
    assert r["actions_taken"] == 2
    assert r["n_deaths"] == 0
    assert r["resets_taken"] == 0
    assert r["levels_completed_seen"] == 2


def test_reset_revives_to_not_finished_not_frozen():
    # After a death the harness must issue RESET (revive), never a frozen non-RESET.
    env = FakeEnv(n_levels=2)
    agent = ScriptedAgent(seq=[9], default=2)  # die once, then no-ops
    r = _drive(env, agent, baseline_actions=[5, 5], max_actions=5 * 10 + 50)

    # final obs is a live NOT_FINISHED frame (revived), with a non-empty frame.
    assert str(r["obs"].state).endswith("NOT_FINISHED")
    assert r["n_deaths"] >= 1
    assert r["resets_taken"] >= 1
    # never won, never wasted the whole budget on a frozen state without resetting
    assert r["distinct_states"] >= 2
