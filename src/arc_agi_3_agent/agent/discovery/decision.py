"""Stage 3 — decision rule. Explore (novelty/information) vs exploit (strategies)."""

from __future__ import annotations

import numpy as np

from .constants import ACTION6, BG, STUCK_K
from .strategies.resource_tracker import ResourceTracker

W_UNKNOWN = 3.0
W_NOVEL = 1.0
W_NOOP = 4.0
W_CLICKINFO = 1.5
W_LETHAL = 100.0         # avoid actions/classes known to end the episode
N_RANDOM_CLICKS = 6      # extra exploratory click samples when ACTION6 available
COMMIT_STEP = 60         # Task 4: after this many actions, stop chasing unknown effects
                         # (hard explore->commit switch; tuned on train envs only)


class DecisionRule:
    def __init__(self, strategies, explore_only: bool = False) -> None:
        self.strategies = strategies           # priority-sorted
        self.explore_only = explore_only
        self.rng: np.random.Generator | None = None
        self._recent_keys: list[int] = []
        self._last_action: int = -1

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._recent_keys = []
        self._last_action = -1

    def step(self, wm) -> tuple[int, dict]:
        key = wm.state_key()
        self._recent_keys.append(key)
        if len(self._recent_keys) > STUCK_K:
            self._recent_keys.pop(0)

        # Stuck escape: identical last-K states -> forced diversification.
        if len(self._recent_keys) == STUCK_K and len(set(self._recent_keys)) == 1:
            return self._forced_escape(wm)

        if not self.explore_only:
            for strat in self.strategies:
                if strat.applicable(wm):
                    proposal = strat.propose(wm)
                    if proposal is not None:
                        self._last_action = proposal[0]
                        return proposal

        return self._explore(wm)

    # ---- explore ----------------------------------------------------------
    def _explore(self, wm) -> tuple[int, dict]:
        assert self.rng is not None
        opts = wm.options()
        unknown = set(wm.unknown_actions())
        noop = wm.known_noop_actions()
        pressure = ResourceTracker.pressure(wm)
        boost = 1.0 - pressure
        # Task 4 hard commit switch: after COMMIT_STEP actions stop paying to probe
        # UNKNOWN effects (the wandering cost). Click information is NOT suppressed —
        # click envs must keep clicking to find a rewarding class.
        unknown_boost = 0.0 if wm.step_index() > COMMIT_STEP else boost

        candidates: list[tuple[float, int, dict]] = []
        for a in opts.action_ids:
            if a == ACTION6:
                continue
            score = float(self.rng.random())
            if a in unknown:
                score += W_UNKNOWN * unknown_boost
            if a in noop:
                score -= W_NOOP
            if wm.is_lethal(a, None):
                score -= W_LETHAL
            candidates.append((score, a, {}))

        if ACTION6 in opts.action_ids:
            for ct in opts.click_targets:
                score = float(self.rng.random()) + W_CLICKINFO * boost
                if ACTION6 in noop:
                    score -= W_NOOP
                if wm.is_lethal(ACTION6, ct.class_key):
                    score -= W_LETHAL
                candidates.append((score, ACTION6, {"x": int(ct.xy[0]), "y": int(ct.xy[1])}))
            for (x, y) in self._random_clicks(wm):
                score = float(self.rng.random()) + W_CLICKINFO * boost
                if ACTION6 in noop:
                    score -= W_NOOP
                candidates.append((score, ACTION6, {"x": int(x), "y": int(y)}))

        if not candidates:
            self._last_action = 1
            return 1, {}
        candidates.sort(key=lambda t: -t[0])
        _, aid, data = candidates[0]
        self._last_action = aid
        return aid, data

    def _random_clicks(self, wm) -> list[tuple[int, int]]:
        if wm.cur_af is None:
            return []
        grid = wm.cur_af.grid
        ys, xs = np.nonzero(grid != BG)
        if len(ys) == 0:
            ys, xs = np.nonzero(np.ones_like(grid))
        k = min(N_RANDOM_CLICKS, len(xs))
        if k == 0:
            return []
        idx = self.rng.choice(len(xs), size=k, replace=False)
        return [(int(xs[i]), int(ys[i])) for i in idx]

    def _forced_escape(self, wm) -> tuple[int, dict]:
        assert self.rng is not None
        opts = wm.options()
        choices = [a for a in opts.action_ids if a != self._last_action and a != ACTION6]
        if choices:
            a = int(self.rng.choice(choices))
            self._last_action = a
            return a, {}
        if ACTION6 in opts.action_ids:
            clicks = self._random_clicks(wm)
            if clicks:
                x, y = clicks[0]
                self._last_action = ACTION6
                return ACTION6, {"x": x, "y": y}
        a = int(opts.action_ids[0]) if opts.action_ids else 1
        self._last_action = a
        return a, {}
