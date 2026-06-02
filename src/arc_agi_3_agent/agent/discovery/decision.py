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


W_RELATIONAL = 1.2       # Task C: constant relational bonus (ablation; fixates -> breaks r11l)
W_UNIFIED = 2.2          # v3: PEAK relational bonus for a fresh top candidate (decays per click)


class DecisionRule:
    def __init__(self, strategies, explore_only: bool = False,
                 relational_explore: bool = False, unified: bool = False) -> None:
        self.strategies = strategies           # priority-sorted
        self.explore_only = explore_only
        # Task C ablation: constant relational bonus (fixates on the top class -> starves the
        # pickup->place sequences r11l needs). Kept only for comparison.
        self.relational_explore = relational_explore
        # v3 UNIFIED goal inference: both signals live, reward arbitrates. Pre-reward, click
        # exploration gets a relational bonus that DECAYS per-click-on-that-class — so each
        # ranked candidate gets early attention (GoalProbe's strength) but the bonus backs off
        # after a few clicks, returning variety so multi-step reward chains stay discoverable
        # (B5's strength). After a click earns reward, ClickToEffect exploits it (reward
        # arbitration). No style is locked in by a first guess or static appearance.
        self.unified = unified
        self.rng: np.random.Generator | None = None
        self._recent_keys: list[int] = []
        self._last_action: int = -1
        self._class_clicks: dict[tuple[int, int], int] = {}   # per-class click count (anti-fixation decay)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._recent_keys = []
        self._last_action = -1
        self._class_clicks = {}

    def step(self, wm) -> tuple[int, dict]:
        # Death is non-terminal (death-model.md Verdict C): on GAME_OVER the only action
        # that un-freezes the env is RESET (level_reset -> revive). Self-drive it so the
        # world model records a consistent RESET decision (the retry keeps the learned
        # model via _on_revive). Mirrors the canonical agent loop.
        if wm.cur_af is not None and str(wm.cur_af.state).endswith("GAME_OVER"):
            self._last_action = 0
            return 0, {}

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

        # candidate = (score, action_id, data, click_class_key | None)
        candidates: list[tuple[float, int, dict, tuple[int, int] | None]] = []
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
            candidates.append((score, a, {}, None))

        if ACTION6 in opts.action_ids:
            # Reward arbitration: the relational nudge is only ON while still discovering
            # (no rewarding click class confirmed yet). Once reward is observed, ClickToEffect
            # (a higher-priority strategy) exploits it and the nudge stays out of the way.
            has_click_reward = bool(wm.rewarding_click_classes())
            use_relational = (self.unified or self.relational_explore) and not has_click_reward
            rel_rank = self._relational_rank(wm) if use_relational else {}
            for ct in opts.click_targets:
                score = float(self.rng.random()) + W_CLICKINFO * boost
                if ACTION6 in noop:
                    score -= W_NOOP
                if wm.is_lethal(ACTION6, ct.class_key):
                    score -= W_LETHAL
                if ct.class_key in rel_rank:
                    rank = rel_rank[ct.class_key]
                    if self.unified:
                        # PEAK bonus for a fresh top candidate, DECAYING per click on that
                        # class -> systematic early probing without permanent fixation.
                        clicks = self._class_clicks.get(ct.class_key, 0)
                        score += (W_UNIFIED / (1.0 + rank)) / (1.0 + clicks)
                    else:
                        score += W_RELATIONAL / (1.0 + rank)   # constant (ablation)
                candidates.append((score, ACTION6, {"x": int(ct.xy[0]), "y": int(ct.xy[1])}, ct.class_key))
            for (x, y) in self._random_clicks(wm):
                score = float(self.rng.random()) + W_CLICKINFO * boost
                if ACTION6 in noop:
                    score -= W_NOOP
                candidates.append((score, ACTION6, {"x": int(x), "y": int(y)}, None))

        if not candidates:
            self._last_action = 1
            return 1, {}
        candidates.sort(key=lambda t: -t[0])
        _, aid, data, ck = candidates[0]
        self._last_action = aid
        if ck is not None:                                  # decay this class's future bonus
            self._class_clicks[ck] = self._class_clicks.get(ck, 0) + 1
        return aid, data

    def _relational_rank(self, wm) -> dict[tuple[int, int], int]:
        """class_key -> rank (0 = most goal-like) by relational features (Task C)."""
        if wm.cur_af is None:
            return {}
        from .interaction import rank_candidates_relational
        cands = rank_candidates_relational(wm.objects(), wm.controllable_obj(), wm.cur_af.active_region)
        rank: dict[tuple[int, int], int] = {}
        for i, c in enumerate(cands):
            rank.setdefault(c.class_key, i)
        return rank

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
