"""Correctness tests for local RHAE scoring.

These are the correctness gate for the entire harness — if scoring is wrong,
every architecture decision downstream is calibrated against a fake target.
Test the formula directly against worked examples from the Tech Report and
spot-check synthetic trajectories.
"""

from __future__ import annotations

import math

import pytest

from arc_agi_3_agent.eval.scoring import (
    LevelResult,
    env_score,
    env_score_from_actions,
    level_score,
    total_score,
)


# ===== Per-level score =====

class TestLevelScore:
    def test_baseline_equals_taken(self):
        # h/a = 1, squared = 1, * 100 = 100
        assert level_score(actions_taken=50, baseline_actions=50, completed=True) == 100.0

    def test_tech_report_worked_example(self):
        # Tech Report §4.1: h=10, a=100 → (10/100)^2 * 100 = 1.0
        assert level_score(actions_taken=100, baseline_actions=10, completed=True) == pytest.approx(1.0)

    def test_super_human_caps_at_115(self):
        # h/a = 10, squared = 100, * 100 = 10000 → capped at 115
        assert level_score(actions_taken=5, baseline_actions=50, completed=True) == 115.0

    def test_uncompleted_is_zero(self):
        assert level_score(actions_taken=50, baseline_actions=50, completed=False) == 0.0

    def test_zero_actions_taken_is_zero(self):
        # Defensive: avoid divide-by-zero in scorecard.py:170
        assert level_score(actions_taken=0, baseline_actions=50, completed=True) == 0.0

    def test_2x_baseline_yields_25(self):
        # h/a = 0.5, squared = 0.25, * 100 = 25
        assert level_score(actions_taken=100, baseline_actions=50, completed=True) == 25.0


# ===== Per-env score (weighted avg + completion cap) =====

class TestEnvScore:
    def test_empty_levels_is_zero(self):
        assert env_score([]) == 0.0

    def test_all_levels_at_baseline(self):
        # 5 levels, all completed at baseline (score 100 each).
        # Weighted avg = (1*100 + 2*100 + 3*100 + 4*100 + 5*100) / 15 = 100.
        # Completion cap = 15/15 * 100 = 100.
        # min(100, 100) = 100.
        levels = [
            LevelResult(i, True, 50, 50) for i in range(1, 6)
        ]
        assert env_score(levels) == 100.0

    def test_partial_completion_caps_env(self):
        # 5-level env, complete only levels 1+2 at baseline.
        # Per-level scores: [100, 100, 0, 0, 0]
        # Weighted avg = (1*100 + 2*100 + 3*0 + 4*0 + 5*0) / 15 = 300/15 = 20.
        # Completion cap = (1+2)/15 * 100 = 20.
        # min(20, 20) = 20. (Tech Report Eq. 2 worked example.)
        levels = [
            LevelResult(1, True, 50, 50),
            LevelResult(2, True, 50, 50),
            LevelResult(3, False, 0, 50),
            LevelResult(4, False, 0, 50),
            LevelResult(5, False, 0, 50),
        ]
        assert env_score(levels) == pytest.approx(20.0)

    def test_completion_cap_with_super_human(self):
        # Tech Report §4.2 "Cap the maximum per-level score" example:
        # If we ace level 1 with super-human efficiency but fail the rest,
        # the env cap (1/15) prevents the 115 from inflating the env score.
        # 5 levels, complete only level 1, take 5 actions vs 50 baseline:
        #   per-level 1 = (50/5)^2 * 100 = 10000, capped at 115.
        # Weighted avg = (1*115 + 0+0+0+0) / 15 = 115/15 ≈ 7.67
        # Completion cap = 1/15 * 100 ≈ 6.67
        # min(7.67, 6.67) ≈ 6.67 — the cap binds.
        levels = [
            LevelResult(1, True, 5, 50),
            LevelResult(2, False, 0, 50),
            LevelResult(3, False, 0, 50),
            LevelResult(4, False, 0, 50),
            LevelResult(5, False, 0, 50),
        ]
        score = env_score(levels)
        assert score == pytest.approx(1 / 15 * 100, abs=1e-6)

    def test_six_level_env_complete_first_three(self):
        # 6 levels (sp80-shape), complete 3 at baseline.
        # Sum of all weights = 1+2+3+4+5+6 = 21.
        # Completed weight = 1+2+3 = 6.
        # Per-level scores: [100, 100, 100, 0, 0, 0]
        # Weighted avg = (100 + 200 + 300) / 21 = 600/21 ≈ 28.57
        # Completion cap = 6/21 * 100 ≈ 28.57. Binds equally.
        levels = [
            LevelResult(i, i <= 3, 50 if i <= 3 else 0, 50) for i in range(1, 7)
        ]
        assert env_score(levels) == pytest.approx(600 / 21)


# ===== Total score (mean across envs) =====

class TestTotalScore:
    def test_empty(self):
        assert total_score([]) == 0.0

    def test_mean(self):
        assert total_score([10.0, 20.0, 30.0]) == 20.0

    def test_single(self):
        assert total_score([42.0]) == 42.0


# ===== env_score_from_actions (scorecard-shape wrapper) =====

class TestEnvScoreFromActions:
    def test_sp80_full_solve_at_baseline(self):
        # sp80 baselines from metadata.json: [39, 58, 25, 148, 96, 152]
        baselines = [39, 58, 25, 148, 96, 152]
        # Hypothetical: complete all 6 levels at baseline (score 100 each).
        # All weights sum to 21. Completion cap = 100. Weighted avg = 100.
        score = env_score_from_actions(
            level_actions=baselines,  # exactly baseline on every level
            baseline_actions=baselines,
            levels_completed=6,
        )
        assert score == pytest.approx(100.0)

    def test_sp80_partial(self):
        # Complete first 3 levels, with action counts 50/50/30 vs baseline 39/58/25.
        # L1: (39/50)^2 * 100 = 60.84
        # L2: (58/50)^2 * 100 = 134.56, capped at 115
        # L3: (25/30)^2 * 100 = 69.44
        # Weighted = 1*60.84 + 2*115 + 3*69.44 + 4*0 + 5*0 + 6*0 = 60.84 + 230 + 208.33 = 499.17
        # Total weight = 21. avg = 499.17/21 ≈ 23.77
        # Cap = (1+2+3)/21 * 100 ≈ 28.57. min → 23.77.
        score = env_score_from_actions(
            level_actions=[50, 50, 30, 0, 0, 0],
            baseline_actions=[39, 58, 25, 148, 96, 152],
            levels_completed=3,
        )
        expected_l1 = (39 / 50) ** 2 * 100
        expected_l2 = 115.0
        expected_l3 = (25 / 30) ** 2 * 100
        expected_avg = (1 * expected_l1 + 2 * expected_l2 + 3 * expected_l3) / 21
        expected_cap = 6 / 21 * 100
        assert score == pytest.approx(min(expected_avg, expected_cap))

    def test_no_levels_completed(self):
        # Random agent that fails level 1 → score is 0.
        score = env_score_from_actions(
            level_actions=[200, 0, 0, 0, 0, 0],
            baseline_actions=[39, 58, 25, 148, 96, 152],
            levels_completed=0,
        )
        assert score == 0.0
