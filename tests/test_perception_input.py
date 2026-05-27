"""Tests for perception_input reduction."""

from __future__ import annotations

import numpy as np
import pytest

from arc_agi_3_agent.data.perception_input import reduce_frame_stack, t_distribution_stats


class TestReduceFrameStack:
    def test_single_frame_t1(self):
        # T=1 → first == last, diff == zeros
        single = np.zeros((1, 64, 64), dtype=np.int16)
        single[0, 5:10, 5:10] = 7
        out = reduce_frame_stack(single)
        assert out.shape == (3, 64, 64)
        assert out.dtype == np.int8
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[2], np.zeros((64, 64), dtype=np.int8))
        assert out[0, 5, 5] == 7

    def test_two_frame_diff(self):
        # T=2 with one cell changing
        stack = np.zeros((2, 64, 64), dtype=np.int16)
        stack[0, 10, 20] = 3
        stack[1, 10, 20] = 8
        out = reduce_frame_stack(stack)
        assert out[0, 10, 20] == 3       # first
        assert out[1, 10, 20] == 8       # last
        assert out[2, 10, 20] == 5       # |8 - 3| = 5
        # Untouched cells: all zero in diff channel
        assert out[2, 0, 0] == 0

    def test_three_frame_takes_max_diff(self):
        # T=3 with cell oscillating: 0 → 5 → 2.
        # Per-step diffs: |5-0|=5, |2-5|=3. max = 5.
        stack = np.zeros((3, 64, 64), dtype=np.int16)
        stack[0, 30, 30] = 0
        stack[1, 30, 30] = 5
        stack[2, 30, 30] = 2
        out = reduce_frame_stack(stack)
        assert out[0, 30, 30] == 0       # first
        assert out[1, 30, 30] == 2       # last
        assert out[2, 30, 30] == 5       # max of |5-0|, |2-5|

    def test_long_stack(self):
        # T=50 — make sure no overflow / weirdness.
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 16, size=(50, 64, 64), dtype=np.int16)
        out = reduce_frame_stack(stack)
        assert out.shape == (3, 64, 64)
        # diff channel max value <= 15 (colors are 0..15)
        assert out[2].max() <= 15
        assert out[2].min() >= 0
        # diff channel non-negative
        assert (out[2] >= 0).all()

    def test_empty_stack_defensive(self):
        # Empty frame list → all-zero output, no exception.
        out = reduce_frame_stack(np.zeros((0, 64, 64), dtype=np.int16))
        assert out.shape == (3, 64, 64)
        assert (out == 0).all()

    def test_accepts_list_input(self):
        # Replay JSONL frames are list-of-list-of-list; ensure we accept that shape.
        as_list = [[[0] * 64 for _ in range(64)] for _ in range(1)]
        as_list[0][5][5] = 9
        out = reduce_frame_stack(as_list)
        assert out[0, 5, 5] == 9
        assert (out[2] == 0).all()


class TestTDistributionStats:
    def test_empty(self):
        assert t_distribution_stats([]) == {"n": 0}

    def test_basic(self):
        # 10 frame events: 7 with T=1, 3 with T>1
        s = t_distribution_stats([1, 1, 1, 1, 1, 1, 1, 5, 10, 20])
        assert s["n"] == 10
        assert s["t1_count"] == 7
        assert s["t1_frac"] == pytest.approx(0.7)
        assert s["t_gt1_count"] == 3
        assert s["t_max"] == 20
