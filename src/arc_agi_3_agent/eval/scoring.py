"""Local RHAE scoring — mirrors arc_agi/scorecard.py:170 exactly.

Tech Report §4.1 (Eq 1, 2, 3) defines:
    S_{l,e} = min(1.15, (h_{l,e} / a_{l,e})^2)              per-level
    E_e    = min(weighted_completed_frac, weighted_avg)     per-env
    T      = mean(E_e) over dataset                         total

The shipping toolkit (arc_agi/scorecard.py:170) multiplies by 100:
    score = ((baseline / taken) ** 2) * 100
    score = min(score, 115.0)

We replicate the 0–115 scoring so harness output is comparable to Kaggle LB.

References:
- Phase 0c §0 OQ5: Kaggle LB displays scorecard.score directly on 0–115 scale.
- Phase 0c §2.2: harness_score_holdout in same units; gate ≥ 10.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelResult:
    """Single level outcome for one env."""

    level_index: int  # 1-indexed (Tech Report w_l = l)
    completed: bool
    actions_taken: int
    baseline_actions: int


def level_score(actions_taken: int, baseline_actions: int, completed: bool) -> float:
    """Per-level RHAE on 0–115 scale.

    Mirrors arc_agi/scorecard.py:170:
        if completed and actions_taken > 0:
            score = ((baseline / taken) ** 2) * 100
            score = min(score, 115.0)
        else:
            score = 0.0
    """
    if not completed:
        return 0.0
    if actions_taken <= 0:
        return 0.0
    raw = ((baseline_actions / actions_taken) ** 2) * 100.0
    return min(raw, 115.0)


def env_score(levels: list[LevelResult]) -> float:
    """Per-env RHAE on 0–115 scale.

    Mirrors arc_agi/scorecard.py:196-206:
        weighted-avg of level scores (weight = 1-indexed level number),
        capped at (sum(weights_of_completed_levels) / sum(all_weights)) * 100.

    Returns 0.0 if levels list is empty.
    """
    if not levels:
        return 0.0
    total_score = 0.0
    total_weight = 0
    completed_weight = 0
    for lvl in levels:
        w = lvl.level_index
        s = level_score(lvl.actions_taken, lvl.baseline_actions, lvl.completed)
        total_score += s * w
        total_weight += w
        if s > 0:
            completed_weight += w
    if total_weight == 0:
        return 0.0
    weighted_avg = total_score / total_weight
    cap = (completed_weight / total_weight) * 100.0
    return min(weighted_avg, cap)


def total_score(env_scores: list[float]) -> float:
    """Total RHAE = mean of env scores. Empty list → 0.0."""
    if not env_scores:
        return 0.0
    return sum(env_scores) / len(env_scores)


def env_score_from_actions(
    level_actions: list[int],
    baseline_actions: list[int],
    levels_completed: int,
) -> float:
    """Convenience wrapper matching scorecard.json field shapes.

    Args:
        level_actions: actions taken per level (full list, including uncompleted)
        baseline_actions: human baseline actions per level (from metadata.json)
        levels_completed: number of completed levels (Tech Report: sequential — completing
            level k means completing levels 1..k)
    """
    levels: list[LevelResult] = []
    for i in range(len(baseline_actions)):
        idx = i + 1  # 1-indexed
        completed = i < levels_completed
        taken = level_actions[i] if i < len(level_actions) else 0
        baseline = baseline_actions[i]
        levels.append(
            LevelResult(
                level_index=idx,
                completed=completed,
                actions_taken=taken,
                baseline_actions=baseline,
            )
        )
    return env_score(levels)
