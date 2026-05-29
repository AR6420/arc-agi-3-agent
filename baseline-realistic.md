# Realistic Random Baseline (Task 0)

**Date:** 2026-05-29 · **Agent:** `BiasedRandomAgent` · **Budget:** real local harness (5× baseline per env via `max_actions_for_env`, + Phase 2.5 early-exits). Holdout = vc33, tu93, sk48, lp85, dc22.

## Result

| metric | value |
|---|---|
| `harness_score_train` (20) | **0.0000** |
| `harness_score_holdout` (5) | **0.0000** |
| envs reaching level 1+ | **0 / 25** |
| nonzero-scoring envs | **none** |
| median actions/env | 98 (max 200) |

## Interpretation

Biased-random under the **real** 5×-baseline budget scores **0.0000** — it completes no level on any env. This is the honest floor.

**The 0.0581 anchor was wrong for this purpose.** That number came from the Kaggle S1 Save&Run with a ~2000-action budget (≈10–60× the local 5×baseline cap). At the budget the leaderboard actually scores under, random gets nothing — consistent with frontier labs scoring <1% on ARC-AGI-3. Skill-acquisition efficiency, not raw exploration, is what the benchmark measures.

## Consequence for Phase 3

- The "beat random" floor is **> 0.0000**. Discovery v0 already clears it on train (0.1844, via r11l completing level 1) — random does not complete any level.
- The real target remains `harness_score_holdout` rising off 0 (Job #1: clear level 1 on unseen envs within 5×baseline).
- Any env where the discovery agent completes level 1 is a strict, qualitative win over random (which completes none).

This baseline is the "worth submitting" floor for the 2026-06-25 submission decision: submit the stable agent that beats 0.0000 by the clearest margin.
