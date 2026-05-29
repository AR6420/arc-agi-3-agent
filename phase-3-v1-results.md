# Phase 3 v1 Results — Budget-Efficient Discovery

**Date:** 2026-05-29 · **Branch:** `main` · Holdout = vc33, tu93, sk48, lp85, dc22 (never tuned on).
**Goal (recalibrated):** clear level 1 on unseen envs within the real 5×-baseline budget. No leaderboard-rank target; rigorous honest effort is the objective.

## Gate decision: **FAIL** — `harness_score_holdout = 0.0000`

| config | train | holdout | envs L1 | nonzero |
|---|---:|---:|---:|---|
| realistic random (Task 0) | 0.0000 | 0.0000 | 0/25 | none |
| discovery v0 (Phase 3) | 0.1844 | 0.0000 | 1/25 | r11l 3.69 |
| v1 +saliency (Task 2) | 0.1125 | 0.0000 | 1/25 | r11l 2.25 |
| v1 full (Tasks 1–5) | **0.1581** | **0.0000** | 1/25 | r11l 3.16 |

**The agent beats the realistic random floor on train** (completes a level random never does) but holdout stays at 0. r11l is the only env that completes level 1 in every configuration.

---

## Task 0 — the honest floor (`baseline-realistic.md`)

Biased-random under the **real** 5×-baseline budget = **0.0000 / 0.0000**, 0/25 levels. The previously-cited 0.0581 was a 2000-action Kaggle Save&Run — ≈10–60× the budget the leaderboard actually scores under. At real budget, random gets nothing. This is the bar a working agent must beat (>0).

## Tasks 1–5 (built, 27 unit tests pass)

1. **Active region + relational saliency** (`saliency.py`): non-bg bbox play-area; per-object saliency = color-rarity + size-atypicality + isolation + dissimilarity-from-controllable → ranked CANDIDATES (no semantic labels — avoids the Approach-A trap). 5 unit tests.
2. **Saliency goal-hypothesis** (movement strategy): pursue the top untried salient object as a candidate goal via lattice BFS to a free neighbor; reach-without-reward → demote, try next. Attacks blocker 1 (goal chicken-and-egg).
3. **Lethality-aware probing**: GAME_OVER labels the triggering (action, click-class) as lethal; decision rule avoids it. (See structural caveat below.)
4. **Two-phase budget split**: hard commit switch at `COMMIT_STEP` actions stops paying to probe *unknown actions* (the wandering cost) while preserving click information.
5. **Cross-level rule persistence**: the learned action→effect / controllable / move-vectors / goal / reward-classes / lethal model is KEPT across level boundaries (only layout-specific state resets). Verified by `test_persistence.py`.

**Latency:** p95 ~35 ms (well under 50 ms). **ls20 rule-recovery: 1/5** (sees 343 transformer events; never completes a level → goal/reward/resource unrecovered) — unchanged from v0.

---

## Honest findings

### 1. No v1 task unlocked a new env at this budget
Holdout stayed 0.000 through every task. The saliency goal-hypothesis (Task 2) requires the **controllable object + move-vectors** to be confirmed first; that confirmation fires on only 3–4 envs and, even when it does, reaching the most-salient object does not trigger reward (the salient object often isn't the goal, or the goal needs more than arrival). Movement envs (incl. tu93 holdout) did not begin scoring.

### 2. A structural limit on in-episode lethality (Task 3)
The harness **breaks on GAME_OVER** and each env is a single `make()`. So a single-life death ends the run — the lethal label it produces has no in-episode future to protect. Lethality learning only helps via (a) an observable *non-fatal* life-loss before the final death, or (b) undo/reset recovery — and the harness can't continue past GAME_OVER to use reset-recovery either. Net in-episode benefit at this budget: ~0. The mechanism is correct and would matter in a multi-life / continue-after-death setting; it is documented, not removed.

### 3. The r11l train delta (3.69 → 3.16) is RNG-ordering, not regression
Refactoring the explore loop changed the interleaving of `np_rng.random()` (candidate scoring) and `np_rng.choice()` (random-click sampling) on the shared generator, shifting r11l's deterministic-but-arbitrary click path. Same seed, different stream order. Not a substantive degradation; holdout unaffected.

### 4. The wall is real and expected
Random = 0 at real budget; frontier labs score <1% on ARC-AGI-3; the board top is ~1.17. From-scratch, single-episode discovery that must *both* infer an unknown goal *and* execute it within 5×baseline is the exact difficulty the benchmark is built to expose. Our substrate is sound (perception, segmentation, action→effect, click-by-class — r11l proves the loop closes); the missing ingredient is budget-efficient goal inference, which humans get from lifetime priors.

---

## What stands (correct, reusable, tested)

- A complete env-agnostic discovery agent: analyser (segmentation, change, structure, saliency, active-region) + world model (action→effect, controllable/selection, attribute/transformer, resource, reward-click, lethal, novelty, cross-level persistence) + decision rule (explore/commit/lethality) + 5 pluggable strategies + harness wiring + ls20 grader.
- 27 unit tests; latency 35 ms p95; fully seeded/deterministic; no per-env hardcoding (verified holdout never tuned; ls20 key only in the grader).
- It strictly beats the realistic random floor on train (1 level completed vs 0).

---

## Submission posture (per updated CLAUDE.md)

- The realistic floor is **0.0000**. The "worth submitting" bar is the stable agent that beats it by the clearest margin; discovery does (train 0.16, 1 env completes a level).
- **Hard submit date 2026-06-25** unless `harness_score_holdout ≥ ~10` triggers earlier (it has not). One submission/day, explicit per-message auth.

## Recommended next directions (joint)

1. **Goal inference without arrival-reward** — the deepest lever. A budget-efficient hypothesis test: try *interacting* with (not just reaching) the top salient candidates and watch for any reward/level signal; rank candidate goals by an env-agnostic prior (centrality, uniqueness, "container/target-like" relational features) rather than pure saliency.
2. **Continue-after-death harness** (if rules permit) so lethality learning has a future to protect — or detect a lives indicator robustly to learn lethality before the final death.
3. **Stronger resource/lives detection** (the deferred secondary) — needed for 4/5 ls20 facts and for #2.

Phase 3 v1 is a rigorous, tested, honest attempt that establishes the true floor (random=0), beats it, and isolates exactly why holdout stays 0. Handing back.
