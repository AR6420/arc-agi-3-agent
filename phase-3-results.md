# Phase 3 Results — Algorithmic Discovery Agent v0

**Date:** 2026-05-29 · **Branch:** `main` · **Hardware:** local (CPU-only agent; no GPU needed)
**Holdout:** vc33, tu93, sk48, lp85, dc22 (never used to tune).

## Gate decision: **FAIL** — `harness_score_holdout = 0.0000` (threshold 10.0)

| metric | value |
|---|---|
| `harness_score_train` (20 envs) | **0.1844** |
| `harness_score_holdout` (5 envs) | **0.0000** |
| envs reaching level 1+ | **1 / 25** (r11l) |
| latency p50 / p95-max | 11 ms / 35 ms (well under 50 ms budget) |
| total wall (25 envs) | 28.4 s |

Per the plan's gate logic (`holdout < 5 → DIAGNOSTIC, do not add next exploit, hand back`), this is a **joint-decision point**, not an autonomous iteration point.

---

## What was built (complete, tested)

Env-agnostic discovery agent under `src/arc_agi_3_agent/agent/discovery/` (no per-env hardcoding; ls20 answer key lives only in the grader):

- **Stage 1 Analyser** — numpy flood-fill segmentation (4- & 8-conn), rotation-invariant shape signatures, stable 3-tier object matching, change detection (moved/appeared/disappeared/recolored/rotated/reshaped + noop/cosmetic/multi_frame), candidate structure (monotone strips, symmetry), per-class click-target generation (the BC-spatial fix).
- **Stage 2 World model** — online action→effect model (stride bucketed by shape_sig), controllable-object + move-vector learning, transformer/attribute tracking, reward-click-class learning, resource-strip promotion, blake2b novelty counts, per-step memory log, `dump_learned_model()` for grading.
- **Stage 3 Decision rule** — explore (unknown-effect + novelty − noop, budget-pressure scaled) vs exploit (priority-ordered strategies), stuck-escape.
- **Stage 4 Strategies** — ResourceTracker (guard), MovementPathfinding (lattice BFS over learned vectors), SelectionUndoTool, ClickToEffect (repeat reward-earning class), AttributeMatching (greedy transformer).
- **Harness wiring** — `--agent discovery` / `discovery_explore_only` / `discovery:<subset>`; full-obs dispatch branch; per-env `archetype_detected`.
- **Tests** — `tests/discovery/` 21 unit tests pass (segmentation, matching, change, options, novelty, world-model scripted sequences, harness dispatch).

---

## Measurements

### Substrate probe (large budget, no early-exit) — substrate WORKS
`scripts/p3_substrate_probe.py` (explore-only, 1500 actions):
- **r11l: reached LEVEL 1 at step 41** (found a rewarding click → archetype=click).
- sk48: 179 distinct states explored, then GAME_OVER (no level).
- vc33 / lp85 / tn36: GAME_OVER at 50 / 78 / 61, 0 levels.

→ The perception + experiment machinery is sound (it explores, segments, finds a rewarding click). The failure is downstream of perception.

### Full agent (all strategies, 25 envs, 5×baseline budget)
- Only **r11l** scored (3.69, 1 level, 86 actions). All other 24 = 0.
- archetype_detected: 4 movement, 1 click, 20 none.
- reached L1+: 1/25.

### ls20 rule-recovery: **1 / 5** (`harness_runs/p3_ls20_recovery/`)
| answer-key fact | recovered |
|---|---|
| transformer changes a tracked attribute | ✅ (343 color/shape events seen) |
| goal defined by target attributes | ❌ |
| reward on match | ❌ |
| a resource depletes per action | ❌ |
| a resource/lives ends the episode at zero | ❌ |

The engine *sees* attribute transformations but never closes the loop (no level completion → no goal/reward anchor; resource-strip detection too fragile under animation noise).

---

## Diagnosis — three structural blockers

1. **Goal is unknowable before the first completion (chicken-and-egg).** The world model only confirms a goal when `levels_completed` increments. Until then, movement/attribute exploits have no target, so MovementPathfinding falls back to novelty-frontier wandering — indistinguishable from exploration. Completing a level by wandering within 5×baseline actions is improbable.

2. **GAME_OVER from blind probing.** Pure-click and mixed envs (vc33, lp85, sk48) die within 50–80 random clicks before a rewarding class is found. One-episode lethality learning can't prevent the death that teaches it. This is the dominant zero-score cause on click envs.

3. **Budget vs discovery tension.** The local scoring budget is 5×baseline (≈30–150 actions/env). Discovery itself consumes much of that. Random reached level 1 on a few envs only at **2000** actions (Kaggle S1) — that budget does not exist locally, so the "explore-only > 0" premise in the plan did not hold (correctly re-attributed here, not a substrate bug).

The spatial-head BC failure IS fixed in principle (click-by-class, not by pixel; r11l proves a rewarding class is found and reused), but per-class clicking still can't beat lethality + the goal-discovery gap inside the budget.

---

## Recommended next steps (joint decision — Hard Rule 4 / plan §4.2)

In rough ROI order:

1. **Goal hypothesis without prior reward.** Treat the most salient/unique static object (rare color, small, distinct from the controllable) as a *candidate goal*; pathfind to it; if reaching it yields reward, lock it — else try the next candidate. Still discovery (hypothesize→test→observe reward), env-agnostic. Directly attacks blocker #1 for movement envs (tu93 holdout is pure movement).

2. **Lethality-aware probing.** Before committing a never-tried click/action, prefer ones acting on objects that already proved safe; treat GAME_OVER as a max-penalty label on the (class, action) and avoid that class. Won't save the first death but caps repeats; pair with undo where available (sk48 holdout has undo).

3. **Stronger resource detection.** Current strip-monotone test is defeated by animation noise. Track a dedicated edge-row/bar region and use longest-monotone-run over a smoothed extent; needed for the resource archetype and 4/5 of the ls20 recovery facts.

4. **Two-phase budget split.** Reserve an explicit fraction of the budget for discovery, the rest for a committed exploit plan, so the agent doesn't wander the whole episode (decision rule has the pressure scalar; make the explore→commit switch hard at a learned fraction).

5. **If the above still miss the gate:** reconsider whether the local 5×baseline budget is the right discovery harness, or whether a small learned goal-recognizer (trained on the 20 train envs' goal cells, env-agnostic features only) is warranted as a *prior over candidate goals* — distinct from the failed BC policy.

---

## Status

Phase 3 v0 is a complete, tested, latency-cheap discovery substrate that provably explores and reuses a learned rewarding click (r11l), but completes only 1/25 levels and scores **holdout 0.0000**. Gate FAIL. No Kaggle submission (non-exempt). Handing back at the diagnostic gate for a joint decision on blockers #1–#3 before any further exploit work.
