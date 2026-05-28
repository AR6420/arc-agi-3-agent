# Phase 2.5 Results — Inference Fixes + Bigger Retrain

**Date:** 2026-05-28
**Branch:** `main`
**Outcome:** **Gate FAIL** — `harness_score_holdout = 0.0000` across every config tried (12 inference variants × 2 weight sets).

---

## TL;DR

The diagnosis at end of Phase 2 (deterministic argmax → stuck-state) was correct but incomplete. Stochastic sampling, top-k clicks, framechange filtering, and a stuck-state escape each move some **train** envs off zero, but no combination touches the **holdout 5** (vc33, tu93, sk48, lp85, dc22). Retraining with a 6.4× larger backbone (560K → 3.6M params), 2× spatial loss weight (0.3 → 0.6), and longer schedule (50 → 250 epoch cap with plateau early-stop) gave virtually identical val_action_acc (0.65 → 0.65) and identical in-env behavior (0/5 holdout, fewer train envs scoring).

**The harness gate is not crossable with the current architecture + training data.** Recommendation: hand back for joint architecture decision.

---

## Part 1 — Harness early-exit fix

Old behavior: skip env if level 1 not reached in **50 actions** (Phase 0c §2.5).

New behavior (Phase 2.5):
1. Per-env level-1 budget = `max(150, 3 × baseline_actions[0])`.
2. Progress-based exit: hash post-action frames; if no new state AND no level advance in 40 consecutive actions, exit.

Validation: re-ran the unchanged argmax agent against the new harness. Result identical to Phase 2: train 0.924, holdout 0.000. Harness change does not perturb the agent's measured score. ✅

---

## Part 2 — Inference-policy sweep (v1 weights, 560K params)

All 11 configs, single seed, sweep wall ~150 s total.

| config | train | **holdout** | nz train | nz holdout |
|---|---:|---:|---:|---:|
| argmax_baseline | 0.924 | **0.000** | 2/20 | 0/5 |
| T0.5 | 1.436 | **0.000** | 5/20 | 0/5 |
| T1.0 | 0.805 | **0.000** | 2/20 | 0/5 |
| T1.5 | 0.119 | **0.000** | 2/20 | 0/5 |
| T2.0 | 0.091 | **0.000** | 1/20 | 0/5 |
| T0.5_topk5 | 2.236 | **0.000** | 9/20 | 0/5 |
| T0.5_topk20 | 1.333 | **0.000** | 8/20 | 0/5 |
| T0.5_topk50 | 1.065 | **0.000** | 8/20 | 0/5 |
| T0.5_topk5_fc | 2.124 | **0.000** | 4/20 | 0/5 |
| **T0.5_topk5_stuck8** | **3.552** | **0.000** | 5/20 | 0/5 |
| T0.5_topk5_fc_stuck8 | 2.134 | **0.000** | 5/20 | 0/5 |

**Observations:**
- T0.5 strictly dominates T≥1.0 (sharper sampling helps; the model has signal, it just gets stuck).
- topk5 > topk20 > topk50 (the spatial head's top-5 pixels are useful; top-50 is noise).
- L1 framechange filter HURTS slightly (2.236 → 2.124 → 1.248). The framechange head's per-action prediction is noisy enough that filtering trims useful actions.
- Stuck-state escape (K=8) gives the biggest single jump on train: 2.24 → 3.55. Direct evidence of stuck-loop dominance in failure cases.
- **Holdout stays at 0.000 throughout.** No inference policy can compensate for what the model itself cannot represent for these 5 envs.

**Best v1 nonzero envs (T0.5_topk5_stuck8):** g50t 35.71, sc25 28.57, r11l 4.76, lf52 1.82, ls20 0.18. All train; none holdout.

---

## Part 3 — Decision gate

`harness_score_holdout = 0.000` ≪ 5.0 → "retraining mandatory" branch per spec. Proceeded to Part 4.

---

## Part 4.0 — Click-tolerance probe

Tested vc33 (pure_click holdout) and tn36 (pure_click train) clicking at the known-good replay click + offsets (±1, ±2, ±3, on both x and y).

| env | target | (0,0) changed | ±1 changed | ±2 changed | ±3 changed |
|---|---|---|---|---|---|
| vc33 | (38, 9) | ✓ diff=1 | ✓ diff=1 | ✓ diff=1 | ✓ diff=1 |
| tn36 | (54, 48) | ✓ diff=1 | ✓ diff=1 | ✓ diff=1 | ✓ diff=1 |

Diff=1 across the board likely indicates a cursor highlight, not progress. **Inconclusive** but does not strongly justify retraining the spatial head as the priority. Top-k sampling already validated as a useful inference-time fix.

---

## Part 4.1 — Retrain (v2 weights, 3.6M params)

### Architecture changes
- Backbone channels: `[32, 64, 64, 128]` → `[64, 128, 128, 192, 192, 256]` (4 → 6 effective blocks).
- Output channels: 128 → 256.
- Spatial head: matched larger upsampling chain.
- Total params: 559,594 → **3,602,378** (6.4× growth).
- VRAM @ batch 128: 1,016 MB → **2,427 MB** (still 5× under 12 GB ceiling).

### Training
- Stage 1: spatial_w 0.3 → **0.6**, epochs 50 → 250, plateau early-stop 10 epochs.
- Stage 2: framechange fine-tune unchanged.

### Result
- Early-stopped at **epoch 38** (val plateau). Wall: 6,791 s ≈ 113 min.
- **Best val_action_acc: 0.6513** (v1: 0.6504; Δ +0.001 — within noise).
- val_action6_hit_rate: 0.047 (v1: 0.060; Δ **−0.013**, the spatial head is slightly WORSE despite 2× loss weight).
- val_framechange_acc: 0.9688 (v1: 0.9676).
- Stage 2 AUC: 0.9142 (v1: 0.8975; Δ +0.017).

**The bigger backbone added no learnable signal.** The same plateau, the same spatial weakness, despite 6× the parameters and 2× the spatial gradient. This is consistent with the data-quality / task-formulation being the bottleneck, not model capacity.

---

## Part 4.2 — Re-evaluate gate (v2 weights)

Full 11-config sweep with v2 weights:

| config | train | **holdout** | nz train | nz holdout |
|---|---:|---:|---:|---:|
| argmax_baseline | 0.139 | **0.000** | 1/20 | 0/5 |
| T0.5 | 1.600 | **0.000** | 5/20 | 0/5 |
| T1.0 | 0.675 | **0.000** | 2/20 | 0/5 |
| T1.5 | 0.000 | **0.000** | 0/20 | 0/5 |
| T2.0 | 0.000 | **0.000** | 0/20 | 0/5 |
| T0.5_topk5 | 1.666 | **0.000** | 6/20 | 0/5 |
| **T0.5_topk20** | **1.866** | **0.000** | 6/20 | 0/5 |
| T0.5_topk50 | 1.627 | **0.000** | 5/20 | 0/5 |
| T0.5_topk5_fc | 1.248 | **0.000** | 5/20 | 0/5 |
| T0.5_topk5_stuck8 | 1.619 | **0.000** | 6/20 | 0/5 |
| T0.5_topk5_fc_stuck8 | 1.248 | **0.000** | 5/20 | 0/5 |

**v2 strictly worse than v1 on the harness.** Best v2 train (1.87) < best v1 train (3.55). Best v2 nonzero envs (T0.5_topk20): ar25 3.13, ft09 4.76, g50t 10.71, ls20 3.57, m0r0 0.85, tr87 14.29.

Note: the **set** of scoring envs differs between v1 and v2 (v1 hit r11l + sc25 + lf52 which v2 zeros; v2 hit ar25 + ft09 + m0r0 + tr87 which v1 zeros). Suggests both models have learned different (poor) parts of the policy space — neither learned the holdout 5.

---

## Diagnosis — why the gate is uncrossable from here

1. **Holdout 5 are categorically unreachable.** Across 22 (11 configs × 2 weight sets) harness runs, no holdout env reached level 1. This is not variance — it's structural.

2. **More capacity ≠ more performance.** v1 (560K params) outperformed v2 (3.6M params) on the harness despite v2 having higher val metrics. The training objective (action-distribution match on human BC) is decoupled from the in-env objective (reach level 1, then progress).

3. **The 0.65 val_action_acc ceiling is intrinsic to the task.** The model can predict 65% of human actions correctly. The other 35% are not "wrong" — they're context-dependent decisions (the human's mental model of the puzzle, what level they're solving, momentum from previous moves). The model has no access to that context.

4. **The framechange head is well-trained but inference-time filtering hurts.** AUC 0.91 on the held-out val set, yet using it to filter actions trims away useful exploratory moves. This suggests calibration mismatch: synth-derived AUC is good, but per-action confidence at inference is overconfident in the wrong direction.

5. **Spatial head is the persistent weak link.** Hit rate 0.05–0.06 regardless of loss weight or backbone size. The model knows roughly *where* (top-k captures it within 5 pixels often) but not *exactly* where.

---

## What I am NOT iterating without explicit user sign-off (per spec Hard Rule 4)

The following architecture changes are NOT being applied autonomously:
- Per-cluster heads (separate action-type head for pure_click / pure_movement / mixed).
- Sequence-aware perception (T-stack temporal conv instead of OQ7's 3-channel reduction).
- A non-BC training objective (RL fine-tune, reward shaping, level-1-reach pretext).
- A different `perception_input` encoding (e.g. raw last frame instead of OQ7).
- A frozen-backbone-then-policy-tune curriculum.

---

## Recommended next steps for joint discussion

In order of expected ROI vs cost:

### 1. **Per-cluster action heads** (cheap, ~1h retrain)
Train three separate action-type heads, one per env cluster. The cluster signal is strong (KL 20.4) but the current shared head averages over them. Hypothesis: pure_click and pure_movement env decisions are different decision processes; sharing dilutes both. Use the cluster-prior fallback at inference.

### 2. **Replay-only validation, env-only test** (free, just analysis)
Confirm whether real human action acc on the holdout 5 envs equals the train envs (it should, since the model never saw them). If holdout action acc ≪ train action acc, the 0.65 plateau is masking generalization failure. If equal, the gap is policy-vs-distribution-match.

### 3. **Reward-shaped fine-tune** (medium, ~3h)
After Stage 1 BC, run a short RL or imitation-with-success-bias phase: weight gradients by whether the trajectory reached level 1+. Targets the in-env objective directly. Risky (RL with 12 GB VRAM, sparse reward) but a known recipe for closing BC-policy gaps.

### 4. **Sequence-aware perception** (expensive, ~6h retrain)
Replace OQ7 reduce with a 4-frame stack (most recent 4 frames as 4 channels of T-history). Doubles input channels but only ~5% backbone param increase. May help on movement envs where motion direction matters.

### 5. **Hand-engineered per-env heuristics + learned residual** (heavy, exploratory)
Use the env-specific knowledge from public env source code (which we're allowed to read per Phase 0b S4) to encode simple "try every action 1..7 first, then click corners" routines as a prior. Learned model fills in the residual. Pure click envs would be cheap to fully heuristic-solve.

---

## File deltas this phase

**Modified:**
- `src/arc_agi_3_agent/eval/harness.py` — per-env early-exit + progress-based exit.
- `src/arc_agi_3_agent/agent/elite_v0.py` — parameterized inference (temperature, spatial_topk, framechange_filter, stuck_detector_K).
- `src/arc_agi_3_agent/training/models.py` — Backbone 560K → 3.6M params.
- `src/arc_agi_3_agent/training/configs/stage1.yaml` — epochs 50 → 250, spatial_w 0.3 → 0.6, checkpoint_every 5 → 10.

**Created:**
- `scripts/phase_2_5_sweep.py` — config grid + auto-log to `phase-2.5-log.md`.
- `scripts/click_tolerance_check.py` — Part 4.0 probe.
- `phase-2.5-log.md` — running sweep log.
- `phase-2.5-results.md` — this doc.

**Archived:**
- `weights/stage1_v1_archive/` (Phase 2 weights).
- `weights/stage2_v1_archive/` (Phase 2 weights).

---

## Status

**Gate: FAIL.** Kaggle submission blocked. Phase 3 blocked. Per CLAUDE.md §1.2 (gate-strict precedent) and Phase 2.5 Hard Rule 1: no submissions until `harness_score_holdout ≥ 10`.

**Next step: joint architecture decision** with user. Recommend starting with diagnostic #2 (replay-only val on holdout) before any new training, then deciding among #1/#3/#4 based on what #2 reveals.
