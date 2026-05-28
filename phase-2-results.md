# Phase 2 Results — E-lite v0

**Date:** 2026-05-27 / 2026-05-28
**Branch:** `main`
**Hardware:** RTX 5070 Ti Laptop (12 GB VRAM), Windows 11, CUDA 13.1 driver, PyTorch 2.11.0+cu128
**Holdout envs:** `vc33`, `tu93`, `sk48`, `lp85`, `dc22` (untouched by any training process)

---

## Gate Decision: **FAIL** ❌

| Metric | Value | Threshold | Status |
|---|---|---|---|
| `harness_score_holdout` | **0.0000** | ≥ 10.0 | FAIL |
| `harness_score_train` | 0.9242 | (informational) | — |

E-lite v0 does not clear the gate for a Kaggle submission. **Phase 3 is blocked** until the architecture is iterated (Phase 2.5) and a re-run clears the holdout gate.

---

## 1. What was built

### 1.1 File inventory (this phase only)

```
src/arc_agi_3_agent/training/
  data.py                       — v3 NPZ Dataset + holdout filter + on-the-fly framechange labels
  models.py                     — ResNet-tiny backbone (~560K params) + 3 heads
  losses.py                     — CE + 0.3*focal(spatial|ACTION6) + 0.1*BCE(framechange)
  synthetic.py                  — OFFLINE-mode biased-random synth rollout generator
  train.py                      — Stage 1 trainer (50ep AdamW cosine, mixed 3:1)
  train_stage2.py               — Stage 2 framechange head fine-tune
  fit_cluster_priors.py         — Stage 3 empirical action priors per env cluster
  configs/stage1.yaml           — Stage 1 hyperparams
  configs/stage2.yaml           — Stage 2 hyperparams
src/arc_agi_3_agent/agent/
  elite_v0.py                   — Inference: backbone + 3 heads + cluster fallback + argmax action / spatial
scripts/
  probe_vram.py                 — 5-step forward+backward VRAM probe
  data_prep/build_split_cache.py — Stratified 90/10 train/val split, SHA-256 cached
  data_prep/composition_report.py — Stage 0 user-review report
tests/training/
  test_holdout_isolation.py     — 100-batch leak check + dataset-length delta check
  test_overfit.py               — single-batch 200-step overfit gate
```

### 1.2 Model architecture

`EliteModel` = 559,594 trainable parameters.

- **Input:** `(B, 3, 64, 64)` int8 → float / 15.0 (OQ7 reduce: channel 0 = first frame, channel 1 = last frame, channel 2 = max-abs-diff over the T-frame animation stack).
- **Backbone:** stem conv3×3(32) → 4 residual blocks `[32, 64, 64, 128]` with strides `[1, 2, 2, 1]` → `(B, 128, 16, 16)`.
- **Action-type head:** GAP → `Linear(128 → 8)`.
- **Spatial head:** `(B, 128, 16, 16) → ConvT(64, stride 2) → ConvT(32, stride 2) → Conv(1, 1×1)` → `(B, 64, 64)` logits.
- **Frame-change head:** `(GAP_feat || action_onehot_8) → Linear(64) → Linear(1)`.

---

## 2. Stage-by-stage results

### Stage A — Infrastructure

| Sanity check | Pass criterion | Observed | Status |
|---|---|---|---|
| holdout leak | 0 leaks / 100 batches | 0 | ✅ |
| dataset delta when holdout dropped | > 1000 rows | 47,731 rows | ✅ |
| single-batch overfit | all 3 head losses < 0.1 after 200 steps | action 0.0013, spatial 0.0003, fc 0.0003 | ✅ |
| VRAM @ batch 128 | < 6 GB | 1,016 MB (1.0 GB) | ✅ |
| param count | < 5 M | 559,594 | ✅ |

### Stage 0 — Data Prep

- **v3 NPZ verified:** `(180144, 3, 64, 64)` int8, schema OK.
- **Synthetic rollouts:** 10,000 steps/env × 20 train envs = **200,000 transitions** in **140 s** total via OFFLINE-mode `BiasedRandomAgent`. Synth framechange rate: **0.811** (post-action frame differs from pre-action in 81% of biased-random transitions).
- **Real BC dataset (holdout-filtered, step > 0):** 132,413 transitions.
- **90/10 train/val split (stratified by env):** 119,181 train / 13,232 val. SHA-256 of indices logged for repro.

### Stage 1 — BC pretrain (50 epochs)

| | Train | Val |
|---|---|---|
| Action acc (best) | 0.7443 | **0.6504** |
| Spatial hit rate (val) | — | 0.0598 |
| Framechange acc (val) | — | 0.9676 |
| Final loss | 0.6224 | 1.2059 |

- Wall: **3,629.8 s** (~60.5 min) — 50 epochs × ~73 s/epoch.
- Sanity gates: epoch 1 val_acc 0.4822 (target ≥ 0.30 ✅), epoch 10 val_acc 0.6233 (target ≥ 0.50 ✅).
- Val accuracy plateaued around epoch 35–50 at ~0.65. **Likely undertrained for harness gate** — user has authorized 100–250 epoch runs for any Phase 2.5 retry.
- Spatial hit rate **0.06** ≫ 1/4096 random baseline (0.00024). 240× over random, but still weak — the click-position head is the bottleneck for harness performance in click-heavy envs (see §5).
- Framechange val acc 0.97 — the head is well-calibrated; reflects high-baseline class imbalance in real BC data.

### Stage 2 — Framechange fine-tune

| Phase | Epochs | Final train acc | Val AUC |
|---|---|---|---|
| baseline (post Stage 1) | — | — | **0.9015** |
| synth_only | 10 | 0.929 | 0.877 |
| combined | 5 | 0.960 | **0.8975** |

- Wall: ~8 min total.
- Target AUC ≥ 0.75 ✅, halt-threshold AUC ≥ 0.65 ✅.
- Slight AUC regression during synth-only fine-tuning (distribution shift between biased-random synth and human-replay real), recovered after combined phase.
- The Stage 2 weights replace the Stage 1 framechange head; backbone + action + spatial heads are frozen.

### Stage 3 — Cluster priors

```
pure_click       (4 envs):  r11l, s5i5, su15, tn36
pure_movement    (5 envs):  g50t, ls20, re86, tr87, wa30
mixed           (11 envs):  ar25, bp35, cd82, cn04, ft09, ka59, lf52, m0r0, sb26, sc25, sp80
```

| KL divergence | Value |
|---|---|
| `KL(pure_click ‖ pure_movement)` | **20.35** |
| `KL(pure_movement ‖ pure_click)` | 19.05 |
| `KL(mixed ‖ pure_click)` | 13.30 |
| `KL(mixed ‖ pure_movement)` | 4.98 |

All KL ≫ 1.0 sanity threshold ✅. Saved to `weights/cluster_priors.json`.

### Stage 4 — E-lite v0 harness eval

```
harness_score_train   = 0.9242   (20 envs, mean RHAE)
harness_score_holdout = 0.0000   (5 envs, mean RHAE)
gate_pass             = False
total_wall_seconds    = 10.3
```

**Per-env scores:**

| env | cluster | score | levels | actions | early-exit | split |
|---|---|---:|---:|---:|---|---|
| re86 | pure_movement | **16.667** | 3 / 8 | 319 | none | train |
| lf52 | mixed | 1.818 | 1 / 10 | 328 | none | train |
| sp80 | mixed | 0.000 | 0 / 6 | 30 | none (GAME_OVER) | train |
| dc22 | mixed | 0.000 | 0 / 6 | 50 | level1@50 | **holdout** |
| lp85 | mixed | 0.000 | 0 / 8 | 50 | level1@50 | **holdout** |
| sk48 | mixed | 0.000 | 0 / 8 | 50 | level1@50 | **holdout** |
| tu93 | pure_movement | 0.000 | 0 / 9 | 50 | level1@50 | **holdout** |
| vc33 | pure_click | 0.000 | 0 / 7 | 50 | level1@50 | **holdout** |
| ar25 / bp35 / cd82 / cn04 / ft09 / g50t / ka59 / ls20 / m0r0 / r11l / s5i5 / sb26 / sc25 / su15 / tn36 / tr87 / wa30 | various | 0.000 | 0 / N | 50 | level1@50 | train |

**Latency** (inference + action build):
- p50 median across envs: **3.07 ms**
- p95 median across envs: **6.31 ms**
- p95 max across envs: **16.35 ms**

✅ Well under the 50 ms p95 inference budget (CLAUDE.md §4.3).

**Aggregate action histogram:**
| action | count |
|---|---:|
| 6 (CLICK) | 1,125 |
| 1 | 279 |
| 4 | 184 |
| 3 | 110 |
| 2 | 59 |
| 7 (UNDO) | 11 |
| 5 | 9 |

Click-heavy (matches human BC training distribution).

---

## 3. Diagnosis — why holdout = 0

### 3.1 Stuck-state argmax problem

23 of 25 envs hit the **50-action early-exit** (Phase 0c §2.5: skip if level 1 not reached in 50 actions). This is **not stochastic failure** — `EliteV0.choose_action` is fully deterministic:

1. Argmax over masked action-type logits.
2. If the chosen action does not change the perception (frame-change head doesn't see a change), the next observation is identical.
3. Identical input → identical logits → identical action.

→ The agent gets stuck in a 1-action loop and burns through the 50-action budget without progressing past level 1.

The 2 envs that scored (`re86` 16.67 and `lf52` 1.82) succeeded because the chosen argmax action happened to advance the state (re86 is `pure_movement` — directional movement always changes state).

### 3.2 Holdout pattern

The 5 holdout envs span the cluster space:
- `vc33` (pure_click): argmax click position is fixed → stuck.
- `tu93` (pure_movement): movement direction is wrong for the puzzle → stuck.
- `sk48`, `lp85`, `dc22` (mixed): argmax action + argmax spatial → fixed (action, x, y) → stuck.

A 6th observation: `sp80` ended with GAME_OVER at 30 actions — the agent took losing actions in a mixed-action env. Same root cause: deterministic walk into a known losing trajectory.

### 3.3 Spatial head weakness

Val spatial hit rate (exact-pixel argmax match) was **0.060** at the end of Stage 1. While this is 240× over the 1/4096 random baseline, it is far below what's needed for click-heavy envs to reach level 1 with deterministic clicking. Even on real human data, the model only guesses the right click pixel 6% of the time.

---

## 4. Recommended next steps

The gate is non-exempt (CLAUDE.md §1.2 Phase 1 precedent). To reach the harness gate ≥ 10 on holdout, iterate before any Phase 3 submission.

### 4.1 Phase 2.5 — high-priority fixes

1. **Stochastic policy at inference.** Replace argmax with temperature-controlled sampling from the action-type softmax. Same for the spatial head. Eliminates the stuck-state loop and should immediately unlock progress on the 18 zero-scoring train envs and most of the holdout. Tunable parameter (no retraining needed).

2. **Longer Stage 1 training.** User has authorized **100–250 epoch runs** (saved as project memory). The 50-epoch run plateaued around val_action_acc 0.65; another 50–200 epochs may close 5–10 percentage points and substantially improve spatial hit rate.

3. **Spatial head reweighting.** Spec gives spatial 0.3× weight in the combined loss. Spatial accuracy is the weakest head — try 0.6 or 1.0 weight in a Phase 2.5 retrain.

### 4.2 Phase 2.5 — secondary

4. **Action-type entropy fallback.** The cluster-prior fallback fires only when `max(softmax) < 0.4`. In stuck-state runs the argmax is often very confident on a wrong action — bump threshold to 0.7 or use entropy directly.

5. **Stuck-state detection.** If the last K observations were identical, force a uniform random action from `available_actions \ {last_action}`. Cheap and robust.

6. **L1 lookahead via framechange head.** The frame-change head is well-trained (AUC 0.90). At inference, for each candidate action, query framechange probability for `(perception, action)`; weight or filter by predicted change. Could replace #4 + #5 in a unified way.

### 4.3 Architecture iteration (Phase 2.6+ if needed)

If 4.1+4.2 still don't clear the gate:
- Larger backbone (target ~3 M params instead of 560 K; still fits 6 GB VRAM ceiling at batch 128).
- Per-cluster heads (separate action-type head for click vs movement envs).
- Sequence-aware perception (T-stack temporal conv instead of 3-channel reduction).

---

## 5. Phase 2 checklist

- [x] Stage A: skeletons + tests + VRAM probe
- [x] Stage 0: synth rollouts (200K), split cache (132K real), composition report
- [x] Stage 1: 50-epoch BC pretrain, val_acc 0.6504
- [x] Stage 2: framechange head fine-tune, val_auc 0.8975
- [x] Stage 3: cluster priors, KL 20.4 (≫ 1.0)
- [x] Stage 4: E-lite v0 assembled, harness eval complete
- [x] Latency under 50 ms p95 budget (3 ms p50, 6 ms p95)
- [x] Holdout never trained on (test verified, no leaks)
- [x] All training runs logged → `runs/stage{0,1,2}/<ts>/`
- [x] All weights cached → `weights/stage{1,2}/` + `weights/cluster_priors.json`
- [x] `phase-2-results.md` (this doc)
- [ ] **Gate met (`harness_score_holdout ≥ 10`)** — FAIL, 0.0000

---

## 6. Hand-off

**Status:** Phase 2 closes with **gate FAIL**. No Kaggle submission is authorized (CLAUDE.md §1.2). Phase 3 is blocked.

**Recommendation:** Proceed to **Phase 2.5** with the stochastic-policy fix (§4.1.1) as the cheapest first iteration. It requires no retraining — only an inference-time code change in `elite_v0.py`. Re-run harness; if `harness_score_holdout ≥ 10`, advance to Phase 3 planning. Otherwise stack §4.1.2 (longer training, weights reload) on top.

The gate-strict precedent from Phase 1 holds: no exemption submissions. Iterate locally until the bar is cleared.
