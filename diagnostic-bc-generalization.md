# Diagnostic — BC Prediction Generalization (holdout vs train)

**Date:** 2026-05-28 · **Weights:** v1 (560K-param, Phase 2 `stage1_v1_archive/best.pt`, val_acc 0.6504)
**Method:** teacher-forced single-step prediction over human replays (NOT in-env rollout). 179,805 transitions, all 25 public envs, `step>0`. Action-type top-1/top-3 computed over logits **masked to available actions**; spatial (ACTION6 only) as top-5 exact pixel + top-5 within Chebyshev radius 3 (±3 px, per the confirmed approximate-click insight). Methodology adversarially verified — forward-BC pairing, holdout isolation, spatial index math, mask symmetry all confirmed (one symmetric caveat: avail-mask sourced from `record[t+1]`, depresses both splits equally, does not affect the gap).

## Aggregates

| split | n | top-1 | top-3 | n(ACT6) | spatial top-5 | top-5 within r3 |
|---|---:|---:|---:|---:|---:|---:|
| **TRAIN** (20) | 132,413 | **0.789** | 0.959 | 33,667 | **0.256** | **0.530** |
| **HOLDOUT** (5) | 47,392 | **0.511** | 0.758 | 22,515 | **0.006** | **0.119** |
| **gap** | | **+0.278** | +0.202 | | **42× drop** | **4.5× drop** |

## Per-env — HOLDOUT

| env | cluster | n | top-1 | top-3 | n6 | sp5 | sp5_r3 | random (Kaggle S1, 2000-act) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| vc33 | pure_click | 4,516 | 0.994 | 1.000 | 4,490 | 0.008 | 0.173 | score 0.003, **L1** |
| lp85 | mixed | 14,212 | 0.985 | 1.000 | 14,000 | 0.007 | 0.124 | score 0.030, **L2** |
| sk48 | mixed+undo | 14,677 | **0.069** | 0.486 | 565 | 0.004 | 0.124 | score 0.283, **L1** |
| tu93 | pure_movement | 5,307 | 0.251 | 0.746 | 0 | — | — | score 0.000, L0 |
| dc22 | mixed | 8,680 | 0.391 | 0.702 | 3,460 | 0.001 | 0.027 | score 0.000, L0 |

(Trained model scored **0.000 in-env on all 5 holdout** in Phase 2.5.)

## Per-env — TRAIN (top-1 sorted, abbreviated)

| env | top-1 | sp5 | sp5_r3 | | env | top-1 | sp5 | sp5_r3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| tn36 | 0.992 | 0.216 | 0.484 | | sc25 | 0.787 | 0.527 | 0.815 |
| r11l | 0.990 | 0.032 | 0.170 | | wa30 | 0.780 | — | — |
| s5i5 | 0.988 | 0.248 | 0.554 | | ar25 | 0.730 | 0.140 | 0.316 |
| su15 | 0.947 | 0.118 | 0.362 | | re86 | 0.713 | — | — |
| sb26 | 0.930 | 0.197 | 0.470 | | tr87 | 0.707 | — | — |
| bp35 | 0.902 | 0.372 | 0.666 | | cd82 | 0.625 | 0.308 | 0.638 |
| lf52 | 0.880 | 0.425 | 0.750 | | m0r0 | 0.604 | 0.271 | 0.542 |
| ka59 | 0.815 | 0.278 | 0.566 | | sp80 | 0.490 | 0.070 | 0.391 |

## Verdict: **B — prediction does NOT generalize**

Top-1 gap **+0.278** (≫ 10-pt threshold). The representation is train-env-specific. Two distinct mechanisms, both pointing the same way:

1. **Spatial head does not transfer — at all.** Train top-5 0.256 → holdout 0.006 (42× collapse); even with the ±3px tolerance, 0.530 → 0.119. The head memorized *where humans clicked in the 20 train envs*; it has not learned a transferable "click the salient/changed object" function. This is the dominant failure.

2. **Action-type generalizes unevenly, and the high holdout numbers are an illusion.** vc33 (0.994) and lp85 (0.985) look "solved" only because those envs are ~99% ACTION6 — "always click" trivially scores 0.99 without any decision generalization. On the one holdout env that demands real action-type discrimination, **sk48 collapses to 0.069** (worse than the 1/|avail| floor — the model confidently picks the wrong type on an unseen mixed+undo env). Strip the click-only freebies and action-type generalization is poor.

### Why this matches the 0.000 in-env holdout
On vc33 the agent picks ACTION6 (correct type, 0.99) but clicks the wrong pixel (sp5 0.008) → no state change → stuck → 0. Random *explores* the click space and stumbles into level 1 (Kaggle S1: vc33 L1, lp85 L2, sk48 L1); our agent *confidently clicks the same wrong place every step*. **Random out-explores us precisely because our learned spatial prior is wrong-but-confident on unseen envs.**

### Implication (per spec)
Not an execution/compounding problem — the model fails at single-step prediction on unseen envs. The BC prior is a **weak seed**, not a policy. Exploration must carry the load. The spatial head specifically needs a transfer-capable formulation (target relative/saliency-based click rather than absolute memorized pixels), and an explore-and-remember loop is required rather than a deterministic-or-sampled BC replay. **Architecture rethink — joint decision (Hard Rule 4); not iterating autonomously.**
