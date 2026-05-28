# ARC-AGI-3 — Phase 0c Architecture Specification

**Author:** Phase 0c — architecture & pre-implementation design.
**Date:** 2026-05-27 (T-34 days from Milestone #1 on 2026-06-30).
**Scope:** Resolve outstanding open questions, pick the architecture, design the local eval harness, sequence the first 5 submissions, spec the training pipeline. **No code, no training, no repo, no submissions.**
**Inputs:** `research-findings.md`, `validation-results.md`, all artifacts in `scripts/validation/` and `data/`.

---

## 0. Open-Question Resolutions

Every resolution cites the evidence that closed it. The 7 Open Questions carried from earlier phases are answered here so Section 1 (architecture decision) can rest on a stable foundation.

### OQ5 — RHAE units / score interpretation

**Resolution:** Kaggle's leaderboard displays `scorecard.score` **directly, on a 0–100 (cap 115) scale.** Tufa Labs' 1.17 means a raw scorecard score of 1.17, i.e. ~1.17% of the theoretical maximum (100% = perfect human-match on every level of every env).

**Evidence** — `.venv313/Lib/site-packages/arc_agi/scorecard.py`:

```python
# Line 170 — per-level score:
score = ((baseline_actions / actions_taken) ** 2) * 100
score = min(score, 115.0)  # Cap at 115

# Lines 196–206 — env score = weighted (1-indexed) avg of level scores,
#   capped at (completed_weight / total_weight) * 100:
for i in range(len(self.level_scores)):
    weight = self.level_indices[i]
    total_score += self.level_scores[i] * weight
    total_weights += weight
    if self.level_scores[i] > 0:
        max_weights += weight
score = total_score / total_weights
max_score = max_weights / total_weights * 100
score = min(score, max_score)

# Line 614 — overall: mean of env scores
avg_score = sum(env_score.score for env_score in environment_list) / len(...)
```

**Consequence for the "≥ 10" gate:** the harness reports a score in the same 0–100 units. **Local gate = `harness_score_holdout ≥ 10.0`** (≈10% of theoretical max — well above the current Kaggle public-LB median of ~0.10–0.15, comfortably above the floor of 0.0, and below the current top-3 at ~0.66–0.68).

### OQ6 — ACTION7 action accounting

**Resolution:** Every `env.step()` call costs 1 action, **including ACTION7**. The initial `env.reset()` issued by user code also costs 1 (the "first RESET free" carve-out in `arc_agi/api.py:330` applies only to the implicit reset inside `arc.make()`, not user-issued resets).

**Evidence — Kaggle parity output (validation-results.md §4.3):**

```
After 1 reset() + 10 step() calls:
  scorecard.actions = 11
  scorecard.resets = 1
```

ACTION7 was not specifically tested (sp80 doesn't expose it), but the rule "every step counts" is symmetric across action IDs in `arc_agi/api.py` — no carve-out exists for ACTION7. **Operating assumption: ACTION7 costs 1 action like every other action.** This shapes the use-undo policy: undo only when the expected next-action improvement is worth the budget loss.

### OQ7 — Animation handling in the perception module

**Resolution:** Reduce variable-length `(T, 64, 64)` frame stacks to **fixed 3-channel `(3, 64, 64)` input**:

```
channel 0 = frame[0]            # pre-animation state (action was issued from here)
channel 1 = frame[-1]           # post-animation state (next action chosen from here)
channel 2 = max-abs-diff        # element-wise max(|frame[t+1] - frame[t]|) across t∈[0..T-1]
                                  → captures the union of moved cells, zero everywhere stable
```

**Justification:**
- T varies 1–404 (Phase 0b §6.5). A flat 3-channel reduction batches cleanly and avoids variable-length sequence modeling for the perception backbone.
- Channel 1 alone (last-frame-only) loses motion info, which is needed for the user-flagged "moving objects" late-game mechanic.
- Full temporal stack (T frames) is expensive and not Phase 0c-justified — the agent acts on the post-animation state; intermediate frames carry mostly redundant signal.
- The 3-channel reduction is reversible enough: channel 2 lets the model detect "something moved here," which is the auxiliary signal worth keeping.

For T=1 (70.9% of frames per Phase 0a §4.5): channel 0 = channel 1, channel 2 = zeros. Model learns to treat zero-motion-mask as "no animation occurred."

### OQ4 — Visibility / fog-of-war per env

**Resolution:** **Zero of 25 public envs have fog-of-war.** All five visibility-audit candidates (dc22, vc33, lf52, g50t, m0r0) were false positives:

| Env | Audit hit | Reality |
|---|---|---|
| dc22 | radius×9 | obfuscated variable `svfovexaz` containing "fov" substring |
| vc33 | radius×4 | sprite name `0012efovcqtrsv` containing "fov" substring |
| lf52 | dark×14 | color constants `BLACK`, `DARK_GRAY` used throughout |
| g50t | radius×2 | obfuscated var `ugfrhsffov` |
| m0r0 | frame-zero×2 | `frame[0,x]=0` and `frame[63,x]=0` draw top/bottom HUD border bars (not fog) |

Source: `scripts/validation/.visibility_audit.json` + targeted greps.

**Consequence:** the user's hand-play observation of "limited visibility in some late levels" must be either (a) a private env mechanic we haven't seen, (b) misclassification of an animation effect or HUD border, or (c) a sprite-occlusion rather than a true fog-of-war. **Design decision: assume full-frame observability for the 25 public envs at training time.** Phase 0c does NOT add a fog-of-war handling component in the perception module; CNN architectures naturally tolerate occluded regions if encountered at eval time.

### OQ1 / OQ2 / OQ3 — Architecture-coupled

Resolved in §1 below.

---

## 1. Architecture Decision

### 1.1 Approach restatement (Phase 0a §6, condensed)

| ID | One-liner | Phase 0a/0b evidence |
|---|---|---|
| A | Hand-crafted primitives + small MLP/GRU policy | Engine sustains >1k FPS; hand-crafted vision is brittle on novel sprites (S4 supports synthetic data generation, but novel-env structure still defeats fixed rules) |
| B | BC-pretrained CNN + online RL fine-tune | 180k forward-BC transitions ready (`bc_transitions_v2.npz`); but COMPETITION-mode = 1 `make()` per env (research-findings §1.11) → no cross-rollout RL signal |
| C | State-graph memory + learned controller | Phase 0a §5.2 — preview Blind Squirrel (6.71% → launch collapse). State-graph blows up under animations and stochasticity |
| D | Slot-attention world model + planner | Strongest academic line; 1–2 weeks just to get slot attention stable on grid input; **deprioritized for M#1** (assigned to Milestone #2) |
| E | Hybrid: BC backbone + frame-change auxiliary + replay-conditioning + 1-step lookahead | Combines every signal; replay-conditioning collapses to cluster marginals on private envs (Phase 0a §6.5.1) — needs reshaping |

### 1.2 Scoring matrix (weights bind the decision)

Axes weighted by Phase 0b empirical findings; weights sum to 100.

| Axis | Weight | A | B | C | D | E-lite (this spec) | E-full |
|---|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 5-week feasibility | 20 | 5 | 4 | 4 | 1 | 4 | 3 |
| 12GB local-dev VRAM | 15 | 5 | 5 | 5 | 2 | 4 | 3 |
| 165ms inference budget | 15 | 5 | 5 | 4 | 1 | 4 | 2 |
| Expected ceiling (mean under ±0.2 variance) | 15 | 1 | 3 | 2 | 5 | 4 | 4 |
| 1-submission/day iteration | 10 | 5 | 3 | 4 | 1 | 4 | 3 |
| Public→private generalization | 15 | 1 | 3 | 1 | 4 | 4 | 3 |
| Engineering risk / week | 10 | 5 | 4 | 4 | 1 | 4 | 2 |
| **Weighted total (×10)** | 100 | **35** | **38** | **32** | **23** | **41** | **30** |

(Scores 1–5: 5 = strong fit, 1 = blocker.)

### 1.3 The pick — **Approach E-lite**

**E-lite components:**
1. **Perception backbone:** ResNet-style CNN (~3–4 M params), input `(3, 64, 64)` one-hot-color × 16 channels → conv stack → spatial feature map + global pooled vector.
2. **Action-type head:** categorical over 8 actions (RESET + ACTION1..7), masked to env's `available_actions` set (constant per env — research-findings §2.6 + Phase 0b Item 5).
3. **Spatial head for ACTION6:** dense 64×64 logits map, only consulted when the chosen action type is ACTION6.
4. **Frame-change auxiliary head (L1):** binary per-(state, action) prediction "did this action change the frame?" → trained as auxiliary loss + used at inference as a candidate filter (skip actions with high predicted prob of being no-ops, except as exploration).
5. **Action-signature gating:** at env first frame, read `available_actions`; pick which head subset to consult (pure_click / pure_movement / mixed routing per Phase 0a §4.3 clusters).
6. **Cluster-level action priors:** trained from 25-env replays grouped by action signature; used at inference as a soft prior on **private envs** where no in-env replay is available.
7. **Online exploration with frame-change bias:** when BC prior entropy exceeds a threshold (signal that the policy is uncertain), sample from the frame-change-weighted action distribution instead of greedy. Resolves OQ3 novelty handling.

**Explicitly excluded** (from E-full):
- ❌ **Inference-time replay-retrieval head** — Phase 0a §6.5.1: collapses to cluster-level marginals on private envs, which we already get from the cluster prior. Engineering cost not earned. **Replays retained at training time as distillation target.**
- ❌ **L2 learned forward model** — 1–2 weeks engineering, training instability on small data, public-set overfitting risk. The 1-step lookahead is implemented via the L1 frame-change head only (cheaper, faster, well-precedented by StochasticGoose).
- ❌ **State-graph memory** — see Approach C; doesn't survive animations and partial observability.
- ❌ **Slot-attention object-centric module** — Milestone #2 candidate; deferred.

### 1.4 Why E-lite beats the alternatives

| Rejected | Disqualifier (one sentence) |
|---|---|
| A | Hand-crafted vision fails on novel sprite vocabularies (260 distinct sprite tags in public set alone; private set likely adds more). |
| B | One-shot COMPETITION mode forbids cross-rollout RL gradient signal; intra-rollout signal alone is insufficient. |
| C | State-graph hash-equality breaks under animation noise and any latent state, both confirmed present in the public envs. |
| D | Slot-attention training on this dataset has not been demonstrated in 5 weeks of focused work in any published source — engineering risk is too high for M#1. |
| E-full | Inference-time replay retrieval on private envs is a no-op (cluster-marginal collapse); engineering cost (retrieval index, k-NN, latency budget) not earned. |

### 1.5 Cross-examination

**Q1 — How does this generalize from 25 public envs to 55 fully-private?**
- Action-type categorical: trained on full action distribution; private envs use the same 7-action vocabulary, masked to per-env `available_actions`.
- Spatial ACTION6 head: trained on broad sprite vocabulary (260 tags); generalizes to novel sprites if it learns object-shape priors instead of object-identity.
- Frame-change head: object-and-action-agnostic in principle (binary did-it-move signal). Generalizes naturally.
- Cluster prior fallback: weak but non-zero signal on private envs (cluster identifies pure_click / pure_movement / mixed at first frame from `available_actions`).
- The exploration bonus (frame-change-guided when BC entropy is high) is the primary novelty-coping mechanism: when the model is uncertain, it prefers actions that change the frame, biasing exploration toward informative moves.

**Q2 — Failure mode? When does this score 0.0 instead of 0.5?**
- If the held-out env's action mechanics are categorically novel (e.g., requires a sequential 2-press combo not present in any public env), the BC policy + frame-change head won't have learned the abstraction. Score = whatever biased random alone achieves.
- If the private env has very tight step budgets (5× cap is < BC's typical action count), the agent gets terminated before reaching late levels.

**Q3 — Most likely 2-week engineering stall component?**
- The **spatial ACTION6 head** is the highest-risk component. Click coords are sparse (single (x,y) target per click in human replays) and the 64×64 output space is large. Risk: head produces uniform/noisy logits and the agent click-spams the center. Mitigation: pretrain ACTION6 head with masked-spatial-reconstruction auxiliary (predict the click location given the previous frames in the replay), use focal loss to handle sparse positives.
- Backup if ACTION6 head fails: fall back to **uniform-over-non-background-cells** click prior — still better than full uniform. Phase 1 must implement this fallback before the spatial head as a safety net.

---

## 2. Local Evaluation Harness Design

### 2.1 Eval split — 20 train / 5 holdout

The 25 public envs split into a training set (20 envs) and a holdout set (5 envs) used **only** for the gate metric `harness_score_holdout`.

**Holdout selection criteria** (one env per criterion, no overlap):

| Slot | Criterion | Pick (with rationale) |
|---|---|---|
| H1 | pure_click signature | **`vc33`** — single-action vocab, simplest test of pure-click head transfer |
| H2 | pure_movement signature | **`tu93`** — 9 levels, mid-depth, no rotation in heavy mid-env vs late |
| H3 | mixed signature + undo support | **`sk48`** — 8 levels, exposes ACTION7, exercises the full action head |
| H4 | mid-env mechanic complexity (highest sprite-tag count) | **`lp85`** — 86 sprite tags, longest source file, stress-tests perception |
| H5 | high lose-call density (challenging dynamics) | **`dc22`** — 5 `self.lose()` calls in source, complex failure conditions |

The remaining **20 envs are the train set** for any replay-trained component: `ar25, bp35, cd82, cn04, ft09, g50t, ka59, lf52, ls20, m0r0, r11l, re86, s5i5, sb26, sc25, sp80, su15, tn36, tr87, wa30`.

The holdout split is fixed in the harness config; never trained on, never evaluated mid-training. The gate "≥ 10" applies only to `harness_score_holdout`.

### 2.2 Score units — match Kaggle 0–115 scale

Local RHAE implementation reuses the exact formula from `arc_agi/scorecard.py:170` and the env aggregation from lines 196–206. The harness imports the SDK's `EnvironmentScoreCalculator` if it's accessible, or re-implements identically (preference: re-implement and unit-test against a synthetic-trajectory fixture, so local and Kaggle stay in lockstep regardless of SDK updates).

The gate **`harness_score_holdout ≥ 10.0`** is in 0–115 units (~10% of theoretical max).

### 2.3 Scoring implementation

```
For each env in {train_set, holdout_set}:
    arc.make(env_id) → env
    obs = env.reset()
    while not done and step_count < 5 * env.baseline_actions_total + safety_margin:
        action = agent.choose_action(obs, env_info)
        obs = env.step(action)
        step_count += 1
    # Per-env RHAE = scorecard formula on (level_actions_taken, level_baseline_actions)
    score_per_env = compute_rhae(level_actions, baseline_actions, levels_completed)

harness_score_train   = mean(score_per_env for env in train_set)
harness_score_holdout = mean(score_per_env for env in holdout_set)
```

`compute_rhae` is the literal port of `scorecard.py:170` + env weighted-avg + completion cap. Unit-tested against synthetic trajectories.

### 2.4 Run protocol

- **Deterministic seed.** Per-env seed = `hash(env_id) & 0xFFFFFFFF`. Same seed → same trajectory.
- **One pass per env.** Single rollout per env per harness run, matching COMPETITION-mode 1-make-per-env semantics.
- **Action cap** = `5 × sum(baseline_actions_per_level) + 50` safety margin. Mirror's Kaggle's per-level termination on a per-env aggregate.
- **OFFLINE mode.** `Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=...)`. Scorecard counters in OFFLINE are decorative under THIS constructor path / `LocalEnvironmentWrapper` (Phase 0b S5; counters DO fire under env-var-driven `Arcade()` no-kwargs pattern — see validation-results.md Phase 1 Findings), so we track actions/levels ourselves at the agent wrapper.

### 2.5 Speed budget — <30 minutes end-to-end

Empirical constraints:
- Engine FPS mean = 2,620 (Phase 0b §5.2).
- Total worst-case actions for 25 envs at 5× cap × ~50 baseline × 7 levels = ~87.5K actions.
- At 30 ms model inference + 1 ms engine = ~30 ms/action → 87.5K × 0.03 = 44 minutes worst case.

**Mitigation:** harness early-exits envs where the agent fails level 1 in <50 actions (no path to learning more from this run); reduces typical wall-clock to **<20 minutes**. Phase 1 verifies.

### 2.6 What gets logged (per harness run)

Written to `harness_runs/<timestamp>/`:
- `summary.json` — overall scores, per-env breakdown
- `per_env/<env_id>.jsonl` — one line per action: `{step, action_id, action_x, action_y, state, levels_completed, frame_hash, latency_ms}`
- `action_histograms.json` — per-env action distribution comparison vs. human replay action distribution
- `latency_p50_p95_p99.json` — model inference latency percentiles (must stay <50ms p95 to meet 165ms ceiling with 3× headroom)
- `surprise_events.jsonl` — frame hashes the agent encountered that don't appear in any training trajectory (high-novelty regions)
- `gate_status.txt` — single line: `PASS` if `harness_score_holdout ≥ 10.0`, else `FAIL: <score>`

---

## 3. Variance Characterization & First-5-Submissions Plan

### 3.1 First 5 submissions — explicit sequence

Every submission must (a) advance architecture knowledge or (b) extract variance signal — ideally both. Submission rate is 1/day; reserving 5 of 34 here.

| # | Date (earliest) | Agent | Purpose | Submit-condition |
|---|---|---|---|---|
| S1 | T-32 (2026-05-29) | **Plumbing-validation random agent.** Biased-random over each env's `available_actions`; ACTION7 only when env exposes it; ACTION6 click at uniform-over-non-background cells. **No model.** Confirms S9 anon-key fix, S10 paths, COMPETITION-mode wiring. | Pipeline validation + score floor | None — first submission; no gate (the gate exists to qualify *learned* agents). |
| S2 | T-31 (2026-05-30) | **Marginal-action-distribution agent.** Sample from BC marginal P(a) per env's action signature cluster. **State-blind** (does not look at the frame). | Confirms BC dataset → agent wiring; establishes "any signal" baseline above random | Local harness `score_holdout ≥ 0` (any non-crash) |
| S3 | T-30 (2026-05-31) | **Re-submit S1 verbatim.** Identical code, identical seeds. | **First variance data point.** Compare to S1's score. If \|S3 − S1\| > 0.1 → variance is meaningful → adjust expectations | None — re-submission |
| S4 | T-29 (2026-06-01) | **First learned agent: E-lite v0.** BC backbone + action-type head + cluster-prior fallback. NO frame-change head, NO spatial head (clicks fall back to uniform-non-background prior). | First real progress milestone; tests core architecture | **Gate active: `harness_score_holdout ≥ 10.0`.** Skip submission and iterate locally if gate fails. |
| S5 | T-28 (2026-06-02) | **Branch:** if S4 ≥ 5: re-submit S4 for variance confirmation. If S4 < 5 or skipped: re-submit S2 (best non-S4 candidate). | Variance signal on either S4 or S2 baseline | Same gate rule as S5 candidate's first submission |

### 3.2 Seed determinism — mandatory

- Every agent must be fully deterministic for a given seed. Same seed + same env + same initial obs → identical action sequence every time.
- Use `random.Random(seed)` and `np.random.default_rng(seed)` everywhere; never `random.random()` or `np.random.rand()` (process-global).
- Per-env seed = `hash((env_id, run_id, agent_version)) & 0xFFFFFFFF`. Variance from re-submission then comes from **scoring-side env-subset selection only** — agent-side variance is netted out.

### 3.3 Submission decision rule (gating)

Before any submission after S3:

1. Run local harness on holdout 5 envs.
2. Verify `harness_score_holdout ≥ 10.0`. If not: skip submission, iterate locally, do not burn the day.
3. Verify the submission is **meaningfully different** from any prior submission (architecture change OR explicit re-submission for variance).
4. State in the chat: remaining quota, `harness_score_holdout` value, one-sentence diff from prior submission.
5. Wait for explicit user authorization in the immediately prior message.
6. Then call `mcp__kaggle__submit_to_competition` (or equivalent).

### 3.4 Variance interpretation thresholds (provisional; revisit after S5)

| Δ between submissions | Interpretation |
|---|---|
| < 0.1 | Noise. Probably the same agent on different env subsets. |
| 0.1 – 0.3 | Ambiguous. Need ≥1 more re-submission to confirm. |
| 0.3 – 0.6 | Likely real architecture signal, but check for variance interaction. |
| > 0.6 | Real signal. The two architectures are meaningfully different in mean. |

These thresholds are derived from the Topic 699208 community report of "0.38 → 0.19" on identical submissions (≈0.2 spread). After S3 vs S1, we'll have an in-house variance estimate and can recalibrate.

---

## 4. Training Pipeline Spec

The pipeline produces the weights for E-lite. Total budget ≤24 hours on RTX 5070 Ti (12 GB VRAM). All stages use `data/bc_transitions_v2.npz` as primary input; supplemented by synthetic rollouts from public env source (S4).

### 4.1 Stage 0 — Data preparation

- Input: `data/bc_transitions_v2.npz` (180,144 forward-BC transitions across 339 replays; 20 train envs after holdout removal — drops to ~155K transitions estimated).
- Generate **synthetic rollouts** from public env source for the 20 train envs only. Strategy: run a biased-random agent in OFFLINE mode for 10K steps per env → label each transition with `frame_changed: bool` and pair (state, action, next_state). Mixed at **1:3 synthetic:real ratio** in the training mini-batches (avoid overfitting to the deterministic-engine no-op patterns of synthetic data).
- Compute and cache the `(3, 64, 64)` perception input from raw state + next_state pairs (channel 0 = state, channel 1 = next_state[last] if animation else state, channel 2 = diff mask).
- Output: `data/bc_input_train.npz`, `data/bc_input_synth.npz`, `data/bc_input_holdout.npz` (holdout cached for harness; never touched at training time).
- Wall-clock: ~2 hours including synthetic rollouts.

### 4.2 Stage 1 — Perception backbone + action-type head pretrain

- Architecture: ResNet-18-equivalent CNN over `(3, 64, 64)` one-hot (3 channels × 16 colors = 48 input channels after one-hot expansion), output spatial feature `(B, 128, 8, 8)` + global pool `(B, 128)`. ~3 M params.
- Heads: action-type categorical (Linear 128 → 8) + spatial map (Conv 128 → 1, upsampled to 64×64).
- Loss: `cross_entropy(action_type_logits, action_type_labels) + 0.3 * focal_loss(spatial_logits, action6_target_mask) + 0.1 * BCE(framechange_logits, framechange_labels)` (framechange as auxiliary).
- Optimizer: AdamW, lr 3e-4, cosine schedule, 50 epochs, batch 128 (≈3 GB VRAM).
- Validation metric: held-out (within-train-set) 10% split → action-type accuracy + ACTION6 spatial IoU.
- Wall-clock: ~6 hours.

### 4.3 Stage 2 — Frame-change head fine-tune

- The frame-change head is already trained as auxiliary in Stage 1. Stage 2 fine-tunes it explicitly on the synthetic dataset (which has 100% of transitions labeled with frame_changed; replays only label this implicitly via state-diff).
- Architecture: small MLP `(128 + action_one_hot_8) → 64 → 1` sigmoid. ~10K params.
- Loss: BCE over frame_changed.
- Optimizer: AdamW, lr 1e-4, 10 epochs on synthetic-only data + 5 epochs on combined.
- Validation: ROC-AUC of frame-change prediction on holdout transitions.
- Wall-clock: ~1 hour.

### 4.4 Stage 3 — Cluster-prior fitting

- For each of {pure_click, pure_movement, mixed}, fit a categorical distribution over actions from the training-set replays in that cluster.
- Output: 3 fixed action-probability vectors used as soft fallback prior at inference.
- Wall-clock: <5 minutes.

### 4.5 Total training budget

| Stage | Wall (h) | VRAM peak |
|---|---:|---:|
| 0 — Data prep | 2 | 0 GB GPU |
| 1 — Backbone + heads | 6 | ~5 GB |
| 2 — Frame-change FT | 1 | ~2 GB |
| 3 — Cluster priors | <0.1 | 0 GB |
| **Total** | **~9 h** | — |

Comfortably under the 24-hour ceiling. Allows ≤2 full retrains per architecture variant within a day, supporting the 5-8-architectures-end-to-end budget.

### 4.6 Checkpointing strategy

- After each stage: save `weights/<stage_name>_<epoch>.pt`. Keep last 3 per stage.
- After Stage 1: `weights/best_backbone_val<metric>.pt` symlinked.
- Final shipped weights: bundled into a Kaggle dataset of size ~20 MB (backbone params dominate). Upload as `arc-agi-3-agent-weights` Kaggle dataset; attached to submission notebook.

### 4.7 Reproducibility

- Pin: `torch==2.12.0+cpu` for CPU dev, `torch==2.12.0+cu121` for GPU; `numpy==2.4.4` (matches Kaggle); `arc-agi==0.9.8`, `arcengine==0.9.3`.
- Set `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds 0 for {python, numpy, torch, torch.cuda}.
- Training run config saved as `runs/<timestamp>/config.json`.

### 4.8 Synthetic-data caveat

Synthetic rollouts from public env source (allowed per S4, Topic 699900) carry a real risk: they learn engine-specific deterministic patterns that don't generalize. **Cap synthetic at 25% of training mini-batches.** Validate by running the trained agent against the **5 holdout envs only** — if synthetic-heavy runs improve train-set score but not holdout score, reduce the ratio.

---

## 5. ARC-AGI-3-Agents Bundle Inspection

Read of `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents/` (mirrored locally at `arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents/`).

### 5.1 Entry-point pattern

`main.py` is the reference CLI: `python main.py --agent=random --game=sp80`. **It is NOT what Kaggle's grader calls.** Kaggle executes the user's notebook end-to-end; whatever code runs in the notebook is the "agent." The grader watches the produced `scorecard.json` artifact.

**Operational consequence:** our Phase 1 submission notebook does NOT import `main.py`. It directly does:

```python
import os
os.environ["ARC_API_KEY"] = "noop"   # S9 workaround — mandatory

from arc_agi import Arcade, OperationMode

ROOT = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
arc = Arcade(
    operation_mode=OperationMode.COMPETITION,
    environments_dir=f'{ROOT}/environment_files',
)
card_id = arc.open_scorecard(tags=['v1'])
for game_info in arc.get_environments():
    env = arc.make(game_info.game_id, scorecard_id=card_id)
    obs = env.reset()
    agent.reset_for_env(game_info)
    while not is_done(obs):
        action = agent.choose_action(obs)
        obs = env.step(action)
scorecard = arc.close_scorecard(card_id)
```

### 5.2 Agent base class contract (`agents/agent.py`)

`Agent` ABC requires:
- `is_done(frames, latest_frame) -> bool` — stop condition for the per-env loop
- `choose_action(frames, latest_frame) -> GameAction` — main decision

Useful base behaviors (Phase 0a §2.2):
- `Agent.main()` loops choose→step→append until `is_done` or `MAX_ACTIONS` exceeded.
- Default `MAX_ACTIONS = 80` — **far too low for our agent**; override to a number derived from the env's baseline_actions × 5.
- `Recorder` integration is optional and not needed for COMPETITION submissions.

### 5.3 Reference Random agent (`agents/templates/random_agent.py`)

Notable patterns:
- On `GameState.NOT_PLAYED` or `GameState.GAME_OVER`: returns `GameAction.RESET`. **This costs 1 action per reset** (OQ6 / Phase 0b S3).
- For complex actions (ACTION6): sets `data={'x': randint(0,63), 'y': randint(0,63)}`.
- Reasoning field is set but optional.

### 5.4 Lifecycle in COMPETITION mode

1. **Per submission:** open one scorecard.
2. **Per env:** call `arc.make(game_id, scorecard_id=card)` exactly once.
3. **Per env-loop:** call `env.reset()` once at the start of the loop (counts 1 action), then `env.step()` until WIN/GAME_OVER or budget exhaustion.
4. **At env-loop end:** move to next env's `arc.make()`. **Cannot re-enter a completed env.**
5. **End of submission:** call `arc.close_scorecard(card_id)` to finalize scoring.

### 5.5 Gotchas the bundle works around (and ones it doesn't)

| Gotcha | Bundle workaround | Our action |
|---|---|---|
| API key required | Reads from `ARC_API_KEY` env var | We must set `"noop"` before any `Arcade(...)` (S9) |
| MAX_ACTIONS default 80 | Configurable per agent subclass | Set to `5 * sum(baseline_actions)` + safety margin |
| OFFLINE scorecard counters | LocalEnvironmentWrapper path only (env-var Arcade() works — Phase 1) | We must track action/level counts ourselves at agent level (S5) |
| First RESET in `arc.make()` is free | Built-in to `api.py:330` | We don't issue an explicit reset before the first step — `env.reset()` returns the initial obs already (the make-time reset is implicit) |

### 5.6 Phase 1 must replicate

- Pre-set `ARC_API_KEY="noop"`.
- Use the correct Kaggle paths (`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/...`).
- Subclass `Agent` with overridden `MAX_ACTIONS`.
- Track actions/levels independently of the OFFLINE scorecard.
- Test the full notebook locally first (against the same env structure mounted at the Kaggle path) before submitting.

---

## 6. Open Issues / Risk Register for Phase 1

These are not blockers for Phase 0c but Phase 1 must surface them:

1. **ACTION6 spatial head training data is sparse.** 56K clicks across 20 train envs → ~2,800 click locations per env on average, in a 64×64 grid. Focal loss + spatial smoothing mandatory.
2. **Variance characterization burns 1 submission per architecture confirm.** Phase 0c plan reserves 5 of 34 for the first cycle; future cycles may need 2-3 confirms per architecture, draining the budget. If we hit ≥0.6 ground-truth score by S6-S8, reduce to 1 confirm per arch and trust local harness for the rest.
3. **Synthetic-data overfitting.** Monitor train vs holdout score gap; if gap exceeds 20 points sustainedly, cut synthetic ratio.
4. **First submission probably scores ≤0.1.** Don't panic. S1's score is calibration for variance, not progress.
5. **Tufa Labs at 1.17 might not be reproducible variance-on.** Their reported score could collapse to 0.7-0.9 on a fresh submission. Watch their LB entries; if their score drops, our top-3 threshold relaxes further.

---

## 7. Phase 0c Done — Phase 1 Handoff

This document + draft `CLAUDE.md` + `phase-1-bootstrap.md` are the Phase 1 inputs. Phase 1 implements:

1. Repo bootstrap per `phase-1-bootstrap.md`.
2. Local eval harness per Section 2 of this doc.
3. Training pipeline Stages 0–3 per Section 4.
4. S1 submission (the plumbing-validation random agent) — after explicit user authorization.

**Status: Phase 0c complete. All 7 Open Questions resolved with evidence. Architecture chosen (E-lite) with documented rationale. Awaiting user review before Phase 1.**
