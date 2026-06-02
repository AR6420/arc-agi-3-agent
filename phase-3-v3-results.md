# Phase 3 v3 Results — Unified Goal Inference (timeboxed, final build)

**Date:** 2026-06-01 · **Branch:** `main` · Local, no submission.
Holdout = vc33, tu93, sk48, lp85, dc22 (never tuned on). 5×Σbaseline budget. Stable seeds.

**Goal of v3:** collapse the proven B5↔GoalProbe trade-off into ONE agent that holds both
signals (relational candidate-ranking + broad sequence-exploration) and lets in-episode reward
arbitrate, so it keeps r11l (B5's strength) AND keeps/raises sk48+lp85 (GoalProbe's gain).

## Outcome: the trade-off is IRREDUCIBLE → timebox fallback triggered

The unified arbitration was built and measured. **No setting recovers both operating points.**
Any non-zero relational click-bias destroys r11l; a zero bias is just B5. The two goal-styles
cannot be arbitrated from reward evidence within budget, because the arbitration signal
(observed reward) only arrives *after* the early click budget is already spent — and for r11l
any relational bias on that early budget starves discovery *before* reward can arbitrate. Per
the timebox discipline (one iteration, explicit fallback), we STOP and document.

---

## Step 1 — Per-env episode-memory boundary (asserted in code)

`DiscoveryAgent.reset_for_env` now RE-INSTANTIATES the world model, strategies, and decider per
env (was: reset a shared object). This is the strongest no-leak guarantee — a fresh object
graph per env, so no learned state (effects, controllable, goal, reward classes, lethal set,
resource) can cross env boundaries; only the env-agnostic *procedure* does. Carrying episode
memory across envs would be the BC overfitting failure.

- Test `tests/discovery/test_unified_goal.py::test_episode_memory_does_not_cross_envs`:
  pollutes env-A memory with every learned-state kind, switches to env-B, asserts a new object
  identity and pristine state.
- **Independent confirmation it was already leak-free:** with the relational bias OFF
  (`W_UNIFIED=0`), the re-instantiated agent reproduces B5's r11l mean **2.934** exactly
  (3.69, 4.76, 0.35) — identical to the v2 shared-object run. So the prior reset path had no
  leak; v3 makes the boundary explicit and assertable.

## Step 2 — Unified decision (both signals live, reward-arbitrated, anti-fixation)

`DecisionRule(unified=True)`: pre-reward, click exploration gets a relational bonus that
**decays per-click-on-that-class** — `(_W_UNIFIED / (1+rank)) / (1+clicks_on_class)`. Intent:
each ranked candidate gets early attention (GoalProbe's strength), then the bonus backs off so
broad/random clicking returns and multi-step reward chains stay discoverable (B5's strength).
Once a click earns reward, `ClickToEffect` exploits it (reward arbitration); the nudge is
disabled once a rewarding class is known. Broad random-cell clicks are retained throughout.
Test `test_unified_spreads_clicks_constant_fixates`: the decaying bonus spreads clicks across
classes far more than the constant-bonus ablation (anti-fixation works as designed).

## Step 3 — Measurement: the irreducibility proof

`W_UNIFIED` sweep on r11l (the trade-off canary), 3 stable seeds:

| W_UNIFIED | r11l scores | r11l mean |
|---:|---|---:|
| **0.0** (≡ B5) | 3.69, 4.76, 0.35 | **2.934** |
| 0.5 | 0.05, 0.01, 0.28 | 0.113 |
| 1.0 | 0.02, 0.71, 0.00 | 0.243 |
| 1.5 | 0.00, 0.00, 0.00 | 0.000 |
| 2.2 | 0.00, 0.00, 0.00 | 0.000 |

**Even the smallest non-zero bias (0.5) craters r11l (2.93 → 0.11).** The anti-fixation decay
did not help — r11l fails on the *first* perturbed clicks, long before the bonus decays.

Worse, unified is **strictly dominated**. Its holdout (5 holdout envs, 3 seeds, W_UNIFIED=2.2)
is **0.0008** (lp85 0.0038; **sk48 0.000** — it *lost* the env GoalProbe unlocked; vc33/tu93/
dc22 0). So:

| | train | holdout | r11l |
|---|---:|---:|---:|
| B5 | **0.147** | 0.0002 | **2.93** |
| GoalProbe | 0.003 | **0.0030** | 0.00 |
| **unified** | ~0.000 | 0.0008 | 0.00 |

The decay spread clicks enough to *lose* sk48 (no persistent probing) yet still perturbed
r11l's early stream enough to break it — **the worst of both, winning nowhere.** Same r11l
regression as GoalProbe and the constant-bias variant: **three independent mechanisms, one
result.**

### Why it is irreducible (the real finding)
r11l's reward is a multi-step pickup→place interaction whose discovery needs the *unbiased*
broad click distribution. Relational ranking is built on visual distinctness/centrality/
containment — exactly the wrong prior for a goal that is a *relation between clicks*, not a
distinct object. So the ranking actively mis-prioritises r11l's reward target, and because the
earliest clicks set up the sequence, even a weak, decaying nudge on them is fatal. There is no
reward signal yet at that point to down-weight the bad prior — arbitration has nothing to act
on until the budget that would have found the reward is already spent. The styles are not
two settings of one dial; they are incompatible allocations of the same scarce early budget.

---

## Submission decision — FLAGGED for the user (not picked autonomously)

Per the fallback: the trade-off is irreducible, so the submission default is one of the two
single agents. Both are reproducible 3-seed means under the fixed harness; realistic random
floor is train 0.098 / holdout 0.0003.

| agent | train | holdout | character |
|---|---:|---:|---|
| random (floor) | 0.098 | 0.0003 | — |
| **B5** (life+retry) | **0.147** | 0.0002 | robust; r11l 2.93; clears lp85 L1 deterministically; ties random on holdout |
| **GoalProbe** | 0.003 | **0.0030** | breaks r11l; raises holdout 15× (sk48+lp85, unseen); loses train |

- **B5** beats the floor on TRAIN (1.5×) and is robust across env types — the safer bet for the
  mixed private 110-env set, but it only ties the floor on holdout.
- **GoalProbe** is the only agent that beats the floor on HOLDOUT (the gate metric, 10×) and
  generalises to two unseen envs — but it sacrifices a whole class of envs (efficient
  sequence-click like r11l), so it is riskier on the private set.

Neither dominates; this is a genuine judgement call about the private-set composition.
**Recommendation deferred to the user** (decision due 2026-06-25). No submission this phase.

## What is kept
- Per-env boundary re-instantiation (correctness; keep).
- `unified_goal` / `discovery_unified` agent + tests — kept as the documented, tested,
  rejected unification attempt (evidence for the irreducibility finding).
- B5 (`discovery`) and GoalProbe (`discovery_goalprobe`) unchanged — the two candidate agents.
- 40 unit tests green.
