# Phase 3 v2 Results — Death Model, Harness Fix, Goal-by-Interaction

**Date:** 2026-06-01 · **Branch:** `main` · Local, no submission.
Holdout = vc33, tu93, sk48, lp85, dc22 (never tuned on). 5×Σbaseline budget per env.

This phase resolved the death model (Task A → Verdict C), fixed the harness that was
silently capping every env at its first death, **discovered and fixed a process-level
determinism bug that made every prior measurement an unreproducible single draw**,
re-baselined random honestly, added per-env life-model learning + retry-with-knowledge
(Task B), and added relational goal-inference-by-interaction (Task C).

---

## Headline

| metric | value | note |
|---|---|---|
| Death model | **Verdict C** (non-terminal) | RESET revives via level_reset; canonical loop never stops on GAME_OVER |
| Harness bug | **confirmed + fixed** | old loop `break`-ed on first GAME_OVER → self-inflicted 0 on every env that died before L1 |
| Determinism bug | **found + fixed** | `per_env_seed` used salted `hash()` → different seed every process; all prior scores were 1 unreproducible draw |
| Random floor (FIXED harness, 5 seeds) | train **0.098 ±0.099**, holdout **0.0003 ±0.0004** | reproducible; was reported 0.0000 |
| Discovery agent B5 (FIXED harness, life+retry) | train **0.147 ±0.094**, holdout **0.0002 ±0.0001** | beats random on train; holdout = floor |
| Discovery agent + Task C (GoalProbe) | train **0.003**, holdout **0.0030** | **trade-off**: holdout 15× up, r11l train → 0 |
| ls20 rule-recovery | **1/5** | + new life model: 30 deaths / mean-life 131 / 4 lethal actions learned |

---

## Task A — Death model (recap, full detail in `death-model.md`)

**Verdict C: death is NON-terminal.** Engine source + canonical agent + empirical probe
agree: `lose()` sets GAME_OVER which freezes only non-RESET actions; RESET does a
`level_reset` (revive on current level, completed-level score preserved, costs 1 action;
engine `_action_count` itself skips RESET but the **scorecard counts RESET as 1 action** —
`arc_agi/scorecard.py:701-704` bumps both `resets` and `actions`). The canonical
`random_agent` has the GAME_OVER stop-case explicitly commented out. Our `eval/harness.py`
broke on GAME_OVER → capped every env at its first death.

---

## Task B — Harness fix + life learning + retry + re-baseline

### B1 — Harness fix (faithful to engine + canonical loop)
`eval/harness.py`: extracted a unit-testable `_run_episode`. On GAME_OVER it now issues
RESET (revive) and **continues**; the episode terminates only on WIN-all-levels or when the
action budget (5×Σbaseline) is exhausted. RESET costs 1 action and is attributed to the
current level's `level_actions` (the honest RHAE cost of retries). All Phase-2.5 early-exit
heuristics (level-1 cap, progress-window) removed — they were the very thing capping retries.

Isolation tests (`tests/discovery/test_harness_death_reset.py`, 4 cases): scripted
death→reset→revive against a fake engine asserts (a) the episode does NOT stop on first
death, (b) action accounting counts the forced RESET, (c) completed levels survive a death,
(d) termination is WIN or budget-exhausted. Live confirmation: random on r11l now does 52
deaths / 52 resets across the full 1215-action budget instead of stopping at death #1.

### Determinism bug (found while re-baselining) — **the most important finding**
`per_env_seed = hash((env_id, run_id, version)) & 0xFFFFFFFF` relied on Python's builtin
`hash()`, which is **salted per process** (PYTHONHASHSEED unset). So every interpreter launch
drew a *different* per-env seed: two back-to-back random runs gave train 0.0091 vs **0.2425**
(11/25 envs differed; sp80 0.082 vs 4.762). **Every score in v0/v1/2.5 was a single
unreproducible draw, not a reproducible measurement.** Fixed with a blake2b-based
`stable_seed` (`src/arc_agi_3_agent/seeding.py`); both agents now seed identically across
processes (verified). This is why prior "holdout 0.0000 / r11l 3.69→3.16" wandered.

### B2 — Re-baselined random floor (FIXED harness, 5 stable seeds)
**train mean 0.0981 ±0.0991 · holdout mean 0.0003 ±0.0004.** Per-seed holdout
`[0.0, 0.0001, 0.0006, 0.001, 0.0]`. Random now *reaches* level 1 on some envs given the
full budget (r11l 4/5, sp80 4/5, m0r0 4/5; holdout lp85 3/5, sk48 1/5) but RHAE ≈ 0 because
it burns thousands of actions to do so. Train mean is dominated by sp80's lucky fast clicks
(mean 1.75, max 4.76) and cd82 (mean 0.48). **The honest holdout floor is ≈ 0.000** — but
note random can now *complete* L1 on 2/5 holdout envs, so the bar is "complete L1
*efficiently*", not merely "complete L1".

### B3 + B4 — Life model + retry-with-knowledge
World model (`world_model.py`): `observe()` now branches on death/revival.
- **Death observe** (`_record_death`): learns lethality generically — a click death blames
  `(ACTION6, class)`; a directional death blames the **cell** the controllable moved into (a
  positional hazard), never the whole direction (which would freeze movement). Counts deaths,
  records the resource extents at death. Nothing per-env is hardcoded (ls20 key stays in the
  grader only).
- **Revival discontinuity** (`_on_revive`): a RESET reloads the level, so the frame jumps back
  to the level start. That jump is NOT attributed as an effect of action 0 (which would poison
  the model). Layout state (objects, strips, tried-candidates) resets; the learned model
  (effects, controllable, move-vectors, goal, reward classes, lethal set, lethal cells) is
  KEPT — so the retry is smarter than the first attempt.
- **Retry-smarter** (`movement.py`): the pathfinder routes around `lethal_cells()` (same layout
  on level_reset → same hazards). The agent self-issues RESET on GAME_OVER (`decision.py`),
  matching the canonical loop and keeping its decision log consistent.

Tests: `tests/discovery/test_life_model.py` (3 cases) — revival doesn't pollute the model,
death records lethality, directional death records a lethal cell not a frozen direction.

### B5 — Discovery agent under the fixed harness (3 stable seeds)
**train mean 0.1471 ±0.0940 · holdout mean 0.0002 ±0.0001.** The agent beats the
re-baselined random floor on TRAIN (0.147 vs 0.098) but on HOLDOUT it is statistically
identical to random (0.0002 vs 0.0003 — both ≈ 0).

| env | H | score mean | max | L1 in N seeds | note |
|---|---|---:|---:|---|---|
| r11l | . | **2.934** | 4.762 | 3/3 | completes L1 every seed (click) — the train driver |
| lf52 | . | 0.009 | 0.018 | 2/3 | |
| lp85 | **H** | 0.001 | 0.002 | **3/3** | completes L1 deterministically (random: 3/5) but RHAE ≈ 0 (budget burn) |
| vc33 tu93 sk48 dc22 | H | 0.000 | 0.000 | 0/3 | uncracked — dominate the holdout mean |

**Reading:** the harness fix + life/retry makes the agent reliable where it already works
(r11l up to 2.93 mean; lp85 L1 now deterministic) but does NOT move holdout off the floor.
The honest cause: completing L1 is not enough — RHAE needs it done *efficiently*, and on the
4 holdout envs it cannot crack (no controllable found / no rewarding click discovered) it
spends the whole budget. Per-env budget context: agents reach first death well before the
budget (r11l ~34, vc33 ~50, sk48 ~251) so retries are plentiful (30–50 per env) — retries
help r11l/lp85 reliability but cannot manufacture a goal the agent never infers.

---

## Task C — Goal inference by interaction (measured separately)

New `interaction.py` ranks objects as CANDIDATE goals by **relational** features (not pure
saliency, no semantic labels): distinctness (rare color / atypical size), centrality
(closeness to the active-region centre), containment ("hollow/frame-like" — bbox ≫ filled
cells), and match-potential (same color/shape as the controllable → a deliver-to-match
target). The `GoalProbe` strategy (priority 55) systematically **interacts** with the top
candidates — clicking each ranked class up to K=2 — and watches for any reward/level signal;
once a click earns reward, `ClickToEffect` takes over. This attacks the v1 blocker directly:
random clicking is budget-inefficient; ranking + probing the best candidate first finds the
rewarding class far sooner. Move-onto and transformer interactions remain with
MovementPathfinding / AttributeMatching.

Enabled via `DiscoveryAgent(goal_by_interaction=True)` / harness agent `discovery_goalprobe`
(off by default so B-vs-C deltas are attributable). Tests: `tests/discovery/test_goal_probe.py`
(4 cases) + relational-ranking unit tests.

### Result (3 stable seeds) — a real trade-off, net-positive on the gate metric

| | random B2 | discovery B5 | **GoalProbe C** |
|---|---:|---:|---:|
| **train** | 0.098 | 0.147 | **0.003** |
| **holdout** | 0.0003 | 0.0002 | **0.0030** |

| env | H | B5 mean | C mean | note |
|---|---|---:|---:|---|
| r11l | . | **2.934** | **0.000** | C **breaks** r11l (the train star) |
| lf52 | . | 0.009 | 0.054 | C up |
| sk48 | **H** | 0.000 | **0.012** | C unlocks (seed 2: 0.037) — unseen env |
| lp85 | **H** | 0.001 | 0.003 | C up (reward class found deterministically) |
| tn36 | . | 0.000 | 0.001 | C up |
| vc33 tu93 dc22 | H | 0.000 | 0.000 | neither cracks |

**Task C lifts holdout 15× (0.0002 → 0.0030) and generalises to two unseen envs (sk48, lp85),
but craters train by breaking r11l (2.93 → 0).** Root cause: r11l's reward is a pickup→place
*sequence*; relational interaction over-focuses clicks on the most "goal-like" object and
starves the sequence. Holdout per-seed `[0.0005, 0.001, 0.0074]` — the gain is real but noisy
(sk48 only on seed 2; lp85 on 2/3).

### Was the trade-off just GoalProbe monopolising? No — tested and falsified.
Hypothesis: GoalProbe (priority 55, fires whenever no reward is known) *replaces* exploration
on dynamic click envs; making relational interaction a *bias* on exploration instead of a
strategy should keep r11l. Implemented (`relational_explore` flag / `discovery_relexplore`):
explore still samples every action/click, click targets just get a decaying relational bonus.
Smoke: **r11l still scores 0** (the bonus makes the agent fixate on the top-ranked class).
**The trade-off is inherent to relational goal-biasing, not a monopoly artifact** — when the
goal is a relation/sequence rather than a distinct object, ranking-by-distinctness mis-leads.
Both Task-C variants are kept as flags (one tested ablation each); neither is the default.

---

## ls20 rule-recovery

Run under the fixed RESET-and-continue loop at full budget (`eval/ls20_recovery.py`), ls20 is
a TRAIN env. **Recovered 1/5** (unchanged headline from v1):
- ✅ `transformer_changes_attr` — 10,382 transformer events seen across color/rotation/shape.
- ❌ `goal_has_target_attrs`, ❌ `reward_on_match` — ls20 never completes a level, so no
  goal/reward is confirmed (goal=None, reward_events=0). Goal inference needs a first success.
- ❌ `resource_depletes`, ❌ `resource_or_lives_terminal` — the row/col strip-extent detector
  does not promote ls20's budget bar (its filled-extent isn't cleanly monotone under our
  measure), so the resource→terminal causal link stays unrecovered.

**New in v2:** the life model *did* learn the death side — 30 deaths / 30 revives,
mean-actions-per-life ≈ 131, 4 lethal actions identified — but that is "deaths happen + which
actions kill", not "a depleting resource causes death". Honest recovery is therefore still
1/5; the gap is the resource detector, a known v1-deferred secondary (ls20 is train, so this
is a legitimate future improvement target, not a holdout concern).

---

## Honest assessment

**What this phase proved.**
1. **The death model was misread by our own harness.** Verdict C is solid (engine + canonical
   + probe). The harness self-inflicted a 0 on every env that died before clearing L1. Fixed.
2. **Every prior number was noise.** The salted-`hash()` seed meant v0/v1/2.5 scores were
   single unreproducible draws (a back-to-back random run swung train 0.009→0.243). This
   alone explains the "r11l 3.69→3.16" wandering we had rationalised as RNG-ordering. With
   `stable_seed` the harness is now reproducible run-to-run, and floors/agents are reported as
   means over seeds — the first honest, repeatable measurements in the project.
3. **The harness fix helps reliability, not the holdout ceiling.** Under the fixed loop the
   agent beats the re-baselined random floor on TRAIN (0.147 vs 0.098, driven by r11l ≈ 2.93)
   and now clears L1 on holdout lp85 *deterministically* (3/3 vs random's lucky 3/5). But
   holdout RHAE stays at the floor (≈ 0.0002 vs 0.0003): completing L1 is not enough — RHAE
   rewards *efficiency*, and on the 4 holdout envs the agent cannot crack it burns the budget.

**Generalisation, stated plainly.** Two operating points on the 5 frozen holdout envs:
- **B5 (robust default):** clears L1 on **1/5 (lp85)** deterministically; train-strong (r11l 2.93).
- **Task C (GoalProbe):** clears/raises **2/5 (lp85 + sk48)** — sk48 was 0 under B5 — lifting
  holdout 15× to 0.0030, but it **breaks r11l** (the train star → 0). Net: better holdout,
  worse train. This is a genuine trade-off (relational goal-bias helps "click a distinct
  object" envs, hurts "do a sequence" envs); the bias-only `relational_explore` variant was
  tried and showed the *same* r11l regression, so the trade-off is inherent, not plumbing.

Neither clears vc33, tu93, dc22. Both results are on envs never tuned on, so the coverage that
does appear (lp85, sk48) is genuine generalisation — modest, noisy, real.

**Why the wall is real (unchanged from v1, now on firmer footing).** From-scratch,
single-episode discovery that must *both* infer an unknown goal *and* execute it efficiently
within 5×baseline is the benchmark's hard core. Our substrate is sound and now correctly
measured: perception, segmentation, action→effect, controllable detection, click-by-class,
life/lethality learning, retry-with-knowledge, and relational candidate ranking all work and
are unit-tested. The missing ingredient remains budget-efficient goal inference on envs whose
goal is neither "reach the salient object" nor "click one class" — exactly what humans get
from lifetime priors.

**Bearing on the 2026-06-25 submission.** The realistic random floor is now a reproducible
**holdout 0.0003 / train 0.098**. The discovery agent beats it on train and ties it on
holdout. Per the updated CLAUDE.md §1.2 the 2026-06-25 entry submits the best *stable* agent
beating the realistic floor by the clearest margin — currently the discovery agent (train
0.147), well short of the ≥10 early-trigger. No submission this phase.

**Reproducibility note.** With `stable_seed`, a given `--run-id` reproduces byte-for-byte
across processes — **verified**: random `run_id=0` re-run in a fresh process gave train
0.1964 == 0.1964, holdout 0.0 == 0.0, **0/25 envs differing** (before the fix, two runs
differed on 11/25 envs). Multi-seed means are over `run_id ∈ {0,1,2}` (agent) / `{0..4}`
(random).
