# Submission Tiebreaker — B5 vs GoalProbe (decision due 2026-06-24)

**Date:** 2026-06-01 · Analysis only, no submission, no agent changes, no holdout tuning.
Inputs: `scenario-taxonomy.md` archetype assignments + the committed 3-seed per-env means
(`p3v2_b5_discovery_ms`, `p3v2_c_goalprobe_ms`). Random floor: train 0.098 / holdout 0.0003.

## TL;DR recommendation: **GoalProbe (tentative)** — but it is genuinely close and rests on one assumption (see the magnitude caveat). Final call to the user on 2026-06-24.

---

## Task A — Public-set archetype composition (proxy for the private 55)

Primary archetype, one per env (cross-cutting resource/selection/undo excluded as primary):

| primary archetype | n | envs |
|---|---:|---|
| attribute_matching | 11 | ar25, cd82, cn04, ft09, ls20, re86, s5i5, sb26, sc25, **sk48ᴴ**, tr87 |
| click_to_effect | 5 | lf52, r11l, tn36, **vc33ᴴ**, **lp85ᴴ** |
| movement_pathfinding | 5 | g50t, ka59, m0r0, wa30, **tu93ᴴ** |
| physics_cascade | 3 | bp35, sp80, su15 |
| ambiguous/unknown | 1 | **dc22ᴴ** |

**Mapping to the goal-style families that actually separate B5 from GoalProbe.** The two agents
are IDENTICAL except in click exploration (B5 = broad/unbiased clicks; GoalProbe = relational
candidate probing). So the family split only bites on **click-goal** envs:

- **distinct-object click** (goal = click ONE relationally-distinct object) → GoalProbe's lever:
  **vc33ᴴ, lp85ᴴ, lf52, tn36** = **4**.
- **sequence/relation click** (goal = a multi-step click interaction, r11l's pickup→place) →
  B5's lever: **r11l** = **1**.
- **lever does not apply** (movement 5, physics 3, attribute 11, dc22 1) → both agents behave
  ~identically; here the choice is a wash = **20** (incl. all 11 attribute envs, where clicking
  the target does not reward without a match, so GoalProbe's probe has no clean hook).

**Composition verdict:** of the envs where the choice *matters at all* (5 click envs), the
distinct-object family (GoalProbe) outnumbers the sequence family (B5) **4 : 1**. The other 20
envs are a wash on this axis. So the public set does NOT have a sequence-click majority — the
opposite: distinct-object click envs are 4× the sequence-click envs.

---

## Task B — Per-archetype agent advantage (grounded in data, not anecdote)

Only **5 / 25** envs score above 0 for either agent; the other 20 tie at 0.0000 (uninformative).

| env | Hold | family | B5 mean | GoalProbe mean | winner |
|---|---|---|---:|---:|---|
| r11l | . | sequence-click | **2.9336** | 0.0000 | **B5** (huge) |
| lf52 | . | distinct-object | 0.0085 | **0.0542** | GoalProbe |
| sk48 | **H** | attribute(mixed) | 0.0000 | **0.0123** | GoalProbe |
| lp85 | **H** | distinct-object | 0.0011 | **0.0025** | GoalProbe |
| tn36 | . | distinct-object | 0.0000 | **0.0011** | GoalProbe |
| *(other 20)* | | various | 0.0000 | 0.0000 | tie |

**Does the family → agent mapping hold?**
- **GoalProbe's side: CONFIRMED.** It wins 3/4 distinct-object click envs (lf52, lp85, tn36; vc33
  stays 0 for both) **plus** sk48 — and 2 of its wins (sk48, lp85) are on UNSEEN holdout envs.
  Its advantage is family-wide and generalises.
- **B5's side: NOT CONFIRMED as a family advantage.** B5's only win is **r11l** — a single env.
  On the other 11 "sequence/relation" envs (the whole attribute_matching family) B5 scores
  **0.0000**, tying GoalProbe. So B5 has no realised *family-wide* edge; it has one
  spectacular **single-env** edge, and that env is in TRAIN (no unseen sequence-click env exists
  in holdout to confirm the capability transfers).

This is the inconsistency the decision rule warns about — here it falls on **B5's** side: the
"sequence family favours B5" claim is really "r11l favours B5". One env, not a family.

---

## The decision, honestly (both views)

This is a fight between two profiles on an RHAE metric that is **magnitude-dominated** (one
env scoring 2.9 outweighs hundreds of envs scoring 0.01):

- **B5 = narrow & deep.** Wins exactly one env, but by 2.93. IF the private 55 contains
  r11l-class (sequence-click place-to-constraint) envs AND that capability transfers, B5's few
  big wins dominate the private RHAE. Public set has ~1/25 such env → ~2/55 expected.
- **GoalProbe = broad & shallow.** Wins 4–5 envs but each ≤ 0.05. Confirmed to generalise to
  unseen envs (sk48, lp85). Beats the floor on the holdout gate metric (0.0030 vs 0.0002, 15×);
  B5 only ties the floor on holdout.

**Why GoalProbe is the recommended tentative default (decision-rule clause 2):**
1. **Gate metric.** We are ranked on holdout-style RHAE; GoalProbe is the only agent that beats
   the floor there. B5 "ties the floor on holdout" = scores ~nothing on the ranking metric.
2. **Confirmed generalisation.** GoalProbe's wins include 2 UNSEEN envs. B5's single win is a
   TRAIN env, and there is **no unseen sequence-click env in holdout to confirm it transfers** —
   its big advantage is unverified on unseen data, exactly the BC failure mode we are wary of.
3. **Breadth of realised wins.** GoalProbe wins 4–5 envs across its family; B5 wins 1. If
   r11l-class envs are rare in the private set, B5 has nothing and GoalProbe's spread scores.

**The one assumption that could flip it (flag loudly):** if the private 55 contains a
meaningful number of r11l-class sequence-click envs (≥ ~3) AND B5's algorithmic r11l capability
transfers to them, B5's magnitude wins would beat GoalProbe's shallow spread on RHAE. We cannot
test this — holdout has zero r11l-class envs. The public taxonomy has exactly one (r11l), so the
base rate looks low (~4%), which is why the recommendation leans GoalProbe — but a user who
believes sequence-click envs are common in the private set should pick **B5**.

---

## Recommendation for 2026-06-24

1. **Default: GoalProbe** (`discovery_goalprobe`) — best on the gate metric, only confirmed
   generalisation, broad realised wins; B5's edge is one unverified train env.
2. **Override to B5** if, by 2026-06-24, there is reason to believe r11l-class sequence-click
   envs are common in the private set (then RHAE magnitude favours B5's deep wins).
3. **Before submission (either way):** re-run the chosen agent's 3 seeds and confirm
   byte-identical reproduction (stable_seed), so the submitted number is the measured one.
4. User makes the final call and authorises the submission per-message on 2026-06-25
   (1/day quota; no autonomous submission).
