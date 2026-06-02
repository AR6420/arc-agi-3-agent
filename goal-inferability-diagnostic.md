# Goal-Inferability Diagnostic — Blind Frame Walkthrough

**Date:** 2026-06-01 · Diagnostic only. No agent, no model wiring, no submission, not scored.

## What this can and cannot claim (read first)
- **CAN establish:** whether the goal is inferable IN PRINCIPLE from the observations a fresh
  reader receives — here, a *shallow* blind read (initial frame + a few generically-chosen
  exploratory frames).
- **CANNOT establish:** that a fresh automated agent could infer the goal within budget. A
  reasoner reading frames with full attention is a weaker, in-principle test, not proof an
  agent could. Every conclusion below is framed this way.
- **Extra caveat that turned out to dominate:** the read was *shallow* — ~3–7 blindly-chosen
  actions per env, vs the agent's hundreds-to-thousands. So "not-inferable from these frames"
  largely means "not inferable from the first handful of exploratory frames", NOT "the goal is
  under-determined from full interactive observation." This confound is large; see §Limitation.

## Integrity & contamination audit (this is the load-bearing part)
- **The main session is CONTAMINATED.** Earlier this session I read `scenario-taxonomy.md`, the
  ls20 framing, and per-env archetype assignments. I therefore **could not be the blind reader**.
  I explicitly set that knowledge aside and did NOT use it to perform the reads.
- **The reads were done by 4 FRESH subagents**, each with none of this session's context, each
  given ONLY one neutral-labeled montage image (`GameA..D`) and a strict instruction to read
  no other file (especially nothing under `environment_files/` and no `*.py` source) and use no
  outside knowledge of the specific game. **All 4 self-reported clean** (each made exactly one
  tool call — Read on its single image — and confirmed no other file access).
- **Rendering did not read source.** Frames were produced by RUNNING the engine (the
  observations the agent receives), never by opening an env `.py`. Sample actions are a fixed,
  env-agnostic explorer policy (each available non-RESET action once, then 3 generic clicks at
  centre + 2 non-background cells), not derived from any known solution.
- **Disclosure:** I (contaminated) viewed ONE montage (`GameD`) once for render-quality QA. That
  does not bias the subagents' reads (they are the data); it only let me confirm the images were
  legible. I did not view the others before reading the subagents' verdicts.
- **Hidden mapping** (kept by the compiler, never shown to the readers): GameA=r11l,
  GameB=vc33, GameC=tu93, GameD=sk48. Chosen per the brief: 1 env our agent CAN crack (r11l,
  GameA) + 3 it cannot, incl. holdout (vc33, tu93, sk48).

---

## Per-env blind reads (from the fresh subagents)

### GameA = r11l  *(the env our agent CAN solve)* — available actions [6]
- **Goal hypothesis (conf low):** reposition a black "+" token (rubber-band-linked through a
  hexagon node to a green "+" anchor) to reach a persistent dashed diamond marker.
- **Evidence:** CLICK moves the black token toward the clicked cell; the connecting line
  reorients; the dashed diamond never moves (→ candidate target by persistence/style only). **No
  HUD, no score, no win event in any frame.**
- **Confirm experiment:** click directly on the dashed diamond; expect a fill/level-advance if
  it is the goal.
- **Verdict: not-inferable-from-these-frames.** Clicks clearly *do something* (move the token),
  but nothing shows what wins; equally consistent with "drag node onto hexagon", "avoid red
  wall", or no goal.

### GameB = vc33  *(holdout; agent fails)* — available actions [6]
- **Goal hypothesis (conf low):** move a yellow agent upward through a gap in a gray bar toward
  an orange top strip.
- **Evidence:** clicks reposition the agent vertically; a tiny yellow segment appears at the
  top-right (possible progress indicator, too small to be sure); **the agent returns to its
  start in the last frame — no monotone progress shown**; no terminal/level event.
- **Confirm experiment:** click above the agent past the gray-bar gap; watch for the top-right
  HUD growing / level-complete; or click onto a red square to test hazard-vs-target.
- **Verdict: not-inferable-from-these-frames.** Cannot distinguish "reach the top" vs "touch
  red" vs "align with the gap" vs other.

### GameC = tu93  *(holdout, pure movement; agent fails)* — available actions [1,2,3,4]
- **Goal hypothesis (conf low):** maze traversal — move a yellow marker (top-left) to a cyan
  tile (bottom-right).
- **Evidence:** classic start/goal/maze color layout (yellow vs cyan, opposite corners, red
  walls). **But across ACTION1–4 the reader could NOT see the yellow marker move at all** — only
  a small change in a bottom magenta bar (candidate budget/timer). So the controllable element
  and per-action effect are not actually demonstrated.
- **Confirm experiment:** repeat one directional action and watch whether yellow translates one
  cell toward cyan and the bar shrinks; or drive into a red wall to test blocking.
- **Verdict: not-inferable-from-these-frames.** Layout is suggestive but nothing is shown to
  move or respond; goal inferred only from static color placement.

### GameD = sk48  *(holdout, mixed; agent fails)* — available actions [1,2,3,4,6,7]
- **Goal hypothesis (conf low):** navigate a pink-bordered player token (ACTION1 visibly moves
  it) along a yellow board toward cyan/dark-red marker squares; a green-on-red vertical bar +
  bottom HUD strip present.
- **Evidence:** ACTION1 jumps the player up with a short trail (clear controllable); ACTION2
  relocates it; clicks cause only minor marker nudges; cyan/red markers shift rows/shade between
  frames (possible align/sort objective); green/red left bar = candidate progress/lives gauge.
- **Confirm experiment:** ACTION1 ×3–5 from start; see if the token walks toward the markers and
  whether reaching one changes state / fills the bar.
- **Verdict: not-inferable-from-these-frames.** Controllable token + that it moves are clear;
  the win condition (reach target? align markers? fill bar?) is not determinable.

---

## Result

**4 / 4 envs: not-inferable-from-these-frames — including r11l, the env our agent solves.**
No fresh reader, with full attention on the rendered frames, could identify a win condition for
any env. For the two pure-movement-ish envs the readers could not even confirm what was
controllable from the few frames.

### The r11l tell (the most useful single finding)
The one env our agent **can** crack is **also** not goal-inferable from a blind static read. So
the agent's r11l success is **not** goal-inference — it is interactive reward-discovery (it
clicks ~hundreds of times until reward happens to fire, then exploits the class). This is
consistent across solved and unsolved envs: **the agent never infers goals; it succeeds only
where blind interaction stumbles into reward within budget, and fails where it does not.**

## Limitation (why this does not cleanly pick a branch)
The read was **shallow**: ~3–7 blindly-chosen actions per env. The agent itself uses
hundreds-to-thousands. Three of four readers explicitly noted they could not see progress / a
win / even the avatar moving *because the exploration was too short* (vc33's agent even returned
to start). So "not-inferable" here is dominated by **under-sampling**, not proven goal
under-determination. A fair in-principle test needs a **deeper interactive blind read** (let the
reasoner choose follow-up actions and watch for a level-advance).

## Which decision branch the evidence points to

- **Branch 1 — the wall is (partly) the TASK: goals are not cheaply inferable from observation.**
  **SUPPORTED for the cheap/early-observation regime.** Even a capable reasoner cannot infer
  these goals from the first handful of exploratory frames. This validates that "infer the goal
  from cheap blind observation" is infeasible — the path our algorithmic agent was implicitly
  betting on does not exist at low observation cost.

- **Branch 2 — goals CLEARLY inferable to a reasoner but the agent missed them → offline
  goal-prior distillation from frames.** **NOT supported by this test.** The goals were *not*
  clearly inferable to the reasoner either, so there is **nothing in the frames alone to
  distill** — a small model trained to predict goals from frames has no signal a capable reader
  couldn't already extract, and the reader extracted none. Frame-based prior distillation is a
  dead end on this evidence.

**Net:** the diagnostic points to **Branch 1**, with the honest caveat that the shallow read
under-tests inferability, and with the r11l tell reframing the agent's wins as
reward-discovery-by-interaction rather than inference.

## A separate, untested lever (flagged, NOT started)
Branch 2's *frame-based* distillation is dead, but a different information source remains legal:
**dev-time extraction of goal priors from the public env SOURCE** (CLAUDE.md §7.2 "read
everything, copy nothing" permits reading the 25 public sources for dev-time priors, capped at
25% synthetic per §4.8). That would distill priors a capable reader *can* get from source (not
from blind frames) into a small fast model that proposes goal hypotheses an explorer tests. It
is a real, scoped candidate — but it is a different lever than this diagnostic tested, it risks
the BC over-fitting failure if priors don't transfer to the 110 private envs, and it must be a
**joint decision**, not started here.

## Recommended next step (for joint decision)
1. If we want to firmly settle Branch 1 vs Branch 2: run a **deeper interactive blind read** —
   a fresh reasoner that issues its own follow-up actions (still no source) over a real budget,
   and report whether a level-advance ever becomes visible/inferable.
2. The only constraint-legal capability lever this surfaces is **source-based goal-prior
   distillation** (above), explicitly flagged for a joint go/no-go, with the over-fitting risk
   stated. No build, no submission taken here.
