# Scenario Taxonomy — Phase 3 Discovery Agent (Part 1)

**Date:** 2026-05-29 · **Status:** review gate — NO agent code until approved.
**Method:** 20 TRAIN env sources read in parallel (one reader each), mechanics extracted, synthesized into experiment-based archetypes, then adversarially critiqued. **Hard Rule 6:** archetype definitions + detection signatures derived ONLY from the 20 train envs. Holdout 5 (vc33, tu93, sk48, lp85, dc22) are placed *provisionally* using surface metadata (`splits.py` notes) + observed S1 random behavior — **NOT** from reading holdout source.

**Core principle.** Every archetype is recognized by **experiment evidence** (issue action → watch board delta), never by static appearance. Static signals are *candidates only*; a click-puzzle and a movement-puzzle can look identical at rest. DISCOVER then EXPLOIT.

---

## 8 archetypes (4 required + 4 emergent)

| # | archetype | freq | tract | cross-cutting | required? |
|---|---|---|---|---|---|
| 1 | movement_pathfinding | high | high | no | ✔ required |
| 2 | attribute_matching (ls20 class) | high | medium | no | ✔ required |
| 3 | click_to_effect | high | medium | no | ✔ required |
| 4 | resource_management | high | high | **yes** | ✔ required |
| 5 | time_pressure_hazard | medium | medium | no | emergent |
| 6 | physics_cascade_simulation | low | medium | no | emergent |
| 7 | selection_indirection | medium | medium | **yes** | emergent |
| 8 | undo_reversibility | medium | high | **yes** | emergent |

Cross-cutting archetypes (4, 7, 8) layer ON TOP of the primary archetype of an env; almost every env carries ≥1.

---

### 1. movement_pathfinding  `freq high · tract high`

**Description.** A single controllable avatar (or small fixed set of pieces) translates by a fixed stride per directional action. Solving is spatial: route avatar (+ pushed objects) to a goal cell around walls, bounded by budget/hazard. Win is POSITIONAL overlap, not attribute equality.

**Detection.**
- *Static (candidate):* one salient sprite ≠ background; wall/blocker tiles; a goal cell of distinct color; corridor layout. None proves it.
- *Experiment (confirm):* issue ACTION1/2/3/4 in turn → confirm **exactly one connected region translates by a constant offset** (1, 3, or 4 cells) in the action direction, all else static, opposite actions reverse it. Zero-delta against specific tiles = walls (record collision footprint). A click puzzle produces NO motion from ACTION1–4 — that's the discriminator.

**Exploit.** Probe motion deltas → build occupancy map → BFS/A* from avatar to candidate goal using learned stride+footprint → execute shortest path. Re-probe after every level reload (stride/walls change). Multi-piece (m0r0): learn each piece's direction-transform first, then plan jointly.

**Exemplars (train):** g50t, ka59, m0r0, wa30. **Holdout:** tu93 (pure_movement) — *provisional*.

**Failure risks.** Stride ≠ 1 (re86/wa30 move 3–4; ls20 by sprite width) desyncs plans. Per-piece direction inversion (m0r0) breaks naive BFS. Pushed objects/followers mutate occupancy mid-path. Moving hazards invalidate static plans → replan each step. **tu93 scored 0.000 for BC — pure-movement is exactly where memorized priors die; the delta-probe must come first.**

---

### 2. attribute_matching  *(the generalized ls20 class)*  `freq high · tract medium`

**Description.** A carried/selected/controlled entity has one or more ATTRIBUTE dimensions — shape, color, rotation, scale, size, count, 3×3-pattern, canvas-fill — that the agent mutates via transformer interactions (stepping on transformer cells, clicking color/rotate/scale buttons, cycling a sprite, moving a basket that paints a canvas). WIN fires when entity attributes EXACTLY match a visible target, independent of or additional to position. Match is on **rendered attribute equality**, not on reaching a cell.

**Detection.**
- *Static (candidate):* a target/template sprite the entity resembles; palette/rotate/scale UI; transformer cells; result-canvas; goal sprites with encoded constraint patterns. Could be inert until interaction proves them live.
- *Experiment (confirm):* (1) **Mutation** — interact with a candidate transformer and confirm the controlled entity's RENDERED attribute changes (recolor/shape-swap/rotate/scale) while position may stay fixed; repeat to learn the CYCLE length (color mod-4, shape mod-6/7, rotation mod-4). **Transformers may be NON-SPATIAL** (ar25 class): the trigger can be an *action type* (selection-cycle, click) or a *state transition* (distance-delta to a target), not a cell you step on. Probe both: does any entity attribute change as a consequence of a selection/click action independent of spatial position? If so, the trigger is the action, not a cell. (2) **Target** — drive one axis toward the candidate target, watch for partial progress (highlight/flash/life-loss/no-op). (3) **Win** — advance fires when ALL axes equal target. Defining delta: an action changes an *attribute* not a *position*, and *equality-to-target* not *cell-occupancy* triggers advance.

**Exploit.** Single-axis probe each transformer (which axis it increments, cycle length). Read target tuple from its sprite. Small search over transformer applications to reach the target tuple respecting cycle modularity, then commit. Canvas/paint variants (cd82, re86): learn which direction/color marks which region, compose the fill. **Decouple axes — never co-vary two until each is mapped.** Treat wrong-match penalty as expensive: simulate the tuple before committing.
- *Constraint-template sub-path (ft09 class):* the target is not a fixed tuple but a *per-instance rule* encoded in the goal sprite's own pixels (e.g. each of 8 neighbor positions flagged "must-equal" vs "must-differ" by that sprite's pixel value at that offset). Exploit: **read the goal sprite's pixel pattern once at episode start to decode the constraint polarity** — this is a static pixel read (pixel-observable, not an action-delta), a legitimate one-time perception step — then drive neighbors to satisfy it. Without decoding polarity the agent cannot tell match from mismatch and will misfire.
- *Rule-induction sub-path (tr87 class) — anti-blowup:* when win depends on a learned left→right *sequence* mapping, brute force is `K^L` (≈7^5) and busts the budget. **Exploit per-position independence: probe each output position separately** (vary one position, hold others, read the partial win/no-op signal) → ~log₂(K) probes per position → ≤~15 actions for L=5 instead of thousands. If the env couples positions (no per-position signal), flag tr87-class as needing a dedicated rule-learning sub-module rather than blind search.

**Exemplars (train):** ls20, ar25, cn04, cd82, re86, tr87, r11l, s5i5, sb26, ft09, sc25. **Holdout:** sk48 (mixed, partial) — *provisional*.

**Failure risks.** Combinatorial blow-up if axes co-mutated (tn36 4-attr; tr87 rule-induction). Implicit transformers (ar25: rotation triggered by DISTANCE-delta to a target, not a cell) easy to miss — cue is an unrelated sprite rotating when you moved. Mismatch still costs budget/lives (sb26, ls20). Constraint-template envs (ft09: per-neighbor match/mismatch) require inferring a per-instance rule, not a fixed target. Rule-induction (tr87: learn a left→right mapping before any match) is the deepest sub-case.

---

### 3. click_to_effect  `freq high · tract medium`

**Description.** Primary control is ACTION6 (click x,y); effect is conditional on the sprite/region under the cursor. Different objects → qualitatively different mutations (toggle block, flip gravity, spawn clones, rotate/scale a color-family, place/merge a piece, select-then-move, run a program animation). The puzzle is discovering the click→effect map per object class, not navigation.

**Detection.**
- *Static (candidate):* button-like UI / color selectors / slot grids; no avatar moving under ACTION1–4. Plus the action-space hint `available_actions == [6]` — note this is a *pre-experiment action-space* signal, NOT a pixel-visible appearance, and is still only a candidate: ACTION6 also appears as a secondary input in non-click-primary envs (cd82 compass-select, ar25 piece-select). Never short-circuit to `click_to_effect` from `available_actions` alone; confirm the experiment signal.
- *Experiment (confirm):* click empty space vs distinct sprites, compare deltas. Confirm (a) ACTION1–4 produce no avatar translation / are unavailable, while (b) ACTION6 at DIFFERENT coords yields DIFFERENT sprite-dependent changes. Background click is often no-op / special gate animation. Signature: **effect is a function of the under-cursor sprite**, and repeated clicks on a toggle oscillate A→B→A.

**Exploit.** Build a **click-affordance map**: probe one representative click per visually-distinct sprite *class*, record its delta (toggle/spawn/select/place/transform/no-op). Identify which class advances the win-check. Two-phase objects: learn click-to-select then click/arrow-to-place. **Cluster clickable cells by under-cursor color/class to predict effect WITHOUT re-probing every pixel — this is the explicit fix for the BC spatial-head failure (memorized absolute pixels → 0.000 holdout).**
- *Place-to-satisfy-constraint sub-path (r11l class):* some click envs are not toggle/spawn but **click a piece from a supply → place it at a target → check a neighbor/connectivity constraint**. Exploit: enumerate supply pieces, probe place→delta, read whether the placement advanced the win-check or was a no-op; treat the target's rendered neighbor pattern as the constraint to satisfy (overlaps with attribute_matching's constraint-template read below). Click-affordance map must include a "place" effect class, not just toggle/select.

**Exemplars (train):** bp35, lf52, r11l, s5i5, su15, tn36. **Holdout:** vc33 (pure_click), lp85 (mixed) — *provisional*.

**Failure risks.** **The dominant historical failure:** BC spatial head memorized absolute pixels → vc33 sp5=0.008. MUST click by salience/under-cursor-class, never learned coordinate. Effects can be multi-frame animations — read delta only after animation settles. Some clicks silently revert on overlap (s5i5): immediate post-click frame misleads. Approx-click tolerance (±1–3 px) sometimes only moves a cosmetic cursor highlight (diff=1) not structural progress — distinguish cosmetic from structural deltas.

---

### 4. resource_management  *(CROSS-CUTTING)*  `freq high · tract high`

**Description.** On nearly every env: **(A)** a discoverable MONOTONE step/action/energy indicator depleting a fixed amount per action (and in a few envs REFILLING at cells/collectibles), bounding episode length; and SEPARATELY **(B)** a discoverable LIFE/retry counter whose depletion ENDS the episode independent of the budget. Both learned by watching a HUD bar/counter delta — never assumed to exist or behave.

**Detection.**
- *Static (candidate):* a bar/segment strip along an edge row (~row 63), numeric HUD, or a row of life cells. Could be decorative; could deplete per-action or per-time; could refill; lose-threshold unknown until observed.
- *Experiment (confirm):* **Budget** — diff the HUD region after each action; confirm monotone decrement of constant size; measure per-action-type cost (su15 merge 2+penalty; ar25 selection/click/undo free). Watch for INCREMENTS at cells/items = refill (sc25 collectibles). At zero, observe whether episode ENDS or level merely resets. **Lives** — a SEPARATE counter dropping only on discrete bad events (ls20 wrong-attribute collision; sp80 4th failed spill), ending episode at 0 even with budget left. Discriminator: budget falls *every* action; lives fall *only on failures*.

**Exploit.** Calibrate per-action-type cost + refill sources FIRST → compute a hard action ceiling and plan within it (also sets the per-env `MAX_ACTIONS` override per CLAUDE.md §9). Prefer free actions for exploration; spend costly actions only on committed moves. Treat lives as a hard simulate-before-commit constraint. Budget the exploration phase explicitly so discovery never exhausts the episode.

**Exemplars (train):** ar25, cd82, cn04, ka59, re86, s5i5, sb26, ls20, sp80, wa30 (≈ every env). **Holdout:** all 5 carry it — *provisional*.

**Failure risks.** Assuming uniform 1-per-action cost when merges/renders/collisions cost more or refills exist. Confusing budget bar with lives counter (different lose semantics). **OFFLINE harness does NOT increment scorecard counters (CLAUDE.md §2.5, §9) — the agent must read the in-frame UI bar locally, not scorecard fields.** Spending the whole budget on discovery.

---

### 5. time_pressure_hazard  `freq medium · tract medium`

**Description.** An AUTONOMOUS element advances every step (or every N steps) REGARDLESS of player action — a wall scrolling left, a chasing enemy/NPC, an obstacle approaching the avatar; episode LOST on contact / threshold-cross. A clock that runs on real moves, distinct from the action-budget; often makes do-nothing impossible.

**Detection.**
- *Static (candidate):* an edge sprite resembling wall/cursor/enemy; an NPC resembling the avatar; a second moving region. Could be inert scenery.
- *Experiment (confirm):* issue the SAME action twice and diff → confirm a NON-controlled sprite moved on its own between steps, **independent of the action chosen**. Count frames between its moves to get its period (every step vs every 2). Drive avatar near it → confirm contact flips a LOSE state. Signature: a sprite whose delta is decoupled from action choice but coupled to step count.

**Exploit.** Estimate arrival time (distance / period) → treat as a deadline constraint on the movement/attribute plan; prefer shortest path; replan every step (hazard map non-stationary). Chasers (ka59 enemy, wa30 NPC): model greedy/BFS pursuit, keep a safe Manhattan margin. Stop exploring once hazard is within striking distance.

**Exemplars (train):** g50t, tn36, ka59, wa30. **Holdout:** none identified — *provisional*.

**Failure risks.** Mistaking hazard for a controllable piece (it ignores input). Under-counting period (−1 per 2 turns vs per turn doubles the deadline; tn36 slows after L5). Static plans get caught at re-convergence. **Cooperative NPCs (wa30 auto-agent) HELP not harm — classify hazard-vs-helper by whether contact loses or progresses.**

---

### 6. physics_cascade_simulation  `freq low · tract medium`

**Description.** Agent sets up a configuration then triggers a DETERMINISTIC multi-frame simulation it doesn't directly control during playout: gravity fall, particle spill/bounce, block merge-on-collision, splitter cloning. Success = PREDICTING where the sim settles. Arrange-then-release, not step-by-step control.

**Detection.**
- *Static (candidate):* falling/particle/mergeable sprites; a dedicated release action (ACTION5) separate from movement; gravity-flip/bounce tiles; numbered/colored mergeables.
- *Experiment (confirm):* one input → MANY frames change over several engine steps (a queued cascade), not a single-cell delta. Confirm determinism (same setup+trigger → same settle). Perturb one element (move a wall/redirector) → watch how the settle pattern shifts. Gravity: after a horizontal move avatar auto-falls until solid. Merge: two like objects collide → one higher-order at midpoint. Spill: particles descend/bounce until none move, then match-check.

**Exploit.** Build a lightweight forward-simulator of the discovered physics (gravity dir, bounce rules, merge table) and PLAN by internal rollout before committing the trigger. Use undo (su15, bp35) to retry releases cheaply. Collector-fill (sp80): search avatar placements routing particles to uncovered collectors. Merge (su15): order clicks to build the quota. Gravity (bp35): pick the horizontal move whose fall lands on the gem, avoiding spikes.

**Exemplars (train):** sp80, su15, bp35. **Holdout:** none identified — *provisional*.

**Failure risks.** Reading the board before the cascade settles → wrong conclusion. Hidden modifiers (sp80 directional remap by rotation; bp35 gravity flip) invert predictions. Limited retries make blind release expensive. Splitter cloning changes object count mid-cascade. Treating the trigger as a single-step action desyncs from queued frames (**Animation T-stack surprise, CLAUDE.md §9**).

---

### 7. selection_indirection  *(CROSS-CUTTING)*  `freq medium · tract medium`

**Description.** The controlled entity is NOT fixed. Agent must first SELECT which piece/grid/basket/avatar it commands (click ACTION6 or cycle ACTION5); directional actions then act ONLY on the current selection. Mis-modeling makes movement *appear* non-deterministic (arrows moved different things at different times).

**Detection.**
- *Static (candidate):* multiple movable-looking sprites; a highlight/cursor; a cycle/click selector; carried/placeable indicators.
- *Experiment (confirm):* issue ACTION5 (or click a piece) → confirm a highlight moves / a different sprite becomes the one responding to ACTION1–4; then a directional action translates the NEWLY selected sprite while the old one stays. Confirm click-select (ACTION6) costs zero budget in some envs. Signature: the identity of the moving sprite changes after a selection action, tied to the last select/cycle.

**Exploit.** Maintain explicit current-selection state; always confirm selection before directional actions; exploit free selection/click for zero-cost exploration. Enumerate the selectable set (cycle ACTION5 once to map it) and plan per-piece sub-goals. Compass/indirection (cd82 selects a *direction*, not a piece): map selector-state → effect table before committing.

**Exemplars (train):** ar25, cn04, ka59, m0r0, re86, sp80, cd82. **Holdout:** sk48 (likely, mixed+undo) — *provisional*.

**Failure risks.** Assuming a fixed avatar → directional actions appear to control random pieces. Cycle order is a fixed circular list (ar25) that must be tracked. Click-select vs click-place ambiguity (m0r0). Selection may cost budget in some envs and not others — verify before relying on it for free exploration.

---

### 8. undo_reversibility  *(CROSS-CUTTING)*  `freq medium · tract high`

**Description.** An explicit UNDO action (ACTION7, or ACTION5 in some envs) pops a saved board state, enabling backtracking. When present it converts the puzzle from one-shot to search-with-backtracking and makes aggressive delta-probing cheap. Presence, cost, depth are discoverable.

**Detection.**
- *Static (candidate):* `available_actions` includes 7 (or documented undo on 5); a visible history hint. ACTION7 is a no-op in some envs.
- *Experiment (confirm):* make a state-changing move, issue the candidate undo, diff → confirm board returns to PRIOR state. Test multi-step rewind and cost: does undo decrement budget (bp35/lf52 cost; lf52 even ADDS +20) or is it free (ar25)? Confirm whether undo rewinds dependent/auto-moved objects (g50t followers).

**Exploit.** Undo confirmed + cheap → explore freely (probe action → read effect → undo) to build the world model at low risk; structure solving as DFS with undo as backtrack. Undo costs budget/lives → ration it. Track undo depth so the agent never rewinds past the stack limit.

**Exemplars (train):** ar25, bp35, g50t, lf52, sb26, su15. **Holdout:** sk48 (mixed+undo) — *provisional*.

**Failure risks.** Assuming undo is free when it costs (lf52 ADDS cost). Assuming it exists when ACTION7 is a no-op. Undo may not restore NPC/auto-agent state. Some envs have NO undo (sp80, ls20) → backtracking plans invalid there.

---

## Worked example — ls20 as attribute_matching + resource (DISCOVERY framing)

ls20 is the canonical attribute_matching env. The agent must **DISCOVER** (never be told) all of the following purely by acting and watching deltas:

| What the agent must discover (experiment) | Generic archetype mechanism |
|---|---|
| Some entity (bottom-left) responds to my actions and carries attributes | controllable + attribute entity (arch 1+2) |
| Certain cells, when the entity passes through, CHANGE its rendered shape/color | transformer interaction (arch 2 mutation probe) |
| There is a target whose shape+color the entity's must equal | target tuple (arch 2 target probe) |
| Advance fires only when entity shape AND color equal the target | win = attribute equality (arch 2 win probe) |
| One HUD bar drops every action and refills at certain cells | budget + refill (arch 4-A) |
| A separate counter drops on bad contact and ends the episode at 0 | lives (arch 4-B) |

**The answer key (held OUTSIDE the agent — for Stage 5 grading only).** User hand-decode of ls20: carried object must match goal shape AND color; white intermediate cells are transformers (shape/color toward target); win on match; yellow bar = step budget, yellow squares refill it; red segments = lives, game-over at zero. **This decode is NOT encoded anywhere in the agent.** At Stage 5 we check whether the agent's *within-episode learned model* independently arrives at equivalents of each row above. Recovery ⇒ the discovery engine works and should transfer to the 55 private scoring envs (no source, no replays, met cold).

---

## Build order for Part 2 Stage 4 (frequency × tractability)

1. **resource_management** — high freq (every env), high tract (read a HUD delta). Build the budget/lives delta-tracker FIRST: it sets the action ceiling for all discovery, so the agent never exhausts itself probing.
2. **movement_pathfinding** — high/high. Single-translation-delta probe is the cleanest discovery signal; BFS/A* over a probed occupancy map is robust. De-risks holdout tu93/dc22/sk48/lp85.
3. **selection_indirection + undo_reversibility** — cross-cutting tools; medium freq but high tract. Build early: they make all later discovery cheaper (free/cheap probing), and mis-modeling selection corrupts the movement probe itself.
4. **click_to_effect** — high/medium. The affordance-by-under-cursor-class map is the explicit fix for the BC spatial-head failure; build salience/class-based, never coordinate-based.
5. **attribute_matching** — high freq, medium tract (multi-axis, cycles, implicit transformers, rule-induction). Biggest payoff; needs single-axis-probe discipline + a small search solver. Build after the cheaper layers supply selection/movement/resource primitives.
6. **time_pressure_hazard** — medium/medium. Layer as a deadline constraint onto movement/attribute plans once those exist.
7. **physics_cascade_simulation** — low freq (3 envs), medium tract. Build last as a specialized arrange-then-release forward-simulator; fewest envs, depends on a working movement/click base.

---

## Critic verdict (independent adversarial agent — relaunched after workflow tail failed)

The workflow's critic agent failed on structured output, so it was relaunched standalone against the committed doc. It found **one P0 my inline self-check missed** + three exploit-path gaps, all now fixed in this revision.

| check | initial | after revision |
|---|---|---|
| Coverage — all 20 train envs assigned | **FAIL** — r11l absent from every exemplar list (its reader's `env` field was a description, not "r11l", so it dropped out of the doc) | **PASS** — r11l added to click_to_effect (primary) + attribute_matching (secondary) |
| Appearance-leak | PASS (minor: `available_actions` is action-space, not appearance) | **PASS** — clarified in click_to_effect detection |
| Holdout-leak | **PASS** (only 20 train sources read; holdout provisional) | **PASS** |
| ls20-encode-risk | **PASS** (generic; answer-key held outside) | **PASS** |
| Missing archetype | **FAIL (P1)** — r11l place-to-satisfy-constraint exploit uncovered | **PASS** — place-to-constraint sub-path added |
| Build-order sanity | PASS | **PASS** |
| Discovery-feasibility (per archetype, observable from frame+levels only) | 6/8 clean; **P1** tr87 rule-induction (7^L blowup) + ft09 constraint read underdocumented; **P2** ar25 non-spatial transformer | **PASS** — tr87 per-position anti-blowup, ft09 static constraint read, ar25 non-spatial transformer all added to attribute_matching |

**Revisions applied (P0→P2):** r11l coverage; click `available_actions` clarification + place-to-constraint sub-path; attribute_matching non-spatial transformer probe; ft09 constraint-template static read; tr87 per-position rule-induction anti-blowup.

**Overall after revision: PASS.** Experiment-grounded, discover-then-exploit, no per-env hardcoding, holdout-clean, all 20 train envs covered, every experiment signal observable from (64×64 frame stack + levels_completed) alone. One residual flag carried into Part 2: tr87-class rule-induction may still need a dedicated sub-module if a per-position win signal proves unavailable — to be confirmed empirically, not assumed.

---

## Review gate

**STOP for review before any agent code (Part 2).** On approval: enter plan mode, surface the Stage 1–4 build plan (analyser → experiment-and-remember loop → decision rule → exploit strategies in the order above), unit-test segmentation + action→effect model in isolation first, then Stage 5 holdout eval with the ls20 rule-recovery comparison.
