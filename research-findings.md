# ARC-AGI-3 — Phase 0a Research Findings

**Author:** Phase 0a research pass for ARC Prize 2026 / ARC-AGI-3 Kaggle competition
**Date:** 2026-05-26 (T-35 days from Milestone #1 on 2026-06-30)
**Scope:** Read-only research and dataset audit. No agent code. No architecture commitments.
**Verification status:** Every claim cited inline. Anything unverifiable from public sources is flagged `[VALIDATE-0b]` for hands-on confirmation in Phase 0b.

---

## 1. Competition Mechanics

### 1.1 The benchmark in one paragraph

ARC-AGI-3 evaluates an agent's ability to play **novel, turn-based 2-D grid environments without instructions**. The agent receives a 64×64 frame of 16 colors and must choose one of at most seven actions. Each environment is a sequence of levels with a hidden win condition that the agent must infer from interaction alone. The benchmark targets four functional pillars — exploration, modeling, goal-setting, planning — and scores by **action efficiency relative to the upper-median first-run human**. [ARC-AGI-3 Technical Report §2.1–2.2](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

### 1.2 Dataset composition

| Subset | Purpose | # Environments |
|---|---|---|
| Public Demo | Format demonstration, public playable | **25** |
| Semi-Private | Frontier-model evaluation (API access partners) | **55** |
| Fully Private | **Official ARC Prize / Kaggle competition target** | **55** |

Total = 135 environments, not the ~150 we had assumed. Public/private ratio is *inverted* vs ARC-AGI-2 — public set is a demo, not a training resource. [Tech Report Table 1, p.11](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf). All replays we possess (342, public) correspond only to the 25-env demo set; the 110 private envs are unseen by every team. [arcprize.org blog: human dataset](https://arcprize.org/blog/arc-agi-3-human-dataset). **Phase 0b clarification (Kaggle data-description page): a single Kaggle submission runs against all 110 private envs; 55 score the Public Leaderboard (visible during competition, the 55 semi-private envs); the other 55 score the Private Leaderboard (revealed at competition end, the 55 fully-private envs — these determine final prize ranking).**

### 1.3 Action space

7 possible actions; each env declares its own subset via `available_actions`. [docs.arcprize.org/actions.md](https://docs.arcprize.org/actions.md)

| ID | Semantics | Data | Keybind |
|---|---|---|---|
| `RESET` | Restart current level / game | — | — |
| `ACTION1` | "Simple action (semantically up)" — game-specific meaning | — | W / ↑ |
| `ACTION2` | "Simple action (semantically down)" | — | S / ↓ |
| `ACTION3` | "Simple action (semantically left)" | — | A / ← |
| `ACTION4` | "Simple action (semantically right)" | — | D / → |
| `ACTION5` | "Simple action — interact / select / rotate / attach / execute" | — | Space / F |
| `ACTION6` | **Click on cell, x,y ∈ 0..63** | `{"x": int, "y": int}` | Mouse click |
| `ACTION7` | **Undo previous action** | — | Ctrl+Z |

Confirmed in the live console (screenshot from user shows SPACEBAR / CLICK / RESET / HELP / UNDO / SELECT buttons; SELECT is not a distinct action in the SDK enum — likely a UI mode toggle for ACTION6 [VALIDATE-0b]).

### 1.4 Observation format

Per the tech report and confirmed by local smoke-test: each cell is one of **16 colors (0–15)** on a **64×64 grid**. The on-wire frame field is a list-of-list-of-list, shape `(T, 64, 64)` where **T ≥ 1**. T > 1 represents a non-interactive **transition animation** that the engine emits between turns. [Tech Report §2.3.1, p.6](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

Empirical evidence from the 340 public replays: 71% of returned frames are `(1, 64, 64)`, the rest are stacks up to **`(404, 64, 64)`** in extreme cases (`env=lp85`). Animation stacks ≥10 frames occur in ~10% of frame events. Implication: the perception module must consume variable-length frame stacks. Picking only the last frame is a reasonable simplification but discards information about velocity / object motion that some envs probably encode through animation (e.g. moving collectables observed in the user's hand-play notes for late levels of certain envs).

### 1.5 The RHAE scoring formula (exact)

Let `a_{l,e}` = actions the agent took to clear level `l` of env `e`. Let `h_{l,e}` = upper-median best human action count for that level (the human baseline). Let `n` = total levels in the env. Let `k` = number of levels the agent completed (sequentially — completing level `k` means completing 1..k).

**Per-level score:**

```
S_{l,e} = min(1.15, (h_{l,e} / a_{l,e})^2)
```

**Per-environment score:**

```
E_e = min(  Σ_{l=1..k} l  /  Σ_{l=1..n} l   ,    Σ_{l=1..n} l · S_{l,e}  /  Σ_{l=1..n} l   )
```

The first term is the **completion cap**: weighted fraction of levels completed. Levels weight `w_l = l` (1-indexed).

**Total score:**

```
T = (1 / |D|) · Σ_{e ∈ D} E_e
```

Worked example (Tech Report §4.1, p.11): "If upper-median best human completed a level in 10 actions, but AI took 100 to complete it, then AI scores `(10/100)^2 = 1%` for that level." [Tech Report Eq. 1–3, p.11–12](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

### 1.6 The per-environment cap (consequence)

5-level env example from the report:
- Complete all 5 → cap 15/15 = 100%
- Complete 4/5 → cap 10/15 ≈ 66.7%
- Complete 3/5 → cap 6/15 = 40%
- Complete 2/5 → cap 3/15 = 20%
- Complete 1/5 → cap 1/15 ≈ 6.7%

**Strategic consequence:** acing levels 1–2 with super-human efficiency is bounded above by ~20% of the env score regardless of how few actions you used. **Depth (completing late levels) dominates efficiency**, which is the opposite shape from what raw squared-ratio scoring suggests at first glance. The benchmark explicitly punishes "skim the easy stuff" strategies.

### 1.7 The 5× action cutoff

> "We impose an action budget of five times the human-baseline median action count per level. That is, for a level with a human median of n actions to completion, the agent is terminated after 5n actions." — [Tech Report §4.3, p.14](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

This is a **compute-control rule, not a scoring rule.** The agent is hard-terminated at 5n; uncompleted levels score 0. Combined with the squared scoring (5× over baseline → 4% of max even without cutoff), the cutoff is mostly a wall-clock saver — but a critical one for the 9-hour Kaggle ceiling (was 12h in Phase 0a; corrected in Phase 0b — see validation-results.md §1 Item 1).

### 1.8 Baseline = upper-median, not 2nd-best

The human baseline is the **upper-median best first-run human action count**. 10 humans are tested per env; rank them by action count on each level, take the upper-median (3rd or 4th place). [Tech Report §4.2 and §5, p.13–17](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

Changelog confirms a methodology shift on 2026-04-14: baseline moved from "2nd-best human" to "median human per level" (the per-level upper-median formulation in the tech report), and the per-level cap rose from 1.0× to 1.15×. [docs.arcprize.org/changelog.md](https://docs.arcprize.org/changelog.md). Any older write-up describing 2nd-best baseline or 1.0× cap is **stale**.

### 1.9 Baselines per env are shipped with the dataset

Each `environment_files/<env>/<hash>/metadata.json` already contains `baseline_actions: [n_1, n_2, ..., n_L]` — the human baseline per level. Sample:

```
sp80 baseline_actions = [39, 58, 25, 148, 96, 152]   # 6 levels
wa30 baseline_actions = [71, 119, 183, 98, 368, 68, 79, 442, 415]   # 9 levels
vc33 baseline_actions = [7, 18, 44, 61, 131, 34, 152]   # 7 levels
ka59 (probe via Arcade.get_environments())
```

Verified via local `Arcade.get_environments()` smoke test. These baselines are exposed even in offline mode and can be used at training time for hard-negative curriculum and at inference time for early-give-up logic. The level counts vary (6 to 9+ in the public set).

### 1.10 Kaggle competition rules — user-supplied, partial verification

The Kaggle page is client-rendered and `WebFetch` returns only the page title; the rules tab is not scrapeable without authentication. **Treat the following user-supplied numbers as the operating assumption; final confirmation by browsing the Kaggle rules tab is a Phase 0b task.**

User-supplied:
- **Milestone #1 deadline: 2026-06-30** (~5 weeks from today).
- **Prizes at Milestone #1:** $25K / $10K / $2.5K (1st / 2nd / 3rd).
- **Submission quota: 1 per day per team.** ~35 total submissions through Milestone #1.
- **Runtime ceiling: 9 hours wall-clock per submission** (corrected from "12 hours" in Phase 0b — Kaggle Code Requirements page; was 6h pre-2026-05-07, raised to 9h after RTX-6000 swap).
- **Hardware: Kaggle RTX 6000 (48 GB VRAM), internet disabled.**
- **Open-source license required: CC0 or MIT-0 for prize eligibility.**

Cross-confirmed by the tech report: "Both competitions are held on Kaggle. As always, participants must open source their solutions in order to receive prize money." Total prize pool 2026 = $2M across ARC-AGI-3 + ARC-AGI-2 tracks. [Tech Report §7, p.21](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

[VALIDATE-0b] — pull the exact Kaggle rules text in Phase 0b once the Kaggle MCP is enabled or via authenticated browser.

### 1.11 Competition mode constraints

The Kaggle competition is forced into `OperationMode.COMPETITION` by the scoring harness. [docs.arcprize.org/toolkit/competition_mode.md](https://docs.arcprize.org/toolkit/competition_mode.md). Constraints:

- **One `make()` call per environment per submission.** No re-entries. No multi-restart per env.
- **Scoring is against the full eval set** even if the agent skips some envs. Skipped env contributes 0 to `T`.
- Only level resets are honored; game resets convert to level resets.
- One scorecard per session; in-flight scorecard queries blocked.

**Implication for design**: every env is one shot. You cannot do n-rollout exploration, pick the best, then re-run. The agent must commit to one trajectory per env. Sample-efficient online adaptation is mandatory.

---

## 2. SDK & Environment Audit

### 2.1 Install (verified locally, 2026-05-26)

The Kaggle dataset bundles 30 wheels in `arc-prize-2026-arc-agi-3/arc_agi_wheels/` (Python 3.12, manylinux2014_x86_64). For local Windows dev, install from PyPI:

```powershell
# create venv on Python 3.13 (3.12+ required by arc-agi pyproject)
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install arc-agi arcengine
```

Verified versions: `arc-agi 0.9.8`, `arcengine 0.9.3`. These match the bundled wheel versions, so submission-time and local-dev behavior should match. Local Windows install pulls Windows-native binaries for numpy/pillow/pydantic-core; Kaggle uses the bundled Linux wheels.

### 2.2 Toolkit shape

```python
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, FrameDataRaw

arc = Arcade(
    operation_mode=OperationMode.OFFLINE,    # or COMPETITION on Kaggle
    environments_dir=r"path/to/environment_files",
)
games = arc.get_environments()              # list[EnvironmentInfo]
card_id = arc.open_scorecard(tags=["..."])
env = arc.make("sp80", scorecard_id=card_id)
obs = env.reset()                            # FrameDataRaw
obs = env.step(GameAction.ACTION1, data={}) # action 1..5/7: data={}
obs = env.step(GameAction.ACTION6, data={"x":32,"y":32})
scorecard = arc.close_scorecard(card_id)
```

`OperationMode` enum members (verified by `list(OperationMode)`): `NORMAL`, `ONLINE`, `OFFLINE`, `COMPETITION`. [docs.arcprize.org/toolkit/arc_agi.md](https://docs.arcprize.org/toolkit/arc_agi.md)

### 2.3 FrameDataRaw fields (verified)

```
game_id          : str       # "sp80-589a99af"
frame            : list[ndarray]  # length T, each (64,64) int — T can be 1..400+
state            : GameState  # NOT_FINISHED | WIN | GAME_OVER | …
levels_completed : int       # was "score" pre-0.9.3 — confirmed via README changelog
win_levels       : int       # total levels in this env
guid             : str
full_reset       : bool
available_actions: list[int] # which 1..7 are legal this turn
```

Confirmed in `ARC-AGI-3-Agents/agents/agent.py:142–155` and via smoke test (`sp80.reset()` returned `state=GameState.NOT_FINISHED, levels_completed=0, win_levels=6, available_actions=[1,2,3,4,5,6]`). [ARC-AGI-3-Agents changelog](https://github.com/arcprize/ARC-AGI-3-Agents/blob/main/README.md)

### 2.4 Local execution speed

Local OFFLINE mode is fast. Smoke-test measurement on Windows Python 3.13: **~5,860 ACTION1 calls/second** against sp80 single-threaded. Docs claim ~2,000 FPS as the engine's design target. [docs.arcprize.org/local-vs-online.md](https://docs.arcprize.org/local-vs-online.md)

Implication: **self-play and replay-driven training are unconstrained.** A million-step training run for a single env takes ~3 minutes engine-side. The training compute bottleneck will be the model, not the environment.

### 2.5 Frame-change detection

Not a first-class SDK feature; the SDK does not provide a "did the frame change?" boolean. The agent has to diff frames itself. StochasticGoose's preview solution was built around predicting frame-change probability per action — an explicit learned signal, not a free observation. See §5.1.

### 2.6 Episode / level / environment / scorecard boundaries

- **Episode**: a single call sequence from `env.reset()` to terminal `state` (WIN or GAME_OVER).
- **Level**: one `levels_completed` increment. The env stays in the same `env` instance.
- **Environment**: a `make()`-ed game. In COMPETITION mode, **one make() per env, period.**
- **Scorecard**: open → run all envs → close. **Two independent TTL layers — do not confuse them.**
  - **Local SDK defaults** (verified in `arc_agi/scorecard.py:24–25`): `DEFAULT_STALE_MINUTES = 15`, `DEFAULT_MAX_OPEN_FOR_MINUTES = 4320` (3 days). Smoke-test log line `idle_for=0:15:00 and max_open_for=3 days, 0:00:00` matches. Overridable via env vars `STALE_MINUTES` and `MAX_OPEN_FOR_MINUTES`.
  - **Server-side ONLINE policy**: the 2026-04-14 changelog's "Maximum scorecard open duration is now capped at 24 hours" applies to the online API, not the local toolkit defaults. The local toolkit's 3-day default exceeds the 24h server cap — irrelevant in OFFLINE / COMPETITION since the toolkit, not the server, is authoritative locally.
  - **Kaggle COMPETITION-mode effective TTL** still [VALIDATE-0b — Kaggle parity notebook]. The 9-hour notebook runtime ceiling is the binding cap regardless.

### 2.7 Reset behavior

`env.reset()` returns to level 1 in OFFLINE mode but **in COMPETITION mode a full game reset is silently converted to a level reset**, per competition_mode docs. So in submission, calling RESET on level 3 puts you back at level 3 start, not env start. Level resets do still cost an action [VALIDATE-0b].

### 2.8 Scorecard output

The closed scorecard is an `EnvironmentScorecard` Pydantic object. Per smoke test, structure is:

```
score                : float    # the final RHAE total
environments[].id    : str
environments[].runs[].guid
environments[].runs[].score
environments[].runs[].levels_completed
environments[].runs[].actions
environments[].runs[].resets
environments[].runs[].state
environments[].runs[].completed       : bool
environments[].runs[].level_scores    : list[float]   # one per level
environments[].runs[].level_actions   : list[int]
environments[].runs[].level_baseline_actions : list[int]
```

`level_baseline_actions` is the upper-median human baseline per level — **the same values shipped in `metadata.json`'s `baseline_actions`**. This is the ground-truth h used in RHAE. The scorecard is fully available offline (contrary to what `local-vs-online.md` implies — that page seems to confuse "no shareable URL" with "no scorecard at all"; the smoke test produced a valid scorecard offline).

### 2.9 Rate limits (online mode only)

600 req/min. 429 with exponential backoff. Irrelevant for Kaggle no-internet runs. [docs.arcprize.org/rate_limits.md](https://docs.arcprize.org/rate_limits.md)

### 2.10 Environment source code is in the bundle

`arc-prize-2026-arc-agi-3/environment_files/<env>/<hash>/<env>.py` is the **full Python source** of each public-set environment. Names are deliberately obfuscated (`bodekplurlf16`, `mfkgvxzkbj`, etc.) to prevent model trivially memorizing semantic identifiers. The source defines sprites, levels, mechanics, win conditions, action remapping (e.g. sp80's level rotation flips ACTION1↔ACTION4), and step-budget per level.

**This source is for the 25 public envs only.** The 110 private envs (55 semi-private + 55 fully-private) are unseen and not bundled. **However**, reading the public-env source teaches us the env-engine vocabulary the studio uses — sprite tags like `sys_click`, `liolfvkveqg` (object marker), `tuvkdkhdokr` (directional), `repwkzbkhxl`, etc. — and the categorical mechanics that appear across the set. Patterns extracted now will generalize to the private set because the private set uses the **same engine and design vocabulary**, by tech-report §3.3 ("All ARC-AGI-3 environments were implemented in a shared runtime using a custom in-house environment engine"). [VALIDATE-0b: do an audit pass on all 25 env .py files to inventory mechanics.]

---

## 3. Hand-Play Observations (from user interview)

Recorded 2026-05-26. Verbatim user answers, restructured for the doc.

### 3.1 Envs played

User has played from the 25-env public demo set. Replays of all 25 are present in `data/human_replays/`. Specific envs the user referenced: `sp80`, `ls20`. The user reports all played envs were eventually solved during hand-play.

### 3.2 ACTION1–5 semantics

**Consistent across envs.** ACTION1–4 are directional (the up/down/left/right mapping holds), ACTION5 is an interact / commit action. Confirmed by env source: `sp80.py` uses ACTION1/2/3/4 as up/down/left/right (with per-level rotation remap via `mfkgvxzkbj`), and ACTION5 triggers the "spill" commit. This stability across envs is a strong prior for the policy network.

### 3.3 ACTION6 (click)

Coordinate-based. The click targets a specific (x,y) cell on the 64×64 grid. User reports: in envs like sp80, clicking on an object **changes the object's color** (i.e. selects-then-edits, not select-then-move). Confirmed by `sp80.py:657–672`: ACTION6 selects a sprite under the click, paints it color 9, and enables movement via ACTION1–4 (so the actual flow is **select → move → commit-with-ACTION5**, with ACTION6 as the select step). Per-env behavior of ACTION6 varies wildly — see action histograms in §4.

### 3.4 ACTION7 (undo)

Present only in select envs. Of 25 public envs, only **5 expose ACTION7** in their replays' action streams: `bp35` (179 undo events), `sk48` (135), `lf52` (51), `su15` (47), `sb26` (6). The rest never use undo — either because the env doesn't expose it, or because it isn't a useful tool given the mechanic. ACTION7 is rare overall (0.26% of human actions; see §4.2).

### 3.5 Recurring mechanics

User reports each env has its own consistent mechanics — within one env, the same rules apply, but mechanics differ env-to-env. The tech report (§3.4) confirms: each environment has multiple mechanics; "environments centered on a single mechanic that scaled in size or difficulty are treated as an anti-pattern." Each env therefore composes several Core-Knowledge primitives (objectness, geometry, basic physics, agentness). [Tech Report §3.4 p.8]

### 3.6 Level progression

> "New mechanics or same additional mechanics introduced mid-env (sometimes both)."

Matches tech report §3.4 design rule: "Difficulty through composition. Later levels are expected to require the accumulation and integration of concepts learned earlier." Level 1 is the **tutorial level** (intentionally easy, random agents may stumble onto a solution). Level k for k > 1 typically adds a new mechanic on top of those already needed. This is exactly why the level-weighted-by-index scoring exists: later levels test composition, earlier levels test single-concept acquisition.

### 3.7 Failure / reset triggers

Env-specific. Example given: `ls20` is step-limit gated. Sp80 (per source) has multiple lose conditions: step-budget exhaustion, spill collision with terminal, and the "4-spill" counter. Tech report §4 confirms the abstract loss conditions are env-defined; there is no universal "death" semantics. The agent must learn each env's failure modes from interaction.

### 3.8 The surprise — limited visibility + moving collectables

> "For some env's final level/stage — limited visibility (just around the player/main-icon) and objects/collectables also have moment (start moving) mid-env."

This is critical and is **not obvious from the action enum or the early levels**. Late levels can:
- Restrict the agent's observable region (fog-of-war around the player avatar)
- Introduce autonomous moving objects (state changes between agent actions, despite the "turn-based" framing — these are the transition animation frames)

Implications:
- The world model cannot assume the full 64×64 grid is informative content. Some envs gate observation by visibility.
- Animations between turns (the variable-T frame stack from §1.4) are not decorative — they encode object velocity and trajectory that the agent needs.
- "Turn-based" is a misnomer for the late-game; it's "the agent chooses one action per turn, but the world may advance several frames between turns."

### 3.9 The Help button

The in-game console exposes a Help button. The user's screenshot shows it produces a generic "Available Controls: arrows / Space / Click" panel that does **not** reveal env mechanics. It's a controls reminder, not a hint system. The SDK does not expose a "help" action — agents have no equivalent affordance. [VALIDATE-0b: confirm by inspecting `env.observation_space` for any help-related fields.]

### 3.10 Console UI vs SDK action set

Buttons visible to the human player on the console: SPACEBAR, CLICK, RESET, HELP, UNDO, SELECT. Greyed when not legal for the current level. Map to SDK:

| Console | SDK |
|---|---|
| Arrow keys | ACTION1 / ACTION2 / ACTION3 / ACTION4 |
| SPACEBAR | ACTION5 |
| CLICK | ACTION6 (with x,y) |
| UNDO | ACTION7 |
| RESET | RESET |
| HELP | not in SDK |
| SELECT | not in SDK — probably a UI mode toggle for ACTION6 [VALIDATE-0b] |

---

## 4. Human Replays Dataset Deep-Dive

### 4.1 What we have

`data/human_replays/arc_agi_3_public_demo_human_testing/public_games-dataset/<env>/*.recording.jsonl` + `testing_feedback_ratings.csv`. 25 env folders, **340 replays** (close to the 342 advertised in [arcprize.org blog](https://arcprize.org/blog/arc-agi-3-human-dataset); the 2 missing are likely deduplicated rejects).

Aggregated stats (computed via `.tmp_replay_stats.py`, summary in `.tmp_replay_stats.json`):

- Total records (frame events): **180,836**
- Total replays: 340
- Mean records/replay: ~532
- Replays terminating in `state=WIN`: **144 / 340 = 42.4%**
- Replays terminating in `state=GAME_OVER`: 836 records but mostly mid-replay GAME_OVER (level fails reset)

**Solved-replay rate per env (WIN endings):** ranges from 14% (bp35: 2/14) to 100% (r11l: 10/10). Median ~46%. Envs with very low solve rate (bp35, sp80, g50t, su15) are *harder for humans*; envs with high solve rate (r11l, cd82, tu93) are easier. Note: even harder envs have at least 2 winning replays each — every env has at least one full-solution trajectory to learn from.

### 4.2 Action distribution

Global action histogram across 180k records (numeric IDs and string IDs co-exist in the dataset — likely two different recorder versions; merging them):

| Action | Count | % of human actions |
|---|---:|---:|
| ACTION6 (click) | 53,876 + 2,471 = **56,347** | 34.0% |
| ACTION4 (right) | 30,729 + 2,578 = 33,307 | 20.1% |
| ACTION3 (left) | 28,301 + 2,776 = 31,077 | 18.7% |
| ACTION1 (up) | 23,932 + 2,893 = 26,825 | 16.2% |
| ACTION2 (down) | 21,699 + 2,615 = 24,314 | 14.7% |
| ACTION5 (interact) | 4,924 + 1,210 = 6,134 | 3.7% |
| ACTION7 (undo) | 418 | 0.25% |
| RESET (id=0 / "RESET") | 1,980 + 94 = 2,074 | 1.25% |

(Percentages of non-RESET non-id-0 actions; totals normalized to ~165,000 game-step actions.)

ACTION6 dominates. Movement is symmetric across the four arrows (skewed slightly toward right/left — consistent with reading-direction priors of human players). ACTION5 is rare. ACTION7 is essentially negligible — humans don't backtrack often, even when they could. Implication: a learned policy that allocates capacity uniformly across actions wastes capacity on ACTION7. **Drop ACTION7 from BC training data when the env doesn't expose it.**

### 4.3 Env families by action profile

Three clusters emerge:

**Pure-click envs (only ACTION6 + RESET):** `bp35`, `lp85`, `r11l`, `s5i5`, `sb26`, `su15`, `tn36`, `vc33`. ~32% of envs. These are pure spatial-reasoning puzzles where the human only clicks cells.

**Pure-movement envs (no ACTION6):** `cn04`, `g50t`, `ls20`, `re86`, `tr87`, `tu93`, `wa30`. ~28% of envs. These play like keyboard-controlled arcade games.

**Mixed envs (ACTION6 + arrows + interact):** `ar25`, `cd82`, `dc22`, `ft09`, `ka59`, `lf52`, `m0r0`, `sc25`, `sk48`, `sp80`. ~40%. These are the most complex — select-then-act paradigms.

**Implication:** the policy can branch by env action signature at the *start* of an env (the first frame's `available_actions` is the discriminator). A click-only env doesn't need the directional head at all; a movement-only env doesn't need a coordinate decoder. This is a free architecture-level prior.

### 4.4 Replay length distribution per env

| Env | n_replays | median frames | min | max | wins | tags (from EnvInfo) |
|---|---:|---:|---:|---:|---:|---|
| ar25 | 10 | 861 | 61 | 1623 | 5 | mixed |
| bp35 | 14 | 599 | 3 | 794 | 2 | click+undo |
| cd82 | 11 | 192 | 23 | 356 | 8 | mixed |
| cn04 | 12 | 620 | 2 | 1522 | 6 | movement+5+6 |
| dc22 | 11 | 600 | 103 | 1429 | 4 | mixed |
| ft09 | 10 | 122 | 15 | 386 | 4 | click-dominant |
| g50t | 12 | 475 | 154 | 1002 | 2 | movement+5 |
| ka59 | 10 | 290 | 35 | 906 | 3 | mixed |
| lf52 | 11 | 1213 | 69 | 1814 | 4 | mixed+undo |
| lp85 | **54** | 227 | 3 | 1312 | 15 | pure click |
| ls20 | 13 | 595 | 22 | 1171 | 6 | pure movement |
| m0r0 | 11 | 995 | 186 | 1608 | 7 | mixed |
| r11l | 10 | 239 | 154 | 694 | 10 | pure click |
| re86 | 11 | 1073 | 348 | 2282 | 5 | movement+5 |
| s5i5 | 11 | 610 | 43 | 1470 | 4 | pure click |
| sb26 | 12 | 183 | 48 | 436 | 5 | click+undo |
| sc25 | 15 | 356 | 23 | 1290 | 10 | mixed |
| sk48 | 14 | 879 | 166 | 2952 | 7 | mixed+undo |
| sp80 | 12 | 272 | 33 | 1163 | 2 | mixed |
| su15 | 13 | 218 | 30 | 653 | 3 | click+undo |
| tn36 | 14 | 300 | 11 | 508 | 6 | pure click |
| tr87 | 12 | 444 | 130 | 690 | 6 | pure movement |
| tu93 | 13 | 441 | 119 | 670 | 9 | pure movement |
| vc33 | 10 | 432 | 10 | 1036 | 6 | pure click |
| wa30 | 14 | 1592 | 91 | 2293 | 5 | movement+5 |

`lp85` has 54 replays (an outlier — probably collected more aggressively because it's the canonical tutorial/early env). Total examples per env are otherwise 10–15 — small for a deep BC pretrain on a single env, but **adequate when shared backbone is pretrained across all 25 and only the head specializes**.

### 4.5 Frame-data shape — animations matter

Distribution of frame-stack length T over 180k frame events:

- T=1 → 128,115 (70.9%)
- T=2 → 16,823 (9.3%)
- T=3 → 8,389 (4.6%)
- T=5 → 5,999 (3.3%)
- T=7 → 5,093 (2.8%)
- T=10 → 2,791 (1.5%)
- T=13 → 4,286 (2.4%)
- T=15 → 1,008 (0.6%)
- Long tails: 100+ frame stacks appear in several envs; **maximum observed T = 404** (in `lp85`).

The 29% of frames with T>1 carry the env's hidden dynamics in the animation. Two strategies for the perception module:
1. Take last frame only (cheap, throws away motion info)
2. Reduce stack via min/max/diff across T (cheap, captures motion masks)
3. Treat as temporal sequence (expensive but most informative)

A pragmatic middle ground: pass `(first, last, diff)` triplets through the CNN — keeps the input shape fixed while preserving motion signal.

### 4.6 BC feasibility on 12 GB VRAM

A ResNet-18-equivalent CNN over 64×64×16-channel one-hot input has ~11M parameters. At batch size 128, mixed-precision, gradient checkpointing optional, **memory footprint is ~3 GB**. Comfortably fits the 12 GB local card with room for replay buffers, optimizer state, and a forward replica.

Training data math:
- 180k frame events, ~165k labeled with an action.
- Filter to **WIN-terminated replays only** (144 replays, ~50–60k action transitions) for clean BC targets, OR keep all replays for behavior dataset (human-quality but not all reach the goal).
- One epoch through 50k transitions on a 11M-param net = ~3 minutes on an RTX 5070 Ti.
- Full training run (50 epochs, BC + auxiliary heads) ≈ 2.5 hours. Easily iterable.

Larger backbones (ResNet-50, ViT-Tiny) also fit. **Compute is not the constraint** for the perception+policy module; the *idea* is the constraint.

### 4.7 What BC alone gets you (and what it doesn't)

BC gets you:
- **Action priors per env** (humans don't undo, click envs need click bias, movement envs need movement bias).
- **Tutorial-level behavior** that beats random with near-certainty.
- **Style of frame-change-eliciting actions** — humans rarely waste moves, so their action distributions are informative even without reward.

BC alone does not give you:
- **Novel-env generalization.** The 110 private envs (55 semi-private + 55 fully-private, both unseen) share the engine but not the specific mechanics. A BC policy trained on 25 envs and dropped on 110 unseen ones will fall back to learned priors but cannot deduce a novel mechanic from one rollout.
- **Online adaptation under one-shot Kaggle constraints.** No way to fine-tune on a private env mid-submission.

### 4.8 BC + online adaptation: what's feasible

Promising directions (none implemented in Phase 0a; just enumerated):

| Approach | VRAM | Train | Inference | Pros | Cons |
|---|---|---|---|---|---|
| BC + auxiliary frame-change head | <4 GB | hours | <5 ms | Cheap, leverages human style + StochasticGoose's preview insight as a learned signal | Doesn't model long-horizon strategy |
| BC + replay-conditioned policy (concat replay tokens) | <6 GB | days | tens of ms | Generalizes via in-context retrieval of similar past situations | Needs nearest-replay retrieval at inference; retrieval index must fit Kaggle disk |
| BC pretrain → online sample-efficient RL (PPO / SAC / IMPALA) | 4–8 GB | days locally, then per-env online updates capped by Kaggle 9 h | <10 ms | Best per-env scores if online updates land | One-shot competition mode forbids cross-rollout RL; only intra-rollout signal is usable |
| Offline RL on replays (CQL / IQL / Decision Transformer) | 6–10 GB | days | <20 ms | Learns from sub-optimal trajectories (the 58% non-WIN replays still encode mechanics) | Generalization to private envs untested in this domain; Decision Transformer needs return tokens we'd have to construct |
| Replay retrieval + slot-attention world model | 8–12 GB | days–weeks | tens of ms | Strongest in-context novelty handling | Most engineering, highest risk in 5 weeks |

12 GB is the binding local constraint. Kaggle has 48 GB but no debug loop. **Anything that needs more than 12 GB to debug locally is effectively un-debuggable** → submission risk balloons.

### 4.9 What this dataset enables that StochasticGoose did NOT have

StochasticGoose competed in the 2025 preview with **no human-replay data** — the human dataset was published 2026-04. Their approach reset the model and buffer between levels and used pure online exploration with a learned frame-change-prediction head. With 340 human replays:

- We can **pretrain the perception backbone** on real human play, not random play.
- We can **train an action prior** that beats random on level 1 immediately.
- We can **measure agent vs human action-by-action divergence** as a training signal.
- We can **cluster envs by action profile** (§4.3) and structure the policy accordingly.

This is the single biggest asymmetry vs prior leaders. **The human dataset is the highest-leverage asset we have access to.**

---

## 5. Prior-Art Failure Analysis (read everything, copy nothing)

### 5.1 StochasticGoose — preview winner, launch collapse

**Preview result (2025-08):** 12.58%, 18 levels completed across 3 hidden eval envs. First place. [arcprize.org/blog/arc-agi-3-preview-30-day-learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)

**Launch result (2026):** collapsed to **0.25%**, recovered to 1.17%. Still #1 on the Kaggle leaderboard (per user-supplied info; [VALIDATE-0b]).

**Approach (from the Medium writeup & GitHub README):** 4-layer CNN backbone (32→256 channels), two heads — one for ACTION1–5 type, one for ACTION6 (x,y) coordinates with conv layers preserving 2D bias. Supervised learning on observed `(state, action) → frame_change` pairs (binary label: did this action change the frame?). 200K-entry experience buffer with hash-based deduplication. **Model + buffer reset between levels.** Exploration biased by predicted frame-change probability. [medium.com/@dries.epos](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db) | [github.com/DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution)

**Failure mode — what assumption broke:**

> "Both winning approaches used an informed search approach, exploring as much of the action space of the environment as possible in the hope of encountering a winning combination by chance." — [Tech Report §6.1, p.20](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

The launch set was deliberately hardened against this. The author had foreshadowed this: "future games will be hardened to reduce brute-force solutions." That foreshadowing came true. The validation pipeline (Tech Report §3.5.1) explicitly stress-tests envs against 1,000,000-step random play and rejects any env where random play solves ≥ 1 in 10,000 — the launch set is, by construction, beyond the reach of biased random search.

**Specific generalization break:** their "biased random exploration through frame-change-predicting CNN" assumes the path to a level's solution is short enough in *action-space* that biased exploration reaches it within ~100k steps. When mechanics require composition (e.g. select object → move → press button at right phase), the action-space search depth explodes. The CNN's frame-change predictor is a 1-step local signal; it gives no credit for "this action enables a useful action 5 turns from now." That's the gap.

**Lessons for us:**
1. **A 1-step learned signal is insufficient** for the launch set. We need either multi-step value estimates, learned world models, or replay-conditioned planning.
2. **Resetting between levels is wasteful.** Levels share an env's mechanics; transferring across levels of the *same* env should be a free win.
3. **Frame-change-prediction is still a useful auxiliary head** — humans almost never choose actions that do nothing (the 1980 "id=0" entries we see in replays are RESETs, not no-ops). Predicting frame-change should be a cheap auxiliary loss alongside the BC main loss.

### 5.2 Blind Squirrel — preview 2nd place

**Score:** 6.71%, 13 levels. [Tech Report §6.1] | [github.com/wd13ca/ARC-AGI-3-Agents](https://github.com/wd13ca/ARC-AGI-3-Agents)

**Approach:** directed state graph over observed frames. Each unique frame hash = node; each action = edge. Construct the graph by exploration, search for terminal states.

**Failure mode:** state-graph methods scale catastrophically when (a) the state space includes non-deterministic transitions (animation transitions count as state changes), (b) levels are not fully observable (the user's "limited visibility" surprise — late-level fog-of-war breaks the assumption that "two frames with the same hash are the same state"), and (c) animations between turns create many "near-equal" hashes that bloat the graph. Tech Report Figure 3 actually celebrates state-graph as a validation tool for the studio — but for an agent, the graph never finishes for the harder envs.

**Lesson:** explicit state graphs are not the path. **Implicit state via a learned representation** that abstracts over animation noise + partial observability is the path.

### 5.3 Frontier LLMs — all < 1% at release

[Tech Report Table 2, p.16](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf):
- Anthropic Opus 4.6 (Max): **0.50%**
- Google Gemini 3.1 Pro Preview: 0.40%
- OpenAI GPT 5.4 (High): 0.20%
- xAI Grok-4.20 (Beta 0309 Reasoning): 0.10%

These are *no-harness, same system prompt* runs on the **semi-private** set, so not directly comparable to Kaggle's fully-private set scores. But the pattern is clear.

**Failure mode:**
1. **Frames-as-tokens is wasteful.** A 64×64 grid is ~12 KB serialized as a comma list per frame; running 500 actions burns 6 MB of context just on observations.
2. **Latency.** A reasoning model at typical 30–100 tok/s on a per-action thought-block needs seconds to minutes per action. With 5n action budgets, levels take hours of API time.
3. **No persistent state between turns.** The LLM has to re-derive the env's mechanics every turn unless the harness carefully feeds back summaries.
4. **Action-format brittleness.** The Tech Report cites prior work where models output free-text actions that the harness then parses; format failures are common. (This is also why the "Read-Grep-Bash" academic agent — see §5.4 — leans on code execution: the model writes Python that produces actions, dodging the parsing problem.)

**Bimodal harness performance** is the most damning observation:

> "in a variant of environment TR87, Opus 4.6 scores 0.0% with no harness and 97.1% with the Duke harness, yet in environment BP35, Opus 4.6 scores 0.0% under both configurations. This is clear evidence that ... Specifically engineered harnesses are not a useful way to measure AGI progress, as their performance on seen environments does not translate to unseen environments, much less to novel domains." — [Tech Report §4.3.1, p.15](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

**Lessons:** (a) generic harnesses that work on a couple of public envs do not transfer; (b) handcrafting per-env strategies is the wrong unit of generalization. We need an architecture whose generalization unit is "mechanics in the engine's vocabulary," not "this specific env."

### 5.4 Academic / community harnesses

**Duke "Read-Grep-Bash" harness (Fox et al., 2026):** Wraps a reasoning model with Python code execution so the model can search and transform its action history. Solves all 3 public envs with near-human action counts. Failure: bimodal — extreme variance across unseen envs. The harness's value is **context management** (compressing history), not strategy. [Tech Report §6.2, p.20] | [blog.alexisfox.dev/arcagi3] (403 to WebFetch; community reference [VALIDATE-0b])

**Arcgentica (Symbolica AI):** Orchestrator-subagent architecture. Top-level orchestrator delegates to specialized subagents, gets back compressed text summaries. Same context-management value proposition. Solves all 3 public envs. [Tech Report §6.3] | [github.com/symbolica-ai/ARC-AGI-3-Agents]

**OpenClaw (ARC Prize Foundation, 2026-05-15):** 5.2% on the public set, $2,912 in API costs. Memory + code-execution tools wrapped around a frontier model. Demonstration baseline; not generalizable. [arcprize.org/leaderboard/community]

**Common pattern in these harnesses:** they fix the LLM's *context window* problem (good) but don't address the LLM's *latency* and *novelty-generalization* problems (bad). All three are compute-heavy at inference — $2,912 for OpenClaw to score 5.2% on the public set is illustrative of the per-action cost.

**Lessons:**
1. Context management is a real problem worth solving — but it's the *easy* problem, and it's already largely solved.
2. The hard problem is **generalization from public-engine mechanics to private-engine mechanics**, and no public attempt has cracked it yet.
3. Compute-heavy LLM harnesses are budget-incompatible with Kaggle's 9 h runtime + 48 GB VRAM no-API constraint. The Kaggle competition is specifically structured to favor self-contained, fast inference — that's our terrain.

### 5.5 Synthesis — what every prior attempt missed

| Approach | What it had right | What it missed |
|---|---|---|
| StochasticGoose | Per-action neural prior, frame-change as cheap signal | No multi-step value, reset between levels, brute-force-derived strategy |
| Blind Squirrel | Explicit state tracking | Graph blows up on stochastic / partial-obs envs |
| Frontier LLMs | Strong general reasoning | Slow, context-hungry, no online adaptation, format brittleness |
| Duke / Arcgentica | Context management | Per-public-env specialization, no novel-env generalization |

**The unfilled gap:** a system that (a) has fast per-action inference, (b) reuses skill across levels of the same env, (c) generalizes across envs by abstracting over the engine's mechanic vocabulary, and (d) has online adaptation within the one-shot per-env Kaggle constraint. **That's the target shape.**

---

## 6. Viable Approach Space (under the modular-system constraint)

All options below assume the architectural constraint: modular system with per-action latency budget ≤ ~200 ms on the Kaggle RTX 6000 (compute math: **110 envs × ~7.3 levels × ~5× of ~50-action human median ≈ 200,000 worst-case actions in 9 h → ~165 ms/action ceiling**; corrected in Phase 0b — was wrongly 55 envs × 12 h ≈ 0.52 s/action in Phase 0a). Comfortable target: **50–100 ms/action**.

### 6.1 Approach A — Hand-crafted perceptual primitives + small learned policy

**Sketch:** Hand-coded vision module extracts objects, agent position, walls, goals via connected-component analysis on the 64×64 grid. A small MLP / GRU policy consumes this structured representation and outputs actions.

**12 GB feasibility:** Trivial — the model is a few hundred KB.
**Train compute:** Minutes.
**Inference latency:** <5 ms.
**Expected ceiling:** Low. The vision module fails on novel sprites; the engine's deliberately-obfuscated semantics (sprite names like `bodekplurlf16`) hint that public-set sprite identities are not the right unit.

**Use as:** Baseline + sanity check + auxiliary signal for a learned model. Not a standalone submission strategy.

### 6.2 Approach B — BC-pretrained CNN policy + online RL fine-tune

**Sketch:** ResNet-style CNN backbone over 64×64×16 one-hot input, two heads (action-type, ACTION6-coords). BC pretrain on 340 human replays. At submission time, freeze backbone; allow heads to fine-tune online on within-env data via low-variance gradient updates (e.g. PPO with tiny rollouts).

**12 GB feasibility:** Easy. ~3 GB peak.
**Train compute:** Hours.
**Inference latency:** ~10 ms.
**Expected ceiling:** Moderate. Online RL within one-shot Kaggle is constrained — you only have one rollout per env, no replays for off-policy gradient.

**Risk:** the one-shot competition rule (§1.11) means online RL barely works. Most "online learning" here would have to be **bandit-style action prior refinement within a level**, not full RL.

### 6.3 Approach C — State-graph memory + learned controller

**Sketch:** Maintain a small graph of distinct frame hashes observed in the current env. Controller is a learned policy that consumes (current frame, recent graph summary, available_actions) and outputs an action. The graph is a memory aid, not a search structure.

**12 GB feasibility:** Easy.
**Train compute:** Hours.
**Inference latency:** ~20 ms with graph encoder.
**Expected ceiling:** Moderate. Blind Squirrel's failure (§5.2) was using the graph as the *search* substrate, not a memory aid. Inverting the role might work better.

**Risk:** animation transitions create graph noise; frame hashing under partial observability is fragile.

### 6.4 Approach D — Object-centric world model (slot-attention) feeding a planner

**Sketch:** Slot-attention or DINO-style encoder produces a set of object tokens per frame. A small Transformer world model predicts next-frame object tokens conditional on action. A planner (MCTS-lite or beam search over the learned model) selects actions.

**12 GB feasibility:** Tight. Slot-attention with 16 slots + small Transformer ≈ 8–10 GB at batch 32. Inference fits easily.
**Train compute:** Days (this is the hardest to train).
**Inference latency:** 50–200 ms depending on planner depth.
**Expected ceiling:** Highest among the options if it trains. **Object-centric representations are the academic state-of-the-art for this exact class of problem** (Spelke priors, Core Knowledge).

**Risk:** highest engineering risk in 5 weeks. Slot attention is finicky on grid inputs; planning over learned models is unstable.

**Milestone #2 candidate.** Approach D is the leading candidate for Milestone #2 (2026-11-02) — when the timeline allows for 8–12 weeks of model development. Phase 0c should not eliminate it from the long-term plan; it's only deprioritized for the 5-week sprint to Milestone #1.

### 6.5 Approach E — Hybrid: BC backbone + frame-change auxiliary head + replay-conditioned head + lightweight planner

**Sketch:**
- **Perception backbone**: ResNet-style CNN, pretrained on 340 replays via masked-frame reconstruction + BC.
- **Action head**: BC-trained categorical over 7 actions + spatial map over 64×64 for ACTION6.
- **Frame-change head**: predicts which actions cause frame change (StochasticGoose's signal, used as auxiliary loss + exploration bias).
- **Replay-conditioned head**: see §6.5.1 below — viability differs sharply between public and private envs.
- **1-step lookahead planner**: see §6.5.2 below — dynamics source is an explicit Phase 0c decision.

**12 GB feasibility:** Tight but workable — ~6–8 GB at training.
**Train compute:** 2–3 days for full stack.
**Inference latency:** ~30–80 ms.
**Expected ceiling:** Highest among realistic-in-5-weeks options. Combines every signal we have access to (BC, replays, frame-change prediction, action availability).

**This is the leading candidate for Phase 0c architecture spec.** Not a commitment — Phase 0c is where the architecture decision lives. Listing here only to show the design space converges naturally around this kind of hybrid.

#### 6.5.1 Replay-conditioned head: public vs private viability

The replay corpus only covers the **25 public envs**. The competition evaluates on **110 private envs** (55 semi-private → Public LB during competition; 55 fully-private → Private LB at end, determines prize ranking) with zero in-corpus replays. Conditioning retrieval has very different strength on the two sets:

- **On the 25 public envs (training-time / dev signal):** retrieve top-k replays from the *same env*. The retrieval target literally played the env we're solving. Next-action priors from these replays are strong — humans almost never waste actions, so their action distribution at a given (env, level, state) is a near-optimal soft prior. Use this aggressively at training time to speed BC convergence and as a sanity-check baseline at inference on public envs.
- **On the 110 private eval envs (where it actually matters):** there are no in-env replays. Retrieval collapses to **cluster-level marginal action distributions** — i.e. "this is a pure-click env, so prefer ACTION6 with these spatial priors" / "this is a pure-movement env, prefer arrows uniformly" / "this is mixed, fall back to a broader distribution" (§4.3 clusters). That is a much weaker signal — not nothing, but mostly redundant with `available_actions` and a uniform prior over the legal subset.

**Implication.** Replay-conditioning is **not a primary lever for the eval set**. Its real value is (a) faster BC training by giving the model a strong distillation target, (b) public-env scaffolding during dev and validation. Phase 0c must explicitly evaluate whether the inference-time retrieval head earns its engineering cost given the public-only utility — a reasonable Phase 0c outcome is to keep retrieval as a training-time mechanism only and drop the inference-time retrieval head.

#### 6.5.2 Dynamics source for the 1-step lookahead planner

The closing formula `argmax(frame-change-likelihood × BC-prior × replay-prior)` is underspecified — where does the frame-change-likelihood for *unexecuted* candidate actions come from? Two options. Phase 0c must pick one explicitly.

- **Option L1 — Binary frame-change head (StochasticGoose-style).** A small head trained on `(state, action) → did the frame change?` as a Bernoulli output. Cheap: <1 ms inference, easy to train as an auxiliary loss alongside BC. Gives a per-action *probability of producing any change* but no information about *what* the next frame looks like. This is exactly the signal StochasticGoose used to bias exploration. Useful for filtering out no-op actions; useless for distinguishing between two actions that both change the frame.
- **Option L2 — Learned forward model.** A small CNN/Transformer that predicts the next frame (or its latent) from the current frame + a candidate action. ~1–2 weeks of focused engineering on top of the BC stack. Enables a *real* 1-step lookahead — the planner can compare candidate next-frames against an expected-goal feature, distinguish productive moves from same-state-shuffles, and (optionally) extend to k-step rollouts. Higher ceiling, materially higher implementation risk and training instability.

**Phase 0c decision criteria.** L1 if the team prioritizes shipping a tight BC+aux baseline by Milestone #1 and treats lookahead as exploration biasing. L2 if early experiments show BC alone clears levels 1–2 reliably but stalls on level 3+ — that's the regime where a real forward model earns its cost. Do not assume either in advance.

### 6.6 Approach matrix

| | A: Primitives | B: BC + RL | C: Graph mem | D: World model | E: Hybrid |
|---|---|---|---|---|---|
| Train compute | minutes | hours | hours | days | 2–3 days |
| Inference (ms) | <5 | ~10 | ~20 | 50–200 | 30–80 |
| 12 GB local debug | trivial | trivial | trivial | tight | workable |
| 5-week feasibility | yes | yes | yes | risky | yes |
| Expected score ceiling | low | moderate | moderate | highest if trained | high |
| Novelty / leaderboard differentiation | low | moderate | low | high | high |

---

## 7. Competitive Intel

### 7.1 Kaggle leaderboard snapshot — user-supplied

**Current top score: ~1.17%** (StochasticGoose / Tufa Labs). Frontier LLM teams (Opus 4.6, GPT 5.4, Gemini 3.1, Grok 4.20) all sub-1% on the related semi-private set. The Kaggle page is client-rendered and not directly fetchable in this session — full leaderboard pull is a [VALIDATE-0b] item.

### 7.2 Identifiable teams and approaches in the wild

- **StochasticGoose (Tufa Labs / Dries Smit):** CNN + biased random search. Preview winner. Launch collapse. Almost certainly retooled by now — [VALIDATE-0b].
- **Blind Squirrel (wd13ca):** State-graph. Probably not competitive at launch.
- **OpenClaw (ARC Prize Foundation):** 5.2% on public set, $2.9k cost. Frontier LLM + memory tools. **Not actually a Kaggle competitor** — it's the foundation's own demo agent. The 5.2% is on the *public set* and is therefore not comparable to Kaggle (private) scores. Their docs make a point that public-set scores are not progress markers.
- **Read-Grep-Bash (Fox, Wang, Rosu, Dhingra — Duke):** Academic harness, evaluated on public envs near human level, but the bimodal-failure result (TR87 97.1%, BP35 0.0%) is published as a caution. Not on Kaggle to our knowledge.
- **Arcgentica (Symbolica AI / Knutsen, Klein):** Orchestrator-subagent. Public envs only.

### 7.3 Realistic top-3 score threshold for Milestone #1 (2026-06-30)

**Low-confidence estimate. Wide error bars.** Given current top is 1.17% and the benchmark was just released to Kaggle a few weeks ago: top-3 by 2026-06-30 will *most likely* require **2–5% RHAE**. Anything ≥5% is podium-safe under that median case. Reasoning:

- Frontier LLMs at sub-1% suggest "free intelligence" is not enough.
- StochasticGoose's 1.17% with no human replays sets a floor that anyone with replays should clear.
- The 1.15× cap means even strong per-level efficiency tops out at ~115% of human; the dominant factor is **levels completed** (per-env cap §1.6).
- Completing 2 out of 6 levels on every env, at human efficiency, scores ~3/21 ≈ 14% on that env, and ~3-5% across the 110 private envs after weighting and partial-success penalties.

**Bracket the uncertainty.**

- **Lower-bound scenario (~1.5–2%).** Kaggle competitions in their early weeks usually have a long tail of stub notebooks (<0.1% scores). If the leaderboard past the top-1 is mostly stubs and frontier-LLM submissions stall sub-1%, top-3 could be reachable at **1.5–2%** — i.e. just clearing the current 1.17% leader by a noticeable margin. [VALIDATE-0b: pull the full leaderboard via Kaggle MCP to see the actual distribution past the top 1.]
- **Median scenario (2–5%).** The estimate above.
- **Upper-bound scenario (~5–8%).** If a serious team (Tufa Labs themselves, a frontier-lab side project, an academic group) ships a BC-pretrained or replay-conditioned agent in the next 4 weeks, the bar can move sharply. Multiple credible teams know the human dataset dropped 2026-04 and have had time to exploit it.

**Operating plan.** Continue planning for the **5–7% stretch target** (Approach E or similar) — it's the configuration we have the highest expected score on and survives both the median and upper scenarios. But recognize that an honest **2% submission is also podium-viable** under the lower scenario, which lowers the bar to "ship *something* solid" before the deadline. Do not over-engineer if a clean 2–3% is in hand with submissions remaining.

### 7.4 Will the bar move?

Probably. Five weeks is enough for ≥1 team to ship a BC-pretrained agent now that the human dataset is public. Conservative read: assume the bar moves to 3–5% by mid-June. Aggressive read: assume someone publishes a 10%+ result and reshapes expectations. We can't outguess this — focus on architecture quality, not on chasing a moving leaderboard.

---

## 8. Risks & Operational Constraints

### 8.1 Submission budget

- **1 submission per team per day** (UI-enforced; Kaggle rules-text boilerplate says "five (5) per day" but the per-competition Submit UI caps at 1/day — confirmed Phase 0b via Topic 689621 + user UI inspection).
- **~34 submissions from today through 2026-06-30** (1/day × 34 days; Phase 0b corrected the original "5/day" misreading from rules-text). With ±0.2 score variance (Phase 0b §S3) requiring ~3–5 confirms per architecture, net architecture-iteration capacity = **5–8 distinct variants end-to-end** before Milestone #1.
- **No submissions in Phase 0.** Treat the first ~10 submissions as architecture validation; the last ~10 as final tuning; keep ~15 as buffer.

### 8.2 Wall-clock

- **9 h per submission** (was 12 h in Phase 0a — corrected Phase 0b).
- Per-action ceiling (worst case): 0.52 s. Comfortable target: 100 ms.
- Training the agent's weights happens *offline*. The Kaggle notebook only does inference + (possibly) lightweight per-env online adaptation.
- All weights must be uploaded as a Kaggle dataset and loaded at submission time (no internet). Dataset size budget: Kaggle datasets cap at ~100 GB per dataset; well within our needs.

### 8.3 VRAM ceilings

- **Local dev: 12 GB (RTX 5070 Ti).** Any model whose training-time footprint exceeds 12 GB cannot be debugged locally. This is the binding constraint on architecture choice.
- **Kaggle eval: 48 GB (RTX 6000).** Larger inference is fine on Kaggle but cannot be debugged → burns submissions.
- **Disk for Kaggle weights dataset:** ~100 GB ceiling per dataset; multiple datasets allowed.

### 8.4 No internet on Kaggle

- All weights uploaded as a private Kaggle dataset and mounted at `/kaggle/input/<name>/`.
- No pip install at runtime — use the bundled wheels in `arc_agi_wheels/` or pre-bake a Kaggle Docker.
- No external HTTP from agent code.
- The `arc_agi` toolkit's `OperationMode.OFFLINE` / `OperationMode.COMPETITION` paths are designed for this exact case.

### 8.5 OSS license requirement

- **Prize-eligible solutions must be CC0 or MIT-0.** [VALIDATE-0b: confirm exact license names allowed.] Tech Report §7 says "participants must open source their solutions in order to receive prize money."
- Implication: avoid GPL-licensed dependencies. ResNet-style backbones from torchvision (BSD), Hugging Face Transformers (Apache-2.0) — both compatible.
- Pretrained weights must also be license-compatible. ImageNet pretraining → OK. Anything trained on non-open data → risk.

### 8.6 RTX-hardware rule

- User-flagged constraint: "RTX-only-for-this-competition rule and ban risk." Likely meaning: the competition specifies RTX 6000 hardware and using other accelerators (TPU, custom) is forbidden. [VALIDATE-0b: confirm exact wording from Kaggle rules.] Practical reading: just stick to Kaggle's standard RTX notebook environment and you're safe.

### 8.7 Kaggle MCP (Phase 0b)

The Kaggle MCP will expose tools like `submit`, `list_datasets`, `upload_dataset`, `get_leaderboard`. The **`submit`** tool is the submission path — it should be **gated by explicit user confirmation** in CLAUDE.md (Phase 0c task). Other read-only tools (list_datasets, get_leaderboard) can be allowlisted. This research doc does not configure the MCP; that happens in Phase 0c.

### 8.8 Env updates between now and submission

The April 2026 changelog shows **15 envs were re-versioned** (tn36, m0r0, r11l, tu93, vc33, sc25, ar25, dc22, cn04, sp80, su15, re86, ka59, s5i5, sk48) with new hash IDs. If the foundation pushes another env update before Milestone #1, any policy that's overfit to env-hash-specific frame statistics could break. Mitigation: train on the *latest* env hashes only, use frame-invariant features (color, object shape, connected components) not pixel-exact matches.

### 8.9 Engine version match

- Bundled: `arc_agi 0.9.8`, `arcengine 0.9.3` (Kaggle wheels).
- Local PyPI install: same versions, verified by `pip show`.
- Future SDK updates would change FrameData fields (the `score` → `levels_completed` rename in 0.9.3 broke pre-0.9.3 code). Pin versions, don't auto-upgrade.

### 8.10 The single biggest risk

**One submission per day + 9-hour eval cycle = feedback loop of >24 h per architecture iteration on the real eval set.** Any bug discovered post-submit costs a full day. Local debugging must catch >95% of bugs before submission. This argues strongly for:
- A faithful offline replay of submission flow (use OFFLINE mode + same scorecard pipeline locally — verified working in §2.8).
- Smoke tests that exercise every env at least once on every code change.
- A staging "shadow eval" suite that runs the full 25 public envs as a sanity gate before any submission.

---

## 9. Open Questions for Phase 0b

To resolve hands-on. Each pinned to a concrete validation step.

1. **Exact Kaggle competition rules** — pull the rules tab via authenticated browser or Kaggle MCP. Specifically: submission quota, runtime ceiling, allowed licenses (CC0 vs MIT-0 vs Apache-2.0), team-size limit, RTX hardware rule wording, Milestone #1 prize structure.
2. **Action 7 cost accounting** — does ACTION7 (undo) consume an action against the 5× budget? Test by stepping a known-undo-supporting env (bp35) and observing `actions` field on the scorecard.
3. **RESET cost in COMPETITION mode** — does a level reset cost an action? Test in OFFLINE mode against sp80 (sp80 resets are easy to trigger via step-budget exhaustion).
4. **Frame-stack semantics on `env.step()`** — does the SDK return all transition frames in `frame: list[ndarray]` or only the last one in COMPETITION mode? Run sp80 a few hundred steps and inspect.
5. **`available_actions` per turn** — is it always the env's full action set, or can it vary turn-by-turn (e.g. ACTION5 only available when something is selected)? sp80 code suggests it can vary (`_get_valid_actions` overrides). Confirm via reset → step → step trace.
6. **Audit all 25 env .py files for mechanic vocabulary** — produce a table of sprite tags, action handlers, lose conditions, mid-env mechanic injections. This is the highest-information offline task before architecture spec.
7. **Replay file integrity** — some records have numeric action IDs (`1..7`), others string (`"ACTION1"..."ACTION7"`). Two recorder versions? Are mid-replay frame events the *response* frames or the *pre-action* frames? Read `agents/recorder.py` callers and confirm.
8. **Kaggle MCP tool surface** — once enabled, document every tool, which are read-only vs submission-path, and what each consumes.
9. **Public-set vs private-set behavior gap** — there's no way to test private-set behavior without a submission. Build a confidence-prediction proxy: for each public env, measure how predictable the agent's score is from its level-1 performance. A noisy predictor → real-eval risk is high.
10. **Animation-frame handling** — does the SDK send animation transitions as a single multi-frame stack inside *one* `step()` response, or as multiple consecutive `step()` responses where the agent is forced to take no-op actions? Replay format shows multi-frame stacks per record, suggesting the former. Verify against live SDK behavior.
11. **Help button** — the user notes there's an in-console Help button. The SDK has no help action. Is the Help button purely a UI affordance for human players, or does it map to a special call (e.g. an instructions hint via a private API)? [Almost certainly UI-only; confirm via SDK source grep.]
12. **Frontier-LLM Kaggle scores** — the 0.5%-and-below numbers in §5.3 are from the semi-private set. What do those same models score on the Kaggle (fully private) set? If unposted, our top-3 threshold estimate (§7.3) needs to be revisited.
13. **Python version on Kaggle.** Local dev uses Python 3.13. Bundled wheels are cp312 (Python 3.12) Linux. The Kaggle competition image's actual Python version is unverified. Run a non-submission Kaggle notebook, print `sys.version`, list `/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_wheels/`, confirm `import arc_agi, arcengine` succeeds. Mismatch invalidates any architecture assumption based on local-Kaggle parity.
14. **Novelty-handling strategy gap.** The 110 private envs will introduce mechanics not present in the 25 public envs (engine vocabulary is shared, but mechanic compositions are new). §4.7 and §5.5 identify the gap but propose no mitigation. Phase 0c must produce a concrete strategy — placeholder candidates: (a) cluster-level priors as soft fallback when in-env replay retrieval is unavailable, (b) exploration bonus weighted by sprite/object novelty relative to public set, (c) frame-change-prediction-guided exploration when BC prior has high entropy. Pick and justify in 0c. Evidence-gathering input from Phase 0b: env source audit (§9 #6) — quantify mechanic diversity across the 25 public envs to estimate the breadth of the engine's vocabulary the model has been exposed to.
15. **Kaggle COMPETITION-mode scorecard TTL.** Local SDK default is 3 days (§2.6). 2026-04-14 changelog mentions a 24-hour cap for the ONLINE API. The effective TTL inside a Kaggle COMPETITION-mode session is unverified. 9-hour Kaggle runtime caps it anyway (corrected Phase 0b — was 12h in Phase 0a), but confirm via a Kaggle notebook run that no premature scorecard expiry occurs.

---

## Appendix A — Local environment summary (as of 2026-05-26)

- Python 3.13.13 venv at `C:\Users\adars\Downloads\ARC-AGI-3\.venv313`.
- Installed: `arc-agi 0.9.8`, `arcengine 0.9.3` + deps (flask, matplotlib, numpy, pillow, pydantic, requests, python-dotenv, blinker, click, contourpy, cycler, fonttools, idna, itsdangerous, jinja2, kiwisolver, markupsafe, packaging, pyparsing, python-dateutil, six, urllib3).
- Verified: `Arcade(operation_mode=OFFLINE, environments_dir=...)` loads all 25 public envs and produces working scorecards offline.
- Engine throughput: ~5,860 steps/s for sp80 single-threaded on Windows.
- 340 replay files inventoried at `data/human_replays/...`. Aggregated stats in `.tmp_replay_stats.json`.

## Appendix B — Files produced this phase

- `research-findings.md` (this document)
- `.tmp_pdf_read.py` (throwaway; extracted Tech Report)
- `.tmp_replay_stats.py` (throwaway; aggregated 340 replays)
- `.tmp_replay_stats.json` (per-env stats — keep, useful for §4.4 and Phase 0b)
- `.tmp_smoke.py` (throwaway; smoke-tested arc_agi offline against sp80)
- `.venv313/` (Python 3.13 venv with `arc-agi` toolkit installed)

All `.tmp_*` files are research artifacts; the canonical deliverable is this document. Recommend keeping `.tmp_replay_stats.json` as input for Phase 0b's full env-mechanic audit.

---

**Status: Phase 0a complete. Awaiting user review before Phase 0b.**
