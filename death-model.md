# Death-Model Investigation (Phase 3 v2, Task A)

**Date:** 2026-05-29 · Local, no submission. Evidence: engine source (`arcengine/base_game.py`), canonical agent loop (`ARC-AGI-3-Agents/agents/`), and empirical drive-to-death (`scripts/death_probe.py`, OFFLINE; the state machine is mode-independent).

## VERDICT: **C — death is NON-TERMINAL until the agent stops resetting.**

GAME_OVER does NOT end an env. The canonical loop RESETs on GAME_OVER and plays on until WIN or the action cap. **Our local harness breaks on GAME_OVER → it caps the agent at its first death → self-inflicted zero on every env that dies before completing level 1.** The harness is WRONG and must be fixed (Task B).

---

## Engine semantics (source-confirmed)

`arcengine/base_game.py`:
- `lose()` (301-303) sets `state = GAME_OVER`. There is **no base-level "lives" field**; lives, where they exist, are env-internal (pixels) and the env decides when to call `lose()`.
- `perform_action` (204-216): once `state ∈ {GAME_OVER, WIN}`, a **non-RESET** action returns an **empty frame** (`frame=[]`) and the state unchanged — the env is frozen.
- `handle_reset` (305-316): on RESET, if `action_count==0 or state==WIN` → `full_reset` (back to level 0, score=0); **else → `level_reset`** (clones the *current* level, `state → NOT_FINISHED`, **`_score` / levels_completed PRESERVED**, `full_reset=False`).
- `levels_completed` == `self._score` (244) — monotonic; not reduced by a death or a level_reset.
- `_action_count` is incremented per non-RESET action (278-279) and is NOT reset by `level_reset` (only by `full_reset`). RESET itself costs 1 action (Phase 1 Issue 2; re-confirmed).

So: a death freezes non-RESET actions, but **RESET after a death does a level_reset → revives play on the current level, keeps completed levels, keeps the action count climbing.**

## Canonical agent loop (what the leaderboard expects)

`ARC-AGI-3-Agents/agents/agent.py:69-89` — `while not is_done() and action_counter <= MAX_ACTIONS`. The loop itself never breaks on GAME_OVER; termination is the agent's `is_done()`.

`agents/templates/random_agent.py:24-44` (the reference agent):
```python
def is_done(...):
    return any([state is GameState.WIN,
                # state is GameState.GAME_OVER,    # <-- COMMENTED OUT, deliberately
               ])
def choose_action(...):
    if state in [NOT_PLAYED, GAME_OVER]:
        action = GameAction.RESET        # revive + continue
    else:
        action = random.choice(non-RESET actions)
```
GAME_OVER is **explicitly not** a stop condition; the agent RESETs and keeps playing. Termination is WIN-all-levels or MAX_ACTIONS.

## Empirical confirmation (`death_probe.py`)

| env | first GAME_OVER | non-RESET in GAME_OVER | RESET after death | lvls preserved | revived |
|---|---|---|---|---|---|
| r11l | step 29 (lvls=1) | state GAME_OVER, frame_len **0** (frozen) | state **NOT_FINISHED**, full_reset=False, frame_len 1 | **lvls=1 kept** | ✅ |
| ls20 | step 135 (lvls=0) | frozen, frame_len 0 | NOT_FINISHED, frame_len 1 | lvls=0 | ✅ |
| vc33 | step 50 (lvls=0) | frozen, frame_len 0 | NOT_FINISHED, frame_len 1 | lvls=0 | ✅ |

r11l hit GAME_OVER 19× in 400 random steps and each RESET revived it; it kept its level-1 completion throughout. Confirms: **non-RESET while dead = frozen empty frame; RESET = level_reset revival; completed levels persist.**

## Answers to Task A questions

1. **Does an episode continue past a within-budget death?** Yes — via RESET. A death sets GAME_OVER (freezing non-RESET actions); RESET (1 action) does a level_reset and play resumes on the current level. An env with internal "lives" stays `NOT_FINISHED` across life-losses and only emits GAME_OVER at true end — the harness already continues through `NOT_FINISHED`, so the only break bug is on GAME_OVER.
2. **RESET after death vs mid-level?** Both do `level_reset` (since `action_count>0`, `state≠WIN`): restart the current level clean, `state→NOT_FINISHED`, score preserved, costs 1 action. (After a WIN, RESET would `full_reset` to level 0.) `ONLY_RESET_LEVELS=true` forces level_reset always.
3. **Does clearing a level restore lives/resource?** Completing a level increments `_score`; the next level loads its own starting budget/layout (`set_level`). Completed-level score is never lost. (Per-env life restoration on level-clear is env-internal; consistent with the ls20 answer key but not a base-engine guarantee.)
4. **Is GAME_OVER the only terminal?** Engine-side, the only "stop" signals are WIN (all levels) and GAME_OVER — and GAME_OVER is escapable via RESET. There is no engine per-level hard step cap; the env's own logic calls `lose()` (e.g. budget exhausted with no lives left). The effective cap is the agent/grader action budget (our harness: 5×Σbaseline).
5. **Does our local harness replicate the engine?** **NO.** `eval/harness.py:144` — `if state.endswith("WIN") or state.endswith("GAME_OVER"): break` — terminates the env loop on the first GAME_OVER. The engine + canonical loop would RESET and continue. The harness undercounts every env that dies before clearing level 1.

## ls20 answer-key check

User hand-play: budget depletion costs one (red) life; episode continues with fewer lives; GAME_OVER only when all lives gone; clearing a level with ≥1 life restores lives. This is **consistent** with the engine model: ls20 keeps `state=NOT_FINISHED` across internal life-losses (refilling its yellow budget, decrementing a red segment) and calls `lose()→GAME_OVER` only at true exhaustion. Empirically ls20 took 135 steps to first GAME_OVER vs r11l's 29 — consistent with a multi-life buffer. Confirming the exact red-segment decrement needs ls20's obfuscated internals / pixel inspection; the SDK-level fact that matters (GAME_OVER is non-terminal via RESET, completed levels persist) is confirmed. **The answer key must NOT be hardcoded; the agent learns life/death per env by observation.**

---

## Consequence (for review)

Fixing the harness to RESET-and-continue on GAME_OVER (until WIN or 5×Σbaseline actions) faithfully replicates the engine + canonical loop and may move holdout off 0 "for free": envs that currently score 0 only because they die before level 1 get retries within budget. The v1 lethality/persistence machinery was likely inert because of THIS bug, not capability — Task B re-measurement is the real test.

**STOP — handing back for review before Task B (harness fix + life learning + retry loop) and Task C (goal-by-interaction).**
