# CLAUDE.md — ARC-AGI-3 Kaggle Competition

**Project:** ARC Prize 2026 / ARC-AGI-3 Kaggle competition.
**Owner:** Adarsh (`adarshreddybms@gmail.com`, Kaggle user enrolled in `arc-prize-2026-arc-agi-3`).
**Authoritative phase docs:** `research-findings.md` (Phase 0a), `validation-results.md` (Phase 0b), `phase-0c-spec.md` (Phase 0c), `phase-1-bootstrap.md` (Phase 1 plan).

This file is read first by every Claude Code session. It defines the operating rules that override default Claude Code behavior for this project. **If anything in this file contradicts a phase doc, this file wins on operational rules and the phase doc wins on technical content.**

---

## 1. Submission discipline (highest-priority rules)

### 1.1 Submission quota — 1 per team per day

The Kaggle Submit UI caps this competition at **1 submission per team per day** (resets every ~24 hours). The generic Kaggle Rules-text §1.2.2.a says "five (5) per day" — this is boilerplate that does not apply here. Authoritative source: Kaggle Submit button + Topic 689621.

**Through Milestone #1 (2026-06-30):** ~34 submissions total. Reserve ≥10 for variance confirmation; net ~24 for architecture iteration. The realistic ceiling is **5–8 distinct architecture variants** end-to-end.

### 1.2 User submission gate — local score ≥ 10 on holdout

**No Kaggle submission is allowed until `harness_score_holdout ≥ 10.0`** on the 5 held-out public envs (`vc33`, `tu93`, `sk48`, `lp85`, `dc22`). Score units are the same as Kaggle's leaderboard (`scorecard.score`, 0–115 scale — see Phase 0c §0 OQ5).

Exceptions (the gate may be skipped only for these specific submissions):
- **S1** — first plumbing-validation random agent (no learned components; gate doesn't apply because there's nothing to gate).
- **S3** — explicit re-submission of S1 verbatim for variance characterization (same agent → same gate disposition).
- Any user-directed exception in the immediately prior user message, stating "skip gate this submission" and justifying.

**Gate is non-exempt going forward (Phase 1 precedent, 2026-05-27).** When offered the S1 plumbing-validation exemption above, the user declined and chose to skip the submission rather than burn a slot on a sub-gate agent. Do not propose future exemptions. Special-handling submissions (variance characterization, etc.) concern WHICH gate-qualifying agent to re-submit — not whether to skip the gate. The S1/S3 carve-outs above remain as-written for historical/spec compatibility but are not to be invoked.

**Hard submission date — 2026-06-25 (Phase 3 v1 directive, 2026-05-29).** To guarantee a real Milestone-#1 leaderboard entry with buffer before 2026-06-30 (room for 1/day + ±0.2 variance), submit the **best available stable agent on 2026-06-25** — i.e. the one that beats the *realistic* random baseline (`baseline-realistic.md`: **0.0000** at the real 5×-baseline budget; the old 0.0581 anchor was a 2000-action Kaggle Save&Run and does not apply) by the clearest margin. **Early trigger:** if `harness_score_holdout ≥ ~10` before 2026-06-25, submit then instead. This is a user-authorized, dated exception to the ≥10 gate for the single 2026-06-25 entry only — it does NOT relax the gate for any other submission, and §1.3 per-message authorization still applies (re-ask in the message immediately before the call).

### 1.3 Per-message authorization required for every Kaggle submission

Every call to `mcp__kaggle__submit_to_competition` (or `kaggle competitions submit` via CLI) requires explicit user authorization **in the message immediately prior to the call**. Past-conversation authorization, "we agreed to submit yesterday," memory of previous OK — none of these count. Re-ask every time.

The authorization message from the user must contain a phrase like:
- "submit to Kaggle"
- "submit S4 now"
- "go ahead with the submission"

Before the call, Claude Code must state in chat:
1. Remaining daily quota (`mcp__kaggle__search_competition_submissions` → count today's).
2. Latest `harness_score_holdout` value (and the gate disposition).
3. One-sentence description of what's different from the previous submission.
4. Then wait for the authorization message.

### 1.4 Submission-path tool gating (Kaggle MCP)

These Kaggle MCP tools are **SUBMISSION-PATH** — never call without §1.3 authorization:

- `mcp__kaggle__submit_to_competition`
- `mcp__kaggle__create_code_competition_submission`
- `mcp__kaggle__start_competition_submission_upload`

These are **READ-ONLY-SAFE** — Claude Code may call freely as needed:

- `mcp__kaggle__get_competition`, `mcp__kaggle__get_competition_leaderboard`, `mcp__kaggle__download_competition_leaderboard`
- `mcp__kaggle__list_competition_data_files`, `mcp__kaggle__list_competition_pages`
- `mcp__kaggle__list_forum_topics`, `mcp__kaggle__get_forum_topic`
- `mcp__kaggle__search_competition_submissions`, `mcp__kaggle__get_competition_submission` (own submissions only)
- `mcp__kaggle__list_models`, `mcp__kaggle__search_datasets`, `mcp__kaggle__search_notebooks`
- `mcp__kaggle__list_notebook_files`, `mcp__kaggle__get_notebook_info`

These are **POTENTIALLY DESTRUCTIVE** — gated by user authorization in the prior message:

- `mcp__kaggle__upload_dataset_file`, `mcp__kaggle__update_dataset_metadata`
- `mcp__kaggle__create_model`, `mcp__kaggle__update_model`, `mcp__kaggle__update_model_variation`
- `mcp__kaggle__save_notebook`, `mcp__kaggle__cancel_notebook_session`

When unsure, treat as gated.

---

## 2. Kaggle environment — non-negotiable submission requirements

Every submission notebook MUST conform to the following or it will crash silently or score 0:

### 2.1 `ARC_API_KEY` workaround (Phase 0b §S9)

Before any `from arc_agi import ...` or `Arcade(...)` instantiation, set:

```python
import os
os.environ["ARC_API_KEY"] = "noop"   # any non-empty string
```

Why: `arc_agi/base.py:172–174` calls `_get_anonymous_api_key()` over HTTPS to `three.arcprize.org` whenever no key is set and mode != OFFLINE. Kaggle submission environment has internet disabled → fetch fails → Arcade constructor crashes before any env interaction. The workaround skips the fetch entirely. Kaggle's grader does not actually need the key for COMPETITION-mode local scoring.

### 2.2 Correct Kaggle paths (Phase 0b §S10)

The dataset mounts at:

```
/kaggle/input/competitions/arc-prize-2026-arc-agi-3/
├── environment_files/<env>/<hash>/<env>.py + metadata.json
├── arc_agi_3_wheels/*.whl
└── ARC-AGI-3-Agents/
```

**Note `competitions/` prefix and `arc_agi_3_wheels` (not `arc_agi_wheels`).** Both wrong in our initial Phase 0b draft; corrected.

### 2.3 Pip install from bundled wheels only

Kaggle's competition environment has internet disabled. Pip must use `--no-index` and the bundled wheels directory:

```python
import sys, subprocess
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
    '--find-links', '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',
    'arc-agi', 'arcengine'
], check=True)
```

Never `pip install arc-agi` without `--no-index` — will silently fall back to PyPI if internet leaks back on.

### 2.4 `OperationMode.COMPETITION` + one make() per env

In COMPETITION mode `arc.make(game_id)` may be called **exactly once per env per submission**. No retries. The agent commits to one trajectory per env. RESET actions issued by the agent count against the action budget (only the implicit reset inside `arc.make()` is free).

### 2.5 Track actions/levels at agent layer (Phase 0b §S5)

In OFFLINE mode (used by the local harness) the scorecard counters do not increment from `env.step()`. **Local correctness validation must count actions and levels at the agent wrapper layer.** The COMPETITION-mode scorecard DOES count correctly on Kaggle (parity test confirmed actions=11 from 1 reset + 10 steps), so the divergence is OFFLINE-only.

---

## 3. Dependency pins

These versions are bundled in Kaggle's wheels directory and known to work end-to-end. Pin in `pyproject.toml`:

```
arc-agi==0.9.8
arcengine==0.9.3
numpy==2.4.4
pydantic==2.13.2
pillow==12.2.0
torch==2.12.0+cpu  # for CPU dev; +cu121 for GPU training
matplotlib==3.10.8
flask==3.1.3        # transitive via arc-agi
```

Do not auto-upgrade. SDK breaking changes happened in 0.9.3 (`score` → `levels_completed` rename — see ARC-AGI-3-Agents changelog) and could happen again.

If `arc_agi` updates to ≥0.10 and we adopt it, **re-validate** every Phase 0b finding before next submission (anon-key fetch behavior, OFFLINE scorecard counter, frame-stack semantics, available_actions constancy). Treat SDK-version bump as a Phase-0b-redo.

---

## 4. Hardware constraints

### 4.1 Local development — 12 GB VRAM (RTX 5070 Ti)

Every training run must fit in 12 GB. Architecture choices are made against this ceiling. Backbone size cap: ~5 M params (ResNet-18-equivalent). Batch size ≤128 at 64×64 input. Use mixed precision (fp16 forward, fp32 master weights) if approaching the ceiling. Never rely on Kaggle's 48 GB to debug architecture choices — anything that needs more than 12 GB to debug locally cannot be iterated on inside the 1-submission/day budget.

### 4.2 Kaggle inference — 48 GB VRAM, 9-hour runtime

`g4-standard-48` (4× RTX 6000 Pro per node), internet disabled, Python 3.12.13. Runtime ceiling = 9 hours. RTX-only-for-this-comp rule (Kaggle Upgraded Accelerators page): using these RTX machines for any other competition's notebook risks site suspension or account ban.

### 4.3 Per-action latency budget — ~165 ms ceiling

Compute math: 110 private envs × ~7.3 levels × 5× action cap × ~50 baseline = ~200K actions worst case. 9 h = 32,400 s. → **165 ms per action ceiling**. Comfortable inference target: **50 ms p95** to leave 3× headroom for engine-side, animation rendering, and tail latencies.

---

## 5. License

### 5.1 Winners license — CC-BY 4.0 (Kaggle Rules §1.5.a)

Final submitted notebook + all weights + all training code must be released under **CC-BY 4.0** to be prize-eligible. Attribution required for any third-party code reused (pretrained model weights, library forks, etc.).

### 5.2 Data access — Apache 2.0

The bundled `environment_files/` and ARC-AGI-3-Agents source are under Apache 2.0 (per Kaggle Rules §1.7). Reuse and modification fine for the competition.

### 5.3 Open-source AI checklist

Per Kaggle Rules: "Submissions are required to have open source system, open source model, and open source weights/parameters, as defined in the checklist from the Open Source AI definition by the Open Source Initiative." Practical interpretation: all model weights uploaded as a public Kaggle dataset; all training code in a public GitHub repo by Milestone #1; no proprietary dependencies.

---

## 6. Reproducibility

### 6.1 Seed determinism — mandatory for every agent

- Every agent must produce identical actions for identical (seed, env, observation) triples.
- Use explicit `random.Random(seed)` and `np.random.default_rng(seed)`; never `random.random()` or `np.random.rand()` (process-global).
- Per-env seed = `hash((env_id, run_id, agent_version)) & 0xFFFFFFFF`.

### 6.2 Training repro

- `torch.use_deterministic_algorithms(True)`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var.
- Fixed seeds: python (0), numpy (0), torch (0), torch.cuda (0).
- Training run config saved to `runs/<timestamp>/config.json`.
- Pin dep versions (§3).

### 6.3 Submission notebook repro

Submission notebook prints at start:
- Git SHA of training code
- Hash of weights file used
- Versions of `arc-agi`, `arcengine`, `torch`, `numpy`
- Per-env seed scheme

This makes every submission self-documenting.

---

## 7. Cross-doc rules

### 7.1 Do not infer team identity from team-name similarity

Phase 0b incident: I assumed `StochasticGoose_v7_final` at rank 30 was a Tufa Labs entry, based on name similarity. User clarified it's an unrelated participant. **Treat similarly-named entries as independent unless verified via account/team links.**

### 7.2 "Read everything, copy nothing" — public envs

Per Phase 0b Surprise S4 + Topic 699900: Greg Kamradt confirmed that reading and deep-copying the public 25-env source code is fair use. **Use this for:** dev-time synthetic-rollout generation (capped at 25% of training data per Phase 0c §4.8), debugging, env-mechanic reference. **Do NOT use this for:** any strategy that hardcodes per-env winning sequences — those don't transfer to the 110 private envs.

### 7.3 Memory & past-chat policy

Future Claude Code sessions on this project must:
1. Read this file (`CLAUDE.md`) first.
2. Read the latest phase doc relevant to the current work.
3. Check `git log` (once a repo exists) for the most recent submissions and their outcomes.
4. Never assume "the previous session said it was OK" — verify against the docs and ask if unclear.

---

## 8. What's allowed per phase

| Phase | Status | Allowed | Forbidden |
|---|---|---|---|
| 0a (research) | ✓ done | research docs only | code, training, repo, submissions |
| 0b (validation) | ✓ done | validation scripts, replay parser, smoke tests | agent code, real training, submissions |
| 0c (spec) | ✓ done | spec docs, CLAUDE.md, bootstrap plan | code, training, repo, submissions |
| **1 (bootstrap + S1)** | **next** | repo, harness, biased-random agent, S1 submission | learned-model code (deferred to Phase 2) |
| 2 (E-lite implementation) | future | training pipeline stages 0-3, S2-S4 submissions | premature architecture pivots |
| 3 (refinement) | future | re-training, S5+, variance characterization | new architectures (lock to E-lite for M#1) |

Architecture pivots between phases require explicit user authorization plus an updated phase doc.

---

## 9. Failure modes to watch for

These have either been observed in Phase 0a/0b or are predictable from the constraints:

- **Stale rules-text reading.** Phase 0b initially trusted Kaggle's generic Rules-text "5/day" — wrong. Always cross-check rules-text against the live Submit UI and against the per-competition discussion forum.
- **Path drift between local and Kaggle.** Phase 0b Surprise S10: the bundle root and wheels directory names changed without notice in our notes. Always verify paths with a `glob('/kaggle/input/**/*.whl')` walk on a non-submission Save & Run before submitting.
- **Score variance ±0.2.** Topic 699208 + planned in Phase 0c §3. Never claim "we hit X" from a single submission. Always require ≥3 confirms before treating a score as the true mean.
- **OFFLINE scorecard mirage.** Phase 0b §S5 (qualified Phase 1): OFFLINE scorecard counters stay at zero under the `Arcade(operation_mode=OFFLINE, ...)` + `LocalEnvironmentWrapper` path used by the local harness. They DO fire under env-var-driven `Arcade()` (no kwargs) used by the Kaggle notebook — see validation-results.md Phase 1 Findings. Agents that rely on scorecard fields for self-verification in the local harness will fail silently; agents on Kaggle can trust the scorecard.
- **Animation T-stack surprise.** Phase 0a §4.5 + Phase 0c OQ7: frame returned from `env.step()` is `(T, 64, 64)` with T up to 404. Agent code must reduce to fixed shape; the addressed reduction is `(first, last, max-abs-diff)` per Phase 0c.
- **MAX_ACTIONS = 80.** ARC-AGI-3-Agents base default is 80. Way too low. Override to `5 * sum(baseline_actions)` per env or the agent terminates before the 5× cap is reached.
- **Stub-name confusion.** Team names on the LB can collide; verify via team_id.

---

## 10. Quick reference — key numbers

| Item | Value | Source |
|---|---|---|
| Milestone #1 deadline | 2026-06-30 | Kaggle Timeline page |
| Milestone #2 deadline | 2026-09-30 | Kaggle Prizes page |
| Final submission | 2026-11-02 | Kaggle Timeline page |
| Submissions per day | 1 | Kaggle UI |
| Runtime per submission | 9 h | Kaggle Code Requirements |
| Eval set | 110 private envs (55 + 55) | Kaggle Data-description |
| Score units | 0–115 (Kaggle LB displays raw) | `arc_agi/scorecard.py:170` |
| Local gate | `harness_score_holdout ≥ 10.0` | Phase 0c §2 |
| Top-3 today | ~0.66–0.68 | Phase 0b LB pull |
| Stretch target | 0.85–1.20 (safe podium under ±0.2 variance) | Phase 0c §3.4 |
| VRAM ceiling (local) | 12 GB (RTX 5070 Ti) | Adarsh's hardware |
| VRAM available (Kaggle) | 48 GB (g4-standard-48) | Kaggle Upgraded Accelerators |
| Public-replay dataset | 180,144 transitions, 339 replays | `data/bc_transitions_v2.npz` |
| Holdout envs | vc33, tu93, sk48, lp85, dc22 | Phase 0c §2.1 |

---

## 11. Onboarding sequence for a new Claude Code session

1. Read this `CLAUDE.md`.
2. Read the latest of: `research-findings.md`, `validation-results.md`, `phase-0c-spec.md`, `phase-1-bootstrap.md`.
3. Read `git log --oneline -30` (once a repo exists) to see recent changes and submissions.
4. Check `mcp__kaggle__search_competition_submissions` for today's submission count.
5. Confirm working directory is the project root (`C:\Users\adars\Downloads\ARC-AGI-3\` or `/workspaces/arc-agi-3-agent` once Phase 1 lands).
6. Confirm `.venv313/Scripts/python` (Windows) or equivalent is the venv.
7. Ask the user what they want to do before initiating any submission-path tool call.

---

**End of CLAUDE.md.** All operational rules in this document carry into every future session. If a rule needs to change, the user updates this file directly; do not silently override.
