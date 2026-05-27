# Phase 1 — Repo Bootstrap Specification

**Author:** Phase 0c spec output — Phase 1 picks this up unchanged.
**Date:** 2026-05-27.
**Scope:** Define the exact repo structure, first-commit contents, and bootstrap operations Phase 1 executes. **No code, no commits, no GitHub pushes in this document — Phase 1 owns execution.**
**Reads first:** `CLAUDE.md`, `phase-0c-spec.md`, `research-findings.md`, `validation-results.md`.

---

## 1. Repo name

**`arc-agi-3-agent`** (lowercase, dash-separated). Rationale: neutral, descriptive, no clever expectations baked in. Avoids over-promising before any submission has scored above the LB median.

GitHub owner: `AdarshReddy0099` (the user's Hugging Face / inferred Kaggle handle). Confirm via `gh auth status` at bootstrap time.

---

## 2. Directory layout

```
arc-agi-3-agent/
├── CLAUDE.md                          # operating rules (from Phase 0c)
├── README.md                          # public-facing project description, CC-BY 4.0 attribution
├── LICENSE                            # CC-BY 4.0 full text (winners license per Kaggle Rules §1.5.a)
├── pyproject.toml                     # pinned dependencies; Python 3.12 target
├── uv.lock                            # OR poetry.lock — pick uv to match the bundled ARC-AGI-3-Agents convention
├── .gitignore                         # excludes data/, weights/, .venv*, .mcp.json, etc.
├── .mcp.json                          # (NOT committed — in .gitignore)
├── docs/
│   ├── research-findings.md           # Phase 0a (copied from project root at bootstrap)
│   ├── validation-results.md          # Phase 0b
│   ├── phase-0c-spec.md               # architecture decision + harness + training spec
│   └── phase-1-bootstrap.md           # this file
├── src/
│   └── arc_agi_3_agent/
│       ├── __init__.py
│       ├── agents/                    # agent classes (Phase 2)
│       │   ├── __init__.py
│       │   ├── base.py                # Agent base class (override of arc-agi-3-agents/agents/agent.py)
│       │   ├── biased_random.py       # S1 / S3 plumbing-validation agent
│       │   ├── marginal_bc.py         # S2 state-blind marginal-action agent
│       │   └── e_lite.py              # S4 main learned agent (Phase 2)
│       ├── data/
│       │   ├── replay_loader.py       # load bc_transitions_v2.npz; train/holdout split
│       │   ├── synth_rollouts.py      # generate synthetic rollouts from public env source (Phase 2)
│       │   └── perception_input.py    # (3, 64, 64) reduction from (T, 64, 64) per Phase 0c OQ7
│       ├── models/                    # NN modules (Phase 2)
│       │   ├── __init__.py
│       │   ├── backbone.py            # ResNet-style perception backbone
│       │   ├── heads.py               # action-type head, ACTION6 spatial head, frame-change aux
│       │   └── cluster_prior.py       # per-action-signature categorical fallback
│       ├── train/                     # training scripts (Phase 2)
│       │   ├── __init__.py
│       │   ├── stage0_data_prep.py
│       │   ├── stage1_backbone.py
│       │   ├── stage2_framechange.py
│       │   └── stage3_cluster_prior.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── harness.py             # local eval harness — Phase 0c §2
│       │   ├── score.py               # RHAE re-implementation, unit-tested against arc_agi
│       │   └── split.py               # holdout vs train env split (vc33/tu93/sk48/lp85/dc22 holdout)
│       └── submit/
│           ├── __init__.py
│           ├── notebook_template.py   # Kaggle submission notebook generator
│           └── precheck.py            # pre-submission gate checks (CLAUDE.md §1.2)
├── tests/                             # pytest suite
│   ├── __init__.py
│   ├── test_score.py                  # RHAE re-impl matches arc_agi.scorecard
│   ├── test_perception_input.py       # (T,64,64) → (3,64,64) reduction is invariant + reversible-enough
│   ├── test_replay_loader.py          # npz integrity, action histogram matches expectations
│   └── test_harness.py                # harness can run a trivial agent end-to-end on 1 env
├── scripts/
│   ├── validation/                    # Phase 0b scripts (copied at bootstrap)
│   │   ├── replay_parser_v2.py
│   │   ├── env_source_audit.py
│   │   ├── engine_stress_test.py
│   │   ├── replay_integrity.py
│   │   ├── visibility_audit.py
│   │   ├── toy_bc_smoke.py
│   │   ├── smoke_offline.py
│   │   └── ... (json/csv/txt outputs are gitignored)
│   ├── make_kaggle_notebook.py        # generate the submission notebook from src/
│   ├── upload_weights_dataset.py      # push trained weights to Kaggle as a private dataset
│   └── pre_submit_check.py            # standalone CLI for the §1.3 gate sequence
├── data/                              # NOT committed (in .gitignore)
│   ├── bc_transitions.npz             # v1 (kept for inverse-model aux)
│   ├── bc_transitions_v2.npz          # primary forward-BC dataset
│   ├── bc_transitions_meta.json
│   ├── bc_transitions_v2_meta.json
│   └── synth_rollouts/                # Phase 2 — generated from public env source
├── weights/                           # NOT committed (in .gitignore)
│   └── ...                            # checkpoints organized per training run
├── runs/                              # NOT committed
│   └── <timestamp>/config.json + logs
└── harness_runs/                      # NOT committed
    └── <timestamp>/summary.json + per_env/*.jsonl
```

Rationale notes:
- **`src/arc_agi_3_agent/`** is a proper installable package (`pip install -e .`). The agent code, eval, and submit modules are importable from the Kaggle submission notebook (which copies the relevant subpackage into `/kaggle/working/` at the top of the notebook).
- **`scripts/validation/` preserved** with original Phase 0b artifacts. These are research, not production — moved as-is. New Phase-1+ scripts live at `scripts/` top level.
- **`data/`, `weights/`, `runs/`, `harness_runs/`** are gitignored. They're machine-generated and large. The canonical inputs (replays NPZ) are reproducible from `scripts/validation/replay_parser_v2.py` + the human-replays data the user already has at `data/human_replays/`.
- **`docs/`** holds all phase docs at commit time. Future Claude Code sessions read these first.
- **`tests/`** is small but real. Phase 1 commits enough tests to verify the harness can score a trivial agent end-to-end before any real architecture work begins.

---

## 3. Top-level file contents at bootstrap

### 3.1 `pyproject.toml`

```toml
[project]
name = "arc-agi-3-agent"
version = "0.1.0"
description = "ARC Prize 2026 — ARC-AGI-3 Kaggle competition agent"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "CC-BY-4.0" }
authors = [{ name = "Adarsh Reddy", email = "adarshreddybms@gmail.com" }]

dependencies = [
    "arc-agi==0.9.8",
    "arcengine==0.9.3",
    "numpy==2.4.4",
    "pydantic==2.13.2",
    "pillow==12.2.0",
    "matplotlib==3.10.8",
    "torch==2.12.0",
    "python-dotenv>=1.2.2",
]

[dependency-groups]
dev = [
    "pytest>=8.4",
    "pytest-cov>=5.0",
    "ruff>=0.11",
    "mypy>=1.15",
    "pre-commit>=4.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

Pinned exactly to Kaggle's bundled-wheel versions (validation-results §4.1) to guarantee local-Kaggle parity.

### 3.2 `README.md` (commit-0 stub)

Contains:
- One-paragraph project description (focus on the benchmark, not on our cleverness).
- Status badge / current Kaggle LB position (manually updated; no CI hook in Phase 1).
- "License: [CC-BY 4.0](LICENSE)" line.
- Quickstart for running the local harness (`python -m arc_agi_3_agent.eval.harness`).
- Pointer to `docs/` for the phase docs.
- Acknowledgments: ARC Prize Foundation, the upstream `ARC-AGI-3-Agents` repo (Apache 2.0 attribution), the human-replays dataset.
- "Read everything, copy nothing" credo line — explicitly cites no other competitor's code is in this repo.

### 3.3 `LICENSE`

Full text of [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode.txt). This is the Kaggle Rules §1.5.a winners license. Mandatory for prize eligibility.

### 3.4 `.gitignore`

Already exists at project root (Phase 0b addendum). Phase 1 copies and extends to also cover `weights/`, `runs/`, `harness_runs/`, `data/synth_rollouts/`, plus Python build artifacts.

### 3.5 `CLAUDE.md`

Copied verbatim from project root at bootstrap.

---

## 4. First commit (commit 0) — exact contents

**Branch:** `main`.
**Message:** `chore: initial repo bootstrap (Phase 0a + 0b + 0c artifacts)`.
**Author:** Adarsh Reddy <adarshreddybms@gmail.com>.

**Files included:**

1. `CLAUDE.md`
2. `README.md` (commit-0 stub per §3.2)
3. `LICENSE` (CC-BY 4.0 full text)
4. `pyproject.toml` (per §3.1)
5. `.gitignore` (extended)
6. `docs/research-findings.md`
7. `docs/validation-results.md`
8. `docs/phase-0c-spec.md`
9. `docs/phase-1-bootstrap.md`
10. `src/arc_agi_3_agent/__init__.py` (empty, with version string)
11. `scripts/validation/replay_parser_v2.py`
12. `scripts/validation/env_source_audit.py`
13. `scripts/validation/engine_stress_test.py`
14. `scripts/validation/replay_integrity.py`
15. `scripts/validation/visibility_audit.py`
16. `scripts/validation/toy_bc_smoke.py`
17. `scripts/validation/smoke_offline.py`

**Files NOT included in commit 0** (gitignored or pending Phase 2):
- `data/bc_transitions*.npz` (too large; reproducible from `replay_parser_v2.py`)
- `.mcp.json` (contains Kaggle Bearer token)
- Any weights, any harness run outputs, any model code (Phase 2).

**Estimated commit-0 size:** ~250 KB total (docs are the bulk).

---

## 5. GitHub repo settings

At repo creation (Phase 1, executed via `gh repo create`):

- **Visibility:** public. CLAUDE.md §5.3 mandates open-source-by-Milestone-#1 anyway; might as well be public from day 0.
- **License:** CC-BY 4.0 (matches `LICENSE` file content; GitHub picks this up automatically).
- **Description:** "ARC Prize 2026 — ARC-AGI-3 Kaggle competition agent. Modular system: BC backbone + frame-change auxiliary + cluster-prior fallback. CC-BY 4.0."
- **Topics:** `kaggle`, `arc-prize`, `arc-agi-3`, `behavioral-cloning`, `agent`.
- **Default branch:** `main`.
- **Branch protection** on `main`: not in Phase 1 (one-person project; protection adds friction without value). Add in Phase 2 if/when a teammate is added.
- **Issues:** enabled. Future bug-tracking lives here.
- **Discussions:** disabled (Kaggle discussions are the canonical forum for this competition).
- **Wiki:** disabled (docs live in `docs/` in repo).
- **Actions:** disabled in Phase 1 (no CI yet). Phase 2 may add a single workflow that runs `pytest` + `ruff check` on PR.

`gh repo create` command:

```bash
gh repo create AdarshReddy0099/arc-agi-3-agent \
    --public \
    --license cc-by-4.0 \
    --description "ARC Prize 2026 — ARC-AGI-3 Kaggle competition agent..." \
    --homepage "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3"
```

Then `git remote add origin` + initial push.

---

## 6. CI/CD policy

**Phase 1: deferred.** Manual local testing only.

**Phase 2 candidate:** single GitHub Actions workflow `.github/workflows/ci.yml`:
- Triggered on push to any branch.
- Runs `ruff check src/ tests/` and `pytest tests/`.
- Caches the venv between runs (Python 3.12, pinned deps).
- Does NOT run the full harness (too slow for CI; takes 20+ minutes).
- Does NOT run training (way too slow; needs GPU).
- Does NOT auto-submit to Kaggle (gated by §1.3 of CLAUDE.md regardless).

**Phase 3 candidate:** nightly cron that runs the full harness on the latest weights and posts the score to a private channel. Optional.

---

## 7. Phase 1 execution sequence

Phase 1 (Claude Code session, after user authorization to proceed) executes in this order:

1. **`gh auth status`** — verify user is logged in.
2. **`gh repo create`** per §5 — produces a public empty repo.
3. **Local `git init`** in `C:\Users\adars\Downloads\ARC-AGI-3\`. Or — easier — bootstrap in a fresh sister directory `C:\Users\adars\Downloads\arc-agi-3-agent\` and copy files in cleanly. Sister directory is the recommended path (no leakage of project-root throwaway files).
4. **Copy files** per §4 list into sister directory.
5. **`pip install -e .`** to verify the package is installable. Run `pytest tests/` (no tests yet, expect "no tests ran"). Run `ruff check src/`.
6. **First commit** per §4.
7. **`git push -u origin main`** — publishes commit 0.
8. **Create Kaggle weights dataset shell**: `kaggle datasets init -p ./weights/` + edit `dataset-metadata.json` + `kaggle datasets create`. Empty dataset for now; Phase 2 fills it.
9. **Write the S1 biased-random agent** in `src/arc_agi_3_agent/agents/biased_random.py`. Single file, ~150 lines. Subclasses `Agent` from the bundled `arc-agi-3-agents` package; overrides `MAX_ACTIONS`.
10. **Write the harness skeleton** in `src/arc_agi_3_agent/eval/harness.py` — runs the agent on the 25 envs (using local OFFLINE mode), computes RHAE per env, prints holdout summary.
11. **Run the harness on S1 (biased-random)** — establishes the local-OFFLINE floor score.
12. **Write the Kaggle notebook template** (per CLAUDE.md §2) — embeds the S1 agent, sets `ARC_API_KEY="noop"`, installs from bundled wheels, runs against all 25 (well, all 110 once on Kaggle), closes scorecard.
13. **Verify the notebook locally** via the parity test approach (or trust the Phase 0b parity test). Do NOT submit yet.
14. **STOP and report back to user.** Phase 1 ends here. Submission S1 happens only after user authorization in a subsequent message.

---

## 8. Out-of-scope for Phase 1

Explicitly Phase 2+ work; do NOT attempt in Phase 1 even if "looks easy":

- Training any neural network.
- Implementing the perception backbone, action-type head, spatial head, frame-change head.
- Generating synthetic rollouts.
- Uploading actual weights (the dataset is a shell only at end of Phase 1).
- Implementing the precise variance-confirmation logic (Phase 2 — once we have S1's actual score).
- Adding CI workflows.
- Adding pre-commit hooks beyond ruff format-check.
- Refactoring `scripts/validation/` (they're frozen research artifacts).

---

## 9. End of Phase 1 — handoff to Phase 2

When Phase 1 ends:

- Repo at `github.com/AdarshReddy0099/arc-agi-3-agent` exists, public, CC-BY 4.0, commit 0 pushed.
- S1 biased-random agent runs locally; harness reports a floor score.
- Kaggle notebook template generated; ready for user authorization to S1-submit.
- Phase 2 spec is implicit in `phase-0c-spec.md` §4 (training pipeline) — Phase 2 starts by implementing Stage 0 (data prep) and works through Stages 1–3.

**The Phase 1 → Phase 2 boundary is the first Kaggle submission (S1).** Phase 2 doesn't depend on S1's score landing — it can start in parallel — but the calendar discipline is: S1 must be submitted within 2 calendar days of Phase 1 completion or we are behind the submissions schedule in Phase 0c §3.1.

---

**Status: Phase 0c bootstrap spec complete. Phase 1 execution awaits user authorization.**
