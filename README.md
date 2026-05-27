# arc-agi-3-agent

ARC Prize 2026 — ARC-AGI-3 Kaggle competition agent.

**Status:** Phase 1 (foundation build). Pre-S1. Plumbing-validation submission pending.

## What this is

A modular agent for the ARC-AGI-3 benchmark: 110 unseen 64×64 grid environments where the agent must infer mechanics, goals, and win-conditions through interaction alone. No instructions, no natural-language interface.

Architecture target (per `docs/phase-0c-spec.md`): Approach E-lite. ResNet-style perception backbone + behavioral-cloning action head + frame-change auxiliary head + cluster-level action priors for novelty fallback. Behavioral-cloning pretrain on 180,144 human-replay transitions; no inference-time replay retrieval. Designed to fit a 12 GB local-dev VRAM ceiling with sub-200ms per-action latency on Kaggle's 9-hour budget.

## Quickstart

```bash
# Python 3.12 required (matches Kaggle bundled wheels: cp312-manylinux)
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -e ".[dev]"

# Re-parse human replays into v3 NPZ (forward-BC, OQ7-correct perception input)
python scripts/data_prep/replay_parser_v3.py

# Run local eval harness with the S1 plumbing agent
python -m arc_agi_3_agent.eval.harness --agent random --output harness_runs/$(date +%Y%m%d_%H%M%S)
```

## Repo layout

```
docs/                       — phase docs (Phase 0a–0c, this Phase 1)
src/arc_agi_3_agent/        — installable package
  agents/                   — agent classes (S1 random, future learned)
  data/                     — replay loaders, perception encoding
  eval/                     — local harness + RHAE scoring
  submit/                   — Kaggle notebook generator
scripts/
  validation/               — Phase 0b research scripts (frozen)
  data_prep/                — Phase 1+ data pipelines
tests/                      — pytest suite (correctness gates)
notebooks/                  — Kaggle submission notebooks
data/                       — replays + v3 NPZ (gitignored, regenerable)
```

## Submission discipline

- **1 submission/day** (Kaggle UI-enforced, not the rules-text 5/day boilerplate).
- **Local gate:** no Kaggle submission until `harness_score_holdout ≥ 10.0` on 5 held-out public envs (`vc33`, `tu93`, `sk48`, `lp85`, `dc22`).
- **Authorization:** every `kaggle competitions submit` requires explicit per-message user authorization. Past-conversation authorization does not count.

See `CLAUDE.md` for the full operational rules.

## License

This project is licensed under **CC-BY 4.0** (the ARC Prize 2026 winners license per Kaggle Rules §1.5.a).

## Acknowledgments

- [ARC Prize Foundation](https://arcprize.org/) for the benchmark, the toolkit (`arc-agi`, `arcengine`), and the human-replays dataset.
- Upstream [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) (Apache 2.0) — base `Agent` class pattern and reference reading.

This repo contains no code copied from other competitors' implementations. Read everything, copy nothing.
