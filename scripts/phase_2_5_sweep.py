"""Phase 2.5 inference-policy sweep. Runs harness across multiple EliteV0 configs.

Logs per-config harness_score_train/holdout to phase-2.5-log.md (markdown table).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "noop")

from arc_agi_3_agent.eval.harness import (
    DEFAULT_ENV_DIR,
    GATE_HOLDOUT_THRESHOLD,
    _get_arcade,
    run_one_env,
)
from arc_agi_3_agent.eval.scoring import total_score
from arc_agi_3_agent.eval.splits import ALL_PUBLIC_ENV_IDS, is_holdout

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_MD = REPO_ROOT / "phase-2.5-log.md"


CONFIGS: list[dict] = [
    {"name": "argmax_baseline", "kwargs": {}},
    # Experiment 1 — action-type temperature sweep
    {"name": "T0.5", "kwargs": {"temperature": 0.5}},
    {"name": "T1.0", "kwargs": {"temperature": 1.0}},
    {"name": "T1.5", "kwargs": {"temperature": 1.5}},
    {"name": "T2.0", "kwargs": {"temperature": 2.0}},
    # Experiment 2 — spatial top-k sampling on top of best T (0.5)
    {"name": "T0.5_topk5", "kwargs": {"temperature": 0.5, "spatial_topk": 5}},
    {"name": "T0.5_topk20", "kwargs": {"temperature": 0.5, "spatial_topk": 20}},
    {"name": "T0.5_topk50", "kwargs": {"temperature": 0.5, "spatial_topk": 50}},
    # Experiment 3 — L1 framechange filter on top of best Exp2 (topk5)
    {"name": "T0.5_topk5_fc", "kwargs": {"temperature": 0.5, "spatial_topk": 5, "framechange_filter": True}},
    # Experiment 4 — stuck-state escape
    {"name": "T0.5_topk5_stuck8", "kwargs": {"temperature": 0.5, "spatial_topk": 5, "stuck_detector_K": 8}},
    {"name": "T0.5_topk5_fc_stuck8", "kwargs": {"temperature": 0.5, "spatial_topk": 5, "framechange_filter": True, "stuck_detector_K": 8}},
]


def _patched_run_one_env(arc, env_id, env_info, agent, log_path, run_id):
    return run_one_env(arc, env_id, env_info, agent, log_path, run_id)


def run_config(name: str, kwargs: dict, env_dir: str, out_root: Path) -> dict:
    from arc_agi_3_agent.agent.elite_v0 import EliteV0

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_env").mkdir(exist_ok=True)

    arc = _get_arcade(env_dir)
    agent = EliteV0(**kwargs)

    envs_info = list(arc.get_environments())
    by_base = {e.game_id.split("-")[0]: e for e in envs_info}

    per_env: list[dict] = []
    t_start = time.perf_counter()
    for base in ALL_PUBLIC_ENV_IDS:
        if base not in by_base:
            continue
        info = by_base[base]
        log_path = out_dir / "per_env" / f"{base}.jsonl"
        r = _patched_run_one_env(arc, info.game_id, info, agent, log_path, 0)
        r["holdout"] = is_holdout(base)
        per_env.append(r)

    total_wall = time.perf_counter() - t_start
    train_scores = [r["score"] for r in per_env if not r["holdout"]]
    holdout_scores = [r["score"] for r in per_env if r["holdout"]]
    train_score = total_score(train_scores)
    holdout_score = total_score(holdout_scores)

    n_train_nonzero = sum(1 for s in train_scores if s > 0)
    n_holdout_nonzero = sum(1 for s in holdout_scores if s > 0)

    summary = {
        "name": name,
        "kwargs": {k: v for k, v in kwargs.items()},
        "harness_score_train": train_score,
        "harness_score_holdout": holdout_score,
        "n_train_nonzero": n_train_nonzero,
        "n_holdout_nonzero": n_holdout_nonzero,
        "wall_seconds": round(total_wall, 1),
        "per_env": [
            {"env": r["env_id"].split("-")[0], "score": r["score"], "lvls": r["levels_completed"],
             "actions": r["actions"], "holdout": r["holdout"]}
            for r in per_env
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{name}] train={train_score:.3f} ({n_train_nonzero}/20 nz) "
        f"holdout={holdout_score:.3f} ({n_holdout_nonzero}/5 nz) wall={total_wall:.1f}s",
        flush=True,
    )
    return summary


def append_md_log(rows: list[dict], header_note: str = "") -> None:
    lines = []
    if not LOG_MD.exists():
        lines.append("# Phase 2.5 Inference-Policy Sweep Log\n")
    lines.append(f"\n## Run {datetime.now().isoformat(timespec='seconds')}\n")
    if header_note:
        lines.append(f"_{header_note}_\n")
    lines.append("\n| config | kwargs | train | holdout | nz train | nz holdout | wall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        kw = ", ".join(f"{k}={v}" for k, v in r["kwargs"].items()) or "—"
        lines.append(
            f"| {r['name']} | `{kw}` | {r['harness_score_train']:.3f} | "
            f"**{r['harness_score_holdout']:.3f}** | "
            f"{r['n_train_nonzero']}/20 | {r['n_holdout_nonzero']}/5 | "
            f"{r['wall_seconds']}s |"
        )
    lines.append("")
    with LOG_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "harness_runs" / "phase_2_5"))
    ap.add_argument("--configs", default=None, help="comma-separated config names; default all")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_names = args.configs.split(",") if args.configs else None
    configs = [c for c in CONFIGS if (selected_names is None or c["name"] in selected_names)]
    print(f"Running {len(configs)} configs.", flush=True)

    results: list[dict] = []
    for cfg in configs:
        r = run_config(cfg["name"], cfg["kwargs"], args.env_dir, out_root)
        results.append(r)

    append_md_log(results, header_note=args.note)
    print(f"\nWrote {LOG_MD}", flush=True)

    best = max(results, key=lambda r: r["harness_score_holdout"])
    print(f"\nBEST: {best['name']}  holdout={best['harness_score_holdout']:.4f}  "
          f"gate_pass={best['harness_score_holdout'] >= GATE_HOLDOUT_THRESHOLD}")


if __name__ == "__main__":
    main()
