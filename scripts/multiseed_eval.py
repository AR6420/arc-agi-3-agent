"""Multi-seed harness driver (Phase 3 v2).

Runs an agent over several run_ids (now process-stable seeds — see seeding.py) and
aggregates per-env + holdout/train score as a DISTRIBUTION (mean/std/min/max). Needed
because biased-random has large cross-seed variance (a lucky fast level-1 can score >4),
so a single seed is not an honest floor.

Usage:
    python scripts/multiseed_eval.py --agent random --seeds 0,1,2,3,4 --out harness_runs/p3v2_b2_random_ms
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "noop")

from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, run_harness
from arc_agi_3_agent.eval.splits import is_holdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="random")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for sd in seeds:
        print(f"\n########## {args.agent} seed/run_id={sd} ##########", flush=True)
        summary = run_harness(args.env_dir, args.agent, out_root / f"seed_{sd}", run_id=sd)
        runs.append(summary)

    # Aggregate per-env across seeds.
    env_ids = [r["env_id"] for r in runs[0]["per_env"]]
    per_env_agg = []
    for i, eid in enumerate(env_ids):
        base = eid.split("-")[0]
        scores = [r["per_env"][i]["score"] for r in runs]
        lvls = [r["per_env"][i]["levels_completed"] for r in runs]
        deaths = [r["per_env"][i].get("n_deaths", 0) for r in runs]
        fdeath = [r["per_env"][i].get("first_death_step") for r in runs]
        nlev = runs[0]["per_env"][i]["n_levels"]
        l1b = runs[0]["per_env"][i].get("level1_baseline")
        per_env_agg.append({
            "env_id": base, "holdout": is_holdout(base), "n_levels": nlev,
            "level1_baseline": l1b,
            "score_mean": round(statistics.mean(scores), 4),
            "score_std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "score_min": round(min(scores), 4), "score_max": round(max(scores), 4),
            "scores": [round(s, 4) for s in scores],
            "levels": lvls, "levels_max": max(lvls),
            "reached_l1_in_n_seeds": sum(1 for x in lvls if x >= 1),
            "deaths": deaths,
            "first_death_steps": fdeath,
        })

    h_train = [r["harness_score_train"] for r in runs]
    h_hold = [r["harness_score_holdout"] for r in runs]
    agg = {
        "agent": args.agent, "seeds": seeds,
        "train_mean": round(statistics.mean(h_train), 4),
        "train_std": round(statistics.pstdev(h_train), 4) if len(h_train) > 1 else 0.0,
        "train_per_seed": [round(x, 4) for x in h_train],
        "holdout_mean": round(statistics.mean(h_hold), 4),
        "holdout_std": round(statistics.pstdev(h_hold), 4) if len(h_hold) > 1 else 0.0,
        "holdout_per_seed": [round(x, 4) for x in h_hold],
        "per_env": per_env_agg,
    }
    (out_root / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print("\n================ MULTISEED AGGREGATE ================")
    print(f"agent={args.agent} seeds={seeds}")
    print(f"train   mean={agg['train_mean']:.4f} std={agg['train_std']:.4f} per_seed={agg['train_per_seed']}")
    print(f"holdout mean={agg['holdout_mean']:.4f} std={agg['holdout_std']:.4f} per_seed={agg['holdout_per_seed']}")
    print(f"\n{'env':6}{'H':2}{'sc_mean':>8}{'sc_std':>7}{'sc_max':>7}{'L1/seeds':>9}{'Lmax':>5}  scores")
    for e in per_env_agg:
        print(f"{e['env_id']:6}{'H' if e['holdout'] else '.':2}"
              f"{e['score_mean']:>8.3f}{e['score_std']:>7.3f}{e['score_max']:>7.3f}"
              f"{e['reached_l1_in_n_seeds']:>4}/{len(seeds):<4}{e['levels_max']:>5}  {e['scores']}")


if __name__ == "__main__":
    main()
