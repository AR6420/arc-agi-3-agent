"""S1 Kaggle submission notebook — biased-random plumbing validator.

Phase 1 — canonical Kaggle submission pattern adapted from
Jeroen Cottaar's "Simplified submission approach" (88 votes, Kaggle Topic 686416).

KEY ARCHITECTURAL POINT (Phase 1 finding, supersedes Phase 0c §5.1):
- COMPETITION-mode Arcade is fundamentally an *online* protocol — it requires a
  reachable arc_agi HTTP server. During real Kaggle competition reruns, the
  grading infrastructure provides a server at `http://gateway:8001/`. During
  Save & Run All (internet OFF), there is no such gateway → use OFFLINE mode
  against the locally bundled env files.
- Branch on env var `KAGGLE_IS_COMPETITION_RERUN`:
    - set → OPERATION_MODE=online, host=gateway, real scoring path.
    - unset → OPERATION_MODE=offline, ENVIRONMENTS_DIR=/kaggle/input/.../environment_files/.
- The ARC_API_KEY S9 workaround is still required (any non-empty string).
- A submission.parquet file at /kaggle/working/ is REQUIRED for Kaggle to
  recognize the submission as valid, even though real scoring uses the gateway's
  internal scorecard, not the parquet.

S1 purpose: validate Kaggle wiring + plumbing. Score floor establishment.
Resolves Phase 1 Issues 2 (RESET counter) + 3 (ACTION7 cost) post-submission.

DO NOT submit without explicit user authorization per CLAUDE.md §1.3.
"""

# %% Cell 1 — Install bundled wheels (no internet; --no-index mandatory)
import subprocess
import sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
     "--find-links",
     "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
     "arc-agi", "python-dotenv"],
    check=True,
)
print("Installed arc-agi + python-dotenv from bundled wheels.")


# %% Cell 2 — Imports + agent definition
import os
import json
import time
import random
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

AGENT_VERSION = "s1_random_agent_v1"
MAX_MOVES_HARD_CAP = 2000  # Per-env hard cap; complemented by 5*sum(baseline)+50 soft cap.


def per_env_seed(env_id: str, run_id: int = 0) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{env_id}|{run_id}|{AGENT_VERSION}".encode()).digest()[:4],
        "big",
    )


class BiasedRandomAgent:
    """Stateless agent rebound per env via reset_for_env().

    For ACTION6, picks a uniformly random color present in the current frame
    then a uniformly random pixel of that color (Jeroen Cottaar's pattern).
    For other actions, samples uniformly from `available_actions` minus RESET.
    """

    def __init__(self) -> None:
        self.rng: random.Random | None = None
        self.np_rng: np.random.Generator | None = None
        self.available_actions: list[int] = []

    def reset_for_env(self, env_id: str, available_actions: list[int], run_id: int = 0) -> None:
        seed = per_env_seed(env_id, run_id)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # Strip RESET (id=0) from voluntary choice — let terminal handling emit it.
        self.available_actions = [a for a in available_actions if a != 0]
        if not self.available_actions:
            self.available_actions = [1]

    def choose_action(self, frame_last) -> tuple[int, dict]:
        assert self.rng is not None
        action_id = self.rng.choice(self.available_actions)
        if action_id == 6:
            return action_id, self._sample_click_xy(frame_last)
        return action_id, {}

    def _sample_click_xy(self, frame_last) -> dict:
        assert self.np_rng is not None and self.rng is not None
        try:
            arr = np.asarray(frame_last, dtype=np.int16)
            if arr.ndim == 3:
                arr = arr[-1]
            colors = np.unique(arr).tolist()
            color = self.rng.choice(colors)
            ys, xs = np.where(arr == color)
            if len(xs) > 0:
                idx = self.rng.randint(0, len(xs) - 1)
                return {"x": int(xs[idx]), "y": int(ys[idx])}
        except Exception:
            pass
        return {"x": int(self.np_rng.integers(0, 64)), "y": int(self.np_rng.integers(0, 64))}


def max_actions_for_env(baseline_actions: list[int]) -> int:
    """Phase 0c §3.1: MAX_ACTIONS = 5 * sum(baseline) + 50, hard-capped at MAX_MOVES_HARD_CAP."""
    soft = 5 * sum(baseline_actions) + 50
    return min(soft, MAX_MOVES_HARD_CAP)


# %% Cell 3 — Branch: real-submission gateway vs Save&Run offline
# Per Jeroen Cottaar's canonical pattern (Topic 686416).
IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
print(f"KAGGLE_IS_COMPETITION_RERUN: {IS_RERUN}")

if IS_RERUN:
    # Real submission grading: Kaggle starts an arc_agi server at gateway:8001.
    # Wait for it to be reachable before proceeding.
    print("Waiting for gateway:8001 ...")
    rc = subprocess.run(
        ["curl", "--fail", "--retry", "999", "--retry-all-errors",
         "--retry-delay", "5", "--retry-max-time", "600",
         "http://gateway:8001/api/games"],
        check=False,
    )
    print(f"Gateway curl rc={rc.returncode}")

    env_content = """SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
"""
else:
    # Save & Run All — no gateway. Use OFFLINE mode against bundled env files.
    env_content = """SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=offline
ENVIRONMENTS_DIR=/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files/
RECORDINGS_DIR=/kaggle/working/server_recording
"""

with open("/kaggle/working/.env", "w") as f:
    f.write(env_content)

import dotenv
dotenv.load_dotenv(dotenv_path="/kaggle/working/.env", override=True)

import arc_agi
from arcengine import GameAction, GameState

arcade = arc_agi.Arcade()
print(f"Arcade initialized. Operation mode env={os.environ.get('OPERATION_MODE')}")

envs = list(arcade.available_environments)
print(f"available_environments → {len(envs)} envs")


# %% Cell 4 — Main eval loop (one make() per env, one trajectory each)
RUN_LOG: list[dict] = []

agent = BiasedRandomAgent()
t_global = time.perf_counter()

for env_info in envs:
    full_id = env_info.game_id
    base_id = full_id.split("-")[0]
    baseline_actions = list(env_info.baseline_actions or [])
    max_actions = max_actions_for_env(baseline_actions) if baseline_actions else 250

    env = arcade.make(full_id)
    # `arcade.make()` already invokes reset internally; initial obs is in env._last_response.
    response = env._last_response
    if response is None:
        RUN_LOG.append({"env": base_id, "error": "no initial response"})
        continue

    avail = list(response.available_actions or [])
    agent.reset_for_env(full_id, avail)

    actions_taken = 0
    levels_seen = int(getattr(response, "levels_completed", 0))
    level_actions_local = [0] * len(baseline_actions)
    action7_count = 0
    reset_count = 0
    t_env_start = time.perf_counter()

    while actions_taken < max_actions:
        state = response.state
        if state == GameState.WIN:
            break

        # Reset on GAME_OVER / NOT_PLAYED — but track that we issued a RESET.
        if state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            response = env.step(GameAction.RESET, {})
            actions_taken += 1
            reset_count += 1
            if response is None:
                break
            continue

        last = response.frame[-1] if isinstance(response.frame, list) and response.frame else response.frame
        action_id, data = agent.choose_action(last)
        ga = GameAction.from_id(action_id)
        response = env.step(ga, data)
        if response is None:
            break
        actions_taken += 1
        if action_id == 7:
            action7_count += 1

        cur = int(getattr(response, "levels_completed", 0))
        if cur < len(baseline_actions):
            level_actions_local[cur] += 1
        if cur > levels_seen:
            levels_seen = cur

    env_wall = time.perf_counter() - t_env_start
    final_state = response.state.name if response else "unknown"
    RUN_LOG.append({
        "env": base_id,
        "full_id": full_id,
        "actions_taken": actions_taken,
        "levels_seen": levels_seen,
        "n_levels": len(baseline_actions),
        "action7_count": action7_count,
        "reset_count": reset_count,
        "available_actions": avail,
        "final_state": final_state,
        "wall_seconds": round(env_wall, 2),
        "level_actions_local": level_actions_local,
        "baseline_actions": baseline_actions,
    })
    print(f"{base_id}: lvls={levels_seen}/{len(baseline_actions)} "
          f"actions={actions_taken} a7={action7_count} reset={reset_count} "
          f"state={final_state} wall={env_wall:.1f}s")

print(f"\nTotal wall: {time.perf_counter() - t_global:.1f}s")


# %% Cell 5 — Scorecard dump (manual runs only; rerun uses gateway scoring)
if not IS_RERUN:
    scorecard = arcade.get_scorecard()
    print(f"\n{'=' * 60}")
    print(f"Score: {scorecard.score:.4f}")
    print(f"Envs completed: {scorecard.total_environments_completed}/{scorecard.total_environments}")
    print(f"Levels completed: {scorecard.total_levels_completed}/{scorecard.total_levels}")
    print(f"Total actions (scorecard): {scorecard.total_actions}")
    print(f"\n{'Game':<20} {'Score':>8} {'Levels':>10} {'Actions':>10} {'Done':>6}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 6}")
    for env_score in scorecard.environments:
        print(
            f"{env_score.id:<20} {env_score.score:>8.2f} "
            f"{env_score.levels_completed:>10} {env_score.actions:>10} "
            f"{'Y' if env_score.completed else 'N':>6}"
        )

    # Phase 1 Issues 2 + 3 resolution: per-env scorecard.actions / scorecard.resets vs local counts.
    print("\n=== ISSUE 2 (RESET counter) + ISSUE 3 (ACTION7 cost) RESOLUTION ===")
    print("Format: env | sc.actions vs local | sc.resets vs local | local_a7_count")
    for env_score in scorecard.environments:
        base = env_score.id.split("-")[0]
        local = next((r for r in RUN_LOG if r["env"] == base), {})
        print(f"  {base:5} | sc.a={env_score.actions} loc={local.get('actions_taken')} | "
              f"sc.r={getattr(env_score, 'resets', 'N/A')} loc={local.get('reset_count')} | "
              f"a7={local.get('action7_count')}")

    # Persist artifact
    artifact = {
        "agent_version": AGENT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scorecard": json.loads(scorecard.model_dump_json()),
        "run_log": RUN_LOG,
    }
    with open("/kaggle/working/s1_run_artifact.json", "w") as f:
        json.dump(artifact, f, indent=2)
    print("\nWrote /kaggle/working/s1_run_artifact.json")


# %% Cell 6 — Write submission.parquet (REQUIRED by Kaggle even though gateway scores)
submission = pd.DataFrame(
    data=[["1_0", "1", True, 1]],
    columns=["row_id", "game_id", "end_of_game", "score"],
)
submission.to_parquet("/kaggle/working/submission.parquet", index=False)
print("Wrote /kaggle/working/submission.parquet")
