"""Render the OBSERVATION frames an explorer sees, for the blind goal-inferability diagnostic.

Integrity: this only RENDERS frames by running the engine (the observations the agent
receives). It does NOT read any env source. Sample actions are a fixed, env-agnostic
explorer policy (every available action once, then a few generic clicks) — NOT derived from
any known solution. Frames are saved as montages with NEUTRAL labels (Game A/B/C/D) so the
blind reader cannot infer the env identity/archetype.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ARC_API_KEY", "noop")

import numpy as np
from PIL import Image, ImageDraw

from arc_agi_3_agent.eval.harness import DEFAULT_ENV_DIR, _frame_last, _get_arcade

# Canonical-ish ARC 16-colour palette (0=bg black).
PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37),
    (255, 255, 255), (160, 32, 240), (139, 69, 19), (255, 105, 180), (0, 255, 255), (128, 128, 0),
]
CELL = 4          # upscale 64 -> 256
OUT = Path("harness_runs/p3_goal_diag")


def colorize(grid: np.ndarray) -> Image.Image:
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v in range(16):
        rgb[grid == v] = PALETTE[v]
    img = Image.fromarray(rgb, "RGB").resize((w * CELL, h * CELL), Image.NEAREST)
    return img


def montage(frames: list[tuple[str, np.ndarray]], title: str) -> Image.Image:
    cols = 3
    rows = (len(frames) + cols - 1) // cols
    fw, fh = 64 * CELL, 64 * CELL
    cap = 22
    pad = 8
    W = cols * fw + (cols + 1) * pad
    H = rows * (fh + cap) + (rows + 1) * pad + 30
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), title, fill=(255, 255, 255))
    for i, (label, grid) in enumerate(frames):
        r, c = divmod(i, cols)
        x = pad + c * (fw + pad)
        y = 30 + pad + r * (fh + cap + pad)
        canvas.paste(colorize(grid), (x, y))
        d.text((x + 2, y + fh + 2), label, fill=(230, 230, 120))
    return canvas


def blind_explore(env, obs):
    """Fixed env-agnostic explorer: each available non-RESET action once, then generic clicks."""
    from arcengine import GameAction
    avail = [a for a in (obs.available_actions or []) if a != 0]
    frames = [("initial frame (T=%d)" % _tlen(obs), _frame_last(obs.frame))]
    rng = np.random.default_rng(0)

    def step(a, data, label):
        nonlocal obs
        st = str(obs.state)
        if st.endswith("GAME_OVER"):
            obs = env.step(GameAction.from_id(0), data={})        # revive
        obs = env.step(GameAction.from_id(a), data=data)
        frames.append((label, _frame_last(obs.frame)))

    for a in [1, 2, 3, 4, 5]:
        if a in avail:
            step(a, {}, f"after ACTION{a}")
    if 6 in avail:
        g = _frame_last(obs.frame)
        nz = np.argwhere(g != 0)
        pts = [(32, 32)]
        if len(nz):
            for _ in range(2):
                yx = nz[int(rng.integers(0, len(nz)))]
                pts.append((int(yx[1]), int(yx[0])))      # (x=col, y=row)
        for (x, y) in pts:
            step(6, {"x": x, "y": y}, f"after CLICK ({x},{y})")
    return frames


def _tlen(obs) -> int:
    f = getattr(obs, "frame", None)
    return len(f) if isinstance(f, list) else 1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # env -> neutral label (mapping kept by the compiler, NOT shown to the blind reader)
    mapping = {"r11l": "GameA", "vc33": "GameB", "tu93": "GameC", "sk48": "GameD"}
    arc = _get_arcade(DEFAULT_ENV_DIR)
    by = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    for env_id, label in mapping.items():
        info = by[env_id]
        card = arc.open_scorecard(tags=[f"diag_{env_id}"])
        env = arc.make(info.game_id, scorecard_id=card)
        obs = env.reset()
        avail = list(obs.available_actions or [])
        frames = blind_explore(env, obs)
        arc.close_scorecard(card)
        title = f"{label} — available actions {sorted(a for a in avail if a != 0)} — blind exploration"
        montage(frames, title).save(OUT / f"{label}.png")
        print(f"{label}: {len(frames)} frames, avail={sorted(a for a in avail if a!=0)} -> {OUT / (label+'.png')}")


if __name__ == "__main__":
    main()
