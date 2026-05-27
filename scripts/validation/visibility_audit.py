"""Per-env visibility audit (Phase 0c OQ4).

Strategy: for each env source file, count occurrences of patterns that
indicate fog-of-war / limited-visibility / per-cell-masking behavior.
The base ARCBaseGame uses RenderableUserDisplay generically; visibility
restrictions show up as ADDITIONAL custom render logic that masks cells
based on agent position or other state.

Reports per-env: lines of code, count of frame-mask / fog / visibility /
distance-from-player patterns. Manual inspection threshold: any env with
multiple hits in the targeted patterns is a visibility candidate.
"""

import json
import re
from collections import Counter
from pathlib import Path

ENV_DIR = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\arc-prize-2026-arc-agi-3\environment_files")
OUT = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.visibility_audit.json")

# Patterns that strongly suggest fog-of-war / limited visibility:
PATTERNS = {
    # mask any cell of the frame array based on a distance / radius
    "frame_assign_mask": re.compile(r"frame\[[^\]]+\]\s*=\s*[^\n]+(?:if|for)"),
    "frame_zero_out": re.compile(r"frame\[[^\]]+\]\s*=\s*(?:0|BACKGROUND|self\.background)"),
    "distance_check": re.compile(r"(?:abs|sqrt|hypot|dist|manhattan|chebyshev|max\(abs).{0,80}(?:player|self\.x|self\.y|agent)", re.IGNORECASE),
    "radius_param": re.compile(r"(?:radius|sight|fov|view_range|visibility|view_dist|sight_radius|reveal)", re.IGNORECASE),
    "frame_loop_mask": re.compile(r"for\s+\w+\s+in\s+range\(\s*(?:64|frame\.shape|height|width)"),
    "render_override": re.compile(r"def\s+(?:render|_render|render_frame|on_render|render_interface)\s*\("),
    "custom_renderable": re.compile(r"class\s+\w+\(RenderableUserDisplay\)"),
    "player_centric": re.compile(r"(?:player|protagonist|main_char|main_sprite|self_pos|agent_pos)\s*[\.\[]"),
    "darkness_color": re.compile(r"(?:\bdark\b|\bblack\b|\bobscur|\bhidden\b|\bcover\b|\bcensor\b|\bshroud\b)", re.IGNORECASE),
}


def audit(env_dir: Path) -> dict:
    py_files = list(env_dir.rglob("*.py"))
    if not py_files:
        return {"env_id": env_dir.name, "error": "no python source"}
    src_path = py_files[0]
    src = src_path.read_text(encoding="utf-8", errors="replace")
    pattern_hits = {name: len(p.findall(src)) for name, p in PATTERNS.items()}
    n_render_classes = pattern_hits["custom_renderable"]
    n_render_methods = pattern_hits["render_override"]
    # Heuristic score: weight true-positive patterns higher
    visibility_score = (
        2 * pattern_hits["frame_assign_mask"]
        + 3 * pattern_hits["frame_zero_out"]
        + 4 * pattern_hits["distance_check"]
        + 5 * pattern_hits["radius_param"]
        + 1 * pattern_hits["frame_loop_mask"]
        + 1 * pattern_hits["player_centric"]
    )
    return {
        "env_id": env_dir.name,
        "src_lines": len(src.splitlines()),
        "n_render_classes": n_render_classes,
        "n_render_methods": n_render_methods,
        "pattern_hits": pattern_hits,
        "visibility_score": visibility_score,
    }


def main():
    env_dirs = sorted([p for p in ENV_DIR.iterdir() if p.is_dir()])
    results = []
    for env_dir in env_dirs:
        # source lives under <env>/<hash>/<env>.py — descend
        hashes = [p for p in env_dir.iterdir() if p.is_dir()]
        if not hashes:
            continue
        r = audit(hashes[0])
        results.append(r)

    # Sort by visibility_score desc
    ranked = sorted(results, key=lambda r: -r.get("visibility_score", 0))
    print("Env | LoC | RenderCls | RenderMeth | VisScore | Patterns")
    print("-" * 100)
    for r in ranked:
        ph = r.get("pattern_hits", {})
        flag = "VIS?" if r.get("visibility_score", 0) >= 8 else ""
        print(
            f"{r['env_id']:5} | {r.get('src_lines',0):4} | "
            f"{r.get('n_render_classes',0):2} | "
            f"{r.get('n_render_methods',0):2} | "
            f"{r.get('visibility_score',0):4} | "
            f"mask={ph.get('frame_assign_mask',0)} "
            f"zero={ph.get('frame_zero_out',0)} "
            f"dist={ph.get('distance_check',0)} "
            f"rad={ph.get('radius_param',0)} "
            f"loop={ph.get('frame_loop_mask',0)} "
            f"player={ph.get('player_centric',0)} "
            f"dark={ph.get('darkness_color',0)} {flag}"
        )

    OUT.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
