"""Full env source audit (Phase 0b Section 2).

For each of the 25 public envs, parse env_files/<env>/<hash>/<env>.py and
metadata.json. Extract structural facts: sprite tags, action handlers,
lose/win conditions, mid-env mechanic triggers, visibility/animation
indicators.

Output a master JSON table at scripts/validation/.env_source_audit.json.
"""

import json
import re
from collections import Counter
from pathlib import Path

ENV_DIR = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\arc-prize-2026-arc-agi-3\environment_files")
OUT = Path(r"C:\Users\adars\Downloads\ARC-AGI-3\scripts\validation\.env_source_audit.json")

ACTION_RE = re.compile(r"\bGameAction\.(ACTION[1-7]|RESET)\b")
LOSE_RE = re.compile(r"\bself\.lose\(\)")
ON_SET_LEVEL_RE = re.compile(r"def\s+on_set_level\s*\(")
HIDDEN_STATE_RE = re.compile(r"def\s+_get_hidden_state")
VALID_ACTIONS_RE = re.compile(r"def\s+_get_valid_actions")
ROTATION_RE = re.compile(r"rotation|rot90|set_rotation")
VISIBILITY_RE = re.compile(r"fog|visib|mask_frame|render_mask|hide_sprite|reveal|opaque|sight", re.IGNORECASE)
ANIMATION_RE = re.compile(r"add_animation|frame_stack|append_frame|self\._frames\.append|render_loop")
SPRITE_TAGS_RE = re.compile(r'tags\s*=\s*\[([^\]]*)\]')
AVAILABLE_ACTIONS_RE = re.compile(r"available_actions\s*=\s*\[([^\]]+)\]")
STEPS_BUDGET_RE = re.compile(r'"steps"\s*:\s*(\d+)')


def audit_env(env_id: str, hash_dir: Path) -> dict:
    py_path = hash_dir / f"{env_id}.py"
    md_path = hash_dir / "metadata.json"
    if not py_path.exists():
        return {"env_id": env_id, "error": f"missing {py_path}"}
    src = py_path.read_text(encoding="utf-8", errors="replace")
    md = json.loads(md_path.read_text(encoding="utf-8")) if md_path.exists() else {}

    actions_referenced = sorted({m.group(1) for m in ACTION_RE.finditer(src)})
    avail = []
    m = AVAILABLE_ACTIONS_RE.search(src)
    if m:
        for tok in re.findall(r"\d+", m.group(1)):
            avail.append(int(tok))

    sprite_tag_lists = SPRITE_TAGS_RE.findall(src)
    sprite_tags = set()
    for taglist in sprite_tag_lists:
        for tg in re.findall(r'"([^"]+)"', taglist):
            sprite_tags.add(tg)

    n_lose = len(LOSE_RE.findall(src))
    has_on_set_level = bool(ON_SET_LEVEL_RE.search(src))
    has_hidden_state = bool(HIDDEN_STATE_RE.search(src))
    has_valid_actions_override = bool(VALID_ACTIONS_RE.search(src))
    has_rotation = bool(ROTATION_RE.search(src))
    has_visibility = bool(VISIBILITY_RE.search(src))
    has_animation = bool(ANIMATION_RE.search(src))

    steps_per_level = [int(x) for x in STEPS_BUDGET_RE.findall(src)]

    has_click = 6 in avail
    has_movement = bool({1, 2, 3, 4} & set(avail))
    has_undo = 7 in avail
    if has_click and not has_movement:
        sig = "pure_click"
    elif has_movement and not has_click:
        sig = "pure_movement"
    elif has_click and has_movement:
        sig = "mixed"
    else:
        sig = f"other:{avail}"

    return {
        "env_id": env_id,
        "hash": hash_dir.name,
        "metadata": {
            "title": md.get("title"),
            "n_levels": len(md.get("baseline_actions", []) or []),
            "baseline_actions": md.get("baseline_actions"),
            "tags": md.get("tags"),
            "default_fps": md.get("default_fps"),
        },
        "available_actions": avail,
        "actions_referenced": actions_referenced,
        "action_signature": sig,
        "has_undo": has_undo,
        "n_sprite_tags": len(sprite_tags),
        "sprite_tags": sorted(sprite_tags),
        "n_lose_calls": n_lose,
        "has_on_set_level": has_on_set_level,
        "has_hidden_state": has_hidden_state,
        "has_valid_actions_override": has_valid_actions_override,
        "has_rotation": has_rotation,
        "has_visibility": has_visibility,
        "has_animation": has_animation,
        "steps_per_level": steps_per_level,
        "src_lines": len(src.splitlines()),
    }


def main():
    env_dirs = sorted([p for p in ENV_DIR.iterdir() if p.is_dir()])
    rows = []
    for env_dir in env_dirs:
        hashes = [p for p in env_dir.iterdir() if p.is_dir()]
        if not hashes:
            continue
        info = audit_env(env_dir.name, hashes[0])
        rows.append(info)
        md = info.get("metadata", {})
        print(
            f"{info['env_id']:5} | "
            f"lvls={md.get('n_levels',0):2} | "
            f"sig={info.get('action_signature',''):<14} | "
            f"tags={info.get('n_sprite_tags',0):2} | "
            f"rot={int(info.get('has_rotation',0))} "
            f"vis={int(info.get('has_visibility',0))} "
            f"anim={int(info.get('has_animation',0))} "
            f"val_act={int(info.get('has_valid_actions_override',0))} "
            f"hidden={int(info.get('has_hidden_state',0))} "
            f"undo={int(info.get('has_undo',0))} | "
            f"lose_calls={info.get('n_lose_calls',0)}"
        )

    all_tags = set()
    for r in rows:
        all_tags.update(r.get("sprite_tags", []))

    sig_count = Counter(r.get("action_signature") for r in rows)
    lvl_count = Counter(r.get("metadata", {}).get("n_levels", 0) for r in rows)

    agg = {
        "n_envs": len(rows),
        "envs_with_rotation": sum(r.get("has_rotation", False) for r in rows),
        "envs_with_visibility_hints": sum(r.get("has_visibility", False) for r in rows),
        "envs_with_animation_hints": sum(r.get("has_animation", False) for r in rows),
        "envs_with_valid_actions_override": sum(r.get("has_valid_actions_override", False) for r in rows),
        "envs_with_undo": sum(r.get("has_undo", False) for r in rows),
        "envs_with_hidden_state": sum(r.get("has_hidden_state", False) for r in rows),
        "envs_by_signature": dict(sig_count),
        "level_count_hist": dict(lvl_count),
        "total_levels": sum(r.get("metadata", {}).get("n_levels", 0) for r in rows),
        "total_distinct_sprite_tags": len(all_tags),
        "all_sprite_tags": sorted(all_tags),
    }

    out = {"summary": agg, "per_env": rows}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== AGGREGATES ===")
    print(json.dumps(agg, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
