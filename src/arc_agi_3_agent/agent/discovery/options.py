"""Option generation — candidate actions + per-CLASS click targets.

One representative click per distinct (shape_sig, color) object class, NOT per
pixel. This is the explicit fix for the BC spatial-head failure (memorized
absolute pixels generalized to 0.000 on holdout click envs).
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyser import AnalysedFrame
from .constants import ACTION6, RESET


@dataclass(frozen=True)
class ClickTarget:
    xy: tuple[int, int]                  # (x=col, y=row)
    class_key: tuple[int, int]           # (shape_sig, color)
    obj_id: int


@dataclass
class Options:
    action_ids: list[int]                # voluntary actions (RESET stripped)
    click_targets: list[ClickTarget]     # only populated if ACTION6 available


def generate_options(af: AnalysedFrame) -> Options:
    action_ids = [a for a in af.available_actions if a != RESET]
    click_targets: list[ClickTarget] = []
    if ACTION6 in af.available_actions:
        seen: set[tuple[int, int]] = set()
        for obj in sorted(af.objects4, key=lambda o: (-o.size, o.id)):
            ck = obj.class_key
            if ck in seen:
                continue
            seen.add(ck)
            click_targets.append(ClickTarget(xy=obj.rep_cell(), class_key=ck, obj_id=obj.id))
    return Options(action_ids=action_ids, click_targets=click_targets)
