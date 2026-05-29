"""Stage 2 — within-episode world model.

Accumulates an action->effect model, controllable-object identity, attribute /
transformer tracking, resource signals, novelty counts and a memory log by
observing (action, board-delta) pairs. Everything is learned from experiment;
nothing about any specific env is assumed.
"""

from __future__ import annotations

import numpy as np

from .analyser import AnalysedFrame, analyse
from .change import diff_frames
from .constants import (
    BG, CONTROLLABLE_CONFIRM, DETERMINISTIC_FRAC, DIRECTIONAL, GRID,
    MIN_TRIALS, NOOP_HI,
)
from .matching import stable_object_ids
from .novelty import novelty_score
from .options import Options, generate_options
from .types import (
    ActionEffect, GoalCandidate, MemoryStep, Object, ResourceSignal, TransformerEvent, TransStat,
)

RESOURCE_CONFIRM = 4      # monotone steps before a strip is promoted to a budget signal


class WorldModel:
    def __init__(self) -> None:
        self.env_id: str | None = None
        self.run_id: int = 0
        self.available_actions: tuple[int, ...] = ()
        self.reset_for_episode()

    # ---- lifecycle -------------------------------------------------------
    def reset(self, env_id: str, available_actions, run_id: int = 0) -> None:
        self.env_id = env_id
        self.run_id = run_id
        self.available_actions = tuple(int(a) for a in available_actions)
        self.reset_for_episode()

    def reset_for_episode(self) -> None:
        self.step_idx = 0
        self.prev_af: AnalysedFrame | None = None
        self.cur_af: AnalysedFrame | None = None
        self.prev_objs: list[Object] = []
        self.cur_objs: list[Object] = []
        self._next_obj_id = 0
        self.last_action: int = -1
        self.last_click: tuple[int, int] | None = None
        self.prev_levels: int = 0
        self.effects: dict[int, ActionEffect] = {}
        self.visit_counts: dict[int, int] = {}
        self.coarse_counts: dict[int, int] = {}
        self.controllable_sig: int | None = None
        self._ctrl_candidates: dict[int, int] = {}     # shape_sig -> consistent count
        self.move_vectors: dict[int, tuple[int, int]] = {}
        self.transformer_events_list: list[TransformerEvent] = []
        self.goal: GoalCandidate | None = None
        self._strip_state: dict[str, dict] = {}        # name -> {last, sign, run}
        self.resources_list: list[ResourceSignal] = []
        self.memory: list[MemoryStep] = []
        self.confirmed_archetype_name: str | None = None
        self._reward_cells: list[tuple[int, int]] = []
        self.reward_click_classes: set[tuple[int, int]] = set()

    def _partial_reset_on_level(self) -> None:
        # New level: occupancy/objects/strides change; keep effects keyed by shape_sig.
        self.prev_objs = []
        self.cur_objs = []
        self._strip_state = {}

    # ---- ingest ----------------------------------------------------------
    def observe(self, obs) -> None:
        """Ingest a post-step observation (full FrameDataRaw)."""
        af = analyse(obs.frame, obs.available_actions, getattr(obs, "levels_completed", 0),
                     str(getattr(obs, "state", "")))
        cur_objs, self._next_obj_id = stable_object_ids(
            self.prev_objs, af.objects4, self._next_obj_id)

        # novelty counts
        self.visit_counts[af.frame_hash] = self.visit_counts.get(af.frame_hash, 0) + 1
        self.coarse_counts[af.coarse_sig] = self.coarse_counts.get(af.coarse_sig, 0) + 1

        levels_delta = af.levels_completed - self.prev_levels

        if self.prev_af is not None and self.last_action >= 0:
            delta = diff_frames(self.prev_objs, cur_objs, self.prev_af.grid, af.grid,
                                levels_delta, motion_cells=af.motion_cells)
            self._update_effect(self.last_action, delta)
            self._update_controllable(self.last_action, delta)
            self._update_transformers(self.last_action, delta, cur_objs)
            self._update_resources(self.last_action, af)
            self._update_goal(levels_delta, cur_objs)
            self._record_click_reward(delta)
            # fill the previous memory step's observed effect
            if self.memory:
                self.memory[-1].observed_delta = delta

        if levels_delta > 0:
            self._partial_reset_on_level()
            cur_objs, self._next_obj_id = stable_object_ids([], af.objects4, self._next_obj_id)

        self.prev_af = af
        self.cur_af = af
        self.prev_objs = cur_objs
        self.cur_objs = cur_objs
        self.prev_levels = af.levels_completed
        self.step_idx += 1

    def record_decision(self, action_id: int, data: dict) -> None:
        """Remember the action just chosen so the NEXT observe can attribute the delta."""
        self.last_action = int(action_id)
        self.last_click = (int(data["x"]), int(data["y"])) if action_id == 6 and data else None
        opts = self.options()
        self.memory.append(MemoryStep(
            step=self.step_idx,
            situation_sig=self.cur_af.frame_hash if self.cur_af else 0,
            options=list(opts.action_ids),
            n_click_targets=len(opts.click_targets),
            decision=int(action_id),
            click_xy=self.last_click,
            levels_completed=self.prev_levels,
            novelty_count=self.visit_counts.get(self.cur_af.frame_hash, 0) if self.cur_af else 0,
        ))

    # ---- update helpers --------------------------------------------------
    def _eff(self, aid: int) -> ActionEffect:
        if aid not in self.effects:
            self.effects[aid] = ActionEffect(action_id=aid)
        return self.effects[aid]

    def _update_effect(self, aid: int, delta) -> None:
        eff = self._eff(aid)
        eff.n_obs += 1
        if delta.is_noop:
            eff.n_noop += 1
        for d in delta.moved():
            sig = self._sig_of(d.obj_id)
            if sig is None or d.translation is None:
                continue
            ts = eff.translations.setdefault(sig, TransStat())
            ts.add(d.translation)
        for d in delta.object_deltas:
            if d.kind == "appeared" and d.new_color is not None:
                eff.appeared_colors[d.new_color] += 1
            elif d.kind == "disappeared" and d.old_color is not None:
                eff.disappeared_colors[d.old_color] += 1
        eff.attr_change_count += len(delta.attr_changes())
        if delta.levels_delta > 0:
            eff.levels_delta_count += 1
        if delta.multi_frame:
            eff.is_cascade_trigger = True
        # undo detection: grid returned to a previously-seen exact state
        if self.cur_af is not None and self.visit_counts.get(self.cur_af.frame_hash, 0) > 1 and aid == 7:
            eff.is_undo = True

    def _sig_of(self, obj_id: int) -> int | None:
        for o in self.cur_objs:
            if o.id == obj_id:
                return o.shape_sig
        for o in self.prev_objs:
            if o.id == obj_id:
                return o.shape_sig
        return None

    def _update_controllable(self, aid: int, delta) -> None:
        if aid not in DIRECTIONAL:
            return
        moved = delta.moved()
        if len(moved) != 1:
            return
        appeared = [d for d in delta.object_deltas if d.kind == "appeared"]
        disappeared = [d for d in delta.object_deltas if d.kind == "disappeared"]
        if appeared or disappeared:
            return
        sig = self._sig_of(moved[0].obj_id)
        if sig is None:
            return
        self._ctrl_candidates[sig] = self._ctrl_candidates.get(sig, 0) + 1
        if self._ctrl_candidates[sig] >= CONTROLLABLE_CONFIRM:
            self.controllable_sig = sig
        # learn the action vector for the controllable
        eff = self._eff(aid)
        if sig in eff.translations:
            ts = eff.translations[sig]
            if ts.is_deterministic(DETERMINISTIC_FRAC):
                self.move_vectors[aid] = ts.mode()

    def _update_transformers(self, aid: int, delta, cur_objs) -> None:
        for d in delta.attr_changes():
            axis = {"recolored": "color", "rotated": "rotation", "reshaped": "shape"}[d.kind]
            old = d.old_color if axis == "color" else (d.old_shape_sig or 0)
            new = d.new_color if axis == "color" else (d.new_shape_sig or 0)
            self.transformer_events_list.append(TransformerEvent(
                obj_id=d.obj_id, axis=axis, old=int(old or 0), new=int(new or 0),
                trigger_action=aid, trigger_xy=self.last_click, trigger_cell_color=None,
            ))

    def _update_resources(self, aid: int, af: AnalysedFrame) -> None:
        prev = self.prev_af
        if prev is None:
            return
        for name, ext in af.strip_extents.items():
            st = self._strip_state.get(name)
            prev_ext = prev.strip_extents.get(name, ext)
            d = ext - prev_ext
            if st is None:
                self._strip_state[name] = {"sign": 0, "run": 0, "promoted": False}
                st = self._strip_state[name]
            if d != 0:
                sign = 1 if d > 0 else -1
                if sign == st["sign"]:
                    st["run"] += 1
                else:
                    st["sign"] = sign
                    st["run"] = 1
                if st["sign"] < 0 and st["run"] >= RESOURCE_CONFIRM and not st["promoted"]:
                    st["promoted"] = True
                    self.resources_list.append(ResourceSignal(
                        region=self._strip_region(name), kind="budget",
                        monotone_per_action=True, depletes_to_terminal=True,
                        per_action_cost={aid: abs(d)},
                    ))

    def _strip_region(self, name: str) -> tuple[int, int, int, int]:
        kind, idx = name.split("_")
        i = int(idx)
        if kind == "row":
            return (i, 0, i, GRID - 1)
        return (0, i, GRID - 1, i)

    def _record_click_reward(self, delta) -> None:
        if self.last_action != 6 or self.last_click is None or delta.levels_delta <= 0:
            return
        x, y = self.last_click
        for o in self.prev_objs:           # objects in the frame that was clicked
            r0, c0, r1, c1 = o.bbox
            if r0 <= y <= r1 and c0 <= x <= c1 and o.mask[y - r0, x - c0]:
                self.reward_click_classes.add(o.class_key)
                return

    def rewarding_click_classes(self) -> set[tuple[int, int]]:
        return set(self.reward_click_classes)

    def _update_goal(self, levels_delta: int, cur_objs) -> None:
        if levels_delta > 0:
            # the controllable's attribute tuple at success is a candidate target
            ctrl = self.controllable_obj()
            if ctrl is not None:
                self.goal = GoalCandidate(kind="match_attr",
                                          target_attr=(ctrl.shape_sig, ctrl.color),
                                          confidence=1.0)

    # ---- read interface (decision + strategies) --------------------------
    def options(self) -> Options:
        if self.cur_af is None:
            return Options(action_ids=[a for a in self.available_actions if a != 0], click_targets=[])
        return generate_options(self.cur_af)

    def objects(self) -> list[Object]:
        return list(self.cur_objs)

    def object_classes(self) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = {}
        for o in self.cur_objs:
            out.setdefault(o.class_key, []).append(o.id)
        return out

    def effect(self, aid: int) -> ActionEffect | None:
        return self.effects.get(aid)

    def all_effects(self) -> dict[int, ActionEffect]:
        return dict(self.effects)

    def unknown_actions(self) -> list[int]:
        return [a for a in self.available_actions
                if a != 0 and (a not in self.effects or self.effects[a].n_obs < MIN_TRIALS)]

    def known_noop_actions(self) -> set[int]:
        return {a for a, e in self.effects.items()
                if e.n_obs >= MIN_TRIALS and e.noop_rate >= NOOP_HI}

    def controllable_obj(self) -> Object | None:
        if self.controllable_sig is None:
            return None
        cands = [o for o in self.cur_objs if o.shape_sig == self.controllable_sig]
        if not cands:
            return None
        return min(cands, key=lambda o: o.size)   # heuristic: avatar is usually small

    def controllable_obj_id(self) -> int | None:
        o = self.controllable_obj()
        return o.id if o else None

    def move_action_vectors(self) -> dict[int, tuple[int, int]]:
        return dict(self.move_vectors)

    def occupancy(self) -> np.ndarray:
        """True = blocked. Non-background cells minus the controllable object."""
        if self.cur_af is None:
            return np.zeros((GRID, GRID), dtype=bool)
        occ = (self.cur_af.grid != BG)
        ctrl = self.controllable_obj()
        if ctrl is not None:
            r0, c0, r1, c1 = ctrl.bbox
            occ[r0:r1 + 1, c0:c1 + 1] &= ~ctrl.mask
        return occ

    def transformer_events(self) -> list[TransformerEvent]:
        return list(self.transformer_events_list)

    def carried_attr(self) -> tuple[int, int] | None:
        o = self.controllable_obj()
        return (o.shape_sig, o.color) if o else None

    def candidate_goal(self) -> GoalCandidate | None:
        return self.goal

    def resources(self) -> list[ResourceSignal]:
        return list(self.resources_list)

    def last_levels_completed(self) -> int:
        return self.prev_levels

    def state_key(self) -> int:
        return self.cur_af.frame_hash if self.cur_af else 0

    def visit_count(self, key: int | None = None) -> int:
        if key is None:
            key = self.state_key()
        return self.visit_counts.get(key, 0)

    def novelty(self, key: int | None = None) -> float:
        return novelty_score(self.visit_count(key))

    def confirmed_archetype(self) -> str | None:
        """Strongest archetype supported by hard evidence (reward-linked first)."""
        if self.reward_click_classes:
            return "click"
        if self.goal is not None and self.goal.kind == "match_attr" and self.transformer_events_list:
            return "attribute"
        if self.controllable_sig is not None:
            return "movement"
        return None

    def step_index(self) -> int:
        return self.step_idx

    def actions_remaining_estimate(self) -> int | None:
        for r in self.resources_list:
            if r.kind in ("budget", "refilling_budget") and r.extent_history:
                return r.extent_history[-1]
        return None

    def dump_learned_model(self) -> dict:
        """For the ls20 rule-recovery grader (read-only inspection)."""
        return {
            "controllable_sig": self.controllable_sig,
            "move_vectors": {a: list(v) for a, v in self.move_vectors.items()},
            "n_transformer_events": len(self.transformer_events_list),
            "transformer_axes": sorted({e.axis for e in self.transformer_events_list}),
            "goal": None if self.goal is None else {
                "kind": self.goal.kind, "target_attr": self.goal.target_attr},
            "resources": [
                {"kind": r.kind, "depletes_to_terminal": r.depletes_to_terminal,
                 "refill_events": r.refill_events, "monotone_per_action": r.monotone_per_action}
                for r in self.resources_list
            ],
            "reward_events": sum(e.levels_delta_count for e in self.effects.values()),
            "confirmed_archetype": self.confirmed_archetype_name,
        }
