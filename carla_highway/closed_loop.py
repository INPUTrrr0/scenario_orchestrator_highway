#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/closed_loop.py — the orchestrator, against a live CARLA ego.

The counterpart of `carla_port/closed_loop.py`. That one subclasses
`v4/orchestrator.Orchestrator` to substitute a live ego into `_world_now()` and
thread `hero_arms` through the red-light directives. This one has an easier job,
because the highway orchestration kernel is *already* headless:

    highway/cutin_orchestrator.py  CutinOrchestrator.tick(asc, atime, ego, t)

`tick` rebases the background scenario to now, re-casts the cut-in role against
the live ego pose, replans the holder toward the ego-relative pin, applies the
actor-actor collision-yield directive, and re-simulates. It touches pygame only
in `draw_panel`, which nothing here calls. So this module owns the *cadence and
the clock*, not the policy: it is wiring, and the orchestration logic stays
upstream where it can be compared against the highway runs.

The clock
---------
`CutinOrchestrator.tick` returns a scenario rebased to the current instant, so
its trajectory frame 0 is *now* and script-local time restarts at zero. The
runner samples actor states at `atime`, which this module advances by the CARLA
timestep and resets on every orchestration tick. That is the same contract
`carla_port/carla_sync.py` describes for `Orchestrator._rebase_here`, reached by
a different route.

Casting is per mode
-------------------
Only `scenario_cutin` is orchestrated. `scenario_hard_brake` and
`scenario_overtake` are ego-policy stress tests: their actors are fully scripted
on purpose, and cast roles or yields would change the very timings the scenarios
were tuned around. `ModeSpec.casting` carries that decision, and `--casting` /
`--no-casting` overrides it.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .highway_ego import Ego
from .highway_map import HighwayFrame
from .scenarios import ModeSpec
from .script_bridge import ROLE_CUTIN, ROLE_NOMINAL, co, mv, se

#: orchestration cadence, matching v4/orchestrator.DT_TICK
DT_TICK = 0.10


@dataclass
class Event:
    """Something the orchestrator did, for the report and the HUD."""
    t: float
    kind: str                 # cast | cutin | yield | outcome
    text: str
    actor: Optional[str] = None


def cutin_spec_of(sc: "se.Scenario") -> Optional[dict]:
    """The `cutin:` block the scenario author put on one of the actors.

    The orchestrator moves this spec between actors as it recasts, so it
    belongs to the scenario, not to whichever actor currently holds it.
    """
    for a in sc.actors:
        if getattr(a, "cutin", None):
            return dict(a.cutin)
    return None


class HighwayClosedLoop:
    """Owns the background scenario, its clock, and the orchestration cadence."""

    def __init__(self, frame: HighwayFrame, background: "se.Scenario",
                 mode: ModeSpec, casting: Optional[bool] = None,
                 cruise: float = 12.0, cutin_spec: Optional[dict] = None,
                 cutin_at: Optional[float] = None):
        self.frame = frame
        self.sc = background
        self.mode = mode
        self.casting = mode.casting if casting is None else bool(casting)
        self.atime = 0.0                  # script time since sc's frame 0
        self.events: List[Event] = []
        self.cutin_at = cutin_at
        self.cutin_start = 0.0
        self._next_tick = 0.0
        self._last_msg = ""
        self.n_recasts = 0

        spec = cutin_spec or cutin_spec_of(background)
        self.orch: Optional["co.CutinOrchestrator"] = None
        if self.casting:
            if spec is None:
                self.casting = False
                self._note(0.0, "cast",
                           "no `cutin:` spec in this scenario; casting disabled")
            else:
                if cutin_at is not None:
                    spec = dict(spec)
                    lc = float(spec.get("lc_duration", 2.0))
                    self.cutin_start = max(0.0, float(cutin_at))
                    spec["t"] = float(cutin_at) + max(lc + 2.0, 4.0)
                    self._next_tick = self.cutin_start
                self.orch = co.CutinOrchestrator(spec, cruise_speed=cruise)
                if cutin_at is not None:
                    self._note(0.0, "cast",
                               f"cut-in starts at t={self.cutin_start:.1f}s, "
                               f"merge by t={spec['t']:.1f}s")
        self.sc.simulate()

    # ------------------------------------------------------------------ #
    def _note(self, t: float, kind: str, text: str,
              actor: Optional[str] = None) -> None:
        self.events.append(Event(t=t, kind=kind, text=text, actor=actor))

    def advance(self, dt: float) -> None:
        """Move the script clock forward one CARLA step."""
        self.atime += dt

    def due(self, t_sim: float) -> bool:
        if self.cutin_at is not None and t_sim < self.cutin_start:
            return False
        return t_sim >= self._next_tick

    def tick(self, ego: Ego, t_sim: float) -> bool:
        """One orchestration step. Returns True if the plans changed."""
        if not self.due(t_sim):
            return False
        self._next_tick = t_sim + DT_TICK
        if self.orch is None:
            return False

        before_holder = self.orch.cutin_id
        before_n = self.orch.n_interventions
        self._recast_if_hopeless(ego, t_sim)
        # CutinOrchestrator.tick rebases to now, so script time restarts.
        self.sc, _ = self.orch.tick(self.sc, self.atime, ego, t_sim)
        self.atime = 0.0

        changed = self.orch.n_interventions != before_n
        if self.orch.cutin_id != before_holder:
            self._note(t_sim, "cast",
                       f"cut-in cast to actor {self.orch.cutin_id}"
                       if self.orch.cutin_id else "no viable cut-in candidate",
                       actor=self.orch.cutin_id)
        msg = getattr(self.orch, "msg", "")
        if msg and msg != self._last_msg:
            self._last_msg = msg
            kind = "outcome" if self.orch.outcome else (
                "yield" if msg.startswith("collision:") else "cutin")
            self._note(t_sim, kind, msg, actor=self.orch.cutin_id)
        return changed

    # ------------------------------------------------------------------ #
    # Recasting
    # ------------------------------------------------------------------ #
    def _recast_if_hopeless(self, ego: Ego, t_sim: float) -> None:
        """Move the cut-in role when the holder can no longer make the pin.

        `CutinOrchestrator.cast` is sticky on *identity*: once `cutin_id` is
        set it re-locks the same actor every tick until the cut-in commits. The
        recast rule lives one level up, in `scenario_editor.py`'s
        `cast_cutin_roles`, which the pygame editor calls and a headless port
        does not — so without this the first pick holds the role forever. That
        is not hypothetical: on `scenario_cutin` the initial tie goes to actor
        2, and it stayed cast at a candidate score of 0.016 while actor 4 sat
        at 0.64 and the cut-in ran out its deadline.

        The upstream rule, reproduced here with upstream's own predicates:

            stickiness is on FEASIBILITY, not score.

        Mid-chase the holder drifts toward the ego's lane, which tanks its
        candidate score — that is progress, not failure. So the role moves only
        when `live_cutin_feasible` says the holder can no longer reach the pin
        by the deadline, and then it goes to the best-scoring actor that still
        can. If nobody can, the holder keeps it and abandons at the deadline,
        which is what upstream does too.
        """
        orch = self.orch
        if orch is None or orch.committed or not orch.cutin_id:
            return
        holder = next((a for a in self.sc.actors
                       if str(a.id) == str(orch.cutin_id)), None)
        if holder is None:
            return
        spec = orch.spec
        theta = ego.theta

        def pose_of(a):
            return a.pose_at_time(self.atime) if a.traj else a.start

        if se.live_cutin_feasible(pose_of(holder), ego.x, ego.y, theta,
                                  ego.v, spec, t_sim):
            return

        cands = [a for a in self.sc.actors
                 if a is not holder
                 and getattr(a, "autonomy", "auto") != "self"
                 and not getattr(a, "block", None)]
        feasible = [a for a in cands
                    if se.live_cutin_feasible(pose_of(a), ego.x, ego.y, theta,
                                              ego.v, spec, t_sim)]
        if not feasible:
            return                     # nobody can; the holder abandons on time

        ego_pose = (ego.x, ego.y, math.degrees(theta))
        lw = self.sc.map.lane_width
        scores = {a.id: co.score_cutin_candidate(pose_of(a), ego_pose, lw)
                  for a in feasible}
        best = max(feasible, key=lambda a: scores[a.id])
        v_req, _ = se.live_cutin_required_speed(pose_of(holder), ego.x, ego.y,
                                                theta, ego.v, spec, t_sim)
        self._note(t_sim, "cast",
                   f"actor {holder.id} can't make the pin (needs "
                   f"{v_req:.1f} m/s) — recast to {best.id} "
                   f"(score {scores[best.id]:.2f})", actor=str(best.id))
        orch.cutin_id = str(best.id)
        orch.n_interventions += 1
        self.n_recasts += 1

    # ---- views ---- #
    @property
    def roles(self) -> Dict[str, str]:
        return dict(self.orch.roles) if self.orch else {}

    @property
    def scores(self) -> Dict[str, float]:
        return dict(self.orch.scores) if self.orch else {}

    @property
    def holder(self) -> Optional[str]:
        return self.orch.cutin_id if self.orch else None

    @property
    def outcome(self) -> Optional[str]:
        """'merged' | 'abandoned' | None (still deciding, or not casting)."""
        return self.orch.outcome if self.orch else None

    @property
    def committed(self) -> bool:
        return bool(self.orch.committed) if self.orch else False

    @property
    def n_interventions(self) -> int:
        return self.orch.n_interventions if self.orch else 0

    @property
    def status(self) -> str:
        if self.orch is None:
            return "scripted actors (no casting in this mode)"
        return self.orch.msg

    def summary(self) -> dict:
        out = {
            "casting": self.casting,
            "holder": self.holder,
            "outcome": self.outcome,
            "committed": self.committed,
            "interventions": self.n_interventions,
            "recasts": self.n_recasts,
            "roles": self.roles,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "events": [{"t": round(e.t, 3), "kind": e.kind, "actor": e.actor,
                        "text": e.text} for e in self.events],
        }
        if self.cutin_at is not None:
            out["cutin_at"] = round(float(self.cutin_at), 3)
            out["cutin_start"] = round(float(self.cutin_start), 3)
            if self.orch is not None:
                out["cutin_deadline"] = round(float(self.orch.spec.get("t", 0)), 3)
        return out
