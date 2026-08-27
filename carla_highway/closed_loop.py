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


class StickyCutinOrchestrator(co.CutinOrchestrator):
    """`CutinOrchestrator` whose cast is sticky on FEASIBILITY, not score.

    This is the one place the port has to override the orchestration kernel
    rather than merely drive it, and the reason is a contradiction inside
    `cast_roles` that makes a merge undetectable:

    * `cast_roles` honours a `lock_id` only while that actor still scores
      above zero, and `score_cutin_candidate` returns exactly zero for an
      actor that is not `cutin_eligible` — *adjacent lane* and ahead.
    * A cut-in succeeds by leaving the adjacent lane. `cutin_is_merged`
      requires the actor within 0.5 m of a pin whose lateral offset is 0, i.e.
      on the ego's own line; `cutin_adjacent` requires at least 0.4 lane widths
      away from it. The two conditions cannot hold at once.

    So on the very tick the holder arrives at the pin, `tick` casts *before* it
    plans, the lock is refused for scoring zero, the role moves to somebody
    else, and `apply_closed_loop_cutin` — the only thing that can return
    "merged" — is never called for the actor that just merged. Every cut-in
    then runs out its deadline and reports `abandoned`. That is what the first
    CARLA runs did, without exception, including the scripted-ego run that
    reproduces upstream's authored conditions.

    `scenario_editor.py`'s own casting, which is what `experiment.py` drives
    and where the repository's successful runs come from, does not have this
    problem: it identifies the holder by *which actor carries the spec* and
    states the rule in as many words — "mid-chase the holder drifts toward the
    ego's lane, which tanks its candidate score — that is progress, not
    failure — so stickiness is on feasibility, not score." This class applies
    that rule to `CutinOrchestrator`, and nothing more: the holder keeps the
    role while `HighwayClosedLoop._recast_if_hopeless` judges it still able to
    make the pin, and `cast_roles` decides everything else exactly as before.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: the holder the port has decided to keep this tick, or None to let
        #: `cast_roles` have its way. Set by `_recast_if_hopeless`.
        self.sticky_id: Optional[str] = None

    def tick(self, asc, atime, ego, clock_t):
        """`CutinOrchestrator.tick`, with the merged actor driving its OWN
        cruise instead of matching the live ego.

        Upstream, an actor that has merged is re-planned every tick at
        `speed = ego.v`:

            if (self.committed and a.id == self.cutin_id
                    and self.outcome == "merged"):
                speed = ego.v

        Against the pygame ego that is harmless — that ego runs a scripted
        speed profile and keeps moving whatever happens in front of it. Against
        a closed-loop ego it is a deadlock, and a total one. The ego brakes for
        the car that just cut in; the merged actor is then commanded to match
        the ego's reduced speed; the smaller gap makes IDM brake harder; the
        actor matches that too. Both spiral to a standstill within a second of
        the merge and neither ever moves again. In the first CARLA cut-in the
        actor went 12.0 -> 0.5 m/s in 0.6 s and both cars sat still for the
        remaining nine seconds of the run.

        `docs/SCENARIOS_AND_VALIDATION.md` asks for "a **nominal speed** (near
        the ego's speed and/or its own cruise), not crawling or abandoned" — so
        its own cruise satisfies the criterion, and it is the only one of the
        two that cannot collapse. The scenario YAML authors a `cruise` for
        every actor; `actor_cruise_speed` reads it.

        Note that the verifier does not catch the deadlock: criterion 4 accepts
        "near the ego's speed", and a stopped actor behind a stopped ego is
        very near it indeed.
        """
        out = super().tick(asc, atime, ego, clock_t)
        if not (self.committed and self.outcome == "merged" and self.cutin_id):
            return out
        sc = out[0]
        for a in sc.actors:
            if str(a.id) != str(self.cutin_id):
                continue
            v_nom = se.actor_cruise_speed(a, default=self.cruise_speed)
            cur = (float(a.maneuvers[0].intercept)
                   if a.maneuvers and hasattr(a.maneuvers[0], "intercept")
                   else None)
            if cur is None or abs(cur - v_nom) > 1e-3:
                co.apply_nominal(a, v_nom,
                                 heading_deg=self.headings.get(a.id))
                sc.simulate()
            break
        return out

    def cast(self, actors, ego, lane_width, sticky: bool = True):
        n_before = self.n_interventions
        castings = super().cast(actors, ego, lane_width, sticky=sticky)
        keep = self.sticky_id
        if (keep is None or self.committed or self.cutin_id == keep
                or not any(str(c.actor_id) == str(keep) for c in castings)):
            return castings
        castings = [co.Casting(actor_id=c.actor_id,
                               role=(ROLE_CUTIN if str(c.actor_id) == str(keep)
                                     else ROLE_NOMINAL),
                               score=c.score)
                    for c in castings]
        self.roles = {c.actor_id: c.role for c in castings}
        self.scores = {c.actor_id: c.score for c in castings}
        self.cutin_id = keep
        # `super().cast` counted moving the role away as an intervention; it
        # did not happen, so do not report it.
        self.n_interventions = n_before
        return castings


class HighwayClosedLoop:
    """Owns the background scenario, its clock, and the orchestration cadence."""

    def __init__(self, frame: HighwayFrame, background: "se.Scenario",
                 mode: ModeSpec, casting: Optional[bool] = None,
                 cruise: float = 12.0, cutin_spec: Optional[dict] = None,
                 cutin_at: Optional[float] = None,
                 cutin_along: Optional[float] = None):
        self.frame = frame
        self.sc = background
        self.mode = mode
        self.casting = mode.casting if casting is None else bool(casting)
        self.atime = 0.0                  # script time since sc's frame 0
        self.events: List[Event] = []
        self.cutin_at = cutin_at
        self.cutin_start = 0.0
        #: sim time the orchestrator first declared the cut-in committed. The
        #: upstream verifier keys every one of its four cut-in checks off this
        #: instant (`docs/SCENARIOS_AND_VALIDATION.md`), so it has to be
        #: recorded when it happens — it cannot be recovered afterwards.
        self.t_commit: Optional[float] = None
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
                if cutin_along is not None:
                    # The pin is authored centre-to-centre against 4.5 m
                    # bodies. CARLA spawns whatever the blueprint is, and on
                    # Town04 that has reached 5.2 m — so the authored 6.0 m is
                    # 1.55 m bumper to bumper and the merge cannot help but
                    # touch. Widening the pin is the port's business, not the
                    # scenario's: `se.cutin_is_merged` and the upstream
                    # verifier both accept anything up to CUTIN_MAX_AHEAD_M
                    # (10 m), so there is room to give without leaving the
                    # window either of them checks.
                    spec = dict(spec)
                    spec["along"] = float(cutin_along)
                if cutin_at is not None:
                    spec = dict(spec)
                    lc = float(spec.get("lc_duration", 2.0))
                    self.cutin_start = max(0.0, float(cutin_at))
                    spec["t"] = float(cutin_at) + max(lc + 2.0, 4.0)
                    self._next_tick = self.cutin_start
                self.orch = StickyCutinOrchestrator(spec, cruise_speed=cruise)
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
        # `committed` latches on either outcome, and the recorded `t_commit` is
        # that instant — the same convention `experiment.py` writes upstream,
        # so a CARLA run and a pygame run mean the same thing by the field. It
        # is only meaningful alongside `success`, and the verifier checks that
        # first.
        if self.t_commit is None and self.orch.committed:
            self.t_commit = t_sim
            self._note(t_sim, "cutin",
                       f"cut-in {self.orch.outcome} by actor "
                       f"{self.orch.cutin_id} at t={t_sim:.2f}s",
                       actor=self.orch.cutin_id)
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
        if orch is None:
            return
        orch.sticky_id = None
        if orch.committed or not orch.cutin_id:
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
            # Still able to make the pin, so it keeps the role — even once it
            # has drifted far enough into the ego's lane that `cast_roles`
            # would score it zero and hand the part to somebody else. See
            # `StickyCutinOrchestrator`.
            orch.sticky_id = str(orch.cutin_id)
            return

        cands = [a for a in self.sc.actors
                 if a is not holder
                 and getattr(a, "autonomy", "auto") != "self"
                 and not getattr(a, "block", None)]
        feasible = [a for a in cands
                    if se.live_cutin_feasible(pose_of(a), ego.x, ego.y, theta,
                                              ego.v, spec, t_sim)]
        if not feasible:
            orch.sticky_id = str(orch.cutin_id)
            return                     # nobody can; the holder abandons on time

        ego_pose = (ego.x, ego.y, math.degrees(theta))
        lw = self.sc.map.lane_width
        scores = {a.id: co.score_cutin_candidate(pose_of(a), ego_pose, lw)
                  for a in feasible}
        # Feasibility and eligibility are different tests, and the recast has
        # to respect BOTH or it deadlocks against the very function it is
        # feeding. `live_cutin_feasible` asks "could this actor still reach the
        # pin by the deadline" — an actor coming up from behind can. But
        # `cast_roles` only casts actors that are `cutin_eligible` (adjacent
        # lane AND already ahead), and it only honours a lock whose score is
        # > 0, so an actor still behind the ego scores 0 and is refused.
        #
        # Handing the role to such an actor produces a two-tick oscillation:
        # this method sets cutin_id, `cast` next tick rejects the lock and
        # falls back to "no viable candidate", cutin_id goes to None, the tick
        # after that re-picks the same infeasible holder, and round it goes at
        # 10 Hz. That is exactly what the first CARLA runs did — 19
        # interventions and 5 recasts inside four seconds, never committing.
        # If nobody eligible can make it, the holder keeps the role and
        # abandons on time, which is what upstream does.
        feasible = [a for a in feasible if scores[a.id] > 0.0]
        if not feasible:
            orch.sticky_id = str(orch.cutin_id)
            return                     # the holder keeps it and abandons on time
        best = max(feasible, key=lambda a: scores[a.id])
        v_req, _ = se.live_cutin_required_speed(pose_of(holder), ego.x, ego.y,
                                                theta, ego.v, spec, t_sim)
        self._note(t_sim, "cast",
                   f"actor {holder.id} can't make the pin (needs "
                   f"{v_req:.1f} m/s) — recast to {best.id} "
                   f"(score {scores[best.id]:.2f})", actor=str(best.id))
        orch.cutin_id = str(best.id)
        orch.sticky_id = str(best.id)
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
            "t_commit": (round(self.t_commit, 3)
                         if self.t_commit is not None else None),
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
