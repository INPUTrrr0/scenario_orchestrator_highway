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
        #: sim time the holder started waiting, or None when it is not
        #: waiting. Only used to note the wait once per episode of waiting.
        self._waiting_since: Optional[float] = None
        #: sim time the proximity gate opened -- the ego first came within
        #: the headway limit of the holder -- or None while it is still shut.
        #: It LATCHES: a cut-in that is genuinely underway must not be
        #: re-gated halfway through, and latching also means the gate cannot
        #: chatter on the headway boundary.
        self._gate_open_at: Optional[float] = None

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
                    self.cutin_start = max(0.0, float(cutin_at))
                    spec["at"] = self.cutin_start
                    # `at` is an EARLIEST time, not an instant. It used to
                    # derive a deadline -- `at + max(lc+2, 4)` -- and the
                    # merge was forced inside it from wherever the actor
                    # happened to be, which at 20 m ahead of the ego is a
                    # lane change on an empty road rather than a cut-in.
                    # Now nothing is forced: the proximity gate decides when
                    # the merge happens and there is no deadline to run out.
                    # The verifier never read `t` (verify_proper_cutin checks
                    # adjacency, success, a commit time, and the merge
                    # station), so dropping it costs the metric nothing.
                    spec["t"] = None
                    # Orchestrate from the start; `at` gates the merge. See
                    # `due`.
                    self._next_tick = 0.0
                self.orch = StickyCutinOrchestrator(spec, cruise_speed=cruise)
                # `load_scenario` ran `resolve_cutins`, which seeded the
                # holder with an authored merge plan seconds before the
                # orchestrator's first tick. The orchestrator owns that plan
                # now and the proximity gate decides when the merge happens,
                # so the seed is dropped in favour of plain cruise: a
                # scenario with a small authored `t` would otherwise begin
                # its lane change before the orchestrator had even woken, and
                # the gate would be deciding about a merge already underway.
                for a in self.sc.actors:
                    if getattr(a, "cutin", None):
                        co.apply_nominal(
                            a, se.actor_cruise_speed(a, default=cruise))
                if cutin_at is not None:
                    hw = float(spec.get("headway", se.CUTIN_NEAR_HEADWAY_S))
                    self._note(0.0, "cast",
                               f"cut-in may begin after t={self.cutin_start:.1f}s "
                               f"and merges once the ego is within {hw:.1f}s "
                               f"headway; no deadline")
        self.sc.simulate()

    # ------------------------------------------------------------------ #
    def _note(self, t: float, kind: str, text: str,
              actor: Optional[str] = None) -> None:
        self.events.append(Event(t=t, kind=kind, text=text, actor=actor))

    def advance(self, dt: float) -> None:
        """Move the script clock forward one CARLA step."""
        self.atime += dt

    def due(self, t_sim: float) -> bool:
        """Whether to orchestrate this tick.

        `at` deliberately does NOT appear here. It gates the MERGE, not the
        orchestration: "the cut-in happens after 3 s" is a statement about
        when the actor may change lanes, not about whether the orchestrator
        is awake. Sleeping until `at` meant the holder ran its authored
        cruise unattended, and with a learned ego that stalls at spawn it
        drifted past 40 m ahead -- where `score_cutin_candidate` collapses to
        0.05, below `cast_roles`'s 0.08 floor, so it was never cast at all
        and the run recorded `holder=None`. Ticking from the start lets the
        wait hold the actor's station until the gate opens.
        """
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
        # Nobody may plan a lane change until the gate has opened, and that
        # has to hold on the tick the holder is FIRST CAST too. On that tick
        # `_recast_if_hopeless` returns early for want of a holder -- it is
        # `orch.tick` below that both casts and plans -- so the gate was
        # never evaluated and the fresh holder planned a chase, whose
        # lane-change burst moved the actor 0.57 m toward the ego at t=0.
        # Every later tick planned a wait, which holds the lateral offset
        # rather than undoing it, so the actor spent the whole run half a
        # metre off its lane centre: a cut-in that visibly starts at t=0 and
        # then stops. Waiting is therefore the DEFAULT while the gate is
        # shut, not something only `_recast_if_hopeless` can switch on.
        if self._gate_open_at is None:
            self.orch.waiting = True
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
        orch.waiting = False
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

        # Two decisions live here and they must not share a test.
        #
        # (a) RECAST -- "is the holder hopeless and is somebody better
        #     available?" -- is judged on the AUTHORED deadline, exactly as it
        #     was before the wait existed. Judging it under the wait's grace
        #     made the holder look feasible for ever, so nothing was ever
        #     hopeless, the role was never handed over, and two of the repo's
        #     own checks went red (`0 recast(s)`, `outcome=None`). Recasting to
        #     a more suitable actor still takes priority over waiting -- the
        #     wait is for when there IS nobody more suitable.
        #
        # (b) CHASE vs WAIT -- reached only once (a) has declined to recast --
        #     is judged on the EXTENDED deadline, the same one in both
        #     directions, with a deadband so the state settles. Using the hard
        #     deadline to enter and the extended one to leave asked an easier
        #     question on the way out than on the way in, and the holder
        #     flipped state on all 17 ticks between t=3.7 and t=7.0 while the
        #     lane change -- planned only by the chase -- was restarted and
        #     discarded every time.
        # (0) THE PROXIMITY GATE, ahead of everything else. Until the ego is
        # close enough to be cut in front of, there is no cut-in to perform
        # and no point asking whether the holder could reach the pin: it
        # waits. The gate latches open, so a merge already underway is never
        # re-gated and the decision cannot chatter on the headway boundary.
        if self._gate_open_at is None:
            earliest = se.cutin_earliest(spec)
            after_earliest = t_sim >= earliest
            near, gap, headway = se.cutin_near_enough(
                pose_of(holder), ego.x, ego.y, theta, ego.v, spec)
            if after_earliest and near:
                self._gate_open_at = t_sim
                limit = float(spec.get("headway", se.CUTIN_NEAR_HEADWAY_S))
                floor = se.cutin_gate_floor(spec)
                self._note(t_sim, "gate",
                           f"ego is within {headway:.2f}s headway of actor "
                           f"{holder.id} at {gap:.1f} m (limit {limit:.1f}s, "
                           f"floor {floor:.1f} m); the cut-in may proceed",
                           actor=str(holder.id))
            else:
                # Not yet. Somebody already in position is more suitable for
                # the action than a holder the ego has not caught up to, so a
                # recast is still allowed -- but only to an actor that is
                # itself near enough, and only once `at` has passed.
                # Otherwise the holder waits, which is also what keeps it
                # from cruising out of casting range.
                orch.sticky_id = str(orch.cutin_id)
                limit = float(spec.get("headway", se.CUTIN_NEAR_HEADWAY_S))
                if not after_earliest:
                    why = (f"the cut-in may not begin before t={earliest:.1f}s "
                           f"(now {t_sim:.1f}s)")
                elif not self._recast_to_nearer(orch, holder, ego, t_sim,
                                                pose_of):
                    floor = se.cutin_gate_floor(spec)
                    why = ((f"it is only {gap:.1f} m ahead, inside the "
                            f"{floor:.1f} m floor — too close to turn in")
                           if gap < floor else
                           (f"the ego is {headway:.1f}s behind it, outside "
                            f"the {limit:.1f}s gate"))
                else:
                    return                 # the role moved to a nearer actor
                self._begin_wait(orch, holder, ego, t_sim, why)
                return

        if se.live_cutin_feasible(pose_of(holder), ego.x, ego.y, theta,
                                  ego.v, spec, t_sim):
            # Still able to make the pin, so it keeps the role — even once it
            # has drifted far enough into the ego's lane that `cast_roles`
            # would score it zero and hand the part to somebody else. See
            # `StickyCutinOrchestrator`.
            orch.sticky_id = str(orch.cutin_id)
            self._resume(orch, holder, t_sim)
            return

        cands = [a for a in self.sc.actors
                 if a is not holder
                 and getattr(a, "autonomy", "auto") != "self"
                 and not getattr(a, "block", None)]
        feasible = [a for a in cands
                    if se.live_cutin_feasible(pose_of(a), ego.x, ego.y, theta,
                                              ego.v, spec, t_sim)]
        if not feasible:
            # Nobody else can make the pin either, so there is no more suitable
            # actor to hand the role to. The holder WAITS rather than running
            # the deadline out: it holds its adjacent lane and eases toward the
            # pin until the geometry comes back or the bounded grace expires.
            orch.sticky_id = str(orch.cutin_id)
            self._wait_or_resume(orch, holder, ego, t_sim,
                                 "no other actor can make the pin")
            return

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
            # Others could reach the pin but none of them is *castable*
            # (`cutin_eligible`: adjacent lane and already ahead), so again
            # there is no more suitable actor. The holder waits.
            orch.sticky_id = str(orch.cutin_id)
            self._wait_or_resume(orch, holder, ego, t_sim,
                                 "no other actor is eligible for the role")
            return
        best = max(feasible, key=lambda a: scores[a.id])
        v_req, _ = se.live_cutin_required_speed(pose_of(holder), ego.x, ego.y,
                                                theta, ego.v, spec, t_sim)
        self._note(t_sim, "cast",
                   f"actor {holder.id} can't make the pin (needs "
                   f"{v_req:.1f} m/s) — recast to {best.id} "
                   f"(score {scores[best.id]:.2f})", actor=str(best.id))
        orch.cutin_id = str(best.id)
        orch.sticky_id = str(best.id)
        orch.waiting = False
        self._waiting_since = None
        orch.n_interventions += 1
        self.n_recasts += 1

    def _recast_to_nearer(self, orch, holder, ego: Ego, t_sim: float,
                          pose_of) -> bool:
        """Hand the role to an actor the ego has ALREADY caught up to.

        Returns True if the role moved. While the gate is shut the question
        "is another actor more suitable?" means "is another actor already in
        position to be cut in front of?", which is a different test from
        `live_cutin_feasible` -- that one asks whether an actor could reach a
        pin, and an actor 40 m up the road can. Casting to a car the ego is
        nowhere near is what the gate exists to prevent, so the candidates
        are filtered on the gate itself and then ranked by the ordinary
        casting score.
        """
        spec = orch.spec
        ego_pose = (ego.x, ego.y, math.degrees(ego.theta))
        lw = self.sc.map.lane_width
        cands = []
        for a in self.sc.actors:
            if a is holder or str(a.id) == "0":
                continue
            if getattr(a, "autonomy", "auto") == "self" or getattr(a, "block", None):
                continue
            near, _, _ = se.cutin_near_enough(pose_of(a), ego.x, ego.y,
                                              ego.theta, ego.v, spec)
            if not near:
                continue
            score = co.score_cutin_candidate(pose_of(a), ego_pose, lw)
            if score > 0.0:
                cands.append((score, a))
        if not cands:
            return False
        score, best = max(cands, key=lambda sa: sa[0])
        self._note(t_sim, "cast",
                   f"the ego has caught up to actor {best.id} but not to "
                   f"actor {holder.id} — recast (score {score:.2f})",
                   actor=str(best.id))
        orch.cutin_id = str(best.id)
        orch.sticky_id = str(best.id)
        orch.waiting = False
        self._waiting_since = None
        orch.n_interventions += 1
        self.n_recasts += 1
        return True

    @staticmethod
    def _wait_grace(orch) -> float:
        """Seconds of grace the holder waits under: the orchestrator's
        override when set, else `spec['wait']`, else the default."""
        if orch.wait_grace_s is not None:
            return float(orch.wait_grace_s)
        return float(orch.spec.get("wait", se.CUTIN_WAIT_GRACE_S))

    def _resume(self, orch, holder, t_sim: float) -> None:
        """Leave the waiting state, noting how long it lasted."""
        if self._waiting_since is None:
            return
        self._note(t_sim, "wait",
                   f"actor {holder.id} can reach the pin again after waiting "
                   f"{t_sim - self._waiting_since:.1f}s — resumes the chase",
                   actor=str(holder.id))
        self._waiting_since = None

    def _wait_or_resume(self, orch, holder, ego: Ego, t_sim: float,
                        why: str) -> None:
        """Decision (b): with nobody more suitable to hand the role to, does
        the holder chase or wait?

        Judged on the wait-extended horizon in BOTH directions, with a speed
        deadband and a minimum dwell so the state cannot chatter at the tick
        rate. A holder that is already waiting has to be
        `CUTIN_WAIT_RESUME_MARGIN` inside the feasible band to go back to
        chasing; one that is not waiting only has to be inside it.
        """
        grace = self._wait_grace(orch)
        waiting_now = self._waiting_since is not None
        dwell_ok = (not waiting_now
                    or (t_sim - self._waiting_since) >= se.CUTIN_WAIT_MIN_DWELL_S)
        margin = se.CUTIN_WAIT_RESUME_MARGIN if waiting_now else 0.0
        pose = holder.pose_at_time(self.atime) if holder.traj else holder.start
        if dwell_ok and se.live_cutin_feasible(pose, ego.x, ego.y, ego.theta,
                                               ego.v, orch.spec, t_sim,
                                               grace=grace, margin=margin):
            # reachable inside the grace: chase it rather than wait
            orch.waiting = False
            self._resume(orch, holder, t_sim)
            return
        self._begin_wait(orch, holder, ego, t_sim, why)

    def _begin_wait(self, orch, holder, ego: Ego, t_sim: float,
                    why: str) -> None:
        """Put the holder into the waiting state and note it once.

        Called only from the branches of `_recast_if_hopeless` that have
        established there is no more suitable actor for the role. The wait is
        the orchestrator's third option next to chase and abandon: the holder
        keeps the part and stops trying to force a merge it cannot finish.
        """
        orch.waiting = True
        if self._waiting_since is not None:
            return                     # already waiting; do not re-note
        self._waiting_since = t_sim
        grace = self._wait_grace(orch)
        v_req, t_rem = se.live_cutin_required_speed(
            holder.pose_at_time(self.atime) if holder.traj else holder.start,
            ego.x, ego.y, ego.theta, ego.v, orch.spec, t_sim)
        self._note(t_sim, "wait",
                   f"actor {holder.id} can't make the pin (needs "
                   f"{v_req:.1f} m/s, {t_rem:.1f}s left) and {why} — "
                   f"waits, deadline +{grace:.1f}s", actor=str(holder.id))

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
            # The wait: how long the holder kept the role while unable to
            # reach the pin, and whether the merge landed only because of the
            # grace. A merge with `merged_after_deadline` true satisfied the
            # scenario's geometry but not its authored timing, so a metric
            # that cares about timing has to read this rather than `outcome`.
            "waited_s": (round(self.orch.waited_s, 3)
                         if self.orch is not None else 0.0),
            "waiting": bool(self.orch.waiting) if self.orch is not None else False,
            "merged_after_deadline": (bool(self.orch.merged_after_deadline)
                                      if self.orch is not None else False),
            "roles": self.roles,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "events": [{"t": round(e.t, 3), "kind": e.kind, "actor": e.actor,
                        "text": e.text} for e in self.events],
        }
        if self.cutin_at is not None:
            out["cutin_at"] = round(float(self.cutin_at), 3)
            out["cutin_start"] = round(float(self.cutin_start), 3)
            if self.orch is not None:
                dl = self.orch.spec.get("t", None)
                out["cutin_deadline"] = (round(float(dl), 3)
                                         if dl is not None else None)
                out["cutin_headway_gate_s"] = round(float(
                    self.orch.spec.get("headway", se.CUTIN_NEAR_HEADWAY_S)), 3)
                out["cutin_gate_open_at"] = (round(self._gate_open_at, 3)
                                             if self._gate_open_at is not None
                                             else None)
                grace = float(self.orch.spec.get("wait", se.CUTIN_WAIT_GRACE_S)
                              if self.orch.wait_grace_s is None
                              else self.orch.wait_grace_s)
                out["cutin_wait_grace"] = round(grace, 3)
        return out
