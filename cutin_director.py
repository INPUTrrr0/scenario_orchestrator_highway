#!/usr/bin/env python3
"""The cut-in orchestrator, shared by the pygame editor and the CARLA runner.

WHAT A CUT-IN IS HERE
---------------------
One actor -- the intention holder -- changes into the ego's lane just in front
of it. The scenario is described by five things:

  1. `trigger`         when the cut-in is ALLOWED: every listed condition
                       must hold (`time_gt`; `ego_speed_gt`, which must hold
                       continuously for more than `ego_speed_hold_s`, 1 s).
  2. `gap_m`           ego FRONT bumper to actor REAR bumper, measured along
                       the road at the CUT-IN INSTANT (default 0.5 m).
  3. `rel_speed_mps`   v_actor - v_ego at the cut-in instant. Positive: the
                       actor comes up faster and crosses the ego's nose.
                       Negative: it cuts in slower and the ego closes on it.
  4. number of actors  one holder plus traffic.
  5. spawns            written per actor, or drawn from a seed.

The CUT-IN INSTANT is the first moment any corner of the actor's body is
inside the ego's lane. Gap, relative speed and time-to-collision are measured
there, from the states the host actually simulated -- not from the plan.

HOW THE HOLDER GETS THERE
-------------------------
The trigger is permission, not a command. Once it fires the holder MAY start
its lane change, and it does so at the first tick the cut-in is reachable:
a lane change started then lands the requested gap and relative speed within
PLAN_MARGIN of the acceleration limits. Until then it keeps getting ready.

Getting ready means planning for the EARLIEST reachable lane-change start at
or after the (predicted) trigger. The trigger time is predicted from the ego's
current speed and acceleration (a time condition is exact), the ego's position
at the resulting cut-in instant is extrapolated, and the holder's longitudinal
plan is solved to arrive there at the requested relative speed. Everything is
re-solved every tick, so the plan follows the ego as it changes what it does.
The holder never matches the ego's speed to wait: it drives a two-phase
acceleration profile toward the rendezvous. If the trigger is not foreseeable
(the ego is not heading for the speed condition), the holder drives its own
cruise speed and nothing happens. If no start within READY_HORIZON_S is
reachable, it closes on the closest miss and keeps looking.

The lane change lasts `lc_duration_s`. Its lateral schedule is fixed from the
moment it starts, so the cut-in instant is known. Between the start and the
cut-in instant the two modes differ:

  * static_commit  the plan solved when the lane change starts is executed as
                   is. The ego's reaction decides whether it is a near miss, a
                   collision or safe.
  * re_aim         the longitudinal plan is re-solved every tick against the
                   ego's latest state, so the 0.5 m / relative-speed geometry
                   is hit whatever the ego does up to the cut-in instant.

From the cut-in instant on, both modes finish the lane change and hold their
speed: a one-shot threat the ego can still brake out of.

MULTIPLE ACTORS
---------------
Every non-ego actor is scored every tick (the probability table): it must be
in a lane adjacent to the ego's, and its score falls as the acceleration its
approach plan needs rises and as its earliest reachable start slips past the
trigger, to zero when no start is reachable. The holder keeps the role while
it can still make the rendezvous, unless another actor is ready
RECAST_READY_MARGIN_S sooner; if it cannot and another actor can, the role
moves. No recasting once the lane change has started. Traffic
keeps its lane and speed and yields to the holder through the existing
collision directive (`cutin_orchestrator.resolve_actor_collisions`).

HOST CONTRACT
-------------
The host owns the clock and the ego. At each orchestration tick it calls

    director.tick(scenario, tau, t, ego)

with `tau` the time into the actors' current plans. The director re-bases
every non-ego actor to now, assigns plans, runs `scenario.simulate()` and the
yield directive; the host then resets its plan clock to zero. Every simulation
step the host calls `director.observe(t, ego, states)` with the vehicles'
actual states, which is where the cut-in instant is detected (interpolated
between steps) and the post-cut-in safety measures are accumulated.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import scenario_editor as se

EGO_ID = "0"
ROAD_HEADING_DEG = 90.0          # straight maps run north: +y forward, -x left

MODE_STATIC_COMMIT = "static_commit"
MODE_RE_AIM = "re_aim"
MODES = (MODE_STATIC_COMMIT, MODE_RE_AIM)

PHASE_APPROACH = "approach"      # before the lane change (trigger fired or not)
PHASE_CUTTING = "cutting"        # lane change started, not yet in the ego's lane
PHASE_CROSSED = "crossed"        # body inside the ego's lane, lane change on
PHASE_MERGED = "merged"          # lane change complete

#: Planning limits for the holder's longitudinal profile, asymmetric the way a
#: real car is: it brakes much harder than it accelerates. A symmetric 4 m/s^2
#: could not follow an IDM ego braking at 7 m/s^2 into the trigger, and the
#: cut-in landed a metre long. A plan outside the limits is infeasible (score
#: 0); A_COMFORT scales the score.
A_MAX = 4.0          # accelerating
D_MAX = 7.0          # braking
A_COMFORT = 1.5
#: How the host draws a lane-changing car's heading, which decides when a
#: corner first enters the ego's lane. The pygame script draws a fixed yaw
#: profile; CARLA derives heading from the direction of motion, capped. At low
#: speed the two differ a lot -- 22 deg of motion yaw against 8 deg scripted --
#: and planning with the wrong one put the measured cut-in 0.12 s early.
HEADING_SCRIPT = "script"
HEADING_MOTION = "motion"
MOTION_YAW_CAP_DEG = 20.0
V_MAX = 40.0
#: Acceleration of a holder that is heading back to its own cruise speed.
CRUISE_ACCEL = 1.5
#: How long the ego is extrapolated at its current acceleration before it is
#: assumed to hold speed -- ZERO: its POSITION is predicted at constant
#: velocity. The plan is re-solved every tick, so genuine acceleration is
#: absorbed as it happens, while extrapolating an estimate amplifies whatever
#: noise the estimate has. Measured across eight timing cases (60 and 15 fps,
#: a start-up frame spike, an autopilot speed dip, braking and accelerating
#: egos, both modes): constant velocity was best or tied in every one, worst
#: gap error 0.22 m, against 0.25 m for a 2 s extrapolation and 0.75 m for a
#: 1 s one. Acceleration is still used to predict WHEN a speed trigger fires.
EGO_ACCEL_HORIZON_S = 0.0
#: The ego's acceleration is a least-squares slope over this window, trusted
#: only once it spans half of it, and ignored below the deadband.
EGO_ACCEL_WINDOW_S = 1.0
EGO_ACCEL_DEADBAND = 0.3
#: Beyond this, a predicted trigger is treated as not foreseeable.
TRIGGER_HORIZON_S = 30.0
#: A speed condition must hold CONTINUOUSLY for longer than this to fire.
SPEED_HOLD_S = 1.0
#: The holder gets ready for the earliest start reachable within this fraction
#: of A_MAX / D_MAX, and starts once it is reachable within the full limits.
#: The gap between the two is hysteresis: the earliest reachable start always
#: sits on the boundary of what is reachable, so checking "go" against the
#: same limits lets every tick's small execution error push it one tick later,
#: forever -- which is what happened without it.
PLAN_MARGIN = 0.85
#: How far past the trigger the earliest reachable start is searched for, and
#: the coarse step of that search (refined to the tick afterwards).
READY_HORIZON_S = 15.0
READY_STEP_S = 0.25
#: The holder moves its planned start earlier only for one reachable at least
#: this much sooner (hysteresis; see `_earliest_start`).
RESTART_SOONER_S = 0.5
#: Score falls with the wait for readiness on this scale; another actor takes
#: the role from a feasible holder only if it is ready this much sooner.
READY_DELAY_SCALE_S = 2.0
RECAST_READY_MARGIN_S = 2.0
#: Plan tail, long enough that no plan runs out between ticks.
TAIL_S = 60.0
#: Seconds of trajectory re-simulated per tick. Hosts sample only the next
#: tick's worth, and the yield directive scans 3 s ahead.
SIM_HORIZON_S = 8.0
#: A holder is "merged" once its centre is this close to the ego lane centre.
MERGED_LAT_TOL_M = 0.5
#: With nobody feasible, a holder gives way only to an actor that would miss
#: the cut-in spot by at least this much less.
RECAST_MISS_MARGIN_M = 2.0
#: The lane change's peak yaw, as `se.Maneuver.pose_at` draws it.
LANE_CHANGE_YAW_DEG = 12.0
#: Traffic in the ego's lane (both opt-in, see CutinSpec). `make_room`: a car
#: ahead of the cut-in spot pulls ahead so that, when the holder lands, its
#: rear bumper is at least ROOM_S0_M + ROOM_T_S * v past the holder's front
#: bumper (and the next car the same past it), accelerating at ROOM_ACCEL, or
#: up to A_MAX when that is too late. After the lane change the room is kept
#: ROOM_AFTER_S ahead of the holder.
ROOM_S0_M = 2.0
ROOM_T_S = 1.0
ROOM_ACCEL = 2.0
ROOM_AFTER_S = 1.0
#: `follow_ego`: a car behind the ego, in its lane or overlapping its body
#: sideways, follows the car ahead of it (the ego first) by IDM with IDM-B's
#: constants, braking no harder than FOLLOW_D_MAX. The plan is FOLLOW_PLAN_S
#: of that acceleration, then its end speed, re-planned every tick.
FOLLOW_T_S = 1.5
FOLLOW_S0_M = 2.0
FOLLOW_A = 1.5
FOLLOW_B = 2.0
FOLLOW_D_MAX = 9.0
FOLLOW_PLAN_S = 0.5
FOLLOW_LAT_MARGIN_M = 0.3


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #
@dataclass
class Trigger:
    """When the cut-in is allowed. Every condition that is set must hold.

    `speed_since` is when the ego's speed last rose above `ego_speed_gt` (None
    while it is not above); the director keeps it from every sample it sees.
    Once fired, the permission is latched: the ego slowing down again later
    does not withdraw it.
    """
    time_gt: Optional[float] = None
    ego_speed_gt: Optional[float] = None
    ego_speed_hold_s: float = SPEED_HOLD_S

    def holds(self, t: float, ego_v: float,
              speed_since: Optional[float] = None) -> bool:
        if self.time_gt is not None and not t > self.time_gt:
            return False
        if self.ego_speed_gt is not None:
            if not ego_v > self.ego_speed_gt or speed_since is None:
                return False
            # strictly longer, and not by float noise on an exact multiple
            if not (t - speed_since > self.ego_speed_hold_s + 1e-6
                    or self.ego_speed_hold_s <= 0.0):
                return False
        return True

    def predict(self, t: float, ego_v: float, ego_a: float,
                speed_since: Optional[float] = None) -> Optional[float]:
        """Absolute time every condition is expected to hold, or None when it
        is not foreseeable from the ego's current motion."""
        when = t
        if self.time_gt is not None:
            when = max(when, self.time_gt + 1e-6)
        if self.ego_speed_gt is not None:
            hold = max(0.0, self.ego_speed_hold_s) + 2e-6
            if ego_v > self.ego_speed_gt:
                since = speed_since if speed_since is not None else t
                when = max(when, since + hold)
            elif ego_a <= 0.05:
                return None
            else:
                when = max(when, t + (self.ego_speed_gt - ego_v) / ego_a + hold)
        if when - t > TRIGGER_HORIZON_S:
            return None
        return when

    def describe(self) -> str:
        parts = []
        if self.time_gt is not None:
            parts.append(f"t > {self.time_gt:g} s")
        if self.ego_speed_gt is not None:
            parts.append(f"ego speed > {self.ego_speed_gt:g} m/s for > "
                         f"{self.ego_speed_hold_s:g} s")
        return " and ".join(parts) or "immediately"

    def to_dict(self) -> dict:
        out = {k: v for k, v in (("time_gt", self.time_gt),
                                 ("ego_speed_gt", self.ego_speed_gt))
               if v is not None}
        if self.ego_speed_gt is not None:
            out["ego_speed_hold_s"] = self.ego_speed_hold_s
        return out


@dataclass
class CutinSpec:
    trigger: Trigger = field(default_factory=Trigger)
    gap_m: float = 0.5
    rel_speed_mps: float = 0.0
    mode: str = MODE_STATIC_COMMIT
    lc_duration_s: float = 2.0
    #: initial intention holder; None lets the score table pick
    holder: Optional[str] = None
    #: notes from translating a legacy spec, surfaced in the report
    notes: List[str] = field(default_factory=list)
    #: ego-lane cars ahead of the cut-in spot pull ahead to leave it free
    make_room: bool = False
    #: cars behind the ego in its lane follow it instead of holding speed
    follow_ego: bool = False
    #: only the holder ever changes lanes, and only one cut-in happens per run
    single_cut_in: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CutinSpec":
        d = dict(d or {})
        legacy = {"at", "along", "lat", "t", "headway", "tail"} & set(d)
        if legacy and "gap_m" not in d and "trigger" not in d:
            return cls.from_legacy(d)
        trig = d.get("trigger") or {}
        mode = str(d.get("mode", MODE_STATIC_COMMIT)).lower().replace("-", "_")
        if mode not in MODES:
            raise ValueError(f"cutin.mode must be one of {MODES}, not {mode!r}")
        return cls(
            trigger=Trigger(
                time_gt=_opt_float(trig.get("time_gt")),
                ego_speed_gt=_opt_float(trig.get("ego_speed_gt")),
                ego_speed_hold_s=float(trig.get("ego_speed_hold_s",
                                                SPEED_HOLD_S))),
            gap_m=float(d.get("gap_m", 0.5)),
            rel_speed_mps=float(d.get("rel_speed_mps", 0.0)),
            mode=mode,
            lc_duration_s=float(d.get("lc_duration_s", 2.0)),
            holder=(str(d["holder"]) if d.get("holder") is not None else None),
            make_room=bool(d.get("make_room", False)),
            follow_ego=bool(d.get("follow_ego", False)),
            single_cut_in=bool(d.get("single_cut_in", False)))

    @classmethod
    def from_legacy(cls, d: dict, lengths: Tuple[float, float] = (4.5, 4.5)
                    ) -> "CutinSpec":
        """An old `{at, along, lat, t, ...}` actor spec, translated.

        `at` was the start time and becomes the trigger; `along` was a
        centre-to-centre distance and becomes a bumper gap; there was no
        relative speed, so it is zero. The deadline `t` and the gate
        `headway` have no counterpart and are dropped.
        """
        notes = []
        at = d.get("at")
        if at is None and d.get("t") is not None:
            at = max(0.0, float(d["t"]) - float(d.get("lc_duration", 2.0)))
            notes.append("legacy `t` (merge deadline) read as a trigger at "
                         "t - lc_duration")
        along = float(d.get("along", 5.0))
        gap = max(0.2, along - 0.5 * (lengths[0] + lengths[1]))
        dropped = sorted({"t", "headway", "tail", "lat"} & set(d))
        if dropped:
            notes.append(f"legacy fields dropped: {', '.join(dropped)}")
        notes.append(f"legacy along={along:g} m centre-to-centre read as "
                     f"gap_m={gap:.2f} m bumper-to-bumper")
        return cls(trigger=Trigger(time_gt=float(at) if at is not None else 0.0),
                   gap_m=gap, rel_speed_mps=0.0, mode=MODE_STATIC_COMMIT,
                   lc_duration_s=float(d.get("lc_duration", 2.0)), notes=notes)

    def to_dict(self) -> dict:
        out = {"trigger": self.trigger.to_dict(), "gap_m": self.gap_m,
               "rel_speed_mps": self.rel_speed_mps, "mode": self.mode,
               "lc_duration_s": self.lc_duration_s}
        if self.make_room:
            out["make_room"] = True
        if self.follow_ego:
            out["follow_ego"] = True
        if self.single_cut_in:
            out["single_cut_in"] = True
        return out


@dataclass
class SpawnSpec:
    """Where the non-ego actors start, relative to the ego.

    Explicit entries: `{lane: left|right|same|<int relative>, offset_m: <m,
    centre-to-centre along the road, + ahead>, speed_mps: <m/s>}`.
    Random: `{seed, count, lanes: [left, right], offset_m: [lo, hi],
    speed_mps: [lo, hi], ego_clear_m}` (`same` in `lanes` draws into the ego's
    lane, no closer than `ego_clear_m`, default 6, centre to centre). A random draw is kept whatever it implies for the
    cut-in (the outcome is recorded); only bodies that would overlap at spawn
    are redrawn, because CARLA cannot spawn them.
    """
    actors: List[dict] = field(default_factory=list)
    random: Optional[dict] = None
    count: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["SpawnSpec"]:
        if not d:
            return None
        return cls(actors=list(d.get("actors") or []),
                   random=dict(d["random"]) if d.get("random") else None,
                   count=(int(d["count"]) if d.get("count") is not None
                          else None))


@dataclass
class EgoState:
    x: float
    y: float
    heading_deg: float
    v: float
    length: float = 4.5
    width: float = 2.0


def _opt_float(v) -> Optional[float]:
    return None if v is None else float(v)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _axes(heading_deg: float) -> Tuple[float, float, float, float]:
    h = math.radians(heading_deg)
    return math.cos(h), math.sin(h), -math.sin(h), math.cos(h)


def along_of(x: float, y: float) -> float:
    fx, fy, _, _ = _axes(ROAD_HEADING_DEG)
    return x * fx + y * fy


def left_of(x: float, y: float) -> float:
    _, _, nx, ny = _axes(ROAD_HEADING_DEG)
    return x * nx + y * ny


def lane_index(m: se.MapConfig, x: float) -> int:
    n = max(1, m.num_lanes)
    half = n * m.lane_width / 2.0
    return int(se.clamp(round((x + half - m.lane_width / 2.0) / m.lane_width),
                        0, n - 1))


def corners(x: float, y: float, heading_deg: float, length: float,
            width: float) -> List[Tuple[float, float]]:
    fx, fy, nx, ny = _axes(heading_deg)
    hl, hw = length / 2.0, width / 2.0
    return [(x + sa * hl * fx + sb * hw * nx, y + sa * hl * fy + sb * hw * ny)
            for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def lane_margin(x: float, y: float, heading_deg: float, length: float,
                width: float, lane_x: float, lane_width: float) -> float:
    """How far the body is from entering the lane centred at `lane_x`:
    positive outside, <= 0 once any corner is inside."""
    return min(abs(cx - lane_x) for cx, _ in
               corners(x, y, heading_deg, length, width)) - lane_width / 2.0


def crossing_fraction(lane_width: float, length: float, width: float,
                      heading: str = HEADING_SCRIPT, speed: float = 10.0,
                      lc_duration: float = 2.0) -> float:
    """Lateral progress u at which a lane change of one lane width puts a
    corner of the body into the target lane, with the yaw the host will draw
    (see HEADING_*). Solved by bisection."""
    def yaw_deg(u: float) -> float:
        if heading == HEADING_MOTION:
            v_lat = lane_width * 6.0 * u * (1.0 - u) / max(1e-6, lc_duration)
            return min(MOTION_YAW_CAP_DEG,
                       math.degrees(math.atan2(v_lat, max(speed, 0.5))))
        return LANE_CHANGE_YAW_DEG * math.sin(math.pi * u)

    def extent(u: float) -> float:
        yaw = math.radians(yaw_deg(u))
        return (lane_width * se._smoothstep(u) + 0.5 * length * math.sin(yaw)
                + 0.5 * width * math.cos(yaw))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if extent(mid) >= lane_width / 2.0:
            hi = mid
        else:
            lo = mid
    return hi


def _u_at(dt_from_start: float, lc_duration: float) -> float:
    return se.clamp(dt_from_start / max(1e-6, lc_duration), 0.0, 1.0)


def _time_at_u(u: float, lc_duration: float) -> float:
    """Inverse of _u_at for the linear u clock."""
    return u * lc_duration


# --------------------------------------------------------------------------- #
# Longitudinal planning
# --------------------------------------------------------------------------- #
@dataclass
class Approach:
    """A longitudinal profile reaching (distance, speed) after tau.

    Two constant-acceleration phases, a1 for tau1 then a2 for the rest, with
    the switch time chosen to MINIMISE THE PEAK acceleration (each phase as a
    fraction of the limit on its own side, accelerating or braking). That choice is
    what makes it work as a receding-horizon planner: while the actor is on
    track, re-solving next tick gives the same peak, so the first phase is
    never deferred. Both simpler options failed in testing -- a switch fixed
    at tau/2 kept executing the gentle half and saturated at the end, and the
    minimum-effort cubic puts its peak at the endpoints and ran out of
    acceleration early. Constant-acceleration phases are also exactly what a
    maneuver can express, so the plan and its execution agree to the metre.
    """
    tau: float
    v0: float
    tau1: float
    a0: float           # first phase
    a_end: float        # second phase
    v_end: float
    feasible: bool
    along_miss_m: float
    speed_miss_mps: float
    samples: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def peak_accel(self) -> float:
        return max(abs(self.a0), abs(self.a_end))

    def v_at(self, dt: float) -> float:
        if dt <= 0.0:
            return self.v0
        if dt <= self.tau1:
            return max(0.0, self.v0 + self.a0 * dt)
        if dt <= self.tau:
            v_m = max(0.0, self.v0 + self.a0 * self.tau1)
            return max(0.0, v_m + self.a_end * (dt - self.tau1))
        return self.v_end


def _two_phase(v0: float, distance: float, v_target: float, tau: float,
               tau1: float) -> Tuple[float, float]:
    """(a1, a2) hitting distance and speed with the switch at tau1."""
    tau2 = tau - tau1
    # v0 + a1*tau1 + a2*tau2 = V
    # v0*tau + a1*(tau1^2/2 + tau1*tau2) + a2*tau2^2/2 = D
    m11, m12, r1 = tau1, tau2, v_target - v0
    m21, m22, r2 = 0.5 * tau1 * tau1 + tau1 * tau2, 0.5 * tau2 * tau2, distance - v0 * tau
    det = m11 * m22 - m12 * m21
    if abs(det) < 1e-12:
        return float("inf"), float("inf")
    return (r1 * m22 - m12 * r2) / det, (m11 * r2 - r1 * m21) / det


def _switch_window(v0: float, distance: float, v_target: float, tau: float,
                   a_max: float, d_max: float, v_max: float
                   ) -> Optional[Tuple[float, float]]:
    """The switch times tau1 at which a two-phase profile reaching (distance,
    v_target) after tau stays within the limits with 0 <= v <= v_max, as an
    interval -- or None when there are none.

    It IS an interval: the distance fixes the speed at the switch,
    v_m = p + q*tau1, and then every limit (on v_m, on a1 = (v_m-v0)/tau1 and
    on a2 = (V-v_m)/(tau-tau1)) is linear in tau1. Exact, where searching for
    the gentlest switch first and checking the limits after is not: at low
    speed that search picked switches whose v_m was negative, called the
    rendezvous unreachable, and pushed a crawling ego's cut-in ten seconds
    later than it needed to be."""
    p = (2.0 * distance - v_target * tau) / tau
    q = (v_target - v0) / tau
    lo, hi = 1e-6 * tau, tau * (1.0 - 1e-6)
    tol = 1e-9
    # each (alpha, beta) is the limit alpha + beta * tau1 >= 0
    for alpha, beta in ((p, q),                                   # v_m >= 0
                        (v_max - p, -q),                          # v_m <= v_max
                        (p - v0, q + d_max),                      # a1 >= -d_max
                        (v0 - p, a_max - q),                      # a1 <= a_max
                        (v_target - p + d_max * tau, -q - d_max),  # a2 >= -d_max
                        (a_max * tau - v_target + p, q - a_max)):  # a2 <= a_max
        alpha += tol
        if abs(beta) < 1e-12:
            if alpha < 0.0:
                return None
        elif beta > 0.0:
            lo = max(lo, -alpha / beta)
        else:
            hi = min(hi, -alpha / beta)
        if lo > hi:
            return None
    return lo, hi


def reachable(v0: float, distance: float, v_target: float, tau: float,
              a_max: float = A_MAX, d_max: float = D_MAX,
              v_max: float = V_MAX) -> bool:
    """Can a two-phase profile within the limits arrive `distance` ahead at
    `v_target` after `tau`?"""
    v0 = max(0.0, v0)
    if tau <= 1e-3:
        return abs(v0 - v_target) < 0.5
    return (v_target >= 0.0 and
            _switch_window(v0, distance, v_target, tau, a_max, d_max, v_max)
            is not None)


def solve_approach(v0: float, distance: float, v_target: float, tau: float,
                   a_max: float = A_MAX, v_max: float = V_MAX,
                   d_max: float = D_MAX, position_first: bool = True,
                   fast: bool = False) -> Approach:
    """`fast` uses a coarser switch-time search, for estimating the miss of an
    unreachable rendezvous."""
    v0 = max(0.0, v0)
    if tau <= 1e-3:
        return Approach(tau, v0, 0.0, 0.0, 0.0, v0, abs(v0 - v_target) < 0.5,
                        distance, v0 - v_target)

    def norm(a: float) -> float:
        # against the limit on its own side: braking 6 then accelerating 3
        # fits a 7/4 car, while the |a|-balanced 4.5/4.5 does not
        return a / a_max if a >= 0.0 else -a / d_max

    n_golden = 20 if fast else 40
    window = (_switch_window(v0, distance, v_target, tau, a_max, d_max, v_max)
              if v_target >= 0.0 else None)
    if window is not None:
        p = (2.0 * distance - v_target * tau) / tau
        q = (v_target - v0) / tau

        def accels(t1: float) -> Tuple[float, float, float]:
            v_m = p + q * t1
            return v_m, (v_m - v0) / t1, (v_target - v_m) / (tau - t1)

        def wpeak(t1: float) -> float:
            _, a1, a2 = accels(t1)
            return max(norm(a1), norm(a2))
        lo, hi = window                 # the peak is quasi-convex in tau1
        for _ in range(n_golden):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            if wpeak(m1) <= wpeak(m2):
                hi = m2
            else:
                lo = m1
        tau1 = 0.5 * (lo + hi)
        v_m, a1, a2 = accels(tau1)
        ap = Approach(tau, v0, tau1, a1, a2, v_target, True, 0.0, 0.0)
        ap.samples = [(0.0, v0), (tau1, max(0.0, v_m)), (tau, v_target)]
        return ap

    def within(a: float) -> bool:
        return -d_max - 1e-9 <= a <= a_max + 1e-9

    def peak(f: float) -> float:
        a1, a2 = _two_phase(v0, distance, v_target, tau, f * tau)
        return max(norm(a1), norm(a2))

    if position_first:
        # The speed is out of reach but the POSITION may not be: land on the
        # spot at the closest speed that can be. Missing the gap is what makes
        # a cut-in into something else; missing the speed only changes how
        # sharp it is.
        a_free = 2.0 * (distance - v0 * tau) / (tau * tau)
        v_free = v0 + a_free * tau
        if within(a_free) and 0.0 <= v_free <= v_max:
            lo_v, hi_v = v_free, se.clamp(v_target, 0.0, v_max)
            for _ in range(30):
                mid = 0.5 * (lo_v + hi_v)
                if reachable(v0, distance, mid, tau, a_max, d_max, v_max):
                    lo_v = mid
                else:
                    hi_v = mid
            ap = solve_approach(v0, distance, lo_v, tau, a_max, v_max,
                                d_max=d_max, position_first=False)
            if not ap.feasible:     # numerically on the edge: take v_free
                ap = solve_approach(v0, distance, v_free, tau, a_max, v_max,
                                    d_max=d_max, position_first=False)
            ap.feasible = False
            ap.speed_miss_mps = ap.v_end - v_target
            return ap
        return solve_approach(v0, distance, v_target, tau, a_max, v_max,
                              d_max=d_max, position_first=False, fast=fast)
    else:
        # Unreachable: the gentlest unconstrained profile, clamped to the
        # limits, and how far short of the spot and the speed it ends.
        n_grid = 16 if fast else 48
        grid = [0.02 + 0.96 * k / n_grid for k in range(n_grid + 1)]
        f = min(grid, key=peak)
        step = 0.96 / n_grid
        lo, hi = max(0.01, f - step), min(0.99, f + step)
        for _ in range(n_golden):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            if peak(m1) <= peak(m2):
                hi = m2
            else:
                lo = m1
        tau1 = 0.5 * (lo + hi) * tau
        a1, a2 = _two_phase(v0, distance, v_target, tau, tau1)
        a1c, a2c = se.clamp(a1, -d_max, a_max), se.clamp(a2, -d_max, a_max)
        v_mc = se.clamp(v0 + a1c * tau1, 0.0, v_max)
        v_endc = se.clamp(v_mc + a2c * (tau - tau1), 0.0, v_max)
        covered = 0.5 * (v0 + v_mc) * tau1 + 0.5 * (v_mc + v_endc) * (tau - tau1)
        ap = Approach(tau, v0, tau1, a1c, a2c, v_endc, False,
                      distance - covered, v_endc - v_target)
    ap.samples = [(0.0, v0), (ap.tau1, ap.v_at(ap.tau1)), (tau, ap.v_at(tau))]
    return ap


def predict_ego(along: float, v: float, a: float, dt: float) -> Tuple[float, float]:
    """Ego (along, speed) after dt: current acceleration for at most
    EGO_ACCEL_HORIZON_S, then constant speed; never backwards."""
    ta = min(max(dt, 0.0), EGO_ACCEL_HORIZON_S)
    if a < 0.0 and v + a * ta < 0.0:
        ta_stop = -v / a
        return along + v * ta_stop + 0.5 * a * ta_stop * ta_stop, 0.0
    v1 = se.clamp(v + a * ta, 0.0, V_MAX)
    s1 = along + 0.5 * (v + v1) * ta
    return s1 + v1 * max(0.0, dt - ta), v1


# --------------------------------------------------------------------------- #
# The director
# --------------------------------------------------------------------------- #
class CutinDirector:
    def __init__(self, spec: CutinSpec, scenario: se.Scenario,
                 ego_id: str = EGO_ID,
                 dims: Optional[Dict[str, Tuple[float, float]]] = None,
                 heading: str = HEADING_SCRIPT):
        self.spec = spec
        self.heading = heading
        self.map = scenario.map
        self.ego_id = str(ego_id)
        self.dims: Dict[str, Tuple[float, float]] = {
            a.id: (a.length, a.width) for a in scenario.actors}
        if dims:
            self.dims.update({str(k): v for k, v in dims.items()})
        self.cruise: Dict[str, float] = {
            a.id: se.actor_cruise_speed(a) for a in scenario.actors
            if a.id != self.ego_id}
        self.holder: Optional[str] = spec.holder
        self.phase = PHASE_APPROACH
        #: when the trigger fired (the cut-in became allowed)
        self.t_trigger: Optional[float] = None
        #: when the holder actually started its lane change
        self.t_lc_start: Optional[float] = None
        #: the lane-change start the holder is currently getting ready for
        self.t_start_plan: Optional[float] = None
        self.t_cross_plan: Optional[float] = None
        self.lat_offset: Optional[float] = None       # holder's lane change
        self.scores: Dict[str, float] = {}
        self.plan_info: Dict[str, dict] = {}
        self.events: List[dict] = []
        self.crossing: Optional[dict] = None
        self.merged_at: Optional[float] = None
        self.n_recasts = 0
        self.post: dict = {"min_gap_m": None, "min_ttc_s": None,
                           "overlap": False}
        self._ego_hist: List[Tuple[float, float]] = []
        self._prev_obs: Optional[dict] = None
        self._hold_after_cross = False
        self._committed_plan = False
        self._tick_dt: Optional[float] = None
        self._last_tick_t: Optional[float] = None
        #: when the ego's speed last rose above the trigger speed
        self._speed_since: Optional[float] = None
        self._speed_last: Optional[Tuple[float, float]] = None
        #: per actor, this tick: (earliest reachable start, reachable?)
        self._ready: Dict[str, Tuple[float, bool]] = {}
        self._foreseeable = False
        self._waiting_noted = False
        #: ego-lane cars already reported as making room (one event each)
        self._room_noted: set = set()
        #: the lane the holder's lane change aims at, once it has started
        self._land_lane: Optional[int] = None
        #: cars follow_ego has driven; they keep following (see _follow_ego)
        self._followed: set = set()
        #: cars seen in the ego's lane; never cast (only with make_room/follow_ego)
        self._shared_lane: set = set()
        for note in spec.notes:
            self._event(0.0, "spec", note)
        self._event(0.0, "spec",
                    f"cut-in allowed when {spec.trigger.describe()}: gap "
                    f"{spec.gap_m:g} m, relative speed "
                    f"{spec.rel_speed_mps:+g} m/s, mode {spec.mode}, lane "
                    f"change {spec.lc_duration_s:g} s")

    # ---- bookkeeping ---- #
    def _event(self, t: float, kind: str, text: str, **data) -> None:
        ev = {"t": round(float(t), 3), "kind": kind, "text": text}
        ev.update(data)
        self.events.append(ev)

    def _ego_accel(self) -> float:
        h = self._ego_hist
        if len(h) < 3 or h[-1][0] - h[0][0] < 0.5 * EGO_ACCEL_WINDOW_S:
            return 0.0
        n = len(h)
        mt = sum(t for t, _ in h) / n
        mv = sum(v for _, v in h) / n
        var = sum((t - mt) ** 2 for t, _ in h)
        if var < 1e-9:
            return 0.0
        a = sum((t - mt) * (v - mv) for t, v in h) / var
        return 0.0 if abs(a) < EGO_ACCEL_DEADBAND else se.clamp(a, -9.0, 5.0)

    def _note_speed(self, t: float, v: float) -> None:
        """Track how long the ego has been above the trigger speed, from every
        sample either host call provides. The crossing is interpolated."""
        thr = self.spec.trigger.ego_speed_gt
        if thr is None:
            return
        last = self._speed_last
        if last is not None and t <= last[0] + 1e-9:
            return
        if v > thr:
            if self._speed_since is None:
                if last is not None and last[1] <= thr and v > last[1]:
                    self._speed_since = (last[0] + (thr - last[1])
                                         / (v - last[1]) * (t - last[0]))
                else:
                    self._speed_since = t
        else:
            self._speed_since = None
        self._speed_last = (t, v)

    def role_of(self, aid: str) -> str:
        if str(aid) == self.ego_id:
            return "ego"
        return "holder" if str(aid) == self.holder else "traffic"

    # ---- the tick ---- #
    def _predict_fire(self, t: float, ego_v: float, a_ego: float) -> Optional[float]:
        """When the trigger will FIRE, not merely hold: the host checks it
        once per tick, so the first tick at or after the condition time. A
        condition that does not hold now cannot fire before the next tick --
        planning for 'now' would start the lane change a tick early. Once the
        trigger has fired, the earliest start is now."""
        if self.t_trigger is not None:
            return t
        cond = self.spec.trigger.predict(t, ego_v, a_ego, self._speed_since)
        if cond is None:
            return None
        dt = self._tick_dt or 0.05
        n = max(1, math.ceil((cond - t) / dt - 1e-9))
        return t + n * dt

    def tick(self, sc: se.Scenario, tau: float, t: float, ego: EgoState) -> None:
        if self._last_tick_t is not None and t - self._last_tick_t > 1e-6:
            self._tick_dt = t - self._last_tick_t
        self._last_tick_t = t
        self._note_speed(t, ego.v)
        self._ego_hist.append((t, ego.v))
        while len(self._ego_hist) > 2 and t - self._ego_hist[0][0] > EGO_ACCEL_WINDOW_S:
            self._ego_hist.pop(0)
        self.dims.setdefault(self.ego_id, (ego.length, ego.width))
        self.dims[self.ego_id] = (ego.length, ego.width)
        a_ego = self._ego_accel()

        states: Dict[str, Tuple[se.Pose, float]] = {}
        for a in sc.actors:
            if a.id == self.ego_id:
                continue
            states[a.id] = self._continue(a, tau)
        sc.simulate(horizon=SIM_HORIZON_S)

        ego_lane_x = self.map.lane_center_x(lane_index(self.map, ego.x))
        if self.spec.make_room or self.spec.follow_ego:
            # With traffic in the ego's lane, a car the ego has shared a lane
            # with is never cast: an ego that changes lanes to pass a slower one
            # would otherwise have it cut straight back in beside it.
            ego_lane = lane_index(self.map, ego.x)
            self._shared_lane.update(aid for aid, (pose, _) in states.items()
                                     if lane_index(self.map, pose[0]) == ego_lane)

        if self.phase == PHASE_APPROACH:
            if (self.t_trigger is None
                    and self.spec.trigger.holds(t, ego.v, self._speed_since)):
                self._permit(t, ego)
            self._score_and_cast(sc, t, ego, a_ego, ego_lane_x, states)
            if self.t_trigger is not None and self.holder is not None:
                t_go, reachable = self._ready.get(self.holder, (math.inf, False))
                if reachable and t_go <= t + 1e-6:
                    self._start_lane_change(t, ego, ego_lane_x, states)
                elif not self._waiting_noted:
                    self._waiting_noted = True
                    self._event(t, "wait",
                                f"actor {self.holder} is not ready to cut in yet: "
                                + (f"earliest reachable start t={t_go:.2f}s"
                                   if reachable else
                                   f"no start within {READY_HORIZON_S:g} s is "
                                   "reachable, closing in"),
                                actor=self.holder)

        holder = next((a for a in sc.actors if a.id == self.holder), None)
        for a in sc.actors:
            a.cutin = None
        if holder is not None:
            holder.cutin = self.spec.to_dict()      # never yields to traffic
            self._plan_holder(holder, t, ego, a_ego, ego_lane_x, states)
        if self.spec.single_cut_in:
            self._keep_one_cut_in(sc, states)
        sc.simulate(horizon=SIM_HORIZON_S)
        pushed = (self._make_room(sc, holder, t, ego, a_ego, states)
                  if self.spec.make_room and holder is not None else set())
        if pushed:
            sc.simulate(horizon=SIM_HORIZON_S)
        self._yields(sc)
        # last: a pairwise yield may have sped a follower up; IDM already brakes
        # it for whatever it would have yielded to
        if self.spec.follow_ego and self._follow_ego(sc, ego, states, pushed):
            sc.simulate(horizon=SIM_HORIZON_S)

    # ---- phases ---- #
    def _continue(self, a: se.Actor, tau: float) -> Tuple[se.Pose, float]:
        """Re-base `a` to now, keeping the rest of its plan, and return its
        current (plan-heading pose, speed)."""
        import maneuvers as mv
        if not a.traj:
            a.compute_schedule()
        speed = (a.speeds[max(0, min(int(round(tau / se.DT)), len(a.speeds) - 1))]
                 if a.speeds else se.actor_cruise_speed(a))
        new = mv.rebase_actor(a, tau)
        a.start, a.maneuvers = new.start, new.maneuvers
        remaining = sum(m.duration for m in a.maneuvers)
        if remaining < TAIL_S / 2.0:
            v_tail = (a.maneuvers[-1].exit_speed() if a.maneuvers
                      and isinstance(a.maneuvers[-1], se.Maneuver) else speed)
            a.maneuvers = list(a.maneuvers) + [se.Maneuver(
                type="go_straight", duration=TAIL_S, intercept=max(0.0, v_tail))]
        return a.start, speed

    def _rendezvous(self, aid: str, pose: se.Pose, t: float, ego: EgoState,
                    a_ego: float, t_cross: float) -> Tuple[float, float, float]:
        """(distance to cover, speed to arrive at, time left) for a cut-in
        instant at `t_cross`."""
        le, _ = self.dims[self.ego_id]
        la, _ = self.dims.get(aid, (4.5, 2.0))
        s_e_now = along_of(ego.x, ego.y)
        s_e, v_e = predict_ego(s_e_now, ego.v, a_ego, t_cross - t)
        s_target = s_e + 0.5 * (le + la) + self.spec.gap_m
        # a car cannot drive backwards: "2 m/s slower than a stopped ego" is
        # zero, and the miss in relative speed is recorded, not planned for
        v_target = max(0.0, v_e + self.spec.rel_speed_mps)
        return s_target - along_of(pose[0], pose[1]), v_target, t_cross - t

    def _approach_for(self, aid: str, pose: se.Pose, v: float, t: float,
                      ego: EgoState, a_ego: float, t_cross: float,
                      scale: float = 1.0, position_first: bool = True,
                      fast: bool = False) -> Approach:
        dist, v_target, tau = self._rendezvous(aid, pose, t, ego, a_ego, t_cross)
        return solve_approach(v, dist, v_target, tau, a_max=A_MAX * scale,
                              d_max=D_MAX * scale,
                              position_first=position_first, fast=fast)

    def _earliest_start(self, aid: str, pose: se.Pose, v: float, t: float,
                        ego: EgoState, a_ego: float, base: float,
                        u_star: float) -> Tuple[float, bool]:
        """When the lane change should start: (start, True) for a reachable
        one, else (the start whose approach misses the spot by least, False).

        Once the trigger has fired (`base` is now), the answer is "now" if a
        lane change started now reaches the cut-in within the full limits.
        The holder keeps the start it is already getting ready for while that
        stays reachable within the full limits. Otherwise it is the earliest
        tick that reaches it within PLAN_MARGIN of them -- see PLAN_MARGIN for
        why the two differ."""
        lc = self.spec.lc_duration_s
        dt = self._tick_dt or 0.05
        stride = max(1, int(round(READY_STEP_S / dt)))
        n_max = max(1, int(READY_HORIZON_S / dt))

        def ok(k: int, scale: float = PLAN_MARGIN) -> bool:
            dist, v_target, tau = self._rendezvous(
                aid, pose, t, ego, a_ego, base + k * dt + u_star * lc)
            return reachable(v, dist, v_target, tau, A_MAX * scale, D_MAX * scale)
        k0 = 0
        if base <= t + 1e-6:
            if ok(0, scale=1.0):
                return base, True
            k0 = 1
        earliest = None
        prev = None
        for k in range(k0, n_max + 1, stride):
            if ok(k):
                lo, hi = (prev if prev is not None else k), k
                while hi - lo > 1:                  # refine to the tick
                    mid = (lo + hi) // 2
                    if ok(mid):
                        hi = mid
                    else:
                        lo = mid
                earliest = hi
                break
            prev = k
        # Keep the start being prepared for while it stays reachable, unless
        # a clearly earlier one now is. Taking the earliest every tick put the
        # plan back on the PLAN_MARGIN boundary each time, where it flipped
        # between "accelerate now" and "brake, then accelerate later" and the
        # start slid away for good; keeping it unconditionally held on to a
        # start chosen while the ego was stopped long after it drove off.
        keep = self.t_start_plan if aid == self.holder else None
        if keep is not None and keep > base + 0.5 * dt:
            k_keep = int(round((keep - base) / dt))
            if (k_keep >= k0 and ok(k_keep, scale=1.0)
                    and (earliest is None
                         or earliest * dt > k_keep * dt - RESTART_SOONER_S)):
                return base + k_keep * dt, True
        if earliest is not None:
            return base + earliest * dt, True

        def miss(k: int) -> float:
            return abs(self._approach_for(
                aid, pose, v, t, ego, a_ego, base + k * dt + u_star * lc,
                scale=PLAN_MARGIN, position_first=False, fast=True).along_miss_m)
        best = min(range(k0, n_max + 1, stride), key=miss)
        return base + best * dt, False

    def _u_star(self, aid: str, ego_v: float) -> float:
        la, wa = self.dims.get(aid, (4.5, 2.0))
        return crossing_fraction(self.map.lane_width, la, wa, self.heading,
                                 max(0.0, ego_v + self.spec.rel_speed_mps),
                                 self.spec.lc_duration_s)

    def _score_and_cast(self, sc, t, ego, a_ego, ego_lane_x, states) -> None:
        trig = self._predict_fire(t, ego.v, a_ego)
        self._foreseeable = trig is not None
        t_trig = trig if trig is not None else t + (self._tick_dt or 0.05)
        scores: Dict[str, float] = {}
        info: Dict[str, dict] = {}
        ready: Dict[str, Tuple[float, bool]] = {}
        ego_lane = lane_index(self.map, ego.x)
        lc = self.spec.lc_duration_s
        for a in sc.actors:
            if a.id == self.ego_id or getattr(a, "autonomy", "auto") == "self":
                continue
            pose, v = states[a.id]
            lane = lane_index(self.map, pose[0])
            if abs(lane - ego_lane) != 1:
                scores[a.id] = 0.0
                info[a.id] = {"eligible": False, "why": "not in a lane next to the ego"}
                continue
            if a.id in self._shared_lane:
                scores[a.id] = 0.0
                info[a.id] = {"eligible": False, "why": "has shared the ego's lane"}
                continue
            u_star = self._u_star(a.id, ego.v)
            t_go, reachable = self._earliest_start(a.id, pose, v, t, ego, a_ego,
                                                   t_trig, u_star)
            ready[a.id] = (t_go, reachable)
            ap = self._approach_for(a.id, pose, v, t, ego, a_ego,
                                    t_go + _time_at_u(u_star, lc))
            delay = max(0.0, t_go - t_trig)
            scores[a.id] = (1.0 / (1.0 + ap.peak_accel / A_COMFORT)
                            / (1.0 + delay / READY_DELAY_SCALE_S)
                            if reachable else 0.0)
            info[a.id] = {"eligible": True, "feasible": reachable,
                          "ready_at_s": round(t_go, 3),
                          "ready_delay_s": round(delay, 3),
                          "peak_accel_mps2": round(ap.peak_accel, 3),
                          "along_miss_m": round(ap.along_miss_m, 3)}
        self.scores, self.plan_info, self._ready = scores, info, ready

        holder_ok = (self.holder is not None and scores.get(self.holder, 0.0) > 0.0)
        if holder_ok:
            mine = ready[self.holder][0]
            sooner = min(((ready[aid][0], aid) for aid, s in scores.items()
                          if s > 0.0 and aid != self.holder), default=None)
            if sooner is not None and sooner[0] <= mine - RECAST_READY_MARGIN_S:
                self.n_recasts += 1
                self._cast(sooner[1], t, f"actor {sooner[1]} is ready "
                           f"{mine - sooner[0]:.1f} s sooner "
                           f"(t={sooner[0]:.2f}s vs t={mine:.2f}s)")
            return
        best = max(((s, aid) for aid, s in scores.items() if s > 0.0),
                   default=None)
        if best is not None:
            if best[1] != self.holder:
                why = ("holder can no longer make the cut-in"
                       if self.holder is not None else "initial cast")
                if self.holder is not None:
                    self.n_recasts += 1
                self._cast(best[1], t, f"{why} (score {best[0]:.2f})")
            return
        # Nobody can make it exactly. The attempt is still made -- by the
        # actor that would miss by least -- and the miss is recorded rather
        # than hidden. An infeasible holder gives way to a clearly closer miss.
        elig = [aid for aid, i in info.items() if i.get("eligible")]
        if not elig:
            return
        pick = min(elig, key=lambda k: abs(info[k]["along_miss_m"]))
        if self.holder is None:
            self._cast(pick, t, "no actor can reach the cut-in exactly; the "
                       "closest attempts it")
            return
        cur = info.get(self.holder)
        cur_miss = (abs(cur["along_miss_m"]) if cur and cur.get("eligible")
                    else float("inf"))
        if pick != self.holder and abs(info[pick]["along_miss_m"]) < cur_miss - RECAST_MISS_MARGIN_M:
            self.n_recasts += 1
            self._cast(pick, t, f"holder cannot make the cut-in and actor "
                       f"{pick} misses by less ({abs(info[pick]['along_miss_m']):.1f} m "
                       f"vs {cur_miss:.1f} m)")

    def _cast(self, aid: str, t: float, why: str) -> None:
        old = self.holder
        self.holder = aid
        self.t_start_plan = None
        self._event(t, "cast", f"actor {aid} holds the cut-in"
                    + (f", replacing {old}" if old else "") + f": {why}",
                    actor=aid, previous=old)

    def _permit(self, t: float, ego: EgoState) -> None:
        self.t_trigger = t
        self._event(t, "trigger",
                    f"{self.spec.trigger.describe()} holds: the cut-in is "
                    f"allowed" + (f" (holder: actor {self.holder})"
                                  if self.holder else ""),
                    actor=self.holder, ego_speed_mps=round(ego.v, 3))

    def _start_lane_change(self, t, ego, ego_lane_x, states) -> None:
        pose, _ = states[self.holder]
        u_star = self._u_star(self.holder, ego.v)
        self.t_lc_start = t
        self.t_cross_plan = t + _time_at_u(u_star, self.spec.lc_duration_s)
        # lateral offset in the maneuver's convention: + is left of heading
        self.lat_offset = left_of(ego_lane_x, 0.0) - left_of(pose[0], 0.0)
        self._land_lane = lane_index(self.map, ego_lane_x)
        self.phase = PHASE_CUTTING
        waited = t - self.t_trigger
        self._event(t, "lane_change",
                    f"actor {self.holder} starts its lane change "
                    f"({'left' if self.lat_offset > 0 else 'right'})"
                    + (f" {waited:.2f} s after the trigger, once the cut-in "
                       f"was reachable" if waited > 1e-6 else "")
                    + f"; cut-in instant planned at t={self.t_cross_plan:.2f}s",
                    actor=self.holder, ego_speed_mps=round(ego.v, 3),
                    waited_s=round(waited, 3))

    def _plan_holder(self, a, t, ego, a_ego, ego_lane_x, states) -> None:
        pose, v = states[a.id]
        lc = self.spec.lc_duration_s
        road = (pose[0], pose[1], ROAD_HEADING_DEG)
        if self.phase == PHASE_APPROACH:
            if not self._foreseeable or a.id not in self._ready:
                a.start, a.maneuvers = road, _cruise_toward(v, self.cruise.get(a.id, v))
                self.plan_info.setdefault(a.id, {})["plan"] = "cruise (trigger not foreseeable)"
                self.t_start_plan = None
                return
            t_go, reachable = self._ready[a.id]
            self.t_start_plan = t_go if reachable else None
            u_star = self._u_star(a.id, ego.v)
            ap = self._approach_for(a.id, pose, v, t, ego, a_ego,
                                    t_go + _time_at_u(u_star, lc))
            # Not reachable yet: close in on the spot without starting to
            # drift over. The lane change is drawn only for a start that is.
            lat = (left_of(ego_lane_x, 0.0) - left_of(pose[0], 0.0)
                   if reachable else 0.0)
            a.start, a.maneuvers = road, _build_plan(t, v, ap, t_go, lc, lat, u0=0.0)
            return
        if self.phase == PHASE_CUTTING:
            u_now = _u_at(t - self.t_lc_start, lc)
            if self.spec.mode == MODE_STATIC_COMMIT and self._committed_plan:
                return          # executing the plan fixed when the lane change started
            ap = self._approach_for(a.id, pose, v, t, ego, a_ego, self.t_cross_plan)
            a.start = road
            a.maneuvers = _build_plan(t, v, ap, self.t_lc_start, lc,
                                      self.lat_offset, u0=u_now)
            self._committed_plan = True
            return
        if self._hold_after_cross:
            # from the cut-in instant on: finish the lane change, hold speed
            u_now = _u_at(t - self.t_lc_start, lc)
            a.start = road
            a.maneuvers = _hold_plan(v, u_now, lc, self.lat_offset)
            self._hold_after_cross = False

    def _keep_one_cut_in(self, sc, states) -> None:
        """Only the holder changes lanes. Background cars never plan a lane
        change of their own (load_director strips authored ones), so a lane
        change in anyone else's plan is left over from a holder the cut-in was
        taken from: it is dropped, and the car keeps to its lane. Once the cut-in
        has happened this also means no second car cuts in."""
        for a in sc.actors:
            if (a.id in (self.ego_id, self.holder) or a.id not in states
                    or getattr(a, "autonomy", "auto") == "self" or not _changing_lanes(a)):
                continue
            pose, v = states[a.id]
            a.start = (self.map.lane_center_x(lane_index(self.map, pose[0])), pose[1],
                       ROAD_HEADING_DEG)
            a.maneuvers = _cruise_toward(v, self.cruise.get(a.id, v))
            self._event(0.0 if self._last_tick_t is None else self._last_tick_t,
                        "single_cut_in", f"actor {a.id} drops its lane change: actor "
                        f"{self.holder} holds the cut-in", actor=a.id)

    def _landing(self, holder, t, ego, a_ego, states
                 ) -> Optional[Tuple[float, float, float]]:
        """(time, along of the holder's centre, its speed) where the holder
        lands in the ego's lane; None while that is not planned yet."""
        pose, v = states[holder.id]
        if self.phase in (PHASE_CROSSED, PHASE_MERGED):
            return t + ROOM_AFTER_S, along_of(pose[0], pose[1]) + v * ROOM_AFTER_S, v
        if self.phase == PHASE_CUTTING and self.t_cross_plan is not None:
            t_land = self.t_cross_plan
        elif self._foreseeable and holder.id in self._ready:
            t_go, _ = self._ready[holder.id]
            t_land = t_go + _time_at_u(self._u_star(holder.id, ego.v),
                                       self.spec.lc_duration_s)
        else:
            return None
        t_land = max(t_land, t + (self._tick_dt or 0.05))
        dist, v_land, _ = self._rendezvous(holder.id, pose, t, ego, a_ego, t_land)
        return t_land, along_of(pose[0], pose[1]) + dist, v_land

    def _make_room(self, sc, holder, t, ego, a_ego, states) -> set:
        """Cars ahead in the lane the holder lands in pull ahead, nearest first,
        so it lands with room in front of it. Until its lane change starts that
        lane is the ego's and "ahead" is ahead of the ego; from then on it is
        the lane the change aims at and ahead of the holder, and only while the
        ego is still in that lane. A car that governs itself or is changing
        lanes keeps its plan but still takes up room. Returns the ids re-planned."""
        land = self._landing(holder, t, ego, a_ego, states)
        if land is None:
            return set()
        t_land, s_land, v_land = land
        tau = t_land - t
        ego_lane = lane_index(self.map, ego.x)
        if self.phase == PHASE_APPROACH or self._land_lane is None:
            lane, s_ref = ego_lane, along_of(ego.x, ego.y)
        elif ego_lane != self._land_lane:
            return set()
        else:
            hp, _ = states[holder.id]
            lane, s_ref = self._land_lane, along_of(hp[0], hp[1])
        ahead = []
        for a in sc.actors:
            if a.id in (self.ego_id, holder.id) or a.id not in states:
                continue
            pose, v = states[a.id]
            s = along_of(pose[0], pose[1])
            if lane_index(self.map, pose[0]) == lane and s > s_ref:
                ahead.append((s, a, pose, v))
        ahead.sort(key=lambda r: r[0])
        front = s_land + 0.5 * self.dims.get(holder.id, (4.5, 2.0))[0]
        v_back = v_land
        changed: set = set()
        for s, a, pose, v in ahead:
            la = self.dims.get(a.id, (a.length, a.width))[0]
            need = front + ROOM_S0_M + ROOM_T_S * v_back - (s + v * tau - 0.5 * la)
            v_end = v
            fixed = (getattr(a, "autonomy", "auto") == "self" or _changing_lanes(a))
            if need > 1e-3 and not fixed:
                acc = ROOM_ACCEL
                if tau * tau < 2.0 * need / acc:
                    acc = min(A_MAX, 2.0 * need / max(tau * tau, 1e-6))
                disc = tau * tau - 2.0 * need / acc
                t1 = tau - math.sqrt(disc) if disc > 0.0 else tau
                t1 = min(t1, max(0.0, (se.CUTIN_MAX_SPEED - v) / acc))
                v_end = v + acc * t1
                gained = acc * t1 * (tau - t1) + 0.5 * acc * t1 * t1
                a.start = (pose[0], pose[1], ROAD_HEADING_DEG)
                a.maneuvers = ([se.Maneuver(type="go_straight", duration=t1,
                                            intercept=v, slope=acc)]
                               if t1 > 1e-3 else [])
                a.maneuvers.append(se.Maneuver(type="go_straight", duration=TAIL_S,
                                               intercept=v_end))
                changed.add(a.id)
                if a.id not in self._room_noted:
                    self._room_noted.add(a.id)
                    self._event(t, "make_room",
                                f"actor {a.id} pulls ahead of the cut-in spot: "
                                f"+{need:.1f} m by t={t_land:.2f}s, "
                                f"{v:.1f} -> {v_end:.1f} m/s"
                                + ("" if gained >= need - 0.05 else
                                   f" (short by {need - gained:.1f} m)"),
                                actor=a.id)
                front = s + v * tau + gained + 0.5 * la
            else:
                front = s + v * tau + 0.5 * la
            v_back = v_end
        return changed

    def _follow_ego(self, sc, ego, states, skip=()) -> bool:
        """Cars behind the ego, in its lane or overlapping it sideways, follow
        by IDM the nearest vehicle ahead whose body overlaps theirs sideways:
        the ego, the holder, a car that governs itself or another follower. A
        car keeps following once it has -- released, it would hold whatever
        speed IDM last gave it, 0 included -- with the same rule for its lead.
        The holder, self-governed cars, cars changing lanes and the ids in
        `skip` lead but are not re-planned. Returns True if a plan changed."""
        s_ego, l_ego = along_of(ego.x, ego.y), left_of(ego.x, ego.y)
        ego_lane = lane_index(self.map, ego.x)
        bodies = [(s_ego, l_ego, ego.v, ego.length, ego.width, self.ego_id)]
        for a in sc.actors:
            if a.id != self.ego_id and a.id in states:
                pose, v = states[a.id]
                la, wa = self.dims.get(a.id, (a.length, a.width))
                bodies.append((along_of(pose[0], pose[1]), left_of(pose[0], pose[1]),
                               v, la, wa, a.id))
        by_id = {a.id: a for a in sc.actors}
        changed = False
        # `v` is the speed it is driving at, not a yield's replacement speed:
        # this plan replaces the yield's, and must not jump
        for s, l, v, la, wa, aid in bodies[1:]:
            a = by_id[aid]
            if (aid == self.holder or aid in skip
                    or getattr(a, "autonomy", "auto") == "self" or _changing_lanes(a)):
                continue
            if aid not in self._followed:
                beside = abs(l - l_ego) < 0.5 * (wa + ego.width) + FOLLOW_LAT_MARGIN_M
                if not (s < s_ego and (beside or lane_index(self.map, states[aid][0][0])
                                       == ego_lane)):
                    continue
                self._followed.add(aid)
            lead = None
            for s2, l2, v2, la2, wa2, aid2 in bodies:
                if (aid2 == aid or s2 <= s
                        or abs(l2 - l) >= 0.5 * (wa + wa2) + FOLLOW_LAT_MARGIN_M):
                    continue
                gap = (s2 - 0.5 * la2) - (s + 0.5 * la)
                if lead is None or gap < lead[0]:
                    lead = (gap, v2)
            v0 = max(0.1, self.cruise.get(aid, v))
            # above its cruise speed (a yield sped it up) it eases back at <= b
            free = max(1.0 - (v / v0) ** 4, -FOLLOW_B / FOLLOW_A)
            if lead is None:
                acc = FOLLOW_A * free
            elif lead[0] <= 0.1:
                acc = -FOLLOW_D_MAX
            else:
                s_star = FOLLOW_S0_M + max(0.0, v * FOLLOW_T_S + v * (v - lead[1])
                                           / (2.0 * math.sqrt(FOLLOW_A * FOLLOW_B)))
                acc = FOLLOW_A * (free - (s_star / lead[0]) ** 2)
            acc = se.clamp(acc, -FOLLOW_D_MAX, FOLLOW_A)
            dur = FOLLOW_PLAN_S
            if acc < 0.0 and v + acc * dur < 0.0:
                dur = v / -acc
            v_end = max(0.0, v + acc * dur)
            pose = states[aid][0]
            a.start = (pose[0], pose[1], ROAD_HEADING_DEG)
            a.maneuvers = [se.Maneuver(type="go_straight", duration=max(dur, 1e-3),
                                       intercept=v, slope=acc),
                           se.Maneuver(type="go_straight", duration=TAIL_S,
                                       intercept=v_end)]
            changed = True
        return changed

    def _yields(self, sc: se.Scenario) -> None:
        import cutin_orchestrator as co
        scores = {a.id: {co.ROLE_CUTIN: (1.0 if a.id == self.holder else 0.0)}
                  for a in sc.actors if a.id != self.ego_id}
        ys = co.resolve_actor_collisions(sc.actors, lambda a, tt: a.pose_at_time(tt),
                                         scores, ego_id=self.ego_id)
        if not ys:
            return
        for inter, v, priv, t_hit in ys:
            inter.start = (inter.start[0], inter.start[1], ROAD_HEADING_DEG)
            inter.maneuvers = [se.Maneuver(type="go_straight", duration=TAIL_S,
                                           intercept=max(0.0, v))]
        sc.simulate(horizon=SIM_HORIZON_S)

    # ---- measurement ---- #
    def observe(self, t: float, ego: EgoState,
                states: Dict[str, Tuple[float, float, float, float]]) -> None:
        """Per simulation step. `states[id] = (x, y, heading_deg, speed)`."""
        self._note_speed(t, ego.v)
        if self.holder is None or self.holder not in states:
            self._prev_obs = None
            return
        hx, hy, hh, hv = states[self.holder]
        la, wa = self.dims.get(self.holder, (4.5, 2.0))
        lw = self.map.lane_width
        ego_lane_x = self.map.lane_center_x(lane_index(self.map, ego.x))
        margin = lane_margin(hx, hy, hh, la, wa, ego_lane_x, lw)
        gap = ((along_of(hx, hy) - la / 2.0)
               - (along_of(ego.x, ego.y) + ego.length / 2.0))
        obs = {"t": t, "margin": margin, "gap": gap, "dv": hv - ego.v,
               "ego_v": ego.v, "v": hv}
        prev = self._prev_obs
        self._prev_obs = obs
        if self.phase == PHASE_CUTTING and margin <= 0.0:
            f = 1.0
            if prev is not None and prev["margin"] > 0.0:
                f = prev["margin"] / (prev["margin"] - margin)
            lerp = (lambda k: obs[k] if prev is None
                    else prev[k] + f * (obs[k] - prev[k]))
            g, dv = lerp("gap"), lerp("dv")
            closing = -dv
            ttc = (0.0 if g <= 0.0 else (g / closing if closing > 1e-6 else None))
            self.crossing = {
                "t": round(lerp("t"), 3), "gap_m": round(g, 3),
                "rel_speed_mps": round(dv, 3), "ego_speed_mps": round(lerp("ego_v"), 3),
                "actor_speed_mps": round(lerp("v"), 3),
                "ttc_s": (round(ttc, 3) if ttc is not None else None),
                "gap_error_m": round(g - self.spec.gap_m, 3),
                "rel_speed_error_mps": round(dv - self.spec.rel_speed_mps, 3)}
            self.phase = PHASE_CROSSED
            self._hold_after_cross = True
            self._event(t, "cut_in",
                        f"actor {self.holder} enters the ego's lane: gap "
                        f"{g:.2f} m (asked {self.spec.gap_m:g}), relative speed "
                        f"{dv:+.2f} m/s (asked {self.spec.rel_speed_mps:+g})"
                        + (f", TTC {ttc:.2f} s" if ttc is not None else ""),
                        actor=self.holder, measured=dict(self.crossing))
        if self.phase in (PHASE_CROSSED, PHASE_MERGED):
            closing = ego.v - hv
            if gap <= 0.0 and margin <= -wa / 2.0:
                self.post["overlap"] = True
            if self.post["min_gap_m"] is None or gap < self.post["min_gap_m"]:
                self.post["min_gap_m"] = round(gap, 3)
            if gap > 0.0 and closing > 1e-6:
                ttc = gap / closing
                if self.post["min_ttc_s"] is None or ttc < self.post["min_ttc_s"]:
                    self.post["min_ttc_s"] = round(ttc, 3)
        if (self.phase == PHASE_CROSSED
                and abs(hx - ego_lane_x) <= MERGED_LAT_TOL_M):
            self.phase = PHASE_MERGED
            self.merged_at = t
            self._event(t, "merged", f"actor {self.holder} is centred in the "
                        f"ego's lane", actor=self.holder)

    # ---- report ---- #
    @property
    def outcome(self) -> str:
        """not_triggered | triggered_not_started (allowed, never reachable) |
        started_not_crossed | cut_in"""
        if self.phase == PHASE_APPROACH:
            return ("not_triggered" if self.t_trigger is None
                    else "triggered_not_started")
        if self.phase == PHASE_CUTTING:
            return "started_not_crossed"
        return "cut_in"

    def summary(self) -> dict:
        def r3(x):
            return round(x, 3) if x is not None else None
        return {"spec": self.spec.to_dict(), "holder": self.holder,
                "phase": self.phase, "outcome": self.outcome,
                "t_trigger": r3(self.t_trigger),
                "t_lane_change": r3(self.t_lc_start),
                "ready_wait_s": (r3(self.t_lc_start - self.t_trigger)
                                 if self.t_lc_start is not None
                                 and self.t_trigger is not None else None),
                "t_cut_in_planned": r3(self.t_cross_plan),
                "cut_in": self.crossing,
                "t_merged": (round(self.merged_at, 3)
                             if self.merged_at is not None else None),
                "after_cut_in": dict(self.post),
                "recasts": self.n_recasts,
                "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "candidates": self.plan_info,
                "events": list(self.events)}


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def _changing_lanes(a) -> bool:
    """Whether an actor's plan still moves it sideways (a lane change under way
    or to come). The ego-lane paths leave such a plan alone: a straight-line
    replacement would strand the car across the lane line."""
    return any(getattr(m, "type", "") == "lane_change" for m in (a.maneuvers or []))


def _cruise_toward(v: float, v_cruise: float) -> List[se.Maneuver]:
    v = max(0.0, v)
    dv = v_cruise - v
    if abs(dv) < 0.05:
        return [se.Maneuver(type="go_straight", duration=TAIL_S, intercept=v)]
    a = CRUISE_ACCEL if dv > 0 else -CRUISE_ACCEL
    return [se.Maneuver(type="go_straight", duration=abs(dv) / CRUISE_ACCEL,
                        intercept=v, slope=a),
            se.Maneuver(type="go_straight", duration=TAIL_S, intercept=v_cruise)]


def _build_plan(t: float, v0: float, ap: Approach, t_lc_start: float,
                lc: float, lat_offset: float, u0: float) -> List[se.Maneuver]:
    """Maneuvers from now: the approach profile to the cut-in instant, the
    lane change from `t_lc_start` over `lc` (continuing from u0 when it has
    already started), then constant speed. Velocity is piecewise linear
    through the profile's samples; every slice is one maneuver."""
    t_trigger = t_lc_start
    t_cross = t + ap.tau
    t_lc0 = max(t_trigger, t)
    t_lc_end = t_trigger + lc
    points = {t, t_cross, t_lc0, t_lc_end}
    points.update(t + dt for dt, _ in ap.samples)
    points = sorted(p for p in points if p >= t - 1e-9)
    points.append(max(points) + TAIL_S)

    def v_at(tt: float) -> float:
        return max(0.0, ap.v_at(tt - t)) if tt <= t_cross else max(0.0, ap.v_end)

    out: List[se.Maneuver] = []
    for ta, tb in zip(points[:-1], points[1:]):
        dur = tb - ta
        if dur <= 1e-6:
            continue
        va, vb = v_at(ta), v_at(tb)
        accel = (vb - va) / dur if tb <= t_cross + 1e-9 else 0.0
        mid = 0.5 * (ta + tb)
        if mid >= t_trigger and ta < t_lc_end - 1e-9:
            out.append(se.Maneuver(type="lane_change", duration=dur,
                                   intercept=va, slope=accel,
                                   lateral_offset=lat_offset,
                                   lat_u0=_u_at(ta - t_trigger, lc),
                                   lat_u1=_u_at(tb - t_trigger, lc)))
        else:
            out.append(se.Maneuver(type="go_straight", duration=dur,
                                   intercept=va, slope=accel))
    return out


def _hold_plan(v: float, u_now: float, lc: float, lat_offset: float
               ) -> List[se.Maneuver]:
    v = max(0.0, v)
    out: List[se.Maneuver] = []
    if u_now < 1.0 - 1e-6:
        out.append(se.Maneuver(type="lane_change", duration=(1.0 - u_now) * lc,
                               intercept=v, lateral_offset=lat_offset,
                               lat_u0=u_now, lat_u1=1.0))
    out.append(se.Maneuver(type="go_straight", duration=TAIL_S, intercept=v))
    return out


# --------------------------------------------------------------------------- #
# Spawning and loading
# --------------------------------------------------------------------------- #
_LANE_WORDS = {"left": -1, "right": +1, "same": 0}


def _rel_lane(v) -> int:
    if isinstance(v, str):
        if v.lower() not in _LANE_WORDS:
            raise ValueError(f"spawn lane must be left/right/same or an int, not {v!r}")
        return _LANE_WORDS[v.lower()]
    return int(v)


PALETTE = [(210, 70, 60), (60, 120, 210), (200, 160, 60), (160, 90, 200),
           (80, 200, 200), (225, 105, 165), (150, 185, 95)]


def spawn_actors(spawn: SpawnSpec, sc: se.Scenario, ego_id: str = EGO_ID,
                 count_override: Optional[int] = None,
                 seed_override: Optional[int] = None) -> List[str]:
    """Replace the scenario's non-ego actors with the spawn spec's. Returns
    notes about the draw."""
    ego = next(a for a in sc.actors if a.id == ego_id)
    ego_lane = lane_index(sc.map, ego.start[0])
    s_ego = along_of(ego.start[0], ego.start[1])
    notes: List[str] = []
    entries: List[dict] = [dict(e) for e in spawn.actors]
    count = count_override if count_override is not None else spawn.count
    rnd = dict(spawn.random) if spawn.random else None
    if seed_override is not None:
        rnd = dict(rnd or {})
        rnd["seed"] = seed_override
    if count is not None and count < len(entries):
        entries = entries[:count]
    if count is not None and count > len(entries):
        if rnd is None:
            raise ValueError(f"{count} actors requested but only {len(entries)} "
                             "spawn entries and no `random` block to draw the rest")
    if rnd is not None:
        rng = random.Random(int(rnd.get("seed", 0)))
        lanes = [_rel_lane(v) for v in (rnd.get("lanes") or ["left", "right"])]
        off_lo, off_hi = rnd.get("offset_m", [-30.0, 30.0])
        v_lo, v_hi = rnd.get("speed_mps", [8.0, 14.0])
        # centre-to-centre distance a draw in the ego's own lane keeps from it
        ego_clear = float(rnd.get("ego_clear_m", 6.0))
        n_draw = (count - len(entries)) if count is not None else int(rnd.get("count", 1))
        placed = [(_rel_lane(e.get("lane", "right")), float(e.get("offset_m", 0.0)))
                  for e in entries]
        for _ in range(max(0, n_draw)):
            for attempt in range(200):
                lane = rng.choice(lanes)
                off = rng.uniform(float(off_lo), float(off_hi))
                clash = (lane == 0 and abs(off) < ego_clear) or any(
                    pl == lane and abs(po - off) < 6.0 for pl, po in placed)
                if not clash:
                    break
            else:
                notes.append("could not place a random actor without overlap; "
                             "skipped")
                continue
            placed.append((lane, off))
            entries.append({"lane": lane, "offset_m": round(off, 3),
                            "speed_mps": round(rng.uniform(float(v_lo), float(v_hi)), 3),
                            "drawn": True})
        notes.append(f"spawn seed {int(rnd.get('seed', 0))}: "
                     f"{sum(1 for e in entries if e.get('drawn'))} drawn")
    actors = [ego]
    n_lanes = max(1, sc.map.num_lanes)
    for i, e in enumerate(entries):
        lane = ego_lane + _rel_lane(e.get("lane", "right"))
        if not 0 <= lane < n_lanes:
            notes.append(f"spawn {i + 1}: lane {lane} is off the road; clamped")
            lane = int(se.clamp(lane, 0, n_lanes - 1))
        x = sc.map.lane_center_x(lane)
        s = s_ego + float(e.get("offset_m", 0.0))
        speed = float(e.get("speed_mps", 12.0))
        aid = str(e.get("id", i + 1))
        start = (x, s, ROAD_HEADING_DEG)
        actors.append(se.Actor(
            id=aid, color=tuple(e.get("color", PALETTE[i % len(PALETTE)])),
            length=float(e.get("length", 4.5)), width=float(e.get("width", 2.0)),
            start=start, cruise=speed,
            maneuvers=[se.Maneuver(type="go_straight", duration=TAIL_S,
                                   intercept=speed)]))
    sc.actors = actors
    sc.simulate()
    return notes


@dataclass
class CutinOverrides:
    """Command-line overrides, identical for every host."""
    trigger_time: Optional[float] = None
    trigger_ego_speed: Optional[float] = None
    trigger_speed_hold_s: Optional[float] = None
    gap_m: Optional[float] = None
    rel_speed_mps: Optional[float] = None
    mode: Optional[str] = None
    lc_duration_s: Optional[float] = None
    num_actors: Optional[int] = None
    spawn_seed: Optional[int] = None

    def apply(self, spec: CutinSpec) -> CutinSpec:
        spec = copy.deepcopy(spec)
        if self.trigger_time is not None or self.trigger_ego_speed is not None:
            spec.trigger = Trigger(time_gt=self.trigger_time,
                                   ego_speed_gt=self.trigger_ego_speed,
                                   ego_speed_hold_s=spec.trigger.ego_speed_hold_s)
        if self.trigger_speed_hold_s is not None:
            spec.trigger.ego_speed_hold_s = self.trigger_speed_hold_s
        if self.gap_m is not None:
            spec.gap_m = self.gap_m
        if self.rel_speed_mps is not None:
            spec.rel_speed_mps = self.rel_speed_mps
        if self.mode is not None:
            spec.mode = self.mode
        if self.lc_duration_s is not None:
            spec.lc_duration_s = self.lc_duration_s
        return spec


def add_cli_arguments(p) -> None:
    """The cut-in flags, shared by every entry point."""
    g = p.add_argument_group("cut-in orchestrator")
    mode = g.add_mutually_exclusive_group()
    mode.add_argument("--static_commit", "--static-commit", dest="cutin_mode",
                      action="store_const", const=MODE_STATIC_COMMIT,
                      help="commit to the plan solved when the lane change "
                           "starts; the ego's response decides the outcome")
    mode.add_argument("--re-aim", "--re_aim", dest="cutin_mode",
                      action="store_const", const=MODE_RE_AIM,
                      help="keep re-aiming at the ego's bumper until the actor "
                           "enters its lane")
    g.add_argument("--trigger-time", type=float, default=None,
                   help="allow the cut-in once t > this (s)")
    g.add_argument("--trigger-ego-speed", type=float, default=None,
                   help="allow the cut-in once the ego has been faster than "
                        "this (m/s) for longer than --trigger-speed-hold")
    g.add_argument("--trigger-speed-hold", type=float, default=None,
                   help=f"how long the speed condition must hold (s, default "
                        f"{SPEED_HOLD_S:g})")
    g.add_argument("--gap", dest="cutin_gap", type=float, default=None,
                   help="ego front bumper to actor rear bumper at the cut-in "
                        "instant (m)")
    g.add_argument("--rel-speed", type=float, default=None,
                   help="actor speed minus ego speed at the cut-in instant (m/s)")
    g.add_argument("--lc-duration", type=float, default=None,
                   help="lane change duration (s)")
    g.add_argument("--num-actors", type=int, default=None,
                   help="number of non-ego actors")
    g.add_argument("--spawn-seed", type=int, default=None,
                   help="seed for the random spawn draw")


def overrides_from_args(args) -> CutinOverrides:
    return CutinOverrides(
        trigger_time=getattr(args, "trigger_time", None),
        trigger_ego_speed=getattr(args, "trigger_ego_speed", None),
        trigger_speed_hold_s=getattr(args, "trigger_speed_hold", None),
        gap_m=getattr(args, "cutin_gap", None),
        rel_speed_mps=getattr(args, "rel_speed", None),
        mode=getattr(args, "cutin_mode", None),
        lc_duration_s=getattr(args, "lc_duration", None),
        num_actors=getattr(args, "num_actors", None),
        spawn_seed=getattr(args, "spawn_seed", None))


def load_director(path: str, sc: se.Scenario,
                  overrides: Optional[CutinOverrides] = None,
                  ego_id: str = EGO_ID,
                  dims: Optional[Dict[str, Tuple[float, float]]] = None,
                  heading: str = HEADING_SCRIPT
                  ) -> Optional[CutinDirector]:
    """Build a director from a scenario YAML, or None when it has no cut-in.

    A top-level `cutin:` block (and optional `spawn:`) is the new form. An
    actor carrying the old `cutin: {at, along, ...}` is translated and becomes
    the initial holder, with the listed actors as the spawn.
    """
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    overrides = overrides or CutinOverrides()
    top = raw.get("cutin")
    legacy_holder = next((a for a in sc.actors if a.id != ego_id and a.cutin), None)
    if top is None and legacy_holder is None:
        return None
    if top is not None:
        spec = CutinSpec.from_dict(top)
    else:
        spec = CutinSpec.from_legacy(dict(legacy_holder.cutin))
        spec.holder = legacy_holder.id
    spec = overrides.apply(spec)
    notes: List[str] = []
    spawn = SpawnSpec.from_dict(raw.get("spawn"))
    if spawn is not None or overrides.num_actors is not None or overrides.spawn_seed is not None:
        if spawn is None:
            spawn = SpawnSpec(actors=[], random={"seed": overrides.spawn_seed or 0})
        notes = spawn_actors(spawn, sc, ego_id, overrides.num_actors,
                             overrides.spawn_seed)
        if spec.holder is not None and spec.holder not in {a.id for a in sc.actors}:
            spec.holder = None
    # a legacy authored merge must not run before the director has spoken
    for a in sc.actors:
        if a.id != ego_id:
            a.cutin = None
            if not a.maneuvers or any(getattr(m, "type", "") == "lane_change"
                                      for m in a.maneuvers):
                v = se.actor_cruise_speed(a)
                a.maneuvers = [se.Maneuver(type="go_straight", duration=TAIL_S,
                                           intercept=v)]
    sc.simulate()
    spec.notes.extend(notes)
    return CutinDirector(spec, sc, ego_id=ego_id, dims=dims, heading=heading)


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
class TrajectoryLog:
    """Driven trajectories of every vehicle, in one format for every host."""

    def __init__(self, lane_map: se.MapConfig, rate_hz: float = 20.0):
        self.map = lane_map
        self.dt = 1.0 / max(rate_hz, 1e-3)
        self._next = 0.0
        self.vehicles: Dict[str, dict] = {}

    def record(self, t: float, states: Dict[str, Tuple[float, float, float, float]],
               roles: Dict[str, str],
               dims: Dict[str, Tuple[float, float]]) -> None:
        if t + 1e-9 < self._next:
            return
        self._next = t + self.dt
        for aid, (x, y, h, v) in states.items():
            veh = self.vehicles.setdefault(str(aid), {
                "length": round(dims.get(aid, (4.5, 2.0))[0], 3),
                "width": round(dims.get(aid, (4.5, 2.0))[1], 3),
                "t": [], "x": [], "y": [], "heading_deg": [], "speed_mps": [],
                "lane": [], "role": []})
            veh["t"].append(round(t, 3))
            veh["x"].append(round(x, 3))
            veh["y"].append(round(y, 3))
            veh["heading_deg"].append(round(h % 360.0, 3))
            veh["speed_mps"].append(round(v, 3))
            veh["lane"].append(lane_index(self.map, x))
            veh["role"].append(roles.get(str(aid), "traffic"))

    def to_dict(self, director: Optional[CutinDirector] = None,
                source: str = "") -> dict:
        n = max(1, self.map.num_lanes)
        return {"format": "highway_trajectories/v1", "source": source,
                "rate_hz": round(1.0 / self.dt, 3),
                "frame": {"road_heading_deg": ROAD_HEADING_DEG,
                          "lane_width_m": self.map.lane_width,
                          "lane_center_x_m": [self.map.lane_center_x(i) for i in range(n)],
                          "lane_index": "0 is the leftmost northbound lane"},
                "vehicles": self.vehicles,
                "cutin": director.summary() if director is not None else None}

    def write(self, path: str, director: Optional[CutinDirector] = None,
              source: str = "") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(director, source), f, indent=1)
