#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/highway_ego.py — the ego policy for the highway scenarios.

Why this file exists at all
---------------------------
`carla_port` is careful to write no ego policy: it subclasses `v4/drivev2.py`'s
`Drive` and inherits `idm_control` / `path_steer` / `autonomous_control`
unmodified. This port cannot do that, for two independent reasons:

1. **Import identity.** `v4/drivev2.py` does `import scenario_editor as se`,
   and that bare name belongs to `highway/scenario_editor.py` in this process
   (see `script_bridge`). Importing drivev2 here would bind it to the wrong
   script layer.

2. **A fixed reference path cannot do these scenarios.** drivev2 builds
   `_ref_path` once, from the junction turn geometry, and pure-pursues it
   forever. That is right for `redlight`/`left_turn`/`right_turn`, where the
   route is decided before the run. But `scenario_hard_brake` asks the ego to
   *change lanes* around a slow lead, and `scenario_overtake` asks it to use
   the **oncoming** lane to get around a stopped car. IDM is longitudinal only:
   on a fixed path it would brake and sit behind the blocker forever, and both
   scenarios would grade as a failure of the harness rather than of the policy.

So the longitudinal law here is drivev2's IDM, restated with its own constants
(they are quoted below with the values from `v4/drivev2.py`, which is the
reference implementation), and the lateral law is drivev2's pure pursuit — but
pointed at a *lane the policy chooses*, with a gap-acceptance layer deciding
which lane that is. That layer is the new part, and it is deliberately small
and legible: this is a baseline to stress, not a driver to be proud of.

An external policy remains the point. `--ego-policy` routes through
`carla_port.ego_driver.PolicyEgoDriver`, which speaks `ego_policy_v1`; this
class is what runs when you do not bring one.

Frame
-----
Everything here is in the script frame of `HighwayFrame`: +y is the ego's
direction of travel, +x is to its right, heading 90 deg is forward. On a
straight road that makes the lateral problem one-dimensional, which is why the
reference path can be a lane centre rather than a polyline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .highway_map import FORWARD_HEADING, HighwayFrame
from .script_bridge import se

# ---- vehicle limits: v4/drivev2.py ---- #
WHEELBASE = 2.8
A_THROTTLE = 5.0
A_BRAKE = 9.0
V_MAX = 18.0
DELTA_MAX = math.radians(32)
DRAG = 1.0

# ---- IDM: v4/drivev2.py ---- #
IDM_V0 = 12.0        # desired/free-flow cruise speed (m/s)
IDM_A = 3.0          # max acceleration (m/s^2)
IDM_B = 3.0          # comfortable/desired deceleration (m/s^2)
IDM_S0 = 2.0         # minimum gap / jam distance, bumper-to-bumper (m)
IDM_T = 1.5          # desired time headway (s)
IDM_DELTA = 4        # acceleration exponent
IDM_LANE_TOL = 2.2   # lateral tolerance (m) for "roughly in my lane"

# ---- pure pursuit: v4/drivev2.py ---- #
LOOKAHEAD_MIN = 3.0
LOOKAHEAD_MAX = 12.0
LOOKAHEAD_GAIN = 0.6

# ---- lane selection (new; see the module docstring) ---- #
LC_GAIN = 0.6            # m/s^2 of IDM improvement needed to bother changing
LC_MIN_INTERVAL = 1.5    # s between committed lane changes
LC_FRONT_GAP = 8.0       # m of clear road needed ahead in the target lane
LC_REAR_GAP = 6.0        # m of clear road needed behind in the target lane
LC_DONE_TOL = 0.5        # m from the lane centre that counts as arrived
STEER_RATE = 4.0         # 1/s cap on normalized steer change (anti-snap)
#: an oncoming lane is only usable if the pass can finish this many times over
ONCOMING_SAFETY = 2.0
#: floor on the ego's closing speed when estimating a pass, so a blocker
#: matching the ego's speed does not make every overtake look infinite
MIN_PASS_CLOSING = 2.0   # m/s
RETURN_CLEAR = 12.0      # m of clear road needed to come home after a pass
CREEP_SPEED = 2.0        # m/s below which a committed lane change may creep
CREEP_ACCEL = 1.0        # m/s^2 floor while creeping out
CREEP_MIN_GAP = 3.0      # m of room needed in the target lane to creep

EGO_LENGTH = 4.5
EGO_WIDTH = 2.0


@dataclass
class Ego:
    """Live ego state in the script frame. Mirrors `cutin_orchestrator.Ego`."""
    x: float
    y: float
    theta: float          # radians, script frame (pi/2 == forward)
    v: float


@dataclass
class Neighbour:
    """A background actor seen in the ego's lane-relative terms."""
    actor_id: str
    lane: int             # nearest lane index (display / candidate filtering)
    lat: float            # script x of the actor — the lateral position that
                          # matters while somebody is between two lanes
    along: float          # signed metres ahead of the ego along +y
    gap: float            # bumper-to-bumper, floored at 0.5 m
    speed: float          # signed along +y (negative == oncoming)
    oncoming: bool


class HighwayEgoPolicy:
    """IDM longitudinally, pure pursuit laterally, gap acceptance to pick a lane."""

    def __init__(self, frame: HighwayFrame, ego: Ego, background: "se.Scenario",
                 atime: float = 0.0, desired_speed: float = IDM_V0,
                 allow_lane_change: bool = True):
        self.frame = frame
        self.ego = ego
        self.asc = background
        self.atime = atime
        self.v0 = float(desired_speed)
        self.allow_lane_change = bool(allow_lane_change)

        self.home_lane = frame.lane_index_of(ego.x)
        self.target_lane = self.home_lane
        self._last_change = -1e9
        self._steer = 0.0
        #: filled every decision, for the HUD and the run report
        self.reason = "hold lane"
        #: edging out at low speed to clear a stopped blocker
        self.creeping = False
        self.n_lane_changes = 0
        self.used_oncoming = False

    # ------------------------------------------------------------------ #
    # Perception
    # ------------------------------------------------------------------ #
    def neighbours(self) -> List[Neighbour]:
        """Every background actor, placed in lane / along terms.

        Uses the orchestrator's own trajectory oracle (`Actor.traj`) at the
        current script time, which is the same source the orchestrator reasons
        about, so the ego and the orchestrator never disagree about where a car
        is.
        """
        out: List[Neighbour] = []
        k = int(round(self.atime / se.DT))
        for a in self.asc.actors:
            if not a.traj:
                continue
            kk = max(0, min(k, len(a.traj) - 1))
            ax, ay, ah = a.traj[kk]
            v = a.speeds[kk] if a.speeds else 0.0
            forward = abs(((ah - FORWARD_HEADING + 180.0) % 360.0) - 180.0) < 90.0
            along = ay - self.ego.y
            half = (getattr(a, "length", 4.5) + EGO_LENGTH) / 2.0
            out.append(Neighbour(
                actor_id=str(a.id), lane=self.frame.lane_index_of(ax), lat=ax,
                along=along, gap=max(abs(along) - half, 0.5),
                speed=(v if forward else -v), oncoming=not forward))
        return out

    def _leader_near(self, x_ref: float, nbrs: List[Neighbour]
                     ) -> Tuple[Optional[float], Optional[float]]:
        """(gap, leader speed) for the closest car ahead within
        `IDM_LANE_TOL` of the lateral position `x_ref`.

        Deliberately NOT "the closest car whose lane index equals mine".
        A car half way through a cut-in belongs to neither lane by nearest
        centre, so an index test makes it invisible until it has finished
        merging — and the ego drives into the space it is merging into. That is
        not hypothetical: it put an ego into the cut-in actor at 2.8 s.

        drivev2's `_idm_leader` gets this right by working in the ego's own
        heading frame with a lateral tolerance rather than by lane identity,
        so that it "naturally picks up e.g. a right-turn merge target once the
        ego is actually in that lane". Same rule here.

        Oncoming cars are never leaders: car-following an oncoming vehicle is
        meaningless. They are handled by `_oncoming_clear`.
        """
        best = None
        for n in nbrs:
            if n.oncoming or n.along <= 0:
                continue
            if abs(n.lat - x_ref) > IDM_LANE_TOL:
                continue
            if best is None or n.along < best.along:
                best = n
        if best is None:
            return None, None
        return best.gap, max(0.0, best.speed)

    def _leader_in(self, lane: int, nbrs: List[Neighbour]
                   ) -> Tuple[Optional[float], Optional[float]]:
        """(gap, leader speed) for the closest car ahead in `lane`."""
        return self._leader_near(self.frame.lane_center_x(lane), nbrs)

    def _idm_accel(self, gap: Optional[float], v_lead: Optional[float]) -> float:
        """drivev2's `idm_control`, before the throttle normalization."""
        v = self.ego.v
        if gap is None:
            return IDM_A * (1.0 - (v / self.v0) ** IDM_DELTA)
        dv = v - v_lead
        s_star = IDM_S0 + max(0.0, v * IDM_T
                              + (v * dv) / (2.0 * math.sqrt(IDM_A * IDM_B)))
        return IDM_A * (1.0 - (v / self.v0) ** IDM_DELTA - (s_star / gap) ** 2)

    # ------------------------------------------------------------------ #
    # Lateral: which lane?
    # ------------------------------------------------------------------ #
    def _lane_clear(self, lane: int, nbrs: List[Neighbour]) -> bool:
        """Is there room to sit in `lane` right now?"""
        x_ref = self.frame.lane_center_x(lane)
        for n in nbrs:
            if n.oncoming or abs(n.lat - x_ref) > IDM_LANE_TOL:
                continue
            if -LC_REAR_GAP < n.along < LC_FRONT_GAP:
                return False
        return True

    def _oncoming_clear(self, lane: int, nbrs: List[Neighbour],
                        pass_distance: float, v_blocker: float = 0.0) -> bool:
        """Is an oncoming lane safe for a pass of `pass_distance` metres?

        The pass takes as long as it takes to get past the car being overtaken,
        so the divisor is the ego's **closing speed on the blocker**, not its
        ground speed. Dividing by ground speed silently assumes the obstacle is
        parked: on `overtake` it estimated a 2.8 s pass that really took 5.5 s
        and put the ego alongside oncoming traffic with 4 m to spare.

        A floor keeps a same-speed blocker from giving an infinite pass time
        (the ego would then never overtake anything moving), and the whole
        thing is required to fit several times over into the time the oncoming
        car needs to arrive. This is the one place the policy has to be
        genuinely conservative, because getting it wrong is a head-on.
        """
        v = max(self.ego.v, 1.0)
        v_rel = max(v - max(v_blocker, 0.0), MIN_PASS_CLOSING)
        t_pass = max(pass_distance, 1.0) / v_rel
        x_ref = self.frame.lane_center_x(lane)
        for n in nbrs:
            if not n.oncoming or n.along <= 0:
                continue
            if abs(n.lat - x_ref) > IDM_LANE_TOL:
                continue
            closing = v + abs(n.speed)
            if closing <= 0.1:
                continue
            if (n.along / closing) < ONCOMING_SAFETY * t_pass:
                return False
        return True

    def _candidate_lanes(self) -> List[int]:
        """Lanes adjacent to the one the ego is in, in preference order:
        same-direction first, oncoming only as a last resort."""
        cur = self.target_lane
        cands = [i for i in (cur - 1, cur + 1) if 0 <= i < self.frame.num_lanes]
        same = [i for i in cands if self.frame.lanes[i].same_direction]
        onc = [i for i in cands if not self.frame.lanes[i].same_direction]
        return same + onc

    def choose_lane(self, nbrs: List[Neighbour], now: float) -> int:
        """Gap acceptance. Returns the lane the ego should be heading for."""
        if not self.allow_lane_change:
            return self.target_lane
        # still executing the last change? finish it first.
        if abs(self.ego.x - self.frame.lane_center_x(self.target_lane)) > LC_DONE_TOL:
            return self.target_lane
        if now - self._last_change < LC_MIN_INTERVAL:
            return self.target_lane

        cur = self.target_lane
        gap_cur, v_cur = self._leader_in(cur, nbrs)
        a_cur = self._idm_accel(gap_cur, v_cur)

        # Prefer home — but only when going home is not simply undoing the
        # reason we left. Testing "is the home lane clear right now" on
        # distance alone makes the ego pull out to overtake, notice the car it
        # is overtaking is still comfortably far away, and merge straight back
        # in behind it; on `overtake` that flip-flop cost the early window and
        # left the ego stopped behind the blocker. So the return decision uses
        # the same IDM comparison as the decision to leave, with the tie going
        # to home.
        if cur != self.home_lane and self._lane_clear(self.home_lane, nbrs):
            g, vl = self._leader_in(self.home_lane, nbrs)
            a_home = self._idm_accel(g, vl)
            if (g is None or g > RETURN_CLEAR) and a_home >= a_cur - LC_GAIN:
                self.reason = f"return to lane {self.home_lane}"
                return self._commit(self.home_lane, now)

        best, best_a = cur, a_cur + LC_GAIN
        best_why = None
        for lane in self._candidate_lanes():
            oncoming_lane = not self.frame.lanes[lane].same_direction
            if not self._lane_clear(lane, nbrs):
                continue
            if oncoming_lane:
                blocker = gap_cur if gap_cur is not None else 0.0
                if not self._oncoming_clear(lane, nbrs,
                                            blocker + 3.0 * EGO_LENGTH,
                                            v_blocker=(v_cur or 0.0)):
                    continue
            g, vl = self._leader_in(lane, nbrs)
            a_new = self._idm_accel(g, vl)
            if oncoming_lane:
                a_new -= 0.5          # mild reluctance; it is the wrong side
            if a_new > best_a:
                best, best_a = lane, a_new
                best_why = ("overtake via oncoming lane" if oncoming_lane
                            else f"lane {cur} blocked; move to {lane}")
        if best != cur:
            self.reason = best_why or f"move to lane {best}"
            return self._commit(best, now)
        return cur

    def _commit(self, lane: int, now: float) -> int:
        if lane != self.target_lane:
            self.n_lane_changes += 1
            self._last_change = now
            if not self.frame.lanes[lane].same_direction:
                self.used_oncoming = True
            self.target_lane = lane
        return lane

    # ------------------------------------------------------------------ #
    # The control law
    # ------------------------------------------------------------------ #
    def lookahead(self) -> float:
        return max(LOOKAHEAD_MIN,
                   min(LOOKAHEAD_MAX, LOOKAHEAD_GAIN * self.ego.v + LOOKAHEAD_MIN))

    def target_point(self) -> Tuple[float, float]:
        """Pure-pursuit target: the target lane's centre, one lookahead ahead.

        On a straight road the reference path is the line x = lane centre, so
        the lookahead point is analytic and there is no polyline to search —
        the highway simplification of drivev2's `path_steer`.
        """
        L = self.lookahead()
        return (self.frame.lane_center_x(self.target_lane), self.ego.y + L)

    def path_steer(self) -> float:
        tx, ty = self.target_point()
        e = self.ego
        L = max(self.lookahead(), 1e-3)
        alpha = math.atan2(ty - e.y, tx - e.x) - e.theta
        alpha = (alpha + math.pi) % (2.0 * math.pi) - math.pi
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), L)
        return max(-1.0, min(1.0, delta / DELTA_MAX))

    def idm_control(self, nbrs: Optional[List[Neighbour]] = None) -> float:
        """Normalized throttle; negative is braking, as in drivev2."""
        nbrs = self.neighbours() if nbrs is None else nbrs
        # brake for whichever is more urgent: the lane we are in, or the lane
        # we are moving into (during a change the ego straddles both).
        gap, vl = self._leader_near(self.ego.x, nbrs)       # anyone on top of me
        a = self._idm_accel(gap, vl)
        tgt_x = self.frame.lane_center_x(self.target_lane)
        changing = abs(self.ego.x - tgt_x) > LC_DONE_TOL
        if changing:
            g2, v2 = self._leader_in(self.target_lane, nbrs)
            a = min(a, self._idm_accel(g2, v2))
            # Creep deadlock. Steering only bites at speed (the bicycle model's
            # yaw rate is proportional to v), and IDM holds a stopped car at its
            # jam distance behind a stopped leader — so an ego halted behind the
            # blocker can never move sideways to get around it, and never moves
            # again. Real drivers edge out; so does this. The creep is allowed
            # only while committed to a lane that is clear, and it is a floor on
            # the demand, not a target speed.
            if (self.ego.v < CREEP_SPEED and self._lane_clear(self.target_lane, nbrs)
                    and (g2 is None or g2 > CREEP_MIN_GAP)):
                a = max(a, CREEP_ACCEL)
                self.creeping = True
            else:
                self.creeping = False
        else:
            self.creeping = False
        if a >= 0:
            return max(0.0, min(1.0, a / A_THROTTLE))
        return max(-1.0, min(0.0, a / A_BRAKE))

    def command(self, now: float = 0.0, dt: float = 1.0 / 60.0
                ) -> Tuple[float, float]:
        """(throttle, steer), both normalized. The policy entry point."""
        nbrs = self.neighbours()
        self.choose_lane(nbrs, now)
        throttle = self.idm_control(nbrs)
        steer_cmd = self.path_steer()
        # slew-limit the steer so a lane-change decision does not snap the wheel
        max_step = STEER_RATE * max(dt, 1e-3)
        self._steer += max(-max_step, min(max_step, steer_cmd - self._steer))
        return throttle, self._steer

    @property
    def status(self) -> str:
        """The lane decision, for the HUD and the report."""
        return self.reason + (" (creeping out)" if self.creeping else "")

    def commanded_accel(self, throttle: float) -> float:
        """The m/s^2 behind the normalized throttle (drivev2's convention)."""
        return throttle * (A_THROTTLE if throttle > 0 else A_BRAKE)

    # ---- kinematic bicycle, for --ego-mode bicycle ---- #
    def integrate(self, throttle: float, steer: float, dt: float) -> None:
        e = self.ego
        if throttle > 0:
            a = A_THROTTLE * throttle
        elif throttle < 0:
            a = A_BRAKE * throttle
        else:
            a = -DRAG if e.v > 0 else 0.0
        e.v = max(0.0, min(V_MAX, e.v + a * dt))
        delta = DELTA_MAX * steer
        e.theta += (e.v / WHEELBASE) * math.tan(delta) * dt
        e.x += e.v * math.cos(e.theta) * dt
        e.y += e.v * math.sin(e.theta) * dt

    # ---- views the rest of the port wants ---- #
    @property
    def heading_deg(self) -> float:
        return math.degrees(self.ego.theta) % 360.0

    def pose(self) -> Tuple[float, float, float]:
        return (self.ego.x, self.ego.y, self.heading_deg)

    @property
    def reference_path(self) -> List[Tuple[float, float]]:
        """The intended path as a polyline, for an external policy's route
        conditioning and for the recorder."""
        x = self.frame.lane_center_x(self.target_lane)
        y0 = self.ego.y
        return [(x, y0 + s) for s in range(0, int(self.frame.length / 2), 2)]

    def lane_offset(self) -> float:
        """Signed distance from the target lane centre — the cross-track error."""
        return self.ego.x - self.frame.lane_center_x(self.target_lane)
