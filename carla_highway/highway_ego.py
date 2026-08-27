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

So the laws are restated here rather than imported, with the values quoted from
the reference implementations:

* **Longitudinal — IDM**, from `v4/drivev2.py`.
* **Lateral — MOBIL**, from the highway `drivev2.py` of PR #1 ("Model-based
  ego for drivev2.py: IDM speed control + MOBIL lane changes"). MOBIL decides
  which lane; a distance-parameterised lateral profile, steered by inverting
  the bicycle model, turns that decision into motion.

The first pass of this file predates PR #1 and used a hand-rolled gap-acceptance
rule (accept a lane if a fixed window ahead and behind is empty, take whichever
lane IDM likes best) driving a pure-pursuit tracker. MOBIL replaces it, and the
three things it brings are not cosmetic:

1. **A safety criterion with units.** Gap acceptance asked "is 6 m behind me
   empty?"; MOBIL asks "how hard would I make that car brake?" and refuses
   above `MOBIL_B_SAFE`. A 6 m gap is fine at matched speed and a collision if
   the follower is closing at 8 m/s, and only the second question can tell
   those apart.
2. **Politeness.** The gain counts the acceleration inflicted on the follower
   left behind and the follower merged in front of, not only the ego's own.
3. **Steering that survives braking.** The lateral profile advances with
   distance travelled, so IDM braking to a crawl mid-change stops the sideways
   motion instead of saturating the wheel (see `LC_DISTANCE`).

What is kept from the first pass, and is not in upstream drivev2: the
contraflow veto (`_oncoming_clear`) and the creep (`CREEP_*`), both of which
`overtake` needs and neither of which upstream's scenarios exercise.

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
from typing import Dict, List, Optional, Tuple

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

#: desired speed for `cutin`, whose YAML authors the ego at 13 m/s. Upstream
#: drivev2 carries the same split (IDM_V0 / IDM_V0_CUTIN): holding 12 here
#: would make the ego slower than the scenario the cut-in was solved against,
#: and the pin would sit further ahead than the author intended.
IDM_V0_CUTIN = 13.0

# ---- MOBIL: drivev2.py, PR #1 "Model-based ego" ---- #
# MOBIL (Minimizing Overall Braking Induced by Lane change) decides WHICH lane;
# the profile below turns that decision into motion. Every acceleration MOBIL
# compares is an IDM acceleration from `_idm_accel`, so the two models stay
# consistent — that pairing is the standard IDM/MOBIL combination, and it is
# why `_idm_accel` takes the vehicle's own (v, v0) rather than reading the
# ego's off self: MOBIL has to ask the same question about hypothetical
# worlds ("what would the car behind me in the next lane be doing if I merged
# in front of it?") and the comparison is only meaningful if both sides come
# out of the identical formula.
MOBIL_P = 0.5            # politeness: how much a neighbour's accel change counts
                         # against our own gain (0 = selfish, 1 = altruistic)
MOBIL_A_THR = 0.15       # switching threshold (m/s^2) the net gain must beat
MOBIL_B_SAFE = 4.0       # hard safety limit (m/s^2): never force the vehicle
                         # behind us in the target lane to brake harder
MOBIL_MIN_INTERVAL = 2.0  # s of cooldown after a committed change (hysteresis)
MOBIL_V_MIN = 3.0        # m/s below which we hold the lane; MOBIL is a highway
                         # model and swapping lanes at walking pace is not a
                         # real manoeuvre
MOBIL_SETTLE_TOL = 0.35  # m: only re-decide once this close to the centre of
                         # the lane already being headed for
#: Keep-home bias (m/s^2), the asymmetric threshold that stops the ego camping
#: out of position after a pass. Upstream this is MOBIL_BIAS_RIGHT and it is
#: keyed to +x; here it is keyed to the lane the scenario spawned the ego in.
#: The two agree whenever home is the rightmost lane, and where they differ it
#: is because these scenarios author the ego into the CENTRE lane of three
#: (`cutin`) — a keep-right bias would walk it out of the lane the cut-in was
#: authored against before the orchestrator ever casts a role.
MOBIL_BIAS_HOME = 0.15
#: Extra threshold (m/s^2) for entering a contraflow lane. Upstream has no
#: such term — with signed velocities an oncoming car is just a leader with a
#: negative v and IDM's dv term reads the true closing speed, which is a
#: cleaner argument. It is kept here because `_oncoming_clear` below is a
#: time-to-arrival veto rather than an acceleration test, and the two want to
#: agree about reluctance; `overtake` is the only mode with a contraflow lane.
MOBIL_BIAS_ONCOMING = 0.5

# ---- lane-change trajectory: drivev2.py, PR #1 ---- #
# How MOBIL's yes/no becomes motion. MOBIL is a discrete model with no lateral
# state at all, so the manoeuvre is a lateral displacement profile and the
# steering that realises it is recovered by inverting the bicycle model. This
# REPLACES the pure-pursuit lane tracker the first pass used, and the reason is
# specific to these scenarios: the profile advances with DISTANCE travelled,
# not wall-clock time, so when IDM brakes the ego to a crawl mid-change — which
# is exactly what it does behind a cut-in — the lateral motion slows with it.
# A lookahead tracker instead keeps demanding the same sideways displacement
# with no forward speed left to make it with, and the steering saturates.
LC_DISTANCE = 30.0       # m of road covered by one full lane change
LC_LAT_KP = 0.35         # feedback gain (1/m) on residual lateral error, so
                         # discretisation drift does not accumulate
LC_YAW_MAX = math.radians(14.0)   # cap on the heading excursion asked for
LC_DONE_TOL = 0.5        # m from the lane centre that counts as arrived
STEER_RATE = 4.0         # 1/s cap on normalized steer change (anti-snap)

# ---- lane occupancy / contraflow (this port) ---- #
LC_FRONT_GAP = 8.0       # m of clear road needed ahead in the target lane
LC_REAR_GAP = 6.0        # m of clear road needed behind in the target lane
#: an oncoming lane is only usable if the pass can finish this many times over
ONCOMING_SAFETY = 2.0
#: floor on the ego's closing speed when estimating a pass, so a blocker
#: matching the ego's speed does not make every overtake look infinite
MIN_PASS_CLOSING = 2.0   # m/s
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
    speed: float          # velocity PROJECTED ON THE ROAD AXIS, so a car
                          # coming the other way reports a negative value
    oncoming: bool
    length: float = EGO_LENGTH
    width: float = EGO_WIDTH


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
        #: the active lane-change manoeuvre — {"x0", "x1", "s"} — or None when
        #: simply holding a lane centre. See `_commit`.
        self.lc: Optional[Dict[str, float]] = None
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
            # Project the actor's speed onto the road axis. `Actor.speeds` is
            # an unsigned magnitude along that actor's OWN heading, so an
            # oncoming car and a car driving away both read "+8" until this
            # is applied — and IDM's dv = v_ego - v_lead then reads +4 for a
            # head-on pair closing at 20 m/s, which makes the ego accelerate
            # at a car coming straight for it. With the projection dv is the
            # true closing speed and no oncoming special case is needed
            # anywhere. Same-direction traffic is untouched: cos(0) = 1.
            align = math.cos(math.radians(ah - FORWARD_HEADING))
            along = ay - self.ego.y
            length = float(getattr(a, "length", EGO_LENGTH))
            half = (length + EGO_LENGTH) / 2.0
            out.append(Neighbour(
                actor_id=str(a.id), lane=self.frame.lane_index_of(ax), lat=ax,
                along=along, gap=max(abs(along) - half, 0.5),
                speed=v * align, oncoming=align < 0.0,
                length=length, width=float(getattr(a, "width", EGO_WIDTH))))
        return out

    # ---- lane occupancy, by body overlap ---- #
    def _lane_actors(self, lane_x: float,
                     nbrs: List[Neighbour]) -> List[Neighbour]:
        """The neighbours whose BODY overlaps the lane centred on `lane_x`.

        Membership is by overlap, not by which lane centre a car is nearest,
        so a car mid-merge belongs to *both* lanes it straddles. A centre-point
        test instead teleports a merging car from one lane to the other the
        instant it crosses the line, and that single-frame flip is enough to
        fool MOBIL badly: the lane the merger is vacating reads empty while its
        body is still sitting in it, and the ego dives into occupied space —
        which is the failure mode a cut-in scenario manufactures on purpose.
        Well-centred cars are unaffected: on a 3.5 m grid the adjacent centre
        is 3.5 m away, past the 1.75 + width/2 threshold.
        """
        half_lane = self.frame.lane_width / 2.0
        return [n for n in nbrs
                if abs(n.lat - lane_x) <= half_lane + n.width / 2.0]

    def _lane_neighbours(self, lane_x: float, nbrs: List[Neighbour]
                         ) -> Tuple[Optional[Neighbour], Optional[Neighbour]]:
        """(leader, follower) for the lane centred on `lane_x`.

        A contraflow lane needs no special handling on the leader side: an
        oncoming car ahead is just a leader with a negative speed, and IDM's dv
        term then reads the true closing rate.

        The one asymmetry: a car BEHIND us pointing the other way is receding,
        not following. It cannot be inconvenienced by our merge, and feeding
        its negative speed into the follower's own IDM (where the argument is
        its speed, not a closing rate) would be meaningless. So the follower
        slot takes same-direction traffic only; the leader slot takes
        everything.
        """
        lead = fol = None
        for n in self._lane_actors(lane_x, nbrs):
            if n.along > 0:
                if lead is None or n.along < lead.along:
                    lead = n
            elif not n.oncoming:
                if fol is None or n.along > fol.along:
                    fol = n
        return lead, fol

    @staticmethod
    def _gap_between(rear: Optional[Neighbour],
                     front: Optional[Neighbour]) -> Optional[float]:
        """Bumper-to-bumper gap between two neighbours, `rear` behind `front`."""
        if front is None or rear is None:
            return None
        return max(front.along - rear.along
                   - front.length / 2.0 - rear.length / 2.0, 0.5)

    @staticmethod
    def _gap_to(lead: Optional[Neighbour]) -> Optional[float]:
        """The ego's bumper-to-bumper gap to `lead`."""
        if lead is None:
            return None
        return max(lead.along - lead.length / 2.0 - EGO_LENGTH / 2.0, 0.5)

    @staticmethod
    def _gap_from(fol: Optional[Neighbour]) -> Optional[float]:
        """`fol`'s bumper-to-bumper gap to the ego."""
        if fol is None:
            return None
        return max(-fol.along - fol.length / 2.0 - EGO_LENGTH / 2.0, 0.5)

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
        return best.gap, best.speed

    def _leader_in(self, lane: int, nbrs: List[Neighbour]
                   ) -> Tuple[Optional[float], Optional[float]]:
        """(gap, leader speed) for the closest car ahead in `lane`."""
        return self._leader_near(self.frame.lane_center_x(lane), nbrs)

    def _idm_accel(self, gap: Optional[float], v_lead: Optional[float],
                   v: Optional[float] = None,
                   v0: Optional[float] = None) -> float:
        """The IDM acceleration (m/s^2) for a vehicle at speed `v` with a leader
        `gap` metres ahead doing `v_lead` — or on an open road when `gap` is
        None. drivev2's `idm_control`, before the throttle normalization.

        `v` and `v0` default to the ego's, which is every call the longitudinal
        law makes. They are parameters because MOBIL asks the same question
        about *other* vehicles in hypothetical worlds, and that comparison is
        only meaningful if both sides come out of this one formula.
        """
        v = self.ego.v if v is None else max(float(v), 0.0)
        v0 = self.v0 if v0 is None else max(float(v0), 1.0)
        free = IDM_A * (1.0 - (v / v0) ** IDM_DELTA)
        if gap is None or v_lead is None:
            return free
        dv = v - v_lead
        s_star = IDM_S0 + max(0.0, v * IDM_T
                              + (v * dv) / (2.0 * math.sqrt(IDM_A * IDM_B)))
        return free - IDM_A * (s_star / max(gap, 0.5)) ** 2

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

    def _mobil_evaluate(self, cur: int, cand: int, nbrs: List[Neighbour]
                        ) -> Tuple[bool, float, str]:
        """MOBIL's two tests for moving from lane `cur` to lane `cand`.

        Safety:    the new follower, once we are in front of it, must not be
                   forced to brake harder than `MOBIL_B_SAFE`.
        Incentive: our own acceleration gain, plus `MOBIL_P` times the gain we
                   inflict on the two followers, must beat the switching
                   threshold (raised or lowered by the keep-home bias).

        Background actors here are open-loop scripts, so a neighbour's desired
        speed is simply the speed it is holding — which makes an unobstructed
        follower's IDM acceleration 0, exactly what its script does. They also
        never actually react to the ego, so the politeness term models a
        courtesy the traffic will not reciprocate; the safety term is what
        genuinely protects them.
        """
        cur_x = self.frame.lane_center_x(cur)
        tgt_x = self.frame.lane_center_x(cand)
        v_e = self.ego.v
        lead_c, fol_c = self._lane_neighbours(cur_x, nbrs)
        lead_t, fol_t = self._lane_neighbours(tgt_x, nbrs)

        # --- us: before (staying) vs after (merged) ---
        a_e_cur = self._idm_accel(self._gap_to(lead_c),
                                  lead_c.speed if lead_c else None)
        a_e_new = self._idm_accel(self._gap_to(lead_t),
                                  lead_t.speed if lead_t else None)

        # --- new follower: before (following lead_t) vs after (following us) ---
        if fol_t is None:
            a_nf_cur = a_nf_new = 0.0
        else:
            a_nf_cur = self._idm_accel(self._gap_between(fol_t, lead_t),
                                       lead_t.speed if lead_t else None,
                                       v=fol_t.speed, v0=fol_t.speed)
            a_nf_new = self._idm_accel(self._gap_from(fol_t), v_e,
                                       v=fol_t.speed, v0=fol_t.speed)
            if a_nf_new < -MOBIL_B_SAFE:
                return False, 0.0, "unsafe for follower (%.1f m/s^2)" % a_nf_new

        # --- old follower: before (following us) vs after (following lead_c) ---
        if fol_c is None:
            a_of_cur = a_of_new = 0.0
        else:
            a_of_cur = self._idm_accel(self._gap_from(fol_c), v_e,
                                       v=fol_c.speed, v0=fol_c.speed)
            a_of_new = self._idm_accel(self._gap_between(fol_c, lead_c),
                                       lead_c.speed if lead_c else None,
                                       v=fol_c.speed, v0=fol_c.speed)

        gain = ((a_e_new - a_e_cur)
                + MOBIL_P * ((a_nf_new - a_nf_cur) + (a_of_new - a_of_cur)))
        thr = MOBIL_A_THR
        toward_home = abs(cand - self.home_lane) < abs(cur - self.home_lane)
        thr += -MOBIL_BIAS_HOME if toward_home else MOBIL_BIAS_HOME
        if not self.frame.lanes[cand].same_direction:
            thr += MOBIL_BIAS_ONCOMING
        if gain <= thr:
            return False, gain, "gain %.2f <= threshold %.2f" % (gain, thr)
        return True, gain, "gain %.2f > threshold %.2f" % (gain, thr)

    def choose_lane(self, nbrs: List[Neighbour], now: float) -> int:
        """MOBIL. Returns the lane the ego should be heading for.

        Only re-decides once the previous change has settled (the ego is within
        `MOBIL_SETTLE_TOL` of its target lane centre) and the cooldown has
        expired, so the ego commits to a manoeuvre instead of dithering on the
        lane line.
        """
        cur = self.target_lane
        if not self.allow_lane_change or self.frame.num_lanes < 2:
            return cur
        tgt_x = self.frame.lane_center_x(cur)
        if (now - self._last_change < MOBIL_MIN_INTERVAL
                or self.ego.v < MOBIL_V_MIN
                or abs(self.ego.x - tgt_x) > MOBIL_SETTLE_TOL):
            return cur                  # crawling, or mid-change: let it finish

        best = None
        for cand in (cur - 1, cur + 1):
            if not 0 <= cand < self.frame.num_lanes:
                continue
            oncoming_lane = not self.frame.lanes[cand].same_direction
            if oncoming_lane:
                # The contraflow veto (this port, not upstream): MOBIL's
                # incentive term is an instantaneous acceleration comparison,
                # and a head-on closing at 25 m/s from 60 m away barely dents
                # it while being exactly the thing that kills you. Ask instead
                # whether the pass FITS in the time the oncoming car needs to
                # arrive. `overtake` is the only mode this fires in.
                if not self._lane_clear(cand, nbrs):
                    continue
                gap_cur, v_cur = self._leader_in(cur, nbrs)
                if not self._oncoming_clear(
                        cand, nbrs, (gap_cur or 0.0) + 3.0 * EGO_LENGTH,
                        v_blocker=max(v_cur or 0.0, 0.0)):
                    continue
            ok, gain, why = self._mobil_evaluate(cur, cand, nbrs)
            if ok and (best is None or gain > best[1]):
                best = (cand, gain, why, oncoming_lane)
        if best is None:
            return cur
        cand, gain, why, oncoming_lane = best
        side = "right" if cand > cur else "left"
        self.reason = ("overtake via oncoming lane" if oncoming_lane
                       else "MOBIL: lane %d -> %d (%s, %s)" % (cur, cand, side, why))
        return self._commit(cand, now)

    def _commit(self, lane: int, now: float) -> int:
        """Turn MOBIL's decision into a manoeuvre.

        MOBIL is a *discrete* model: its output is one bit, and in the traffic
        simulations it was written for a lane is an integer index and the
        change is instantaneous. It has no lateral position, no heading and no
        steering angle anywhere in it, so something has to bridge that bit and
        a car with a steering wheel. `self.lc` is that bridge — a lateral
        displacement profile, parameterised by distance travelled.
        """
        if lane != self.target_lane:
            self.n_lane_changes += 1
            self._last_change = now
            if not self.frame.lanes[lane].same_direction:
                self.used_oncoming = True
            self.lc = {"x0": self.ego.x,
                       "x1": self.frame.lane_center_x(lane), "s": 0.0}
            self.target_lane = lane
        return lane

    # ------------------------------------------------------------------ #
    # The control law
    # ------------------------------------------------------------------ #
    @staticmethod
    def _smoothstep(f: float) -> Tuple[float, float, float]:
        """S(f), S'(f), S''(f) for the quintic 10f^3 - 15f^4 + 6f^5 — the
        lateral shape, its slope and its curvature.

        The script layer's own `lane_change` maneuver uses the cubic 3f^2-2f^3,
        and for a scripted actor that is fine: its heading is decorative, so
        nobody differentiates the profile twice. Here the second derivative IS
        the steering command, and the cubic has S''(0)=6, S''(1)=-6 — it would
        demand a step onto full lock at the start of every lane change and
        another step off at the end. The quintic is the minimum-jerk profile
        with S'=S''=0 at both ends, so the steering rises from zero and returns
        to zero.
        """
        f = max(0.0, min(1.0, f))
        return (f ** 3 * (10.0 - 15.0 * f + 6.0 * f * f),
                30.0 * f * f * (1.0 - f) ** 2,
                60.0 * f * (1.0 - f) * (1.0 - 2.0 * f))

    def lane_target_lateral(self) -> Tuple[float, float, float]:
        """Desired (x, dx/dt, d2x/dt2) this instant: the lane-change profile
        while one is running, otherwise hold the target lane centre.

        The profile is parameterised by distance (f = s / LC_DISTANCE), so
        converting its shape derivatives into time derivatives brings in the
        current speed by the chain rule: dx/dt = d*S'(f)*v/L and
        d2x/dt2 = d*S''(f)*(v/L)^2. That v^2 is exactly what cancels when
        `lane_change_steer` divides by v^2 to get curvature, which is what
        makes the manoeuvre speed-independent.
        """
        if self.lc is None:
            return (self.frame.lane_center_x(self.target_lane), 0.0, 0.0)
        lc = self.lc
        d, L = lc["x1"] - lc["x0"], LC_DISTANCE
        v = self.ego.v
        f, df, ddf = self._smoothstep(lc["s"] / L)
        return (lc["x0"] + d * f, d * df * v / L, d * ddf * (v / L) ** 2)

    def lane_change_steer(self) -> float:
        """Steering with no path-tracker involved: invert the kinematic
        bicycle model on the lane-change profile.

        The profile gives lateral velocity and acceleration directly, so for a
        car moving along the road at v the heading it needs is
        psi = atan(n_dot / v) and the path curvature is kappa = n_ddot / v^2
        (small angle). The bicycle model says a steering angle delta produces
        curvature tan(delta)/L, so delta = atan(L * kappa). That term is pure
        feedforward — the steering the manoeuvre *implies*. A proportional term
        on the residual lateral and heading error absorbs integration drift,
        and is what keeps this honest in `--ego-mode physics`, where the ego
        state comes back from CARLA rather than from this file's own
        integrator.

        Everything below is in the ego's LEFT-positive lateral frame, because
        that is the frame the bicycle model steers in: a positive steering
        angle raises theta, which turns the car left. Lanes are laid out along
        script x with +y forward, so left is -x — hence the sign flip on the
        way in. `carla_port.actuation.CarlaEgoActuator` makes the further flip
        into CARLA's clockwise yaw convention; it is not this file's business.
        """
        e = self.ego
        v = max(e.v, 1.0)
        x_des, xd_des, xdd_des = self.lane_target_lateral()
        n_err = -(e.x - x_des)                # + => we are LEFT of the target
        nd_des, ndd_des = -xd_des, -xdd_des
        delta_ff = math.atan(WHEELBASE * ndd_des / (v * v))
        road_theta = math.radians(FORWARD_HEADING)
        psi_des = road_theta + max(-LC_YAW_MAX,
                                   min(LC_YAW_MAX, math.atan2(nd_des, v)))
        psi_err = math.atan2(math.sin(psi_des - e.theta),
                             math.cos(psi_des - e.theta))
        psi_err -= LC_LAT_KP * n_err          # left of target => steer right
        delta = delta_ff + math.atan(WHEELBASE * psi_err / v)
        return max(-1.0, min(1.0, delta / DELTA_MAX))

    def advance_lane_change(self, ds: float) -> None:
        """Advance the active manoeuvre by the distance just travelled, and
        retire it once the profile completes. A stopped ego makes no progress —
        which is the correct behaviour, not a stall.
        """
        if self.lc is None:
            return
        self.lc["s"] += max(0.0, ds)
        if self.lc["s"] >= LC_DISTANCE:
            self.lc = None

    def idm_control(self, nbrs: Optional[List[Neighbour]] = None) -> float:
        """Normalized throttle; negative is braking, as in drivev2."""
        nbrs = self.neighbours() if nbrs is None else nbrs
        # Brake for whichever leader is most urgent. Three queries, because
        # each catches something the others miss:
        #
        #   ego-centred   anyone sitting on top of the ego right now, found by
        #                 lateral distance from the ego's own line rather than
        #                 by lane identity. A car half way through a cut-in
        #                 belongs to neither lane by nearest centre, and an
        #                 index test makes it invisible until it has finished
        #                 merging — which put an ego into the cut-in actor at
        #                 2.8 s before this query existed.
        #   current lane  the lane the ego is nearest, by BODY OVERLAP, and
        #                 including contraflow: an oncoming car ahead is a
        #                 leader with a negative speed, so IDM's dv term reads
        #                 the true closing rate and the ego brakes for a
        #                 head-on instead of ignoring it. Upstream drivev2
        #                 relies on exactly this.
        #   target lane   during a change the ego straddles both, and either
        #                 can block it.
        gap, vl = self._leader_near(self.ego.x, nbrs)       # anyone on top of me
        a = self._idm_accel(gap, vl)
        here = self.frame.lane_index_of(self.ego.x)
        for lane in {here, self.target_lane}:
            lead, _ = self._lane_neighbours(self.frame.lane_center_x(lane), nbrs)
            if lead is not None:
                a = min(a, self._idm_accel(self._gap_to(lead), lead.speed))
        tgt_x = self.frame.lane_center_x(self.target_lane)
        changing = abs(self.ego.x - tgt_x) > LC_DONE_TOL
        if changing:
            g2, v2 = self._leader_in(self.target_lane, nbrs)
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
        """(throttle, steer), both normalized. The policy entry point.

        IDM sets the speed, MOBIL picks the lane, and the lateral profile that
        decision starts supplies the steering. No path-tracking controller is
        involved.
        """
        nbrs = self.neighbours()
        self.choose_lane(nbrs, now)
        throttle = self.idm_control(nbrs)
        steer_cmd = self.lane_change_steer()
        # Advance the profile by the distance the ego is about to cover. In
        # `--ego-mode physics` the true travelled distance is only known next
        # tick, when CARLA reports the new pose; v*dt is that to first order
        # and the LC_LAT_KP feedback term mops up the difference.
        self.advance_lane_change(max(self.ego.v, 0.0) * max(dt, 0.0))
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
