#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drivev2.py — watch the ego drive itself; the orchestrator keeps its family.

Same session as drive.py, but the ego is driven fully autonomously
(Drive.autonomous_control) — no keyboard input. Longitudinal control is an IDM
(Intelligent Driver Model) car-following policy: throttle/brake to keep a safe
gap+headway behind whichever background actor is closest ahead and roughly in
the ego's own lane, accelerating toward a desired cruise speed on an open road
(Drive.idm_control). Lateral control is MOBIL (Minimizing Overall Braking
Induced by Lane change), which decides *which lane* to be in
(Drive.mobil_step). Both models score every option with the same IDM
acceleration (Drive._idm_accel), which is what makes IDM+MOBIL a coherent pair
rather than two bolted-together heuristics.

MOBIL is a discrete model: its output is one bit — change lane or don't — and
in the traffic simulations it comes from, a lane is an integer index and the
change is instantaneous. There is no lateral position, heading or steering
angle anywhere in it. Drive.begin_lane_change bridges that gap without a
path-tracking controller: the decision starts a lateral displacement profile
in time (the same smoothstep the repo's own `lane_change` maneuver uses), and
Drive.lane_change_steer recovers the steering by *inverting the bicycle
model* on that profile — the manoeuvre's own lateral acceleration gives the
curvature, curvature gives the steering angle, plus a small proportional term
for drift. So the overtake really is IDM for speed and MOBIL for everything
lateral; there is no pure-pursuit follower, no reference polyline and no
lookahead point.

Contraflow lanes (see scenario_overtake.yaml) need no special case. Actor
velocities are projected onto the road axis in Drive._lane_actors and so are
signed, which means an oncoming car ahead is simply a leader with a negative
velocity and IDM's dv term reads the true closing speed. The ego may enter an
opposing lane to overtake; whether it does is decided by the resulting
acceleration — distance-sensitive, and reassessed every tick — not by any
blanket rule about where the oncoming car is.

What this does NOT give you is commitment. MOBIL is memoryless: it asks "is
that lane better right now", not "can I finish a pull-out, pass and tuck-back
before that car arrives". An overtake begun against a distant oncoming car
can therefore turn bad mid-manoeuvre, and there is no abort path once the
LC_DISTANCE profile is running. A real two-way overtake wants gap acceptance
on top of this.

On the intersection map MOBIL is inert (one lane per direction, nothing to
change into) and the route is straight through the 4-way from an axis-aligned
start, so the correct steering there is exactly zero. Restoring the left/right
turn modes would need a real path-follower again — that is what the deleted
pure-pursuit follower was for.

Background actors stay open-loop maneuver scripts — they never sense the ego or
each other directly. Every tick the orchestrator evaluates the red-light family
(D1/D2/D3, script-grounded, against a forward prediction of the ego) and applies
the minimal causal intervention to the *other* actors when needed — it never
steers the ego. The whole session is one rollout and can be recorded to an mp4
under outputs/.

Note the ego is a closed-loop IDM driver in "cutin" mode too: it no longer
replays the YAML ego script (the hand-authored 13->4 m/s brake), it brakes when
and only when the merging actor actually enters its lane.

Controls (session control only — the ego drives itself)
  R        start/stop recording        Esc / Q   quit

Modes
  intersection (default) — random actors at a 4-way; red-light orchestrator
  cutin                  — straight 3-lane highway; closed-loop cut-in actor

Usage
  python3 drivev2.py [--seed N] [--record]
  python3 drivev2.py --mode cutin [--scenario path] [--record]
  python3 drivev2.py --mode cutin --headless --duration 8 --seed 1
Requires pygame; recording requires ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pygame                          # noqa: E402
import scenario_editor as se          # noqa: E402
import maps as mp                     # noqa: E402
import directives as dv               # noqa: E402
import directives_script as ds        # noqa: E402
import maneuvers as mv                # noqa: E402
import cutin_orchestrator as co       # noqa: E402  (shared closed-loop cut-in)

# view / window
VIEW = 42.0
SCALE = 10.0
SIZE = int(2 * VIEW * SCALE)          # 840
TOP = 52
BOT = 46
W, H = SIZE, TOP + SIZE + BOT
FPS = 30
DT = 1.0 / FPS
ORCH_EVERY = 3          # run the orchestrator every N frames (physics/render every frame)
CAP_T = 9.0             # cap actor timeline length (keeps re-simulation cheap)

# bicycle model
WHEELBASE = 2.8
A_THROTTLE = 5.0
A_BRAKE = 9.0
V_MAX = 18.0
DELTA_MAX = math.radians(32)
DRAG = 1.0                            # gentle coast deceleration

# IDM (Intelligent Driver Model) — autonomous longitudinal ego control.
# Steering is not an IDM concept; see Drive.idm_control / Drive.mobil_step.
IDM_V0 = 12.0        # desired/free-flow cruise speed (m/s) — a comfortable
                     # target speed, distinct from V_MAX (the hard physical cap)
IDM_V0_CUTIN = 13.0  # same, on the highway (matches the cut-in YAML's ego cruise,
                     # so the ego holds the speed that scenario was authored around)
IDM_A = 3.0          # max acceleration (m/s^2)
IDM_B = 3.0          # comfortable/desired deceleration (m/s^2)
IDM_S0 = 2.0         # minimum gap / jam distance, bumper-to-bumper (m)
IDM_T = 1.5          # desired time headway (s)
IDM_DELTA = 4        # acceleration exponent (standard IDM value)
IDM_LANE_TOL = 2.2   # lateral tolerance (m, in the ego's own heading frame)
                     # for "roughly in my lane" when looking for a leader

# MOBIL (Minimizing Overall Braking Induced by Lane change) — autonomous
# lateral *decisions* on the straight (highway) map: which lane to be in.
# MOBIL picks a target lane; begin_lane_change/lane_change_steer turn that
# decision into motion. Every acceleration it compares is an IDM acceleration
# (_idm_accel), so the two models stay consistent — that pairing is the
# standard IDM/MOBIL combination.
MOBIL_P = 0.5            # politeness: how much a neighbour's accel change counts
                         # against our own gain (0 = selfish, 1 = altruistic)
MOBIL_A_THR = 0.15       # switching threshold (m/s^2) — the net gain a change
                         # must beat, so we don't swap lanes over rounding noise
MOBIL_B_SAFE = 4.0       # hard safety limit (m/s^2): never force the vehicle
                         # behind us in the target lane to brake harder than this
MOBIL_BIAS_RIGHT = 0.15  # keep-right bias (m/s^2): makes moving right cheaper
                         # and moving left dearer, so the ego drifts back right
                         # after a pass instead of camping in the fast lane
MOBIL_MIN_INTERVAL = 2.0  # s of cooldown after a committed change (hysteresis)
MOBIL_V_MIN = 3.0        # m/s: below this, hold the lane. MOBIL is a highway
                         # model — swapping lanes at walking pace isn't a real
                         # manoeuvre, and it's exactly where pure pursuit
                         # saturates (short lookahead + a full lane of offset
                         # demands more steering angle than DELTA_MAX)
MOBIL_SETTLE_TOL = 0.35  # m: only re-decide once we're this close to the centre
                         # of the lane we're already heading for

# Lane-change trajectory — how MOBIL's yes/no decision becomes motion. MOBIL
# itself has no lateral state (see Drive.begin_lane_change), so the manoeuvre
# is a lateral displacement profile in time, and the steering that realises it
# is recovered by inverting the bicycle model (Drive.lane_change_steer).
LC_DISTANCE = 30.0       # m of road covered by one full lane change. The
                         # profile advances with DISTANCE, not wall-clock time,
                         # which is what makes it robust: IDM can brake to a
                         # crawl mid-change (it does, behind a cut-in), and a
                         # time-based profile would then demand the same
                         # sideways motion with no forward speed left to do it
                         # with — heading blows up and the steering saturates.
                         # On distance, curvature works out to
                         # d*S''(f)/LC_DISTANCE^2: entirely speed-independent,
                         # and lateral motion simply stops when the car does.
LC_LAT_KP = 0.35         # feedback gain (1/m) on residual lateral error, so
                         # discretisation drift doesn't accumulate
LC_YAW_MAX = math.radians(14.0)   # cap on the heading excursion we ask for
EGO_LEN = 4.5            # the ego body length this file draws/collides with

SIGNALS = {"N": "green", "S": "green", "E": "red", "W": "red"}
ACTOR_COLORS = [(210, 90, 80), (240, 175, 65), (80, 140, 220), (170, 110, 220),
                (80, 200, 200), (150, 190, 100), (225, 120, 175), (200, 140, 70)]

GRASS = (32, 44, 34)
ROAD = (60, 60, 66)
LINE = (220, 210, 120)
EDGE = (200, 200, 200)
BAR = (20, 22, 28)
TXT = (230, 232, 238)
MUTED = (150, 156, 168)
EGO_COL = (90, 200, 120)

DEFAULT_CUTIN = os.path.join(HERE, "scenarios", "scenario_cutin.yaml")
CAM = [0.0, 0.0]  # camera center (world m); follows ego in cut-in mode


@dataclass
class Ego:
    x: float
    y: float
    theta: float          # radians
    v: float


def w2s(x, y):
    return (int(SIZE / 2 + (x - CAM[0]) * SCALE),
            int(TOP + SIZE / 2 - (y - CAM[1]) * SCALE))


# --------------------------------------------------------------------------- #
# Random scenario configuration
# --------------------------------------------------------------------------- #
LEG_POSE = {  # inbound leg -> (lane-fixed coord, heading deg, axis) at distance d
    "EN": lambda d: (d, 1.75, 180.0),      # east arm, westbound
    "WS": lambda d: (-d, -1.75, 0.0),      # west arm, eastbound
    "NW": lambda d: (-1.75, d, 270.0),     # north arm, southbound
    "SE": lambda d: (1.75, -d, 90.0),      # south arm, northbound (ego uses this)
}


def spawn(seed: int, arm: float = 60.0, lw: float = 3.5):
    """Return (ego, actors_scenario) for a random configuration."""
    rng = random.Random(seed)
    ego = Ego(x=1.75, y=-rng.uniform(40, 52), theta=math.radians(90),
              v=rng.uniform(5, 9))
    actors: List[se.Actor] = []
    k = rng.randint(3, 6)
    legs = ["EN", "WS", "NW"]              # conflicting/ crossing approaches
    for i in range(k):
        leg = rng.choice(legs)
        d = rng.uniform(20, 56)
        cx, cy, hd = {"EN": (d, 1.75, 180.0), "WS": (-d, -1.75, 0.0),
                      "NW": (-1.75, d, 270.0)}[leg]
        speed = rng.uniform(7, 14)
        route = rng.choice(["straight", "straight", "left", "right"])
        man = mv.build_route_maneuvers(cx, cy, hd, speed, route, lw, arm)
        # speed profiles over the scenario: sometimes a stop-and-go or a ramp
        r = rng.random()
        if r < 0.25 and man:                       # brief stop partway
            man.insert(1, se.Maneuver(type="stop", duration=rng.uniform(0.6, 1.4)))
        elif r < 0.45 and man and man[0].type == "go_straight":
            man[0].type = "accelerate"             # ramp up
            man[0].slope = rng.uniform(1.0, 3.0)
        actors.append(se.Actor(id=str(i + 1),
                               color=ACTOR_COLORS[i % len(ACTOR_COLORS)],
                               length=4.5, width=2.0, start=(cx, cy, hd),
                               maneuvers=man))
    sc = se.Scenario(map=se.MapConfig(lw, arm), actors=actors, pixels_per_meter=6.0)
    _cap_scenario(sc)
    return ego, sc



def spawn_cutin(path: str):
    """Load a straight-road cut-in YAML: ego from actor 0, others stay scripted."""
    sc = se.load_scenario(path)
    if sc.map.kind != "straight":
        raise ValueError(f"cut-in scenario must have map.kind=straight, got {sc.map.kind!r}")
    by_id = {a.id: a for a in sc.actors}
    if "0" not in by_id:
        raise ValueError("cut-in scenario needs actor id 0 (ego template)")
    ego_a = by_id["0"]
    v0 = 0.0
    if ego_a.maneuvers and isinstance(ego_a.maneuvers[0], se.Maneuver):
        v0 = ego_a.maneuvers[0].velocity_at(0.0)
    ego = Ego(x=ego_a.start[0], y=ego_a.start[1],
              theta=math.radians(ego_a.start[2]), v=max(v0, 0.0))
    others = [a for a in sc.actors if a.id != "0"]
    if not others:
        raise ValueError("cut-in scenario needs at least one non-ego actor")
    asc = se.Scenario(map=se.clone_map(sc.map), actors=others,
                      pixels_per_meter=sc.pixels_per_meter)
    asc.simulate()
    return ego, asc


def _cap_scenario(sc: se.Scenario) -> None:
    """Trim each actor's last maneuver so total <= CAP_T — bounds re-simulation
    cost without changing near-term behavior."""
    for a in sc.actors:
        a.compute_schedule()
        if a.total > CAP_T and a.maneuvers:
            a.maneuvers[-1].duration = max(0.1, a.maneuvers[-1].duration
                                           - (a.total - CAP_T))
    sc.simulate()


# --------------------------------------------------------------------------- #
# The driving session
# --------------------------------------------------------------------------- #
class Drive:
    def __init__(self, seed: int, record: bool, headless: bool,
                 mode: str = "intersection", scenario: Optional[str] = None):
        self.mode = mode
        self.seed = seed
        self.scenario_path = scenario
        self.prm = dv.Params(H=9.0)
        ds.SWEEP_STRIDE = 4              # coarse body sweep (~0.067s) for real-time
        if mode == "cutin":
            path = scenario or DEFAULT_CUTIN
            self.ego, self.asc = spawn_cutin(path)
            self.scenario_path = path
        else:
            self.ego, self.asc = spawn(seed)
        # desired cruise speed + the route the ego intends to drive; both are
        # fixed for the session, so build the reference path once, up front
        self.idm_v0 = IDM_V0_CUTIN if mode == "cutin" else IDM_V0
        # MOBIL state — straight (highway) map only; an intersection arm has a
        # single lane per direction, so there is nothing to change into
        self._road_theta = self.ego.theta      # lane direction, fixed for the
                                               # session (the ego's heading
                                               # tilts mid-change; lanes don't)
        self.lanes = self._lane_centers()      # world x of each lane centre
        self.target_lane = (self._lane_index(self.ego.x) if self.lanes else None)
        self.mobil_cooldown = 0.0
        self.mobil_msg = ""
        self.n_lane_changes = 0
        # active lane-change manoeuvre (None when tracking a lane centre)
        self.lc: Optional[dict] = None
        self.atime = 0.0                 # elapsed since the actors' last re-base
        self.clock_t = 0.0               # absolute session time (never rebased)
        # cut-in constraint (ego-relative at fixed t) — tracked here since
        # rebasing rebuilds Actor objects and would drop the spec
        self.cutin_spec: Optional[dict] = None
        self.cutin_id: Optional[str] = None
        self.cutin_committed = False
        self.cutin_outcome: Optional[str] = None  # "merged" | "abandoned"
        self.cutin_heading: Optional[float] = None  # actor's nominal lane heading
        if mode == "cutin":
            for a in self.asc.actors:
                if getattr(a, "cutin", None):
                    self.cutin_spec = dict(a.cutin)
                    self.cutin_id = a.id
                    self.cutin_heading = (a.start[2] if len(a.start) > 2
                                          else 90.0)
                    break
        self.standing: Dict[str, tuple] = {}
        self.pursuer: Optional[str] = None      # the red-runner currently tracking the ego
        self.interv_msg = ""
        self.flash = 0
        self.n_interventions = 0
        self._fcount = 0
        self.headless = headless
        self.hit: Optional[str] = None
        pygame.init()
        flags = 0
        self.screen = pygame.display.set_mode((W, H), flags)
        if not headless:
            title = ("drivev2 — cut-in (autonomous ego)" if mode == "cutin"
                     else f"drivev2 — seed {seed} (autonomous ego)")
            pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_sm = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)
        self.clock = pygame.time.Clock()
        self.outdir = os.path.join(HERE, "outputs")
        os.makedirs(self.outdir, exist_ok=True)
        self.ff = None
        self.outfile = None
        self.verdict = None
        if record:
            self._start_record()
        if mode == "intersection":
            self.orchestrate()               # populate the first verdict
        else:
            self.interv_msg = "scripted cut-in (ego free)"

    # ---- recording ---- #
    def _start_record(self):
        if self.ff is not None:
            return
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = "cutin" if self.mode == "cutin" else f"seed{self.seed}"
        base = os.path.join(self.outdir, f"drive_{tag}_{ts}")
        self.outfile = base + ".mp4"
        self.cmd_file = base + ".commands.json"
        self.cmd_log: List[dict] = []      # ego actuation commands over the recording
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-loglevel", "error",
             self.outfile], stdin=subprocess.PIPE)

    def _record_cmd(self, throttle: float, steer: float):
        """Log one frame of ego actuation (only while recording)."""
        if self.ff is None:
            return
        self.cmd_log.append({
            "t": round(self._fcount * DT, 3),
            "throttle": round(throttle, 3), "steer": round(steer, 3),
            "x": round(self.ego.x, 3), "y": round(self.ego.y, 3),
            "heading": round(math.degrees(self.ego.theta), 2),
            "v": round(self.ego.v, 3)})

    def _stop_record(self):
        if self.ff is not None:
            self.ff.stdin.close()
            self.ff.wait()
            self.ff = None
            with open(self.cmd_file, "w") as f:
                json.dump({"seed": self.seed, "fps": FPS, "dt": DT,
                           "wheelbase": WHEELBASE, "v_max": V_MAX,
                           "commands": self.cmd_log}, f, indent=1)
            print(f"commands saved: {self.cmd_file}")

    def _grab_frame(self):
        if self.ff is not None:
            self.ff.stdin.write(pygame.image.tostring(self.screen, "RGB"))

    # ---- ego dynamics (kinematic bicycle) ---- #
    def integrate_ego(self, throttle: float, steer: float):
        e = self.ego
        if throttle > 0:
            a = A_THROTTLE * throttle
        elif throttle < 0:
            a = A_BRAKE * throttle          # brake toward 0
        else:
            a = -DRAG if e.v > 0 else 0.0   # coast
        e.v = max(0.0, min(V_MAX, e.v + a * DT))
        delta = DELTA_MAX * steer
        e.theta += (e.v / WHEELBASE) * math.tan(delta) * DT
        e.x += e.v * math.cos(e.theta) * DT
        e.y += e.v * math.sin(e.theta) * DT

    # ---- autonomous ego control (IDM, longitudinal only) ---- #
    def _idm_leader(self) -> Tuple[Optional[float], Optional[float]]:
        """The closest background actor ahead of the ego and roughly in its
        lane — found in the ego's own heading frame (forward distance > 0,
        lateral offset within IDM_LANE_TOL), not by lane index or leg/route
        identity, so it naturally picks up a cut-in actor at the moment that
        actor's body actually crosses into the ego's lane. Returns
        (gap, v_lead) bumper-to-bumper — gap floored at 0.5m to keep the IDM
        formula's division well-behaved — or (None, None) if nothing
        qualifies (open road: IDM reduces to accelerating toward idm_v0)."""
        if self.lanes:
            # Highway: ask by lane, in the ROAD frame. Using the ego's own
            # heading here would be wrong mid-change — a 14deg yaw excursion
            # throws the lateral test by >4m at 20m ahead, i.e. more than a
            # lane, so the ego would track a car in the wrong lane exactly
            # when it can least afford to. During a change both the lane we
            # are leaving and the one we are entering can block us, so take
            # whichever leader is closer.
            cands = {self.target_lane, self._lane_index(self.ego.x)}
            best = None
            for li in cands:
                lead, _ = self._lane_neighbors(self.lanes[li])
                if lead is None:
                    continue
                g = self._gap_to(lead)
                if best is None or g < best[0]:
                    best = (g, lead[1])
            return best if best else (None, None)
        e = self.ego
        ch, sh = math.cos(e.theta), math.sin(e.theta)
        self.asc.simulate()
        k = int(round(self.atime / se.DT))
        best_fwd, best_gap, best_v = None, None, None
        for a in self.asc.actors:
            kk = max(0, min(k, len(a.traj) - 1))
            ax, ay, _ = a.traj[kk]
            dx, dy = ax - e.x, ay - e.y
            fwd = dx * ch + dy * sh
            lat = -dx * sh + dy * ch
            if fwd <= 0 or abs(lat) > IDM_LANE_TOL:
                continue
            if best_fwd is None or fwd < best_fwd:
                best_fwd = fwd
                best_gap = max(fwd - 4.5 / 2 - a.length / 2, 0.5)
                best_v = a.speeds[kk]
        return best_gap, best_v

    def _idm_accel(self, v: float, gap: Optional[float] = None,
                   v_lead: Optional[float] = None,
                   v0: Optional[float] = None) -> float:
        """The IDM acceleration (m/s^2) for a vehicle at speed `v` with a
        leader `gap` metres ahead doing `v_lead` — or on an open road when
        `gap` is None. Factored out of idm_control because MOBIL has to ask
        the same question about *hypothetical* worlds ("what would that car
        behind me in the left lane be doing if I merged in front of it?"),
        and the comparison is only meaningful if both sides come from the
        identical formula. `v0` defaults to the ego's desired speed."""
        v0 = self.idm_v0 if v0 is None else max(v0, 1.0)
        free = IDM_A * (1.0 - (max(v, 0.0) / v0) ** IDM_DELTA)
        if gap is None or v_lead is None:
            return free
        dv_rel = v - v_lead
        s_star = IDM_S0 + max(0.0, v * IDM_T
                              + (v * dv_rel) / (2.0 * math.sqrt(IDM_A * IDM_B)))
        return free - IDM_A * (s_star / max(gap, 0.5)) ** 2

    def idm_control(self) -> float:
        """Autonomous longitudinal control via the Intelligent Driver Model:
        accelerate toward self.idm_v0, braking as needed for a safe
        gap+headway behind the leader from _idm_leader (or freely toward
        idm_v0 if there is none). IDM is a *longitudinal* car-following model
        only — it has no notion of steering; see mobil_step for that.
        Returns a throttle in [-1, 1] (negative = brake): the IDM
        acceleration mapped through this car's A_THROTTLE / A_BRAKE."""
        gap, v_lead = self._idm_leader()
        accel = self._idm_accel(self.ego.v, gap, v_lead)
        if accel >= 0:
            return max(0.0, min(1.0, accel / A_THROTTLE))
        return max(-1.0, min(0.0, accel / A_BRAKE))

    # ---- MOBIL lane-change decisions (straight/highway map only) ---- #
    def _lane_centers(self) -> List[float]:
        """World x of every lane centre, or [] on a map without lanes to
        change between (the intersection arms are one lane per direction)."""
        m = self.asc.map
        if getattr(m, "kind", "intersection") != "straight":
            return []
        return [m.lane_center_x(i) for i in range(max(1, m.num_lanes))]

    def _lane_index(self, x: float) -> int:
        """Index of the lane whose centre is nearest world-x `x`."""
        return min(range(len(self.lanes)), key=lambda i: abs(self.lanes[i] - x))

    def _lane_actors(self, lane_x: float):
        """Every background actor currently in the lane centred on `lane_x`,
        as (fwd, v, length, opposing) — `fwd` is the signed along-road offset
        of its centre from the ego's (+ = ahead), and `opposing` flags a car
        pointing back at us.

        `v` is the actor's velocity PROJECTED ONTO THE ROAD AXIS, and so is
        signed: a car coming the other way reports a negative value. This is
        the whole reason IDM can reason about contraflow traffic. Actor
        speeds out of the simulator (se.Maneuver.velocity_at) are unsigned
        magnitudes along each actor's *own* heading, so an oncoming car and a
        car driving away from us both read "+8" until they are projected.
        Feeding the raw magnitude to IDM makes dv = v_ego - v_lead read +4
        for a head-on pair closing at 20 m/s, and the ego cheerfully
        accelerates at a car coming straight at it. With the projection, dv
        is the true closing speed and the s* term does the rest — no special
        case for oncoming traffic is needed anywhere. Same-direction traffic
        is unaffected: cos(0) = +1, so the projection is a no-op there.

        Membership is by BODY OVERLAP, not by which lane centre the car is
        nearest, so a car mid-merge belongs to *both* lanes it is straddling.
        A centre-point test instead teleports a merging car from one lane to
        the other the instant it crosses the lane line, and that single-frame
        flip is enough to fool MOBIL badly: the lane the merger is vacating
        reads as empty while its body is still sitting in it, and the ego
        dives into occupied space. Overlap keeps the merger in both lanes
        until it has genuinely cleared one. Well-centred cars are unaffected
        — on a 3.5m grid the adjacent centre is 3.5m away, past the
        1.75+width/2 threshold."""
        e = self.ego
        ch, sh = math.cos(self._road_theta), math.sin(self._road_theta)
        half_lane = self.asc.map.lane_width / 2.0
        self.asc.simulate()
        k = int(round(self.atime / se.DT))
        out = []
        for a in self.asc.actors:
            kk = max(0, min(k, len(a.traj) - 1))
            ax, ay, ahd = a.traj[kk]
            dx, dy = ax - e.x, ay - e.y
            # lateral coord on a +y road is world x
            if abs(ax - lane_x) > half_lane + a.width / 2.0:
                continue
            fwd = dx * ch + dy * sh
            align = math.cos(math.radians(ahd) - self._road_theta)
            out.append((fwd, a.speeds[kk] * align, a.length, align < 0.0))
        return out

    def _lane_neighbors(self, lane_x: float):
        """(leader, follower) for the lane centred on `lane_x` — the nearest
        actor ahead of / behind the ego there, as (fwd, v, length), or None.

        A contraflow lane needs no special handling: an oncoming car ahead is
        just a leader with a negative v, and IDM's dv term then reads the true
        closing speed. There is deliberately no veto on entering such a lane —
        whether the ego may overtake into it is decided by the resulting
        acceleration, which is distance-sensitive, rather than by a blanket
        rule that ignores where the oncoming car actually is.

        The one asymmetry: a car BEHIND us pointing the other way is receding,
        not following. It cannot be inconvenienced by our merge, and feeding
        its negative v into the follower's own IDM (where v is its speed, not
        a closing rate) would be meaningless. So the follower slot takes
        same-direction traffic only; the leader slot takes everything."""
        lead = fol = None
        for fwd, v, ln, opp in self._lane_actors(lane_x):
            if fwd > 0 and (lead is None or fwd < lead[0]):
                lead = (fwd, v, ln)
            elif fwd <= 0 and not opp and (fol is None or fwd > fol[0]):
                fol = (fwd, v, ln)
        return lead, fol

    @staticmethod
    def _gap_between(rear, front) -> Optional[float]:
        """Bumper-to-bumper gap between two (fwd, v, length) records, rear
        behind front. None when there is no vehicle in front."""
        if front is None:
            return None
        if rear is None:
            return None
        return max(front[0] - rear[0] - front[2] / 2.0 - rear[2] / 2.0, 0.5)

    def _gap_to(self, lead) -> Optional[float]:
        """Ego's bumper-to-bumper gap to a (fwd, v, length) leader."""
        if lead is None:
            return None
        return max(lead[0] - lead[2] / 2.0 - EGO_LEN / 2.0, 0.5)

    def _gap_from(self, fol) -> Optional[float]:
        """A (fwd, v, length) follower's bumper-to-bumper gap to the ego."""
        if fol is None:
            return None
        return max(-fol[0] - fol[2] / 2.0 - EGO_LEN / 2.0, 0.5)

    def _mobil_evaluate(self, cur_x: float, tgt_x: float, to_right: bool):
        """MOBIL's two tests for moving from the lane at `cur_x` to the one at
        `tgt_x`. Returns (ok, gain, reason).

        Safety:    the new follower, once we are in front of it, must not be
                   forced to brake harder than MOBIL_B_SAFE.
        Incentive: our own acceleration gain, plus MOBIL_P times the gain we
                   inflict on the two followers, must beat the switching
                   threshold (raised/lowered by the keep-right bias).

        Background actors here are open-loop constant-speed scripts, so a
        neighbour's "desired speed" is simply the speed it is holding — that
        makes an unobstructed follower's IDM acceleration 0, which is exactly
        what its script does. They also never actually react to us; the
        politeness term therefore models courtesy the traffic won't
        reciprocate, while the safety term is what genuinely protects them."""
        v_e = self.ego.v
        lead_c, fol_c = self._lane_neighbors(cur_x)
        lead_t, fol_t = self._lane_neighbors(tgt_x)

        # --- us: before (staying) vs after (merged) ---
        a_e_cur = self._idm_accel(v_e, self._gap_to(lead_c),
                                  lead_c[1] if lead_c else None)
        a_e_new = self._idm_accel(v_e, self._gap_to(lead_t),
                                  lead_t[1] if lead_t else None)

        # --- new follower: before (following lead_t) vs after (following us) ---
        if fol_t is None:
            a_nf_cur = a_nf_new = 0.0
        else:
            a_nf_cur = self._idm_accel(fol_t[1], self._gap_between(fol_t, lead_t),
                                       lead_t[1] if lead_t else None, v0=fol_t[1])
            a_nf_new = self._idm_accel(fol_t[1], self._gap_from(fol_t), v_e,
                                       v0=fol_t[1])
            if a_nf_new < -MOBIL_B_SAFE:
                return False, 0.0, "unsafe for follower (%.1f m/s^2)" % a_nf_new

        # --- old follower: before (following us) vs after (following lead_c) ---
        if fol_c is None:
            a_of_cur = a_of_new = 0.0
        else:
            a_of_cur = self._idm_accel(fol_c[1], self._gap_from(fol_c), v_e,
                                       v0=fol_c[1])
            a_of_new = self._idm_accel(fol_c[1], self._gap_between(fol_c, lead_c),
                                       lead_c[1] if lead_c else None, v0=fol_c[1])

        gain = ((a_e_new - a_e_cur)
                + MOBIL_P * ((a_nf_new - a_nf_cur) + (a_of_new - a_of_cur)))
        thr = MOBIL_A_THR + (-MOBIL_BIAS_RIGHT if to_right else MOBIL_BIAS_RIGHT)
        if gain <= thr:
            return False, gain, "gain %.2f <= threshold %.2f" % (gain, thr)
        return True, gain, "gain %.2f > threshold %.2f" % (gain, thr)

    def mobil_step(self) -> None:
        """Pick this tick's target lane. Only re-decides once the previous
        change has settled (the ego is within MOBIL_SETTLE_TOL of its target
        lane centre) and the cooldown has expired, so the ego commits to a
        manoeuvre instead of dithering on the lane line. Committing hands
        off to begin_lane_change, which starts the lateral profile that
        lane_change_steer then turns into steering."""
        if not self.lanes or len(self.lanes) < 2:
            return
        self.mobil_cooldown = max(0.0, self.mobil_cooldown - DT)
        tgt_x = self.lanes[self.target_lane]
        if (self.mobil_cooldown > 0.0
                or self.ego.v < MOBIL_V_MIN
                or abs(self.ego.x - tgt_x) > MOBIL_SETTLE_TOL):
            return                      # crawling, or mid-change: let it finish
        cur = self.target_lane
        best = None
        for cand in (cur - 1, cur + 1):
            if not 0 <= cand < len(self.lanes):
                continue
            to_right = cand > cur       # +x is the ego's right on a +y road
            ok, gain, why = self._mobil_evaluate(self.lanes[cur],
                                                 self.lanes[cand], to_right)
            if ok and (best is None or gain > best[1]):
                best = (cand, gain, why)
        if best is None:
            return
        cand, gain, why = best
        side = "right" if cand > cur else "left"
        self.begin_lane_change(cand)
        self.mobil_cooldown = MOBIL_MIN_INTERVAL
        self.n_lane_changes += 1
        self.flash = 10
        self.mobil_msg = "MOBIL: lane %d -> %d (%s, %s)" % (cur, cand, side, why)

    # ---- lane-change trajectory + the steering that realises it ---- #
    def begin_lane_change(self, lane: int) -> None:
        """Turn MOBIL's decision into a manoeuvre. MOBIL is a *discrete*
        model — its output is one bit, "change or not", and in the traffic
        simulations it was written for a lane is an integer index and the
        change is instantaneous. It has no lateral position, no heading and
        no steering angle anywhere in it, so something has to bridge the gap
        between that bit and a car with a steering wheel. That bridge is
        this: a lateral displacement profile in time, from the lane we're in
        to the lane MOBIL picked.

        The profile is the same smoothstep the repo's own `lane_change`
        maneuver uses (se.Maneuver.pose_at), so the ego's lane change has the
        identical shape as a scripted actor's. Duration scales mildly with
        speed: a change is a roughly fixed *distance* manoeuvre, so at low
        speed it needs longer."""
        # `s` is distance travelled into the manoeuvre, not elapsed time — see
        # LC_DISTANCE. No speed scaling is needed anywhere as a result.
        self.lc = {"x0": self.ego.x, "x1": self.lanes[lane], "s": 0.0}
        self.target_lane = lane

    @staticmethod
    def _smoothstep(f: float) -> Tuple[float, float, float]:
        """S(f), S'(f), S''(f) for the quintic 10f^3-15f^4+6f^5 — the lateral
        shape, its slope and its curvature.

        The repo's own `lane_change` maneuver uses the cubic 3f^2-2f^3
        instead, and for a scripted actor that is fine: its heading is
        decorative, so nobody differentiates the profile twice. Here the
        second derivative IS the steering command, and the cubic has
        S''(0)=6, S''(1)=-6 — it would demand a step onto full lock at the
        start of every lane change and another step off at the end. The
        quintic is the minimum-jerk profile with S'=S''=0 at both ends, so
        the steering rises from zero and returns to zero."""
        f = max(0.0, min(1.0, f))
        return (f ** 3 * (10.0 - 15.0 * f + 6.0 * f * f),
                30.0 * f * f * (1.0 - f) ** 2,
                60.0 * f * (1.0 - f) * (1.0 - 2.0 * f))

    def lane_target_lateral(self) -> Tuple[float, float, float]:
        """Desired (x, dx/dt, d2x/dt2) this instant: the lane-change profile
        while one is running, otherwise just hold the target lane centre.

        The profile is parameterised by distance (f = s / LC_DISTANCE), so
        converting its shape derivatives into time derivatives brings in the
        current speed by the chain rule: dx/dt = d*S'(f)*v/L and
        d2x/dt2 = d*S''(f)*(v/L)^2. The v^2 is exactly what cancels when
        lane_change_steer divides by v^2 to get curvature."""
        if self.lc is None:
            return (self.lanes[self.target_lane], 0.0, 0.0) if self.lanes \
                else (self.ego.x, 0.0, 0.0)
        lc = self.lc
        d, L = lc["x1"] - lc["x0"], LC_DISTANCE
        v = self.ego.v
        s, ds, dds = self._smoothstep(lc["s"] / L)
        return (lc["x0"] + d * s, d * ds * v / L, d * dds * (v / L) ** 2)

    def lane_change_steer(self) -> float:
        """Steering for the highway, with no path-tracker involved: invert
        the kinematic bicycle model on the lane-change profile.

        The profile gives lateral velocity and acceleration directly, so for
        a car moving along the road at v: the heading it needs is
        psi = atan(n_dot / v), and the path curvature is kappa = n_ddot / v^2
        (small-angle). The bicycle model says a steering angle delta produces
        curvature tan(delta)/L, so delta = atan(L * kappa). That term is pure
        feedforward — it is the steering the manoeuvre *implies*. A small
        proportional term on the residual lateral and heading error absorbs
        integration drift. No lookahead point, no reference polyline, no
        closest-point search: the manoeuvre defines the steering.

        Everything below is in the ego's LEFT-positive lateral frame, because
        that is the frame the bicycle model steers in: a positive steering
        angle raises theta, which turns the car left. The lanes are laid out
        along world x (maps.py builds the strip along +y), and for a
        northbound road left is -x — so world-x quantities flip sign on the
        way in. `sgn` is that flip, written so a southbound road works too."""
        e = self.ego
        v = max(e.v, 1.0)
        x_des, xd_des, xdd_des = self.lane_target_lateral()
        sgn = -math.sin(self._road_theta)     # d(left) / d(world x)
        n_err = sgn * (e.x - x_des)           # + => we are LEFT of the target
        nd_des, ndd_des = sgn * xd_des, sgn * xdd_des
        delta_ff = math.atan(WHEELBASE * ndd_des / (v * v))
        # desired heading: road direction tilted by the lateral velocity
        psi_des = self._road_theta + max(-LC_YAW_MAX,
                                         min(LC_YAW_MAX, math.atan2(nd_des, v)))
        psi_err = math.atan2(math.sin(psi_des - e.theta),
                             math.cos(psi_des - e.theta))
        psi_err -= LC_LAT_KP * n_err          # left of target => steer right
        delta = delta_ff + math.atan(WHEELBASE * psi_err / v)
        return max(-1.0, min(1.0, delta / DELTA_MAX))

    def advance_lane_change(self) -> None:
        """Advance the active manoeuvre by the distance just travelled, and
        retire it once the profile completes. A stopped ego makes no
        progress — which is the correct behaviour, not a stall."""
        if self.lc is None:
            return
        self.lc["s"] += self.ego.v * DT
        if self.lc["s"] >= LC_DISTANCE:
            self.lc = None

    def autonomous_control(self) -> Tuple[float, float]:
        """(throttle, steer) for the fully autonomous ego.

        Highway: IDM sets speed, MOBIL decides the lane, and the lane-change
        profile it starts supplies the steering (lane_change_steer). No
        path-tracking controller is involved.

        Intersection: the route is straight through the 4-way from an
        axis-aligned start, so the correct steering is exactly zero and there
        is nothing to track. (Turns would need a real path-follower again —
        see the note in the module docstring.)"""
        if not self.lanes:
            return self.idm_control(), 0.0
        self.mobil_step()
        steer = self.lane_change_steer()
        self.advance_lane_change()
        return self.idm_control(), steer

    # ---- directive world: ego prediction + scripted actors ---- #
    def _world(self) -> se.Scenario:
        base = mv.rebase_scenario(self.asc, self.atime)   # actors at "now"
        if getattr(self.asc.map, "kind", "intersection") == "straight":
            span = self.asc.map.length + 12.0
        else:
            span = (2 * self.asc.map.arm_length + 12.0)
        ego_v = max(self.ego.v, 0.6)
        ego_dur = min(span / ego_v, self.prm.H)   # predict over the horizon only
        ego_act = se.Actor(id="0", color=EGO_COL, length=4.5, width=2.0,
                           start=(self.ego.x, self.ego.y, math.degrees(self.ego.theta)),
                           maneuvers=[se.Maneuver(type="go_straight",
                                                  intercept=ego_v, slope=0.0,
                                                  duration=ego_dur)])
        w = se.Scenario(map=se.clone_map(base.map),
                        actors=[ego_act] + base.actors, pixels_per_meter=6.0)
        w.simulate()
        return w

    def orchestrate(self):
        """Pursuer orchestration: every tick, designate an uncommitted red-runner
        and re-aim it onto the ego's predicted crossing (slow it to wait if the ego
        coasts, speed it if the ego bolts) — rather than sitting idle whenever a
        collision is merely *predicted*. Also clears any third-vehicle interferer."""
        w = self._world()
        self.verdict = ds.evaluate(w, "0", SIGNALS, self.prm, fast=True)
        # candidate retimes that put an uncommitted red-runner on the ego (cheapest first)
        opts = ds.d1_options(w, "0", self.prm, turns=("__keep__",), verify=False)
        plan = {}                                    # actor -> target speed
        if opts:
            # keep shadowing with the same pursuer while it stays viable
            opt = next((o for o in opts if o[1] == self.pursuer), opts[0])
            plan[opt[1]] = opt[3]
            new_pursuer = opt[1]
        else:
            new_pursuer = None

        if plan:
            self.asc = mv.rebase_scenario(self.asc, self.atime)
            self.atime = 0.0
            by = {a.id: a for a in self.asc.actors}
            for aid, v_t in plan.items():
                if aid in by:
                    mv.retime_actor(by[aid], float(v_t))
            _cap_scenario(self.asc)
            if new_pursuer != self.pursuer:
                self.n_interventions += 1
                self.flash = 10
            self.interv_msg = ("pursuer %s -> %.1f m/s"
                               % (new_pursuer, plan.get(new_pursuer, 0.0))
                               if new_pursuer else "clearing interferer")
        else:
            self.interv_msg = "no uncommitted red-runner can reach you"
        self.pursuer = new_pursuer

    def orchestrate_cutin(self):
        """Closed-loop cut-in (delegates to cutin_orchestrator): chase the pin
        glued to the live ego until merge, or abandon past deadline `cutin.t`."""
        spec = self.cutin_spec
        if spec is None or self.cutin_committed or self.hit:
            return
        base = mv.rebase_scenario(self.asc, self.atime)
        a = next((x for x in base.actors if x.id == self.cutin_id), None)
        if a is None:
            return
        old_v = (a.maneuvers[0].intercept
                 if a.maneuvers and isinstance(a.maneuvers[0], se.Maneuver)
                 else 0.0)
        status, msg = co.apply_closed_loop_cutin(a, self.ego, spec, self.clock_t,
                                                 heading_deg=self.cutin_heading)
        self.asc = base
        self.atime = 0.0
        self.asc.simulate()
        self.interv_msg = msg
        if status in ("merged", "abandoned"):
            self.cutin_committed = True
            self.cutin_outcome = status
            if status == "abandoned":
                self.flash = 8
            return
        new_v = a.maneuvers[0].intercept
        if abs(new_v - old_v) > 1.0:
            self.flash = 8
            self.n_interventions += 1

    def actor_poses(self):
        self.asc.simulate()
        k = int(round(self.atime / se.DT))
        out = []
        for a in self.asc.actors:
            kk = max(0, min(k, len(a.traj) - 1))
            out.append((a, a.traj[kk]))
        return out

    def real_collision(self) -> Optional[str]:
        eb = _rect(self.ego.x, self.ego.y, math.degrees(self.ego.theta), 4.5, 2.0)
        for a, pose in self.actor_poses():
            if dv.rects_overlap(eb, _rect(*pose, a.length, a.width)):
                return a.id
        return None

    # ---- stepping ---- #
    def step(self, throttle: float, steer: float):
        self.integrate_ego(throttle, steer)
        self.atime += DT
        self.clock_t += DT
        self._fcount += 1
        if self.mode == "intersection" and self._fcount % ORCH_EVERY == 0:
            self.orchestrate()
        elif self.mode == "cutin" and self._fcount % ORCH_EVERY == 0:
            self.orchestrate_cutin()
        if self.flash > 0:
            self.flash -= 1
        hit = self.real_collision()
        if hit and self.hit is None:
            self.hit = hit
            self.flash = 20
            self.interv_msg = f"collision with actor {hit}"

    # ---- rendering ---- #
    def _update_camera(self):
        if self.mode == "cutin":
            look = 8.0
            CAM[0] = self.ego.x + look * math.cos(self.ego.theta)
            CAM[1] = self.ego.y + look * math.sin(self.ego.theta)
        else:
            CAM[0], CAM[1] = 0.0, 0.0

    def render(self):
        self._update_camera()
        s = self.screen
        s.fill(GRASS)
        extend = 40.0 if getattr(self.asc.map, "kind", "") == "straight" else 0.0
        mp.draw_map(s, self.asc.map, w2s, road=ROAD, line=LINE, edge=EDGE,
                    extend_y=extend)
        if getattr(self.asc.map, "kind", "intersection") != "straight":
            self._draw_signals()
        for a, pose in self.actor_poses():
            self._draw_body(pose, a.length, a.width, tuple(a.color), a.id)
        self._draw_body((self.ego.x, self.ego.y, math.degrees(self.ego.theta)),
                        4.5, 2.0, EGO_COL, "0", ring=True)
        mp.draw_rulers(s, w2s, self._s2w,
                       pygame.Rect(0, TOP, SIZE, SIZE),
                       font=self.font_sm)
        self._draw_strips()
        pygame.display.flip()

    def _s2w(self, sx: float, sy: float):
        return ((sx - SIZE / 2) / SCALE + CAM[0],
                CAM[1] - (sy - TOP - SIZE / 2) / SCALE)

    def _draw_signals(self):
        """Traffic-light dots on intersection arms (drive-mode only)."""
        lw = self.asc.map.lane_width
        cx, cy = w2s(0, 0)
        for arm_k, (sx, sy) in {"N": (0, lw), "S": (0, -lw), "E": (lw, 0),
                                "W": (-lw, 0)}.items():
            px, py = w2s(sx, sy)
            vx, vy = px - cx, py - cy
            nrm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
            px, py = px + vx / nrm * 16, py + vy / nrm * 16
            red = SIGNALS.get(arm_k) == "red"
            pygame.draw.circle(self.screen, (220, 40, 40) if red else (60, 200, 80),
                               (int(px), int(py)), 6)

    def _draw_body(self, pose, L, Wd, col, label, ring=False):
        pts = [w2s(*c) for c in _corners(*pose, L, Wd)]
        pygame.draw.polygon(self.screen, col, pts)
        pygame.draw.polygon(self.screen, (18, 18, 18), pts, 1)
        pygame.draw.line(self.screen, (250, 250, 250), pts[0], pts[1], 3)
        if ring:
            c = w2s(pose[0], pose[1])
            pygame.draw.circle(self.screen, (255, 255, 255), c, 3)
        lp = w2s(pose[0], pose[1])
        t = self.font_sm.render(label, True, (10, 10, 10))
        self.screen.blit(t, (lp[0] - t.get_width() // 2, lp[1] - 7))

    def _draw_strips(self):
        s = self.screen
        pygame.draw.rect(s, BAR, (0, 0, W, TOP))
        title = "drivev2 — cut-in" if self.mode == "cutin" else "drivev2"
        s.blit(self.font_big.render(title, True, TXT), (10, 6))
        s.blit(self.font.render(f"speed {self.ego.v:4.1f} m/s", True, TXT), (150, 10))
        if self.mode == "cutin":
            status = "HIT" if self.hit else "cut-in actor scripted"
            col = (224, 82, 82) if self.hit else (88, 194, 106)
            r = pygame.Rect(360, 8, 200, 22)
            pygame.draw.rect(s, col, r, border_radius=4)
            s.blit(self.font_sm.render(status, True, (10, 10, 10)), (r.x + 8, r.y + 4))
            if self.lanes:
                changing = self.lc is not None
                lane_txt = "lane %d/%d%s  changes %d" % (
                    self.target_lane, len(self.lanes) - 1,
                    " (changing)" if changing else "", self.n_lane_changes)
                s.blit(self.font_sm.render(lane_txt, True,
                                           (255, 205, 40) if changing else MUTED),
                       (580, 12))
        else:
            v = self.verdict
            if v is not None:
                for i, (nm, ev) in enumerate((("D1", v.d1), ("D2", v.d2), ("D3", v.d3))):
                    c = (88, 194, 106) if ev.value else (224, 82, 82)
                    r = pygame.Rect(320 + i * 62, 8, 56, 22)
                    pygame.draw.rect(s, c, r, border_radius=4)
                    s.blit(self.font_sm.render(f"{nm}:{'OK' if ev.value else 'X'}", True,
                                               (10, 10, 10)), (r.x + 6, r.y + 4))
                s.blit(self.font_sm.render(f"hero {v.hero}", True, MUTED), (320 + 3 * 62, 12))
        if self.ff is not None:
            pygame.draw.circle(s, (230, 60, 60), (W - 20, 16), 7)
            s.blit(self.font_sm.render("REC", True, (230, 120, 120)), (W - 52, 10))
        # bottom strip: intervention + controls
        pygame.draw.rect(s, BAR, (0, H - BOT, W, BOT))
        col = (56, 178, 198) if self.flash > 0 else MUTED
        if self.mode == "cutin":
            msg = self.mobil_msg or self.interv_msg or \
                "highway: IDM speed + MOBIL lane choice"
        else:
            msg = ("orchestrator: " + self.interv_msg) if self.interv_msg else \
                  "orchestrator: monitoring…"
        s.blit(self.font_sm.render(msg[:110], True, col), (10, H - BOT + 6))
        s.blit(self.font_sm.render("R record   Esc quit   "
                                   "(ego: IDM speed + MOBIL lane change)",
                                   True, MUTED), (10, H - BOT + 26))

    # ---- loops ---- #
    def loop(self):
        alive = True
        while alive:
            self.clock.tick(FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        alive = False
                    elif e.key == pygame.K_r:
                        self._stop_record() if self.ff else self._start_record()
            throttle, steer = self.autonomous_control()
            self.step(throttle, steer)
            self._record_cmd(throttle, steer)
            self.render()
            self._grab_frame()
        self._stop_record()
        pygame.quit()

    def run_headless(self, duration: float, policy=None):
        """Auto-drive for `duration` s (default: IDM + path-following),
        recording. Pass `policy(self, t) -> (throttle, steer)` to override the
        autonomous controller with something else for testing."""
        steps = int(duration / DT)
        for i in range(steps):
            if policy:
                throttle, steer = policy(self, i * DT)
            else:
                throttle, steer = self.autonomous_control()
            self.step(throttle, steer)
            self._record_cmd(throttle, steer)
            self.render()
            self._grab_frame()
        self._stop_record()
        pygame.quit()


def _corners(x, y, hd, L, Wd):
    h = math.radians(hd)
    fx, fy = math.cos(h), math.sin(h)
    px, py = -math.sin(h), math.cos(h)
    a, b = L / 2, Wd / 2
    return [(x + fx * a + px * b, y + fy * a + py * b),
            (x + fx * a - px * b, y + fy * a - py * b),
            (x - fx * a - px * b, y - fy * a - py * b),
            (x - fx * a + px * b, y - fy * a + py * b)]


def _rect(x, y, hd, L, Wd):
    return dv.rect_corners(x, y, hd, L, Wd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("intersection", "cutin"),
                    default="intersection",
                    help="intersection (default) or cutin (3-lane highway)")
    ap.add_argument("--scenario", default=None,
                    help="YAML for --mode cutin (default: scenarios/scenario_cutin.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="no interaction; auto-drive for --duration and record")
    ap.add_argument("--duration", type=float, default=8.0)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    d = Drive(seed, record=(args.record or args.headless), headless=args.headless,
              mode=args.mode, scenario=args.scenario)
    if args.headless:
        d.run_headless(args.duration)
        print(f"mode={args.mode} seed {seed}  ->  {d.outfile}"
              + (f"  hit={d.hit}" if d.hit else ""))
    else:
        d.loop()
        if d.outfile:
            print(f"recording saved: {d.outfile}")


if __name__ == "__main__":
    main()