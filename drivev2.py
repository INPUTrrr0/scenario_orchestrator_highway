#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drivev2.py — watch the ego drive itself; the orchestrator keeps its family.

Same session as drive.py, but the ego is driven fully autonomously
(Drive.autonomous_control) — no keyboard input. Longitudinal control is an IDM
(Intelligent Driver Model) car-following policy: throttle/brake to keep a safe
gap+headway behind whichever background actor is closest ahead and roughly in
the ego's own lane, accelerating toward a desired cruise speed on an open road
(Drive.idm_control). Lateral control is a pure-pursuit path-follower
(Drive.path_steer) driving the ego along the route the scenario assumes for it
— straight through the 4-way in "intersection" mode, hold-your-lane down the
highway in "cutin" mode — the same geometry _world()'s ego-prediction uses, so
the ego's real path and the orchestrator's predicted path agree.

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
# Steering is not an IDM concept; see Drive.idm_control / Drive.path_steer.
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
        self._ref_path = self._build_reference_path()   # for path_steer()
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

    def idm_control(self) -> float:
        """Autonomous longitudinal control via the Intelligent Driver Model:
        accelerate toward self.idm_v0, braking as needed for a safe
        gap+headway behind the leader from _idm_leader (or freely toward
        idm_v0 if there is none). IDM is a *longitudinal* car-following model
        only — it has no notion of steering; see path_steer for that.
        Returns a throttle in [-1, 1] (negative = brake): the IDM
        acceleration mapped through this car's A_THROTTLE / A_BRAKE."""
        v = self.ego.v
        gap, v_lead = self._idm_leader()
        if gap is None:
            accel = IDM_A * (1.0 - (v / self.idm_v0) ** IDM_DELTA)
        else:
            dv_rel = v - v_lead
            s_star = IDM_S0 + max(0.0, v * IDM_T
                                  + (v * dv_rel) / (2.0 * math.sqrt(IDM_A * IDM_B)))
            accel = IDM_A * (1.0 - (v / self.idm_v0) ** IDM_DELTA
                             - (s_star / gap) ** 2)
        if accel >= 0:
            return max(0.0, min(1.0, accel / A_THROTTLE))
        return max(-1.0, min(0.0, accel / A_BRAKE))

    # ---- path-following steering (straight through the 4-way in
    # intersection mode; hold the spawn lane in cutin mode) ---- #
    LOOKAHEAD_MIN = 3.0
    LOOKAHEAD_MAX = 12.0
    LOOKAHEAD_GAIN = 0.6   # lookahead = clip(GAIN*v + MIN, MIN, MAX)

    def _build_reference_path(self, step: float = 0.25) -> List[Tuple[float, float]]:
        """Dense (x,y) waypoints for the ego's intended route this session,
        built once at spawn time from the ego's start pose.

        On a straight map (cutin) the intended route is just "keep going" —
        build_route_maneuvers is intersection geometry and doesn't apply — so
        the path is a ray along the start heading, run out past the end of the
        road. On the intersection map it reuses the exact same route geometry
        (mv.build_route_maneuvers, straight through) that _world()'s
        ego-prediction assumes, so the path the ego actually drives and the
        path the orchestrator predicts for it agree with each other. The
        path's SHAPE is speed-invariant (only timing scales with speed), so
        building it with speed=1.0 makes elapsed maneuver-time equal
        arc-length traveled — a convenient way to sample evenly without
        needing the real drive speed."""
        e = self.ego
        if getattr(self.asc.map, "kind", "intersection") == "straight":
            span = self.asc.map.length + 40.0
            ch, sh = math.cos(e.theta), math.sin(e.theta)
            n = max(2, int(span / step))
            return [(e.x + ch * step * i, e.y + sh * step * i) for i in range(n + 1)]
        lw, arm = self.asc.map.lane_width, self.asc.map.arm_length
        man = mv.build_route_maneuvers(e.x, e.y, 90.0, 1.0, "straight", lw, arm)
        pts: List[Tuple[float, float]] = []
        pose = (e.x, e.y, 90.0)
        for m in man:
            n = max(2, int(m.duration / step))
            for k in range(n + 1):
                x, y, _ = m.pose_at(pose, m.duration * k / n)
                pts.append((x, y))
            pose = m.end_pose(pose)
        return pts

    def path_steer(self) -> float:
        """Pure-pursuit steering toward a lookahead point on the reference
        path (_build_reference_path): find the closest path point to the
        ego's current position, walk forward along the path by a
        speed-scaled lookahead distance, and steer toward that target.
        Standard, simple, and robust for this speed range."""
        path, e = self._ref_path, self.ego
        best_i, best_d2 = 0, float("inf")
        for i, (px, py) in enumerate(path):
            d2 = (px - e.x) ** 2 + (py - e.y) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        lookahead = max(self.LOOKAHEAD_MIN,
                        min(self.LOOKAHEAD_MAX,
                            self.LOOKAHEAD_GAIN * e.v + self.LOOKAHEAD_MIN))
        tgt = path[-1]
        acc = 0.0
        for i in range(best_i, len(path) - 1):
            acc += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if acc >= lookahead:
                tgt = path[i + 1]
                break
        alpha = math.atan2(tgt[1] - e.y, tgt[0] - e.x) - e.theta
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), lookahead)
        return max(-1.0, min(1.0, delta / DELTA_MAX))

    def autonomous_control(self) -> Tuple[float, float]:
        """(throttle, steer) for the fully autonomous ego: IDM for
        longitudinal (idm_control), pure-pursuit path-following for lateral
        (path_steer) — holding the ego on the same route the orchestrator's
        own ego-prediction assumes."""
        return self.idm_control(), self.path_steer()

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
            msg = self.interv_msg or "cut-in: IDM ego holds lane, actor merges ahead"
        else:
            msg = ("orchestrator: " + self.interv_msg) if self.interv_msg else \
                  "orchestrator: monitoring…"
        s.blit(self.font_sm.render(msg[:110], True, col), (10, H - BOT + 6))
        s.blit(self.font_sm.render("R record   Esc quit   "
                                   "(ego is IDM + pure-pursuit autonomous)",
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
