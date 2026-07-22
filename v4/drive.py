#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/drive.py — drive the ego yourself; the orchestrator keeps the red-light family.

You control the ego with a kinematic bicycle model (throttle/brake + steering).
Background actors are spawned from a random seed (varied start positions, speed
profiles, and routes — turn or straight). Every tick the orchestrator evaluates the
red-light family (D1/D2/D3, script-grounded, against a forward prediction of your
ego) and applies the minimal causal intervention to the *other* actors when needed
— it never steers your ego. The whole session is one rollout and can be recorded to
an mp4 under v4/outputs/.

Controls
  ↑ / W    throttle          ↓ / S    brake / reverse-to-stop
  ← / A    steer left        → / D    steer right
  R        start/stop recording        Esc / Q   quit

Usage
  python3 drive.py [--seed N] [--record]
  python3 drive.py --headless --duration 8 --record --seed 3   # no display, auto-drive
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
V2 = os.path.join(HERE, "..", "v2")
for p in (V2, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pygame                          # noqa: E402
import scenario_editor as se          # noqa: E402
import directives as dv               # noqa: E402
import directives_script as ds        # noqa: E402
import maneuvers as mv                # noqa: E402

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


@dataclass
class Ego:
    x: float
    y: float
    theta: float          # radians
    v: float


def w2s(x, y):
    return int(SIZE / 2 + x * SCALE), int(TOP + SIZE / 2 - y * SCALE)


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
    def __init__(self, seed: int, record: bool, headless: bool):
        self.seed = seed
        self.prm = dv.Params(H=9.0)
        ds.SWEEP_STRIDE = 4              # coarse body sweep (~0.067s) for real-time
        self.ego, self.asc = spawn(seed)
        self.atime = 0.0                 # elapsed since the actors' last re-base
        self.standing: Dict[str, tuple] = {}
        self.interv_msg = ""
        self.flash = 0
        self.n_interventions = 0
        self._fcount = 0
        self.headless = headless
        pygame.init()
        flags = 0
        self.screen = pygame.display.set_mode((W, H), flags)
        if not headless:
            pygame.display.set_caption(f"v4 drive — seed {seed}")
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
        self.orchestrate()               # populate the first verdict

    # ---- recording ---- #
    def _start_record(self):
        if self.ff is not None:
            return
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        base = os.path.join(self.outdir, f"drive_seed{self.seed}_{ts}")
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

    # ---- directive world: ego prediction + scripted actors ---- #
    def _world(self) -> se.Scenario:
        base = mv.rebase_scenario(self.asc, self.atime)   # actors at "now"
        span = (2 * self.asc.map.arm_length + 12.0)
        ego_v = max(self.ego.v, 0.6)
        ego_dur = min(span / ego_v, self.prm.H)   # predict over the horizon only
        ego_act = se.Actor(id="0", color=EGO_COL, length=4.5, width=2.0,
                           start=(self.ego.x, self.ego.y, math.degrees(self.ego.theta)),
                           maneuvers=[se.Maneuver(type="go_straight",
                                                  intercept=ego_v, slope=0.0,
                                                  duration=ego_dur)])
        w = se.Scenario(map=se.MapConfig(base.map.lane_width, base.map.arm_length),
                        actors=[ego_act] + base.actors, pixels_per_meter=6.0)
        w.simulate()
        return w

    def orchestrate(self):
        w = self._world()
        self.verdict = ds.evaluate(w, "0", SIGNALS, self.prm, fast=True)
        rr = ds.repair(w, "0", SIGNALS, self.prm, fast=True)
        if rr.feasible and rr.interventions:
            key = {iv.actor: (("retime", round(float(iv.value), 1))
                              if iv.kind == "retime" else ("reroute", iv.value))
                   for iv in rr.interventions if iv.actor != "0"}
            if key and key != {a: self.standing.get(a) for a in key}:
                self.asc = mv.rebase_scenario(self.asc, self.atime)
                self.atime = 0.0
                by = {a.id: a for a in self.asc.actors}
                lw, arm = self.asc.map.lane_width, self.asc.map.arm_length
                for iv in rr.interventions:
                    if iv.actor == "0" or iv.actor not in by:
                        continue
                    if iv.kind == "retime":
                        mv.retime_actor(by[iv.actor], float(iv.value))
                    else:
                        mv.reroute_actor(by[iv.actor], str(iv.value), lw, arm)
                _cap_scenario(self.asc)
                self.standing.update(key)
                self.interv_msg = "  ".join(str(iv) for iv in rr.interventions)
                self.flash = 10
                self.n_interventions += 1
        elif self.verdict.ok:
            pass

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
        self._fcount += 1
        if self._fcount % ORCH_EVERY == 0:      # orchestrate a few times per second
            self.orchestrate()
        if self.flash > 0:
            self.flash -= 1

    # ---- rendering ---- #
    def render(self):
        s = self.screen
        s.fill(GRASS)
        lw, arm = self.asc.map.lane_width, self.asc.map.arm_length
        b = lw * SCALE
        cx, cy = w2s(0, 0)
        pygame.draw.rect(s, ROAD, (cx - b, TOP, 2 * b, SIZE))
        pygame.draw.rect(s, ROAD, (0, cy - b, SIZE, 2 * b))
        for p in range(int(b + 1.6 * SCALE), int(SIZE / 2), int(3.6 * SCALE)):
            for sgn in (1, -1):
                pygame.draw.line(s, LINE, (cx, cy + sgn * p),
                                 (cx, cy + sgn * min(p + 2 * SCALE, SIZE / 2)), 2)
                pygame.draw.line(s, LINE, (cx + sgn * p, cy),
                                 (cx + sgn * min(p + 2 * SCALE, SIZE / 2), cy), 2)
        for arm_k, (sx, sy) in {"N": (0, lw), "S": (0, -lw), "E": (lw, 0),
                                "W": (-lw, 0)}.items():
            px, py = w2s(sx, sy)
            vx, vy = px - cx, py - cy
            n = max((vx * vx + vy * vy) ** 0.5, 1e-6)
            px, py = px + vx / n * 16, py + vy / n * 16
            red = SIGNALS.get(arm_k) == "red"
            pygame.draw.circle(s, (220, 40, 40) if red else (60, 200, 80),
                               (int(px), int(py)), 6)
        # actors
        for a, pose in self.actor_poses():
            self._draw_body(pose, a.length, a.width, tuple(a.color), a.id)
        # ego
        self._draw_body((self.ego.x, self.ego.y, math.degrees(self.ego.theta)),
                        4.5, 2.0, EGO_COL, "0", ring=True)
        self._draw_strips()
        pygame.display.flip()

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
        s.blit(self.font_big.render("v4 drive", True, TXT), (10, 6))
        s.blit(self.font.render(f"speed {self.ego.v:4.1f} m/s", True, TXT), (150, 10))
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
        msg = ("orchestrator: " + self.interv_msg) if self.interv_msg else \
              "orchestrator: monitoring…"
        s.blit(self.font_sm.render(msg[:110], True, col), (10, H - BOT + 6))
        s.blit(self.font_sm.render("↑/↓ throttle/brake   ←/→ steer   R record   "
                                   "Esc quit", True, MUTED), (10, H - BOT + 26))

    # ---- loops ---- #
    def loop(self):
        alive = True
        while alive:
            self.clock.tick(FPS)
            throttle = steer = 0.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        alive = False
                    elif e.key == pygame.K_r:
                        self._stop_record() if self.ff else self._start_record()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 1.0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                throttle = -1.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer = 1.0
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer = -1.0
            self.step(throttle, steer)
            self._record_cmd(throttle, steer)
            self.render()
            self._grab_frame()
        self._stop_record()
        pygame.quit()

    def run_headless(self, duration: float, policy=None):
        """Auto-drive for `duration` s (default: accelerate then hold), recording."""
        steps = int(duration / DT)
        for i in range(steps):
            if policy:
                throttle, steer = policy(self, i * DT)
            else:
                throttle, steer = (1.0 if self.ego.v < 12 else 0.0), 0.0
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
    d = Drive(seed, record=(args.record or args.headless), headless=args.headless)
    if args.headless:
        d.run_headless(args.duration)
        print(f"seed {seed}  ->  {d.outfile}")
    else:
        d.loop()
        if d.outfile:
            print(f"recording saved: {d.outfile}")


if __name__ == "__main__":
    main()
