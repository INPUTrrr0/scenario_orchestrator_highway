#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cutin_orchestrator.py — closed-loop cut-in orchestration with role casting.

Standalone session (pygame): ego is user-driven; background actors are cast by
the orchestrator into either:
  * cutin   — chase the ego-relative pin until merge / deadline (green light)
  * nominal — keep driving straight at cruise speed (grey light)

Role casting picks the most promising non-ego actor each orchestration tick
(while no cut-in is committed) using a simple score: adjacent-lane preference ×
longitudinal proximity to a preferred cut-in station ahead of the ego.

Usage
  .venv/bin/python cutin_orchestrator.py [--seed N] [--actors K]
  .venv/bin/python cutin_orchestrator.py --headless --duration 8 --seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pygame                          # noqa: E402
import scenario_editor as se          # noqa: E402
import maps as mp                     # noqa: E402
import maneuvers as mv                # noqa: E402

# ---- view ---- #
VIEW = 42.0
SCALE = 10.0
SIZE = int(2 * VIEW * SCALE)          # 840
TOP = 52
BOT = 46
PANEL_W = 200
W, H = SIZE, TOP + SIZE + BOT
FPS = 30
DT = 1.0 / FPS
ORCH_EVERY = 3

# bicycle
WHEELBASE = 2.8
A_THROTTLE = 5.0
A_BRAKE = 9.0
V_MAX = 18.0
DELTA_MAX = math.radians(32)
DRAG = 1.0

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
PANEL_BG = (22, 24, 30)
PANEL_BANNER = (36, 40, 50)
DOT_ON = (72, 190, 100)
DOT_OFF = (90, 94, 105)
ROLE_CUTIN = "cutin"
ROLE_NOMINAL = "nominal"

DEFAULT_CUTIN_SPEC = {
    "t": 4.0, "along": 6.0, "lat": 0.0, "lc_duration": 2.0, "tail": 4.0,
}
CRUISE_SPEED = 12.0
CAM = [0.0, 0.0]


@dataclass
class Ego:
    x: float
    y: float
    theta: float          # radians
    v: float


@dataclass
class Casting:
    actor_id: str
    role: str             # ROLE_CUTIN | ROLE_NOMINAL
    score: float


def w2s(x, y):
    return (int(SIZE / 2 + (x - CAM[0]) * SCALE),
            int(TOP + SIZE / 2 - (y - CAM[1]) * SCALE))


# --------------------------------------------------------------------------- #
# Spawn: ego + random fleet on a straight multi-lane road
# --------------------------------------------------------------------------- #
def spawn_fleet(seed: int, n_actors: int = 4,
                num_lanes: int = 3, lane_width: float = 3.5,
                length: float = 120.0, cruise: float = CRUISE_SPEED
                ) -> Tuple[Ego, se.Scenario]:
    """Random northbound fleet. Ego in center lane; others in random lanes."""
    rng = random.Random(seed)
    m = se.MapConfig(lane_width=lane_width, kind="straight",
                     num_lanes=num_lanes, length=length)
    ego_lane = num_lanes // 2
    ego_x = m.lane_center_x(ego_lane)
    ego_y = -length / 2.0 + 18.0
    ego = Ego(x=ego_x, y=ego_y, theta=math.radians(90), v=cruise)

    actors: List[se.Actor] = []
    used: List[Tuple[int, float]] = []
    for i in range(max(1, n_actors)):
        for _try in range(40):
            lane = rng.randrange(num_lanes)
            y = ego_y + rng.uniform(6.0, 45.0)
            if all(lane != ul or abs(y - uy) > 8.0 for ul, uy in used):
                used.append((lane, y))
                break
        else:
            lane = i % num_lanes
            y = ego_y + 10.0 + 9.0 * i
            used.append((lane, y))
        x = m.lane_center_x(lane)
        speed = cruise + rng.uniform(-1.5, 1.5)
        start = (x, y, 90.0)
        actors.append(se.Actor(
            id=str(i + 1),
            color=ACTOR_COLORS[i % len(ACTOR_COLORS)],
            length=4.5, width=2.0, start=start,
            maneuvers=se.cruise_plan(start, speed),
        ))
    asc = se.Scenario(map=m, actors=actors, pixels_per_meter=6.0)
    asc.simulate()
    return ego, asc


# --------------------------------------------------------------------------- #
# Role casting
# --------------------------------------------------------------------------- #
def score_cutin_candidate(actor_pose: se.Pose, ego_pose: se.Pose,
                          lane_width: float) -> float:
    """Higher = better cut-in candidate.

    Prefers an adjacent lane and a station roughly `along` metres ahead of the
    ego. Same-lane cars score near zero (no cut-in to perform).
    `ego_pose` is (x, y, heading_deg).
    """
    along, lat = se.world_to_ego_offset(ego_pose, actor_pose[0], actor_pose[1])
    # lateral: adjacent lane ~ lane_width; same lane ~ 0
    abs_lat = abs(lat)
    if abs_lat < 0.4 * lane_width:
        lat_score = 0.05                      # already in ego lane
    elif abs_lat < 1.6 * lane_width:
        lat_score = 1.0                       # adjacent
    else:
        lat_score = 0.35                      # two lanes over — still possible
    # longitudinal: prefer ~8–20 m ahead; penalize far ahead / behind
    if along < -8.0:
        along_score = 0.05
    elif along > 50.0:
        along_score = 0.1
    else:
        along_score = 1.0 / (1.0 + abs(along - 14.0) / 12.0)
    return lat_score * along_score


def cast_roles(actors: List[se.Actor], ego_pose: se.Pose, lane_width: float,
               lock_id: Optional[str] = None) -> List[Casting]:
    """Pick one cut-in actor (or keep `lock_id`); everyone else drives nominal."""
    scored: List[Tuple[float, str]] = []
    for a in actors:
        pose = a.start
        scored.append((score_cutin_candidate(pose, ego_pose, lane_width), a.id))
    scored.sort(reverse=True)
    if lock_id and any(aid == lock_id for _, aid in scored):
        chosen = lock_id
    elif scored and scored[0][0] > 0.08:
        chosen = scored[0][1]
    else:
        chosen = None
    by_score = {aid: s for s, aid in scored}
    out: List[Casting] = []
    for a in actors:
        role = ROLE_CUTIN if a.id == chosen else ROLE_NOMINAL
        out.append(Casting(actor_id=a.id, role=role,
                           score=by_score.get(a.id, 0.0)))
    return out


# --------------------------------------------------------------------------- #
# Closed-loop cut-in (moved from drive.orchestrate_cutin)
# --------------------------------------------------------------------------- #
def apply_closed_loop_cutin(actor: se.Actor, ego: Ego, spec: dict,
                            clock_t: float, heading_deg: Optional[float] = None
                            ) -> Tuple[str, str]:
    """Replan `actor` toward the live ego-relative pin.

    Returns (status, message) where status is one of:
      'chasing' | 'merged' | 'abandoned'
    Mutates actor.start / actor.maneuvers in place.
    """
    along = float(spec.get("along", 1.0))
    lat = float(spec.get("lat", 0.0))
    wx, wy = se.live_cutin_pin(ego.x, ego.y, ego.theta, along, lat)
    hd = heading_deg if heading_deg is not None else math.degrees(ego.theta)
    start = (actor.start[0], actor.start[1], hd)

    if se.cutin_is_merged(actor.start, wx, wy, ego.theta):
        actor.start = start
        actor.maneuvers = se.cruise_plan(start, ego.v)
        return "merged", "cut-in merged — actor matching ego"

    t_rem = se.closed_loop_cutin_horizon(spec, clock_t)
    if t_rem <= 0.0:
        actor.start = start
        actor.maneuvers = se.cruise_plan(start, ego.v)
        return ("abandoned",
                f"cut-in abandoned — past t={float(spec['t']):.1f}s, cruising")

    actor.start = start
    actor.maneuvers = se.solve_closed_loop_cutin(
        start, wx, wy, t_rem, ego.v,
        lc_duration=float(spec.get("lc_duration", 2.0)),
        tail=30.0)
    new_v = actor.maneuvers[0].intercept
    return ("chasing",
            f"chase pin +{along:.1f}m @ t={clock_t:.1f}s: "
            f"actor {actor.id} -> {new_v:.1f} m/s")


def apply_nominal(actor: se.Actor, speed: float,
                  heading_deg: Optional[float] = None) -> None:
    """Keep driving straight at cruise speed."""
    hd = (heading_deg if heading_deg is not None
          else (actor.start[2] if len(actor.start) > 2 else 90.0))
    start = (actor.start[0], actor.start[1], hd)
    actor.start = start
    actor.maneuvers = se.cruise_plan(start, speed)


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
class CutinOrchestrator:
    """Role-casting + closed-loop cut-in over a fleet of scripted actors."""

    def __init__(self, cutin_spec: Optional[dict] = None,
                 cruise_speed: float = CRUISE_SPEED):
        self.spec = dict(cutin_spec or DEFAULT_CUTIN_SPEC)
        self.cruise_speed = float(cruise_speed)
        self.roles: Dict[str, str] = {}
        self.scores: Dict[str, float] = {}
        self.cutin_id: Optional[str] = None
        self.committed = False
        self.outcome: Optional[str] = None     # merged | abandoned
        self.msg = "role casting…"
        self.n_interventions = 0
        self.flash = 0

    def cast(self, actors: List[se.Actor], ego: Ego, lane_width: float,
             sticky: bool = True) -> List[Casting]:
        """(Re)assign roles. Sticky keeps the current cut-in actor while viable."""
        lock = self.cutin_id if (sticky and not self.committed) else None
        ego_pose = (ego.x, ego.y, math.degrees(ego.theta))
        castings = cast_roles(actors, ego_pose, lane_width, lock_id=lock)
        self.roles = {c.actor_id: c.role for c in castings}
        self.scores = {c.actor_id: c.score for c in castings}
        new_id = next((c.actor_id for c in castings if c.role == ROLE_CUTIN),
                      None)
        if new_id != self.cutin_id and not self.committed:
            self.cutin_id = new_id
            self.n_interventions += 1
            self.flash = 8
            if new_id:
                self.msg = (f"cast actor {new_id} as cut-in "
                            f"(score {self.scores.get(new_id, 0):.2f})")
            else:
                self.msg = "no viable cut-in candidate — all nominal"
        return castings

    def tick(self, asc: se.Scenario, atime: float, ego: Ego, clock_t: float
             ) -> Tuple[se.Scenario, float]:
        """One orchestration step: rebase, cast, plan each actor, resimulate."""
        base = mv.rebase_scenario(asc, atime)
        lw = base.map.lane_width
        if not self.committed:
            self.cast(base.actors, ego, lw, sticky=True)

        for a in base.actors:
            role = self.roles.get(a.id, ROLE_NOMINAL)
            if role == ROLE_CUTIN and not self.committed:
                status, msg = apply_closed_loop_cutin(a, ego, self.spec, clock_t)
                self.msg = msg
                if status in ("merged", "abandoned"):
                    self.committed = True
                    self.outcome = status
                    self.flash = 10
                    self.n_interventions += 1
            else:
                # committed cut-in actor (or never cast) → cruise
                speed = (ego.v if (self.committed and a.id == self.cutin_id)
                         else self.cruise_speed)
                apply_nominal(a, speed)

        base.simulate()
        return base, 0.0

    # ---- panel ---- #
    def draw_panel(self, surface: pygame.Surface,
                   font: pygame.font.Font, font_sm: pygame.font.Font,
                   actor_ids: List[str],
                   origin: Tuple[int, int] = (36, TOP + 36)) -> pygame.Rect:
        """Top-left 'orchestrator' role-casting card."""
        roles = {aid: self.roles.get(aid, ROLE_NOMINAL) for aid in actor_ids}
        return draw_role_panel(surface, font, font_sm, roles, origin)


# --------------------------------------------------------------------------- #
# Role-casting panel (reused by scenario_editor's window)
# --------------------------------------------------------------------------- #
def draw_role_panel(surface: pygame.Surface,
                    font: pygame.font.Font, font_sm: pygame.font.Font,
                    roles: Dict[str, str],
                    origin: Tuple[int, int] = (36, TOP + 36),
                    width: int = PANEL_W) -> pygame.Rect:
    """Card with an 'orchestrator' banner and one row per actor: a round dot
    that is green when the actor is cast as the cut-in, grey when nominal."""
    x, y = origin
    row_h = 26
    header_h = 28
    pad = 10
    h = header_h + pad + max(1, len(roles)) * row_h + pad
    rect = pygame.Rect(x, y, width, h)

    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, (70, 74, 86), rect, 1, border_radius=6)
    ban = pygame.Rect(x, y, width, header_h)
    pygame.draw.rect(surface, PANEL_BANNER, ban,
                     border_top_left_radius=6, border_top_right_radius=6)
    surface.blit(font.render("orchestrator", True, TXT), (x + 10, y + 6))

    yy = y + header_h + pad // 2
    for aid, role in roles.items():
        on = role == ROLE_CUTIN
        cx, cy = x + 18, yy + row_h // 2
        pygame.draw.circle(surface, DOT_ON if on else DOT_OFF, (cx, cy), 7)
        pygame.draw.circle(surface, (20, 22, 28), (cx, cy), 7, 1)
        surface.blit(font_sm.render(f"actor {aid}", True, TXT),
                     (x + 34, yy + 5))
        tag = "cut-in" if on else "cruise"
        tc = DOT_ON if on else MUTED
        surface.blit(font_sm.render(tag, True, tc), (x + width - 58, yy + 5))
        yy += row_h
    return rect


# --------------------------------------------------------------------------- #
# Session UI (drive the ego; orchestrator casts + scripts the fleet)
# --------------------------------------------------------------------------- #
class CutinSession:
    def __init__(self, seed: int, n_actors: int = 4, headless: bool = False,
                 cutin_spec: Optional[dict] = None):
        self.seed = seed
        self.ego, self.asc = spawn_fleet(seed, n_actors=n_actors)
        self.orch = CutinOrchestrator(cutin_spec=cutin_spec,
                                      cruise_speed=CRUISE_SPEED)
        self.atime = 0.0
        self.clock_t = 0.0
        self._fcount = 0
        self.hit: Optional[str] = None
        self.headless = headless

        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        if not headless:
            pygame.display.set_caption(
                f"cut-in orchestrator — seed {seed}")
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_sm = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 18,
                                            bold=True)
        self.clock = pygame.time.Clock()

        # initial cast so the panel is populated before the first frame
        self.orch.cast(self.asc.actors, self.ego, self.asc.map.lane_width,
                       sticky=False)

    def integrate_ego(self, throttle: float, steer: float):
        e = self.ego
        if throttle > 0:
            a = A_THROTTLE * throttle
        elif throttle < 0:
            a = A_BRAKE * throttle
        else:
            a = -DRAG if e.v > 0 else 0.0
        e.v = max(0.0, min(V_MAX, e.v + a * DT))
        delta = DELTA_MAX * steer
        e.theta += (e.v / WHEELBASE) * math.tan(delta) * DT
        e.x += e.v * math.cos(e.theta) * DT
        e.y += e.v * math.sin(e.theta) * DT

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
            if _rects_overlap(eb, _rect(*pose, a.length, a.width)):
                return a.id
        return None

    def step(self, throttle: float, steer: float):
        self.integrate_ego(throttle, steer)
        self.atime += DT
        self.clock_t += DT
        self._fcount += 1
        if self._fcount % ORCH_EVERY == 0:
            self.asc, self.atime = self.orch.tick(
                self.asc, self.atime, self.ego, self.clock_t)
        if self.orch.flash > 0:
            self.orch.flash -= 1
        hit = self.real_collision()
        if hit and self.hit is None:
            self.hit = hit
            self.orch.flash = 20
            self.orch.msg = f"collision with actor {hit}"

    def _update_camera(self):
        look = 8.0
        CAM[0] = self.ego.x + look * math.cos(self.ego.theta)
        CAM[1] = self.ego.y + look * math.sin(self.ego.theta)

    def render(self):
        self._update_camera()
        s = self.screen
        s.fill(GRASS)
        mp.draw_map(s, self.asc.map, w2s, road=ROAD, line=LINE, edge=EDGE,
                    extend_y=40.0)
        for a, pose in self.actor_poses():
            ring = (self.orch.roles.get(a.id) == ROLE_CUTIN)
            self._draw_body(pose, a.length, a.width, tuple(a.color), a.id,
                            ring=ring)
        self._draw_body((self.ego.x, self.ego.y, math.degrees(self.ego.theta)),
                        4.5, 2.0, EGO_COL, "0", ring=True)
        # cut-in pin for the cast actor
        if self.orch.cutin_id and not self.orch.committed:
            sp = self.orch.spec
            px, py = se.live_cutin_pin(self.ego.x, self.ego.y, self.ego.theta,
                                       float(sp.get("along", 6.0)),
                                       float(sp.get("lat", 0.0)))
            self._draw_pin(px, py)
        mp.draw_rulers(s, w2s, self._s2w, pygame.Rect(0, TOP, SIZE, SIZE),
                       font=self.font_sm)
        self.orch.draw_panel(s, self.font_big, self.font_sm,
                             [a.id for a in self.asc.actors])
        self._draw_strips()
        pygame.display.flip()

    def _s2w(self, sx: float, sy: float):
        return ((sx - SIZE / 2) / SCALE + CAM[0],
                CAM[1] - (sy - TOP - SIZE / 2) / SCALE)

    def _draw_pin(self, x, y):
        p = w2s(x, y)
        pygame.draw.line(self.screen, (240, 160, 60), (p[0], p[1] - 14),
                         (p[0], p[1] + 4), 2)
        pygame.draw.circle(self.screen, (240, 160, 60), (p[0], p[1] - 14), 5)

    def _draw_body(self, pose, L, Wd, col, label, ring=False):
        pts = [w2s(*c) for c in _corners(*pose, L, Wd)]
        pygame.draw.polygon(self.screen, col, pts)
        pygame.draw.polygon(self.screen, (18, 18, 18), pts, 1)
        pygame.draw.line(self.screen, (250, 250, 250), pts[0], pts[1], 3)
        if ring:
            c = w2s(pose[0], pose[1])
            pygame.draw.circle(self.screen, (255, 255, 255), c, 5, 2)
        lp = w2s(pose[0], pose[1])
        t = self.font_sm.render(label, True, (10, 10, 10))
        self.screen.blit(t, (lp[0] - t.get_width() // 2, lp[1] - 7))

    def _draw_strips(self):
        s = self.screen
        pygame.draw.rect(s, BAR, (0, 0, W, TOP))
        s.blit(self.font_big.render("cut-in orchestrator", True, TXT), (10, 8))
        s.blit(self.font.render(f"seed {self.seed}   "
                                f"ego {self.ego.v:4.1f} m/s   "
                                f"t={self.clock_t:4.1f}s", True, TXT),
               (230, 14))
        if self.hit:
            badge = "HIT"
            col = (224, 82, 82)
        elif self.orch.outcome == "merged":
            badge = "MERGED"
            col = (88, 194, 106)
        elif self.orch.outcome == "abandoned":
            badge = "ABANDONED"
            col = (200, 160, 60)
        else:
            badge = "CASTING"
            col = (56, 178, 198)
        r = pygame.Rect(W - 130, 10, 110, 28)
        pygame.draw.rect(s, col, r, border_radius=4)
        s.blit(self.font.render(badge, True, (10, 10, 10)), (r.x + 12, r.y + 5))

        pygame.draw.rect(s, BAR, (0, H - BOT, W, BOT))
        col = (56, 178, 198) if self.orch.flash > 0 else MUTED
        s.blit(self.font_sm.render(self.orch.msg[:110], True, col),
               (10, H - BOT + 6))
        s.blit(self.font_sm.render(
            "↑/↓ throttle/brake   ←/→ steer   Esc quit", True, MUTED),
               (10, H - BOT + 26))

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
            self.render()
        pygame.quit()

    def run_headless(self, duration: float):
        steps = int(duration / DT)
        for _ in range(steps):
            cruise = CRUISE_SPEED
            throttle = 1.0 if self.ego.v < cruise else 0.0
            self.step(throttle, 0.0)
            self.render()
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
    # axis-aligned AABB of the oriented body — coarse but fine for hit flash
    pts = _corners(x, y, hd, L, Wd)
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def _rects_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main():
    ap = argparse.ArgumentParser(description="cut-in orchestrator (role casting)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--actors", type=int, default=4,
                    help="number of non-ego actors to spawn")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--deadline", type=float, default=4.0,
                    help="cut-in deadline t (seconds)")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    spec = dict(DEFAULT_CUTIN_SPEC)
    spec["t"] = float(args.deadline)
    sess = CutinSession(seed, n_actors=args.actors, headless=args.headless,
                        cutin_spec=spec)
    if args.headless:
        sess.run_headless(args.duration)
        print(f"seed={seed} actors={args.actors} "
              f"cutin={sess.orch.cutin_id} outcome={sess.orch.outcome} "
              f"hit={sess.hit} roles={sess.orch.roles}")
    else:
        sess.loop()


if __name__ == "__main__":
    main()
