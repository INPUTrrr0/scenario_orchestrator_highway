#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directives.py — v2 declarative directive layer (see v2/DESIGN.md).

Defines scenario families as modal-temporal directives grounded in a predicate
library, evaluates them on abstracted scenario states with their predicted
evolutions, and computes minimal *causal* interventions (retime/reroute) when a
directive fails.

Headless; depends on pyyaml only (and optionally on scenario_editor.py in the
same directory, for the --state <scenario file> adapter).

CLI:
  python3 directives.py --demo
  python3 directives.py --state FILE.yaml [--time T] [--signals N=green,E=red,...]
  add --ramp [A_MAX] for the C0-velocity (ramp) variant.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

INF = float("inf")

# --------------------------------------------------------------------------- #
# 0. Parameters


@dataclass
class Params:
    H: float = 15.0            # prediction horizon (s)
    v_max: float = 20.0        # max commandable speed (m/s)
    dt: float = 0.10           # temporal sampling step for G/F fallback (s)
    dt_d3: float = 0.50        # sampling step for D3's G (atom self-lookahead)
    ramp: Optional[float] = None  # a_max (m/s^2) for the C0 variant; None = jumps
    r_margin: float = 0.5      # conflict radius margin (m)
    pad: float = 0.20          # protected-window padding (s)
    g_min: float = 2.0         # min corridor gap before 'blocks' (m)
    w_v: float = 1.0           # cost per m/s of retime
    w_r: float = 5.0           # cost of reroute
    path_step: float = 0.25    # path polyline resolution (m)


# --------------------------------------------------------------------------- #
# 1. Geometry


def norm_ang(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def rot_pt(p: Tuple[float, float], deg: float) -> Tuple[float, float]:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def rect_corners(x, y, hdg, length, width):
    r = math.radians(hdg)
    c, s = math.cos(r), math.sin(r)
    hl, hw = length / 2.0, width / 2.0
    out = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        out.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return out


def rects_overlap(ra, rb) -> bool:
    """Separating-axis test for two convex quads."""
    for quad_a, quad_b in ((ra, rb), (rb, ra)):
        for i in range(4):
            ax, ay = quad_a[i]
            bx, by = quad_a[(i + 1) % 4]
            nx, ny = -(by - ay), (bx - ax)
            amin = min(px * nx + py * ny for px, py in quad_a)
            amax = max(px * nx + py * ny for px, py in quad_a)
            bmin = min(px * nx + py * ny for px, py in quad_b)
            bmax = max(px * nx + py * ny for px, py in quad_b)
            if amax < bmin or bmax < amin:
                return False
    return True


def _seg_x(p1, p2, q1, q2) -> Optional[Tuple[float, float]]:
    """Params (t, u) in [0,1]^2 where segments p and q cross, else None."""
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    qpx, qpy = q1[0] - p1[0], q1[1] - p1[1]
    t = (qpx * sy - qpy * sx) / den
    u = (qpx * ry - qpy * rx) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (min(max(t, 0.0), 1.0), min(max(u, 0.0), 1.0))
    return None


class Path:
    """Polyline path with arclength parametrization."""

    def __init__(self, pts: List[Tuple[float, float]]):
        self.pts = pts
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + math.dist(pts[i - 1], pts[i]))
        self.cum = cum
        self.length = cum[-1]
        self._near = None  # cached indices of segments near the intersection

    def _seg_at(self, s: float) -> Tuple[int, float]:
        if s <= 0:
            return 0, 0.0
        if s >= self.length:
            return len(self.pts) - 2, 1.0
        lo, hi = 0, len(self.cum) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        span = self.cum[hi] - self.cum[lo]
        f = (s - self.cum[lo]) / span if span > 0 else 0.0
        return lo, f

    def pos_at(self, s: float) -> Tuple[float, float]:
        i, f = self._seg_at(s)
        (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
        return (ax + f * (bx - ax), ay + f * (by - ay))

    def heading_at(self, s: float) -> float:
        i, _ = self._seg_at(min(max(s, 0.0), self.length - 1e-6))
        (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
        return math.degrees(math.atan2(by - ay, bx - ax))

    def project(self, p: Tuple[float, float]) -> Tuple[float, float]:
        """(s, lateral distance) of the closest point on the path."""
        px, py = p
        best_s, best_d2 = 0.0, INF
        for i in range(len(self.pts) - 1):
            ax, ay = self.pts[i]
            bx, by = self.pts[i + 1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 == 0 else min(max(((px - ax) * vx + (py - ay) * vy) / L2, 0.0), 1.0)
            cx, cy = ax + t * vx, ay + t * vy
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.cum[i] + t * math.sqrt(L2)
        return best_s, math.sqrt(best_d2)

    def near_idx(self, R: float = 14.0) -> List[int]:
        if self._near is None:
            self._near = [
                i for i in range(len(self.pts) - 1)
                if (abs(self.pts[i][0]) <= R and abs(self.pts[i][1]) <= R)
                or (abs(self.pts[i + 1][0]) <= R and abs(self.pts[i + 1][1]) <= R)
            ]
        return self._near

    def min_dist_to(self, p: Tuple[float, float]) -> Tuple[float, float]:
        """(s, distance) of closest approach to point p, near the intersection."""
        px, py = p
        best = (0.0, INF)
        for i in self.near_idx():
            ax, ay = self.pts[i]
            bx, by = self.pts[i + 1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 == 0 else min(max(((px - ax) * vx + (py - ay) * vy) / L2, 0.0), 1.0)
            cx, cy = ax + t * vx, ay + t * vy
            d = math.hypot(px - cx, py - cy)
            if d < best[1]:
                best = (self.cum[i] + t * math.sqrt(L2), d)
        return best


_CROSS_CACHE: Dict[Tuple[int, int], Optional[Tuple[float, float]]] = {}


def first_crossing(pa: Path, pb: Path) -> Optional[Tuple[float, float]]:
    """(s_a, s_b) of the first transversal crossing (in a's order), else None."""
    if pa is pb:
        return None
    key = (id(pa), id(pb))
    if key in _CROSS_CACHE:
        return _CROSS_CACHE[key]
    best = None
    for i in pa.near_idx():
        p1, p2 = pa.pts[i], pa.pts[i + 1]
        for j in pb.near_idx():
            hit = _seg_x(p1, p2, pb.pts[j], pb.pts[j + 1])
            if hit is not None:
                sa = pa.cum[i] + hit[0] * (pa.cum[i + 1] - pa.cum[i])
                sb = pb.cum[j] + hit[1] * (pb.cum[j + 1] - pb.cum[j])
                if best is None or sa < best[0]:
                    best = (sa, sb)
    _CROSS_CACHE[key] = best
    return best


# --------------------------------------------------------------------------- #
# 2. Map, legs, routes


IN_LEGS = {"SE": 0.0, "EN": 90.0, "NW": 180.0, "WS": 270.0}  # rotation from canonical SE
TURNS = ("straight", "right", "left")
ROUTES = {  # (inbound leg, turn) -> outbound leg (right-hand traffic)
    "SE": {"straight": "NE", "right": "ES", "left": "WN"},
    "EN": {"straight": "WN", "right": "NE", "left": "SW"},
    "NW": {"straight": "SW", "right": "WN", "left": "ES"},
    "WS": {"straight": "ES", "right": "SW", "left": "NE"},
}
# leg: (heading, lane axis, lane sign, along axis, along sign, inbound)
LEG_DEFS = {
    "SE": (90.0, "x", +1, "y", -1, True),
    "EN": (180.0, "y", +1, "x", +1, True),
    "NW": (270.0, "x", -1, "y", +1, True),
    "WS": (0.0, "y", -1, "x", -1, True),
    "NE": (90.0, "x", +1, "y", +1, False),
    "ES": (0.0, "y", -1, "x", +1, False),
    "WN": (180.0, "y", +1, "x", -1, False),
    "SW": (270.0, "x", -1, "y", -1, False),
}
OPP_ARM = {"N": "S", "S": "N", "E": "W", "W": "E"}


def arm_of(leg: str) -> str:
    return leg[0]


def _line(a, b, step):
    n = max(2, int(math.ceil(math.dist(a, b) / step)) + 1)
    return [(a[0] + (b[0] - a[0]) * k / (n - 1), a[1] + (b[1] - a[1]) * k / (n - 1))
            for k in range(n)]


def _arc(center, r, a0, a1, step):
    n = max(3, int(math.ceil(r * abs(math.radians(a1 - a0)) / step)) + 1)
    out = []
    for k in range(n):
        a = math.radians(a0 + (a1 - a0) * k / (n - 1))
        out.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
    return out


_PATH_CACHE: Dict[tuple, Path] = {}


@dataclass(frozen=True)
class MapCfg:
    lane_width: float = 3.5
    arm_length: float = 60.0


def route_path(mc: MapCfg, leg_in: str, turn: str, prm: Params) -> Path:
    """Full lane-following path: approach -> intersection (fillet) -> outbound."""
    key = (mc.lane_width, mc.arm_length, leg_in, turn)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    w, step = mc.lane_width, prm.path_step
    h, b, L = w / 2.0, w, mc.arm_length + w
    pts = _line((h, -L), (h, -b), step)                       # canonical approach (SE)
    if turn == "straight":
        pts += _line((h, -b), (h, L), step)[1:]
    elif turn == "right":                                     # tangent fillet, r = w/2
        pts += _arc((b, -b), h, 180.0, 90.0, step)[1:]
        pts += _line((b, -h), (L, -h), step)[1:]
    else:                                                     # left: tangent fillet, r = 3w/2
        pts += _arc((-b, -b), b + h, 0.0, 90.0, step)[1:]
        pts += _line((-b, h), (-L, h), step)[1:]
    ang = IN_LEGS[leg_in]
    p = Path([rot_pt(q, ang) for q in pts])
    _PATH_CACHE[key] = p
    return p


def ray_path(x, y, hdg, R=140.0) -> Path:
    r = math.radians(hdg)
    return Path([(x, y), (x + R * math.cos(r), y + R * math.sin(r))])


def stopline_s(mc: MapCfg, path: Path, leg_in: str) -> float:
    """Arclength of the stop line (box entry) on a route path."""
    entry = rot_pt((mc.lane_width / 2.0, -mc.lane_width), IN_LEGS[leg_in])
    return path.project(entry)[0]


# --------------------------------------------------------------------------- #
# 3. Concrete state


@dataclass
class ActorState:
    id: str
    x: float
    y: float
    heading: float
    speed: float
    length: float = 4.5
    width: float = 2.0


@dataclass
class State:
    map: MapCfg
    actors: List[ActorState]
    ego: str = "0"
    signals: Optional[Dict[str, str]] = None  # arm -> 'red'|'green'; None -> assumed
    label: str = ""


# --------------------------------------------------------------------------- #
# 4. Recognition machinery (concrete -> abstract)


@dataclass
class AbsActor:
    id: str
    x: float
    y: float
    heading: float
    speed: float
    length: float
    width: float
    region: str            # 'approach' | 'intersection' | 'exit' | 'offmap'
    leg: Optional[str]     # inbound leg (approach/intersection) or outbound leg (exit)
    turn: Optional[str]    # recognized route turn (None when not on a route)
    path: Path             # predicted path (route path, or ray fallback)
    s0: float              # current arclength along path
    committed: bool        # past the decision point (stop line)
    conf: float            # recognition confidence in (0, 1]


@dataclass
class AbsState:
    map: MapCfg
    prm: Params
    ego: str
    actors: Dict[str, AbsActor]
    signals: Dict[str, str]
    signals_assumed: bool
    label: str = ""

    def phase(self, arm: str) -> str:
        return self.signals.get(arm, "green")


def _match_leg(x, y, hdg, leg, mc) -> Optional[Tuple[float, float]]:
    """(lateral offset, heading error) if the pose matches the leg, else None."""
    heading, lax, lsg, aax, asg, _in = LEG_DEFS[leg]
    h, b = mc.lane_width / 2.0, mc.lane_width
    lane_c = x if lax == "x" else y
    along = x if aax == "x" else y
    lat = abs(lane_c - lsg * h)
    dh = abs(norm_ang(hdg - heading))
    if lat <= 1.6 and dh <= 35.0 and along * asg > b - 0.1:
        return lat, dh
    return None


def recognize(state: State, prm: Params) -> AbsState:
    mc = state.map
    b = mc.lane_width
    actors: Dict[str, AbsActor] = {}
    for a in state.actors:
        region, leg, turn, path, conf = "offmap", None, None, None, 0.2
        inside = abs(a.x) <= b + 0.1 and abs(a.y) <= b + 0.1
        if inside:
            # fit all routes; best residual wins
            best = None
            for leg_in in IN_LEGS:
                for t in TURNS:
                    p = route_path(mc, leg_in, t, prm)
                    s, lat = p.project((a.x, a.y))
                    dh = abs(norm_ang(a.heading - p.heading_at(s)))
                    if lat <= 1.8 and dh <= 50.0:
                        score = lat + dh / 30.0
                        if best is None or score < best[0]:
                            best = (score, leg_in, t, p, s, lat, dh)
                    del s, lat, dh
            if best is not None:
                _, leg, turn, path, s0, lat, dh = best
                region = "intersection"
                conf = max(0.2, (1 - lat / 2.0) * (1 - dh / 60.0))
        else:
            best = None
            for lg in LEG_DEFS:
                m = _match_leg(a.x, a.y, a.heading, lg, mc)
                if m is not None and (best is None or m[0] < best[1][0]):
                    best = (lg, m)
            if best is not None:
                leg, (lat, dh) = best
                conf = max(0.2, (1 - lat / 2.0) * (1 - dh / 45.0))
                if LEG_DEFS[leg][5]:
                    region, turn = "approach", "straight"   # default route hypothesis
                    path = route_path(mc, leg, turn, prm)
                    conf *= 0.9  # route is a default, not observed
                else:
                    region = "exit"
        if path is None:
            path = ray_path(a.x, a.y, a.heading)
            s0 = 0.0
        elif region != "intersection":
            s0, _ = path.project((a.x, a.y))
        committed = region != "approach"
        actors[a.id] = AbsActor(a.id, a.x, a.y, a.heading, a.speed, a.length,
                                a.width, region, leg, turn, path, s0, committed,
                                round(conf, 3))
    # signals: given, or assumed family-consistently (ego arm + opposite green)
    if state.signals:
        signals, assumed = dict(state.signals), False
    else:
        ego = actors.get(state.ego)
        ego_arm = arm_of(ego.leg) if ego and ego.leg else "S"
        signals = {arm: "red" for arm in "NESW"}
        signals[ego_arm] = "green"
        signals[OPP_ARM[ego_arm]] = "green"
        assumed = True
    return AbsState(mc, prm, state.ego, actors, signals, assumed, state.label)


def route_options(aa: AbsActor) -> List[str]:
    if aa.leg is None or aa.leg not in IN_LEGS:
        return [aa.turn] if aa.turn else []
    if aa.committed:
        return [aa.turn]
    rest = [t for t in TURNS if t != aa.turn]
    return [aa.turn] + rest


# --------------------------------------------------------------------------- #
# 5. Evolution model (speed profiles, branch set)


class Profile:
    """Piecewise speed profile: pieces of (t0, s0, v0, a)."""

    def __init__(self, pieces):
        self.pieces = pieces

    @staticmethod
    def const(v: float) -> "Profile":
        return Profile([(0.0, 0.0, max(v, 0.0), 0.0)])

    @staticmethod
    def retime(v0: float, v1: float, a_max: Optional[float]) -> "Profile":
        """Adopt target v1 from t=0: jump (a_max None) or one ramp then plateau."""
        v0, v1 = max(v0, 0.0), max(v1, 0.0)
        if a_max is None or abs(v1 - v0) < 1e-9:
            return Profile([(0.0, 0.0, v1, 0.0)])
        a = a_max if v1 > v0 else -a_max
        T = (v1 - v0) / a
        return Profile([(0.0, 0.0, v0, a), (T, (v0 + v1) / 2.0 * T, v1, 0.0)])

    def target_v(self) -> float:
        return self.pieces[-1][2]

    def _piece_end(self, k: int) -> float:
        return self.pieces[k + 1][0] if k + 1 < len(self.pieces) else INF

    def v_at(self, t: float) -> float:
        for k in range(len(self.pieces) - 1, -1, -1):
            t0, s0, v0, a = self.pieces[k]
            if t >= t0 - 1e-12:
                return max(v0 + a * (t - t0), 0.0)
        return self.pieces[0][2]

    def s_at(self, t: float) -> float:
        for k in range(len(self.pieces) - 1, -1, -1):
            t0, s0, v0, a = self.pieces[k]
            if t >= t0 - 1e-12:
                tau = t - t0
                if a < 0:  # clamp at standstill
                    tau = min(tau, v0 / -a) if v0 > 0 else 0.0
                return s0 + v0 * tau + 0.5 * a * tau * tau
        return 0.0

    def arrival(self, d: float) -> Optional[float]:
        """First t with s(t) >= d; None if never reached."""
        if d <= 0:
            return 0.0
        for k, (t0, s0, v0, a) in enumerate(self.pieces):
            t_end = self._piece_end(k)
            dd = d - s0
            if a == 0:
                if v0 <= 1e-9:
                    continue
                tau = dd / v0
            else:
                disc = v0 * v0 + 2 * a * dd
                if disc < 0:
                    continue
                r = math.sqrt(disc)
                cands = [(-v0 + r) / a, (-v0 - r) / a]
                cands = [c for c in cands if c >= -1e-9]
                if a < 0 and v0 > 0:  # only before standstill
                    cands = [c for c in cands if c <= v0 / -a + 1e-9]
                if not cands:
                    continue
                tau = min(cands)
            t_hit = t0 + tau
            if t_hit <= t_end + 1e-9:
                return max(t_hit, 0.0)
        return None


class Evolution:
    """An evolution: per actor a (path, s0, profile). Controls overlay applies
    causal interventions (retime/reroute) at t=0."""

    def __init__(self, astate: AbsState, prm: Params, controls: Optional[dict] = None):
        self.astate, self.prm = astate, prm
        self.items: Dict[str, Tuple[Path, float, Profile]] = {}
        controls = controls or {}
        for aid, aa in astate.actors.items():
            path, s0 = aa.path, aa.s0
            c = controls.get(aid, {})
            if "turn" in c and not aa.committed and aa.leg in IN_LEGS:
                path = route_path(astate.map, aa.leg, c["turn"], prm)
                s0 = path.project((aa.x, aa.y))[0]
            prof = Profile.const(aa.speed)
            if "speed" in c:
                prof = Profile.retime(aa.speed, c["speed"], prm.ramp)
            self.items[aid] = (path, s0, prof)

    def s_at(self, aid: str, t: float) -> float:
        path, s0, prof = self.items[aid]
        return s0 + prof.s_at(t)

    def pose_at(self, aid: str, t: float) -> Tuple[float, float, float]:
        path, s0, prof = self.items[aid]
        s = min(s0 + prof.s_at(t), path.length)
        x, y = path.pos_at(s)
        return x, y, path.heading_at(s)

    def conflict(self, a: str, b: str) -> Optional[Tuple[float, float]]:
        """(d_a, d_b): remaining distances (from t=0) to the paths' crossing."""
        pa, s0a, _ = self.items[a]
        pb, s0b, _ = self.items[b]
        cr = first_crossing(pa, pb)
        if cr is None:
            return None
        return cr[0] - s0a, cr[1] - s0b


def conflict_radius(a: AbsActor, b: AbsActor, prm: Params) -> float:
    return (a.length + b.width) / 2.0 + prm.r_margin


def occupancy_window(prof: Profile, d: float, r: float) -> Optional[Tuple[float, float]]:
    """Global-time interval during which the actor is within +-r of a point at
    remaining distance d along its path. None if already past or never reached."""
    if d < -r:
        return None
    t0 = 0.0 if d - r <= 0 else prof.arrival(d - r)
    if t0 is None:
        return None
    t1 = prof.arrival(d + r)
    return (t0, t1 if t1 is not None else INF)


def _ivx(w1, w2) -> Optional[Tuple[float, float]]:
    if w1 is None or w2 is None:
        return None
    lo, hi = max(w1[0], w2[0]), min(w1[1], w2[1])
    return (lo, hi) if lo <= hi else None


def collision_overlap(astate, evo, a: str, b: str) -> Optional[Tuple[float, float]]:
    """Closed form: global-time window during which a and b co-occupy their
    conflict point (piecewise-constant velocities). None -> no predicted collision."""
    A, B = astate.actors[a], astate.actors[b]
    cf = evo.conflict(a, b)
    if cf is None:
        return None
    da, db = cf
    wa = occupancy_window(evo.items[a][2], da, conflict_radius(A, B, astate.prm))
    wb = occupancy_window(evo.items[b][2], db, conflict_radius(B, A, astate.prm))
    return _ivx(wa, wb)


def collidable(ctx, hero: str) -> Tuple[bool, str]:
    """Dia F collide(hero, ego) at ctx.t: exists an admissible hero control
    (route x target speed, adopted at ctx.t) whose arrival at a conflict point
    with ego falls inside ego's occupancy window. Closed form (DESIGN 5, D2)."""
    prm, astate, evo, t = ctx.prm, ctx.astate, ctx.evo, ctx.t
    if hero is None or hero not in astate.actors:
        return False, "no hero candidate"
    hh = astate.actors[hero]
    ego = astate.actors[astate.ego]
    v_now = evo.items[hero][2].v_at(t)
    for turn in route_options(hh):
        if turn is None:
            continue
        if hh.leg in IN_LEGS and turn != hh.turn:
            bp = route_path(astate.map, hh.leg, turn, prm)
            bs0 = bp.project((hh.x, hh.y))[0]
        else:
            bp, bs0, _ = evo.items[hero]
        cr = first_crossing(bp, evo.items[astate.ego][0])
        if cr is None:
            continue
        r_h = conflict_radius(hh, ego, prm)
        r_e = conflict_radius(ego, hh, prm)
        d_rem = (cr[0] - bs0) - evo.items[hero][2].s_at(t)
        if d_rem < -r_h:
            continue  # hero already past this conflict point
        d_ego = cr[1] - evo.items[astate.ego][1]
        we = occupancy_window(evo.items[astate.ego][2], d_ego, r_e)
        if we is None or we[1] <= t:
            continue  # ego window over (or ego never arrives)
        if prm.ramp is None:
            lo = t + max(d_rem - r_h, 0.0) / prm.v_max
            hi = INF
        else:
            fast = Profile.retime(v_now, prm.v_max, prm.ramp)
            tf = fast.arrival(max(d_rem - r_h, 0.0))
            lo = t + (tf if tf is not None else INF)
            stop_d = v_now * v_now / (2.0 * prm.ramp)
            if stop_d <= max(d_rem - r_h, 0.0):
                hi = INF  # can stop short of the conflict: delay arbitrarily
            else:
                brake = Profile.retime(v_now, 0.0, prm.ramp)
                tb = brake.arrival(d_rem + r_h)
                hi = t + (tb if tb is not None else INF)
        if _ivx((lo, hi), (max(we[0], t), we[1])) is not None:
            return True, f"route {turn}"
    return False, "no admissible control reaches ego's window"


# --------------------------------------------------------------------------- #
# 6. Predicate library (grounding)


@dataclass
class Ev:
    value: bool
    witness: dict = field(default_factory=dict)
    t_fail: Optional[float] = None
    expl: str = ""
    conf: float = 1.0
    info: dict = field(default_factory=dict)


@dataclass
class Ctx:
    astate: AbsState
    evo: Evolution
    t: float
    prm: Params
    bind: dict
    tstar: Optional[float] = None
    protected: Optional[Tuple[float, float]] = None

    def at(self, t: float) -> "Ctx":
        return replace(self, t=t)

    def resolve(self, ref: str) -> Optional[str]:
        return self.bind.get(ref, ref if ref in self.astate.actors else None)

    def horizon(self, le) -> float:
        if le == "T*":
            return self.tstar if self.tstar is not None else self.prm.H
        if le == "H":
            return self.prm.H
        return float(le)


def p_collide(ctx: Ctx, a: str, b: str) -> Ev:
    ov = collision_overlap(ctx.astate, ctx.evo, a, b)
    conf = ctx.astate.actors[a].conf * ctx.astate.actors[b].conf
    if ov and ov[0] <= ctx.t <= ov[1]:
        return Ev(True, expl=f"co-occupy conflict point during [{ov[0]:.2f},{ov[1]:.2f}]",
                  conf=conf, info={"window": ov})
    return Ev(False, expl="no co-occupancy now", conf=conf,
              info={"window": ov} if ov else {})


def p_runs_red(ctx: Ctx, v: str) -> Ev:
    aa = ctx.astate.actors[v]
    if aa.leg is None or aa.leg not in IN_LEGS and aa.region != "intersection":
        return Ev(False, expl="not on an approach", conf=aa.conf)
    arm = arm_of(aa.leg) if aa.leg else None
    if arm is None or ctx.astate.phase(arm) != "red":
        return Ev(False, expl=f"approach {arm} not red", conf=aa.conf)
    if aa.region == "intersection":
        return Ev(True, expl=f"in intersection, entered on red ({arm})", conf=aa.conf)
    if aa.region != "approach":
        return Ev(False, expl="already exited", conf=aa.conf)
    sl = stopline_s(ctx.astate.map, aa.path, aa.leg)
    d = sl - aa.s0 - ctx.evo.items[v][2].s_at(ctx.t)
    if d <= 0:
        return Ev(True, expl=f"crossed stop line on red ({arm})", conf=aa.conf)
    ta = ctx.evo.items[v][2].arrival(sl - aa.s0)
    if ta is not None and ta <= ctx.prm.H:
        return Ev(True, expl=f"will cross stop line on red ({arm}) at t={ta:.2f}",
                  conf=aa.conf)
    return Ev(False, expl="never crosses stop line within horizon", conf=aa.conf)


def _first_collision_within(ctx: Ctx, a: str, b: str, t_hi: float) -> Optional[float]:
    ov = collision_overlap(ctx.astate, ctx.evo, a, b)
    if ov is None:
        return None
    lo = max(ov[0], ctx.t)
    return lo if lo <= min(ov[1], t_hi) else None


def p_blocks(ctx: Ctx, w: str, hero: str) -> Ev:
    """w sits in hero's corridor ahead and the gap closes below g_min before t*."""
    astate, evo, prm = ctx.astate, ctx.evo, ctx.prm
    ww, hh = astate.actors[w], astate.actors[hero]
    hpath, hs0, hprof = evo.items[hero]
    s_w, lat = hpath.project((ww.x, ww.y))
    if lat > 1.4 or s_w <= hs0:
        return Ev(False, expl="not ahead in hero's corridor")
    t_hi = ctx.tstar if ctx.tstar is not None else prm.H
    wprof = evo.items[w][2]
    body = (ww.length + hh.length) / 2.0
    t = ctx.t
    while t <= t_hi + 1e-9:
        gap = (s_w + wprof.s_at(t)) - (hs0 + hprof.s_at(t)) - body
        if gap < prm.g_min:
            return Ev(True, t_fail=t, expl=f"corridor gap {gap:.1f} m < {prm.g_min} m at t={t:.2f}")
        t += 0.1
    return Ev(False, expl="gap stays open")


def p_occupies_conflict(ctx: Ctx, w: str, hero: str, ego: str) -> Ev:
    if ctx.protected is None:
        return Ev(False, expl="no planned collision window")
    astate, evo, prm = ctx.astate, ctx.evo, ctx.prm
    hpath = evo.items[hero][0]
    cr = first_crossing(hpath, evo.items[ego][0])
    if cr is None:
        return Ev(False, expl="no conflict point")
    P = hpath.pos_at(cr[0])
    wa = astate.actors[w]
    r_w = (wa.length + max(astate.actors[hero].width, astate.actors[ego].width)) / 2.0 \
        + prm.r_margin
    s_c, dist = evo.items[w][0].min_dist_to(P)
    if dist > r_w:
        return Ev(False, expl="path clears the conflict point")
    d_w = s_c - evo.items[w][1]
    win = occupancy_window(evo.items[w][2], d_w, r_w)
    c0, c1 = ctx.protected
    hit = _ivx(win, (c0 - prm.pad, c1 + prm.pad))
    if hit:
        return Ev(True, expl=f"occupies conflict point during [{hit[0]:.2f},{hit[1]:.2f}]"
                             f" inside protected window", t_fail=hit[0])
    return Ev(False, expl="clear of protected window")


def p_interferes(ctx: Ctx, w: str, hero: str, ego: str) -> Ev:
    t_hi = ctx.tstar if ctx.tstar is not None else ctx.prm.H
    conf = ctx.astate.actors[w].conf
    tc = _first_collision_within(ctx, w, ego, t_hi)
    if tc is not None:
        return Ev(True, t_fail=tc, conf=conf,
                  expl=f"would collide with ego at t={tc:.2f} (< t*)")
    tc = _first_collision_within(ctx, w, hero, t_hi)
    if tc is not None:
        return Ev(True, t_fail=tc, conf=conf,
                  expl=f"would collide with hero at t={tc:.2f} (< t*)")
    bl = p_blocks(ctx, w, hero)
    if bl.value:
        return Ev(True, t_fail=bl.t_fail, expl="blocks hero: " + bl.expl, conf=conf)
    oc = p_occupies_conflict(ctx, w, hero, ego)
    if oc.value:
        return Ev(True, t_fail=oc.t_fail, expl=oc.expl, conf=conf)
    return Ev(False, expl="no interference", conf=conf)


PREDICATES = {
    "collide": p_collide,
    "runs_red": p_runs_red,
    "interferes": p_interferes,
    "blocks": p_blocks,
    "occupies_conflict": p_occupies_conflict,
}


# --------------------------------------------------------------------------- #
# 7. Formal language (modal-temporal DSL)


class Formula:
    def eval(self, ctx: Ctx) -> Ev:
        raise NotImplementedError

    def __invert__(self):
        return Not(self)

    def __and__(self, o):
        return And(self, o)


@dataclass
class Atom(Formula):
    name: str
    args: tuple

    def __init__(self, name, *args):
        if name not in PREDICATES:
            raise ValueError(f"unknown predicate: {name}")
        self.name, self.args = name, args

    def eval(self, ctx: Ctx) -> Ev:
        ids = []
        for a in self.args:
            r = ctx.resolve(a)
            if r is None:
                return Ev(False, expl=f"unbound role '{a}'")
            ids.append(r)
        return PREDICATES[self.name](ctx, *ids)

    def __repr__(self):
        return f"{self.name}({', '.join(self.args)})"


class Not(Formula):
    def __init__(self, child):
        self.child = child

    def eval(self, ctx):
        e = self.child.eval(ctx)
        return Ev(not e.value, e.witness, e.t_fail, e.expl, e.conf, e.info)

    def __repr__(self):
        return f"¬{self.child!r}"


class And(Formula):
    def __init__(self, *children):
        self.children = children

    def eval(self, ctx):
        conf, info = 1.0, {}
        for c in self.children:
            e = c.eval(ctx)
            conf = min(conf, e.conf)
            if not e.value:
                return Ev(False, e.witness, e.t_fail, e.expl, conf, e.info)
            info.update(e.info)
        return Ev(True, {}, None, "all conjuncts hold", conf, info)

    def __repr__(self):
        return " ∧ ".join(repr(c) for c in self.children)


class F(Formula):
    """Eventually within `le` seconds, along the current evolution."""

    def __init__(self, child, le="H"):
        self.child, self.le = child, le

    def eval(self, ctx):
        hor = ctx.horizon(self.le)
        if isinstance(self.child, Atom) and self.child.name == "collide":
            a = ctx.resolve(self.child.args[0])
            b = ctx.resolve(self.child.args[1])
            if a is None or b is None:
                return Ev(False, expl="unbound role")
            ov = collision_overlap(ctx.astate, ctx.evo, a, b)
            conf = ctx.astate.actors[a].conf * ctx.astate.actors[b].conf
            hit = _ivx(ov, (ctx.t, ctx.t + hor)) if ov else None
            if hit:
                return Ev(True, conf=conf,
                          expl=f"collision during [{hit[0]:.2f},{hit[1]:.2f}] s",
                          info={"window": hit, "t_star": (hit[0] + hit[1]) / 2.0})
            return Ev(False, conf=conf, expl="occupancy windows never overlap")
        t = ctx.t
        while t <= ctx.t + hor + 1e-9:
            e = self.child.eval(ctx.at(t))
            if e.value:
                return Ev(True, e.witness, None, f"holds at t={t:.2f}: " + e.expl,
                          e.conf, e.info)
            t += ctx.prm.dt
        return Ev(False, expl="never holds within horizon")

    def __repr__(self):
        return f"F[≤{self.le}] {self.child!r}"


class G(Formula):
    """Always within `le` seconds, along the current evolution."""

    def __init__(self, child, le="T*", dt=None):
        self.child, self.le, self.dt = child, le, dt

    def eval(self, ctx):
        hor = ctx.horizon(self.le)
        dt = self.dt or ctx.prm.dt
        t, conf = ctx.t, 1.0
        while t <= ctx.t + hor + 1e-9:
            e = self.child.eval(ctx.at(t))
            conf = min(conf, e.conf)
            if not e.value:
                return Ev(False, e.witness, t, f"fails at t={t:.2f}: " + e.expl,
                          conf, e.info)
            t += dt
        return Ev(True, {}, None, f"holds throughout [{ctx.t:.2f},{ctx.t + hor:.2f}]",
                  conf)

    def __repr__(self):
        return f"G[≤{self.le}] {self.child!r}"


class Dia(Formula):
    """Possibly: over the admissible branch set of the hero (route x speed)."""

    def __init__(self, child):
        self.child = child

    def eval(self, ctx):
        hero = ctx.bind.get("hero")
        # analytic when child = F collide(hero, ego)  (the D2 pattern)
        if (isinstance(self.child, F) and isinstance(self.child.child, Atom)
                and self.child.child.name == "collide"):
            ok, why = collidable(ctx, hero)
            conf = ctx.astate.actors[hero].conf if hero in (ctx.astate.actors or {}) else 1.0
            return Ev(ok, expl=("reachable: " + why) if ok else why,
                      conf=conf, t_fail=None if ok else ctx.t)
        # generic fallback: finite grid over hero controls
        if hero is None or hero not in ctx.astate.actors:
            return Ev(False, expl="no hero candidate")
        aa = ctx.astate.actors[hero]
        base = getattr(ctx.evo, "_controls", {})
        for turn in route_options(aa):
            for k in range(0, 11):
                v = ctx.prm.v_max * k / 10.0
                trial = dict(base)
                trial[hero] = {"speed": v, "turn": turn}
                evo2 = Evolution(ctx.astate, ctx.prm, trial)
                e = self.child.eval(replace(ctx, evo=evo2))
                if e.value:
                    return Ev(True, expl=f"witness control: v'={v:.1f}, route {turn}",
                              conf=e.conf)
        return Ev(False, expl="no branch satisfies", t_fail=ctx.t)

    def __repr__(self):
        return f"◇ {self.child!r}"


class Exists(Formula):
    """Existential over concrete vehicles (excluding bound roles)."""

    def __init__(self, var, child, exclude=("ego",)):
        self.var, self.child, self.exclude = var, child, exclude

    def eval(self, ctx):
        excl = {ctx.bind.get(r) for r in self.exclude if ctx.bind.get(r)}
        reasons = []
        for aid in sorted(ctx.astate.actors):
            if aid in excl:
                continue
            ctx2 = replace(ctx, bind={**ctx.bind, self.var: aid})
            e = self.child.eval(ctx2)
            if e.value:
                return Ev(True, {self.var: aid, **e.witness}, e.t_fail,
                          f"{self.var}={aid}: " + e.expl, e.conf, e.info)
            reasons.append(f"{aid}: {e.expl}")
        return Ev(False, expl="; ".join(reasons) if reasons else "no vehicles")

    def __repr__(self):
        return f"∃{self.var} {self.child!r}"


# --------------------------------------------------------------------------- #
# 8. The red-light-violation family


D1 = Exists("v", And(Atom("runs_red", "v"), F(Atom("collide", "v", "ego"), le="H")))
D2 = G(Dia(F(Atom("collide", "hero", "ego"), le="H")), le="T*")
D3 = G(Not(Exists("w", Atom("interferes", "w", "hero", "ego"),
               exclude=("ego", "hero"))), le="T*", dt=0.5)

DIRECTIVES = [
    ("D1", "some vehicle shall run the red and collide with the ego", D1),
    ("D2", "at all times, some vehicle could still collide with the ego", D2),
    ("D3", "other vehicles would not interfere with the scenario", D3),
]


@dataclass
class FamilyResult:
    d1: Ev
    d2: Ev
    d3: Ev
    hero: Optional[str]
    t_star: Optional[float]
    protected: Optional[Tuple[float, float]]
    evo: Evolution

    @property
    def ok(self) -> bool:
        return self.d1.value and self.d2.value and self.d3.value


def d1_repair_options(astate: AbsState, prm: Params, controls: dict) -> list:
    """Causal repairs making D1 true: (cost, aid, turn, v_target, expl)."""
    evo = Evolution(astate, prm, controls)
    ego = astate.actors[astate.ego]
    out = []
    for aid in sorted(astate.actors):
        if aid == astate.ego:
            continue
        aa = astate.actors[aid]
        if aa.leg is None or arm_of(aa.leg) not in "NESW" \
                or astate.phase(arm_of(aa.leg)) != "red" or aa.region == "exit":
            continue
        v_now = evo.items[aid][2].v_at(0.0)
        for turn in route_options(aa):
            if turn is None:
                continue
            if turn != aa.turn:
                bp = route_path(astate.map, aa.leg, turn, prm)
                bs0 = bp.project((aa.x, aa.y))[0]
            else:
                bp, bs0, _ = evo.items[aid]
            cr = first_crossing(bp, evo.items[astate.ego][0])
            if cr is None:
                continue
            d = cr[0] - bs0
            r_h = conflict_radius(aa, ego, prm)
            if d < -r_h:
                continue
            we = occupancy_window(evo.items[astate.ego][2],
                                  cr[1] - evo.items[astate.ego][1],
                                  conflict_radius(ego, aa, prm))
            if we is None or we[1] <= 0:
                continue
            t_mid = (max(we[0], 0.0) + we[1]) / 2.0
            if t_mid <= 1e-6:
                continue
            if prm.ramp is None:
                v_cands = [d / t_mid] if d > 0 else [v_now]
            else:
                v_t = _bisect_target(v_now, d, t_mid, prm)
                # midpoint unreachable under the ramp: clamp to the nearest
                # achievable arrival (floor it / brake) and verify the overlap
                v_cands = [v_t] if v_t is not None else [prm.v_max, 0.0]
            for v_t in v_cands:
                if v_t is None or not (0.0 <= v_t <= prm.v_max):
                    continue
                trial = {**controls, aid: {**controls.get(aid, {}), "speed": v_t,
                                           **({"turn": turn} if turn != aa.turn else {})}}
                ov = collision_overlap(astate, Evolution(astate, prm, trial),
                                       aid, astate.ego)
                if ov is None:
                    continue
                t_hit = (max(ov[0], 0.0) + ov[1]) / 2.0
                cost = prm.w_v * abs(v_t - v_now) + (prm.w_r if turn != aa.turn else 0.0)
                out.append((cost, aid, turn, v_t,
                            f"retime {aid} -> {v_t:.2f} m/s"
                            + (f" + reroute {turn}" if turn != aa.turn else "")
                            + f" (arrive conflict at t={t_hit:.2f})"))
                break
    out.sort(key=lambda o: o[0])
    return out


def _bisect_target(v_now, d, t_mid, prm) -> Optional[float]:
    """Ramp variant: target speed whose ramped arrival at d equals t_mid."""

    def arr(v):
        a = Profile.retime(v_now, v, prm.ramp).arrival(d)
        return a if a is not None else INF

    lo, hi = 0.0, prm.v_max
    if arr(hi) > t_mid:      # even flooring it arrives too late
        return None
    if arr(1e-3) < t_mid:    # even crawling arrives too early (nearly on top)
        return 1e-3
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if arr(mid) > t_mid:
            lo = mid
        else:
            hi = mid
    return hi


def best_candidate(astate: AbsState, prm: Params, controls: dict):
    opts = d1_repair_options(astate, prm, controls)
    return opts[0][1] if opts else None


def evaluate_family(astate: AbsState, prm: Params,
                    controls: Optional[dict] = None) -> FamilyResult:
    controls = controls or {}
    evo = Evolution(astate, prm, controls)
    evo._controls = controls
    ctx = Ctx(astate, evo, 0.0, prm, {"ego": astate.ego})
    d1 = D1.eval(ctx)
    if d1.value:
        hero = d1.witness.get("v")
        t_star = d1.info.get("t_star", prm.H)
        protected = d1.info.get("window")
    else:
        hero = best_candidate(astate, prm, controls)
        t_star, protected = prm.H, None
    ctx2 = replace(ctx, bind={**ctx.bind, "hero": hero} if hero else ctx.bind,
                   tstar=t_star, protected=protected)
    d2 = D2.eval(ctx2) if hero else Ev(False, expl="no hero candidate exists")
    d3 = D3.eval(ctx2)
    return FamilyResult(d1, d2, d3, hero, t_star, protected, evo)


# --------------------------------------------------------------------------- #
# 9. Minimal causal intervention


@dataclass
class Intervention:
    kind: str        # 'retime' | 'reroute'
    actor: str
    value: object
    cost: float
    why: str

    def __str__(self):
        v = f"{self.value:.2f} m/s" if self.kind == "retime" else str(self.value)
        return f"{self.kind}({self.actor} -> {v})  cost {self.cost:.2f}   [{self.why}]"


@dataclass
class RepairResult:
    feasible: bool
    interventions: List[Intervention]
    cost: float
    final: FamilyResult
    reason: str = ""


def _body_sweep_clear(astate, prm, controls, w, others, t_hi) -> bool:
    """Dense oriented-body check: w touches none of `others` before t_hi.
    Catches near-miss geometry the centerline-crossing abstraction can't see
    (e.g. corner clips between neighbouring turn fillets)."""
    evo = Evolution(astate, prm, controls)
    wa = astate.actors[w]
    t = 0.0
    while t <= t_hi + 1e-9:
        rw = rect_corners(*evo.pose_at(w, t), wa.length, wa.width)
        for o in others:
            oa = astate.actors[o]
            if rects_overlap(rw, rect_corners(*evo.pose_at(o, t),
                                              oa.length, oa.width)):
                return False
        t += 0.05
    return True


def _fix_interferer(astate, prm, controls, w, res) -> Optional[List[Intervention]]:
    """Min-cost causal edit of w clearing the full `interferes` predicate,
    validated concretely by a dense body sweep against hero and ego."""
    aa = astate.actors[w]
    v_now = Evolution(astate, prm, controls).items[w][2].v_at(0.0)

    def clears(trial_controls) -> bool:
        evo2 = Evolution(astate, prm, trial_controls)
        ctx = Ctx(astate, evo2, 0.0, prm,
                  {"ego": astate.ego, "hero": res.hero},
                  tstar=res.t_star, protected=res.protected)
        if p_interferes(ctx, w, res.hero, astate.ego).value:
            return False
        others = [o for o in (res.hero, astate.ego) if o and o != w]
        return _body_sweep_clear(astate, prm, trial_controls, w, others,
                                 (res.t_star or prm.H) + 0.5)

    cands = []
    # retime: scan speeds by increasing |dv|, refine winner
    speeds = sorted({round(k * 0.25, 2) for k in range(int(prm.v_max / 0.25) + 1)},
                    key=lambda v: abs(v - v_now))
    for v in speeds:
        trial = {**controls, w: {**controls.get(w, {}), "speed": v}}
        if clears(trial):
            cands.append((prm.w_v * abs(v - v_now),
                          [Intervention("retime", w, v, prm.w_v * abs(v - v_now),
                                        "clear interference")]))
            break
    # reroute (only if uncommitted)
    if not aa.committed and aa.leg in IN_LEGS:
        for turn in TURNS:
            if turn == aa.turn:
                continue
            trial = {**controls, w: {**controls.get(w, {}), "turn": turn}}
            if clears(trial):
                cands.append((prm.w_r,
                              [Intervention("reroute", w, turn, prm.w_r,
                                            "route away from the scenario")]))
                break
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1]


def repair(astate: AbsState, prm: Params) -> RepairResult:
    controls: dict = {}
    plan: List[Intervention] = []
    res = evaluate_family(astate, prm, controls)
    for _ in range(4):
        if res.ok:
            break
        if not res.d1.value:
            opts = d1_repair_options(astate, prm, controls)
            if not opts:
                return RepairResult(False, plan, sum(i.cost for i in plan), res,
                                    "D1 unrepairable: no vehicle on a conflicting red"
                                    " approach can causally reach ego's window"
                                    " (the past is not available for editing)")
            cost, aid, turn, v_t, why = opts[0]
            c = dict(controls.get(aid, {}))
            c["speed"] = v_t
            if turn != astate.actors[aid].turn:
                c["turn"] = turn
                plan.append(Intervention("reroute", aid, turn, prm.w_r, why))
            plan.append(Intervention("retime", aid, v_t,
                                     prm.w_v * abs(v_t - astate.actors[aid].speed), why))
            controls[aid] = c
        elif not res.d2.value:
            # restore collidability margin: re-solve the hero's retime
            opts = [o for o in d1_repair_options(astate, prm, controls)
                    if o[1] == res.hero]
            if not opts:
                return RepairResult(False, plan, sum(i.cost for i in plan), res,
                                    f"D2 unrepairable for hero {res.hero}: committed"
                                    " past the conflict point")
            cost, aid, turn, v_t, why = opts[0]
            controls[aid] = {**controls.get(aid, {}), "speed": v_t}
            plan.append(Intervention("retime", aid, v_t,
                                     prm.w_v * abs(v_t - astate.actors[aid].speed), why))
        else:  # D3
            w = res.d3.witness.get("w")
            if w is None:
                break
            fix = _fix_interferer(astate, prm, controls, w, res)
            if fix is None:
                return RepairResult(False, plan, sum(i.cost for i in plan), res,
                                    f"D3 unrepairable: interferer {w} cannot be"
                                    " causally cleared")
            for iv in fix:
                c = dict(controls.get(w, {}))
                if iv.kind == "retime":
                    c["speed"] = iv.value
                else:
                    c["turn"] = iv.value
                controls[w] = c
                plan.append(iv)
        res = evaluate_family(astate, prm, controls)
    ok = res.ok
    return RepairResult(ok, plan, sum(i.cost for i in plan), res,
                        "" if ok else "unresolved after max repair passes")


def verify_collision(astate: AbsState, prm: Params, controls: dict,
                     hero: str, t_hi: float) -> Optional[float]:
    """Dense oriented-rectangle check that ego and hero bodies truly overlap."""
    evo = Evolution(astate, prm, controls or {})
    e, h = astate.actors[astate.ego], astate.actors[hero]
    t = 0.0
    while t <= t_hi + 1.0:
        xe, ye, he_ = evo.pose_at(astate.ego, t)
        xh, yh, hh_ = evo.pose_at(hero, t)
        if rects_overlap(rect_corners(xe, ye, he_, e.length, e.width),
                         rect_corners(xh, yh, hh_, h.length, h.width)):
            return t
        t += 1.0 / 30.0
    return None


# --------------------------------------------------------------------------- #
# 10. State loading, adapter, demo, CLI


def mkstate(label, actors, signals=None, ego="0", lane_width=3.5, arm_length=60.0):
    acts = [ActorState(str(a[0]), *a[1:]) for a in actors]
    return State(MapCfg(lane_width, arm_length), acts, ego, signals, label)


def load_state_file(path: str, T: Optional[float], signals: Optional[dict]) -> State:
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    acts = doc.get("actors", [])
    is_scenario = any("maneuvers" in (a or {}) or "start" in (a or {}) for a in acts)
    if is_scenario:
        return state_from_scenario(path, T or 0.0, signals)
    if T is not None:
        raise SystemExit("--time applies to scenario files only; "
                         "a state file has no clock")
    mp = doc.get("map", {})
    st = State(MapCfg(float(mp.get("lane_width", 3.5)),
                      float(mp.get("arm_length", 60.0))),
               [ActorState(str(a["id"]), float(a["x"]), float(a["y"]),
                           float(a["heading"]), float(a["speed"]),
                           float(a.get("length", 4.5)), float(a.get("width", 2.0)))
                for a in acts],
               str(doc.get("ego", "0")),
               signals or doc.get("signals"),
               os.path.basename(path))
    return st


def state_from_scenario(path: str, T: float, signals: Optional[dict]) -> State:
    """Adapter: sample a v0/v1 scenario file at clock T (headless simulation)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import scenario_editor as se
    sc = se.load_scenario(path)
    sc.simulate()
    period = sc.period
    k = int(round((T % period) / se.DT))
    actors = []
    for a in sc.actors:
        k2 = min(k, len(a.traj) - 1)
        x, y, hdg = a.traj[k2]
        v = a.speeds[k2]
        actors.append(ActorState(str(a.id), x, y, hdg, v, a.length, a.width))
    return State(MapCfg(sc.map.lane_width, sc.map.arm_length), actors, "0",
                 signals, f"{os.path.basename(path)} @ T={T:g}s")


def parse_signals(txt: str) -> dict:
    out = {}
    for part in txt.split(","):
        arm, _, phase = part.partition("=")
        arm, phase = arm.strip().upper(), phase.strip().lower()
        if arm not in "NESW" or phase not in ("red", "green"):
            raise SystemExit(f"bad --signals entry: {part}")
        out[arm] = phase
    return out


def _arc_pose(cx, cy, r, ang, dh, v):
    """Pose on a fillet arc at position-angle `ang` (deg), heading offset dh."""
    a = math.radians(ang)
    return (cx + r * math.cos(a), cy + r * math.sin(a), ang + dh, v)


def complex_demo_states() -> List[State]:
    """Populated-world demo inspired by v0 scenario_v20: actors on all legs,
    mid-turn actors inside the intersection, parked vehicles off-road, and two
    states sampled from the inspiring realization itself (if present)."""
    # v20 parks these on the west grass at x ~ -53; pulled in to stay in view
    parked = [("5", -34.0, 10.5, 224.2, 0.0), ("6", -29.33, 9.67, 227.6, 0.0)]

    def enl(ang, v):   # on the EN-left fillet (center (3.5,-3.5), r=5.25), CCW
        return _arc_pose(3.5, -3.5, 5.25, ang, +90.0, v)

    def ser(ang, v):   # on the SE-right fillet (center (3.5,-3.5), r=1.75), CW
        return _arc_pose(3.5, -3.5, 1.75, ang, -90.0, v)

    x1 = enl(95.0, 3.5)        # mid-left-turn, before ego's lane crossing
    x1past = enl(170.0, 8.0)   # mid-left-turn, already past the crossing
    x9 = ser(110.0, 5.0)       # mid-right-turn ahead of ego, clearing its lane
    states = [
        mkstate("X1: populated world, mid-turn hero on course (all hold)",
                [("0", 1.75, -15.0, 90.0, 12.0), ("1", *x1),
                 ("2", 26.0, 1.75, 180.0, 10.2), ("3", 40.0, 1.75, 180.0, 10.2),
                 ("4", -30.0, -1.75, 0.0, 0.0), *parked,
                 ("7", -1.75, 36.0, 270.0, 6.0), ("9", *x9)]),
        mkstate("X2: turner already through, approach hero too slow",
                [("0", 1.75, -26.0, 90.0, 12.0), ("1", *x1past),
                 ("2", 30.0, 1.75, 180.0, 6.0), ("3", 55.0, 1.75, 180.0, 6.0),
                 ("4", -30.0, -1.75, 0.0, 0.0), *parked,
                 ("7", -1.75, 32.0, 270.0, 8.0)]),
        mkstate("X3: southbound car sweeps the conflict and merges with hero",
                [("0", 1.75, -15.0, 90.0, 12.0), ("1", *x1),
                 ("3", 40.0, 1.75, 180.0, 10.0), ("4", -30.0, -1.75, 0.0, 0.0),
                 *parked, ("7", -1.75, 9.0, 270.0, 8.0), ("9", *x9)]),
        mkstate("X4: slow lead blocks the hero and hits the ego first",
                [("0", 1.75, -26.0, 90.0, 12.0), ("1", *x1past),
                 ("2", 26.0, 1.75, 180.0, 10.2), ("3", 12.0, 1.75, 180.0, 3.0),
                 *parked]),
        mkstate("X5: only candidate is stopped at the line (wake it)",
                [("0", 1.75, -26.0, 90.0, 12.0), ("2", 45.0, 1.75, 180.0, 8.0),
                 ("4", -12.0, -1.75, 0.0, 0.0), *parked,
                 ("7", -1.75, 32.0, 270.0, 8.0), ("9", *x9)]),
        mkstate("X6: ego about to clear - too late to stage anything",
                [("0", 1.75, -6.0, 90.0, 14.0), ("1", *x1past),
                 ("2", 40.0, 1.75, 180.0, 10.0), ("3", 55.0, 1.75, 180.0, 10.0),
                 ("4", -35.0, -1.75, 0.0, 0.0), *parked,
                 ("7", -1.75, 16.0, 270.0, 8.0)]),
    ]
    v20 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "v0", "scenarios", "scenario_v20.yaml")
    if os.path.exists(v20):
        for i, (T, tag) in enumerate(
                ((4.5, "the sampled realization satisfies the family"),
                 (2.0, "sampled early - repair wakes the stopping car")), start=7):
            st = state_from_scenario(v20, T, None)
            st.label = f"X{i}: scenario_v20 @ T={T:g}s ({tag})"
            states.append(st)
    states.append(
        mkstate("X9: two interferers at once (both causally cleared)",
                [("0", 1.75, -15.0, 90.0, 12.0), ("1", *x1),
                 ("3", 8.0, 1.75, 180.0, 4.0), ("4", -30.0, -1.75, 0.0, 0.0),
                 *parked, ("7", -1.75, 9.0, 270.0, 8.0), ("9", *x9)]))
    return states


def demo_states() -> List[State]:
    E = ("0", 1.75, -30.0, 90.0, 12.0)
    return [
        mkstate("S1: hero on collision course (all directives hold)",
                [E, ("1", 30.0, 1.75, 180.0, 10.68)]),
        mkstate("S2: hero too slow, misses ego's window",
                [E, ("1", 30.0, 1.75, 180.0, 6.0)]),
        mkstate("S3: hero already through; uncommitted candidate behind",
                [E, ("1", -8.0, 1.75, 180.0, 12.0), ("2", 45.0, 1.75, 180.0, 12.0)]),
        mkstate("S4: slow lead vehicle ahead of hero (blocks + hits ego)",
                [E, ("1", 30.0, 1.75, 180.0, 10.68), ("2", 15.0, 1.75, 180.0, 4.0)]),
        mkstate("S5: crossing vehicle sweeps the conflict during the window",
                [E, ("1", 30.0, 1.75, 180.0, 10.68), ("2", -12.0, -1.75, 0.0, 6.0)]),
        mkstate("S6: nobody on a conflicting red approach (infeasible)",
                [E, ("2", -1.75, 25.0, 270.0, 10.0)]),
    ]


def fmt_ev(e: Ev) -> str:
    mark = "✓" if e.value else "✗"
    bits = [mark]
    if e.witness:
        bits.append("witness " + ",".join(f"{k}={v}" for k, v in e.witness.items()))
    if not e.value and e.t_fail is not None:
        bits.append(f"t_fail={e.t_fail:.2f}")
    bits.append(e.expl)
    bits.append(f"(conf {e.conf:.2f})")
    return "  ".join(bits)


def report(state: State, prm: Params) -> RepairResult:
    astate = recognize(state, prm)
    print("=" * 78)
    print(f"{state.label or 'state'}   "
          f"[{'ramp a_max=%.1f' % prm.ramp if prm.ramp else 'instantaneous velocity changes'}]")
    sig = " ".join(f"{a}={astate.signals[a]}" for a in "NESW")
    print(f"  signals: {sig}" + ("   (assumed family-consistent)" if astate.signals_assumed else ""))
    for aid in sorted(astate.actors):
        aa = astate.actors[aid]
        role = " (ego)" if aid == astate.ego else ""
        route = f"{aa.leg} {aa.turn}" if aa.turn else (aa.leg or "?")
        print(f"  actor {aid}{role}: {aa.region:<12} route {route:<12} "
              f"v={aa.speed:.2f}  conf {aa.conf:.2f}")
    res = evaluate_family(astate, prm)
    for (name, gloss, _), e in zip(DIRECTIVES, (res.d1, res.d2, res.d3)):
        print(f"  {name} [{gloss}]")
        print(f"     {fmt_ev(e)}")
    if res.ok:
        print(f"  planned collision t* = {res.t_star:.2f} s   -> no intervention needed")
        t = verify_collision(astate, prm, {}, res.hero, res.t_star)
        print(f"  verify: bodies overlap at t={t:.2f} s" if t is not None
              else "  verify: WARNING - no dense-body overlap found")
        return RepairResult(True, [], 0.0, res)
    rr = repair(astate, prm)
    if not rr.feasible:
        print(f"  INFEASIBLE: {rr.reason}")
        return rr
    print("  minimal causal intervention:")
    for iv in rr.interventions:
        print(f"     {iv}")
    print(f"     total cost {rr.cost:.2f}")
    f = rr.final
    marks = " ".join(f"{n}={'✓' if e.value else '✗'}"
                     for (n, _, _), e in zip(DIRECTIVES, (f.d1, f.d2, f.d3)))
    print(f"  re-evaluation: {marks}   hero={f.hero}  t*={f.t_star:.2f} s")
    controls = {}
    for iv in rr.interventions:
        c = controls.setdefault(iv.actor, {})
        c["speed" if iv.kind == "retime" else "turn"] = iv.value
    t = verify_collision(astate, prm, controls, f.hero, f.t_star)
    print(f"  verify: bodies overlap at t={t:.2f} s" if t is not None
          else "  verify: WARNING - no dense-body overlap found")
    return rr


def main():
    ap = argparse.ArgumentParser(description="v2 directive layer (see DESIGN.md)")
    ap.add_argument("--demo", nargs="?", const="basic",
                    choices=["basic", "complex", "all"], default=None,
                    help="run built-in demo states (basic | complex | all)")
    ap.add_argument("--state", help="state YAML or v0/v1 scenario YAML")
    ap.add_argument("--time", type=float, default=None,
                    help="clock T for scenario files (adapter)")
    ap.add_argument("--signals", type=str, default=None,
                    help="e.g. N=green,S=green,E=red,W=red")
    ap.add_argument("--ramp", nargs="?", const=3.0, default=None, type=float,
                    help="C0-velocity variant with a_max (default 3 m/s^2)")
    args = ap.parse_args()
    prm = Params(ramp=args.ramp)
    signals = parse_signals(args.signals) if args.signals else None
    if args.state:
        st = load_state_file(args.state, args.time, signals)
        report(st, prm)
    elif args.demo:
        states = []
        if args.demo in ("basic", "all"):
            states += demo_states()
        if args.demo in ("complex", "all"):
            states += complex_demo_states()
        for st in states:
            if signals:
                st.signals = signals
            report(st, prm)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
