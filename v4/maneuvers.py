#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/maneuvers.py — maneuver-script surgery shared by the kernel and the
script-based directive layer.

Everything here operates on the v0/v2 maneuver representation (the one shared
ground truth, P1): re-basing (freeze the elapsed prefix), retime/reroute
(constant-speed edits along a route), route construction for spawns, and the
user perturbation vocabulary. Kept separate from orchestrator.py so
directives_script.py can build/verify trial worlds without a circular import.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
if V2 not in sys.path:
    sys.path.insert(0, V2)

import scenario_editor as se          # noqa: E402

BIG_DUR = 60.0
TURN_TYPES = se.TURN_TYPES
LONG_TYPES = se.LONGITUDINAL_TYPES

LEG_SPAWN = {  # inbound leg -> (x, y, heading) at the arm end (right-hand lane)
    "SE": lambda lw, arm: (lw / 2, -arm + 2.0, 90.0),
    "EN": lambda lw, arm: (arm - 2.0, lw / 2, 180.0),
    "WS": lambda lw, arm: (-arm + 2.0, -lw / 2, 0.0),
    "NW": lambda lw, arm: (-lw / 2, arm - 2.0, 270.0),
}
LEG_HEADING = {"SE": 90.0, "EN": 180.0, "WS": 0.0, "NW": 270.0}
LEG_LANE = {"SE": ("x", +1), "EN": ("y", +1), "WS": ("y", -1), "NW": ("x", -1)}
PALETTE = [(90, 190, 110), (210, 70, 60), (60, 120, 210), (200, 160, 60),
           (160, 90, 200), (80, 200, 200), (150, 185, 95), (225, 105, 165)]


# --------------------------------------------------------------------------- #
# Re-basing: freeze the elapsed prefix, keep the remaining suffix
# --------------------------------------------------------------------------- #
def _truncate_maneuver(m: se.Maneuver, t_i: float) -> se.Maneuver:
    """The remaining tail of maneuver `m` after local time t_i, starting from
    the pose reached at t_i (trajectory-preserving)."""
    t_i = max(0.0, min(t_i, m.duration))
    rem = m.duration - t_i
    if m.type in TURN_TYPES:
        frac = t_i / max(1e-6, m.duration)
        return se.Maneuver(type=m.type, radius=m.radius,
                           angle=m.angle * (1.0 - frac), duration=rem)
    if m.type == "stop":
        return se.Maneuver(type="stop", duration=rem)
    return se.Maneuver(type=m.type, intercept=m.velocity_at(t_i),
                       slope=m.slope, duration=rem)


def _pose_speed_at(a: se.Actor, tau: float) -> Tuple[se.Pose, float]:
    k = int(round(tau / se.DT))
    if a.traj:
        k = max(0, min(k, len(a.traj) - 1))
        return a.traj[k], a.speeds[k]
    return a.start, 0.0


def rebase_actor(a: se.Actor, tau: float) -> se.Actor:
    a.compute_schedule()
    pose, _ = _pose_speed_at(a, tau)
    segs = a.maneuvers
    if tau >= a.total - 1e-9 or not segs:
        new_segs: List = []
    else:
        i = a.active_index(tau)
        t_i = tau - a.cum[i]
        head = segs[i]
        tail = ([_truncate_maneuver(head, t_i)] if isinstance(head, se.Maneuver)
                else [copy.deepcopy(head)])
        new_segs = tail + [copy.deepcopy(s) for s in segs[i + 1:]]
    return se.Actor(id=a.id, color=a.color, length=a.length, width=a.width,
                    start=pose, maneuvers=new_segs)


def rebase_scenario(sc: se.Scenario, tau: float) -> se.Scenario:
    sc.simulate()
    actors = [rebase_actor(a, tau) for a in sc.actors]
    new = se.Scenario(map=se.MapConfig(sc.map.lane_width, sc.map.arm_length),
                      actors=actors, pixels_per_meter=sc.pixels_per_meter)
    new.simulate()
    return new


# --------------------------------------------------------------------------- #
# Speed / route edits (constant-speed along the existing geometry)
# --------------------------------------------------------------------------- #
def _geom_distance(m: se.Maneuver) -> float:
    if m.type in TURN_TYPES:
        return m.arc_length()
    if m.type == "stop":
        return 0.0
    return m.distance(m.duration)


def retime_actor(a: se.Actor, v_target: float) -> None:
    """Adopt target speed `v_target` along the actor's existing route, in place.
    Geometry preserved; only speed changes. v_target == 0 -> yield/stop."""
    if v_target <= 1e-3:
        a.maneuvers = [se.Maneuver(type="stop", duration=BIG_DUR)]
        return
    out: List[se.Maneuver] = []
    for m in a.maneuvers:
        if not isinstance(m, se.Maneuver):
            out.append(m)
            continue
        if m.type in TURN_TYPES:
            out.append(se.Maneuver(type=m.type, radius=m.radius, angle=m.angle,
                                   duration=m.arc_length() / v_target))
        else:
            dist = _geom_distance(m)
            if dist <= 1e-6:
                continue
            out.append(se.Maneuver(type="go_straight", intercept=v_target,
                                   slope=0.0, duration=dist / v_target))
    if not out:
        out = [se.Maneuver(type="go_straight", intercept=v_target, slope=0.0,
                           duration=BIG_DUR)]
    a.maneuvers = out


def _snap_heading(hdg: float) -> float:
    return min((0.0, 90.0, 180.0, 270.0, 360.0),
               key=lambda h: abs(((hdg - h + 180) % 360) - 180)) % 360


def _dist_to_intersection(x: float, y: float, hdg: float, lw: float) -> float:
    h = _snap_heading(hdg)
    if h == 90:
        return max(0.0, (-lw) - y)
    if h == 270:
        return max(0.0, y - lw)
    if h == 0:
        return max(0.0, (-lw) - x)
    return max(0.0, x - lw)


def build_route_maneuvers(x: float, y: float, hdg: float, speed: float,
                          turn: str, lw: float, arm: float) -> List[se.Maneuver]:
    """Constant-speed suffix from (x,y,hdg): approach -> (turn) -> exit."""
    speed = max(1.0, speed)
    r = 6.0
    d_app = _dist_to_intersection(x, y, hdg, lw)
    if turn == "straight":
        span = d_app + 2 * lw + arm + 6.0
        return [se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                            duration=span / speed)]
    segs: List[se.Maneuver] = []
    if d_app > 1e-3:
        segs.append(se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                                duration=d_app / speed))
    ttype = "turn_left" if turn == "left" else "turn_right"
    tm = se.Maneuver(type=ttype, radius=r, angle=90.0, duration=1.0)
    tm.duration = tm.arc_length() / speed
    segs.append(tm)
    segs.append(se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                            duration=(arm + 6.0) / speed))
    return segs


def reroute_actor(a: se.Actor, turn: str, lw: float, arm: float) -> None:
    """Best-effort route change: rebuild the suffix to follow `turn` at the
    actor's current speed from its current pose."""
    _, v = _pose_speed_at(a, 0.0)
    if v <= 1e-3 and a.maneuvers and isinstance(a.maneuvers[0], se.Maneuver):
        v = a.maneuvers[0].velocity_at(0.0)
    x, y, hdg = a.start
    a.maneuvers = build_route_maneuvers(x, y, hdg, max(v, 6.0), turn, lw, arm)


# --------------------------------------------------------------------------- #
# Perturbations (user maneuver-script edits; may target the ego)
# --------------------------------------------------------------------------- #
@dataclass
class Perturbation:
    kind: str                       # add_actor | set_maneuver | set_speed | remove_actor
    actor: Optional[str] = None
    leg: Optional[str] = None
    turn: Optional[str] = None
    speed: Optional[float] = None
    x: Optional[float] = None        # explicit spawn pose (add_actor); else the arm end
    y: Optional[float] = None
    heading: Optional[float] = None
    note: str = ""                   # optional label for the tree/delta

    def summary(self) -> str:
        if self.kind == "add_actor":
            where = self.note or (f"{self.leg} @ ({self.x:g},{self.y:g})"
                                  if self.x is not None else self.leg)
            return f"add actor on {where} ({self.turn}, {self.speed:g} m/s)"
        if self.kind == "set_maneuver":
            return f"set actor {self.actor} route -> {self.turn}"
        if self.kind == "set_speed":
            return f"set actor {self.actor} speed -> {self.speed:g} m/s"
        if self.kind == "remove_actor":
            return f"remove actor {self.actor}"
        return self.kind


def apply_perturbation(sc: se.Scenario, p: Perturbation) -> se.Scenario:
    """Apply a perturbation to an already-re-based scenario (returns a new sc)."""
    lw, arm = sc.map.lane_width, sc.map.arm_length
    actors = [copy.deepcopy(a) for a in sc.actors]
    by_id = {a.id: a for a in actors}
    if p.kind == "add_actor":
        nid = str(max([int(a.id) for a in actors if a.id.isdigit()] + [-1]) + 1)
        if p.x is not None:
            x, y = p.x, p.y
            hdg = p.heading if p.heading is not None else LEG_HEADING[p.leg]
        else:
            x, y, hdg = LEG_SPAWN[p.leg](lw, arm)
        color = PALETTE[len(actors) % len(PALETTE)]
        man = build_route_maneuvers(x, y, hdg, p.speed or 10.0,
                                    p.turn or "straight", lw, arm)
        actors.append(se.Actor(id=nid, color=color, length=4.5, width=2.0,
                               start=(x, y, hdg), maneuvers=man))
    elif p.kind == "remove_actor":
        actors = [a for a in actors if a.id != p.actor]
    elif p.kind == "set_maneuver" and p.actor in by_id:
        reroute_actor(by_id[p.actor], p.turn or "left", lw, arm)
    elif p.kind == "set_speed" and p.actor in by_id:
        retime_actor(by_id[p.actor], p.speed or 0.0)
    new = se.Scenario(map=se.MapConfig(lw, arm), actors=actors,
                      pixels_per_meter=sc.pixels_per_meter)
    new.simulate()
    return new
