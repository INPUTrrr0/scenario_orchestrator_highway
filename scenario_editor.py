#!/usr/bin/env python3
"""
Scenario Editor v1
==================
Load an orchestrated intersection scenario (YAML), visualize it in a bird's-eye
(BEV) pygame view, play/loop it, and edit per-segment behavior.

v1 model (see DESIGN.md):
  * An actor has a sequential list of SEGMENTS, each either a maneuver
    (go_straight / turn_left / turn_right / accelerate / decelerate / stop,
    unchanged from v0) or a FUNCTION: a reactive node graph mapping
    observations (other actors, the map, time) to actions.
  * A function segment has two sinks: OUT:speed (m/s) and OUT:yaw (heading,
    deg CCW from East). Bound sinks drive that attribute each tick; unbound
    attributes hold their entry value. Motion is a kinematic unicycle.
  * Simulation is tick-based (fixed dt = 1/60 s) with a cached trajectory,
    recomputed after every edit. Functions observe the PREVIOUS tick's world
    state, so evaluation order across actors doesn't matter.
  * All multi-choice UI selections use dropdowns (segment type, node values,
    edge operations) — nothing click-cycles.
  * Edits are logged to edit_history.yaml; saving writes a version-numbered
    file and records a branching provenance graph in provenance.yaml.

Coordinates: graph/Cartesian, meters, origin at intersection center,
x = East, y = North (y up), heading in degrees CCW from East.

Usage:
    python scenario_editor.py [scenario.yaml]
    python scenario_editor.py --validate scenario.yaml
    python scenario_editor.py scenario.yaml --capture out.mp4 [--fps N --loops N]
    python scenario_editor.py scenario.yaml --snapshot out.png [--select ID --time T]

Controls:
    Space / Play button : play-pause (loops forever)
    Reset / Save        : clock -> 0 / write next version + provenance
    Click actor (paused): select; opens the segment editor below
    type / prev / next  : dropdown segment type; step through segments
    Maneuver segments   : drag curve endpoints or edit fields (v0)
    Function segments   : double-click canvas = new node; click node = choose
                          its value; drag from a node's right port = wire
                          (to a node BODY: new binary op combining the two,
                          works for op outputs too; to an op's input SLOT:
                          fill or rewire that slot; to empty: unary op; to an
                          OUT sink: bind); click an op = choose operation;
                          <> badge swaps inputs; Delete = remove selected
                          node + everything downstream
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import yaml

Pose = Tuple[float, float, float]  # (x, y, heading_deg)
DT = 1.0 / 60.0                    # simulation tick

LONGITUDINAL_TYPES = {"go_straight", "accelerate", "decelerate"}
TURN_TYPES = {"turn_left", "turn_right"}
# Lateral lane change: forward motion from a velocity curve + smooth lateral
# offset (+left / −right of heading). Heading returns to the start heading.
LANE_CHANGE_TYPES = {"lane_change"}
VELOCITY_CURVE_TYPES = LONGITUDINAL_TYPES | LANE_CHANGE_TYPES
ALL_MANEUVER_TYPES = LONGITUDINAL_TYPES | TURN_TYPES | LANE_CHANGE_TYPES | {"stop"}
# dropdown order for the segment `type:` button
SEGMENT_TYPES = ["go_straight", "lane_change", "turn_left", "turn_right",
                 "accelerate", "decelerate", "stop", "function"]

# ---- function-graph type system ------------------------------------------- #
SCALAR, POINT = "scalar", "point"
# op -> (input signature, output type); 'T' is a generic unified from inputs
OPS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "add": ((SCALAR, SCALAR), SCALAR), "sub": ((SCALAR, SCALAR), SCALAR),
    "mul": ((SCALAR, SCALAR), SCALAR), "div": ((SCALAR, SCALAR), SCALAR),
    "min": ((SCALAR, SCALAR), SCALAR), "max": ((SCALAR, SCALAR), SCALAR),
    "lt":  ((SCALAR, SCALAR), SCALAR), "gt":  ((SCALAR, SCALAR), SCALAR),
    "neg": ((SCALAR,), SCALAR), "abs": ((SCALAR,), SCALAR),
    "not": ((SCALAR,), SCALAR),
    "if":  ((SCALAR, "T", "T"), "T"),
    "dist": ((POINT, POINT), SCALAR), "bearing": ((POINT, POINT), SCALAR),
    "midpoint": ((POINT, POINT), POINT),
    "x": ((POINT,), SCALAR), "y": ((POINT,), SCALAR), "mag": ((POINT,), SCALAR),
}
OP_ORDER = ["add", "sub", "mul", "div", "min", "max", "lt", "gt", "if",
            "dist", "bearing", "midpoint", "neg", "abs", "not", "x", "y", "mag"]
ASYMMETRIC_OPS = {"sub", "div", "lt", "gt"}   # input order matters -> swap badge
EPS = 1e-6

MAP_POINT_NAMES = ["center", "cross_NE", "cross_NW", "cross_SE", "cross_SW"]


def map_point_coords(name: str, lane_width: float) -> Tuple[float, float]:
    d = lane_width / 2.0
    return {"center": (0.0, 0.0), "cross_NE": (d, d), "cross_NW": (-d, d),
            "cross_SE": (d, -d), "cross_SW": (-d, -d)}[name]


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# Map layout + rendering live in maps.py; re-exported so existing
# `import scenario_editor as se; se.MapConfig` callers keep working.
from maps import (MapConfig, clone_map, map_from_dict,  # noqa: E402
                  draw_map as draw_map_surface, draw_rulers)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Maneuver:
    type: str
    duration: float = 1.0
    # velocity curve for longitudinal / lane_change: v(t) = intercept + slope*t
    intercept: float = 0.0     # initial speed v0 (m/s)
    slope: float = 0.0         # acceleration a (m/s^2)
    # turn geometry
    radius: float = 5.0
    angle: float = 90.0
    # lane_change: lateral displacement in meters (+left / −right of heading)
    lateral_offset: float = 3.5

    @property
    def curve_kind(self) -> str:
        return "velocity" if self.type in VELOCITY_CURVE_TYPES else "none"

    def arc_length(self) -> float:
        return self.radius * math.radians(self.angle)

    def velocity_at(self, t: float) -> float:
        if self.type == "stop":
            return 0.0
        if self.type in TURN_TYPES:
            return self.arc_length() / max(1e-6, self.duration)
        return max(0.0, self.intercept + self.slope * t)

    def distance(self, t: float) -> float:
        if self.type == "stop":
            return 0.0
        if self.type in TURN_TYPES:
            return self.arc_length() * clamp(t / max(1e-6, self.duration), 0.0, 1.0)
        t_eff = t
        if self.slope < 0:
            t_stop = -self.intercept / self.slope if self.slope != 0 else t
            t_eff = clamp(t, 0.0, max(0.0, t_stop))
        return self.intercept * t_eff + 0.5 * self.slope * t_eff * t_eff

    def pose_at(self, start: Pose, t: float) -> Pose:
        sx, sy, sh = start
        h = math.radians(sh)
        if self.type == "stop":
            return (sx, sy, sh)
        if self.type in LONGITUDINAL_TYPES:
            s = self.distance(t)
            return (sx + s * math.cos(h), sy + s * math.sin(h), sh)
        if self.type == "lane_change":
            frac = clamp(t / max(1e-6, self.duration), 0.0, 1.0)
            s_lat = frac * frac * (3.0 - 2.0 * frac)
            along = self.distance(t)
            lat = self.lateral_offset * s_lat
            nx, ny = -math.sin(h), math.cos(h)
            yaw = (12.0 if self.lateral_offset >= 0 else -12.0) * math.sin(math.pi * frac)
            return (sx + along * math.cos(h) + lat * nx,
                    sy + along * math.sin(h) + lat * ny,
                    sh + yaw)
        frac = clamp(t / max(1e-6, self.duration), 0.0, 1.0)
        if self.type == "turn_left":
            cx, cy = sx - self.radius * math.sin(h), sy + self.radius * math.cos(h)
            phi0 = math.atan2(sy - cy, sx - cx)
            delta = math.radians(self.angle) * frac
            return (cx + self.radius * math.cos(phi0 + delta),
                    cy + self.radius * math.sin(phi0 + delta), sh + self.angle * frac)
        if self.type == "turn_right":
            cx, cy = sx + self.radius * math.sin(h), sy - self.radius * math.cos(h)
            phi0 = math.atan2(sy - cy, sx - cx)
            delta = -math.radians(self.angle) * frac
            return (cx + self.radius * math.cos(phi0 + delta),
                    cy + self.radius * math.sin(phi0 + delta), sh - self.angle * frac)
        return (sx, sy, sh)

    def end_pose(self, start: Pose) -> Pose:
        return self.pose_at(start, self.duration)

    def exit_speed(self) -> float:
        return self.velocity_at(self.duration)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "duration": round(self.duration, 4)}
        if self.type in VELOCITY_CURVE_TYPES:
            d["curve"] = {"v0": round(self.intercept, 4), "accel": round(self.slope, 4)}
            if self.type == "lane_change":
                d["lateral_offset"] = round(self.lateral_offset, 4)
        elif self.type in TURN_TYPES:
            d["radius"] = round(self.radius, 4)
            d["angle"] = round(self.angle, 4)
        return d


@dataclass
class Node:
    """One node of a function graph.  kind:
    obs (source, field) | const (value) | map_point (name | point) |
    time (which: global|segment) | op (op, inputs)."""
    id: str
    kind: str
    pos: Tuple[float, float] = (40.0, 40.0)   # editor-canvas coordinates
    source: str = "self"          # obs: 'self' or an actor id
    value: float = 1.0            # const
    name: Optional[str] = None    # map_point: named catalog entry
    point: Optional[Tuple[float, float]] = None   # map_point: custom (x, y)
    which: str = "global"         # time: 'global' | 'segment'
    op: str = "add"               # op
    inputs: Optional[List[str]] = None            # op: node ids, slot order
    field: str = "speed"          # obs: 'speed' | 'pos' | 'heading'
    # NOTE: `field` must stay the LAST annotation here — it shadows
    # dataclasses.field inside this class body once defined.

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        self.pos = tuple(self.pos)
        if self.point is not None:
            self.point = tuple(self.point)

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "kind": self.kind,
                   "pos": [round(self.pos[0], 1), round(self.pos[1], 1)]}
        if self.kind == "obs":
            d["source"] = self.source
            d["field"] = self.field
        elif self.kind == "const":
            d["value"] = round(self.value, 6)
        elif self.kind == "map_point":
            if self.name:
                d["name"] = self.name
            else:
                p = self.point or (0.0, 0.0)
                d["point"] = [round(p[0], 4), round(p[1], 4)]
        elif self.kind == "time":
            d["which"] = self.which
        elif self.kind == "op":
            d["op"] = self.op
            d["inputs"] = list(self.inputs)
        return d


@dataclass
class Function:
    """A reactive segment: node graph -> OUT:speed / OUT:yaw sinks."""
    duration: float = 4.0
    nodes: List[Node] = field(default_factory=list)
    out_speed: Optional[str] = None
    out_yaw: Optional[str] = None
    type: str = "function"          # uniform with Maneuver.type
    sim_exit_speed: float = 0.0     # filled by the simulator

    @property
    def curve_kind(self) -> str:
        return "graph"

    def exit_speed(self) -> float:
        return self.sim_exit_speed

    def by_id(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def node(self, nid: Optional[str]) -> Optional[Node]:
        return self.by_id().get(nid) if nid else None

    def next_node_id(self) -> str:
        mx = -1
        for n in self.nodes:
            if n.id.startswith("n"):
                try:
                    mx = max(mx, int(n.id[1:]))
                except ValueError:
                    pass
        return f"n{mx + 1}"

    def to_dict(self) -> dict:
        out = {}
        if self.out_speed:
            out["speed"] = self.out_speed
        if self.out_yaw:
            out["yaw"] = self.out_yaw
        return {"type": "function", "duration": round(self.duration, 4),
                "out": out, "nodes": [n.to_dict() for n in self.nodes]}


Segment = Union[Maneuver, Function]


# --------------------------------------------------------------------------- #
# Function-graph analysis & evaluation
# --------------------------------------------------------------------------- #
def node_value_type(fn: Function, nid: Optional[str],
                    _seen: Optional[set] = None) -> Optional[str]:
    """SCALAR / POINT, or None if unknown (missing, cyclic, unresolved T)."""
    node = fn.node(nid)
    if node is None:
        return None
    if node.kind == "obs":
        return POINT if node.field == "pos" else SCALAR
    if node.kind in ("const", "time"):
        return SCALAR
    if node.kind == "map_point":
        return POINT
    # op
    if node.op not in OPS:
        return None
    seen = _seen or set()
    if nid in seen:
        return None
    sig, out = OPS[node.op]
    if out != "T":
        return out
    for i, st in enumerate(sig):
        if st == "T" and i < len(node.inputs):
            t = node_value_type(fn, node.inputs[i], seen | {nid})
            if t:
                return t
    return None


def op_is_complete(fn: Function, node: Node) -> bool:
    if node.kind != "op" or node.op not in OPS:
        return False
    sig, _ = OPS[node.op]
    if len(node.inputs) < len(sig):
        return False
    by = fn.by_id()
    return all(i in by for i in node.inputs)


def slot_accepts(fn: Function, op_node: Node, slot: int, src_type: Optional[str]) -> bool:
    """Can a value of src_type be wired into `slot` of op_node?"""
    sig, _ = OPS[op_node.op]
    if slot >= len(sig):
        return False
    st = sig[slot]
    if st == "T":
        # unify with any already-wired T slot of known type
        for i, s2 in enumerate(sig):
            if s2 == "T" and i < len(op_node.inputs) and i != slot:
                t = node_value_type(fn, op_node.inputs[i])
                if t and src_type and t != src_type:
                    return False
        return True
    return src_type is None or src_type == st


def compatible_ops(fn: Function, wired: List[Optional[str]]) -> List[str]:
    """Ops whose signature is consistent with the wired input types (prefix)."""
    result = []
    for op in OP_ORDER:
        sig, _ = OPS[op]
        k = min(len(sig), len(wired))
        ok, tbind = True, None
        for i in range(k):
            st, wt = sig[i], wired[i]
            if wt is None:
                continue
            if st == "T":
                if tbind is None:
                    tbind = wt
                elif tbind != wt:
                    ok = False
                    break
            elif st != wt:
                ok = False
                break
        if ok:
            result.append(op)
    return result


def would_cycle(fn: Function, op_id: str, src_id: str) -> bool:
    """True if wiring src into op would create a cycle (op reachable from src)."""
    by = fn.by_id()
    stack, seen = [src_id], set()
    while stack:
        nid = stack.pop()
        if nid == op_id:
            return True
        if nid in seen:
            continue
        seen.add(nid)
        n = by.get(nid)
        if n and n.kind == "op":
            stack.extend(n.inputs)
    return False


def downstream_ids(fn: Function, nid: str) -> set:
    """nid plus every op that (transitively) consumes it."""
    dead = {nid}
    changed = True
    while changed:
        changed = False
        for n in fn.nodes:
            if n.kind == "op" and n.id not in dead and any(i in dead for i in n.inputs):
                dead.add(n.id)
                changed = True
    return dead


def _apply_op(op: str, vals: list):
    try:
        if op == "add":
            return vals[0] + vals[1]
        if op == "sub":
            return vals[0] - vals[1]
        if op == "mul":
            return vals[0] * vals[1]
        if op == "div":
            d = vals[1]
            return vals[0] / (d if abs(d) > EPS else math.copysign(EPS, d or 1.0))
        if op == "min":
            return min(vals[0], vals[1])
        if op == "max":
            return max(vals[0], vals[1])
        if op == "lt":
            return 1.0 if vals[0] < vals[1] else 0.0
        if op == "gt":
            return 1.0 if vals[0] > vals[1] else 0.0
        if op == "neg":
            return -vals[0]
        if op == "abs":
            return abs(vals[0])
        if op == "not":
            return 0.0 if abs(vals[0]) > EPS else 1.0
        if op == "if":
            return vals[1] if abs(vals[0]) > EPS else vals[2]
        if op == "dist":
            return math.hypot(vals[1][0] - vals[0][0], vals[1][1] - vals[0][1])
        if op == "bearing":
            return math.degrees(math.atan2(vals[1][1] - vals[0][1],
                                           vals[1][0] - vals[0][0]))
        if op == "midpoint":
            return ((vals[0][0] + vals[1][0]) / 2.0, (vals[0][1] + vals[1][1]) / 2.0)
        if op == "x":
            return vals[0][0]
        if op == "y":
            return vals[0][1]
        if op == "mag":
            return math.hypot(vals[0][0], vals[0][1])
    except (TypeError, IndexError):
        return None
    return None


def _eval_node(fn: Function, nid: Optional[str], ctx: dict,
               memo: dict, stack: set):
    """Value of node nid (float | (x,y) | None). ctx keys:
    self, world {id: (x, y, heading, speed)}, T, t, lane_width."""
    if nid is None:
        return None
    if nid in memo:
        return memo[nid]
    if nid in stack:            # cycle guard (shouldn't happen via the editor)
        return None
    node = fn.node(nid)
    if node is None:
        return None
    val = None
    if node.kind == "const":
        val = node.value
    elif node.kind == "time":
        val = ctx["t"] if node.which == "segment" else ctx["T"]
    elif node.kind == "map_point":
        if node.name in MAP_POINT_NAMES:
            val = map_point_coords(node.name, ctx["lane_width"])
        elif node.point is not None:
            val = tuple(node.point)
    elif node.kind == "obs":
        aid = ctx["self"] if node.source == "self" else node.source
        w = ctx["world"].get(aid)
        if w is not None:
            if node.field == "pos":
                val = (w[0], w[1])
            elif node.field == "heading":
                val = w[2]
            else:
                val = w[3]
    elif node.kind == "op" and node.op in OPS:
        sig, _ = OPS[node.op]
        if len(node.inputs) >= len(sig):
            stack.add(nid)
            vals = [_eval_node(fn, i, ctx, memo, stack) for i in node.inputs[:len(sig)]]
            stack.discard(nid)
            if all(v is not None for v in vals):
                val = _apply_op(node.op, vals)
    memo[nid] = val
    return val


def eval_outputs(fn: Function, ctx: dict) -> Tuple[Optional[float], Optional[float]]:
    """(speed, yaw) — each None if unbound or the graph is incomplete."""
    memo: dict = {}
    v = _eval_node(fn, fn.out_speed, ctx, memo, set()) if fn.out_speed else None
    h = _eval_node(fn, fn.out_yaw, ctx, memo, set()) if fn.out_yaw else None
    if isinstance(v, tuple):
        v = None
    if isinstance(h, tuple):
        h = None
    return v, h


# --------------------------------------------------------------------------- #
# Actor / Scenario
# --------------------------------------------------------------------------- #
@dataclass
class Actor:
    id: str
    color: Tuple[int, int, int]
    length: float
    width: float
    start: Pose
    maneuvers: List[Segment] = field(default_factory=list)   # mixed segments
    # cut-in constraint: {"t","along","lat","lc_duration","tail"} — when set,
    # the actor's maneuvers are auto-generated by resolve_cutins() so its lane
    # change completes at time `t` at ego_pose(t) + along*fwd + lat*left.
    # Offsets are ego-relative; if the ego's script changes, the world place
    # moves with the ego but the relative cut-off stays the same.
    cutin: Optional[dict] = None
    # block-cut-in constraint: {"t","duration","along","lat"} — the ego wants
    # to change lanes and this actor denies the gap: it speeds up so that from
    # time `t` through `t+duration` it holds a station `along` metres ahead /
    # `lat` metres lateral of the ego (lat = the ego's target lane).  Missing
    # t/duration/lat are auto-derived from the ego's first scripted
    # lane_change.  The orchestrator moves this spec between fully-autonomous
    # actors (role casting), just like `cutin`.
    block: Optional[dict] = None
    # nominal cruise speed (m/s). When set and no maneuvers are given in the
    # YAML, a straight-cruise plan is auto-generated; after an unsuccessful
    # cut-in the actor returns to this speed and keeps driving straight.
    cruise: Optional[float] = None
    # mixed autonomy — who governs this actor:
    #   "auto": fully autonomous — no independent plan; follows whatever the
    #           orchestrator casts (cut-in / nominal cruising)
    #   "self": fully self-governed — executes only user-issued intents
    #           (speed up, brake, cut-in pin, edited segments); the
    #           orchestrator neither casts it nor touches its plan
    autonomy: str = "auto"
    # filled by Scenario.simulate():
    cum: List[float] = field(default_factory=list)
    total: float = 0.0
    traj: List[Pose] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    final_pose: Pose = (0.0, 0.0, 0.0)

    def compute_schedule(self) -> None:
        self.cum = [0.0]
        for m in self.maneuvers:
            self.cum.append(self.cum[-1] + max(0.0, m.duration))
        self.total = self.cum[-1]

    def active_index(self, phase: float) -> int:
        for i in range(len(self.maneuvers)):
            if self.cum[i] <= phase < self.cum[i + 1]:
                return i
        return max(0, len(self.maneuvers) - 1)

    def pose_at_time(self, phase: float) -> Pose:
        if not self.traj:
            return self.start
        i = int(round(phase / DT))
        return self.traj[max(0, min(i, len(self.traj) - 1))]

    def to_dict(self) -> dict:
        d = {"id": self.id, "color": list(self.color),
             "length": self.length, "width": self.width,
             "start": {"x": round(self.start[0], 4), "y": round(self.start[1], 4),
                       "heading": round(self.start[2], 4)}}
        if self.cruise is not None:
            # cruise actors: maneuvers are derived (cruise plan / cut-in
            # solver), so only the cruise speed is persisted
            d["cruise"] = round(float(self.cruise), 4)
        else:
            d["maneuvers"] = [m.to_dict() for m in self.maneuvers]
        if self.cutin:
            d["cutin"] = {k: round(float(v), 4) for k, v in self.cutin.items()}
        if self.block:
            d["block"] = {k: round(float(v), 4) for k, v in self.block.items()}
        if self.autonomy != "auto":
            d["autonomy"] = self.autonomy
        return d


class _SimState:
    __slots__ = ("si", "entry_pose", "entry_speed", "fpose", "last_v")

    def __init__(self, a: Actor):
        self.si = 0
        self.entry_pose = a.start
        self.entry_speed = 0.0
        if a.maneuvers and isinstance(a.maneuvers[0], Maneuver):
            self.entry_speed = 0.0  # maneuvers carry their own v0
        self.fpose = a.start
        self.last_v = 0.0


@dataclass
class Scenario:
    map: MapConfig
    actors: List[Actor]
    pixels_per_meter: float = 6.0

    @property
    def period(self) -> float:
        return max([a.total for a in self.actors] + [1e-6])

    # ---- fixed-step simulation with cached trajectories (D3) ---- #
    def _step(self, a: Actor, st: _SimState, T: float,
              snap: Dict[str, Tuple[float, float, float, float]]
              ) -> Tuple[Pose, float]:
        segs = a.maneuvers
        if not segs:
            return a.start, 0.0
        while st.si < len(segs) and T >= a.cum[st.si + 1] - 1e-9:
            seg = segs[st.si]
            if isinstance(seg, Function):
                end, ex = st.fpose, st.last_v
                seg.sim_exit_speed = ex
            else:
                end = seg.end_pose(st.entry_pose)
                ex = seg.exit_speed()
            st.si += 1
            st.entry_pose, st.entry_speed = end, ex
            if st.si < len(segs) and isinstance(segs[st.si], Function):
                st.fpose, st.last_v = end, ex
        if st.si >= len(segs):
            return st.entry_pose, 0.0
        seg = segs[st.si]
        t = T - a.cum[st.si]
        if isinstance(seg, Maneuver):
            return seg.pose_at(st.entry_pose, t), seg.velocity_at(t)
        # function segment (kinematic unicycle; observes previous tick `snap`)
        ctx = {"self": a.id, "world": snap, "T": T, "t": t,
               "lane_width": self.map.lane_width}
        v, h = eval_outputs(seg, ctx)
        if v is None:
            v = st.entry_speed          # held
        v = clamp(v, 0.0, 60.0)
        if h is None:
            h = st.entry_pose[2]        # held
        x, y = st.fpose[0], st.fpose[1]
        pose = (x, y, h)
        hr = math.radians(h)
        st.fpose = (x + v * math.cos(hr) * DT, y + v * math.sin(hr) * DT, h)
        st.last_v = v
        return pose, v

    def simulate(self) -> None:
        for a in self.actors:
            a.compute_schedule()
            a.traj, a.speeds = [], []
        period = self.period
        n = int(math.ceil(period / DT)) + 1
        states = [_SimState(a) for a in self.actors]
        # initial snapshot (tick 0 observations = initial world state)
        snap: Dict[str, Tuple[float, float, float, float]] = {}
        for a in self.actors:
            v0 = 0.0
            if a.maneuvers and isinstance(a.maneuvers[0], Maneuver):
                v0 = a.maneuvers[0].velocity_at(0.0)
            snap[a.id] = (a.start[0], a.start[1], a.start[2], v0)
        for k in range(n):
            T = k * DT
            cur: Dict[str, Tuple[float, float, float, float]] = {}
            for a, st in zip(self.actors, states):
                pose, v = self._step(a, st, T, snap)
                a.traj.append(pose)
                a.speeds.append(v)
                cur[a.id] = (pose[0], pose[1], pose[2], v)
            snap = cur
        for a, st in zip(self.actors, states):
            a.final_pose = st.entry_pose if st.si >= len(a.maneuvers) \
                else (a.traj[-1] if a.traj else a.start)

    def to_dict(self) -> dict:
        return {"map": self.map.to_dict(),
                "render": {"pixels_per_meter": self.pixels_per_meter},
                "actors": [a.to_dict() for a in self.actors]}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_node(nd: dict) -> Node:
    return Node(
        id=str(nd["id"]),
        kind=str(nd.get("kind", "const")),
        pos=tuple(nd.get("pos", [40.0, 40.0])),
        source=str(nd.get("source", "self")),
        value=float(nd.get("value", 1.0)),
        name=nd.get("name"),
        point=tuple(nd["point"]) if nd.get("point") is not None else None,
        which=str(nd.get("which", "global")),
        op=str(nd.get("op", "add")),
        inputs=[str(i) for i in (nd.get("inputs") or [])],
        field=str(nd.get("field", "speed")),
    )


def _parse_segment(md: dict, default_lane_width: float = 3.5) -> Segment:
    if md.get("type") == "function":
        out = md.get("out") or {}
        return Function(
            duration=float(md.get("duration", 4.0)),
            nodes=[_parse_node(nd) for nd in (md.get("nodes") or [])],
            out_speed=(str(out["speed"]) if out.get("speed") else None),
            out_yaw=(str(out["yaw"]) if out.get("yaw") else None),
        )
    curve = md.get("curve", {}) or {}
    return Maneuver(
        type=md["type"],
        duration=float(md.get("duration", 1.0)),
        intercept=float(curve.get("v0", curve.get("intercept", 0.0))),
        slope=float(curve.get("accel", curve.get("slope", 0.0))),
        radius=float(md.get("radius", 5.0)),
        angle=float(md.get("angle", 90.0)),
        lateral_offset=float(md.get("lateral_offset", default_lane_width)),
    )


def load_scenario(path: str) -> Scenario:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    mapcfg = map_from_dict(raw.get("map", {}) or {})
    ppm = float((raw.get("render", {}) or {}).get("pixels_per_meter", 6.0))
    actors: List[Actor] = []
    for ad in raw.get("actors", []):
        st = ad.get("start", {})
        cu = ad.get("cutin")
        bl = ad.get("block")
        cr = ad.get("cruise")
        a = Actor(
            id=str(ad["id"]),
            color=tuple(ad.get("color", [200, 80, 80])),
            length=float(ad.get("length", 4.5)),
            width=float(ad.get("width", 2.0)),
            start=(float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                   float(st.get("heading", 0.0))),
            maneuvers=[_parse_segment(md, mapcfg.lane_width)
                       for md in ad.get("maneuvers", [])],
            cutin={k: float(v) for k, v in cu.items()} if cu else None,
            # `block: {}` means "derive everything from the ego's script"
            block=(({k: float(v) for k, v in bl.items()} or {"along": 0.0})
                   if bl is not None else None),
            cruise=float(cr) if cr is not None else None,
            autonomy=("self" if str(ad.get("autonomy", "auto")).lower()
                      == "self" else "auto"),
        )
        if not a.maneuvers and a.cruise is not None:
            a.maneuvers = cruise_plan(a.start, a.cruise,
                                      duration=NOMINAL_CRUISE_T)
        actors.append(a)
    sc = Scenario(map=mapcfg, actors=actors, pixels_per_meter=ppm)
    sc.simulate()
    resolve_cutins(sc)
    resolve_blocks(sc)
    return sc


# --------------------------------------------------------------------------- #
# Cut-in constraint solver
#
# An actor with a `cutin` spec commits to completing a lane change at a fixed
# *time* `t`, at a pose relative to the ego at that same time:
#     world_target(t) = ego_pose(t) + along * forward + lat * left
# Editing the ego's speed moves where the ego (and therefore the pin) is at
# time `t`, but the cut-off stays "along metres in front of the ego at t".
# --------------------------------------------------------------------------- #
CUTIN_MIN_SPEED = 0.5
CUTIN_MAX_SPEED = 40.0


def _heading_axes(hd_deg: float) -> Tuple[float, float, float, float]:
    h = math.radians(hd_deg)
    fx, fy = math.cos(h), math.sin(h)
    return fx, fy, -math.sin(h), math.cos(h)   # forward, left


def cutin_world_target(ego: Actor, spec: dict) -> Tuple[float, float, float]:
    """World (x, y) and time `t` for an ego-relative cut-in spec (scripted)."""
    t = max(0.0, float(spec["t"]))
    ex, ey, eh = ego.pose_at_time(t)
    fx, fy, nx, ny = _heading_axes(eh)
    along = float(spec.get("along", 1.0))
    lat = float(spec.get("lat", 0.0))
    return ex + along * fx + lat * nx, ey + along * fy + lat * ny, t


def live_cutin_pin(ego_x: float, ego_y: float, ego_theta_rad: float,
                   along: float, lat: float) -> Tuple[float, float]:
    """World pin attached to a *live* ego pose (moves as the ego moves)."""
    fx, fy = math.cos(ego_theta_rad), math.sin(ego_theta_rad)
    nx, ny = -fy, fx
    return ego_x + along * fx + lat * nx, ego_y + along * fy + lat * ny


def cutin_is_merged(actor_pose: Pose, pin_x: float, pin_y: float,
                    ego_theta_rad: float, lat_tol: float = 0.5,
                    along_tol: float = 4.0) -> bool:
    """True when the actor is near the pin in the ego's frame (merge done)."""
    fx, fy = math.cos(ego_theta_rad), math.sin(ego_theta_rad)
    nx, ny = -fy, fx
    dx, dy = actor_pose[0] - pin_x, actor_pose[1] - pin_y
    return abs(dx * nx + dy * ny) < lat_tol and abs(dx * fx + dy * fy) < along_tol


def world_to_ego_offset(ego_pose: Pose, wx: float, wy: float
                        ) -> Tuple[float, float]:
    """Project a world point into (along, lat) in the ego's body frame."""
    ex, ey, eh = ego_pose
    fx, fy, nx, ny = _heading_axes(eh)
    dx, dy = wx - ex, wy - ey
    return dx * fx + dy * fy, dx * nx + dy * ny


CUTIN_FEASIBLE_MIN_V = 2.0     # absolute floor for a viable chase speed
CUTIN_FEASIBLE_EGO_FRAC = 0.6  # ...and no slower than this fraction of ego_v:
#                                crawling far below the ego's speed to let the
#                                pin catch up is not a cut-in — drop the role


def live_cutin_required_speed(actor_pose: Pose, ego_x: float, ego_y: float,
                              ego_theta: float, ego_v: float,
                              spec: dict, now: float) -> Tuple[float, float]:
    """(required cruise speed, t_rem) for `actor_pose` to reach the live
    ego-relative pin by the deadline `spec.t` (same math as the chase solver:
    the pin advances at ~ego_v, so v = ego_v + along_error / t_rem)."""
    t_rem = closed_loop_cutin_horizon(spec, now)
    wx, wy = live_cutin_pin(ego_x, ego_y, ego_theta,
                            float(spec.get("along", 1.0)),
                            float(spec.get("lat", 0.0)))
    fx, fy = math.cos(ego_theta), math.sin(ego_theta)
    err = (wx - actor_pose[0]) * fx + (wy - actor_pose[1]) * fy
    return ego_v + err / max(t_rem, 0.4), t_rem


def live_cutin_feasible(actor_pose: Pose, ego_x: float, ego_y: float,
                        ego_theta: float, ego_v: float,
                        spec: dict, now: float) -> bool:
    """Can this actor still make the cut-in?  False when the deadline is (all
    but) gone or the required speed is outside what a car would do — e.g. the
    ego braked hard and the pin fell hopelessly far behind the actor."""
    v_req, t_rem = live_cutin_required_speed(actor_pose, ego_x, ego_y,
                                             ego_theta, ego_v, spec, now)
    if t_rem <= 0.3:
        return False
    lo = max(CUTIN_FEASIBLE_MIN_V, CUTIN_FEASIBLE_EGO_FRAC * ego_v)
    return lo <= v_req <= CUTIN_MAX_SPEED


def closed_loop_cutin_horizon(spec: dict, now: float) -> float:
    """Seconds left until the cut-in deadline `spec.t`.

    Returns <= 0 when the deadline has passed — callers should then abandon
    the cut-in (cruise straight) instead of chasing the pin forever.
    """
    t_cut = float(spec.get("t", now))
    return t_cut - now


NOMINAL_CRUISE_T = 8.0     # scripted-playback length of a generated cruise plan


def cruise_plan(start: Pose, speed: float, duration: float = 30.0
                ) -> List[Maneuver]:
    """Straight cruise used after a successful merge or an abandoned cut-in."""
    return [Maneuver(type="go_straight", duration=float(duration),
                     intercept=max(speed, CUTIN_MIN_SPEED))]


def actor_cruise_speed(a: Actor, default: float = 12.0) -> float:
    """The speed an actor returns to when driving nominally."""
    if a.cruise is not None:
        return float(a.cruise)
    if a.maneuvers and isinstance(a.maneuvers[0], Maneuver):
        return max(float(a.maneuvers[0].intercept), CUTIN_MIN_SPEED)
    return default


def solve_closed_loop_cutin(start: Pose, pin_x: float, pin_y: float,
                            t_rem: float, ego_v: float,
                            lc_duration: float = 2.0,
                            tail: float = 30.0,
                            ref_lat: float = 3.5) -> List[Maneuver]:
    """Closed-loop plan toward a pin glued to a moving ego.

    The pin advances at ~ego_v, so the actor must cover
        along_error + ego_v * t_rem
    in time t_rem → v = ego_v + along_error / t_rem.  That lets the actor
    slow down when it is ahead of the pin and speed up when it is behind,
    instead of locking a constant world point.

    Lane changes use short bursts (~2.5 m/s lateral). Frequent replans would
    otherwise keep restarting a long smoothstep near its flat start and the
    merge would never finish.
    """
    sx, sy, sh = start
    h = math.radians(sh)
    fx, fy = math.cos(h), math.sin(h)
    nx, ny = -fy, fx
    dx, dy = pin_x - sx, pin_y - sy
    along_err = dx * fx + dy * fy
    lat_err = dx * nx + dy * ny
    t_rem = max(t_rem, 0.4)
    v = clamp(ego_v + along_err / t_rem, CUTIN_MIN_SPEED, CUTIN_MAX_SPEED)
    plan: List[Maneuver] = []
    need_lc = abs(lat_err) > 0.15
    # LC burst sized against the ~0.05 s replan cadence: each replan restarts
    # the smoothstep at zero lateral velocity, so only its first slice ever
    # executes.  Early-smoothstep progress is ~3*(tau/lc)^2 * lat, hence
    # lc = sqrt(3*lat*tau/R) yields an effective lateral rate of R ~ 2.5 m/s
    # regardless of how large the remaining offset is.
    lc = (clamp(math.sqrt(0.06 * abs(lat_err)), 0.12, min(0.9, t_rem))
          if need_lc else 0.0)
    # start the lane-change immediately: delaying it until the actor has
    # caught the pin longitudinally burns the deadline (and never opens the
    # conflict the collision-directive needs in order to clear the slot).
    if need_lc:
        plan.append(Maneuver(type="lane_change", duration=lc, intercept=v,
                             lateral_offset=lat_err))
    else:
        plan.append(Maneuver(type="go_straight", duration=min(0.5, t_rem),
                             intercept=v))
    plan.append(Maneuver(type="go_straight", duration=float(tail),
                         intercept=max(v, ego_v)))
    return plan


def solve_cutin_maneuvers(start: Pose, wx: float, wy: float, t_arr: float,
                          lc_duration: float = 2.0, tail: float = 4.0,
                          tail_speed: float = 0.0) -> List[Maneuver]:
    """Constant-speed plan from `start` that finishes a lane change at world
    (wx, wy) exactly `t_arr` seconds from now:
    [go_straight t1] + [lane_change lc] + [go_straight tail].
    The tail cruises at max(plan speed, tail_speed) so a slower merge does
    not park the actor in front of a faster ego."""
    sx, sy, sh = start
    h = math.radians(sh)
    dx, dy = wx - sx, wy - sy
    d_along = dx * math.cos(h) + dy * math.sin(h)
    lat = -dx * math.sin(h) + dy * math.cos(h)
    t_arr = max(t_arr, 0.4)
    lc = clamp(float(lc_duration), 0.3, t_arr)
    t1 = t_arr - lc
    v = clamp(d_along / t_arr, CUTIN_MIN_SPEED, CUTIN_MAX_SPEED)
    plan: List[Maneuver] = []
    if t1 > 0.05:
        plan.append(Maneuver(type="go_straight", duration=t1, intercept=v))
    plan.append(Maneuver(type="lane_change", duration=lc, intercept=v,
                         lateral_offset=lat))
    plan.append(Maneuver(type="go_straight", duration=float(tail),
                         intercept=max(v, tail_speed)))
    return plan


def _same_plan(cur: List[Segment], new: List[Maneuver]) -> bool:
    if len(cur) != len(new):
        return False
    for c, n in zip(cur, new):
        if not isinstance(c, Maneuver) or c.type != n.type:
            return False
        if (abs(c.duration - n.duration) > 1e-3
                or abs(c.intercept - n.intercept) > 1e-3
                or abs(c.lateral_offset - n.lateral_offset) > 1e-3):
            return False
    return True


def resolve_cutins(sc: Scenario, ego_id: str = "0") -> bool:
    """Re-plan every actor carrying a `cutin` spec against the ego's current
    script.  Returns True if any plan changed (scenario is re-simulated)."""
    ego = next((a for a in sc.actors if a.id == ego_id), None)
    targets = [a for a in sc.actors if a.cutin and a is not ego]
    if ego is None or not targets:
        return False
    if not ego.traj:
        sc.simulate()
    changed = False
    for a in targets:
        spec = a.cutin
        if "t" not in spec:
            continue
        wx, wy, t_arr = cutin_world_target(ego, spec)
        k = max(0, min(int(round(t_arr / DT)), len(ego.speeds) - 1))
        v_ego = ego.speeds[k] if ego.speeds else 0.0
        plan = solve_cutin_maneuvers(
            a.start, wx, wy, t_arr,
            lc_duration=float(spec.get("lc_duration", 2.0)),
            tail=float(spec.get("tail", 4.0)),
            tail_speed=v_ego)
        if not _same_plan(a.maneuvers, plan):
            a.maneuvers = plan
            changed = True
    if changed:
        sc.simulate()
    return changed


# --------------------------------------------------------------------------- #
# Block-cut-in solver
#
# The ego wants to change lanes; the blocker denies the gap.  A `block` spec
# {"t","duration","along","lat"} makes its holder speed up so that from time
# `t` through `t+duration` it holds a station `along` metres ahead / `lat`
# metres lateral of the ego — `lat` being the ego's target lane — so merging
# into that lane is dangerous.  The blocker never steers: being alongside in
# the target lane is the whole block.
# --------------------------------------------------------------------------- #
BLOCK_LEAD = 1.0      # be in position this long before the ego's lane change
BLOCK_TAIL_PAD = 1.0  # keep blocking a little past the lane-change end


def derive_block_spec(ego: Actor, spec: dict) -> bool:
    """Fill missing t/duration/lat of a block spec from the ego's first
    scripted lane_change.  Returns False when they cannot be determined."""
    if all(k in spec for k in ("t", "duration", "lat")):
        return True
    t_acc = 0.0
    for m in ego.maneuvers:
        if isinstance(m, Maneuver) and m.type == "lane_change":
            spec.setdefault("t", max(0.4, t_acc - BLOCK_LEAD))
            spec.setdefault("duration",
                            float(m.duration) + BLOCK_LEAD + BLOCK_TAIL_PAD)
            spec.setdefault("lat", float(m.lateral_offset))
            return True
        t_acc += max(0.0, float(m.duration))
    return False


def block_window(spec: dict) -> Tuple[float, float]:
    """(start, end) of the blocking window on the session clock."""
    t0 = float(spec.get("t", 0.0))
    return t0, t0 + float(spec.get("duration", 3.0))


def block_world_target(ego: Actor, spec: dict) -> Tuple[float, float]:
    """World hold-point of a scripted block at spec time `t`."""
    t0 = max(0.0, float(spec.get("t", 0.0)))
    ex, ey, eh = ego.pose_at_time(t0)
    fx, fy, nx, ny = _heading_axes(eh)
    along = float(spec.get("along", 0.0))
    lat = float(spec.get("lat", 0.0))
    return ex + along * fx + lat * nx, ey + along * fy + lat * ny


def solve_block_maneuvers(start: Pose, ego: Actor, spec: dict,
                          cruise_v: float, tail: float = 8.0
                          ) -> List[Maneuver]:
    """Longitudinal block plan: speed up to arrive alongside the ego when its
    lane change starts, hold that station (ego speed) through the window,
    then return to cruise."""
    t0 = max(0.4, float(spec["t"]))
    dur = max(0.5, float(spec.get("duration", 3.0)))
    px, py = block_world_target(ego, spec)
    sx, sy, sh = start
    h = math.radians(sh)
    d_along = (px - sx) * math.cos(h) + (py - sy) * math.sin(h)
    v1 = clamp(d_along / t0, CUTIN_MIN_SPEED, CUTIN_MAX_SPEED)
    k = max(0, min(int(round(t0 / DT)), len(ego.speeds) - 1))
    v_e = ego.speeds[k] if ego.speeds else v1
    return [Maneuver(type="go_straight", duration=t0, intercept=v1),
            Maneuver(type="go_straight", duration=dur,
                     intercept=max(v_e, CUTIN_MIN_SPEED)),
            Maneuver(type="go_straight", duration=float(tail),
                     intercept=max(cruise_v, CUTIN_MIN_SPEED))]


def resolve_blocks(sc: Scenario, ego_id: str = "0") -> bool:
    """Re-plan every actor carrying a `block` spec against the ego's current
    script.  Returns True if any plan changed (scenario is re-simulated)."""
    ego = next((a for a in sc.actors if a.id == ego_id), None)
    targets = [a for a in sc.actors if a.block and a is not ego]
    if ego is None or not targets:
        return False
    if not ego.traj:
        sc.simulate()
    changed = False
    for a in targets:
        spec = a.block
        if not derive_block_spec(ego, spec):
            continue
        plan = solve_block_maneuvers(a.start, ego, spec,
                                     actor_cruise_speed(a))
        if not _same_plan(a.maneuvers, plan):
            a.maneuvers = plan
            changed = True
    if changed:
        sc.simulate()
    return changed


BLOCK_SAFE_GAP = 8.0  # m — a merge with less longitudinal clearance is unsafe


def ego_lane_gap(blocker_pose: Pose, ex: float, ey: float,
                 lane_width: float) -> Optional[float]:
    """Longitudinal gap to the blocker when the ego's center is inside the
    blocker's lane (measured in the blocker's lane-aligned frame), else None."""
    bx, by, bh = blocker_pose
    h = math.radians(bh)
    lat = -(ex - bx) * math.sin(h) + (ey - by) * math.cos(h)
    if abs(lat) >= 0.5 * lane_width:
        return None
    return abs((ex - bx) * math.cos(h) + (ey - by) * math.sin(h))


def scripted_block_result(ego: Actor, blocker: Actor, spec: dict,
                          lane_width: float) -> Optional[Tuple[bool, float]]:
    """Grade a scripted block.  The scripted ego merges no matter what, so
    success means it never found a safe gap: returns (denied, min_gap) where
    min_gap is the smallest longitudinal clearance the ego has while inside
    the blocker's lane during the window, and denied = min_gap < BLOCK_SAFE_GAP.
    None when the ego never enters the lane in the window."""
    if not ego.traj or not blocker.traj:
        return None
    t0, t_end = block_window(spec)
    gap_min: Optional[float] = None
    t = t0
    while t <= t_end:
        ex, ey, _ = ego.pose_at_time(t)
        g = ego_lane_gap(blocker.pose_at_time(t), ex, ey, lane_width)
        if g is not None:
            gap_min = g if gap_min is None else min(gap_min, g)
        t += 5 * DT
    if gap_min is None:
        return None
    return gap_min < BLOCK_SAFE_GAP, gap_min


def _validate_function(a: Actor, i: int, fn: Function, actor_ids: set) -> None:
    where = f"actor {a.id} segment {i} (function)"
    ids = [n.id for n in fn.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{where}: duplicate node ids")
    by = fn.by_id()
    # acyclicity (three-color DFS over op inputs)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: 0 for nid in by}

    def dfs(nid: str):
        color[nid] = GRAY
        n = by[nid]
        if n.kind == "op":
            for j in n.inputs:
                if j not in by:
                    raise ValueError(f"{where}: node {nid} input {j!r} missing")
                if color[j] == GRAY:
                    raise ValueError(f"{where}: cycle through node {j}")
                if color[j] == WHITE:
                    dfs(j)
        color[nid] = BLACK

    for nid in by:
        if color[nid] == WHITE:
            dfs(nid)
    for n in fn.nodes:
        if n.kind == "op":
            if n.op not in OPS:
                raise ValueError(f"{where}: unknown op {n.op!r}")
            sig, _ = OPS[n.op]
            if len(n.inputs) < len(sig):
                raise ValueError(f"{where}: op node {n.id} ({n.op}) incomplete "
                                 f"({len(n.inputs)}/{len(sig)} inputs)")
            for slot, inp in enumerate(n.inputs[:len(sig)]):
                t = node_value_type(fn, inp)
                st = sig[slot]
                if st != "T" and t is not None and t != st:
                    raise ValueError(f"{where}: op {n.id} ({n.op}) slot {slot} "
                                     f"expects {st}, got {t}")
        elif n.kind == "obs":
            if n.source != "self" and n.source not in actor_ids:
                raise ValueError(f"{where}: node {n.id} observes unknown "
                                 f"actor {n.source!r}")
            if n.field not in ("speed", "pos", "heading"):
                raise ValueError(f"{where}: node {n.id} unknown field {n.field!r}")
        elif n.kind == "map_point":
            if n.name is not None and n.name not in MAP_POINT_NAMES:
                raise ValueError(f"{where}: unknown map point {n.name!r}")
            if n.name is None and n.point is None:
                raise ValueError(f"{where}: map_point node {n.id} has no name/point")
        elif n.kind == "time":
            if n.which not in ("global", "segment"):
                raise ValueError(f"{where}: node {n.id} unknown time {n.which!r}")
        elif n.kind != "const":
            raise ValueError(f"{where}: unknown node kind {n.kind!r}")
    for label, nid in (("OUT:speed", fn.out_speed), ("OUT:yaw", fn.out_yaw)):
        if nid is not None:
            if nid not in by:
                raise ValueError(f"{where}: {label} bound to missing node {nid!r}")
            if node_value_type(fn, nid) != SCALAR:
                raise ValueError(f"{where}: {label} must be bound to a scalar")


def validate_scenario(path: str) -> Scenario:
    """Load and check a scenario against the CURRENT format. Raises on problems."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    for ad in raw.get("actors", []):
        for md in ad.get("maneuvers", []) or []:
            if "length" in md or "slope" in (md.get("curve") or {}):
                raise ValueError("old format ('length'/'slope'); regenerate this file")
    sc = load_scenario(path)
    if not sc.actors:
        raise ValueError("scenario has no actors")
    ids = [a.id for a in sc.actors]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate actor ids: {ids}")
    actor_ids = set(ids)
    for a in sc.actors:
        for i, m in enumerate(a.maneuvers):
            if m.duration <= 0:
                raise ValueError(f"actor {a.id} segment {i}: duration must be > 0")
            if isinstance(m, Function):
                _validate_function(a, i, m, actor_ids)
            elif m.type not in ALL_MANEUVER_TYPES:
                raise ValueError(f"actor {a.id} segment {i}: unknown type {m.type!r}")
    return sc


# --------------------------------------------------------------------------- #
# Persistence: edit log + versioned save + provenance graph
# --------------------------------------------------------------------------- #
class Persistence:
    def __init__(self, scenarios_dir: str, loaded_file: Optional[str]):
        self.dir = scenarios_dir
        os.makedirs(self.dir, exist_ok=True)
        self.prov_path = os.path.join(self.dir, "provenance.yaml")
        self.hist_path = os.path.join(self.dir, "edit_history.yaml")
        self.versions: List[dict] = self._load_provenance()
        self.base_version = self._resolve_base(loaded_file)

    def _load_provenance(self) -> List[dict]:
        if os.path.exists(self.prov_path):
            data = yaml.safe_load(open(self.prov_path)) or {}
            return list(data.get("versions", []))
        return []

    def _write_provenance(self) -> None:
        with open(self.prov_path, "w") as f:
            yaml.safe_dump({"versions": self.versions}, f, sort_keys=False)

    def _next_number(self) -> int:
        return (max([v["version"] for v in self.versions]) + 1) if self.versions else 1

    def _resolve_base(self, loaded_file: Optional[str]) -> Optional[int]:
        if loaded_file:
            base = os.path.basename(loaded_file)
            for v in self.versions:
                if v.get("file") == base:
                    return v["version"]
            n = self._next_number()
            self.versions.append({"version": n, "file": base, "parent": None,
                                  "created": datetime.now().isoformat(timespec="seconds")})
            self._write_provenance()
            return n
        return None

    def log_edit(self, actor_id: str, mi: int, mtype: str,
                 param: str, old: float, new: float) -> None:
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "base_version": self.base_version, "actor_id": actor_id,
                 "maneuver_index": mi, "maneuver_type": mtype,
                 "parameter": param, "old_value": round(old, 4),
                 "new_value": round(new, 4)}
        with open(self.hist_path, "a") as f:
            f.write(yaml.safe_dump([entry], sort_keys=False))

    def log_structural(self, action: str, actor_id: str, **extra) -> None:
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "base_version": self.base_version, "action": action,
                 "actor_id": actor_id}
        entry.update(extra)
        with open(self.hist_path, "a") as f:
            f.write(yaml.safe_dump([entry], sort_keys=False))

    def save_version(self, scenario: Scenario) -> str:
        n = self._next_number()
        fname = f"scenario_v{n}.yaml"
        fpath = os.path.join(self.dir, fname)
        with open(fpath, "w") as f:
            yaml.safe_dump(scenario.to_dict(), f, sort_keys=False)
        self.versions.append({"version": n, "file": fname, "parent": self.base_version,
                              "created": datetime.now().isoformat(timespec="seconds")})
        self._write_provenance()
        self.base_version = n
        return fname


# --------------------------------------------------------------------------- #
# GUI (pygame)
# --------------------------------------------------------------------------- #
def run_gui(scenario: Scenario, persistence: Optional[Persistence],
            capture: Optional[str] = None, fps: int = 30, loops: int = 1,
            snapshot: Optional[str] = None, select_id: Optional[str] = None,
            at_time: float = 0.0, auto_drive: bool = False,
            on_frame: Optional[callable] = None) -> None:
    """Interactive editor; `capture` records an MP4 headlessly; `snapshot`
    renders one paused frame (optionally with an actor selected) to a PNG.

    Experiment hooks (used by experiment.py, no effect on normal use):
      * auto_drive — enter Drive mode immediately on startup.
      * on_frame(snapshot) — called every driving frame with a dict of the
        live state (clock, ego pose+speed, per-actor displayed poses, cut-in /
        block holders + outcomes).  Return False to end the run."""
    import pygame

    import cutin_orchestrator as co   # role casting: scoring + panel drawing

    pygame.init()
    WIDTH = 1100
    TOPBAR_H = 56
    CANVAS_H = 540          # bird's-eye view region (D8: smaller than v0)
    SUBWIN_H = 400          # segment editor, stacked BELOW the BEV (D8: bigger)
    HEIGHT = TOPBAR_H + CANVAS_H + SUBWIN_H
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Scenario Editor v1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 16)
    font_sm = pygame.font.SysFont("consolas,menlo,monospace", 13)
    font_big = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)

    # colors
    C_GRASS = (32, 44, 34)
    C_ROAD = (60, 60, 66)
    C_LINE = (220, 210, 120)
    C_EDGE = (200, 200, 200)
    C_BAR = (24, 26, 32)
    C_BTN = (54, 58, 70)
    C_BTN_HL = (80, 120, 200)
    C_TEXT = (230, 230, 235)
    C_PANEL = (30, 32, 40)
    C_SEL = (255, 220, 40)
    C_AXIS = (150, 150, 160)
    C_CURVE = (120, 200, 255)
    C_BAD = (235, 90, 90)
    C_WIRE = (140, 190, 240)
    NODE_COLORS = {"obs": (46, 60, 88), "const": (52, 52, 62),
                   "map_point": (44, 68, 52), "time": (64, 52, 76),
                   "op": (78, 62, 42)}
    NODE_W, NODE_H = 122, 34

    ppm = scenario.pixels_per_meter
    canvas_cx = WIDTH // 2
    canvas_cy = TOPBAR_H + CANVAS_H // 2

    def w2s(wx: float, wy: float) -> Tuple[int, int]:
        return int(canvas_cx + wx * ppm), int(canvas_cy - wy * ppm)

    def s2w(sx: float, sy: float) -> Tuple[float, float]:
        return (sx - canvas_cx) / ppm, (canvas_cy - sy) / ppm

    # ---- state ----
    playing = True
    T = 0.0
    drive_mode = False
    live_ego: Optional[dict] = None      # {x,y,theta_rad,v} while Drive is on
    drive_profile: Optional[List[float]] = None  # scripted ego v(t) to follow
    #                                      hands-off in Drive; keys override
    cutin_committed = False
    cutin_outcome: Optional[str] = None  # "merged" | "abandoned" | None
    block_committed = False
    # live/final block status: "breached" while ego is in the target lane
    # (clears back to None / "feasible" if ego returns before the window ends);
    # "blocked" | "breached" once committed at window end
    block_outcome: Optional[str] = None
    cutin_phase = 0.0                    # time into replanned actors' current plan
    # canonical spawn poses — closed-loop drive mutates Actor.start, so Reset
    # / re-enter Drive must restore from this snapshot (updated on spawn edits)
    spawn_poses: Dict[str, Pose] = {a.id: a.start for a in scenario.actors}
    live_rebase: set = set()             # nominal actors the collision
    #                                      directive has taken over (drive clock)
    selected: Optional[int] = None       # actor index
    man_index = 0                        # segment index within selected actor
    focus_field: Optional[str] = None    # field name | 'time' | 'nv:<node id>'
    edit_buffer = ""
    dragging: Optional[str] = None       # 'left'|'right'|'spawn'|'rotate'
    #                                      |'wire'|'node'|'nodepress'
    drag_grab: Optional[Tuple[float, float]] = None
    drag_orig: Optional[Pose] = None
    cutin_drag: Optional[int] = None     # actor index whose pin is dragged
    cutin_orig: Optional[dict] = None    # spec snapshot for edit logging
    status_msg = ""
    status_until = 0.0
    # bicycle-model constants (same feel as drive.py)
    EGO_WHEELBASE = 2.8
    EGO_A_THROTTLE = 5.0
    EGO_A_BRAKE = 9.0
    EGO_V_MAX = 18.0
    EGO_DELTA_MAX = math.radians(32)
    EGO_DRAG = 1.0
    # function-editor state
    sel_node: Optional[str] = None
    wire_from: Optional[str] = None
    node_press: Optional[Tuple[int, int]] = None
    node_orig: Optional[Tuple[float, float]] = None
    last_canvas_click = (0, (0, 0))      # (ms, pos) for double-click detection
    mouse_pos = (0, 0)
    # dropdown widget state: {'rect','items':[(label,payload)],'cb','scroll'}
    dropdown: Optional[dict] = None
    # orchestrator-panel autonomy buttons: actor id -> rect (set each frame)
    orch_buttons: Dict[str, "pygame.Rect"] = {}
    DD_ITEM_H = 22
    DD_MAX_VIS = 12

    # ---- top bar rects ----
    btn_play = pygame.Rect(WIDTH // 2 - 55, 10, 110, 36)
    btn_reset = pygame.Rect(WIDTH // 2 - 190, 10, 110, 36)
    btn_save = pygame.Rect(WIDTH // 2 + 80, 10, 100, 36)
    btn_drive = pygame.Rect(WIDTH // 2 + 190, 10, 100, 36)
    btn_add = pygame.Rect(WIDTH - 240, 10, 105, 36)
    btn_del = pygame.Rect(WIDTH - 130, 10, 105, 36)
    time_field = pygame.Rect(52, 14, 84, 28)

    ADD_PRESETS = [(1.75, -58, 90), (58, 1.75, 180), (-58, -1.75, 0), (-1.75, 58, 270)]
    ADD_PALETTE = [(90, 190, 110), (210, 70, 60), (60, 120, 210),
                   (200, 160, 60), (160, 90, 200), (80, 200, 200)]

    # ---- geometry helpers ----
    def subwin_rect() -> pygame.Rect:
        return pygame.Rect(0, HEIGHT - SUBWIN_H, WIDTH, SUBWIN_H)

    def plot_rect() -> pygame.Rect:
        sw = subwin_rect()
        return pygame.Rect(sw.x + 55, sw.y + 56, 470, SUBWIN_H - 96)

    def node_canvas_rect() -> pygame.Rect:
        sw = subwin_rect()
        return pygame.Rect(sw.x + 8, sw.y + 48, sw.width - 16, sw.height - 76)

    def dur_field_rect() -> pygame.Rect:
        sw = subwin_rect()
        return pygame.Rect(sw.x + 380, sw.y + 12, 76, 26)

    def field_specs(m: Maneuver):
        if m.type == "go_straight":
            return [("intercept", "speed"), ("duration", "duration")]
        if m.type == "lane_change":
            return [("intercept", "speed"), ("lateral_offset", "lat_off"),
                    ("duration", "duration")]
        if m.type in ("accelerate", "decelerate"):
            return [("intercept", "v0"), ("slope", "accel"), ("duration", "duration")]
        if m.type in ("turn_left", "turn_right"):
            return [("radius", "radius"), ("angle", "angle"), ("duration", "duration")]
        return [("duration", "duration")]

    def field_rects(m: Maneuver) -> dict:
        sw = subwin_rect()
        x = sw.x + 640
        return {name: pygame.Rect(x, sw.y + 56 + i * 42, 90, 26)
                for i, (name, _lbl) in enumerate(field_specs(m))}

    def field_label(m: Maneuver, name: str) -> str:
        return next((lbl for n, lbl in field_specs(m) if n == name), name)

    def header_buttons() -> dict:
        sw = subwin_rect()
        y = sw.y + 12
        return {"type": pygame.Rect(sw.right - 420, y, 160, 26),
                "prev": pygame.Rect(sw.right - 254, y, 48, 26),
                "next": pygame.Rect(sw.right - 200, y, 48, 26),
                "add_mvr": pygame.Rect(sw.right - 146, y, 64, 26),
                "del_mvr": pygame.Rect(sw.right - 76, y, 64, 26)}

    def cur_segment() -> Optional[Segment]:
        if selected is None:
            return None
        a = scenario.actors[selected]
        if not a.maneuvers:
            return None
        return a.maneuvers[min(man_index, len(a.maneuvers) - 1)]

    def fe_active() -> bool:
        return (not playing and selected is not None
                and isinstance(cur_segment(), Function))

    def set_status(msg: str) -> None:
        nonlocal status_msg, status_until
        status_msg = msg
        status_until = T + 3.0

    def log_struct(action: str, **extra) -> None:
        if persistence is not None and selected is not None:
            persistence.log_structural(action, scenario.actors[selected].id, **extra)

    def maybe_resolve_cutins() -> None:
        """Re-cast the cut-in / block roles, then re-solve plans against the
        (possibly edited) ego script."""
        nonlocal man_index
        if drive_mode:
            return
        recast = cast_cutin_roles()
        recast = cast_block_roles() or recast
        changed = resolve_cutins(scenario)
        changed = resolve_blocks(scenario) or changed
        if (changed or recast) and selected is not None:
            n = len(scenario.actors[selected].maneuvers)
            if n:
                man_index = min(man_index, n - 1)
        if recast:
            scenario.simulate()

    def ego_actor() -> Optional[Actor]:
        return next((a for a in scenario.actors if a.id == "0"), None)

    def cutin_holder() -> Optional[Actor]:
        """The actor currently cast as the cut-in (owns the spec)."""
        return next((a for a in scenario.actors
                     if a.id != "0" and a.cutin), None)

    def block_holder() -> Optional[Actor]:
        """The actor currently cast as the block-cut-in (owns the spec)."""
        return next((a for a in scenario.actors
                     if a.id != "0" and a.block), None)

    def cast_cutin_roles() -> bool:
        """Role casting: hand the cut-in spec to the best-placed actor.

        Scores every non-ego actor (adjacent lane + proximity to a station
        ahead of the ego); the winner takes the `cutin` spec, the previous
        holder goes back to nominal straight cruising.  Hysteresis (1.25x)
        keeps the role from flapping.  Returns True if the role moved.
        """
        ego = ego_actor()
        holder = cutin_holder()
        others = [a for a in scenario.actors if a.id != "0"]
        if ego is None or holder is None or len(others) < 2:
            return False
        if holder.autonomy == "self":
            return False   # user-owned intent — the orchestrator hands off
        # only fully-autonomous actors without another role can be conscripted
        cands = [a for a in others
                 if a is not holder and a.autonomy != "self" and not a.block]
        if not cands:
            return False
        lw = scenario.map.lane_width
        if drive_mode and live_ego is not None:
            e = live_ego
            ego_pose = (e["x"], e["y"], math.degrees(e["theta"]))

            def pose_of(a: Actor) -> Pose:
                # holder plans are rebased every replan tick (clock =
                # cutin_phase); nominal actors still run their original
                # script on the session clock T — same as the renderer
                if a.cutin or a.block:
                    return a.pose_at_time(cutin_phase) if a.traj else a.start
                return a.pose_at_time(T % scenario.period)

            # Recast only when the holder can no longer make the pin by the
            # deadline (e.g. the ego braked hard and the pin fell hopelessly
            # far behind the holder). Mid-chase the holder drifts toward the
            # ego's lane, which tanks its *candidate* score — that is
            # progress, not failure — so stickiness is on feasibility, not
            # score.
            spec = holder.cutin
            if live_cutin_feasible(pose_of(holder), e["x"], e["y"],
                                   e["theta"], e["v"], spec, T):
                return False
            feas = [a for a in cands
                    if live_cutin_feasible(pose_of(a), e["x"], e["y"],
                                           e["theta"], e["v"], spec, T)]
            if not feas:
                return False   # nobody can do it; holder abandons at deadline
            scores = {a.id: co.score_cutin_candidate(pose_of(a), ego_pose, lw)
                      for a in feas}
            best = max(feas, key=lambda a: scores[a.id])
            v_req, _ = live_cutin_required_speed(
                pose_of(holder), e["x"], e["y"], e["theta"], e["v"], spec, T)
            set_status(f"orchestrator: actor {holder.id} can't make the pin "
                       f"(needs {v_req:.1f} m/s) — recast to {best.id}")
        else:
            ego_pose = ego.start

            def pose_of(a: Actor) -> Pose:
                return a.start

            scores = {a.id: co.score_cutin_candidate(pose_of(a), ego_pose, lw)
                      for a in cands + [holder]}
            best = max(cands, key=lambda a: scores[a.id])
            if scores[best.id] <= 1.25 * scores[holder.id]:
                return False
            set_status(f"orchestrator: cut-in recast {holder.id} -> {best.id} "
                       f"(score {scores[best.id]:.2f} vs {scores[holder.id]:.2f})")
        # move the spec; the old holder goes back to nominal cruising
        if drive_mode:
            # capture current poses BEFORE the swap (pose_of branches on
            # who owns the spec)
            hp, bp = pose_of(holder), pose_of(best)
        best.cutin, holder.cutin = holder.cutin, None
        v = actor_cruise_speed(holder)
        if drive_mode:
            holder.start = (hp[0], hp[1], holder.start[2])
            holder.maneuvers = cruise_plan(holder.start, v, duration=30.0)
            # rebase the new holder to "now" so its first replan doesn't
            # read a pose from its stale spawn-anchored script
            best.start = bp
            best.maneuvers = cruise_plan(bp, actor_cruise_speed(best),
                                         duration=30.0)
            scenario.simulate()
        else:
            dur = max(8.0, float(getattr(ego, "total", 0.0) or 8.0))
            holder.maneuvers = cruise_plan(holder.start, v, duration=dur)
        return True

    def cast_block_roles() -> bool:
        """Role casting for the block-cut-in: hand the `block` spec to the
        fully-autonomous actor best placed to deny the ego's target-lane gap
        (in the target lane, slightly behind the ego, so speeding up closes
        the gap).  Same hysteresis / feasibility pattern as the cut-in."""
        ego = ego_actor()
        holder = block_holder()
        others = [a for a in scenario.actors if a.id != "0"]
        if ego is None or holder is None or len(others) < 2:
            return False
        if holder.autonomy == "self":
            return False   # user-owned intent — the orchestrator hands off
        cands = [a for a in others
                 if a is not holder and a.autonomy != "self" and not a.cutin]
        if not cands:
            return False
        spec = holder.block
        if not derive_block_spec(ego, spec):
            return False
        lw = scenario.map.lane_width
        target_lat = float(spec.get("lat", 0.0))
        if drive_mode and live_ego is not None:
            e = live_ego
            ego_pose = (e["x"], e["y"], math.degrees(e["theta"]))

            def pose_of(a: Actor) -> Pose:
                if a.cutin or a.block:
                    return a.pose_at_time(cutin_phase) if a.traj else a.start
                return a.pose_at_time(T % scenario.period)

            # sticky on feasibility: can the holder still reach / hold the
            # station before the blocking window closes? (same chase math
            # as the cut-in, deadline = end of window)
            _, t_end = block_window(spec)
            pseudo = {"t": t_end, "along": spec.get("along", 0.0),
                      "lat": target_lat}
            if live_cutin_feasible(pose_of(holder), e["x"], e["y"],
                                   e["theta"], e["v"], pseudo, T):
                return False
            feas = [a for a in cands
                    if live_cutin_feasible(pose_of(a), e["x"], e["y"],
                                           e["theta"], e["v"], pseudo, T)]
            if not feas:
                return False   # nobody can block; holder releases at window end
            scores = {a.id: co.score_block_candidate(pose_of(a), ego_pose,
                                                     lw, target_lat)
                      for a in feas}
            best = max(feas, key=lambda a: scores[a.id])
            set_status(f"orchestrator: actor {holder.id} can't hold the "
                       f"block — recast to {best.id}")
        else:
            ego_pose = ego.start

            def pose_of(a: Actor) -> Pose:
                return a.start

            scores = {a.id: co.score_block_candidate(pose_of(a), ego_pose,
                                                     lw, target_lat)
                      for a in cands + [holder]}
            best = max(cands, key=lambda a: scores[a.id])
            if scores[best.id] <= 1.25 * scores[holder.id]:
                return False
            set_status(f"orchestrator: block recast {holder.id} -> {best.id} "
                       f"(score {scores[best.id]:.2f} vs "
                       f"{scores[holder.id]:.2f})")
        if drive_mode:
            hp, bp = pose_of(holder), pose_of(best)
        best.block, holder.block = holder.block, None
        v = actor_cruise_speed(holder)
        if drive_mode:
            holder.start = (hp[0], hp[1], holder.start[2])
            holder.maneuvers = cruise_plan(holder.start, v, duration=30.0)
            best.start = bp
            best.maneuvers = cruise_plan(bp, actor_cruise_speed(best),
                                         duration=30.0)
            scenario.simulate()
        else:
            dur = max(8.0, float(getattr(ego, "total", 0.0) or 8.0))
            holder.maneuvers = cruise_plan(holder.start, v, duration=dur)
        return True

    def make_autonomy_cb(actor: Actor):
        def cb(mode: str) -> None:
            if actor.autonomy == mode:
                return
            actor.autonomy = mode
            label = ("fully autonomous" if mode == "auto"
                     else "fully self-governed")
            set_status(f"actor {actor.id}: {label}")
            if persistence:
                persistence.log_structural("set_autonomy", actor.id, mode=mode)
        return cb

    def remember_spawn(a: Actor) -> None:
        spawn_poses[a.id] = a.start

    def restore_spawns() -> None:
        """Put every actor back at its canonical spawn (pre-drive) pose."""
        for a in scenario.actors:
            if a.id in spawn_poses:
                a.start = spawn_poses[a.id]

    def enter_drive_mode() -> None:
        nonlocal drive_mode, playing, T, live_ego, cutin_committed, selected
        nonlocal cutin_phase, cutin_outcome, drive_profile
        nonlocal block_committed, block_outcome, live_rebase
        restore_spawns()
        ego = ego_actor()
        if ego is None:
            set_status("no ego actor (id 0) to drive")
            return
        T = 0.0
        cutin_phase = 0.0
        v0 = 0.0
        if ego.maneuvers and isinstance(ego.maneuvers[0], Maneuver):
            v0 = ego.maneuvers[0].velocity_at(0.0)
        live_ego = {"x": ego.start[0], "y": ego.start[1],
                    "theta": math.radians(ego.start[2]), "v": max(v0, 0.0)}
        cutin_committed = False
        cutin_outcome = None
        block_committed = False
        block_outcome = None
        live_rebase = set()
        drive_mode = True
        playing = True
        selected = None
        resolve_cutins(scenario)   # seed plans; closed-loop takes over while driving
        resolve_blocks(scenario)
        # hands-off the live ego follows the scripted speed profile (so a
        # scripted sudden slowdown plays out even in Drive); keys override
        scenario.simulate()
        drive_profile = list(ego.speeds) if ego.speeds else None
        set_status("DRIVE ON — ego follows its script; WASD/arrows override")

    def exit_drive_mode() -> None:
        nonlocal drive_mode, playing, T, live_ego, cutin_committed, cutin_phase
        nonlocal cutin_outcome, drive_profile, block_committed, block_outcome
        nonlocal live_rebase
        drive_mode = False
        live_ego = None
        drive_profile = None
        cutin_committed = False
        cutin_outcome = None
        block_committed = False
        block_outcome = None
        live_rebase = set()
        cutin_phase = 0.0
        playing = False
        T = 0.0
        restore_spawns()
        resolve_cutins(scenario)
        resolve_blocks(scenario)
        set_status("DRIVE OFF — back to scripted playback")

    def toggle_drive_mode() -> None:
        if drive_mode:
            exit_drive_mode()
        else:
            enter_drive_mode()

    def integrate_live_ego(throttle: float, steer: float, dt: float) -> None:
        e = live_ego
        if e is None:
            return
        if throttle > 0:
            a = EGO_A_THROTTLE * throttle
        elif throttle < 0:
            a = EGO_A_BRAKE * throttle
        else:
            a = -EGO_DRAG if e["v"] > 0 else 0.0
        e["v"] = max(0.0, min(EGO_V_MAX, e["v"] + a * dt))
        delta = EGO_DELTA_MAX * steer
        e["theta"] += (e["v"] / EGO_WHEELBASE) * math.tan(delta) * dt
        e["x"] += e["v"] * math.cos(e["theta"]) * dt
        e["y"] += e["v"] * math.sin(e["theta"]) * dt

    def live_resolve_cutins() -> None:
        """Closed-loop orchestration while driving: the cut-in holder chases
        its pin until merge / deadline, and the block holder speeds up to
        hold a station in the ego's target lane until its window closes."""
        nonlocal cutin_committed, cutin_phase, man_index, cutin_outcome
        nonlocal block_committed, block_outcome, live_rebase
        if live_ego is None:
            return
        e = live_ego
        changed = False
        replanned: set = set()
        if not cutin_committed:
            changed = cast_cutin_roles() or changed   # roles may move mid-drive
        if not block_committed:
            changed = cast_block_roles() or changed
        for a in scenario.actors:
            if a.id == "0":
                continue
            if a.cutin and not cutin_committed:
                spec = a.cutin
                if not a.traj:
                    scenario.simulate()
                pose = a.pose_at_time(cutin_phase)
                along = float(spec.get("along", 1.0))
                lat = float(spec.get("lat", 0.0))
                wx, wy = live_cutin_pin(e["x"], e["y"], e["theta"], along, lat)
                # keep the actor's own plan-frame heading (lane-aligned):
                # copying the ego's live heading would make the actor steer
                # whenever the user steers.  a.start[2] is stable across
                # replans (unlike pose[2], which carries the lane-change's
                # temporary yaw mid-maneuver).
                start = (pose[0], pose[1], a.start[2])
                replanned.add(a.id)
                if cutin_is_merged(pose, wx, wy, e["theta"]):
                    a.start = start
                    a.maneuvers = cruise_plan(start, e["v"])
                    cutin_committed = True
                    cutin_outcome = "merged"
                    changed = True
                    set_status("cut-in merged — actor matching ego")
                    continue
                t_rem = closed_loop_cutin_horizon(spec, T)
                if t_rem <= 0.0:
                    # deadline passed without a merge — drop the cut-in and
                    # return to the actor's own cruise speed, straight ahead
                    v_nom = actor_cruise_speed(a)
                    a.start = start
                    a.maneuvers = cruise_plan(start, v_nom)
                    cutin_committed = True
                    cutin_outcome = "abandoned"
                    changed = True
                    set_status(f"cut-in abandoned — past "
                               f"t={float(spec['t']):.2f}s, "
                               f"back to cruise {v_nom:.1f} m/s")
                    continue
                a.start = start
                a.maneuvers = solve_closed_loop_cutin(
                    start, wx, wy, t_rem, e["v"],
                    lc_duration=float(spec.get("lc_duration", 2.0)),
                    tail=30.0)
                changed = True
            elif a.block and not block_committed:
                spec = a.block
                if "t" not in spec or "lat" not in spec:
                    continue   # underivable without an ego lane-change script
                if not a.traj:
                    scenario.simulate()
                pose = a.pose_at_time(cutin_phase)
                # blocker never steers: hold its own lane-aligned heading,
                # never the ego's (see the cut-in branch note above)
                start = (pose[0], pose[1], a.start[2])
                t0, t_end = block_window(spec)
                replanned.add(a.id)
                gap = ego_lane_gap(start, e["x"], e["y"],
                                   scenario.map.lane_width)
                # live re-eval: ego in the target lane → BREACHED pin; if it
                # lane-changes back before the window ends, flip to FEASIBLE
                # and keep blocking (a temporary visit must not abort the role)
                if gap is not None:
                    if block_outcome != "breached":
                        set_status(f"block BREACHED — ego in target lane "
                                   f"(gap {gap:.1f} m); still holding until "
                                   f"t={t_end:.1f}s")
                    block_outcome = "breached"
                else:
                    if block_outcome == "breached":
                        set_status("block FEASIBLE again — ego left the "
                                   "target lane; still holding")
                    block_outcome = None
                if T > t_end:
                    # finalize from the *current* occupancy: still in lane
                    # → breached for good; otherwise the merge was denied
                    v_nom = actor_cruise_speed(a)
                    a.start = start
                    a.maneuvers = cruise_plan(start, v_nom)
                    block_committed = True
                    if gap is not None:
                        block_outcome = "breached"
                        set_status(f"block FAILED — ego still in lane at "
                                   f"t={t_end:.1f}s (gap {gap:.1f} m); "
                                   f"back to cruise {v_nom:.1f} m/s")
                    else:
                        block_outcome = "blocked"
                        set_status(f"block SUCCEEDED — merge denied through "
                                   f"t={t_end:.1f}s; back to cruise "
                                   f"{v_nom:.1f} m/s")
                    changed = True
                    continue
                # chase / hold the station glued to the live ego: close the
                # along-error by the window start, then track it tightly
                wx, wy = live_cutin_pin(e["x"], e["y"], e["theta"],
                                        float(spec.get("along", 0.0)),
                                        float(spec.get("lat", 0.0)))
                fx, fy = math.cos(e["theta"]), math.sin(e["theta"])
                err = (wx - pose[0]) * fx + (wy - pose[1]) * fy
                hz = max(t0 - T, 0.8)
                v = clamp(e["v"] + err / hz, CUTIN_MIN_SPEED, CUTIN_MAX_SPEED)
                a.start = start
                a.maneuvers = cruise_plan(start, v)
                changed = True
        if changed:
            # everyone rendered on the replan clock but NOT replanned this
            # tick (e.g. a committed holder already cruising) must be rebased
            # before the phase resets, or they would snap back in time
            for a in scenario.actors:
                if (a.id != "0" and (a.cutin or a.block or a.id in live_rebase)
                        and a.id not in replanned):
                    p = a.pose_at_time(cutin_phase)
                    v = (a.maneuvers[0].intercept
                         if a.maneuvers and isinstance(a.maneuvers[0], Maneuver)
                         else actor_cruise_speed(a))
                    a.start = p
                    a.maneuvers = cruise_plan(p, v)
            cutin_phase = 0.0
            scenario.simulate()
            if selected is not None:
                n = len(scenario.actors[selected].maneuvers)
                if n:
                    man_index = min(man_index, n - 1)

        # collision directive: predicted body overlap → lower-priority actor
        # yields.  Owners (cut-in / block) outrank nominal traffic; among
        # owners the panel score ("probability") of the held action decides.
        # Example: cut-in owner blocked by a car in the target lane → that
        # car is sped forward to open a safe gap.
        def pose_at(a: Actor, t: float) -> Pose:
            if a.cutin or a.block or a.id in live_rebase:
                return a.pose_at_time(cutin_phase + t)
            return a.pose_at_time((T + t) % scenario.period)

        ego_a = ego_actor()
        scores: Dict[str, Dict[str, float]] = {}
        if ego_a is not None:
            ego_pose = (e["x"], e["y"], math.degrees(e["theta"]))
            lw = scenario.map.lane_width
            bspec = next((a.block for a in scenario.actors if a.block), None)
            target_lat = (float(bspec["lat"])
                          if bspec and "lat" in bspec else None)
            for a in scenario.actors:
                if a.id == "0":
                    continue
                p = pose_at(a, 0.0)
                d = {co.ROLE_CUTIN: co.score_cutin_candidate(p, ego_pose, lw)}
                if target_lat is not None:
                    d[co.ROLE_BLOCK] = co.score_block_candidate(
                        p, ego_pose, lw, target_lat)
                scores[a.id] = d
        yields = co.resolve_actor_collisions(scenario.actors, pose_at, scores)
        if yields:
            # only the interferer is rewritten — the action owner keeps the
            # plan just solved (lane-change / block hold).  After `changed`
            # the phase is already 0 and starts are current; otherwise rebase
            # the interferer from the pose it occupies *now*.
            for inter, v, priv, t_hit in yields:
                p = pose_at(inter, 0.0)
                hd = inter.start[2] if len(inter.start) > 2 else p[2]
                inter.start = (p[0], p[1], hd)
                inter.maneuvers = cruise_plan(inter.start, v)
                live_rebase.add(inter.id)
            if not changed:
                for a in scenario.actors:
                    if a.id == "0" or a.id in {y[0].id for y in yields}:
                        continue
                    if a.cutin:
                        continue   # never convert an in-progress cut-in to cruise
                    if a.block or a.id in live_rebase:
                        p = pose_at(a, 0.0)
                        v = (a.maneuvers[0].intercept
                             if a.maneuvers and isinstance(a.maneuvers[0], Maneuver)
                             else actor_cruise_speed(a))
                        a.start = (p[0], p[1], a.start[2] if len(a.start) > 2 else p[2])
                        a.maneuvers = cruise_plan(a.start, v)
                cutin_phase = 0.0
            scenario.simulate()
            set_status(
                f"collision: actor {yields[0][0].id} yields to "
                f"{yields[0][2].id} (hit in {yields[0][3]:.1f}s) → "
                f"{yields[0][1]:.1f} m/s")

    # ---- dropdown widget (all multi-choice selection; nothing cycles) ----
    def open_dropdown(anchor: pygame.Rect, items: List[Tuple[str, object]], cb) -> None:
        nonlocal dropdown
        if not items:
            return
        w = min(300, max(140, max(font_sm.size(lbl)[0] for lbl, _ in items) + 26))
        vis = min(len(items), DD_MAX_VIS)
        h = vis * DD_ITEM_H + 4
        x = min(anchor.x, WIDTH - w - 4)
        y = anchor.bottom + 2
        if y + h > HEIGHT - 4:
            y = max(4, anchor.y - h - 2)
        dropdown = {"rect": pygame.Rect(x, y, w, h), "items": items,
                    "cb": cb, "scroll": 0}

    def dropdown_click(mx: int, my: int) -> None:
        nonlocal dropdown
        dd = dropdown
        dropdown = None
        if dd is None or not dd["rect"].collidepoint(mx, my):
            return
        idx = dd["scroll"] + (my - dd["rect"].y - 2) // DD_ITEM_H
        if 0 <= idx < len(dd["items"]):
            dd["cb"](dd["items"][idx][1])

    def draw_dropdown() -> None:
        if dropdown is None:
            return
        r = dropdown["rect"]
        pygame.draw.rect(screen, (20, 22, 28), r)
        pygame.draw.rect(screen, C_BTN_HL, r, 1)
        items = dropdown["items"]
        sc0 = dropdown["scroll"]
        for row, (lbl, _) in enumerate(items[sc0:sc0 + DD_MAX_VIS]):
            ir = pygame.Rect(r.x + 2, r.y + 2 + row * DD_ITEM_H, r.w - 4, DD_ITEM_H)
            if ir.collidepoint(*mouse_pos):
                pygame.draw.rect(screen, (54, 70, 100), ir)
            screen.blit(font_sm.render(lbl, True, C_TEXT), (ir.x + 8, ir.y + 4))
        if len(items) > DD_MAX_VIS:
            frac0 = sc0 / len(items)
            frac1 = (sc0 + DD_MAX_VIS) / len(items)
            pygame.draw.rect(screen, (90, 94, 105),
                             pygame.Rect(r.right - 5, r.y + int(frac0 * r.h),
                                         3, max(8, int((frac1 - frac0) * r.h))))

    # ---- actor management ----
    def next_actor_id() -> str:
        nums = []
        for a in scenario.actors:
            try:
                nums.append(int(a.id))
            except (ValueError, TypeError):
                pass
        return str(max(nums) + 1) if nums else "0"

    def spawn_pose_for(i: int) -> Pose:
        """Where a freshly added actor goes. Straight maps get a lane slot near
        the south end; intersections keep the four arm presets."""
        m = scenario.map
        if m.kind != "straight":
            return ADD_PRESETS[i % len(ADD_PRESETS)]
        n = max(1, m.num_lanes)
        row = i // n
        y = min(-m.half_length() + 12.0 + 8.0 * row, m.half_length() - 6.0)
        return (m.lane_center_x(i % n), y, 90.0)

    def do_add_actor() -> None:
        nonlocal selected, man_index, sel_node, playing
        i = len(scenario.actors)
        sx, sy, hd = spawn_pose_for(i)
        a = Actor(id=next_actor_id(), color=ADD_PALETTE[i % len(ADD_PALETTE)],
                  length=4.5, width=2.0, start=(sx, sy, hd), cruise=12.0,
                  maneuvers=cruise_plan((sx, sy, hd), 12.0,
                                        duration=NOMINAL_CRUISE_T))
        scenario.actors.append(a)
        remember_spawn(a)
        scenario.simulate()
        if persistence:
            persistence.log_structural("add_actor", a.id)
        selected = len(scenario.actors) - 1
        man_index = 0
        sel_node = None
        playing = False          # so the segment editor opens on the new actor
        set_status(f"added actor {a.id}")

    def do_remove_actor() -> None:
        nonlocal selected, man_index, sel_node
        if selected is None:
            return
        a = scenario.actors.pop(selected)
        spawn_poses.pop(a.id, None)
        scenario.simulate()
        if persistence:
            persistence.log_structural("remove_actor", a.id)
        selected = None
        man_index = 0
        sel_node = None
        set_status(f"removed actor {a.id}")

    def do_add_segment() -> None:
        nonlocal man_index, sel_node
        if selected is None:
            return
        a = scenario.actors[selected]
        v_in = a.maneuvers[man_index].exit_speed() if a.maneuvers else 10.0
        new_m = Maneuver(type="go_straight", duration=2.0,
                         intercept=(v_in if v_in > 0 else 10.0), slope=0.0)
        at = man_index + 1 if a.maneuvers else 0
        a.maneuvers.insert(at, new_m)
        scenario.simulate()
        man_index = at
        sel_node = None
        log_struct("add_maneuver", maneuver_index=at, maneuver_type="go_straight")
        set_status(f"added segment to actor {a.id} at {at}")

    def do_del_segment() -> None:
        nonlocal man_index, sel_node
        if selected is None:
            return
        a = scenario.actors[selected]
        if len(a.maneuvers) <= 1:
            set_status("cannot delete the last segment")
            return
        removed = a.maneuvers.pop(man_index)
        scenario.simulate()
        log_struct("del_maneuver", maneuver_index=man_index,
                   maneuver_type=removed.type)
        man_index = min(man_index, len(a.maneuvers) - 1)
        sel_node = None
        set_status(f"deleted segment from actor {a.id}")

    def do_set_segment_type(new_type: str) -> None:
        nonlocal focus_field, sel_node
        if selected is None:
            return
        a = scenario.actors[selected]
        m = a.maneuvers[man_index]
        old_type = m.type
        if new_type == old_type:
            return
        v_in = a.maneuvers[man_index - 1].exit_speed() if man_index > 0 else 10.0
        if new_type == "function":
            a.maneuvers[man_index] = Function(duration=m.duration)
        else:
            if isinstance(m, Function):
                m = Maneuver(type=new_type, duration=m.duration)
                a.maneuvers[man_index] = m
            m.type = new_type
            if new_type in TURN_TYPES:
                if m.radius <= 0:
                    m.radius = 5.0
                if m.angle <= 0:
                    m.angle = 90.0
            if new_type == "lane_change":
                if abs(m.lateral_offset) < 1e-6:
                    m.lateral_offset = scenario.map.lane_width
                m.intercept = v_in if v_in > 0 else 10.0
                m.slope = 0.0
                if m.duration < 0.5:
                    m.duration = 2.0
            if new_type in LONGITUDINAL_TYPES:
                m.intercept = v_in
                if new_type == "go_straight":
                    m.slope = 0.0
                elif new_type == "accelerate":
                    m.slope = 2.0 if m.slope <= 0 else m.slope
                else:
                    m.slope = -2.0 if m.slope >= 0 else m.slope
            if new_type == "stop":
                m.slope, m.intercept = 0.0, 0.0
        scenario.simulate()
        focus_field = None
        sel_node = None
        log_struct("set_maneuver_type", maneuver_index=man_index,
                   old_type=old_type, new_type=new_type)
        set_status(f"actor {a.id} segment {man_index}: {old_type} -> {new_type}")

    def commit_spawn_edit() -> None:
        if selected is None or drag_orig is None:
            return
        a = scenario.actors[selected]
        remember_spawn(a)
        if a.start != drag_orig and persistence:
            persistence.log_structural(
                "move_actor", a.id,
                old_start={"x": round(drag_orig[0], 3), "y": round(drag_orig[1], 3),
                           "heading": round(drag_orig[2], 3)},
                new_start={"x": round(a.start[0], 3), "y": round(a.start[1], 3),
                           "heading": round(a.start[2], 3)})
            set_status(f"moved actor {a.id} spawn")

    # ---- parameter editing ----
    def apply_param(param: str, new_val: float) -> None:
        a = scenario.actors[selected]
        m = a.maneuvers[man_index]
        old = getattr(m, param)
        if param == "duration":
            new_val = max(0.05, new_val)
        elif param == "radius":
            new_val = max(0.5, new_val)
        elif param == "angle":
            new_val = clamp(new_val, 1.0, 179.0)
        elif param == "intercept":
            new_val = max(0.0, new_val)
        elif param == "lateral_offset":
            new_val = clamp(new_val, -14.0, 14.0)
        setattr(m, param, new_val)
        scenario.simulate()
        if persistence:
            persistence.log_edit(a.id, man_index, m.type, param, old, new_val)
        set_status(f"{a.id}.{m.type}.{param}: {old:.3g} -> {new_val:.3g}")

    def plot_maps(m: Maneuver, pr: pygame.Rect):
        tmax = max(1e-6, m.duration)
        v0, v1 = m.intercept, m.intercept + m.slope * m.duration
        vmax = max(v0, v1, 1.0) * 1.15
        vmin = min(v0, v1, 0.0) - 0.15 * abs(max(v0, v1, 1.0))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        def t2x(t): return pr.x + (t / tmax) * pr.width
        def v2y(v): return pr.bottom - (v - vmin) / (vmax - vmin) * pr.height
        def y2v(y): return vmin + (pr.bottom - y) / pr.height * (vmax - vmin)
        return t2x, v2y, y2v, tmax, vmin, vmax

    # ================= function editor ================= #
    def node_rect(node: Node) -> pygame.Rect:
        c = node_canvas_rect()
        return pygame.Rect(int(c.x + node.pos[0]), int(c.y + node.pos[1]),
                           NODE_W, NODE_H)

    def port_center(node: Node) -> Tuple[int, int]:
        r = node_rect(node)
        return (r.right, r.centery)

    def slot_centers(node: Node) -> List[Tuple[int, int]]:
        r = node_rect(node)
        sig, _ = OPS.get(node.op, ((), SCALAR))
        n = max(1, len(sig))
        return [(r.left, r.y + int(r.h * (i + 1) / (n + 1))) for i in range(n)]

    def swap_badge_rect(node: Node) -> pygame.Rect:
        r = node_rect(node)
        return pygame.Rect(r.right - 20, r.y + 2, 18, 12)

    def sink_rects() -> Dict[str, pygame.Rect]:
        c = node_canvas_rect()
        return {"speed": pygame.Rect(c.right - 150, c.y + 46, 140, 30),
                "yaw": pygame.Rect(c.right - 150, c.y + 96, 140, 30)}

    def node_label(node: Node) -> str:
        if node.kind == "obs":
            who = "self" if node.source == "self" else f"actor {node.source}"
            return f"{who}.{node.field}"
        if node.kind == "const":
            return "const"
        if node.kind == "map_point":
            return f"map.{node.name}" if node.name else "map.custom"
        if node.kind == "time":
            return f"time.{node.which}"
        return node.op

    def node_sub_label(node: Node) -> Optional[str]:
        if node.kind == "const":
            return f"{node.value:.4g}"
        if node.kind == "map_point" and not node.name:
            p = node.point or (0.0, 0.0)
            return f"{p[0]:.3g},{p[1]:.3g}"
        return None

    def clamp_node_pos(x: float, y: float) -> Tuple[float, float]:
        c = node_canvas_rect()
        return (clamp(x, 0, c.w - NODE_W), clamp(y, 0, c.h - NODE_H))

    def value_catalog() -> List[Tuple[str, dict]]:
        a = scenario.actors[selected]
        items: List[Tuple[str, dict]] = [
            ("const", {"kind": "const"}),
            ("time.global", {"kind": "time", "which": "global"}),
            ("time.segment", {"kind": "time", "which": "segment"}),
        ]
        for f in ("speed", "pos", "heading"):
            items.append((f"self.{f}", {"kind": "obs", "source": "self", "field": f}))
        for other in scenario.actors:
            if other.id == a.id:
                continue
            for f in ("speed", "pos", "heading"):
                items.append((f"actor {other.id}.{f}",
                              {"kind": "obs", "source": other.id, "field": f}))
        for name in MAP_POINT_NAMES:
            items.append((f"map.{name}", {"kind": "map_point", "name": name}))
        items.append(("map.custom (x,y)", {"kind": "map_point", "name": None}))
        return items

    def fe_set_value(node: Node, spec: dict) -> None:
        fn = cur_segment()
        old = node_label(node)
        node.kind = spec["kind"]
        if node.kind == "obs":
            node.source = spec["source"]
            node.field = spec["field"]
        elif node.kind == "time":
            node.which = spec["which"]
        elif node.kind == "map_point":
            node.name = spec.get("name")
            if node.name is None and node.point is None:
                node.point = (0.0, 0.0)
        # value-type change may invalidate downstream wires; simplest safe rule:
        # keep them (they render red / evaluate None until fixed)
        scenario.simulate()
        log_struct("fn_set_value", maneuver_index=man_index, node=node.id,
                   old=old, new=node_label(node))
        set_status(f"node {node.id}: {old} -> {node_label(node)}")

    def fe_set_op(node: Node, newop: str) -> None:
        fn = cur_segment()
        old = node.op
        node.op = newop
        arity = len(OPS[newop][0])
        if len(node.inputs) > arity:
            node.inputs = node.inputs[:arity]
        scenario.simulate()
        log_struct("fn_set_op", maneuver_index=man_index, node=node.id,
                   old=old, new=newop)
        set_status(f"node {node.id}: {old} -> {newop}")

    def fe_add_const(cx: float, cy: float) -> None:
        nonlocal sel_node
        fn = cur_segment()
        nid = fn.next_node_id()
        px, py = clamp_node_pos(cx, cy)
        fn.nodes.append(Node(id=nid, kind="const", pos=(px, py), value=1.0))
        sel_node = nid
        scenario.simulate()
        log_struct("fn_add_node", maneuver_index=man_index, node=nid)
        set_status(f"added node {nid} (click it to choose a value)")

    def fe_delete_selected() -> None:
        nonlocal sel_node
        fn = cur_segment()
        if sel_node is None or fn.node(sel_node) is None:
            return
        dead = downstream_ids(fn, sel_node)
        fn.nodes = [n for n in fn.nodes if n.id not in dead]
        unbound = []
        if fn.out_speed in dead:
            fn.out_speed = None
            unbound.append("speed")
        if fn.out_yaw in dead:
            fn.out_yaw = None
            unbound.append("yaw")
        scenario.simulate()
        log_struct("fn_delete_node", maneuver_index=man_index, node=sel_node,
                   removed=sorted(dead), unbound=unbound)
        set_status(f"deleted {len(dead)} node(s)")
        sel_node = None

    def fe_bind_sink(which: str, src: str) -> None:
        fn = cur_segment()
        if node_value_type(fn, src) != SCALAR:
            set_status(f"OUT:{which} needs a scalar")
            return
        if which == "speed":
            fn.out_speed = src
        else:
            fn.out_yaw = src
        scenario.simulate()
        log_struct("fn_bind_out", maneuver_index=man_index, out=which, node=src)
        set_status(f"bound OUT:{which} <- {src}")

    def fe_unbind_sink(which: str) -> None:
        fn = cur_segment()
        old = fn.out_speed if which == "speed" else fn.out_yaw
        if old is None:
            set_status(f"OUT:{which}: drag a scalar node's port here to bind")
            return
        if which == "speed":
            fn.out_speed = None
        else:
            fn.out_yaw = None
        scenario.simulate()
        log_struct("fn_unbind_out", maneuver_index=man_index, out=which, node=old)
        set_status(f"unbound OUT:{which}")

    def fe_hit(mx: int, my: int):
        """('sink', key) | ('port', id) | ('swap', id) | ('slot', (id, idx)) |
        ('node', id) | ('canvas', None)"""
        fn = cur_segment()
        for key, r in sink_rects().items():
            if r.collidepoint(mx, my):
                return ("sink", key)
        for node in reversed(fn.nodes):
            px, py = port_center(node)
            if (mx - px) ** 2 + (my - py) ** 2 <= 64:
                return ("port", node.id)
            if (node.kind == "op" and node.op in ASYMMETRIC_OPS
                    and len(node.inputs) >= 2
                    and swap_badge_rect(node).collidepoint(mx, my)):
                return ("swap", node.id)
            if node.kind == "op":
                for i, (sx_, sy_) in enumerate(slot_centers(node)):
                    if (mx - sx_) ** 2 + (my - sy_) ** 2 <= 81:
                        return ("slot", (node.id, i))
            if node_rect(node).collidepoint(mx, my):
                return ("node", node.id)
        return ("canvas", None)

    def fe_mousedown(mx: int, my: int) -> None:
        nonlocal dragging, wire_from, sel_node, node_press, node_orig
        nonlocal last_canvas_click, focus_field
        fn = cur_segment()
        kind, ident = fe_hit(mx, my)
        focus_field = None
        if kind == "sink":
            fe_unbind_sink(ident)
        elif kind == "port":
            dragging = "wire"
            wire_from = ident
        elif kind == "swap":
            node = fn.node(ident)
            node.inputs[0], node.inputs[1] = node.inputs[1], node.inputs[0]
            scenario.simulate()
            log_struct("fn_swap_inputs", maneuver_index=man_index, node=ident)
            set_status(f"swapped inputs of {ident}")
        elif kind in ("node", "slot"):
            nid = ident if kind == "node" else ident[0]
            sel_node = nid
            node = fn.node(nid)
            dragging = "nodepress"
            node_press = (mx, my)
            node_orig = node.pos
        else:  # canvas
            sel_node = None
            now = pygame.time.get_ticks()
            lt, lp = last_canvas_click
            if now - lt < 400 and abs(mx - lp[0]) < 6 and abs(my - lp[1]) < 6:
                c = node_canvas_rect()
                fe_add_const(mx - c.x - NODE_W // 2, my - c.y - NODE_H // 2)
                last_canvas_click = (0, (0, 0))
            else:
                last_canvas_click = (now, (mx, my))

    def fe_motion(mx: int, my: int) -> None:
        nonlocal dragging
        if dragging == "nodepress" and node_press is not None:
            if abs(mx - node_press[0]) > 4 or abs(my - node_press[1]) > 4:
                dragging = "node"
        if dragging == "node" and sel_node is not None:
            fn = cur_segment()
            node = fn.node(sel_node)
            if node and node_press and node_orig:
                nx = node_orig[0] + (mx - node_press[0])
                ny = node_orig[1] + (my - node_press[1])
                node.pos = clamp_node_pos(nx, ny)

    def fe_default_binary(fn: Function, a_id: str, b_id: str) -> Optional[str]:
        ta, tb = node_value_type(fn, a_id), node_value_type(fn, b_id)
        for op in compatible_ops(fn, [ta, tb]):
            if len(OPS[op][0]) >= 2:
                return op
        return None

    def fe_default_unary(fn: Function, a_id: str) -> Optional[str]:
        ta = node_value_type(fn, a_id)
        for op in compatible_ops(fn, [ta]):
            if len(OPS[op][0]) == 1:
                return op
        return None

    def fe_wire_release(mx: int, my: int) -> None:
        nonlocal sel_node
        fn = cur_segment()
        src = wire_from
        if src is None or fn is None or fn.node(src) is None:
            return
        kind, ident = fe_hit(mx, my)
        if kind == "sink":
            fe_bind_sink(ident, src)
            return
        if kind == "slot":
            # fill (or rewire) a specific input slot of an op node
            oid, idx = ident
            tgt = fn.node(oid)
            slot = idx if idx < len(tgt.inputs) else len(tgt.inputs)
            if not slot_accepts(fn, tgt, slot, node_value_type(fn, src)):
                set_status(f"{tgt.op} slot {slot + 1} rejects that type")
                return
            if would_cycle(fn, tgt.id, src):
                set_status("refused: that wire would create a cycle")
                return
            if slot < len(tgt.inputs):
                old = tgt.inputs[slot]
                if old == src:
                    return
                tgt.inputs[slot] = src
                scenario.simulate()
                log_struct("fn_wire", maneuver_index=man_index, src=src,
                           dst=tgt.id, slot=slot, replaced=old)
                set_status(f"rewired {tgt.id}.{tgt.op}[{slot + 1}]: {old} -> {src}")
            else:
                tgt.inputs.append(src)
                scenario.simulate()
                log_struct("fn_wire", maneuver_index=man_index,
                           src=src, dst=tgt.id, slot=slot)
                set_status(f"wired {src} -> {tgt.id}.{tgt.op}[{slot + 1}]")
            return
        if kind in ("node", "port", "swap") and ident != src:
            # drop on a node BODY (value or op output): combine into a new
            # binary op — click the new op afterwards to choose the operation
            tgt = fn.node(ident)
            op = fe_default_binary(fn, src, ident)
            if op is None:
                set_status("no operation takes those two input types")
                return
            nid = fn.next_node_id()
            ra, rb = node_rect(fn.node(src)), node_rect(tgt)
            c = node_canvas_rect()
            px, py = clamp_node_pos(max(ra.x, rb.x) - c.x + NODE_W + 40,
                                    (ra.y + rb.y) / 2 - c.y)
            fn.nodes.append(Node(id=nid, kind="op", pos=(px, py),
                                 op=op, inputs=[src, ident]))
            sel_node = nid
            scenario.simulate()
            log_struct("fn_wire", maneuver_index=man_index,
                       src=src, dst=ident, new_op=nid, op=op)
            set_status(f"{nid} = {op}({src}, {ident}) — click it to change the op")
            return
        if kind == "canvas":
            op = fe_default_unary(fn, src)
            if op is None:
                set_status("no unary operation takes that type")
                return
            nid = fn.next_node_id()
            c = node_canvas_rect()
            px, py = clamp_node_pos(mx - c.x, my - c.y - NODE_H // 2)
            fn.nodes.append(Node(id=nid, kind="op", pos=(px, py),
                                 op=op, inputs=[src]))
            sel_node = nid
            scenario.simulate()
            log_struct("fn_wire", maneuver_index=man_index, src=src,
                       new_op=nid, op=op)
            set_status(f"{nid} = {op}({src}) — click it to change the op")

    def fe_node_click(mx: int, my: int) -> None:
        """A press+release on a node without movement."""
        nonlocal focus_field, edit_buffer
        fn = cur_segment()
        node = fn.node(sel_node)
        if node is None:
            return
        r = node_rect(node)
        if node.kind == "op":
            wired = [node_value_type(fn, i) for i in node.inputs]
            ops = compatible_ops(fn, wired)

            def sig_str(op):
                sig, out = OPS[op]
                short = {"scalar": "S", "point": "P", "T": "T"}
                return (f"{op}  ({','.join(short[s] for s in sig)})->"
                        f"{short[out]}")
            open_dropdown(r, [(sig_str(o), o) for o in ops],
                          lambda o, n=node: fe_set_op(n, o))
            return
        sub = node_sub_label(node)
        if sub is not None and my > r.y + NODE_H // 2:
            focus_field = f"nv:{node.id}"
            edit_buffer = ""
            return
        open_dropdown(r, value_catalog(),
                      lambda spec, n=node: fe_set_value(n, spec))

    def draw_function_editor(a: Actor, fn: Function) -> None:
        sw = subwin_rect()
        # duration field in the header row
        dr = dur_field_rect()
        screen.blit(font.render("dur", True, C_TEXT), (dr.x - 40, dr.y + 4))
        focused = (focus_field == "duration")
        pygame.draw.rect(screen, (18, 20, 26), dr)
        pygame.draw.rect(screen, C_BTN_HL if focused else (90, 94, 105), dr, 2)
        shown = edit_buffer if focused else f"{fn.duration:.4g}"
        screen.blit(font.render(shown, True, C_TEXT), (dr.x + 6, dr.y + 4))

        c = node_canvas_rect()
        pygame.draw.rect(screen, (16, 18, 24), c)
        pygame.draw.rect(screen, (70, 74, 85), c, 1)
        screen.set_clip(c)
        by = fn.by_id()
        # edges into op nodes
        for node in fn.nodes:
            if node.kind != "op":
                continue
            slots = slot_centers(node)
            for i, inp in enumerate(node.inputs):
                srcn = by.get(inp)
                if srcn and i < len(slots):
                    pygame.draw.line(screen, C_WIRE, port_center(srcn), slots[i], 2)
        # edges into sinks
        sinks = sink_rects()
        for key, nid in (("speed", fn.out_speed), ("yaw", fn.out_yaw)):
            if nid and by.get(nid):
                r = sinks[key]
                pygame.draw.line(screen, (140, 230, 150),
                                 port_center(by[nid]), (r.left, r.centery), 2)
        # wire preview
        if dragging == "wire" and wire_from and by.get(wire_from):
            pygame.draw.line(screen, C_SEL, port_center(by[wire_from]), mouse_pos, 1)
        # nodes
        for node in fn.nodes:
            r = node_rect(node)
            col = NODE_COLORS.get(node.kind, C_BTN)
            pygame.draw.rect(screen, col, r, border_radius=5)
            incomplete = node.kind == "op" and not op_is_complete(fn, node)
            border = C_SEL if node.id == sel_node else (
                C_BAD if incomplete else (110, 114, 125))
            pygame.draw.rect(screen, border, r, 2, border_radius=5)
            sub = node_sub_label(node)
            lbl = font_sm.render(node_label(node), True, C_TEXT)
            if sub is None:
                screen.blit(lbl, (r.centerx - lbl.get_width() // 2,
                                  r.centery - lbl.get_height() // 2))
            else:
                screen.blit(lbl, (r.centerx - lbl.get_width() // 2, r.y + 3))
                if focus_field == f"nv:{node.id}":
                    txt = edit_buffer + "_"
                    scol = C_SEL
                else:
                    txt = sub
                    scol = (180, 220, 255)
                s2 = font_sm.render(txt, True, scol)
                screen.blit(s2, (r.centerx - s2.get_width() // 2, r.y + 18))
            # output port
            pygame.draw.circle(screen, C_CURVE, port_center(node), 5)
            # input slots
            if node.kind == "op":
                for i, sc_ in enumerate(slot_centers(node)):
                    if i < len(node.inputs):
                        pygame.draw.circle(screen, (140, 230, 150), sc_, 4)
                    else:
                        pygame.draw.circle(screen, C_BAD, sc_, 4, 1)
                if node.op in ASYMMETRIC_OPS and len(node.inputs) >= 2:
                    br = swap_badge_rect(node)
                    pygame.draw.rect(screen, (40, 42, 50), br, border_radius=3)
                    sb = font_sm.render("<>", True, C_TEXT)
                    screen.blit(sb, (br.centerx - sb.get_width() // 2, br.y - 1))
        # sinks
        for key, r in sinks.items():
            bound = fn.out_speed if key == "speed" else fn.out_yaw
            pygame.draw.rect(screen, (36, 44, 40) if bound else (40, 40, 46), r,
                             border_radius=5)
            pygame.draw.rect(screen, (140, 230, 150) if bound else (110, 114, 125),
                             r, 2, border_radius=5)
            unit = "m/s" if key == "speed" else "deg"
            txt = f"OUT:{key} ({unit})" if not bound else f"OUT:{key} <- {bound}"
            sl = font_sm.render(txt, True, C_TEXT)
            screen.blit(sl, (r.x + 8, r.centery - sl.get_height() // 2))
        screen.set_clip(None)
        hint = ("2xclick: new node | click node: value | drag port: wire "
                "(body=combine op, slot=fill/rewire, empty=unary, OUT=bind) | Del: delete")
        screen.blit(font_sm.render(hint, True, (140, 144, 155)),
                    (c.x, subwin_rect().bottom - 22))

    # ---- drawing (map / actors / bars) ----
    def draw_map():
        screen.fill(C_GRASS)
        draw_map_surface(screen, scenario.map, w2s,
                         road=C_ROAD, line=C_LINE, edge=C_EDGE)

    def actor_corners_world(pose: Pose, a: Actor):
        x, y, hd = pose
        h = math.radians(hd)
        fx, fy = math.cos(h), math.sin(h)
        px, py = -math.sin(h), math.cos(h)
        L, W = a.length / 2, a.width / 2
        return [(x + fx * L + px * W, y + fy * L + py * W),
                (x + fx * L - px * W, y + fy * L - py * W),
                (x - fx * L - px * W, y - fy * L - py * W),
                (x - fx * L + px * W, y - fy * L + py * W)]

    def rotation_handle_world(a: Actor) -> Tuple[float, float]:
        sx, sy, hd = a.start
        h = math.radians(hd)
        r = a.length / 2 + 2.5
        return (sx + math.cos(h) * r, sy + math.sin(h) * r)

    def point_in_pose(a: Actor, pose: Pose, wx: float, wy: float) -> bool:
        cx, cy, hd = pose
        h = math.radians(hd)
        dx, dy = wx - cx, wy - cy
        along = dx * math.cos(h) + dy * math.sin(h)
        lat = -dx * math.sin(h) + dy * math.cos(h)
        return abs(along) <= a.length / 2 + 0.5 and abs(lat) <= a.width / 2 + 0.5

    def displayed_pose(a: Actor) -> Pose:
        """World pose an actor is drawn at this frame (also what the
        experiment recorder samples, so records match the picture)."""
        if drive_mode and live_ego is not None and a.id == "0":
            return (live_ego["x"], live_ego["y"], math.degrees(live_ego["theta"]))
        if drive_mode and (a.cutin or a.block or a.id in live_rebase):
            return a.pose_at_time(cutin_phase)
        return a.pose_at_time(T % scenario.period)

    def build_frame_snapshot(dt: float) -> dict:
        """State handed to the experiment `on_frame` hook each driving tick."""
        e = live_ego or {"x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0}
        ch, bh = cutin_holder(), block_holder()
        return {
            "T": T, "dt": dt, "period": scenario.period,
            "ego": (e["x"], e["y"], math.degrees(e["theta"]), e["v"]),
            "actors": {a.id: displayed_pose(a)
                       for a in scenario.actors if a.id != "0"},
            "cutin_holder": ch.id if ch else None,
            "cutin_outcome": cutin_outcome,
            "cutin_committed": cutin_committed,
            "block_holder": bh.id if bh else None,
            "block_outcome": block_outcome,
            "block_committed": block_committed,
        }

    def draw_actor(idx: int, a: Actor):
        x, y, hd = displayed_pose(a)
        pts = [w2s(*c) for c in actor_corners_world((x, y, hd), a)]
        pygame.draw.polygon(screen, a.color, pts)
        pygame.draw.polygon(screen, (20, 20, 20), pts, 1)
        pygame.draw.line(screen, (250, 250, 250), pts[0], pts[1], 3)
        if idx == selected:
            pygame.draw.polygon(screen, C_SEL, pts, 3)
            if not playing and not drive_mode:
                spts = [w2s(*c) for c in actor_corners_world(a.start, a)]
                pygame.draw.polygon(screen, C_SEL, spts, 2)
                sc0 = w2s(a.start[0], a.start[1])
                hpt = w2s(*rotation_handle_world(a))
                pygame.draw.line(screen, C_SEL, sc0, hpt, 2)
                pygame.draw.circle(screen, C_SEL, hpt, 6)
                tag = font_sm.render("spawn (drag to move, handle to rotate)",
                                     True, C_SEL)
                screen.blit(tag, (spts[3][0], spts[3][1] + 4))
        label = font_sm.render(a.id, True, C_TEXT)
        lp = w2s(x, y)
        screen.blit(label, (lp[0] - label.get_width() // 2, lp[1] - 8))
        if drive_mode and live_ego is not None and a.id == "0":
            spd = font_sm.render(f"{live_ego['v']:.1f} m/s", True, C_SEL)
            screen.blit(spd, (lp[0] - spd.get_width() // 2, lp[1] + 10))

    def draw_cutin_pins():
        ego = ego_actor()
        if ego is None:
            return
        for i, a in enumerate(scenario.actors):
            spec = a.cutin
            if not spec or "t" not in spec:
                continue
            t_c = float(spec["t"])
            along = float(spec.get("along", 1.0))
            lat = float(spec.get("lat", 0.0))
            if drive_mode and live_ego is not None:
                e = live_ego
                ex, ey = e["x"], e["y"]
                wx, wy = live_cutin_pin(ex, ey, e["theta"], along, lat)
            else:
                wx, wy, _ = cutin_world_target(ego, spec)
                ex, ey, _ = ego.pose_at_time(t_c)
            px, py = w2s(wx, wy)
            if drive_mode and cutin_outcome == "abandoned":
                col = (140, 140, 150)
                note = "  ABANDONED — cruising"
            elif drive_mode and cutin_outcome == "merged":
                col = (90, 200, 120)
                note = "  MERGED"
            elif i == selected:
                col = C_SEL
                note = "  (drag pin)" if not playing else ""
            else:
                col = (255, 200, 60)
                note = "  (drag pin)" if (not playing and not drive_mode) else ""
            pygame.draw.line(screen, (col[0] // 2, col[1] // 2, col[2] // 2),
                             w2s(ex, ey), (px, py), 1)
            pygame.draw.line(screen, col, (px, py), (px, py - 18), 2)
            pygame.draw.polygon(screen, col,
                                [(px, py - 18), (px + 12, py - 13), (px, py - 8)])
            pygame.draw.circle(screen, col, (px, py), 4)
            pygame.draw.circle(screen, (20, 20, 20), (px, py), 4, 1)
            if not playing or drive_mode:
                tag = font_sm.render(
                    f"cut-in @ t={t_c:.2f}s  "
                    f"+{along:.1f}m ahead / {lat:+.1f}m lat{note}",
                    True, col)
                screen.blit(tag, (px + 14, py - 24))
        # block-cut-in hold points ("no entry" marker in the ego's target lane)
        for a in scenario.actors:
            spec = a.block
            if not spec or a.id == "0" or "t" not in spec or "lat" not in spec:
                continue
            t0, t_end = block_window(spec)
            along = float(spec.get("along", 0.0))
            lat = float(spec.get("lat", 0.0))
            if drive_mode and live_ego is not None:
                e = live_ego
                ex, ey = e["x"], e["y"]
                wx, wy = live_cutin_pin(ex, ey, e["theta"], along, lat)
            else:
                wx, wy = block_world_target(ego, spec)
                ex, ey, _ = ego.pose_at_time(t0)
            px, py = w2s(wx, wy)
            if drive_mode and block_outcome == "blocked":
                col = (90, 200, 120)
                note = "  BLOCKED — merge denied"
            elif drive_mode and block_outcome == "breached":
                # live (window still open) or final (committed): same pin look
                col = (140, 140, 150)
                note = ("  BREACHED — ego in lane"
                        if not block_committed
                        else "  BREACHED — ego merged")
            elif drive_mode and not block_committed:
                col = (230, 95, 80)
                note = "  FEASIBLE — holding"
            else:
                col = (230, 95, 80)
                note = ""
                if not drive_mode:
                    # scripted grade: did the ego ever find a safe gap?
                    res = scripted_block_result(ego, a, spec,
                                                scenario.map.lane_width)
                    if res is not None:
                        denied, gap = res
                        if denied:
                            col = (90, 200, 120)
                            note = f"  GAP DENIED — ego merges at {gap:.1f}m"
                        else:
                            note = f"  ineffective — ego finds {gap:.1f}m gap"
            pygame.draw.line(screen, (col[0] // 2, col[1] // 2, col[2] // 2),
                             w2s(ex, ey), (px, py), 1)
            pygame.draw.circle(screen, col, (px, py), 9, 2)
            pygame.draw.line(screen, col, (px - 6, py), (px + 6, py), 3)
            if not playing or drive_mode:
                tag = font_sm.render(
                    f"block @ t={t0:.1f}s for {t_end - t0:.1f}s  "
                    f"{lat:+.1f}m lat{note}", True, col)
                screen.blit(tag, (px + 14, py - 24))

    def draw_button(rect, label, active=False, enabled=True):
        col = C_BTN_HL if active else C_BTN
        if not enabled:
            col = (44, 46, 52)
        pygame.draw.rect(screen, col, rect, border_radius=6)
        pygame.draw.rect(screen, (90, 94, 105), rect, 1, border_radius=6)
        txt = font.render(label, True, C_TEXT if enabled else (120, 120, 130))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                          rect.centery - txt.get_height() // 2))

    def draw_topbar():
        pygame.draw.rect(screen, C_BAR, pygame.Rect(0, 0, WIDTH, TOPBAR_H))
        draw_button(btn_reset, "Reset")
        draw_button(btn_play, "Pause" if playing else "Play", active=playing)
        draw_button(btn_save, "Save")
        draw_button(btn_drive, "Drive", active=drive_mode)
        draw_button(btn_add, "+ Actor", enabled=not drive_mode)
        draw_button(btn_del, "- Actor",
                    enabled=(selected is not None and not drive_mode))
        screen.blit(font.render("T=", True, C_TEXT), (18, 18))
        focused = (focus_field == "time")
        pygame.draw.rect(screen, (18, 20, 26), time_field)
        pygame.draw.rect(screen, C_BTN_HL if focused else (90, 94, 105), time_field, 2)
        shown = edit_buffer if focused else f"{T % scenario.period:.2f}"
        screen.blit(font.render(shown, True, C_TEXT), (time_field.x + 6, time_field.y + 5))
        if drive_mode and live_ego is not None:
            mode = f"DRIVE  ego {live_ego['v']:.1f} m/s"
        else:
            mode = "PLAYING" if playing else "PAUSED"
        info = f"/ {scenario.period:.2f}s   {mode}"
        screen.blit(font.render(info, True, C_TEXT), (time_field.right + 10, 18))

    def draw_subwindow():
        sw = subwin_rect()
        pygame.draw.rect(screen, C_PANEL, sw)
        pygame.draw.line(screen, (80, 84, 95), (sw.x, sw.y), (sw.right, sw.y), 2)
        if status_msg and T < status_until:
            st = font_sm.render(status_msg, True, (150, 220, 150))
            screen.blit(st, (sw.right - st.get_width() - 16, sw.bottom - 22))
        m = cur_segment()
        if playing or selected is None or m is None:
            hint = "Pause and click an actor to edit its segments; " \
                   "use + Actor / - Actor to add or remove."
            screen.blit(font.render(hint, True, (150, 154, 165)), (sw.x + 20, sw.y + 22))
            return
        a = scenario.actors[selected]
        title = f"Actor {a.id}   segment {man_index + 1}/{len(a.maneuvers)}"
        screen.blit(font_big.render(title, True, C_TEXT), (sw.x + 16, sw.y + 14))
        hb = header_buttons()
        draw_button(hb["type"], f"type: {m.type}")
        draw_button(hb["prev"], "prev")
        draw_button(hb["next"], "next")
        draw_button(hb["add_mvr"], "+seg")
        draw_button(hb["del_mvr"], "-seg", enabled=(len(a.maneuvers) > 1))

        if isinstance(m, Function):
            draw_function_editor(a, m)
            return
        pr = plot_rect()
        if m.curve_kind == "velocity":
            pygame.draw.rect(screen, (18, 20, 26), pr)
            pygame.draw.rect(screen, C_AXIS, pr, 1)
            t2x, v2y, _, tmax, vmin, vmax = plot_maps(m, pr)
            screen.blit(font_sm.render("velocity (m/s)", True, C_AXIS), (pr.x - 4, pr.y - 18))
            screen.blit(font_sm.render("time (s)", True, C_AXIS),
                        (pr.right - 60, pr.bottom + 6))
            if vmin < 0 < vmax:
                zy = v2y(0)
                pygame.draw.line(screen, (70, 74, 85), (pr.x, zy), (pr.right, zy), 1)
            p_left = (t2x(0), v2y(m.intercept))
            p_right = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
            pygame.draw.line(screen, C_CURVE, p_left, p_right, 2)
            pygame.draw.circle(screen, C_SEL, (int(p_left[0]), int(p_left[1])), 6)
            if m.type not in ("go_straight", "lane_change"):
                pygame.draw.circle(screen, C_SEL, (int(p_right[0]), int(p_right[1])), 6)
            phase = T % scenario.period
            if a.cum[man_index] <= phase < a.cum[man_index + 1]:
                mx = t2x(phase - a.cum[man_index])
                pygame.draw.line(screen, (250, 120, 120), (mx, pr.y), (mx, pr.bottom), 1)
        else:
            note = {"turn_left": "left turn — set radius, angle, duration",
                    "turn_right": "right turn — set radius, angle, duration",
                    "stop": "stop — hold position for duration"}.get(m.type, "")
            screen.blit(font.render(note, True, (150, 154, 165)), (pr.x, pr.y + 8))

        for name, rect in field_rects(m).items():
            screen.blit(font.render(field_label(m, name), True, C_TEXT),
                        (rect.x - 90, rect.y + 4))
            focused = (focus_field == name)
            pygame.draw.rect(screen, (18, 20, 26), rect)
            pygame.draw.rect(screen, C_BTN_HL if focused else (90, 94, 105), rect, 2)
            shown = edit_buffer if focused else f"{getattr(m, name):.4g}"
            screen.blit(font.render(shown, True, C_TEXT), (rect.x + 6, rect.y + 4))
        hint = "drag curve endpoints or click a field and type (Enter). +/-seg add/delete."
        screen.blit(font_sm.render(hint, True, (140, 144, 155)),
                    (pr.x, subwin_rect().bottom - 24))

    # ---- event handling ----
    def handle_click(mx, my):
        nonlocal playing, T, selected, man_index, focus_field, edit_buffer, dragging
        nonlocal drag_grab, drag_orig, sel_node, cutin_drag, cutin_orig
        if btn_play.collidepoint(mx, my):
            playing = not playing
            return
        if btn_drive.collidepoint(mx, my):
            toggle_drive_mode()
            return
        if btn_reset.collidepoint(mx, my):
            T = 0.0
            if drive_mode:
                # re-enter drive from canonical spawns (not drifted live starts)
                enter_drive_mode()
            else:
                restore_spawns()
                resolve_cutins(scenario)
            return
        if btn_save.collidepoint(mx, my):
            if persistence:
                fname = persistence.save_version(scenario)
                set_status(f"saved {fname} (parent v{persistence.versions[-1]['parent']})")
            return
        # orchestrator-panel autonomy dropdowns (usable while driving too)
        for aid, br in orch_buttons.items():
            if br.collidepoint(mx, my):
                actor = next((a for a in scenario.actors if a.id == aid), None)
                if actor is not None:
                    open_dropdown(br,
                                  [("fully autonomous", "auto"),
                                   ("fully self-governed", "self")],
                                  make_autonomy_cb(actor))
                return
        if drive_mode:
            return   # no scripted editing while driving
        if time_field.collidepoint(mx, my):
            playing = False
            focus_field = "time"
            edit_buffer = ""
            return
        if btn_add.collidepoint(mx, my):
            do_add_actor()
            return
        if btn_del.collidepoint(mx, my) and selected is not None:
            do_remove_actor()
            return
        # subwindow interactions (only when visible)
        m = cur_segment()
        if not playing and selected is not None and m is not None:
            sw = subwin_rect()
            hb = header_buttons()
            a = scenario.actors[selected]
            if hb["prev"].collidepoint(mx, my):
                man_index = (man_index - 1) % len(a.maneuvers)
                focus_field = None
                sel_node = None
                return
            if hb["next"].collidepoint(mx, my):
                man_index = (man_index + 1) % len(a.maneuvers)
                focus_field = None
                sel_node = None
                return
            if hb["add_mvr"].collidepoint(mx, my):
                do_add_segment(); focus_field = None; return
            if hb["del_mvr"].collidepoint(mx, my):
                do_del_segment(); focus_field = None; return
            if hb["type"].collidepoint(mx, my):
                open_dropdown(hb["type"], [(t, t) for t in SEGMENT_TYPES],
                              do_set_segment_type)
                return
            if isinstance(m, Function):
                if dur_field_rect().collidepoint(mx, my):
                    focus_field = "duration"
                    edit_buffer = ""
                    return
                # node canvas is handled by fe_mousedown (router), not here
            else:
                for key, rect in field_rects(m).items():
                    if rect.collidepoint(mx, my):
                        focus_field = key
                        edit_buffer = ""
                        return
                if m.curve_kind == "velocity":
                    pr = plot_rect()
                    t2x, v2y, _, tmax, _, _ = plot_maps(m, pr)
                    pl = (t2x(0), v2y(m.intercept))
                    prg = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
                    if (mx - pl[0]) ** 2 + (my - pl[1]) ** 2 < 100:
                        dragging = "left"; focus_field = None; return
                    if m.type not in ("go_straight", "lane_change") \
                            and (mx - prg[0]) ** 2 + (my - prg[1]) ** 2 < 100:
                        dragging = "right"; focus_field = None; return
            if sw.collidepoint(mx, my):
                return
        # BEV interactions (paused only)
        if not playing and TOPBAR_H < my < TOPBAR_H + CANVAS_H:
            wx, wy = s2w(mx, my)
            for i, a in enumerate(scenario.actors):
                if not a.cutin or "t" not in a.cutin:
                    continue
                ego = next((x for x in scenario.actors if x.id == "0"), None)
                if ego is None:
                    break
                cx, cy, _ = cutin_world_target(ego, a.cutin)
                px, py = w2s(cx, cy)
                if (mx - px) ** 2 + (my - py) ** 2 < 144:
                    dragging = "cutin"
                    cutin_drag = i
                    cutin_orig = dict(a.cutin)
                    selected = i
                    focus_field = None
                    return
            if selected is not None:
                a = scenario.actors[selected]
                hpt = w2s(*rotation_handle_world(a))
                if (mx - hpt[0]) ** 2 + (my - hpt[1]) ** 2 < 100:
                    dragging = "rotate"; drag_orig = a.start; focus_field = None
                    T = 0.0
                    return
                if point_in_pose(a, a.start, wx, wy):
                    dragging = "spawn"; drag_grab = (wx, wy); drag_orig = a.start
                    focus_field = None
                    T = 0.0
                    return
            phase = T % scenario.period
            for i, a in enumerate(scenario.actors):
                if point_in_pose(a, a.pose_at_time(phase), wx, wy):
                    selected = i
                    man_index = a.active_index(phase) if a.total > phase else 0
                    focus_field = None
                    sel_node = None
                    return

    def handle_drag(mx, my):
        if dragging == "cutin" and cutin_drag is not None:
            a = scenario.actors[cutin_drag]
            ego = next((x for x in scenario.actors if x.id == "0"), None)
            if ego is None or not a.cutin:
                return
            # pin is ego-relative at the *current scrub time*: dragging stamps
            # both t and the (along, lat) offset from wherever the ego is now
            t_now = T % scenario.period
            ego_pose = ego.pose_at_time(t_now)
            wx, wy = s2w(mx, my)
            along, lat = world_to_ego_offset(ego_pose, wx, wy)
            a.cutin["t"] = round(t_now, 3)
            a.cutin["along"] = round(along, 2)
            a.cutin["lat"] = round(lat, 2)
            maybe_resolve_cutins()
            return
        if dragging in ("spawn", "rotate") and selected is not None:
            a = scenario.actors[selected]
            wx, wy = s2w(mx, my)
            if dragging == "spawn" and drag_grab is not None:
                nx = drag_orig[0] + (wx - drag_grab[0])
                ny = drag_orig[1] + (wy - drag_grab[1])
                a.start = (nx, ny, a.start[2])
            else:
                hd = math.degrees(math.atan2(wy - a.start[1], wx - a.start[0]))
                a.start = (a.start[0], a.start[1], round(hd, 1))
            scenario.simulate()
            return
        m = cur_segment()
        if not isinstance(m, Maneuver) or m.curve_kind != "velocity":
            return
        pr = plot_rect()
        _, _, y2v, tmax, _, _ = plot_maps(m, pr)
        val = y2v(clamp(my, pr.y, pr.bottom))
        if dragging == "left":
            apply_param("intercept", val)
        elif dragging == "right":
            apply_param("slope", (val - m.intercept) / max(1e-6, m.duration))

    def commit_field():
        nonlocal focus_field, edit_buffer, T
        if focus_field is None:
            return
        try:
            if focus_field == "time":
                T = max(0.0, float(edit_buffer))
            elif focus_field.startswith("nv:"):
                fn = cur_segment()
                node = fn.node(focus_field[3:]) if isinstance(fn, Function) else None
                if node is not None:
                    if node.kind == "const":
                        old = node.value
                        node.value = float(edit_buffer)
                        log_struct("fn_set_value", maneuver_index=man_index,
                                   node=node.id, old=old, new=node.value)
                    elif node.kind == "map_point" and not node.name:
                        xs, ys = edit_buffer.split(",")
                        old = node.point
                        node.point = (float(xs), float(ys))
                        log_struct("fn_set_value", maneuver_index=man_index,
                                   node=node.id, old=list(old or ()),
                                   new=list(node.point))
                    scenario.simulate()
                    set_status(f"node {node.id} updated")
            else:
                apply_param(focus_field, float(edit_buffer))
        except (ValueError, IndexError):
            set_status("invalid value")
        focus_field = None
        edit_buffer = ""

    def draw_orch_panel():
        """Top-left role-casting card (only for orchestrated scenarios)."""
        nonlocal orch_buttons
        others = [a for a in scenario.actors if a.id != "0"]
        if not any(a.cutin or a.block for a in others):
            orch_buttons = {}
            return
        roles = {a.id: (co.ROLE_CUTIN if a.cutin
                        else co.ROLE_BLOCK if a.block
                        else co.ROLE_NOMINAL)
                 for a in others}
        autonomy = {a.id: a.autonomy for a in others}
        # per-cell candidate scores at the poses currently on screen
        ego = ego_actor()
        scores: Optional[Dict[str, Dict[str, float]]] = None
        if ego is not None:
            if drive_mode and live_ego is not None:
                ego_pose = (live_ego["x"], live_ego["y"],
                            math.degrees(live_ego["theta"]))
            else:
                ego_pose = ego.pose_at_time(T % scenario.period)
            lw = scenario.map.lane_width
            bspec = next((a.block for a in others if a.block), None)
            target_lat = (float(bspec["lat"])
                          if bspec and "lat" in bspec else None)

            def pose_now(a: Actor) -> Pose:
                if drive_mode and (a.cutin or a.block):
                    return a.pose_at_time(cutin_phase)
                return a.pose_at_time(T % scenario.period)

            scores = {}
            for a in others:
                p = pose_now(a)
                d = {co.ROLE_CUTIN: co.score_cutin_candidate(p, ego_pose, lw)}
                if target_lat is not None:
                    d[co.ROLE_BLOCK] = co.score_block_candidate(
                        p, ego_pose, lw, target_lat)
                scores[a.id] = d
        orch_buttons = co.draw_role_panel(screen, font, font_sm, roles,
                                          origin=(36, TOPBAR_H + 36),
                                          autonomy=autonomy, scores=scores)

    def render_frame():
        draw_map()
        for i, a in enumerate(scenario.actors):
            draw_actor(i, a)
        draw_cutin_pins()
        draw_rulers(screen, w2s, s2w,
                    pygame.Rect(0, TOPBAR_H, WIDTH, CANVAS_H),
                    font=font_sm)
        draw_orch_panel()
        draw_subwindow()
        draw_topbar()
        draw_dropdown()
        pygame.display.flip()

    # ---- headless capture ----
    if capture:
        import subprocess
        total_frames = max(1, int(round(scenario.period * loops * fps)))
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(fps),
             "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
             "-loglevel", "error", capture],
            stdin=subprocess.PIPE)
        playing = True
        for n in range(total_frames):
            T = n / fps
            pygame.event.pump()
            render_frame()
            ff.stdin.write(pygame.image.tostring(screen, "RGB"))
        ff.stdin.close()
        ff.wait()
        pygame.quit()
        return

    # ---- headless snapshot (one paused frame, optional selection) ----
    if snapshot:
        playing = False
        T = max(0.0, at_time)
        if select_id is not None:
            for i, a in enumerate(scenario.actors):
                if a.id == str(select_id):
                    selected = i
                    phase = T % scenario.period
                    man_index = a.active_index(phase) if a.total > phase else 0
        pygame.event.pump()
        render_frame()
        pygame.image.save(screen, snapshot)
        pygame.quit()
        return

    # ---- interactive main loop ----
    if auto_drive:
        enter_drive_mode()          # experiment harness starts already driving
    running = True
    frame_i = 0
    while running:
        dt = clock.tick(60) / 1000.0
        frame_i += 1
        if playing:
            if drive_mode and live_ego is not None:
                keys = pygame.key.get_pressed()
                throttle = steer = 0.0
                manual = False
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    throttle, manual = 1.0, True
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    throttle, manual = -1.0, True
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    steer = 1.0
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    steer = -1.0
                if not manual and drive_profile:
                    # autopilot: track the scripted ego speed profile
                    k = min(int(T / DT), len(drive_profile) - 1)
                    err = drive_profile[k] - live_ego["v"]
                    throttle = clamp(err / 1.5, -1.0, 1.0)
                integrate_live_ego(throttle, steer, dt)
                T += dt
                cutin_phase += dt
                if frame_i % 3 == 0:
                    live_resolve_cutins()
                if on_frame is not None:
                    # experiment recorder; may ask to end the run
                    if on_frame(build_frame_snapshot(dt)) is False:
                        running = False
            else:
                T += dt
        mouse_pos = pygame.mouse.get_pos()
        # keep cut-in plans consistent with any ego edits (cheap no-op check)
        if frame_i % 15 == 0 and dragging is None and not drive_mode:
            maybe_resolve_cutins()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if dropdown is not None:
                    dropdown_click(*event.pos)
                elif fe_active() and node_canvas_rect().collidepoint(event.pos):
                    fe_mousedown(*event.pos)
                else:
                    handle_click(*event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging == "wire":
                    fe_wire_release(*event.pos)
                elif dragging == "node" and sel_node is not None and node_orig:
                    fn = cur_segment()
                    node = fn.node(sel_node) if isinstance(fn, Function) else None
                    if node is not None and tuple(node.pos) != tuple(node_orig):
                        log_struct("fn_move_node", maneuver_index=man_index,
                                   node=sel_node,
                                   old=[round(v, 1) for v in node_orig],
                                   new=[round(v, 1) for v in node.pos])
                elif dragging == "nodepress":
                    fe_node_click(*event.pos)
                elif dragging in ("spawn", "rotate"):
                    commit_spawn_edit()
                elif dragging == "cutin" and cutin_drag is not None:
                    a = scenario.actors[cutin_drag]
                    if cutin_orig is not None and persistence is not None \
                            and (a.cutin.get("t") != cutin_orig.get("t")
                                 or a.cutin.get("along") != cutin_orig.get("along")
                                 or a.cutin.get("lat") != cutin_orig.get("lat")):
                        persistence.log_structural(
                            "move_cutin_target", a.id,
                            old={k: cutin_orig.get(k)
                                 for k in ("t", "along", "lat")},
                            new={k: a.cutin.get(k)
                                 for k in ("t", "along", "lat")})
                    set_status(
                        f"cut-in @ t={a.cutin['t']:.2f}s  "
                        f"+{a.cutin['along']:.1f}m ahead / "
                        f"{a.cutin.get('lat', 0):+.1f}m lat; plan re-solved")
                dragging = None
                cutin_drag = None
                cutin_orig = None
                drag_grab = None
                drag_orig = None
                wire_from = None
                node_press = None
                node_orig = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                if dragging in ("nodepress", "node", "wire"):
                    fe_motion(*event.pos)
                else:
                    handle_drag(*event.pos)
            elif event.type == pygame.MOUSEWHEEL and dropdown is not None:
                mx_sc = len(dropdown["items"]) - DD_MAX_VIS
                if mx_sc > 0:
                    dropdown["scroll"] = int(clamp(dropdown["scroll"] - event.y,
                                                   0, mx_sc))
            elif event.type == pygame.KEYDOWN:
                if dropdown is not None:
                    if event.key == pygame.K_ESCAPE:
                        dropdown = None
                elif focus_field is not None:
                    if event.key == pygame.K_RETURN:
                        commit_field()
                    elif event.key == pygame.K_ESCAPE:
                        focus_field = None; edit_buffer = ""
                    elif event.key == pygame.K_BACKSPACE:
                        edit_buffer = edit_buffer[:-1]
                    elif event.unicode in "0123456789.-+eE,":
                        edit_buffer += event.unicode
                else:
                    if event.key == pygame.K_SPACE:
                        playing = not playing
                    elif event.key == pygame.K_ESCAPE:
                        if drive_mode:
                            exit_drive_mode()
                        elif sel_node is not None:
                            sel_node = None
                        else:
                            selected = None
                    elif event.key == pygame.K_f:
                        toggle_drive_mode()
                    elif drive_mode:
                        pass
                    elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        if fe_active() and sel_node is not None:
                            fe_delete_selected()
                        elif selected is not None:
                            do_remove_actor()
                    elif event.key == pygame.K_a:
                        do_add_actor()

        render_frame()

    pygame.quit()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Driving scenario editor")
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, "scenarios")
    ap.add_argument("scenario", nargs="?",
                    default=os.path.join(default_dir, "scenario_v1.yaml"),
                    help="path to a scenario YAML")
    ap.add_argument("--scenarios-dir", default=None,
                    help="folder for versioned saves / provenance (default: scenario's folder)")
    ap.add_argument("--validate", action="store_true",
                    help="load the scenario, report validity against the current format, and exit")
    ap.add_argument("--capture", default=None, metavar="OUT.mp4",
                    help="headless: record the whole window to an MP4 (no GUI) and exit")
    ap.add_argument("--fps", type=int, default=30, help="capture frame rate")
    ap.add_argument("--loops", type=int, default=1, help="number of loops to capture")
    ap.add_argument("--snapshot", default=None, metavar="OUT.png",
                    help="headless: render one paused frame to a PNG and exit")
    ap.add_argument("--select", default=None, metavar="ID",
                    help="with --snapshot: actor id to select")
    ap.add_argument("--time", type=float, default=0.0,
                    help="with --snapshot: paused clock time")
    args = ap.parse_args()

    if args.validate:
        try:
            sc = validate_scenario(args.scenario)
            print(f"OK: {args.scenario} — {len(sc.actors)} actor(s), "
                  f"period {sc.period:.2f}s")
            sys.exit(0)
        except Exception as e:
            print(f"INVALID: {args.scenario} — {e}")
            sys.exit(1)

    scenario = load_scenario(args.scenario)
    if args.capture:
        run_gui(scenario, None, capture=args.capture, fps=args.fps, loops=args.loops)
        print(f"captured {args.capture} "
              f"({scenario.period * args.loops:.2f}s x{args.loops} @ {args.fps}fps)")
    elif args.snapshot:
        run_gui(scenario, None, snapshot=args.snapshot,
                select_id=args.select, at_time=args.time)
        print(f"snapshot {args.snapshot} (T={args.time}, select={args.select})")
    else:
        sdir = args.scenarios_dir or os.path.dirname(os.path.abspath(args.scenario))
        persistence = Persistence(sdir, args.scenario)
        run_gui(scenario, persistence)


if __name__ == "__main__":
    # helper modules import `scenario_editor`; alias it to this running module
    # so they share our classes instead of loading a second copy
    sys.modules.setdefault("scenario_editor", sys.modules[__name__])
    main()
