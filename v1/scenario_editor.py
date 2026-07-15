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
                          (to node: binary op, to empty: unary op, to an op:
                          fill next input slot, to OUT sink: bind); click an
                          op = choose operation; <> badge swaps inputs;
                          Delete = remove selected node + everything downstream
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
ALL_MANEUVER_TYPES = LONGITUDINAL_TYPES | TURN_TYPES | {"stop"}
# dropdown order for the segment `type:` button (function is the 7th type)
SEGMENT_TYPES = ["go_straight", "turn_left", "turn_right",
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


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Maneuver:
    type: str
    duration: float = 1.0
    # velocity curve for longitudinal maneuvers: v(t) = intercept + slope*t
    intercept: float = 0.0     # initial speed v0 (m/s)
    slope: float = 0.0         # acceleration a (m/s^2)
    # turn geometry
    radius: float = 5.0
    angle: float = 90.0

    @property
    def curve_kind(self) -> str:
        return "velocity" if self.type in LONGITUDINAL_TYPES else "none"

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
        if self.type in LONGITUDINAL_TYPES:
            d["curve"] = {"v0": round(self.intercept, 4), "accel": round(self.slope, 4)}
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
        return {"id": self.id, "color": list(self.color),
                "length": self.length, "width": self.width,
                "start": {"x": round(self.start[0], 4), "y": round(self.start[1], 4),
                          "heading": round(self.start[2], 4)},
                "maneuvers": [m.to_dict() for m in self.maneuvers]}


@dataclass
class MapConfig:
    lane_width: float = 3.5
    arm_length: float = 60.0


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
        return {"map": {"lane_width": self.map.lane_width,
                        "arm_length": self.map.arm_length},
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


def _parse_segment(md: dict) -> Segment:
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
    )


def load_scenario(path: str) -> Scenario:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    mp = raw.get("map", {}) or {}
    mapcfg = MapConfig(lane_width=float(mp.get("lane_width", 3.5)),
                       arm_length=float(mp.get("arm_length", 60.0)))
    ppm = float((raw.get("render", {}) or {}).get("pixels_per_meter", 6.0))
    actors: List[Actor] = []
    for ad in raw.get("actors", []):
        st = ad.get("start", {})
        actors.append(Actor(
            id=str(ad["id"]),
            color=tuple(ad.get("color", [200, 80, 80])),
            length=float(ad.get("length", 4.5)),
            width=float(ad.get("width", 2.0)),
            start=(float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                   float(st.get("heading", 0.0))),
            maneuvers=[_parse_segment(md) for md in ad.get("maneuvers", [])],
        ))
    sc = Scenario(map=mapcfg, actors=actors, pixels_per_meter=ppm)
    sc.simulate()
    return sc


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
            at_time: float = 0.0) -> None:
    """Interactive editor; `capture` records an MP4 headlessly; `snapshot`
    renders one paused frame (optionally with an actor selected) to a PNG."""
    import pygame

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
    selected: Optional[int] = None       # actor index
    man_index = 0                        # segment index within selected actor
    focus_field: Optional[str] = None    # field name | 'time' | 'nv:<node id>'
    edit_buffer = ""
    dragging: Optional[str] = None       # 'left'|'right'|'spawn'|'rotate'
    #                                      |'wire'|'node'|'nodepress'
    drag_grab: Optional[Tuple[float, float]] = None
    drag_orig: Optional[Pose] = None
    status_msg = ""
    status_until = 0.0
    # function-editor state
    sel_node: Optional[str] = None
    wire_from: Optional[str] = None
    node_press: Optional[Tuple[int, int]] = None
    node_orig: Optional[Tuple[float, float]] = None
    last_canvas_click = (0, (0, 0))      # (ms, pos) for double-click detection
    mouse_pos = (0, 0)
    # dropdown widget state: {'rect','items':[(label,payload)],'cb','scroll'}
    dropdown: Optional[dict] = None
    DD_ITEM_H = 22
    DD_MAX_VIS = 12

    # ---- top bar rects ----
    btn_play = pygame.Rect(WIDTH // 2 - 55, 10, 110, 36)
    btn_reset = pygame.Rect(WIDTH // 2 - 190, 10, 110, 36)
    btn_save = pygame.Rect(WIDTH // 2 + 80, 10, 110, 36)
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

    def do_add_actor() -> None:
        nonlocal selected, man_index, sel_node
        i = len(scenario.actors)
        sx, sy, hd = ADD_PRESETS[i % len(ADD_PRESETS)]
        a = Actor(id=next_actor_id(), color=ADD_PALETTE[i % len(ADD_PALETTE)],
                  length=4.5, width=2.0, start=(sx, sy, hd),
                  maneuvers=[Maneuver(type="go_straight", duration=8.0,
                                      intercept=12.0, slope=0.0)])
        scenario.actors.append(a)
        scenario.simulate()
        if persistence:
            persistence.log_structural("add_actor", a.id)
        selected = len(scenario.actors) - 1
        man_index = 0
        sel_node = None
        set_status(f"added actor {a.id}")

    def do_remove_actor() -> None:
        nonlocal selected, man_index, sel_node
        if selected is None:
            return
        a = scenario.actors.pop(selected)
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
        """('sink', key) | ('port', id) | ('swap', id) | ('node', id) | ('canvas', None)"""
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
        elif kind == "node":
            sel_node = ident
            node = fn.node(ident)
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
        if kind in ("node", "port", "swap") and ident != src:
            tgt = fn.node(ident)
            if tgt.kind == "op":
                sig, _ = OPS[tgt.op]
                if len(tgt.inputs) >= len(sig):
                    set_status(f"{tgt.op} has no free input slot")
                    return
                slot = len(tgt.inputs)
                if not slot_accepts(fn, tgt, slot, node_value_type(fn, src)):
                    set_status(f"{tgt.op} slot {slot + 1} rejects that type")
                    return
                if would_cycle(fn, tgt.id, src):
                    set_status("refused: that wire would create a cycle")
                    return
                tgt.inputs.append(src)
                scenario.simulate()
                log_struct("fn_wire", maneuver_index=man_index,
                           src=src, dst=tgt.id, slot=slot)
                set_status(f"wired {src} -> {tgt.id}.{tgt.op}[{slot + 1}]")
            else:
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
                "(node=op, empty=unary, op=next slot, OUT=bind) | Del: delete")
        screen.blit(font_sm.render(hint, True, (140, 144, 155)),
                    (c.x, subwin_rect().bottom - 22))

    # ---- drawing (map / actors / bars) ----
    def draw_map():
        screen.fill(C_GRASS)
        arm = scenario.map.arm_length
        half = scenario.map.lane_width
        x0, y0 = w2s(-arm, half)
        x1, y1 = w2s(arm, -half)
        pygame.draw.rect(screen, C_ROAD, pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        x0, y0 = w2s(-half, arm)
        x1, y1 = w2s(half, -arm)
        pygame.draw.rect(screen, C_ROAD, pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        dash = 3.0
        d = -arm
        while d < arm:
            if abs(d) > half:
                a1 = w2s(d, 0); a2 = w2s(min(d + dash, arm), 0)
                pygame.draw.line(screen, C_LINE, a1, a2, 2)
                b1 = w2s(0, d); b2 = w2s(0, min(d + dash, arm))
                pygame.draw.line(screen, C_LINE, b1, b2, 2)
            d += dash * 2
        for sx, sy, ex, ey in [(-half, -half, 0, -half), (0, half, half, half),
                               (-half, half, -half, 0), (half, -half, half, 0)]:
            pygame.draw.line(screen, C_EDGE, w2s(sx, sy), w2s(ex, ey), 2)

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

    def draw_actor(idx: int, a: Actor):
        phase = T % scenario.period
        x, y, hd = a.pose_at_time(phase)
        pts = [w2s(*c) for c in actor_corners_world((x, y, hd), a)]
        pygame.draw.polygon(screen, a.color, pts)
        pygame.draw.polygon(screen, (20, 20, 20), pts, 1)
        pygame.draw.line(screen, (250, 250, 250), pts[0], pts[1], 3)
        if idx == selected:
            pygame.draw.polygon(screen, C_SEL, pts, 3)
            if not playing:
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
        draw_button(btn_add, "+ Actor")
        draw_button(btn_del, "- Actor", enabled=(selected is not None))
        screen.blit(font.render("T=", True, C_TEXT), (18, 18))
        focused = (focus_field == "time")
        pygame.draw.rect(screen, (18, 20, 26), time_field)
        pygame.draw.rect(screen, C_BTN_HL if focused else (90, 94, 105), time_field, 2)
        shown = edit_buffer if focused else f"{T % scenario.period:.2f}"
        screen.blit(font.render(shown, True, C_TEXT), (time_field.x + 6, time_field.y + 5))
        info = f"/ {scenario.period:.2f}s   {'PLAYING' if playing else 'PAUSED'}"
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
            if m.type != "go_straight":
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
        nonlocal drag_grab, drag_orig, sel_node
        if btn_play.collidepoint(mx, my):
            playing = not playing
            return
        if btn_reset.collidepoint(mx, my):
            T = 0.0
            return
        if btn_save.collidepoint(mx, my):
            if persistence:
                fname = persistence.save_version(scenario)
                set_status(f"saved {fname} (parent v{persistence.versions[-1]['parent']})")
            return
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
                    if m.type != "go_straight" \
                            and (mx - prg[0]) ** 2 + (my - prg[1]) ** 2 < 100:
                        dragging = "right"; focus_field = None; return
            if sw.collidepoint(mx, my):
                return
        # BEV interactions (paused only)
        if not playing and TOPBAR_H < my < TOPBAR_H + CANVAS_H:
            wx, wy = s2w(mx, my)
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

    def render_frame():
        draw_map()
        for i, a in enumerate(scenario.actors):
            draw_actor(i, a)
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
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if playing:
            T += dt
        mouse_pos = pygame.mouse.get_pos()

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
                dragging = None
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
                        if sel_node is not None:
                            sel_node = None
                        else:
                            selected = None
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
    ap = argparse.ArgumentParser(description="Intersection scenario editor (v1)")
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
    main()
