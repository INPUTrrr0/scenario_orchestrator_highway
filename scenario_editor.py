#!/usr/bin/env python3
"""
Scenario Editor
===============
Load an orchestrated intersection scenario (YAML), visualize it in a bird's-eye
(BEV) pygame view, play/loop it, and edit per-maneuver timing curves.

Model (see DESIGN.md):
  * Chained path primitives: each maneuver is a geometric segment chained at the
    actually-reached end pose of the previous maneuver (no teleports).
  * Linear timing curve whose meaning is per-maneuver:
        - geometric maneuvers  -> progress-fraction vs time
        - accelerate/decelerate -> velocity vs time (slope = accel, intercept = v0)
  * Actors run maneuvers sequentially, all in parallel on one looping global
    clock; an actor that finishes its list holds its final pose until loop reset.
  * Map: a 4-way intersection (one N-S carriageway, one E-W carriageway).
  * Edits are logged to edit_history.yaml; saving writes a version-numbered file
    and records a branching provenance graph in provenance.yaml.

Coordinates: graph/Cartesian, meters, origin at intersection center,
x = East, y = North (y up), heading in degrees CCW from East.

Usage:
    python scenario_editor.py [scenario.yaml]

Controls:
    Space / Play button : play-pause (loops forever)
    Reset button        : clock -> 0
    Save button         : write next version + update provenance
    Click actor (paused): select; opens timing-curve editor below
    prev / next         : step through the selected actor's maneuvers
    drag curve endpoints or edit the slope/intercept/duration fields
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import yaml

Pose = Tuple[float, float, float]  # (x, y, heading_deg)

# Maneuver classes whose linear curve is interpreted as velocity(t) rather than
# progress(t).
VELOCITY_KINDS = {"accelerate", "decelerate"}
GEOMETRIC_KINDS = {"go_straight", "turn_left", "turn_right", "accelerate", "decelerate"}
ALL_MANEUVER_TYPES = GEOMETRIC_KINDS | {"stop"}
# order used when cycling a maneuver's type in the editor
MANEUVER_TYPE_CYCLE = ["go_straight", "turn_left", "turn_right",
                       "accelerate", "decelerate", "stop"]


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Maneuver:
    type: str
    duration: float = 1.0
    slope: float = 0.0
    intercept: float = 0.0
    # geometry params (only some apply per type)
    length: float = 0.0
    radius: float = 0.0
    angle: float = 90.0

    @property
    def curve_kind(self) -> str:
        return "velocity" if self.type in VELOCITY_KINDS else "progress"

    # progress fraction u in [0, 1] at maneuver-local time t
    def progress(self, t: float) -> float:
        if self.type == "stop":
            return 0.0
        if self.curve_kind == "progress":
            return clamp(self.slope * t + self.intercept, 0.0, 1.0)
        # velocity kind: integrate v(t) = intercept + slope*t -> distance -> u
        t_eff = t
        if self.slope < 0:  # do not let velocity go negative
            t_stop = -self.intercept / self.slope if self.slope != 0 else t
            t_eff = clamp(t, 0.0, max(0.0, t_stop))
        s = self.intercept * t_eff + 0.5 * self.slope * t_eff * t_eff
        if self.length <= 0:
            return 0.0
        return clamp(s / self.length, 0.0, 1.0)

    # world pose after travelling fraction u along this segment, given start pose
    def pose_at(self, start: Pose, u: float) -> Pose:
        sx, sy, sh = start
        h = math.radians(sh)
        if self.type == "stop":
            return (sx, sy, sh)
        if self.type in ("go_straight", "accelerate", "decelerate"):
            x = sx + self.length * u * math.cos(h)
            y = sy + self.length * u * math.sin(h)
            return (x, y, sh)
        if self.type == "turn_left":  # CCW arc
            cx = sx - self.radius * math.sin(h)
            cy = sy + self.radius * math.cos(h)
            phi0 = math.atan2(sy - cy, sx - cx)
            delta = math.radians(self.angle) * u
            x = cx + self.radius * math.cos(phi0 + delta)
            y = cy + self.radius * math.sin(phi0 + delta)
            return (x, y, sh + self.angle * u)
        if self.type == "turn_right":  # CW arc
            cx = sx + self.radius * math.sin(h)
            cy = sy - self.radius * math.cos(h)
            phi0 = math.atan2(sy - cy, sx - cx)
            delta = -math.radians(self.angle) * u
            x = cx + self.radius * math.cos(phi0 + delta)
            y = cy + self.radius * math.sin(phi0 + delta)
            return (x, y, sh - self.angle * u)
        # unknown type -> hold
        return (sx, sy, sh)

    # pose reached at end of the maneuver's own duration (used for chaining)
    def end_pose(self, start: Pose) -> Pose:
        return self.pose_at(start, self.progress(self.duration))

    def geom_length(self) -> float:
        """Arc length of the segment's geometry (meters)."""
        if self.type in ("go_straight", "accelerate", "decelerate"):
            return self.length
        if self.type in ("turn_left", "turn_right"):
            return self.radius * math.radians(self.angle)
        return 0.0

    def exit_speed(self) -> float:
        """Path speed (m/s) at the end of this maneuver — used to seed the next."""
        if self.type == "stop":
            return 0.0
        if self.curve_kind == "velocity":
            return max(0.0, self.intercept + self.slope * self.duration)
        # progress kind: path speed = arc_length * du/dt, and du/dt = slope
        return max(0.0, self.geom_length() * self.slope)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "duration": round(self.duration, 4),
                   "curve": {"slope": round(self.slope, 4),
                             "intercept": round(self.intercept, 4)}}
        if self.type in ("go_straight", "accelerate", "decelerate"):
            d["length"] = round(self.length, 4)
        elif self.type in ("turn_left", "turn_right"):
            d["radius"] = round(self.radius, 4)
            d["angle"] = round(self.angle, 4)
        return d


@dataclass
class Actor:
    id: str
    color: Tuple[int, int, int]
    length: float
    width: float
    start: Pose
    maneuvers: List[Maneuver] = field(default_factory=list)
    # precomputed:
    start_poses: List[Pose] = field(default_factory=list)
    cum: List[float] = field(default_factory=list)
    total: float = 0.0
    final_pose: Pose = (0.0, 0.0, 0.0)

    def build_path(self) -> None:
        p = self.start
        self.start_poses = []
        self.cum = [0.0]
        for m in self.maneuvers:
            self.start_poses.append(p)
            p = m.end_pose(p)
            self.cum.append(self.cum[-1] + max(0.0, m.duration))
        self.total = self.cum[-1]
        self.final_pose = p

    def active_index(self, phase: float) -> int:
        # index of maneuver active at local phase (assumes phase < total)
        for i in range(len(self.maneuvers)):
            if self.cum[i] <= phase < self.cum[i + 1]:
                return i
        return max(0, len(self.maneuvers) - 1)

    def pose_at_time(self, phase: float) -> Pose:
        if self.total <= 0 or not self.maneuvers:
            return self.final_pose or self.start
        if phase >= self.total:
            return self.final_pose
        i = self.active_index(phase)
        t_local = phase - self.cum[i]
        u = self.maneuvers[i].progress(t_local)
        return self.maneuvers[i].pose_at(self.start_poses[i], u)

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


@dataclass
class Scenario:
    map: MapConfig
    actors: List[Actor]
    pixels_per_meter: float = 6.0

    def build_paths(self) -> None:
        for a in self.actors:
            a.build_path()

    @property
    def period(self) -> float:
        return max([a.total for a in self.actors] + [1e-6])

    def to_dict(self) -> dict:
        return {"map": {"lane_width": self.map.lane_width, "arm_length": self.map.arm_length},
                "render": {"pixels_per_meter": self.pixels_per_meter},
                "actors": [a.to_dict() for a in self.actors]}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
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
        mans: List[Maneuver] = []
        for md in ad.get("maneuvers", []):
            curve = md.get("curve", {}) or {}
            mans.append(Maneuver(
                type=md["type"],
                duration=float(md.get("duration", 1.0)),
                slope=float(curve.get("slope", 0.0)),
                intercept=float(curve.get("intercept", 0.0)),
                length=float(md.get("length", 0.0)),
                radius=float(md.get("radius", 0.0)),
                angle=float(md.get("angle", 90.0)),
            ))
        actors.append(Actor(
            id=str(ad["id"]),
            color=tuple(ad.get("color", [200, 80, 80])),
            length=float(ad.get("length", 4.5)),
            width=float(ad.get("width", 2.0)),
            start=(float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                   float(st.get("heading", 0.0))),
            maneuvers=mans,
        ))
    sc = Scenario(map=mapcfg, actors=actors, pixels_per_meter=ppm)
    sc.build_paths()
    return sc


def validate_scenario(path: str) -> Scenario:
    """Load and check a scenario against the CURRENT format. Raises on problems."""
    sc = load_scenario(path)
    if not sc.actors:
        raise ValueError("scenario has no actors")
    ids = [a.id for a in sc.actors]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate actor ids: {ids}")
    for a in sc.actors:
        for i, m in enumerate(a.maneuvers):
            if m.type not in ALL_MANEUVER_TYPES:
                raise ValueError(f"actor {a.id} maneuver {i}: unknown type {m.type!r}")
            if m.duration <= 0:
                raise ValueError(f"actor {a.id} maneuver {i}: duration must be > 0")
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
            # register the loaded file as a root version
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
        self.base_version = n  # new working base; branch by reloading an earlier one
        return fname


# --------------------------------------------------------------------------- #
# GUI (pygame)
# --------------------------------------------------------------------------- #
def run_gui(scenario: Scenario, persistence: Persistence) -> None:
    import pygame

    pygame.init()
    WIDTH = 1100
    TOPBAR_H = 56
    CANVAS_H = 720          # bird's-eye view region
    SUBWIN_H = 250          # timing-curve editor, stacked BELOW the BEV
    HEIGHT = TOPBAR_H + CANVAS_H + SUBWIN_H
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Scenario Editor")
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
    man_index = 0                        # maneuver index within selected actor
    focus_field: Optional[str] = None    # 'slope' | 'intercept' | 'duration'
    edit_buffer = ""
    dragging: Optional[str] = None       # 'left'|'right' (curve) | 'spawn'|'rotate'
    drag_grab: Optional[Tuple[float, float]] = None   # world point where a spawn drag began
    drag_orig: Optional[Pose] = None                  # actor start pose at drag begin
    status_msg = ""
    status_until = 0.0

    # ---- top bar rects ----
    btn_play = pygame.Rect(WIDTH // 2 - 55, 10, 110, 36)
    btn_reset = pygame.Rect(WIDTH // 2 - 190, 10, 110, 36)
    btn_save = pygame.Rect(WIDTH // 2 + 80, 10, 110, 36)
    btn_add = pygame.Rect(WIDTH - 240, 10, 105, 36)   # add a new actor
    btn_del = pygame.Rect(WIDTH - 130, 10, 105, 36)   # remove selected actor
    time_field = pygame.Rect(52, 14, 84, 28)   # editable current-time scrubber

    # default inbound-leg spawns (SE, EN, WS, NW), each a straight-through route
    ADD_PRESETS = [(1.75, -58, 90), (58, 1.75, 180), (-58, -1.75, 0), (-1.75, 58, 270)]
    ADD_PALETTE = [(90, 190, 110), (210, 70, 60), (60, 120, 210),
                   (200, 160, 60), (160, 90, 200), (80, 200, 200)]

    # ---- subwindow geometry (computed when visible) ----
    def subwin_rect() -> pygame.Rect:
        return pygame.Rect(0, HEIGHT - SUBWIN_H, WIDTH, SUBWIN_H)

    def plot_rect() -> pygame.Rect:
        sw = subwin_rect()
        return pygame.Rect(sw.x + 55, sw.y + 56, 470, SUBWIN_H - 96)

    def geom_field_names(m: Maneuver) -> List[str]:
        if m.type in ("go_straight", "accelerate", "decelerate"):
            return ["length"]
        if m.type in ("turn_left", "turn_right"):
            return ["radius", "angle"]
        return []

    def field_rects(m: Maneuver) -> dict:
        sw = subwin_rect()
        base_y = sw.y + 56
        colA_x, colB_x = sw.x + 620, sw.x + 850
        rects = {"slope": pygame.Rect(colA_x, base_y, 90, 26),
                 "intercept": pygame.Rect(colA_x, base_y + 42, 90, 26),
                 "duration": pygame.Rect(colA_x, base_y + 84, 90, 26)}
        for i, name in enumerate(geom_field_names(m)):
            rects[name] = pygame.Rect(colB_x, base_y + i * 42, 90, 26)
        return rects

    def field_value(m: Maneuver, name: str) -> float:
        return getattr(m, name)

    def header_buttons() -> dict:
        sw = subwin_rect()
        y = sw.y + 12
        return {"type": pygame.Rect(sw.right - 420, y, 160, 26),
                "prev": pygame.Rect(sw.right - 254, y, 48, 26),
                "next": pygame.Rect(sw.right - 200, y, 48, 26),
                "add_mvr": pygame.Rect(sw.right - 146, y, 64, 26),
                "del_mvr": pygame.Rect(sw.right - 76, y, 64, 26)}

    def cur_maneuver() -> Optional[Maneuver]:
        if selected is None:
            return None
        a = scenario.actors[selected]
        if not a.maneuvers:
            return None
        return a.maneuvers[man_index]

    def set_status(msg: str) -> None:
        nonlocal status_msg, status_until
        status_msg = msg
        status_until = T + 3.0

    def next_actor_id() -> str:
        nums = []
        for a in scenario.actors:
            try:
                nums.append(int(a.id))
            except (ValueError, TypeError):
                pass
        return str(max(nums) + 1) if nums else "0"

    def do_add_actor() -> None:
        nonlocal selected, man_index
        i = len(scenario.actors)
        sx, sy, hd = ADD_PRESETS[i % len(ADD_PRESETS)]
        a = Actor(id=next_actor_id(), color=ADD_PALETTE[i % len(ADD_PALETTE)],
                  length=4.5, width=2.0, start=(sx, sy, hd),
                  maneuvers=[Maneuver(type="go_straight", duration=8.0,
                                      slope=0.125, intercept=0.0, length=116.0)])
        a.build_path()
        scenario.actors.append(a)
        persistence.log_structural("add_actor", a.id)
        selected = len(scenario.actors) - 1
        man_index = 0
        set_status(f"added actor {a.id}")

    def do_remove_actor() -> None:
        nonlocal selected, man_index
        if selected is None:
            return
        a = scenario.actors.pop(selected)
        persistence.log_structural("remove_actor", a.id)
        selected = None
        man_index = 0
        set_status(f"removed actor {a.id}")

    def do_add_maneuver() -> None:
        nonlocal man_index
        if selected is None:
            return
        a = scenario.actors[selected]
        # insert a straight segment that continues at the current exit speed
        v_in = a.maneuvers[man_index].exit_speed() if a.maneuvers else 10.0
        length = 20.0
        new_m = Maneuver(type="go_straight", duration=2.0,
                         slope=(v_in / length if v_in > 0 else 0.5),
                         intercept=0.0, length=length)
        at = man_index + 1 if a.maneuvers else 0
        a.maneuvers.insert(at, new_m)
        a.build_path()
        man_index = at
        persistence.log_structural("add_maneuver", a.id,
                                   maneuver_index=at, maneuver_type="go_straight")
        set_status(f"added maneuver to actor {a.id} at {at}")

    def do_del_maneuver() -> None:
        nonlocal man_index
        if selected is None:
            return
        a = scenario.actors[selected]
        if len(a.maneuvers) <= 1:
            set_status("cannot delete the last maneuver")
            return
        removed = a.maneuvers.pop(man_index)
        a.build_path()
        persistence.log_structural("del_maneuver", a.id,
                                   maneuver_index=man_index,
                                   maneuver_type=removed.type)
        man_index = min(man_index, len(a.maneuvers) - 1)
        set_status(f"deleted maneuver from actor {a.id}")

    def do_cycle_maneuver_type() -> None:
        nonlocal focus_field
        if selected is None:
            return
        a = scenario.actors[selected]
        m = a.maneuvers[man_index]
        old_type = m.type
        idx = MANEUVER_TYPE_CYCLE.index(old_type) if old_type in MANEUVER_TYPE_CYCLE else -1
        new_type = MANEUVER_TYPE_CYCLE[(idx + 1) % len(MANEUVER_TYPE_CYCLE)]
        old_kind = m.curve_kind
        m.type = new_type
        new_kind = m.curve_kind
        # speed the actor is carrying into this maneuver (so accel/decel are
        # continuous with the incoming speed rather than snapping to 0)
        v_in = a.maneuvers[man_index - 1].exit_speed() if man_index > 0 else 10.0
        # fill sensible geometry defaults for the new type
        if new_type in ("go_straight", "accelerate", "decelerate") and m.length <= 0:
            m.length = 20.0
        if new_type in ("turn_left", "turn_right"):
            if m.radius <= 0:
                m.radius = 5.0
            if m.angle <= 0:
                m.angle = 90.0
        # reset the curve when the curve *kind* changes (progress <-> velocity)
        if new_kind != old_kind:
            if new_kind == "progress":
                # keep the incoming speed as a constant cruise
                m.intercept = 0.0
                m.slope = (v_in / m.geom_length()) if m.geom_length() > 0 \
                    else 1.0 / max(0.05, m.duration)
            else:  # velocity: start from the incoming speed, then accel/decel
                m.intercept = v_in
                m.slope = 2.0 if new_type == "accelerate" else -2.0
        # nudge signs so accel/decel stay meaningful even without a kind change
        if new_type == "accelerate" and m.slope <= 0:
            m.slope = 2.0
        if new_type == "decelerate" and m.slope >= 0:
            m.slope = -2.0
        if new_type == "stop":
            m.slope, m.intercept = 0.0, 0.0
        a.build_path()
        focus_field = None
        persistence.log_structural("set_maneuver_type", a.id, maneuver_index=man_index,
                                   old_type=old_type, new_type=new_type)
        set_status(f"actor {a.id} maneuver {man_index}: {old_type} -> {new_type}")

    def commit_spawn_edit() -> None:
        # log the net spawn change once, on drag release
        if selected is None or drag_orig is None:
            return
        a = scenario.actors[selected]
        if a.start != drag_orig:
            persistence.log_structural(
                "move_actor", a.id,
                old_start={"x": round(drag_orig[0], 3), "y": round(drag_orig[1], 3),
                           "heading": round(drag_orig[2], 3)},
                new_start={"x": round(a.start[0], 3), "y": round(a.start[1], 3),
                           "heading": round(a.start[2], 3)})
            set_status(f"moved actor {a.id} spawn")

    # ---- editing helpers ----
    def apply_param(param: str, new_val: float) -> None:
        a = scenario.actors[selected]
        m = a.maneuvers[man_index]
        old = getattr(m, param)
        if param == "duration":
            new_val = max(0.05, new_val)
        setattr(m, param, new_val)
        a.build_path()
        persistence.log_edit(a.id, man_index, m.type, param, old, new_val)
        set_status(f"{a.id}.{m.type}.{param}: {old:.3g} -> {new_val:.3g}")

    def plot_maps(m: Maneuver, pr: pygame.Rect):
        tmax = max(1e-6, m.duration)
        if m.curve_kind == "progress":
            vmin, vmax = -0.05, 1.05
        else:
            v0, v1 = m.intercept, m.intercept + m.slope * m.duration
            vmax = max(v0, v1, 1.0) * 1.15
            vmin = min(v0, v1, 0.0) - 0.15 * abs(max(v0, v1, 1.0))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1.0

        def t2x(t): return pr.x + (t / tmax) * pr.width
        def v2y(v): return pr.bottom - (v - vmin) / (vmax - vmin) * pr.height
        def y2v(y): return vmin + (pr.bottom - y) / pr.height * (vmax - vmin)
        return t2x, v2y, y2v, tmax, vmin, vmax

    # ---- drawing ----
    def draw_map():
        screen.fill(C_GRASS)
        arm = scenario.map.arm_length
        half = scenario.map.lane_width  # half road width = one lane each way
        # E-W road
        x0, y0 = w2s(-arm, half)
        x1, y1 = w2s(arm, -half)
        pygame.draw.rect(screen, C_ROAD, pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        # N-S road
        x0, y0 = w2s(-half, arm)
        x1, y1 = w2s(half, -arm)
        pygame.draw.rect(screen, C_ROAD, pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        # dashed center lines
        dash = 3.0
        d = -arm
        while d < arm:
            if abs(d) > half:  # skip intersection box
                a1 = w2s(d, 0); a2 = w2s(min(d + dash, arm), 0)
                pygame.draw.line(screen, C_LINE, a1, a2, 2)
                b1 = w2s(0, d); b2 = w2s(0, min(d + dash, arm))
                pygame.draw.line(screen, C_LINE, b1, b2, 2)
            d += dash * 2
        # stop lines
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
        # heading indicator (front edge)
        pygame.draw.line(screen, (250, 250, 250), pts[0], pts[1], 3)
        if idx == selected:
            pygame.draw.polygon(screen, C_SEL, pts, 3)
            # spawn marker (draggable) + rotation handle, only when paused
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
        # editable current-time field: click and type a time to scrub there
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
            screen.blit(st, (sw.right - st.get_width() - 16, sw.bottom - 26))
        m = cur_maneuver()
        if playing or selected is None or m is None:
            hint = "Pause and click an actor to edit its timing curve; " \
                   "use + Actor / - Actor to add or remove."
            screen.blit(font.render(hint, True, (150, 154, 165)), (sw.x + 20, sw.y + 22))
            return
        a = scenario.actors[selected]
        title = f"Actor {a.id}   maneuver {man_index + 1}/{len(a.maneuvers)}"
        screen.blit(font_big.render(title, True, C_TEXT), (sw.x + 16, sw.y + 14))
        hb = header_buttons()
        draw_button(hb["type"], f"type: {m.type}")
        draw_button(hb["prev"], "prev")
        draw_button(hb["next"], "next")
        draw_button(hb["add_mvr"], "+mvr")
        draw_button(hb["del_mvr"], "-mvr", enabled=(len(a.maneuvers) > 1))

        pr = plot_rect()
        pygame.draw.rect(screen, (18, 20, 26), pr)
        pygame.draw.rect(screen, C_AXIS, pr, 1)
        t2x, v2y, _, tmax, vmin, vmax = plot_maps(m, pr)
        ylab = "velocity (m/s)" if m.curve_kind == "velocity" else "progress"
        screen.blit(font_sm.render(ylab, True, C_AXIS), (pr.x - 52, pr.y - 18))
        screen.blit(font_sm.render("time (s)", True, C_AXIS),
                    (pr.right - 60, pr.bottom + 6))
        # zero line for velocity
        if vmin < 0 < vmax:
            zy = v2y(0)
            pygame.draw.line(screen, (70, 74, 85), (pr.x, zy), (pr.right, zy), 1)
        # the curve
        p_left = (t2x(0), v2y(m.intercept))
        p_right = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
        pygame.draw.line(screen, C_CURVE, p_left, p_right, 2)
        pygame.draw.circle(screen, C_SEL, (int(p_left[0]), int(p_left[1])), 6)
        pygame.draw.circle(screen, C_SEL, (int(p_right[0]), int(p_right[1])), 6)
        # current-time marker if this maneuver is active now
        phase = T % scenario.period
        if a.cum[man_index] <= phase < a.cum[man_index + 1]:
            tl = phase - a.cum[man_index]
            mx = t2x(tl)
            pygame.draw.line(screen, (250, 120, 120), (mx, pr.y), (mx, pr.bottom), 1)

        # fields: timing (slope/intercept/duration) + geometry (length | radius,angle)
        fr = field_rects(m)
        for key, rect in fr.items():
            screen.blit(font.render(key, True, C_TEXT), (rect.x - 78, rect.y + 4))
            focused = (focus_field == key)
            pygame.draw.rect(screen, (18, 20, 26), rect)
            pygame.draw.rect(screen, C_BTN_HL if focused else (90, 94, 105), rect, 2)
            shown = edit_buffer if focused else f"{field_value(m, key):.4g}"
            screen.blit(font.render(shown, True, C_TEXT), (rect.x + 6, rect.y + 4))
        hint = "drag curve endpoints or a field + type (Enter). +/- mvr add/delete."
        screen.blit(font_sm.render(hint, True, (140, 144, 155)),
                    (pr.x, pr.bottom + 10))

    # ---- event handling ----
    def handle_click(mx, my):
        nonlocal playing, T, selected, man_index, focus_field, edit_buffer, dragging
        nonlocal drag_grab, drag_orig
        if btn_play.collidepoint(mx, my):
            playing = not playing
            return
        if btn_reset.collidepoint(mx, my):
            T = 0.0
            return
        if btn_save.collidepoint(mx, my):
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
        if not playing and selected is not None and cur_maneuver() is not None:
            sw = subwin_rect()
            hb = header_buttons()
            a = scenario.actors[selected]
            if hb["prev"].collidepoint(mx, my):
                man_index = (man_index - 1) % len(a.maneuvers)
                focus_field = None
                return
            if hb["next"].collidepoint(mx, my):
                man_index = (man_index + 1) % len(a.maneuvers)
                focus_field = None
                return
            if hb["add_mvr"].collidepoint(mx, my):
                do_add_maneuver(); focus_field = None; return
            if hb["del_mvr"].collidepoint(mx, my):
                do_del_maneuver(); focus_field = None; return
            if hb["type"].collidepoint(mx, my):
                do_cycle_maneuver_type(); return
            m = cur_maneuver()
            for key, rect in field_rects(m).items():
                if rect.collidepoint(mx, my):
                    focus_field = key
                    edit_buffer = ""
                    return
            pr = plot_rect()
            t2x, v2y, _, tmax, _, _ = plot_maps(m, pr)
            pl = (t2x(0), v2y(m.intercept))
            prg = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
            if (mx - pl[0]) ** 2 + (my - pl[1]) ** 2 < 100:
                dragging = "left"; focus_field = None; return
            if (mx - prg[0]) ** 2 + (my - prg[1]) ** 2 < 100:
                dragging = "right"; focus_field = None; return
            if sw.collidepoint(mx, my):
                return  # click inside panel, no actor pick
        # BEV interactions (paused only) — canvas region, never the subwindow
        if not playing and TOPBAR_H < my < TOPBAR_H + CANVAS_H:
            wx, wy = s2w(mx, my)
            # spawn drag / rotate on the selected actor's start marker
            if selected is not None:
                a = scenario.actors[selected]
                hpt = w2s(*rotation_handle_world(a))
                if (mx - hpt[0]) ** 2 + (my - hpt[1]) ** 2 < 100:
                    dragging = "rotate"; drag_orig = a.start; focus_field = None
                    T = 0.0  # jump to spawn so the marker and body coincide
                    return
                if point_in_pose(a, a.start, wx, wy):
                    dragging = "spawn"; drag_grab = (wx, wy); drag_orig = a.start
                    focus_field = None
                    T = 0.0
                    return
            # otherwise pick an actor by its (moving) body
            phase = T % scenario.period
            for i, a in enumerate(scenario.actors):
                if point_in_pose(a, a.pose_at_time(phase), wx, wy):
                    selected = i
                    man_index = a.active_index(phase) if a.total > phase else 0
                    focus_field = None
                    return

    def handle_drag(mx, my):
        # spawn move / heading rotate on the selected actor
        if dragging in ("spawn", "rotate") and selected is not None:
            a = scenario.actors[selected]
            wx, wy = s2w(mx, my)
            if dragging == "spawn" and drag_grab is not None:
                nx = drag_orig[0] + (wx - drag_grab[0])
                ny = drag_orig[1] + (wy - drag_grab[1])
                a.start = (nx, ny, a.start[2])
            else:  # rotate about the start point
                hd = math.degrees(math.atan2(wy - a.start[1], wx - a.start[0]))
                a.start = (a.start[0], a.start[1], round(hd, 1))
            a.build_path()
            return
        m = cur_maneuver()
        if m is None:
            return
        pr = plot_rect()
        _, _, y2v, tmax, _, _ = plot_maps(m, pr)
        val = y2v(clamp(my, pr.y, pr.bottom))
        if m.curve_kind == "progress":
            val = clamp(val, 0.0, 1.0)
        if dragging == "left":
            apply_param("intercept", val)
        elif dragging == "right":
            new_slope = (val - m.intercept) / max(1e-6, m.duration)
            apply_param("slope", new_slope)

    def commit_field():
        nonlocal focus_field, edit_buffer, T
        if focus_field is None:
            return
        try:
            val = float(edit_buffer)
            if focus_field == "time":
                T = max(0.0, val)
            else:
                apply_param(focus_field, val)
        except ValueError:
            set_status("invalid number")
        focus_field = None
        edit_buffer = ""

    # ---- main loop ----
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if playing:
            T += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handle_click(*event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging in ("spawn", "rotate"):
                    commit_spawn_edit()
                dragging = None
                drag_grab = None
                drag_orig = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                handle_drag(*event.pos)
            elif event.type == pygame.KEYDOWN:
                if focus_field is not None:
                    if event.key == pygame.K_RETURN:
                        commit_field()
                    elif event.key == pygame.K_ESCAPE:
                        focus_field = None; edit_buffer = ""
                    elif event.key == pygame.K_BACKSPACE:
                        edit_buffer = edit_buffer[:-1]
                    elif event.unicode in "0123456789.-+eE":
                        edit_buffer += event.unicode
                else:
                    if event.key == pygame.K_SPACE:
                        playing = not playing
                    elif event.key == pygame.K_ESCAPE:
                        selected = None
                    elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) \
                            and selected is not None:
                        do_remove_actor()
                    elif event.key == pygame.K_a:
                        do_add_actor()

        draw_map()
        for i, a in enumerate(scenario.actors):
            draw_actor(i, a)
        draw_subwindow()   # always stacked below the BEV (no overlap)
        draw_topbar()
        pygame.display.flip()

    pygame.quit()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Intersection scenario editor")
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, "scenarios")
    ap.add_argument("scenario", nargs="?",
                    default=os.path.join(default_dir, "scenario_v1.yaml"),
                    help="path to a scenario YAML")
    ap.add_argument("--scenarios-dir", default=None,
                    help="folder for versioned saves / provenance (default: scenario's folder)")
    ap.add_argument("--validate", action="store_true",
                    help="load the scenario, report validity against the current format, and exit")
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
    sdir = args.scenarios_dir or os.path.dirname(os.path.abspath(args.scenario))
    persistence = Persistence(sdir, args.scenario)
    run_gui(scenario, persistence)


if __name__ == "__main__":
    main()
