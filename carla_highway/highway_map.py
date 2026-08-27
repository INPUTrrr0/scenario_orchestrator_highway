#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/highway_map.py — fit the highway script frame onto a real road.

The counterpart of `carla_port/carla_map.py`. That module fits ONE signalized
junction; this one fits ONE straight, multi-lane stretch of road, because that
is the world the highway orchestrator's scripts live in:

    highway/maps.py  MapConfig(kind="straight", num_lanes=N, lane_width=w,
                               length=L)
    lane_center_x(i) = -N*w/2 + w/2 + i*w        (i = 0 is the leftmost lane)

Coordinate convention
---------------------
Identical to `carla_port.carla_map.IntersectionFrame`, deliberately — the two
frames are interchangeable everywhere the CARLA-side mechanics are used
(`carla_adapter`, `carla_sync`, `carla_video`, `carla_obs`), and duplicating the
convention with a different sign somewhere would be a silent disaster.

    script frame:  origin at the road centre, x to the driver's RIGHT,
                   y along the direction of travel, heading in degrees CCW
                   from +x (so the scripts' `heading: 90` is "forward").

    X = a.x + x_s*cos(theta) + y_s*sin(theta)
    Y = a.y + x_s*sin(theta) - y_s*cos(theta)
    yaw = theta - h_s

`theta` is the CARLA yaw of the script's +x axis. Forward (`h_s = 90`) must map
to the ego lane's travel direction `D`, so **theta = D + 90**; that also puts
script +x 90 degrees clockwise of travel in CARLA's left-handed frame, i.e. on
the driver's right, as the scripts assume.

`lane_width` is measured from the real lane centre spacing rather than taken
from `Waypoint.lane_width`, for the same reason `carla_map` does it: the script
places actors at `lane_center_x(i)` and those places have to be real lanes.

Scenario shapes
---------------
`scenario_cutin` and `scenario_hard_brake` need N same-direction lanes;
`scenario_overtake` needs an oncoming lane, because the ego has to use it to
get around the stopped blocker. `discover()` takes both requirements
(`lanes`, `two_way`) and picks the road section that fits them best.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from carla_port.carla_api import carla

from .script_bridge import se

#: script heading (deg CCW from +x) that means "forward" to every highway script
FORWARD_HEADING = 90.0

#: How far the real lane may wander from the script's straight line before
#: the fit stops extending. Half a lane is far too much — an actor placed at
#: the far end would straddle the marking — so this is kept well inside one.
MAX_LATERAL_DEV = 0.35   # metres


def _norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Lanes
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    """One driving lane of the fitted road section, in script coordinates."""
    lane_id: int
    offset: float                 # script x of the lane centre
    same_direction: bool          # travels with the ego
    width: float                  # CARLA's own lane width, for reference
    waypoint: object = field(repr=False, default=None)   # at the frame origin

    @property
    def heading(self) -> float:
        """Script heading of traffic in this lane."""
        return FORWARD_HEADING if self.same_direction else (FORWARD_HEADING + 180.0) % 360.0


# --------------------------------------------------------------------------- #
# The frame
# --------------------------------------------------------------------------- #
class HighwayFrame:
    """Placement of the highway scripts' straight road onto a real CARLA road.

    Exposes the same conversion surface as
    `carla_port.carla_map.IntersectionFrame`, so the script-agnostic CARLA
    mechanics accept either.
    """

    def __init__(self, world, anchor, theta: float, lane_width: float,
                 length: float, lanes: Sequence[Lane],
                 road_id: Optional[int] = None, section_id: Optional[int] = None,
                 warnings: Optional[List[str]] = None):
        self.world = world
        self.anchor = anchor                      # carla.Location, road centre
        self.theta = float(theta)                 # deg, CARLA yaw of script +x
        self.lane_width = float(lane_width)
        self.length = float(length)
        self.lanes = list(lanes)
        self.road_id = road_id
        self.section_id = section_id
        self.warnings = list(warnings or [])
        self._ct = math.cos(math.radians(self.theta))
        self._st = math.sin(math.radians(self.theta))

    # ---- the shape the script layer sees ---- #
    @property
    def num_lanes(self) -> int:
        return len(self.lanes)

    @property
    def arm_length(self) -> float:
        """Half the usable road length.

        Not a highway concept; provided because the script `MapConfig` carries
        the field and because `carla_port.carla_video.default_top_span` frames
        the bird's-eye view with it.
        """
        return self.length / 2.0

    def map_config(self) -> "se.MapConfig":
        return se.MapConfig(lane_width=self.lane_width, kind="straight",
                            num_lanes=self.num_lanes, length=self.length)

    def lane_center_x(self, index: int) -> float:
        """Script x of lane `index` (0 = leftmost), as the script computes it."""
        return self.map_config().lane_center_x(index)

    def same_direction_lanes(self) -> List[Lane]:
        return [ln for ln in self.lanes if ln.same_direction]

    def oncoming_lanes(self) -> List[Lane]:
        return [ln for ln in self.lanes if not ln.same_direction]

    def lane_index_of(self, x: float) -> int:
        """Index of the lane whose centre is nearest script x."""
        n = max(1, self.num_lanes)
        best, best_d = 0, float("inf")
        for i in range(n):
            d = abs(self.lane_center_x(i) - x)
            if d < best_d:
                best, best_d = i, d
        return best

    def two_way_ok(self, want_forward: bool) -> bool:
        """Does the fitted road have a lane running the requested way?"""
        return any(ln.same_direction == want_forward for ln in self.lanes)

    def lane_index_for_direction(self, want_forward: bool,
                                 prefer: int = 0) -> Optional[int]:
        """Index of the lane running `want_forward`, nearest to `prefer`."""
        cands = [i for i, ln in enumerate(self.lanes)
                 if ln.same_direction == want_forward]
        if not cands:
            return None
        return min(cands, key=lambda i: abs(i - prefer))

    def heading_of_lane(self, index: int) -> float:
        """Script heading of traffic in lane `index`."""
        if 0 <= index < len(self.lanes):
            return self.lanes[index].heading
        return FORWARD_HEADING

    # ---- coordinate conversion (the ONE place it happens) ---- #
    def to_carla_xy(self, x: float, y: float) -> Tuple[float, float]:
        return (self.anchor.x + x * self._ct + y * self._st,
                self.anchor.y + x * self._st - y * self._ct)

    def to_carla_yaw(self, heading: float) -> float:
        return _norm180(self.theta - heading)

    def to_carla_location(self, x: float, y: float, z: Optional[float] = None):
        X, Y = self.to_carla_xy(x, y)
        return carla.Location(x=X, y=Y, z=self.anchor.z if z is None else z)

    def to_carla_transform(self, x: float, y: float, heading: float,
                           z: Optional[float] = None):
        return carla.Transform(self.to_carla_location(x, y, z),
                               carla.Rotation(pitch=0.0,
                                              yaw=self.to_carla_yaw(heading),
                                              roll=0.0))

    def from_carla_xy(self, X: float, Y: float) -> Tuple[float, float]:
        dx, dy = X - self.anchor.x, Y - self.anchor.y
        return (dx * self._ct + dy * self._st, dx * self._st - dy * self._ct)

    def from_carla_yaw(self, yaw: float) -> float:
        return (self.theta - yaw) % 360.0

    def from_carla_transform(self, tf) -> Tuple[float, float, float]:
        x, y = self.from_carla_xy(tf.location.x, tf.location.y)
        return (x, y, self.from_carla_yaw(tf.rotation.yaw))

    def to_carla_velocity(self, speed: float, heading: float):
        r = math.radians(self.to_carla_yaw(heading))
        return carla.Vector3D(x=speed * math.cos(r), y=speed * math.sin(r), z=0.0)

    def from_carla_velocity(self, v) -> float:
        return math.hypot(v.x, v.y)

    # ---- extent ---- #
    def on_road(self, x: float, y: float, margin: float = 0.5) -> bool:
        """Is a script point inside the fitted stretch of road?"""
        half_w = self.num_lanes * self.lane_width / 2.0 + margin
        return abs(x) <= half_w and abs(y) <= self.length / 2.0 + margin

    def ground_z(self, x: float, y: float, fallback: Optional[float] = None) -> float:
        """Road surface height under a script point."""
        z = self.anchor.z if fallback is None else fallback
        try:
            wp = self.world.get_map().get_waypoint(self.to_carla_location(x, y, z),
                                                   project_to_road=True)
        except RuntimeError:
            return z
        return wp.transform.location.z if wp is not None else z

    # ---- compatibility shims for the shared CARLA mechanics ---- #
    def traffic_light(self, arm: Optional[str] = None):
        """No signals on the fitted highway section.

        `carla_port.carla_obs` calls this when it has an `ego_arm`; the highway
        port never sets one, so this exists only to keep the surface identical.
        """
        return None

    # ---- diagnostics ---- #
    def lane_fit_error(self) -> float:
        """Largest gap between a real lane centre and where the script's
        `lane_center_x` puts that lane. The number that says whether scripted
        actors will land on real asphalt."""
        if not self.lanes:
            return 0.0
        return max(abs(ln.offset - self.lane_center_x(i))
                   for i, ln in enumerate(self.lanes))

    def describe(self) -> str:
        out = [f"road {self.road_id}.{self.section_id} @ "
               f"({self.anchor.x:.1f}, {self.anchor.y:.1f}, {self.anchor.z:.1f})",
               f"theta={self.theta:.2f} deg  lane_width={self.lane_width:.2f} m  "
               f"num_lanes={self.num_lanes}  length={self.length:.1f} m  "
               f"lane_fit_err={self.lane_fit_error():.2f} m"]
        for i, ln in enumerate(self.lanes):
            out.append(f"  lane {i}: carla_id={ln.lane_id:+d} "
                       f"x={ln.offset:+.2f} (script {self.lane_center_x(i):+.2f}) "
                       f"w={ln.width:.2f} "
                       f"{'forward' if ln.same_direction else 'ONCOMING'}")
        for w in self.warnings:
            out.append(f"  ! {w}")
        return "\n".join(out)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    @classmethod
    def discover(cls, world, lanes: int = 3, two_way: bool = False,
                 min_length: float = 120.0, road_id: Optional[int] = None,
                 seed_step: float = 5.0, straight_tol_deg: float = 4.0,
                 walk_step: float = 5.0, max_length: float = 400.0,
                 through_junctions: bool = True,
                 max_lateral_dev: float = MAX_LATERAL_DEV
                 ) -> "HighwayFrame":
        """Fit the script's straight road onto the best-matching CARLA section.

        `lanes` is how many lanes the scenario needs. With `two_way` the set
        must contain at least one oncoming lane (scenario_overtake); otherwise
        every lane must run the same way (cutin / hard_brake).

        Selection prefers, in order: enough lanes of the right kind, the
        straightest section, then the longest, then the most evenly spaced.
        """
        cmap = world.get_map()
        cands = _candidate_sections(cmap, seed_step=seed_step, road_id=road_id)
        if not cands:
            raise RuntimeError("no non-junction driving lanes found in this map")

        scored: List[Tuple[tuple, dict]] = []
        rejected: Dict[str, int] = {}
        for seed in cands:
            fit = _fit_section(seed, lanes=lanes, two_way=two_way,
                               min_length=min_length, straight_tol_deg=straight_tol_deg,
                               walk_step=walk_step, max_length=max_length,
                               through_junctions=through_junctions,
                               max_lateral_dev=max_lateral_dev)
            if isinstance(fit, str):
                rejected[fit] = rejected.get(fit, 0) + 1
                continue
            scored.append((fit["key"], fit))

        if not scored:
            why = ", ".join(f"{k} x{v}" for k, v in
                            sorted(rejected.items(), key=lambda kv: -kv[1]))
            raise RuntimeError(
                f"no straight section with {lanes} "
                f"{'two-way' if two_way else 'same-direction'} lanes and "
                f"{min_length:.0f} m of run in this map (rejections: {why})")

        scored.sort(key=lambda s: s[0])
        best = scored[0][1]
        return cls(world=world, anchor=best["anchor"], theta=best["theta"],
                   lane_width=best["lane_width"], length=best["length"],
                   lanes=best["lanes"], road_id=best["road_id"],
                   section_id=best["section_id"], warnings=best["warnings"])


# --------------------------------------------------------------------------- #
# Discovery helpers
# --------------------------------------------------------------------------- #
def _candidate_sections(cmap, seed_step: float, road_id: Optional[int]) -> List:
    """One representative waypoint per (road, section, lane) group.

    `generate_waypoints` hands back a dense grid; collapsing it to one seed per
    road section keeps the fit O(sections) instead of O(waypoints).
    """
    seen = set()
    out = []
    for wp in cmap.generate_waypoints(seed_step):
        if wp.is_junction:
            continue
        if getattr(wp, "lane_type", None) is not None and \
                wp.lane_type != carla.LaneType.Driving:
            continue
        if road_id is not None and wp.road_id != road_id:
            continue
        key = (wp.road_id, wp.section_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(wp)
    return out


def _siblings(wp, max_lanes: int = 10) -> List:
    """Every driving lane waypoint in `wp`'s road section.

    Walks `get_left_lane` / `get_right_lane` from `wp`, deduping by lane id.
    Ordering is NOT trusted: crossing the centre line flips what "left" means,
    so the caller re-derives each lane's position geometrically.
    """
    found = {wp.lane_id: wp}
    for step in ("left", "right"):
        cur, n = wp, 0
        while n < max_lanes:
            n += 1
            try:
                nxt = cur.get_left_lane() if step == "left" else cur.get_right_lane()
            except (RuntimeError, AttributeError):
                break
            if nxt is None or nxt.road_id != wp.road_id or \
                    nxt.section_id != wp.section_id or nxt.lane_id in found:
                break
            if getattr(nxt, "lane_type", None) is not None and \
                    nxt.lane_type != carla.LaneType.Driving:
                # a non-driving lane (shoulder, median) still separates lanes;
                # keep walking past it without recording it
                cur = nxt
                continue
            found[nxt.lane_id] = nxt
            cur = nxt
    return list(found.values())


def _straight_run(wp, forward: bool, step: float, tol_deg: float,
                  max_dist: float, through_junctions: bool = True,
                  max_lateral_dev: float = MAX_LATERAL_DEV) -> float:
    """Metres of lane from `wp` that stay within `max_lateral_dev` of straight.

    At a fork (including inside a junction, where `next()` offers one waypoint
    per turn) the continuation is the branch whose heading best matches the
    start heading — driving straight on. Without that the run stops dead at the
    first cross street, which is why a town map appears to have no straight
    longer than ~20 m even along a road that visibly runs for hundreds.

    The binding criterion is **cumulative lateral deviation**, not per-step
    heading error. The script frame is a straight line; what breaks a scenario
    is an actor placed far from the anchor ending up off the real lane, and
    that is exactly this distance. A heading tolerance alone does not bound it:
    4 degrees per 5 m step integrates to metres over a 200 m fit, which is how
    Town05 produced actors 1.46 m off the lane centre while every per-step
    heading check passed.

    `through_junctions=False` restores the strict behaviour (stop at any
    junction), for a scenario that must not cross traffic.
    """
    tf0 = wp.transform
    yaw0 = tf0.rotation.yaw
    r = math.radians(yaw0)
    fx, fy = math.cos(r), math.sin(r)
    ox, oy = tf0.location.x, tf0.location.y
    cur, dist = wp, 0.0
    while dist < max_dist:
        nxts = cur.next(step) if forward else cur.previous(step)
        nxts = [w for w in (nxts or [])]
        if not through_junctions:
            nxts = [w for w in nxts if not w.is_junction]
        if not nxts:
            break
        # keep going straight: the branch closest to the original heading
        nxt = min(nxts, key=lambda w: abs(_norm180(w.transform.rotation.yaw - yaw0)))
        if abs(_norm180(nxt.transform.rotation.yaw - yaw0)) > tol_deg:
            break
        L = nxt.transform.location
        # perpendicular distance from the straight axis through the seed
        dev = abs(-(L.x - ox) * fy + (L.y - oy) * fx)
        if dev > max_lateral_dev:
            break
        cur, dist = nxt, dist + step
    return dist


def _fit_section(seed, lanes: int, two_way: bool, min_length: float,
                 straight_tol_deg: float, walk_step: float, max_length: float,
                 through_junctions: bool = True,
                 max_lateral_dev: float = MAX_LATERAL_DEV):
    """Try to fit the script road onto `seed`'s section.

    Returns a fit dict, or a short string naming why it was rejected.
    """
    sibs = _siblings(seed)
    if len(sibs) < lanes:
        return "too-few-lanes"

    # Geometry in the seed's own frame: +x to the seed's right, +y forward.
    yaw0 = seed.transform.rotation.yaw
    r = math.radians(yaw0)
    fx, fy = math.cos(r), math.sin(r)          # forward unit vector (CARLA)
    o = seed.transform.location

    entries = []
    for wp in sibs:
        L = wp.transform.location
        dx, dy = L.x - o.x, L.y - o.y
        # CARLA is left-handed, so right-of-travel is forward rotated +90 in
        # yaw: R = (-sin r, cos r) = (-fy, fx). That is exactly the script's
        # +x axis (theta = yaw0 + 90), so this projection IS the script x.
        lateral = -dx * fy + dy * fx
        same = abs(_norm180(wp.transform.rotation.yaw - yaw0)) < 90.0
        entries.append({"wp": wp, "lat": lateral, "same": same,
                        "width": float(getattr(wp, "lane_width", 3.5) or 3.5),
                        "lane_id": int(wp.lane_id)})

    same_dir = [e for e in entries if e["same"]]
    if not same_dir:
        return "no-forward-lane"
    if two_way:
        if not any(not e["same"] for e in entries):
            return "not-two-way"
        pool = entries
    else:
        pool = same_dir
    if len(pool) < lanes:
        return "too-few-lanes-of-kind"

    # Keep the `lanes` lanes closest together laterally, always including the
    # seed's own lane, so a 3-lane pick out of a 4-lane road stays contiguous.
    pool.sort(key=lambda e: e["lat"])
    best_window = None
    for i in range(0, len(pool) - lanes + 1):
        win = pool[i:i + lanes]
        if two_way and not any(not e["same"] for e in win):
            continue
        if two_way and not any(e["same"] for e in win):
            continue
        spread = win[-1]["lat"] - win[0]["lat"]
        if best_window is None or spread < best_window[0]:
            best_window = (spread, win)
    if best_window is None:
        return "no-contiguous-window"
    win = best_window[1]

    # Even spacing is what makes MapConfig.lane_center_x land on real lanes.
    gaps = [win[k + 1]["lat"] - win[k]["lat"] for k in range(len(win) - 1)]
    lane_width = (sum(gaps) / len(gaps)) if gaps else \
        float(win[0]["width"])
    if lane_width <= 0.5:
        return "degenerate-lane-width"
    spacing_err = max((abs(g - lane_width) for g in gaps), default=0.0)

    # Longitudinal run, measured on the seed lane and capped by the shortest
    # sibling: an actor in any lane has to have road under it for the whole run.
    fwd = min(_straight_run(e["wp"], True, walk_step, straight_tol_deg,
                            max_length / 2.0, through_junctions, max_lateral_dev)
              for e in win)
    bwd = min(_straight_run(e["wp"], False, walk_step, straight_tol_deg,
                            max_length / 2.0, through_junctions, max_lateral_dev)
              for e in win)
    length = min(2.0 * min(fwd, bwd), max_length)
    if length < min_length:
        return "too-short"

    # The ego travels with the seed lane; the frame's forward is that direction.
    theta = yaw0 + FORWARD_HEADING

    # Anchor: the lateral centroid of the chosen lanes, at the seed's station.
    centroid_lat = sum(e["lat"] for e in win) / len(win)
    # move from the seed to the centroid along the seed's right-hand direction
    rx, ry = -fy, fx                            # CARLA right-of-travel unit vector
    anchor = carla.Location(x=o.x + centroid_lat * rx,
                            y=o.y + centroid_lat * ry,
                            z=o.z)

    lane_objs = [Lane(lane_id=e["lane_id"], offset=e["lat"] - centroid_lat,
                      same_direction=e["same"], width=e["width"], waypoint=e["wp"])
                 for e in win]

    warnings: List[str] = []
    if spacing_err > 0.35:
        warnings.append(f"lane spacing varies by {spacing_err:.2f} m; "
                        "scripted lane centres may sit off-centre")
    n_on = sum(1 for ln in lane_objs if ln.same_direction)
    if two_way and n_on == len(lane_objs):
        warnings.append("two-way requested but every chosen lane runs forward")

    # Lower is better on every component.
    key = (round(spacing_err, 2), -round(length, 0), round(best_window[0], 1))
    return {"key": key, "anchor": anchor, "theta": theta,
            "lane_width": lane_width, "length": length, "lanes": lane_objs,
            "road_id": int(seed.road_id), "section_id": int(seed.section_id),
            "warnings": warnings}
