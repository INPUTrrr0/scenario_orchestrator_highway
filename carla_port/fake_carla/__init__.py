#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/fake_carla — an offline test double for the CARLA 0.9.16 API.

Purpose: run the port's validation suite (coordinate round-trip, maneuver
placement, retime, reroute, rebasing, predicted vs realized collision) on a
machine with no CARLA server. It is NOT a simulator: it advances a frame
counter, holds whatever transforms are written to it, and reports oriented-box
overlaps between vehicles as collision events. Nothing here is used by
carla_runner.py against a real server.

The synthetic map is a signalized 4-way intersection whose geometry is the
script layer's own (v2/directives.route_path), deliberately placed at a
non-trivial anchor and rotation so the coordinate conversion is actually
exercised rather than accidentally being the identity.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return f"Vector3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Location(Vector3D):
    def distance(self, other) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2
                         + (self.z - other.z) ** 2)

    def __repr__(self):
        return f"Location(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)

    def __repr__(self):
        return f"Rotation(pitch={self.pitch:.2f}, yaw={self.yaw:.2f}, roll={self.roll:.2f})"


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()

    def __repr__(self):
        return f"Transform({self.location}, {self.rotation})"


class BoundingBox:
    def __init__(self, location=None, extent=None):
        self.location = location if location is not None else Location()
        self.extent = extent if extent is not None else Vector3D(2.25, 1.0, 0.75)
        self.rotation = Rotation()


class LaneType(Enum):
    NONE = 0
    Driving = 1
    Any = 2


class AttachmentType(Enum):
    Rigid = 0
    SpringArm = 1


class TrafficLightState(Enum):
    Red = 0
    Yellow = 1
    Green = 2
    Off = 3
    Unknown = 4

    def __str__(self):
        return f"TrafficLightState.{self.name}"


class WorldSettings:
    def __init__(self, synchronous_mode=False, no_rendering_mode=False,
                 fixed_delta_seconds=0.0):
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.substepping = True
        self.max_substep_delta_time = 0.01
        self.max_substeps = 10


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Map: lanes as polylines, waypoints as (lane, s)
# --------------------------------------------------------------------------- #
class _Lane:
    def __init__(self, lane_id_tuple, pts, is_junction, junction_id,
                 lane_width, successors=None, predecessors=None):
        self.road_id, self.lane_id, self.section_id = lane_id_tuple
        self.pts = pts
        self.is_junction = is_junction
        self.junction_id = junction_id
        self.lane_width = lane_width
        self.successors: List["_Lane"] = successors or []
        self.predecessors: List["_Lane"] = predecessors or []
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + math.dist(pts[i - 1], pts[i]))
        self.cum = cum
        self.length = cum[-1]

    def at(self, s: float) -> Tuple[float, float, float]:
        s = min(max(s, 0.0), self.length)
        i = 0
        while i + 2 < len(self.pts) and self.cum[i + 1] < s:
            i += 1
        span = self.cum[i + 1] - self.cum[i]
        f = (s - self.cum[i]) / span if span > 1e-12 else 0.0
        ax, ay = self.pts[i]
        bx, by = self.pts[i + 1]
        h = math.degrees(math.atan2(by - ay, bx - ax))
        return (ax + f * (bx - ax), ay + f * (by - ay), h)


class Waypoint:
    _next_id = [1]

    def __init__(self, lane: _Lane, s: float, world_map: "Map"):
        self._lane, self._s, self._map = lane, min(max(s, 0.0), lane.length), world_map
        self.id = Waypoint._next_id[0]
        Waypoint._next_id[0] += 1

    # ---- attributes ---- #
    @property
    def road_id(self): return self._lane.road_id

    @property
    def lane_id(self): return self._lane.lane_id

    @property
    def section_id(self): return self._lane.section_id

    @property
    def s(self): return self._s

    @property
    def is_junction(self): return self._lane.is_junction

    @property
    def junction_id(self): return self._lane.junction_id

    @property
    def lane_width(self): return self._lane.lane_width

    @property
    def lane_type(self): return LaneType.Driving

    @property
    def transform(self) -> Transform:
        x, y, h = self._lane.at(self._s)
        X, Y = self._map.script_to_world(x, y)
        return Transform(Location(X, Y, self._map.z0),
                         Rotation(0.0, self._map.script_yaw_to_world(h), 0.0))

    def get_junction(self):
        return self._map.junction

    # ---- traversal ---- #
    def next(self, distance: float) -> List["Waypoint"]:
        s = self._s + distance
        if s <= self._lane.length:
            return [Waypoint(self._lane, s, self._map)]
        over = s - self._lane.length
        return [Waypoint(nl, min(over, nl.length), self._map)
                for nl in self._lane.successors]

    def previous(self, distance: float) -> List["Waypoint"]:
        s = self._s - distance
        if s >= 0.0:
            return [Waypoint(self._lane, s, self._map)]
        over = -s
        return [Waypoint(pl, max(pl.length - over, 0.0), self._map)
                for pl in self._lane.predecessors]

    def __repr__(self):
        return (f"Waypoint(road={self.road_id}, lane={self.lane_id}, "
                f"s={self._s:.1f}, junction={self.is_junction})")


class Junction:
    def __init__(self, jid, bbox, pairs):
        self.id = jid
        self.bounding_box = bbox
        self._pairs = pairs

    def get_waypoints(self, lane_type=LaneType.Driving):
        return list(self._pairs)


class Map:
    """Synthetic signalized 4-way intersection in script geometry, rigidly
    placed at (x0, y0, z0) with rotation `theta` (CARLA yaw of script +x)."""

    ARMS = ("N", "E", "S", "W")
    IN_LEG = {"S": "SE", "E": "EN", "N": "NW", "W": "WS"}
    ROUTES = {"SE": {"straight": "NE", "right": "ES", "left": "WN"},
              "EN": {"straight": "WN", "right": "NE", "left": "SW"},
              "NW": {"straight": "SW", "right": "WN", "left": "ES"},
              "WS": {"straight": "ES", "right": "SW", "left": "NE"}}
    OUT_ARM = {"NE": "N", "ES": "E", "WN": "W", "SW": "S"}

    def __init__(self, name="FakeTown", lane_width=3.5, arm_length=60.0,
                 x0=137.0, y0=-42.0, z0=0.4, theta=37.0):
        self.name = name
        self.lane_width = lane_width
        self.arm_length = arm_length
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.theta = theta
        self._ct = math.cos(math.radians(theta))
        self._st = math.sin(math.radians(theta))
        self._build()

    # ---- placement ---- #
    def script_to_world(self, x, y):
        return (self.x0 + x * self._ct + y * self._st,
                self.y0 + x * self._st - y * self._ct)

    def world_to_script(self, X, Y):
        dx, dy = X - self.x0, Y - self.y0
        return (dx * self._ct + dy * self._st, dx * self._st - dy * self._ct)

    def script_yaw_to_world(self, heading):
        return _norm180(self.theta - heading)

    def world_yaw_to_script(self, yaw):
        return (self.theta - yaw) % 360.0

    # ---- construction ---- #
    def _build(self):
        lw, arm = self.lane_width, self.arm_length
        h = lw / 2.0
        # approach (inbound) and exit (outbound) lane geometry, script frame
        approach = {                              # arm -> (start, end)
            "S": ((h, -arm), (h, -lw)),
            "N": ((-h, arm), (-h, lw)),
            "E": ((arm, h), (lw, h)),
            "W": ((-arm, -h), (-lw, -h)),
        }
        exit_ = {                                 # outbound leg -> (start, end)
            "NE": ((h, lw), (h, arm)),
            "SW": ((-h, -lw), (-h, -arm)),
            "ES": ((lw, -h), (arm, -h)),
            "WN": ((-lw, h), (-arm, h)),
        }
        self.lanes: Dict[str, _Lane] = {}
        rid = 0
        for a, (p0, p1) in approach.items():
            self.lanes[f"app_{a}"] = _Lane((rid, -1, 0), [p0, p1], False, -1, lw)
            rid += 1
        for leg, (p0, p1) in exit_.items():
            self.lanes[f"exit_{leg}"] = _Lane((rid, -1, 0), [p0, p1], False, -1, lw)
            rid += 1
        self.junction_id = 900
        for a in self.ARMS:
            leg_in = self.IN_LEG[a]
            for turn in ("straight", "left", "right"):
                pts = self._junction_pts(leg_in, turn)
                lane = _Lane((rid, -1, 0), pts, True, self.junction_id, lw)
                rid += 1
                self.lanes[f"jn_{a}_{turn}"] = lane
                out_leg = self.ROUTES[leg_in][turn]
                ex = self.lanes[f"exit_{out_leg}"]
                lane.successors = [ex]
                ex.predecessors.append(lane)
                app = self.lanes[f"app_{a}"]
                app.successors.append(lane)
                lane.predecessors = [app]
        bbox = BoundingBox(Location(self.x0, self.y0, self.z0),
                           Vector3D(lw * 1.6, lw * 1.6, 3.0))
        pairs = []
        for a in self.ARMS:
            for turn in ("straight", "left", "right"):
                lane = self.lanes[f"jn_{a}_{turn}"]
                pairs.append((Waypoint(lane, 0.0, self), Waypoint(lane, lane.length, self)))
        self.junction = Junction(self.junction_id, bbox, pairs)

    def _junction_pts(self, leg_in, turn):
        """The in-box portion of the script's own route geometry."""
        from ..script_bridge import dv
        p = dv.route_path(dv.MapCfg(self.lane_width, self.arm_length), leg_in,
                          turn, dv.Params())
        b = self.lane_width + 1e-6
        pts = [q for q in p.pts if abs(q[0]) <= b and abs(q[1]) <= b]
        return pts if len(pts) >= 2 else [p.pts[0], p.pts[-1]]

    # ---- queries ---- #
    def get_waypoint(self, location, project_to_road=True,
                     lane_type=LaneType.Driving) -> Optional[Waypoint]:
        x, y = self.world_to_script(location.x, location.y)
        best, best_d = None, float("inf")
        for lane in self.lanes.values():
            for k in range(0, int(lane.length / 0.5) + 1):
                s = min(k * 0.5, lane.length)
                px, py, _ = lane.at(s)
                d = math.hypot(px - x, py - y)
                if d < best_d:
                    best, best_d = (lane, s), d
        if best is None or (not project_to_road and best_d > 0.2):
            return None
        return Waypoint(best[0], best[1], self)

    def get_topology(self):
        out = []
        for lane in self.lanes.values():
            a = Waypoint(lane, 0.0, self)
            for succ in lane.successors:
                out.append((a, Waypoint(succ, 0.0, self)))
            if not lane.successors:
                out.append((a, Waypoint(lane, lane.length, self)))
        return out

    def generate_waypoints(self, distance):
        out = []
        for lane in self.lanes.values():
            s = 0.0
            while s <= lane.length:
                out.append(Waypoint(lane, s, self))
                s += distance
        return out

    def get_spawn_points(self):
        return [Waypoint(l, 0.0, self).transform for l in self.lanes.values()]


# --------------------------------------------------------------------------- #
# Actors
# --------------------------------------------------------------------------- #
class ActorBlueprint:
    def __init__(self, bp_id, extent, attrs=None):
        self.id = bp_id
        self.bounding_box = BoundingBox(Location(), Vector3D(*extent))
        self._attrs = dict(attrs or {})

    def has_attribute(self, name): return name in self._attrs

    def get_attribute(self, name): return self._attrs.get(name)

    def set_attribute(self, name, value): self._attrs[name] = value

    def __repr__(self): return f"ActorBlueprint({self.id})"


class BlueprintLibrary:
    def __init__(self, bps): self._bps = list(bps)

    def filter(self, pattern):
        import fnmatch
        pat = pattern if "*" in pattern else pattern + "*"
        return [b for b in self._bps if fnmatch.fnmatch(b.id, pat)]

    def find(self, bp_id):
        for b in self._bps:
            if b.id == bp_id:
                return b
        raise IndexError(f"blueprint {bp_id} not found")

    def __iter__(self): return iter(self._bps)


class Actor:
    def __init__(self, actor_id, type_id, transform, world, extent=(2.25, 1.0, 0.75)):
        self.id = actor_id
        self.type_id = type_id
        self._tf = transform
        self._velocity = Vector3D()
        self._world = world
        self.bounding_box = BoundingBox(Location(), Vector3D(*extent))
        self.attributes = {"role_name": ""}
        self._physics = True
        self.is_alive = True

    def get_transform(self): return self._tf

    def get_world(self): return self._world

    def get_location(self): return self._tf.location

    def set_transform(self, transform): self._tf = transform

    def get_velocity(self): return self._velocity

    def set_target_velocity(self, v): self._velocity = v

    def set_target_angular_velocity(self, v): pass

    def set_simulate_physics(self, enabled=True): self._physics = enabled

    def destroy(self):
        self.is_alive = False
        self._world._remove(self)
        return True

    def __repr__(self): return f"Actor(id={self.id}, type={self.type_id})"


class VehicleControl:
    """A recorded control. The double has no dynamics, so it is only ever
    stored — enough to exercise a control-emitting ego policy's plumbing
    offline, not to move a vehicle."""

    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False,
                 reverse=False, manual_gear_shift=False, gear=0):
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = bool(hand_brake)
        self.reverse = bool(reverse)
        self.manual_gear_shift = bool(manual_gear_shift)
        self.gear = int(gear)

    def __repr__(self):
        return (f"VehicleControl(throttle={self.throttle:.3f}, "
                f"steer={self.steer:.3f}, brake={self.brake:.3f})")


class WheelPhysicsControl:
    def __init__(self, max_steer_angle=70.0):
        self.max_steer_angle = float(max_steer_angle)


class VehiclePhysicsControl:
    def __init__(self, wheels=None):
        self.wheels = list(wheels or [WheelPhysicsControl()] * 4)


class Vehicle(Actor):
    """A vehicle in the double. Physics is not simulated, so a control is
    recorded rather than integrated; `apply_control` exists so the code paths
    that drive an ego through VehicleControl can be exercised without a server."""

    SPEED_LIMIT_KPH = 50.0

    def apply_control(self, control):
        self._control = control

    def get_control(self):
        return getattr(self, "_control", None) or VehicleControl()

    def get_speed_limit(self):
        return self.SPEED_LIMIT_KPH

    def get_physics_control(self):
        return VehiclePhysicsControl()


class Sensor(Actor):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._callback = None
        self.parent = None

    def listen(self, callback): self._callback = callback

    def stop(self): self._callback = None


class TrafficLight(Actor):
    def __init__(self, actor_id, transform, world, arm, stop_wp):
        super().__init__(actor_id, "traffic.traffic_light", transform, world)
        self._state = TrafficLightState.Green
        self._frozen = False
        self.arm = arm
        self._stop_wp = stop_wp

    def get_state(self): return self._state

    @property
    def state(self): return self._state

    def set_state(self, state): self._state = state

    def freeze(self, freeze): self._frozen = bool(freeze)

    def is_frozen(self): return self._frozen

    def get_stop_waypoints(self): return [self._stop_wp]

    def get_affected_lane_waypoints(self): return [self._stop_wp]

    def get_pole_index(self): return Map.ARMS.index(self.arm)


class CollisionEvent:
    def __init__(self, frame, actor, other_actor, impulse):
        self.frame = frame
        self.actor = actor
        self.other_actor = other_actor
        self.normal_impulse = impulse


class ActorList(list):
    def filter(self, pattern):
        import fnmatch
        pat = pattern if "*" in pattern else pattern + "*"
        return ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pat))


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
_VEHICLE_BPS = [
    ("vehicle.audi.a2", (1.85, 0.89, 0.77)),
    ("vehicle.tesla.model3", (2.40, 1.08, 0.74)),
    ("vehicle.nissan.patrol", (2.30, 0.94, 0.93)),
    ("vehicle.mini.cooper", (1.90, 0.90, 0.74)),
    ("vehicle.carlamotors.carlacola", (2.60, 1.30, 1.25)),
]


class World:
    def __init__(self, world_map: Map):
        self._map = world_map
        self._actors: Dict[int, Actor] = {}
        self._sensors: List[Sensor] = []
        self._next_id = 100
        self.frame = 0
        self.elapsed = 0.0
        self._settings = WorldSettings()
        self.id = 1
        self._tls: Dict[str, TrafficLight] = {}
        self._spawn_traffic_lights()

    # ---- setup ---- #
    def _spawn_traffic_lights(self):
        for arm in Map.ARMS:
            lane = self._map.lanes[f"app_{arm}"]
            stop_wp = Waypoint(lane, lane.length, self._map)
            tl = TrafficLight(self._alloc_id(), stop_wp.transform, self, arm,
                              stop_wp)
            self._actors[tl.id] = tl
            self._tls[arm] = tl

    def _alloc_id(self):
        self._next_id += 1
        return self._next_id

    # ---- API ---- #
    def get_map(self): return self._map

    def get_settings(self): return self._settings

    def apply_settings(self, settings):
        self._settings = settings
        return self.frame

    def get_blueprint_library(self):
        bps = [ActorBlueprint(i, e, {"number_of_wheels": "4", "role_name": "autopilot"})
               for i, e in _VEHICLE_BPS]
        bps.append(ActorBlueprint("sensor.other.collision", (0.1, 0.1, 0.1)))
        return BlueprintLibrary(bps)

    def get_actors(self, actor_ids=None):
        if actor_ids is None:
            return ActorList(self._actors.values())
        return ActorList(self._actors[i] for i in actor_ids if i in self._actors)

    def get_actor(self, actor_id): return self._actors.get(actor_id)

    def get_spectator(self):
        spec = getattr(self, "_spectator", None)
        if spec is None:
            spec = Actor(self._alloc_id(), "spectator", Transform(), self)
            self._spectator = spec
        return spec

    def try_spawn_actor(self, blueprint, transform, attach_to=None):
        try:
            return self.spawn_actor(blueprint, transform, attach_to=attach_to)
        except RuntimeError:
            return None

    def spawn_actor(self, blueprint, transform, attach_to=None):
        ext = blueprint.bounding_box.extent
        if blueprint.id.startswith("sensor."):
            s = Sensor(self._alloc_id(), blueprint.id, transform, self,
                       (ext.x, ext.y, ext.z))
            s.parent = attach_to
            self._actors[s.id] = s
            self._sensors.append(s)
            return s
        for other in self._actors.values():
            if not isinstance(other, Vehicle):
                continue
            if _overlap(_corners(transform, ext),
                        _corners(other.get_transform(), other.bounding_box.extent)):
                raise RuntimeError("spawn point occupied")
        v = Vehicle(self._alloc_id(), blueprint.id, transform, self,
                    (ext.x, ext.y, ext.z))
        v.attributes["role_name"] = str(blueprint.get_attribute("role_name") or "")
        self._actors[v.id] = v
        return v

    def _remove(self, actor):
        self._actors.pop(actor.id, None)
        if actor in self._sensors:
            self._sensors.remove(actor)

    def get_traffic_lights_from_waypoint(self, waypoint, distance):
        """Exact-lane matches first, then anything else within `distance`,
        nearest first — the ordering the real API's forward lane search gives."""
        exact, near = [], []
        for tl in self._tls.values():
            wp = tl._stop_wp
            d = wp.transform.location.distance(waypoint.transform.location)
            if wp.road_id == waypoint.road_id and wp.lane_id == waypoint.lane_id:
                exact.append(tl)
            elif d <= distance:
                near.append((d, tl))
        return exact + [tl for _, tl in sorted(near, key=lambda t: t[0])]

    def get_traffic_lights_in_junction(self, junction_id):
        return list(self._tls.values()) if junction_id == self._map.junction_id else []

    def freeze_all_traffic_lights(self, frozen):
        for tl in self._tls.values():
            tl.freeze(frozen)

    def tick(self, seconds=10.0):
        self.frame += 1
        self.elapsed += self._settings.fixed_delta_seconds or 0.05
        self._detect_collisions()
        return self.frame

    def wait_for_tick(self, seconds=10.0):
        return self.tick()

    # ---- collision reporting (the only "physics" the double has) ---- #
    def _detect_collisions(self):
        vehicles = [a for a in self._actors.values() if isinstance(a, Vehicle)]
        boxes = {v.id: _corners(v.get_transform(), v.bounding_box.extent)
                 for v in vehicles}
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                if not _overlap(boxes[a.id], boxes[b.id]):
                    continue
                imp = Vector3D(1000.0, 0.0, 0.0)
                for sensor in self._sensors:
                    if sensor.parent is a and sensor._callback:
                        sensor._callback(CollisionEvent(self.frame, a, b, imp))
                    elif sensor.parent is b and sensor._callback:
                        sensor._callback(CollisionEvent(self.frame, b, a, imp))


class Client:
    def __init__(self, host="localhost", port=2000, worker_threads=0):
        self.host, self.port = host, port
        self._timeout = 10.0
        self._world = World(Map())

    def set_timeout(self, seconds): self._timeout = seconds

    def get_client_version(self): return "0.9.16-fake"

    def get_server_version(self): return "0.9.16-fake"

    def get_world(self): return self._world

    def load_world(self, map_name, *a, **kw):
        self._world = World(Map(name=map_name))
        return self._world

    def reload_world(self, *a, **kw): return self._world

    def get_available_maps(self): return ["/Game/Carla/Maps/FakeTown"]

    def apply_batch_sync(self, commands, do_tick=False): return []


# --------------------------------------------------------------------------- #
# Oriented-box overlap (SAT), for the double's collision reporting
# --------------------------------------------------------------------------- #
def _corners(tf, extent):
    cx, cy = tf.location.x, tf.location.y
    r = math.radians(tf.rotation.yaw)
    c, s = math.cos(r), math.sin(r)
    ex, ey = extent.x, extent.y
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey))]


def _overlap(ra, rb) -> bool:
    for rect in (ra, rb):
        for i in range(len(rect)):
            ax, ay = rect[i]
            bx, by = rect[(i + 1) % len(rect)]
            nx, ny = -(by - ay), (bx - ax)
            pa = [nx * x + ny * y for x, y in ra]
            pb = [nx * x + ny * y for x, y in rb]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True
