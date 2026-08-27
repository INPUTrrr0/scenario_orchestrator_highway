#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_obs.py — the `state` observation an external ego policy sees.

The port's own ego policy (`carla_ego.EgoPolicy`, drivev2's IDM + pure-pursuit)
reads the script world directly, so it needs no observation at all. An
*external* ego policy does: `scenario_orchestration`'s `ego_policy_v1` interface
is "given an observation in the declared observation space, return an action in
the declared action space", and the observation space this port can provide is
`state` — an object-centric description of the scene in the ego's own frame.

This module builds that document. It owns no policy and no control law; it is
the sensor side of `ego_driver.PolicyEgoDriver`.

Frame
-----
Everything is in the **ego frame CARLA itself implies**: metres, `+x` forward,
`+y` right, yaw in radians relative to the ego heading. That is exactly what
`carla_garage.transfuser_utils.get_relative_transform` returns, which is what
the object-centric planners in this family are trained on, so the conversion is
done here with CARLA transform matrices rather than through the script frame.
Going via the script frame would work too — it is the same rigid motion — but it
mirrors `y` (README section 7), and a policy trained on CARLA's handedness must
not be handed a mirrored scene.

The one input that starts life in the script frame is the route: it comes from
the ego's own reference path (`EgoPolicy._ref_path`, the geometry the
orchestrator's ego prediction also assumes), so it is converted script -> CARLA
world -> ego frame, in that order.

What is deliberately NOT here
-----------------------------
The BEV raster. Every released PlanT 2.0 checkpoint needs one, but it is that
repository's own representation, produced by its own renderer from its own
prebuilt town rasters. Handing it in as an injected callable keeps this module
free of any one policy's internals; see `scenario_orchestration/bev.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .carla_adapter import BindingSet
# Frame/script types are needed for ANNOTATIONS ONLY (PEP 563: this
# module has `from __future__ import annotations`, so they are never
# evaluated at runtime). Importing them under TYPE_CHECKING keeps this
# module independent of WHICH map frame and WHICH script layer are in
# use, so carla_highway/ can reuse it with a HighwayFrame.
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                # pragma: no cover
    from .carla_map import IntersectionFrame

#: Objects further than this from the ego are not serialized at all. The policy
#: applies its own (tighter, model-specific) range gate; this is only here to
#: keep the document small on a busy map.
RANGE_M = 75.0

#: Route sampling, matching the reference agents in this family: one point per
#: metre, starting `ROUTE_FIRST_M` ahead, `ROUTE_POINTS` of them.
ROUTE_POINTS = 20
ROUTE_FIRST_M = 2.5
ROUTE_STEP_M = 1.0

#: A traffic light is only described while it is this close.
LIGHT_RANGE_M = 30.0

#: CARLA speed limits are per-map; 50 km/h is the urban default and the value
#: the object-centric planners fall back to.
DEFAULT_SPEED_LIMIT_KPH = 50.0

#: `carla.TrafficLightState` -> the state string the observation carries.
LIGHT_STATES = {"Red": "Red", "Yellow": "Yellow", "Green": "Green",
                "Off": "Green", "Unknown": "Green"}


def _normalize_angle(rad: float) -> float:
    """Wrap to (-pi, pi]."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class EgoFrame:
    """The ego's pose, as the rigid transform that maps world -> ego frame.

    Held as the three rows of the transposed rotation plus the translation so
    the conversion is a handful of multiplications per point and needs no numpy;
    the observation builder runs once per control step for every actor in range.
    """

    x: float
    y: float
    z: float
    yaw_rad: float

    @classmethod
    def of(cls, carla_actor) -> "EgoFrame":
        tf = carla_actor.get_transform()
        loc, rot = tf.location, tf.rotation
        return cls(x=float(loc.x), y=float(loc.y), z=float(loc.z),
                   yaw_rad=math.radians(float(rot.yaw)))

    def to_ego(self, X: float, Y: float, Z: Optional[float] = None
               ) -> Tuple[float, float, float]:
        """A CARLA world point in the ego frame (+x forward, +y right)."""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy = X - self.x, Y - self.y
        dz = 0.0 if Z is None else Z - self.z
        return (dx * c + dy * s, -dx * s + dy * c, dz)

    def relative_yaw(self, yaw_deg: float) -> float:
        """Another body's CARLA yaw, relative to the ego heading, in radians."""
        return _normalize_angle(math.radians(float(yaw_deg)) - self.yaw_rad)


@dataclass
class ObservationBuilder:
    """Builds one `state` observation per control step.

    `bev` is the policy-specific raster source: a callable taking the ego's
    CARLA actor and returning whatever that policy's `bev` field expects, or
    None. It is injected rather than imported so this module stays policy
    agnostic.
    """

    world: object
    frame: IntersectionFrame
    bindings: BindingSet
    ego_id: str
    ego_arm: str = "S"
    bev: Optional[Callable[[object], object]] = None
    range_m: float = RANGE_M
    route_points: int = ROUTE_POINTS
    route_first_m: float = ROUTE_FIRST_M
    route_step_m: float = ROUTE_STEP_M
    speed_limit_kph: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def build(self, ego_actor, ref_path: Sequence[Tuple[float, float]],
              ego_script_pose: Tuple[float, float, float],
              ego_speed: float) -> Dict[str, object]:
        """The observation for the current CARLA state.

        `ref_path` and `ego_script_pose` are in the SCRIPT frame — they come
        from the ego's own reference path — and are converted here. Everything
        else is read from CARLA.
        """
        ego = EgoFrame.of(ego_actor)
        observation: Dict[str, object] = {
            "ego": {"speed_mps": float(ego_speed)},
            "objects": self._objects(ego, ego_actor),
            "route": self._route(ego, ref_path, ego_script_pose),
            "speed_limit_kph": self._speed_limit(ego_actor),
        }
        if self.bev is not None:
            raster = self.bev(ego_actor)
            if raster is not None:
                observation["bev"] = {"semantic_classes": raster}
        return observation

    # ------------------------------------------------------------------ #
    # Objects
    # ------------------------------------------------------------------ #
    def _objects(self, ego: EgoFrame, ego_actor) -> List[Dict[str, object]]:
        """Every vehicle, walker and blocking traffic light the ego can see.

        Read from CARLA rather than from the maneuver script on purpose: an ego
        policy is being evaluated on what a driver could perceive, and CARLA is
        also the only place non-orchestrated actors exist. The two agree for
        orchestrated actors — the synchronizer wrote them this tick.
        """
        out: List[Dict[str, object]] = []
        ego_carla_id = getattr(ego_actor, "id", None)
        for actor in self._actors():
            # Excluded twice on purpose: by CARLA identity, and by script id in
            # case the caller handed a different handle to the same vehicle.
            if getattr(actor, "id", None) == ego_carla_id:
                continue
            if self.bindings.script_id_of(actor) == self.ego_id:
                continue
            kind = self._classify(actor)
            if kind is None:
                continue
            tf = actor.get_transform()
            x, y, z = ego.to_ego(tf.location.x, tf.location.y, tf.location.z)
            if x * x + y * y > self.range_m ** 2:
                continue
            extent = self._extent(actor)
            obj: Dict[str, object] = {
                "type": kind,
                "position": [x, y, z],
                "yaw_rad": ego.relative_yaw(tf.rotation.yaw),
                "speed_mps": self._speed(actor),
                "extent": list(extent),
                "type_id": getattr(actor, "type_id", None),
            }
            # The script id, when the actor is one of ours. Not part of the
            # observation contract, but it is what makes a recorded trace
            # readable next to the orchestrator's own decisions.
            script_id = self.bindings.script_id_of(actor)
            obj["id"] = script_id if script_id is not None else \
                f"carla:{getattr(actor, 'id', '?')}"
            out.append(obj)
        out.extend(self._lights(ego))
        return out

    def _actors(self) -> List[object]:
        try:
            actors = self.world.get_actors()
        except (RuntimeError, AttributeError):        # pragma: no cover
            return []
        try:
            return list(actors.filter("*vehicle*")) + list(actors.filter("*walker*"))
        except (AttributeError, TypeError):
            # The offline double exposes a plain list with no filter().
            return [a for a in actors
                    if self._classify(a) is not None]

    @staticmethod
    def _classify(actor) -> Optional[str]:
        """The observation's object class for a CARLA actor, or None to skip."""
        type_id = str(getattr(actor, "type_id", "") or "")
        if type_id.startswith("vehicle"):
            return "car"
        if type_id.startswith("walker"):
            return "walker"
        return None

    @staticmethod
    def _extent(actor) -> Tuple[float, float, float]:
        """CARLA half-extents (length, width, height), which is what
        `carla.BoundingBox.extent` reports and what the policies expect."""
        box = getattr(actor, "bounding_box", None)
        extent = getattr(box, "extent", None)
        if extent is None:                            # pragma: no cover
            return (2.4, 1.0, 0.8)
        return (float(extent.x), float(extent.y), float(extent.z))

    @staticmethod
    def _speed(actor) -> float:
        try:
            v = actor.get_velocity()
        except (RuntimeError, AttributeError):        # pragma: no cover
            return 0.0
        return math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2)

    # ------------------------------------------------------------------ #
    # Traffic lights
    # ------------------------------------------------------------------ #
    def _lights(self, ego: EgoFrame) -> List[Dict[str, object]]:
        """The ego approach's own light, described at its stop line.

        Only the light governing the ego is described, and only while it is red
        or amber: that is the reference agents' own filtering, and a green light
        carries no constraint. The position is the stop line rather than the
        light head, because that is where a planner has to stop.
        """
        light = None
        try:
            light = self.frame.traffic_light(self.ego_arm)
        except (KeyError, AttributeError, RuntimeError):   # pragma: no cover
            return []
        if light is None:
            return []
        state = LIGHT_STATES.get(str(getattr(light, "state", "Green")), "Green")
        if state == "Green":
            return []
        out: List[Dict[str, object]] = []
        for wp in self._stop_lines(light):
            tf = wp.transform
            x, y, z = ego.to_ego(tf.location.x, tf.location.y, tf.location.z)
            if x * x + y * y > LIGHT_RANGE_M ** 2:
                continue
            out.append({
                "type": "traffic_light",
                "position": [x, y, z],
                "yaw_rad": ego.relative_yaw(tf.rotation.yaw),
                "state": state,
                "extent": [1.5, 1.5, 0.5],
            })
        return out

    @staticmethod
    def _stop_lines(light) -> List[object]:
        try:
            return list(light.get_stop_waypoints())
        except (AttributeError, RuntimeError):        # pragma: no cover
            return []

    # ------------------------------------------------------------------ #
    # Route
    # ------------------------------------------------------------------ #
    def _route(self, ego: EgoFrame, ref_path: Sequence[Tuple[float, float]],
               ego_script_pose: Tuple[float, float, float]
               ) -> List[List[float]]:
        """`route_points` ego-frame points along the ego's intended route.

        The route is the ego's own reference path — the geometry
        `EgoPolicy._build_reference_path` builds from `mv.build_route_maneuvers`
        and the orchestrator's ego prediction assumes — resampled to one point
        per `route_step_m` from `route_first_m` ahead of the ego's current
        position along the path. Route conditioning is what tells a planner
        which way through the junction it is meant to go, so it has to be the
        route the scenario is about.

        Points are never invented: a route that runs out is reported short and
        the policy pads it, so a shortfall is visible rather than fabricated.
        """
        if not ref_path:
            return []
        ex, ey, _ = ego_script_pose
        start = self._nearest_index(ref_path, ex, ey)

        # Arc length from the ego's projection onto the path.
        picked: List[List[float]] = []
        target = self.route_first_m
        travelled = 0.0
        i = start
        while i + 1 < len(ref_path) and len(picked) < self.route_points:
            (ax, ay), (bx, by) = ref_path[i], ref_path[i + 1]
            seg = math.hypot(bx - ax, by - ay)
            while target <= travelled + seg + 1e-9 and len(picked) < self.route_points:
                f = 0.0 if seg <= 1e-9 else (target - travelled) / seg
                px, py = ax + (bx - ax) * f, ay + (by - ay) * f
                X, Y = self.frame.to_carla_xy(px, py)
                x, y, _z = ego.to_ego(X, Y, None)
                picked.append([x, y])
                target += self.route_step_m
            travelled += seg
            i += 1
        return picked

    @staticmethod
    def _nearest_index(path: Sequence[Tuple[float, float]],
                       x: float, y: float) -> int:
        best_i, best_d2 = 0, float("inf")
        for i, (px, py) in enumerate(path):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i

    # ------------------------------------------------------------------ #
    def _speed_limit(self, ego_actor) -> float:
        if self.speed_limit_kph is not None:
            return float(self.speed_limit_kph)
        try:
            limit = float(ego_actor.get_speed_limit())
        except (RuntimeError, AttributeError, TypeError):
            return DEFAULT_SPEED_LIMIT_KPH
        # CARLA reports 0 until the vehicle has passed a speed-limit sign.
        return limit if limit > 1.0 else DEFAULT_SPEED_LIMIT_KPH
