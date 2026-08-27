#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_adapter.py — script actor <-> CARLA actor bindings.

Ownership split (design document, "Actor Representation"):

    script actor owns : identity, remaining maneuver sequence, planned
                        trajectory (Actor.traj), planned speeds (Actor.speeds),
                        vehicle dimensions, semantic role
    CARLA actor owns  : current transform, velocity, bounding box / collision
                        participation, rendering and sensors

The binding is an explicit `script actor id -> carla.Actor` map, never list
position, so rebasing (which rebuilds every se.Actor object) cannot scramble it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# Frame/script types are needed for ANNOTATIONS ONLY (PEP 563: this
# module has `from __future__ import annotations`, so they are never
# evaluated at runtime). Importing them under TYPE_CHECKING keeps this
# module independent of WHICH map frame and WHICH script layer are in
# use, so carla_highway/ can reuse it with a HighwayFrame.
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                # pragma: no cover
    from .carla_map import IntersectionFrame
    from .script_bridge import se


# --------------------------------------------------------------------------- #
# The state exchanged across the boundary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScriptState:
    """One actor's planned state at one instant, in script world coordinates
    (metres, degrees CCW from +x, y up)."""
    actor_id: str
    x: float
    y: float
    heading: float
    speed: float

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.heading)


# --------------------------------------------------------------------------- #
# Named conversions (the design document asks for these by name)
# --------------------------------------------------------------------------- #
def script_state_to_carla_transform(state: ScriptState, frame: IntersectionFrame,
                                    z: Optional[float] = None):
    return frame.to_carla_transform(state.x, state.y, state.heading, z)


def carla_actor_to_script_state(actor_id: str, carla_actor,
                                frame: IntersectionFrame) -> ScriptState:
    """CARLA -> script. Present and unit-tested, but deliberately NOT wired into
    the run loop: the initial kinematic port keeps the script authoritative and
    does not rebase from CARLA state (design document, "Rebasing"). This is the
    seam a later physics/controller port grows into."""
    tf = carla_actor.get_transform()
    x, y, h = frame.from_carla_transform(tf)
    return ScriptState(actor_id, x, y, h,
                       frame.from_carla_velocity(carla_actor.get_velocity()))


def script_speed_to_carla_velocity(speed: float, heading: float,
                                   frame: IntersectionFrame):
    return frame.to_carla_velocity(speed, heading)


# --------------------------------------------------------------------------- #
# Bindings
# --------------------------------------------------------------------------- #
@dataclass
class CarlaActorBinding:
    script_actor_id: str
    carla_actor: object
    blueprint_id: str = ""
    extent: Tuple[float, float] = (4.5, 2.0)      # (length, width) in metres
    ground_z: float = 0.0

    @property
    def carla_id(self) -> int:
        return self.carla_actor.id


class BindingSet:
    """script actor id <-> CARLA actor, both directions."""

    def __init__(self, frame: IntersectionFrame):
        self.frame = frame
        self._by_script: Dict[str, CarlaActorBinding] = {}
        self._by_carla: Dict[int, CarlaActorBinding] = {}
        self.unbound: List[str] = []              # script actors that failed to spawn

    def add(self, binding: CarlaActorBinding) -> None:
        self._by_script[binding.script_actor_id] = binding
        self._by_carla[binding.carla_id] = binding

    def get(self, script_id: str) -> Optional[CarlaActorBinding]:
        return self._by_script.get(script_id)

    def script_id_of(self, carla_actor_or_id) -> Optional[str]:
        cid = (carla_actor_or_id if isinstance(carla_actor_or_id, int)
               else getattr(carla_actor_or_id, "id", None))
        b = self._by_carla.get(cid)
        return b.script_actor_id if b else None

    def __iter__(self) -> Iterator[CarlaActorBinding]:
        return iter(self._by_script.values())

    def __len__(self) -> int:
        return len(self._by_script)

    def ids(self) -> List[str]:
        return list(self._by_script)

    def destroy(self) -> None:
        for b in list(self._by_script.values()):
            try:
                b.carla_actor.destroy()
            except RuntimeError:
                pass
        self._by_script.clear()
        self._by_carla.clear()


# --------------------------------------------------------------------------- #
# Spawning
# --------------------------------------------------------------------------- #
def _four_wheeled(blueprint_library) -> List:
    bps = []
    for bp in blueprint_library.filter("vehicle.*"):
        try:
            if int(bp.get_attribute("number_of_wheels")) != 4:
                continue
        except (RuntimeError, ValueError, AttributeError):
            pass
        bps.append(bp)
    return sorted(bps, key=lambda b: b.id)


def _bp_extent(bp) -> Optional[Tuple[float, float]]:
    """(length, width) from the blueprint's declared bounding box.

    CARLA 0.9.16 does NOT expose a bounding box on carla.ActorBlueprint, so on a
    real server this returns None and blueprint choice falls back to round-robin.
    `adopt_carla_extents` then copies the SPAWNED vehicle's real box into the
    script actor, which is what actually keeps predicted and realized collision
    geometry in agreement."""
    bb = getattr(bp, "bounding_box", None)
    if bb is None:
        return None
    return (2.0 * bb.extent.x, 2.0 * bb.extent.y)


def spawn_bindings(world, frame: IntersectionFrame, scenario: "se.Scenario",
                   z_offset: float = 0.10, simulate_physics: bool = True,
                   blueprint_filter: str = "vehicle.*",
                   adopt_carla_extents: bool = True,
                   snap_to_ground: bool = True,
                   models: Optional[Dict[str, str]] = None,
                   colors: Optional[Dict[str, str]] = None,
                   notes: Optional[List[str]] = None) -> BindingSet:
    """Spawn one CARLA vehicle per script actor at that actor's initial pose.

    Script actors that cannot be spawned (occupied spot, off-road pose) are
    reported in `BindingSet.unbound`. They stay in the scenario and keep being
    reasoned about by the orchestrator; they simply have no CARLA body.

    `adopt_carla_extents` copies each spawned vehicle's real bounding box back
    into the script actor's length/width, so PREDICTED body overlap (the
    orchestrator's oriented-rectangle sweep) and REALIZED CARLA collisions use
    the same geometry. It mutates only length/width, never the maneuver plan.

    `models` maps script actor id -> a blueprint id or filter (`"*"` is the
    catch-all key, applied to any actor without its own entry); `colors` does
    the same for an `"R,G,B"` paint. Both are *preferences*: an id that matches
    nothing in this build falls back to the body-size match below and appends a
    line to `notes` rather than failing the run.

    Left to itself, the size match hands every actor with the same declared
    body the same blueprint — on Town04 that was a 5.2 m box truck for the
    whole fleet, which is both hard to read in a video and 0.7 m longer than
    the 4.5 m body the scenario's geometry was authored against.
    """
    library = world.get_blueprint_library()
    if blueprint_filter != "vehicle.*":
        pool = sorted(library.filter(blueprint_filter), key=lambda b: b.id)
    else:
        pool = _four_wheeled(library)
    if not pool:
        raise RuntimeError(f"no blueprints matched {blueprint_filter!r}")

    models = dict(models or {})
    colors = dict(colors or {})
    notes = notes if notes is not None else []

    bindings = BindingSet(frame)
    for i, actor in enumerate(scenario.actors):
        want = (actor.length, actor.width)
        aid = str(actor.id)
        wanted = models.get(aid, models.get("*"))
        bp = None
        if wanted:
            bp = _named_blueprint(library, wanted)
            if bp is None:
                notes.append(f"no blueprint matched {wanted!r} for actor {aid}; "
                             "fell back to the closest body-size match")
        if bp is None:
            bp = _closest_blueprint(pool, want, i)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"script_{actor.id}")
        paint = colors.get(aid, colors.get("*"))
        if paint and bp.has_attribute("color"):
            try:
                bp.set_attribute("color", paint)
            except (RuntimeError, ValueError):
                notes.append(f"blueprint {bp.id} refused colour {paint!r} "
                             f"for actor {aid}")
        x, y, h = actor.start
        z = frame.anchor.z
        if snap_to_ground:
            z = _ground_z(world, frame, x, y, fallback=z)
        tf = frame.to_carla_transform(x, y, h, z + z_offset)
        carla_actor = world.try_spawn_actor(bp, tf)
        if carla_actor is None:                   # nudge up and retry once
            tf = frame.to_carla_transform(x, y, h, z + z_offset + 0.5)
            carla_actor = world.try_spawn_actor(bp, tf)
        if carla_actor is None:
            bindings.unbound.append(actor.id)
            continue
        carla_actor.set_simulate_physics(simulate_physics)
        bb = carla_actor.bounding_box
        extent = (2.0 * bb.extent.x, 2.0 * bb.extent.y)
        if adopt_carla_extents:
            actor.length, actor.width = extent
        bindings.add(CarlaActorBinding(script_actor_id=actor.id,
                                       carla_actor=carla_actor,
                                       blueprint_id=bp.id, extent=extent,
                                       ground_z=z))
    if adopt_carla_extents:
        scenario.simulate()                       # dimensions changed; refresh
    return bindings


def _named_blueprint(library, wanted: str):
    """A blueprint for an explicit id or filter, or None if nothing matches.

    Accepts a full id (`vehicle.audi.tt`), a bare model (`audi.tt`) or a
    wildcard (`vehicle.audi.*`); ties break on id so the choice is stable
    across runs.
    """
    for pattern in (wanted, f"vehicle.{wanted}"):
        try:
            found = sorted(library.filter(pattern), key=lambda b: b.id)
        except RuntimeError:
            found = []
        if found:
            return found[0]
    return None


def _closest_blueprint(pool: List, want: Tuple[float, float], index: int):
    """Pick the blueprint whose body is closest to the script actor's declared
    dimensions; deterministic, and falls back to round-robin when CARLA does not
    expose blueprint bounding boxes."""
    scored = []
    for bp in pool:
        ext = _bp_extent(bp)
        if ext is None:
            continue
        scored.append((abs(ext[0] - want[0]) + abs(ext[1] - want[1]), bp.id, bp))
    if scored:
        scored.sort(key=lambda s: (s[0], s[1]))
        return scored[0][2]
    return pool[index % len(pool)]


def _ground_z(world, frame: IntersectionFrame, x: float, y: float,
              fallback: float) -> float:
    """Road surface height under a script-frame point."""
    loc = frame.to_carla_location(x, y, fallback)
    try:
        wp = world.get_map().get_waypoint(loc, project_to_road=True)
    except RuntimeError:
        return fallback
    return wp.transform.location.z if wp is not None else fallback


def pose_round_trip_error(frame: IntersectionFrame, pose, transform
                          ) -> Tuple[float, float]:
    """(position error m, heading error deg) between a script pose and a CARLA
    transform read back from the simulator. Used by the validation suite."""
    x, y, h = pose
    rx, ry, rh = frame.from_carla_transform(transform)
    dh = abs((h - rh + 180.0) % 360.0 - 180.0)
    return (math.hypot(rx - x, ry - y), dh)
