#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_collision.py — REALIZED collisions, from CARLA sensors.

Two collision mechanisms exist in this port and both are kept:

  predicted : planned trajectories -> oriented body overlap. Owned by
              v4/directives_script.py (collide_time / collision_now) and used
              by evaluate() and repair(). Untouched by this module.
  realized  : CARLA collision sensors. Execution-level ground truth, observed
              here.

The predicted path is NOT replaced by CARLA collision sensors: the orchestrator
must reason about collisions that have not happened yet, which no sensor can
report.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .carla_api import carla
from .carla_adapter import BindingSet

COLLISION_BP = "sensor.other.collision"


@dataclass(frozen=True)
class RealizedCollision:
    """One CARLA collision event, expressed in script-actor terms."""
    frame: int
    sim_time: float
    actor_id: str                     # script actor id of the sensor's parent
    other_id: Optional[str]           # script actor id, or None if not bound
    other_type: str                   # CARLA type_id of the other body
    impulse: float                    # |normal impulse| (N*s)

    def __str__(self) -> str:
        other = self.other_id if self.other_id is not None else self.other_type
        return (f"realized collision: {self.actor_id} x {other} "
                f"@ {self.sim_time:.2f}s (impulse {self.impulse:.0f} N*s)")


class CollisionMonitor:
    """Attaches a collision sensor to every bound actor and collects events.

    Sensor callbacks run on the client's I/O thread, so events land in a deque
    and are drained from the main loop after world.tick().
    """

    def __init__(self, world, bindings: BindingSet,
                 min_impulse: float = 0.0, dedupe_pairs: bool = True):
        self.world = world
        self.bindings = bindings
        self.min_impulse = min_impulse
        self.dedupe_pairs = dedupe_pairs
        self.sensors: List = []
        self.history: List[RealizedCollision] = []
        self._queue: Deque = deque()
        self._sim_time = 0.0
        self._seen_pairs: set = set()

    # ---- setup ---- #
    def attach(self) -> int:
        bp = self.world.get_blueprint_library().find(COLLISION_BP)
        for binding in self.bindings:
            sensor = self.world.spawn_actor(bp, carla.Transform(),
                                            attach_to=binding.carla_actor)
            sid = binding.script_actor_id
            sensor.listen(lambda event, sid=sid: self._queue.append((sid, event)))
            self.sensors.append(sensor)
        return len(self.sensors)

    def set_time(self, sim_time: float) -> None:
        """Timestamp applied to events drained next. Called by the runner with
        the simulation time of the tick that produced them."""
        self._sim_time = sim_time

    # ---- per-tick ---- #
    def drain(self) -> List[RealizedCollision]:
        """Pop everything the sensors reported since the last call."""
        out: List[RealizedCollision] = []
        while self._queue:
            sid, event = self._queue.popleft()
            other = event.other_actor
            other_id = self.bindings.script_id_of(other)
            imp = event.normal_impulse
            mag = (imp.x ** 2 + imp.y ** 2 + imp.z ** 2) ** 0.5
            if mag < self.min_impulse:
                continue
            if self.dedupe_pairs:
                key = tuple(sorted((sid, other_id or f"carla:{other.id}")))
                if key in self._seen_pairs:
                    continue
                self._seen_pairs.add(key)
            rc = RealizedCollision(frame=getattr(event, "frame", 0),
                                   sim_time=self._sim_time,
                                   actor_id=sid, other_id=other_id,
                                   other_type=getattr(other, "type_id", "?"),
                                   impulse=mag)
            out.append(rc)
            self.history.append(rc)
        return out

    def involving(self, script_id: str) -> List[RealizedCollision]:
        return [c for c in self.history
                if c.actor_id == script_id or c.other_id == script_id]

    # ---- teardown ---- #
    def destroy(self) -> None:
        for s in self.sensors:
            try:
                s.stop()
            except RuntimeError:
                pass
            try:
                s.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()
