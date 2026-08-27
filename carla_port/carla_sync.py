#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_sync.py — script state -> CARLA state, every step.

The invariant this module exists to hold:

    At each simulation step, every orchestrated actor is explicitly
    synchronized to the state prescribed by the current maneuver-script
    trajectory.

Sampling. `Actor.traj` / `Actor.speeds` are on the script's fixed grid
(se.DT = 1/60 s). The CARLA step may be anything, so states are linearly
interpolated in position and speed and angularly interpolated in heading. With
fixed_delta_seconds == se.DT the samples land exactly on grid frames and the
interpolation is a no-op.

Rebasing. `tau` is measured from the orchestrator's `t_base`, which
`Orchestrator._rebase_here()` moves forward whenever it rebases. Sampling
`t_sim - orch.t_base` therefore stays correct and continuous across every
rebase without this module knowing anything about how rebasing works.

CARLA -> script rebasing is NOT done here. The initial kinematic port keeps the
script authoritative; `carla_adapter.carla_actor_to_script_state` is the seam a
later port grows into.
"""
from __future__ import annotations

import math
from typing import Dict

from .carla_api import carla
from .carla_adapter import BindingSet, ScriptState, script_state_to_carla_transform
# Frame/script types are needed for ANNOTATIONS ONLY (PEP 563: this
# module has `from __future__ import annotations`, so they are never
# evaluated at runtime). Importing them under TYPE_CHECKING keeps this
# module independent of WHICH map frame and WHICH script layer are in
# use, so carla_highway/ can reuse it with a HighwayFrame.
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                # pragma: no cover
    from .carla_map import IntersectionFrame

# The script trajectory grid. Both script layers (v2/scenario_editor.py and
# highway/scenario_editor.py) define DT = 1/60 s; asserting it here rather
# than importing keeps the module script-layer agnostic.
DT = 1.0 / 60.0

# how the scripted state is pushed into CARLA
PHYSICS = "physics"      # physics on: set_transform + set_target_velocity.
                         # CARLA reports realized collisions and impulses.
KINEMATIC = "kinematic"  # physics off: set_transform only. Perfectly rigid
                         # playback, but UE generates far fewer hit events.


# --------------------------------------------------------------------------- #
# Sampling the script
# --------------------------------------------------------------------------- #
def sample_script_state(actor: "se.Actor", tau: float) -> ScriptState:
    """Actor state at script-local time `tau` (seconds since the actor's
    trajectory frame 0). Clamps at both ends: before the start it holds the
    initial pose, past the end it holds the final pose at zero speed."""
    if not actor.traj:
        x, y, h = actor.start
        return ScriptState(actor.id, x, y, h, 0.0)
    n = len(actor.traj)
    if tau <= 0.0:
        x, y, h = actor.traj[0]
        return ScriptState(actor.id, x, y, h, actor.speeds[0])
    f = tau / DT
    k = int(math.floor(f))
    if k >= n - 1:
        x, y, h = actor.traj[-1]
        return ScriptState(actor.id, x, y, h, actor.speeds[-1])
    u = f - k
    x0, y0, h0 = actor.traj[k]
    x1, y1, h1 = actor.traj[k + 1]
    dh = (h1 - h0 + 180.0) % 360.0 - 180.0        # shortest-arc heading lerp
    return ScriptState(actor.id,
                       x0 + u * (x1 - x0),
                       y0 + u * (y1 - y0),
                       (h0 + u * dh) % 360.0,
                       actor.speeds[k] + u * (actor.speeds[k + 1] - actor.speeds[k]))


def world_states(scenario: "se.Scenario", tau: float) -> Dict[str, ScriptState]:
    """Every actor's planned state at script-local time `tau`."""
    return {a.id: sample_script_state(a, tau) for a in scenario.actors}


# --------------------------------------------------------------------------- #
# Pushing it into CARLA
# --------------------------------------------------------------------------- #
class StateSynchronizer:
    """Applies script states to bound CARLA actors."""

    def __init__(self, frame: IntersectionFrame, bindings: BindingSet,
                 mode: str = PHYSICS, z_offset: float = 0.05,
                 sync_velocity: bool = True, reground_every: int = 0):
        if mode not in (PHYSICS, KINEMATIC):
            raise ValueError(f"unknown sync mode {mode!r}")
        self.frame = frame
        self.bindings = bindings
        self.mode = mode
        self.z_offset = z_offset
        self.sync_velocity = sync_velocity
        self.reground_every = reground_every       # 0 = never re-query ground z
        self._n = 0

    def apply(self, states: Dict[str, ScriptState]) -> int:
        """Synchronize every bound actor to its planned state. Returns the
        number of actors actually written."""
        written = 0
        for binding in self.bindings:
            st = states.get(binding.script_actor_id)
            if st is None:
                continue
            self.apply_one(binding, st)
            written += 1
        self._n += 1
        return written

    def apply_one(self, binding, st: ScriptState) -> None:
        z = binding.ground_z
        if self.reground_every and self._n % self.reground_every == 0:
            z = self._ground_z(st, z)
            binding.ground_z = z
        tf = script_state_to_carla_transform(st, self.frame, z + self.z_offset)
        actor = binding.carla_actor
        actor.set_transform(tf)
        if self.mode == PHYSICS and self.sync_velocity:
            # set_target_velocity requires physics; it makes CARLA's own
            # get_velocity() readback, collision impulses and any downstream
            # sensor agree with the scripted speed.
            actor.set_target_velocity(
                self.frame.to_carla_velocity(st.speed, st.heading))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    def _ground_z(self, st: ScriptState, fallback: float) -> float:
        loc = self.frame.to_carla_location(st.x, st.y, fallback)
        try:
            wp = self.frame.world.get_map().get_waypoint(loc, project_to_road=True)
        except RuntimeError:
            return fallback
        return wp.transform.location.z if wp is not None else fallback


# --------------------------------------------------------------------------- #
# Read-back (validation and diagnostics only)
# --------------------------------------------------------------------------- #
def readback_errors(frame: IntersectionFrame, bindings: BindingSet,
                    states: Dict[str, ScriptState]) -> Dict[str, tuple]:
    """{script id: (position error m, heading error deg, speed error m/s)} between
    the planned state and what CARLA reports. Diagnostic only — nothing in the
    loop consumes it."""
    from .carla_adapter import carla_actor_to_script_state
    out = {}
    for binding in bindings:
        st = states.get(binding.script_actor_id)
        if st is None:
            continue
        got = carla_actor_to_script_state(binding.script_actor_id,
                                          binding.carla_actor, frame)
        dh = abs((st.heading - got.heading + 180.0) % 360.0 - 180.0)
        out[binding.script_actor_id] = (math.hypot(got.x - st.x, got.y - st.y),
                                        dh, abs(got.speed - st.speed))
    return out
