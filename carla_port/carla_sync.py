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

# where a background actor's YAW comes from
HEADING_PLAN = "plan"      # the script's own heading field, written verbatim
HEADING_MOTION = "motion"  # the direction the actor is actually travelling
HEADING_MODES = (HEADING_PLAN, HEADING_MOTION)

#: Below this much movement in one step the direction of travel is numerical
#: noise, so the last good heading is held rather than recomputed.
HEADING_MIN_STEP = 0.02      # m
#: Low-pass on the raw direction of travel. The commanded positions are
#: themselves slightly uneven across a replan, and differentiating them
#: amplifies that; this is short enough to track a real lane change (~2 s) and
#: long enough to reject a one-step wobble.
HEADING_TAU = 0.12           # s
#: Hard cap on how fast the rendered yaw may turn. A road vehicle at speed does
#: not exceed this, and it bounds anything the filter has not already removed.
HEADING_RATE_MAX = 90.0      # deg/s
#: How far the rendered yaw may sit from the plan's own heading.
#:
#: Direction of travel is a vehicle's yaw only while the plan is something a
#: vehicle could drive. A closed-loop cut-in replanned every 0.10 s is not: each
#: replan demands the whole remaining lateral correction inside the remaining
#: time, so the merging actor's commanded speed collapses (12 -> 2.9 m/s across
#: one merge) while its lateral rate stays near 4.6 m/s. Its true direction of
#: travel is then 50 deg off the road, and rendering that honestly gives a car
#: crabbing sideways down the highway.
#:
#: So the direction of travel supplies the LEAN and the plan's heading supplies
#: the reference it leans from. 20 deg is a hard swerve and comfortably past
#: the 14 deg the repository's own ego policy allows itself (`LC_YAW_MAX` in
#: drivev2), so a real manoeuvre is never clipped and a physically impossible
#: one is.
HEADING_MAX_SLIP = 20.0      # deg


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
                 sync_velocity: bool = True, reground_every: int = 0,
                 heading: str = HEADING_MOTION):
        if mode not in (PHYSICS, KINEMATIC):
            raise ValueError(f"unknown sync mode {mode!r}")
        if heading not in HEADING_MODES:
            raise ValueError(f"unknown heading mode {heading!r}; expected one "
                             f"of {', '.join(HEADING_MODES)}")
        self.frame = frame
        self.bindings = bindings
        self.mode = mode
        self.z_offset = z_offset
        self.sync_velocity = sync_velocity
        self.reground_every = reground_every       # 0 = never re-query ground z
        self.heading = heading
        self._n = 0
        self._last_xy: Dict[str, tuple] = {}
        self._yaw: Dict[str, float] = {}

    def apply(self, states: Dict[str, ScriptState], dt: float = DT) -> int:
        """Synchronize every bound actor to its planned state. Returns the
        number of actors actually written."""
        written = 0
        for binding in self.bindings:
            st = states.get(binding.script_actor_id)
            if st is None:
                continue
            self.apply_one(binding, st, dt)
            written += 1
        self._n += 1
        return written

    def heading_of(self, st: ScriptState, dt: float) -> float:
        """The yaw to render this actor with.

        `HEADING_PLAN` writes the script's heading field verbatim. That field
        is not a vehicle attitude — it is an artifact of how the current plan
        was framed, and a closed-loop orchestrator reframes it constantly. On
        `scenario_cutin` the cut-in actor is replanned every 0.10 s from its
        NOMINAL LANE heading (`CutinOrchestrator.headings`, recorded at first
        sight), deliberately, so the manoeuvre's temporary yaw cannot compound
        into the next plan frame. Within each tick the fresh lane-change
        maneuver then swings the heading back out. The result is a sawtooth:
        the merging actor's yaw oscillated +/-5 deg at 10 Hz through the whole
        merge.

        In pygame that is invisible — a scripted actor's heading is decorative,
        drawn as a small rectangle and never differentiated. Here it is fed to
        `set_transform` sixty times a second on a 3D vehicle body, and the car
        visibly shakes.

        `HEADING_MOTION` renders the direction the actor is actually
        travelling, low-passed and rate-capped. On a smooth path that IS a
        vehicle's yaw, it is continuous by construction, and it cannot feed
        back into any decision: the orchestrator reasons about `Actor.traj`,
        never about what CARLA was told.
        """
        aid = st.actor_id
        prev = self._last_xy.get(aid)
        self._last_xy[aid] = (st.x, st.y)
        if self.heading == HEADING_PLAN or prev is None:
            self._yaw[aid] = st.heading
            return st.heading
        cur = self._yaw.get(aid, st.heading)
        dx, dy = st.x - prev[0], st.y - prev[1]
        if math.hypot(dx, dy) < HEADING_MIN_STEP:
            return cur                      # stopped: hold, do not spin
        raw = math.degrees(math.atan2(dy, dx))
        # keep the lean inside what a vehicle can actually hold — see
        # HEADING_MAX_SLIP
        slip = (raw - st.heading + 180.0) % 360.0 - 180.0
        raw = st.heading + max(-HEADING_MAX_SLIP, min(HEADING_MAX_SLIP, slip))
        err = (raw - cur + 180.0) % 360.0 - 180.0
        alpha = (1.0 - math.exp(-dt / HEADING_TAU)) if dt > 0.0 else 1.0
        cap = HEADING_RATE_MAX * max(dt, 1e-6)
        step = max(-cap, min(cap, err * alpha))
        cur = (cur + step) % 360.0
        self._yaw[aid] = cur
        return cur

    def apply_one(self, binding, st: ScriptState, dt: float = DT) -> None:
        z = binding.ground_z
        if self.reground_every and self._n % self.reground_every == 0:
            z = self._ground_z(st, z)
            binding.ground_z = z
        heading = self.heading_of(st, dt)
        st = ScriptState(st.actor_id, st.x, st.y, heading, st.speed)
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
