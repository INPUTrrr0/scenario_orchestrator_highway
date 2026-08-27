#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/ego_driver.py — drive the ego from an external ego policy.

The port ships one ego policy of its own: `carla_ego.EgoPolicy`, which is
`v4/drivev2.py`'s IDM + pure-pursuit, reading the script world directly. This
module adds the other case — an ego policy that lives in **another repository**
and speaks `scenario_orchestration`'s `ego_policy_v1` interface:

    observation (state[+sensor])  ->  policy.act(...)  ->  action (control)

The observation is `state` — the object-centric document `carla_obs.py` builds
— plus, for a policy that asks for one, a `sensor` block carrying the camera
images its own rig produced. A policy asks by exposing `sensors()`; see
`carla_sensors.py` for why the rig is the policy's to choose and not this
module's.

`PolicyEgoDriver` is the whole of that wiring. It owns no policy and no control
law: it builds the observation (`carla_obs.ObservationBuilder`), asks the policy
object for an action, and applies the action's `control` block as a
`carla.VehicleControl`.

Why `control` and not `waypoints`
---------------------------------
A waypoint-emitting policy is only reproducible if the controller that turns its
waypoints into pedals is the one it was tuned and published with. The policies
in this family therefore return their own `control` alongside the waypoints,
computed by their own lateral and longitudinal controllers, and this driver
consumes that. Re-deriving a controller here would silently change the numbers
while still calling them the policy's. Waypoints, target speed and everything
else the action carries are recorded in the trace but do not steer the car.

Decision rate
-------------
CARLA runs at the script's own 1/60 s step so that sampled maneuver states land
exactly on trajectory frames (README section 3). An external policy was neither
trained nor evaluated at 60 Hz — the CARLA Leaderboard runs agents at 20 Hz — so
the policy is stepped at `hz` and its last control is held in between, rather
than being queried once per physics tick. Holding a control between decisions is
what the leaderboard does too; querying a transformer sixty times a simulated
second would be both slower and off-distribution.

What stays with drivev2's policy
--------------------------------
`EgoPolicy` is still constructed when an external driver is in charge, but only
as the *measurement and prediction* companion: it owns the reference path, so it
supplies the route the external policy is conditioned on, the cross-track error
the run report quotes, and the route progress. It is never asked for a command.
That keeps "how well did the ego track the route the scenario is about" a
comparable number across every policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .carla_obs import ObservationBuilder
from .carla_sensors import CameraRig, rig as named_rig, specs_from
# Annotation only (PEP 563) — keeps this driver usable with any map frame,
# so carla_highway can host external ego_policy_v1 policies too.
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                # pragma: no cover
    from .carla_map import IntersectionFrame
from .carla_adapter import BindingSet

#: How often an external policy is asked for a decision, in Hz. 20 Hz is the
#: CARLA Leaderboard's agent rate, which is what these policies are evaluated at.
DEFAULT_POLICY_HZ = 20.0


class EgoDriverError(RuntimeError):
    """An external ego policy could not be stepped, or gave nothing usable."""


@dataclass
class EgoContext:
    """Everything a driver needs about the run it has been attached to."""

    world: object
    frame: IntersectionFrame
    bindings: BindingSet
    ego_id: str
    ego_actor: object
    mode: str
    policy: object                 # carla_ego.EgoPolicy, the measurement companion
    fixed_delta: float


class EgoDriver:
    """Produces the ego's `carla.VehicleControl` for one simulation step.

    The runner owns when this is called (step E2 of the loop) and applies what
    it returns; a driver never touches the world itself.
    """

    name = "ego_driver"

    def attach(self, ctx: EgoContext) -> None:
        """Bind to a run. Called once, after the ego vehicle exists."""

    def reset(self) -> None:
        """Start a new episode."""

    def control(self, dt: float):
        """The VehicleControl for this step."""
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the driver holds."""

    def metadata(self) -> Dict[str, Any]:
        """What this driver is, for the run report."""
        return {"driver": self.name}


class PolicyEgoDriver(EgoDriver):
    """Drive the ego from an `ego_policy_v1` policy object.

    `policy` is whatever the policy repository's `build_policy()` returned. Only
    two things are required of it: an `act(observation) -> action` method, and an
    action carrying a `control` block. `reset()`, `close()` and `metadata()` are
    used when present and ignored when absent, so a minimal policy still works.
    """

    def __init__(self, policy: Any, name: str = "external",
                 hz: float = DEFAULT_POLICY_HZ,
                 bev: Optional[Any] = None,
                 ego_arm: str = "S",
                 keep_actions: int = 0):
        self.policy = policy
        self.name = str(name)
        self.hz = float(hz)
        self.bev = bev
        self.ego_arm = str(ego_arm)
        #: How many actions to retain verbatim for the trace. 0 keeps none; the
        #: summary counters below are kept regardless. A full action carries the
        #: predicted waypoints and path, so keeping every one of them for a
        #: twenty-second run is megabytes.
        self.keep_actions = int(keep_actions)

        self.ctx: Optional[EgoContext] = None
        self.observations: Optional[ObservationBuilder] = None
        self.rig: Optional[CameraRig] = None
        self.steps = 0                     # policy decisions taken
        self.calls = 0                     # control() invocations
        self.actions: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self._control = None               # the held VehicleControl
        self._next_decision = 0.0          # seconds of driver time
        self._clock = 0.0
        self._last: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def attach(self, ctx: EgoContext) -> None:
        self.ctx = ctx
        self.observations = ObservationBuilder(
            world=ctx.world, frame=ctx.frame, bindings=ctx.bindings,
            ego_id=ctx.ego_id, ego_arm=self.ego_arm, bev=self.bev)
        # A BEV source that needs the ego vehicle gets it here, so a missing
        # dependency or town raster fails during setup rather than on tick one.
        prepare = getattr(self.bev, "prepare", None)
        if callable(prepare):
            prepare(ctx.ego_actor)
        self._build_rig(ctx)
        loader = getattr(self.policy, "load", None)
        if callable(loader):
            # Loaded here, not on the first tick: a policy that cannot build its
            # network should fail while the run is still being set up.
            try:
                loader()
            except ImportError as exc:
                raise ImportError(
                    f"ego policy {self.name!r} could not load: {exc}. Its own "
                    "inference stack has to be importable in this interpreter, "
                    "alongside the CARLA API") from exc

    def _build_rig(self, ctx: EgoContext) -> None:
        """Attach whatever rig the policy asked for, before the run starts.

        Two ways to ask, both optional: `sensors()` returning CameraSpecs (or
        plain dicts, so a policy repository need not import this port), or
        `camera_rig` naming one of `carla_sensors.RIGS`. A policy that does
        neither is a `state`-only policy and gets no rig — which is every
        policy that worked before this module existed.
        """
        declared = None
        sensors = getattr(self.policy, "sensors", None)
        if callable(sensors):
            declared = specs_from(sensors())
        else:
            name = getattr(self.policy, "camera_rig", None)
            if name:
                declared = named_rig(str(name))
        if not declared:
            return
        self.rig = CameraRig(ctx.world, ctx.ego_actor, declared).spawn()
        if not self.rig.active:
            # Not fatal: the offline test double has no camera blueprints, and
            # a policy that cannot see must say so itself rather than have this
            # driver guess that the run is worthless.
            self.notes.append(
                f"camera rig requested but no camera attached: "
                f"{'; '.join(self.rig.failed) or 'no sensors created'}")
        elif self.rig.failed:
            self.notes.append(f"camera rig partly attached: "
                              f"{'; '.join(self.rig.failed)}")

    def reset(self) -> None:
        self.steps = 0
        self.calls = 0
        self.actions.clear()
        self._control = None
        self._clock = 0.0
        self._next_decision = 0.0
        self._last = {}
        resetter = getattr(self.policy, "reset", None)
        if callable(resetter):
            resetter()

    def close(self) -> None:
        if self.rig is not None:
            self.rig.destroy()
        closer = getattr(self.policy, "close", None)
        if callable(closer):
            closer()

    # ------------------------------------------------------------------ #
    def control(self, dt: float):
        """One step: decide if due, otherwise hold the last decision."""
        from .carla_api import carla

        if self.ctx is None or self.observations is None:
            raise EgoDriverError("PolicyEgoDriver.control() before attach()")
        self.calls += 1
        if self._control is not None and self._clock + 1e-9 < self._next_decision:
            self._clock += dt
            return self._control

        interval = 1.0 / self.hz if self.hz > 0 else 0.0
        self._next_decision = self._clock + interval
        self._clock += dt

        action = self._act()
        self._control = self._to_carla_control(action, carla)
        self.steps += 1
        if len(self.actions) < self.keep_actions:
            self.actions.append(self._summarize(action))
        return self._control

    # ------------------------------------------------------------------ #
    def _act(self) -> Dict[str, Any]:
        ctx = self.ctx
        policy = ctx.policy
        observation = self.observations.build(
            ego_actor=ctx.ego_actor,
            ref_path=policy.reference_path,
            ego_script_pose=policy.pose(),
            ego_speed=policy.ego.v)
        if self.rig is not None and self.rig.active:
            observation["sensor"] = {"cameras": self.rig.capture(self._frame())}
        act = getattr(self.policy, "act", None) or getattr(self.policy, "step", None)
        if not callable(act):
            act = self.policy if callable(self.policy) else None
        if act is None:
            raise EgoDriverError(
                f"ego policy {self.name!r} exposes no act()/step()/__call__() to "
                "step; ego_policy_v1 requires one of them")
        action = act(observation)
        if not isinstance(action, dict):
            raise EgoDriverError(
                f"ego policy {self.name!r} returned {type(action).__name__}, not a "
                "mapping; ego_policy_v1 actions are JSON-compatible mappings")
        self._last = action
        return action

    def _frame(self) -> Optional[int]:
        """The world frame the images must be stamped with — see
        `carla_sensors.CameraRig.capture`."""
        try:
            return self.ctx.world.get_snapshot().frame
        except (AttributeError, RuntimeError):
            return None

    def _to_carla_control(self, action: Dict[str, Any], carla):
        control = action.get("control")
        if not isinstance(control, dict):
            raise EgoDriverError(
                f"ego policy {self.name!r} returned no 'control' block. This port "
                "actuates control only (see the module docstring): a policy that "
                "emits waypoints must also return the control its own "
                f"lateral/longitudinal controllers produce. Action keys: "
                f"{sorted(action)}")
        steer = _unit(control.get("steer", 0.0), "steer")
        throttle = _clamp01(control.get("throttle", 0.0))
        brake = _clamp01(control.get("brake", 0.0))
        # No sign flip and no rescaling: `steer` is already normalized in CARLA's
        # own convention (+ = right), which is how the reference agents in this
        # family assign it to carla.VehicleControl.steer. The negation
        # carla_ego.CarlaEgoActuator makes is there because drivev2's steer is in
        # the SCRIPT frame, where headings run the other way.
        return carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                    brake=float(brake),
                                    hand_brake=bool(control.get("hand_brake", False)),
                                    reverse=bool(control.get("reverse", False)))

    @staticmethod
    def _summarize(action: Dict[str, Any]) -> Dict[str, Any]:
        """The parts of an action worth keeping per step."""
        keep = {}
        for key in ("target_speed_mps", "control", "meta"):
            if key in action:
                keep[key] = action[key]
        waypoints = action.get("waypoints")
        if isinstance(waypoints, list):
            keep["waypoints"] = waypoints
        return keep

    # ------------------------------------------------------------------ #
    def metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "driver": "external_policy",
            "policy": self.name,
            "actuates": "control",
            "decision_hz": self.hz,
            "decisions": self.steps,
            "control_steps": self.calls,
            "bev": self.bev.describe() if hasattr(self.bev, "describe") else
                   (None if self.bev is None else "provided"),
            "observation_space": "state+sensor" if (
                self.rig is not None and self.rig.active) else "state",
            "camera_rig": self.rig.describe() if self.rig is not None else None,
        }
        described = getattr(self.policy, "metadata", None)
        if callable(described):
            try:
                meta["policy_metadata"] = described()
            except Exception as exc:            # pragma: no cover - diagnostics
                meta["policy_metadata_error"] = f"{type(exc).__name__}: {exc}"
        last_meta = (self._last or {}).get("meta")
        if isinstance(last_meta, dict):
            meta["last_action_meta"] = last_meta
        if self.notes:
            meta["notes"] = list(self.notes)
        return meta


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _unit(value: Any, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise EgoDriverError(f"control.{what} is not a number: {value!r}") from exc
    if not math.isfinite(v):
        raise EgoDriverError(f"control.{what} is not finite: {value!r}")
    return max(-1.0, min(1.0, v))
