#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/runner.py — connect, spawn, run, grade, report. Entry point.

The highway counterpart of `carla_port/carla_runner.py`, and deliberately the
same shape: synchronous CARLA at the script's own 1/60 s step, background actors
placed from the maneuver-script oracle every tick, the ego driven closed-loop by
a policy, an mp4 with a verdict HUD, and a JSON report.

The loop, per step:

    A. read the ego back from CARLA (CARLA integrated it, so CARLA is the
       authority on where it is)
    B. orchestrate when due (HighwayClosedLoop; cut-in casting in `cutin` mode)
    C. sample the script world at the current script time
    D. write the background actors into CARLA; the ego is driven, not placed
    E. actuate the ego from the policy
    F. world.tick()
    G. collect collisions, grade, capture a video frame
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from carla_port.actuation import CarlaEgoActuator, LongitudinalPID
from carla_port.carla_adapter import carla_actor_to_script_state, spawn_bindings
from carla_port.carla_api import carla, use_fake
from carla_port.carla_collision import CollisionMonitor
from carla_port.carla_sync import (KINEMATIC, PHYSICS, StateSynchronizer,
                                   sample_script_state, world_states)

from . import scenarios as sc_mod
from .closed_loop import DT_TICK, HighwayClosedLoop
from .highway_ego import (DELTA_MAX, EGO_LENGTH, EGO_WIDTH, IDM_V0,
                          IDM_V0_CUTIN, V_MAX, Ego, HighwayEgoPolicy)
from .highway_map import FORWARD_HEADING, HighwayFrame
from .script_bridge import DT, se

BICYCLE = "bicycle"
PHYSICS_EGO = "physics"
EGO_MODES = (PHYSICS_EGO, BICYCLE)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SO_DIR = os.path.join(REPO_ROOT, "scenario_orchestration")
DEFAULT_VIDEO_DIR = os.path.join(HERE, "outputs")
DEFAULT_REPORT_DIR = os.path.join(HERE, "reports")

#: Shorthand names -> policy.json fields for `--policy NAME`.
POLICY_SHORTCUTS = {
    "simlingo": {
        "name": "simlingo",
        "interface": "ego_policy_v1",
        "observation_space": "sensor",
        "action_space": "control",
        "repository": "third_party/simlingo",
        "entry_point": "scenario_orchestration/policy.py",
    },
    "tfv6": {
        "name": "tfv6",
        "interface": "ego_policy_v1",
        "observation_space": "sensor",
        "action_space": "control",
        "repository": "third_party/tfv6",
        "entry_point": "scenario_orchestration/policy.py",
    },
}


def _load_external_ego_driver(cfg: "RunConfig", companion: HighwayEgoPolicy):
    """Load an ego_policy_v1 repository and wrap it in PolicyEgoDriver."""
    if SO_DIR not in sys.path:
        sys.path.insert(0, SO_DIR)
    from contract import PolicyRequest
    import policies as pol_mod
    from carla_port.ego_driver import EgoContext, PolicyEgoDriver

    if cfg.policy_request:
        req = PolicyRequest.from_json(cfg.policy_request)
    elif cfg.policy in POLICY_SHORTCUTS:
        spec = dict(POLICY_SHORTCUTS[cfg.policy])
        env_root = os.environ.get(f"{cfg.policy.upper()}_ROOT")
        if not env_root:
            env_root = os.path.join(os.path.dirname(REPO_ROOT), "third_party",
                                    cfg.policy)
        req = PolicyRequest(**spec,
                            parameters={"repository_path": env_root})
    else:
        raise ValueError(f"unknown --policy {cfg.policy!r}; "
                         f"have {sorted(POLICY_SHORTCUTS)}")
    loaded = pol_mod.load_policy(req, harness_root=None, repo_root=REPO_ROOT)
    driver = PolicyEgoDriver(loaded.policy, name=loaded.name, hz=cfg.policy_hz)
    return driver, loaded


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    scenario: str = sc_mod.CUTIN
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 20.0
    load_timeout: float = 120.0
    town: Optional[str] = "Town04"
    duration: Optional[float] = None
    linger: float = 1.0
    fixed_delta: float = DT
    no_rendering: bool = False
    # world fitting
    road_id: Optional[int] = None
    min_length: Optional[float] = None
    # actors
    base: Optional[str] = None
    sync_mode: str = PHYSICS
    z_offset: float = 0.10
    reground_every: int = 0
    adopt_carla_extents: bool = True
    #: CARLA blueprint for the background actors. A compact hatch/coupe keeps
    #: the fleet legible in the video AND close to the 4.5 m body the scenario
    #: geometry is authored against; the body-size match on its own picked a
    #: 5.2 m box truck for every actor on Town04.
    actor_model: str = "vehicle.audi.tt"
    actor_color: str = "200,30,30"
    #: the ego is painted apart from the traffic — the scripts draw it green.
    ego_model: Optional[str] = None
    ego_color: str = "90,190,110"
    ego: str = sc_mod.EGO_ID
    # ego
    ego_mode: str = PHYSICS_EGO
    scripted_ego: bool = False
    desired_speed: float = IDM_V0
    no_lane_change: bool = False
    lane_change: bool = False
    ego_policy: bool = True
    policy: Optional[str] = None          # simlingo | tfv6 | None = highway IDM
    policy_request: Optional[str] = None    # path to policy.json
    policy_hz: float = 20.0
    # orchestration
    casting: Optional[bool] = None
    cruise: float = 12.0
    cutin_at: Optional[float] = None   # hold the cut-in off until this time
    cutin_along: Optional[float] = None  # override the pin distance, m ahead
    # outputs
    video: Optional[str] = None
    video_dir: str = DEFAULT_VIDEO_DIR
    video_view: str = "both"
    video_size: Tuple[int, int] = (720, 540)
    video_fps: float = 30.0
    video_top_span: Optional[float] = None
    hud: bool = True
    no_video: bool = False
    weather: str = "clear"
    spectator: bool = True
    report: Optional[str] = None
    #: where to write the run in the UPSTREAM verifier's schema (the one
    #: `scripts/scenario_verify.py` reads). Defaults beside `--report`.
    verify_report: Optional[str] = None
    #: seconds between recorded trajectory samples. The verifier walks the
    #: trajectory linearly and its `pose_at_time` takes the last sample at or
    #: before t, so the sampling period is the timing resolution of every
    #: check it makes. 20 Hz puts the commit-instant pose within 0.6 m at
    #: highway speed, well inside the 10 m station window.
    traj_dt: float = 0.05
    verbose: bool = True


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
@dataclass
class Interaction:
    """What the ego did relative to one background actor."""
    actor_id: str
    oncoming: bool
    min_gap: float = float("inf")
    passed: bool = False
    _was_ahead: Optional[bool] = None

    def update(self, along: float, gap: float) -> None:
        self.min_gap = min(self.min_gap, gap)
        ahead = along > 0
        if self._was_ahead is None:
            self._was_ahead = ahead
        elif self._was_ahead and not ahead:
            self.passed = True          # it was in front, now it is behind
            self._was_ahead = False


class HighwayRun:
    """One CARLA run of one highway scenario."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.client = None
        self.world = None
        self.frame: Optional[HighwayFrame] = None
        self.scenario = None            # full, incl. ego (what gets spawned)
        self.loop: Optional[HighwayClosedLoop] = None
        self.policy: Optional[HighwayEgoPolicy] = None
        self.ego_driver = None              # PolicyEgoDriver when --policy is set
        self._external_policy_name: Optional[str] = None
        self.ego_actor = None
        self.actuator: Optional[CarlaEgoActuator] = None
        self.bindings = None
        self.sync = None
        self.collisions = None
        self.recorder = None
        self.t_sim = 0.0
        self.notes: List[str] = []
        self.realized: List[object] = []
        self.interactions: Dict[str, Interaction] = {}
        self._orig_settings = None
        self._orig_weather = None
        self._xtrack_max = 0.0
        self._ego_start_y = 0.0
        #: (x, y, speed) of the ego this step, whoever is driving it
        self._ego_xyv: Optional[Tuple[float, float, float]] = None
        #: script id -> (length, width), as spawned. Written after
        #: spawn_bindings adopts the real CARLA bounding boxes, so the
        #: gaps reported here are between the bodies that actually collide.
        self._extents: Dict[str, Tuple[float, float]] = {}
        #: [t, x, y, heading_deg, speed] rows, in the SCRIPT frame — which is
        #: the frame the upstream verifier assumes (+y along the road, lanes
        #: separated in x, headings in degrees), so no conversion is needed on
        #: the way out. Read back from CARLA, not from the commanded states:
        #: the point of verifying a CARLA run is to check what the simulator
        #: actually did with the plan.
        self._ego_traj: List[List[float]] = []
        self._actor_traj: Dict[str, List[List[float]]] = {}
        self._spawns: Dict[str, List[float]] = {}
        self._cruise: Dict[str, float] = {}
        self._next_traj_t = 0.0

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    def _map_name(self) -> str:
        try:
            return str(self.world.get_map().name).rsplit("/", 1)[-1]
        except (RuntimeError, AttributeError):
            return "?"

    def _already_on(self, town: str) -> bool:
        """CARLA reports map names as e.g. 'Carla/Maps/Town04'."""
        return self._map_name() == str(town).rsplit("/", 1)[-1]

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        cfg = self.cfg
        if cfg.duration is None:
            cfg.duration = sc_mod.spec(cfg.scenario).duration
        self._log(f"connecting to CARLA at {cfg.host}:{cfg.port}")
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(cfg.timeout)
        self.world = self.client.get_world()
        if cfg.town and not self._already_on(cfg.town):
            # Loading a town streams a lot of assets and routinely takes far
            # longer than a normal RPC; the working timeout is restored after.
            self.client.set_timeout(max(cfg.timeout, cfg.load_timeout))
            self._log(f"loading {cfg.town} (timeout {cfg.load_timeout:.0f}s)")
            self.world = self.client.load_world(cfg.town)
            self.client.set_timeout(cfg.timeout)
        else:
            self._log(f"already on {self._map_name()}; not reloading")

        self._orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = cfg.fixed_delta
        s.no_rendering_mode = cfg.no_rendering
        self.world.apply_settings(s)
        self._log(f"synchronous mode, fixed_delta_seconds={cfg.fixed_delta:.5f} "
                  f"(script DT={DT:.5f}, orchestration every {DT_TICK:.2f}s)")
        if abs(cfg.fixed_delta - DT) > 1e-9:
            self.notes.append(
                f"fixed_delta_seconds {cfg.fixed_delta:.5f} != script DT "
                f"{DT:.5f}; scripted states are interpolated between frames")
        self._apply_weather()

        # ---- the road ---- #
        spec = sc_mod.spec(cfg.scenario)
        self.frame = sc_mod.discover_frame(self.world, cfg.scenario,
                                           min_length=cfg.min_length,
                                           road_id=cfg.road_id)
        self._log(self.frame.describe())
        self.notes.extend(self.frame.warnings)

        # ---- the scenario, refitted onto it ---- #
        authored = sc_mod.load(cfg.scenario, cfg.base)
        self.scenario, notes = sc_mod.retarget(authored, self.frame)
        self.notes.extend(notes)
        for n in notes:
            self._log(f"  ! {n}")
        ego_actor, background = sc_mod.split_ego(self.scenario, cfg.ego)
        self.ego_actor = ego_actor
        self._ego_start_y = ego_actor.start[1]

        # ---- CARLA bodies ---- #
        models = {"*": cfg.actor_model} if cfg.actor_model else {}
        colors = {"*": cfg.actor_color} if cfg.actor_color else {}
        if cfg.ego_model:
            models[str(cfg.ego)] = cfg.ego_model
        if cfg.ego_color:
            colors[str(cfg.ego)] = cfg.ego_color
        self.bindings = spawn_bindings(
            self.world, self.frame, self.scenario, z_offset=cfg.z_offset,
            simulate_physics=(cfg.sync_mode == PHYSICS),
            adopt_carla_extents=cfg.adopt_carla_extents,
            models=models, colors=colors, notes=self.notes)
        self._log(f"spawned {len(self.bindings)}/{len(self.scenario.actors)} "
                  f"CARLA vehicles")
        if self.bindings.unbound:
            self.notes.append(
                "script actors with no CARLA body (spawn refused): "
                + ",".join(self.bindings.unbound))
        background.simulate()

        # spawn_bindings has just written the real CARLA bounding boxes back
        # into the script actors, so this is the geometry that collides.
        self._extents = {str(a.id): (float(a.length), float(a.width))
                         for a in self.scenario.actors}

        self.collisions = CollisionMonitor(self.world, self.bindings)
        self._log(f"attached {self.collisions.attach()} collision sensors")

        # ---- orchestration ---- #
        # A negative --cutin-at means "no delay": run the scenario on its own
        # authored deadline. That is the default, and it has to be, because
        # holding the cut-in off is not free. The actors cruise nominally while
        # the orchestrator is asleep, and on `scenario_cutin` they are faster
        # than the ego — so six seconds of silence puts every candidate 30-50 m
        # ahead of the pin, and the run then measures nothing but the
        # orchestrator failing to find anyone who can fall back that far.
        cutin_at = cfg.cutin_at
        if cutin_at is not None and cutin_at < 0:
            cutin_at = None
        self.loop = HighwayClosedLoop(self.frame, background, spec,
                                      casting=cfg.casting, cruise=cfg.cruise,
                                      cutin_at=cutin_at,
                                      cutin_along=cfg.cutin_along)
        if cutin_at is not None and self.loop.orch is not None:
            deadline = float(self.loop.orch.spec.get("t", cutin_at))
            self._log(f"cut-in starts at t={cutin_at:.1f}s (merge by t={deadline:.1f}s)")
        self._log(f"orchestration: {self.loop.status}")
        self._check_cutin_clearance(ego_actor, background)

        # ---- the ego ---- #
        if not cfg.scripted_ego:
            self._build_ego(ego_actor)

        for a in background.actors:
            aid = str(a.id)
            self.interactions[aid] = Interaction(
                actor_id=aid, oncoming=not sc_mod._is_forward(a.start[2]))
            # Spawn pose + body, in the verifier's [x, y, heading, L, W] form.
            self._spawns[aid] = [round(float(a.start[0]), 3),
                                 round(float(a.start[1]), 3),
                                 round(float(a.start[2]), 3),
                                 round(float(getattr(a, "length", 4.5)), 3),
                                 round(float(getattr(a, "width", 2.0)), 3)]
            cr = getattr(a, "cruise", None)
            if cr is not None:
                self._cruise[aid] = float(cr)
        ex, ey, eh = ego_actor.start[0], ego_actor.start[1], ego_actor.start[2]
        self._spawns[str(cfg.ego)] = [round(float(ex), 3), round(float(ey), 3),
                                      round(float(eh), 3),
                                      round(float(getattr(ego_actor, "length", 4.5)), 3),
                                      round(float(getattr(ego_actor, "width", 2.0)), 3)]

        self.sync = StateSynchronizer(self.frame, self.bindings,
                                      mode=cfg.sync_mode, z_offset=cfg.z_offset,
                                      reground_every=cfg.reground_every)
        if cfg.spectator:
            self._place_spectator()
        if not cfg.no_video:
            self._build_recorder()

        # place everyone at their initial state and let CARLA settle one tick
        self.sync.apply(world_states(self.scenario, 0.0))
        self.world.tick()
        if self.recorder is not None:
            self.recorder.start()

    #: Bumper-to-bumper metres a cut-in pin should leave once real CARLA bodies
    #: are in play. Below this, contact is expected rather than a bug.
    #:
    #: Measured, not guessed: `scenario_cutin`'s `along: 6.0` works out to
    #: 1.55 m against the bodies Town04 spawns (3.7 m ego, 5.2 m actor) and
    #: still makes contact, in BOTH ego modes. A merge is a lateral sweep, so
    #: the corner of the merging car passes closer than the straight-line
    #: bumper gap suggests; the threshold has to sit above the clearance that
    #: was observed to fail.
    CUTIN_TIGHT_CLEARANCE = 2.5

    def _check_cutin_clearance(self, ego_actor, background) -> None:
        """Say up front how much room the authored cut-in pin actually leaves.

        `cutin: {along: 6.0}` places the merging car 6 m ahead of the ego
        CENTRE TO CENTRE. In the script world the bodies are a nominal
        4.5 x 2.0 m and nothing enforces contact anyway — pygame has no physics,
        so a pin that overlaps is simply drawn overlapping. Here
        `spawn_bindings(adopt_carla_extents=True)` replaces those numbers with
        the spawned vehicle's real bounding box, and CARLA's collision sensors
        do enforce it. A pin that was comfortable on screen can then leave
        about a metre of bumper gap, and the merge grazes the ego.

        That is a property of the scenario meeting real geometry, not a fault
        in the port or the ego policy — the scripted ego from the authored YAML
        touches too. It is worth saying out loud, because otherwise the run
        looks like the ego policy failed.
        """
        if self.loop.orch is None:
            return
        spec = getattr(self.loop.orch, "spec", None) or {}
        along = abs(float(spec.get("along", 0.0)))
        lat = abs(float(spec.get("lat", 0.0)))
        if lat > 0.5 * self.frame.lane_width:
            return                     # the pin is not in the ego's lane
        ego_len = float(getattr(ego_actor, "length", EGO_LENGTH))
        # the role can be cast to any of them, so take the worst case
        others = [float(getattr(a, "length", 4.5)) for a in background.actors]
        longest = max(others) if others else 4.5
        clearance = along - (ego_len + longest) / 2.0
        msg = (f"cut-in pin is along={along:.1f} m centre-to-centre; with the "
               f"spawned bodies ({ego_len:.1f} m ego, up to {longest:.1f} m "
               f"actor) that is {clearance:.2f} m bumper to bumper")
        if clearance < self.CUTIN_TIGHT_CLEARANCE:
            self.notes.append(
                msg + " — contact during the merge is expected here, and is "
                "the scenario's geometry meeting real vehicle extents, not "
                "the ego policy failing. Widen `along` in the scenario YAML "
                "to give the merge room.")
        self._log(f"  {msg}")

    def _build_ego(self, ego_actor) -> None:
        from carla_port.ego_driver import EgoContext

        x, y, h = ego_actor.start
        v0 = ego_actor.speeds[0] if getattr(ego_actor, "speeds", None) else \
            self.cfg.desired_speed
        ego = Ego(x=x, y=y, theta=math.radians(h), v=float(v0))
        allow_lc = sc_mod.spec(self.cfg.scenario).ego_lane_changes
        if self.cfg.no_lane_change:
            allow_lc = False
        elif self.cfg.lane_change:                    # --lane-change overrides
            allow_lc = True
        self.policy = HighwayEgoPolicy(
            self.frame, ego, self.loop.sc, atime=0.0,
            desired_speed=self.cfg.desired_speed,
            allow_lane_change=allow_lc)
        binding = self.bindings.get(self.cfg.ego)
        if binding is None:
            self.notes.append("the ego has no CARLA body; falling back to the "
                              "scripted ego")
            self.policy = None
            return

        use_external = bool(self.cfg.policy or self.cfg.policy_request)
        if use_external:
            if self.cfg.ego_mode != PHYSICS_EGO:
                raise RuntimeError(
                    f"--ego-mode {self.cfg.ego_mode!r} cannot drive an external "
                    f"ego policy: it emits carla.VehicleControl, which needs "
                    f"'{PHYSICS_EGO}'")
            binding.carla_actor.set_simulate_physics(True)
            driver, loaded = _load_external_ego_driver(self.cfg, self.policy)
            self.ego_driver = driver
            self._external_policy_name = loaded.name
            driver.attach(EgoContext(
                world=self.world, frame=self.frame, bindings=self.bindings,
                ego_id=self.cfg.ego, ego_actor=binding.carla_actor,
                mode=self.cfg.scenario, policy=self.policy,
                fixed_delta=self.cfg.fixed_delta))
            driver.reset()
            meta = driver.metadata()
            self._log(
                f"ego: external policy {meta.get('policy', '?')} "
                f"-> VehicleControl at {meta.get('decision_hz', '?')} Hz "
                f"(state read back from CARLA); highway IDM kept for route "
                f"measurement and orchestration only")
            return

        binding.carla_actor.set_simulate_physics(self.cfg.ego_mode == PHYSICS_EGO)
        self.actuator = CarlaEgoActuator(binding.carla_actor,
                                         pid=LongitudinalPID(),
                                         delta_max=DELTA_MAX, v_max=V_MAX)
        self._log(f"ego: policy=highway IDM+lane-select"
                  f"{'' if allow_lc else ' (lane-keeping)'}, "
                  f"mode={self.cfg.ego_mode}, "
                  f"home lane {self.policy.home_lane}, "
                  f"max_steer={self.actuator.max_steer_deg:.0f} deg")

    def _apply_weather(self) -> None:
        if self.cfg.weather != "clear":
            return
        try:
            self._orig_weather = self.world.get_weather()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        except (RuntimeError, AttributeError):
            self._orig_weather = None

    def _place_spectator(self) -> None:
        try:
            spec = self.world.get_spectator()
        except (RuntimeError, AttributeError):
            return
        loc = self.frame.to_carla_location(0.0, 0.0, self.frame.anchor.z + 60.0)
        spec.set_transform(carla.Transform(
            loc, carla.Rotation(pitch=-90.0, yaw=self.frame.to_carla_yaw(90.0))))

    def _build_recorder(self) -> None:
        from carla_port.carla_video import Recorder
        cfg = self.cfg
        path = cfg.video
        if not path:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(cfg.video_dir, exist_ok=True)
            path = os.path.join(cfg.video_dir, f"{cfg.scenario}_{stamp}.mp4")
        ego_actor = None
        b = self.bindings.get(cfg.ego)
        if b is not None:
            ego_actor = b.carla_actor
        elif cfg.video_view in ("chase", "both"):
            self.notes.append("no CARLA body for the ego; the chase camera was "
                              "skipped and only the top view recorded")
        try:
            self.recorder = Recorder(
                self.world, self.frame, path, view=cfg.video_view,
                width=cfg.video_size[0], height=cfg.video_size[1],
                fps=cfg.video_fps, sim_fps=1.0 / cfg.fixed_delta, hud=cfg.hud,
                top_span=cfg.video_top_span)
            n = self.recorder.attach(ego_actor)
            if getattr(self.recorder, "hud_unavailable", False):
                self.notes.append("Pillow is not installed; the video has no HUD")
            self._log(f"recording {n} camera(s) ({cfg.video_view}) at "
                      f"{cfg.video_fps:g} fps -> {path}")
        except Exception as exc:                      # ffmpeg / camera trouble
            self.notes.append(f"video disabled: {type(exc).__name__}: {exc}")
            self.recorder = None

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        cfg = self.cfg
        dt = cfg.fixed_delta
        n_steps = int(round((cfg.duration + cfg.linger) / dt))
        for _ in range(n_steps):
            # A. read the ego back from CARLA
            self._read_ego()
            # B. orchestrate when due
            if self.policy is not None:
                self.loop.tick(self.policy.ego, self.t_sim)
                self.policy.asc = self.loop.sc
                self.policy.atime = self.loop.atime
            elif self.loop.casting and self.ego_actor is not None:
                # Casting needs a live ego pose, not a policy. Sampling the
                # scripted plan gives one, so `--scripted-ego --scenario cutin`
                # reproduces the upstream conditions the YAML was authored for
                # (an ego that brakes 13 -> 4 m/s) and is the way to compare
                # this port's casting against a highway run.
                st = sample_script_state(self.ego_actor, self.t_sim)
                self.loop.tick(Ego(x=st.x, y=st.y,
                                   theta=math.radians(st.heading),
                                   v=st.speed), self.t_sim)
                self._ego_xyv = (st.x, st.y, st.speed)
            # C/D. sample the script world and write the background in
            t_next = self.t_sim + dt
            self.loop.advance(dt)
            states = world_states(self.loop.sc, self.loop.atime)
            if self.policy is None and self.ego_actor is not None:
                # Scripted ego: place it from its own, never-rebased plan.
                # Taking the whole scenario here would also overwrite the
                # BACKGROUND with un-rebased states and quietly undo every
                # orchestration decision made so far.
                states[str(self.cfg.ego)] = sample_script_state(self.ego_actor,
                                                                t_next)
            self.sync.apply(states)
            # E. actuate the ego
            self._drive_ego(dt)
            # F. advance CARLA
            self.collisions.set_time(t_next)
            self.world.tick()
            self.t_sim = t_next
            # G. collect
            for rc in self.collisions.drain():
                self.realized.append(rc)
                self._log(f"  t={self.t_sim:6.2f}  {rc}")
            self._track(states)
            self._sample_traj()
            if self.recorder is not None:
                self.recorder.capture(self._hud())
        return self.report()

    def _read_ego(self) -> None:
        """Publish where the ego is this step, whoever is driving it.

        A scripted ego has no policy object to ask, so it is sampled from its
        own plan. Without that branch every per-actor gap and the distance
        travelled came back `None` for `--scripted-ego`, which is exactly the
        run you want the numbers from when comparing against upstream.
        """
        if self.policy is None:
            if self.ego_actor is not None:
                st = sample_script_state(self.ego_actor, self.t_sim)
                self._ego_xyv = (st.x, st.y, st.speed)
            return
        if self.cfg.ego_mode == PHYSICS_EGO:
            b = self.bindings.get(self.cfg.ego)
            if b is not None:
                st = carla_actor_to_script_state(self.cfg.ego, b.carla_actor,
                                                 self.frame)
                e = self.policy.ego
                e.x, e.y, e.theta, e.v = (st.x, st.y, math.radians(st.heading),
                                          st.speed)
        e = self.policy.ego
        self._ego_xyv = (e.x, e.y, e.v)
        # cross-track is the controller's tracking error, so it stops
        # being meaningful the moment something hits the EGO
        if not self.ego_collisions():
            self._xtrack_max = max(self._xtrack_max,
                                   abs(self.policy.lane_offset()))

    def _drive_ego(self, dt: float) -> None:
        if self.policy is None:
            return
        b = self.bindings.get(self.cfg.ego)
        if b is None:
            return
        if self.ego_driver is not None:
            b.carla_actor.apply_control(self.ego_driver.control(dt))
            return
        throttle, steer = self.policy.command(now=self.t_sim, dt=dt)
        if self.cfg.ego_mode == BICYCLE:
            self.policy.integrate(throttle, steer, dt)
            e = self.policy.ego
            tf = self.frame.to_carla_transform(
                e.x, e.y, math.degrees(e.theta) % 360.0, b.ground_z + self.cfg.z_offset)
            b.carla_actor.set_transform(tf)
            b.carla_actor.set_target_velocity(
                self.frame.to_carla_velocity(e.v, math.degrees(e.theta) % 360.0))
            return
        ctrl, _v_target = self.actuator.control(self.policy, throttle, steer, dt)
        b.carla_actor.apply_control(ctrl)

    def _track(self, states: Dict[str, object]) -> None:
        if self._ego_xyv is None:
            return
        ex, ey, _ = self._ego_xyv
        el, ew = self._extents.get(str(self.cfg.ego), (EGO_LENGTH, EGO_WIDTH))
        for aid, it in self.interactions.items():
            st = states.get(aid)
            if st is None:
                continue
            al, aw = self._extents.get(aid, (4.5, 2.0))
            along = st.y - ey
            lat = st.x - ex
            # Separation between two road-aligned boxes, not a radius: on a
            # straight road the ego passing 3.5 m abeam is a clean lane apart,
            # but hypot() - half_length calls that a zero gap. The extents are
            # the spawned CARLA bodies', so this is the clearance that decides
            # whether the collision sensor fires.
            d_long = abs(along) - (el + al) / 2.0
            d_lat = abs(lat) - (ew + aw) / 2.0
            gap = max(d_long, d_lat, 0.0)
            it.update(along, gap)

    # ------------------------------------------------------------------ #
    # Trajectory recording, in the upstream verifier's schema
    # ------------------------------------------------------------------ #
    def _sample_traj(self) -> None:
        """One [t, x, y, heading, v] row per vehicle, decimated to `traj_dt`.

        Poses come back from CARLA through `carla_actor_to_script_state`, not
        from the commanded script states: `sync_mode=physics` writes a target
        velocity and lets the physics settle, so the two differ, and it is the
        simulated motion the verifier is being asked about.
        """
        if self.bindings is None or self.t_sim + 1e-9 < self._next_traj_t:
            return
        self._next_traj_t = self.t_sim + max(self.cfg.traj_dt, 1e-3)
        t = round(self.t_sim, 4)
        ego_id = str(self.cfg.ego)
        for b in self.bindings:
            aid = str(b.script_actor_id)
            try:
                st = carla_actor_to_script_state(aid, b.carla_actor, self.frame)
            except RuntimeError:
                continue            # actor destroyed mid-run; drop the sample
            row = [t, round(st.x, 3), round(st.y, 3),
                   round(st.heading % 360.0, 3), round(st.speed, 3)]
            if aid == ego_id:
                self._ego_traj.append(row)
            else:
                self._actor_traj.setdefault(aid, []).append(row)
        if ego_id not in self.bindings.ids() and self._ego_xyv is not None:
            ex, ey, ev = self._ego_xyv
            hd = (self.policy.heading_deg if self.policy is not None
                  else FORWARD_HEADING)
            self._ego_traj.append([t, round(ex, 3), round(ey, 3),
                                   round(hd % 360.0, 3), round(ev, 3)])

    def verify_report(self) -> dict:
        """The run in the schema `scripts/scenario_verify.py` reads.

        The port's own `report()` grades in CARLA's terms — did the collision
        sensor fire, did the orchestrator declare a merge. That is a different
        question from the upstream one, which is asked of the *trajectories*
        and does not trust the orchestrator's own verdict: the cut-in verifier
        re-derives where everybody was at the commit instant and checks the
        four criteria in `docs/SCENARIOS_AND_VALIDATION.md` against the
        recorded motion. Emitting this file is what lets a CARLA run be graded
        by exactly the same code as a pygame run.

        Frames line up without conversion: the script frame this port works in
        is the frame the upstream scenarios are authored in (+y along the road,
        lanes separated in x, headings in degrees).
        """
        mode = self.cfg.scenario
        ego_id = str(self.cfg.ego)
        out: Dict[str, object] = {
            "scenario": mode,
            "source": "carla_highway",
            "town": self.cfg.town,
            "lane_width": round(self.frame.lane_width, 4),
            "duration": round(self.t_sim, 3),
            "spawns": dict(self._spawns),
            "cruise": dict(self._cruise),
            "vehicle_dims": {aid: {"length": round(l, 3), "width": round(w, 3)}
                             for aid, (l, w) in sorted(self._extents.items())},
            "ego_trajectory": self._ego_traj,
            "actor_trajectories": self._actor_traj,
        }
        roles = self.loop.roles or {}
        if roles:
            out["actor_roles"] = {str(k): str(v) for k, v in roles.items()}
        if mode == sc_mod.CUTIN:
            holder = self.loop.holder
            if holder is not None:
                out["cast"] = {"cutin": str(holder)}
            # `success` is the orchestrator's own claim; the verifier requires
            # it to be True and then goes on to disbelieve everything else,
            # re-deriving the merge from the trajectories.
            out["cutin"] = {
                "performer": (str(holder) if holder is not None else None),
                "success": self.loop.outcome == "merged",
                "outcome": self.loop.outcome,
                "t_commit": (round(self.loop.t_commit, 3)
                             if self.loop.t_commit is not None else None),
            }
        return out

    # ------------------------------------------------------------------ #
    def _role_panel(self):
        """Intention matrix for the video HUD (same card as the pygame editor)."""
        from carla_highway.script_bridge import ROLE_CUTIN, ROLE_BLOCK, ROLE_NOMINAL, co

        sc = self.loop.sc if self.loop else None
        if sc is None:
            return None, None, None
        ego = str(self.cfg.ego)
        others = [a for a in sc.actors if str(a.id) != ego]
        if not others:
            return None, None, None

        has_cutin_block = any(getattr(a, "cutin", None) or getattr(a, "block", None)
                              for a in others)
        live = self.loop.roles if (self.loop and self.loop.casting) else {}

        if self.loop and self.loop.casting:
            # Until the first cast, show everyone as nominal (delayed cut-in).
            roles = {str(a.id): live.get(str(a.id), ROLE_NOMINAL)
                     for a in others} if live else {
                str(a.id): ROLE_NOMINAL for a in others}
        else:
            def assigned(a) -> str:
                if getattr(a, "cutin", None):
                    return ROLE_CUTIN
                if getattr(a, "block", None):
                    return ROLE_BLOCK
                role = getattr(a, "role", None)
                return str(role) if role else ROLE_NOMINAL
            roles = {str(a.id): assigned(a) for a in others}
        columns = co.panel_columns_for(roles, has_cutin_block=has_cutin_block)

        scores: dict = {}
        flat = self.loop.scores if self.loop else {}
        for a in others:
            aid = str(a.id)
            cell: dict = {}
            if aid in flat:
                # Live casting score is the cut-in placement score.
                cell[ROLE_CUTIN] = float(flat[aid])
            if self.policy is not None:
                ego_pose = (self.policy.ego.x, self.policy.ego.y,
                            math.degrees(self.policy.ego.theta))
                pose = (a.pose_at_time(self.loop.atime)
                        if getattr(a, "traj", None) else a.start)
                lw = sc.map.lane_width
                for rkey, _lbl in columns:
                    if rkey in (ROLE_NOMINAL, ROLE_CUTIN, ROLE_BLOCK) or rkey in cell:
                        continue
                    fn = co.STRESS_SCORE_FN.get(rkey)
                    if fn is None:
                        continue
                    cell[rkey] = float(fn(pose, ego_pose, lw))
            if cell:
                scores[aid] = cell
        return roles, columns, (scores or None)

    def _hud(self):
        from carla_port.carla_video import Hud
        p = self.policy
        lane = p.frame.lane_index_of(p.ego.x) if p else -1
        line3 = (f"lane {lane} -> {p.target_lane}   changes {p.n_lane_changes}"
                 f"   {p.status}" if p else "scripted ego")
        if p and self._external_policy_name:
            line3 = f"ego {self._external_policy_name}   " + line3
        grade = self._grade()
        roles, columns, scores = self._role_panel()
        return Hud(t=self.t_sim, mode=self.cfg.scenario, seed=0,
                   town=self.cfg.town or "", junction=self.frame.road_id,
                   ego_speed=(p.ego.v if p else 0.0),
                   ego_mode=self.cfg.ego_mode,
                   xtrack=self._xtrack_max,
                   line3=line3,
                   badges=[(k.upper(), v) for k, v in grade["checks"].items()],
                   hero=self.loop.holder, hero_label="cut-in",
                   ok=grade["success"],
                   interventions=self.loop.n_interventions,
                   last_intervention=self.loop.status,
                   realized=(str(self.ego_collisions()[-1])
                             if self.ego_collisions() else ""),
                   roles=roles, role_columns=columns, role_scores=scores)

    def ego_collisions(self) -> List[object]:
        """Realized collisions the EGO was part of.

        The grade asks whether the policy under test crashed, so a background
        actor's collision must not fail the run. They happen for a mundane
        reason: the scripts run for 40 s while the fitted straight is finite,
        so an actor that outlives the scenario drives off the end of the fit
        and into the scenery — long after the part being measured. Those are
        reported, never graded.
        """
        ego = str(self.cfg.ego)
        return [rc for rc in self.realized
                if str(getattr(rc, "actor_id", "")) == ego
                or str(getattr(rc, "other_id", "")) == ego]

    def _grade(self) -> dict:
        """Per-scenario success, in the terms each scenario was written in."""
        mode = self.cfg.scenario
        p = self.policy
        hit = bool(self.ego_collisions())
        checks: Dict[str, bool] = {"no_hit": not hit}
        if mode == sc_mod.CUTIN:
            out = self.loop.outcome
            checks["merged"] = (out == "merged")
            success = (out == "merged") and not hit
            detail = f"cut-in {out or 'undecided'}"
        elif mode == sc_mod.HARD_BRAKE:
            lead = self.interactions.get("1")
            passed = bool(lead and lead.passed)
            checks["passed"] = passed
            success = passed and not hit
            detail = ("evaded the slow lead" if passed else
                      "stuck behind the slow lead")
        else:                                   # overtake
            blocker = self.interactions.get("1")
            passed = bool(blocker and blocker.passed)
            home = bool(p and p.target_lane == p.home_lane)
            checks["passed"] = passed
            checks["home"] = home
            success = passed and home and not hit
            detail = ("passed the blocker and returned" if passed and home
                      else "passed, still out of lane" if passed
                      else "never got past the blocker")
        return {"success": bool(success), "checks": checks, "detail": detail}

    def report(self) -> dict:
        p = self.policy
        grade = self._grade()
        rep = {
            "scenario": self.cfg.scenario,
            "town": self.cfg.town,
            "duration": round(self.t_sim, 3),
            "fixed_delta": self.cfg.fixed_delta,
            "frame": {
                "road_id": self.frame.road_id,
                "section_id": self.frame.section_id,
                "theta": round(self.frame.theta, 3),
                "lane_width": round(self.frame.lane_width, 3),
                "num_lanes": self.frame.num_lanes,
                "length": round(self.frame.length, 1),
                "lane_fit_error": round(self.frame.lane_fit_error(), 4),
                "lanes": [{"index": i, "carla_lane_id": ln.lane_id,
                           "x": round(ln.offset, 3),
                           "oncoming": not ln.same_direction}
                          for i, ln in enumerate(self.frame.lanes)],
            },
            "ego": {
                "mode": ("scripted" if p is None else self.cfg.ego_mode),
                "policy": ("scripted" if p is None
                           else (self._external_policy_name or
                                 "highway IDM + lane select")),
                "home_lane": (p.home_lane if p else None),
                "final_lane": (p.frame.lane_index_of(p.ego.x) if p else None),
                "lane_changes": (p.n_lane_changes if p else 0),
                "used_oncoming_lane": (p.used_oncoming if p else False),
                "max_cross_track": round(self._xtrack_max, 3),
                "distance": (round(self._ego_xyv[1] - self._ego_start_y, 2)
                             if self._ego_xyv else None),
                "final_speed": (round(self._ego_xyv[2], 2)
                                if self._ego_xyv else None),
            },
            "orchestration": self.loop.summary(),
            "body_extents": {aid: [round(l, 2), round(w, 2)]
                             for aid, (l, w) in sorted(self._extents.items())},
            "interactions": {
                aid: {"oncoming": it.oncoming,
                      "min_gap": (round(it.min_gap, 3)
                                  if it.min_gap != float("inf") else None),
                      "passed": it.passed}
                for aid, it in self.interactions.items()},
            "realized_collisions": [str(rc) for rc in self.realized],
            "ego_collisions": [str(rc) for rc in self.ego_collisions()],
            "grade": grade,
            "notes": self.notes,
        }
        if self.recorder is not None:
            rep["video"] = self.recorder.path
        if self.ego_driver is not None:
            rep["ego_driver"] = self.ego_driver.metadata()
        return rep

    def teardown(self) -> None:
        if self.ego_driver is not None:
            try:
                self.ego_driver.close()
            except Exception as exc:
                self.notes.append(f"ego driver close: {exc}")
        try:
            if self.recorder is not None:
                self.recorder.close()
        except Exception as exc:
            self.notes.append(f"video finalize failed: {exc}")
        for closer in (getattr(self.collisions, "destroy", None),
                       getattr(self.bindings, "destroy", None)):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        try:
            if self._orig_weather is not None:
                self.world.set_weather(self._orig_weather)
            if self._orig_settings is not None:
                self.world.apply_settings(self._orig_settings)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
def summarize(rep: dict) -> str:
    g = rep["grade"]
    o = rep["orchestration"]
    e = rep["ego"]
    f = rep["frame"]
    out = [
        "",
        f"=== {rep['scenario']} on {rep['town']} "
        f"(road {f['road_id']}.{f['section_id']}, {f['num_lanes']} lanes, "
        f"{f['length']:.0f} m, fit {f['lane_fit_error']:.2f} m) ===",
        f"  ego        {e['policy']} / {e['mode']}: "
        f"{e['lane_changes']} lane change(s)"
        + (", used the oncoming lane" if e["used_oncoming_lane"] else "")
        + f", {e['distance']} m travelled, cross-track max {e['max_cross_track']} m",
        f"  orchestr.  casting={o['casting']} holder={o['holder']} "
        f"outcome={o['outcome']} interventions={o['interventions']}",
    ]
    for aid, it in sorted(rep["interactions"].items()):
        out.append(f"  actor {aid}    min gap {it['min_gap']} m  "
                   f"{'PASSED' if it['passed'] else 'not passed'}"
                   + ("  (oncoming)" if it["oncoming"] else ""))
    for rc in rep["ego_collisions"]:
        out.append(f"  COLLISION  {rc}")
    for rc in rep["realized_collisions"]:
        if rc not in rep["ego_collisions"]:
            out.append(f"  (background) {rc}")
    checks = "  ".join(f"{k}{'+' if v else '-'}" for k, v in g["checks"].items())
    out.append(f"  VERDICT    {'PASS' if g['success'] else 'FAIL'}  [{checks}]  "
               f"{g['detail']}")
    for n in rep["notes"]:
        out.append(f"  ! {n}")
    if rep.get("video"):
        out.append(f"  video      {rep['video']}")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m carla_highway.runner",
        description="Run a highway orchestration scenario in CARLA.")
    p.add_argument("--scenario", default=sc_mod.CUTIN, choices=list(sc_mod.MODES),
                   help="which scenario to run (default: cutin)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--load-timeout", type=float, default=120.0,
                   help="separate, longer timeout for load_world")
    p.add_argument("--town", default="Town04",
                   help="CARLA town to load; Town04 has both the multi-lane "
                        "highway and a two-way road (default: Town04)")
    p.add_argument("--duration", type=float, default=None,
                   help="simulated seconds; default is per-scenario "
                        "(cutin 12, hard_brake 14, overtake 20 — the "
                        "oncoming car only reaches the corridor at t~8)")
    p.add_argument("--linger", type=float, default=1.0)
    p.add_argument("--fixed-delta", type=float, default=DT)
    p.add_argument("--road-id", type=int, default=None,
                   help="pick the road section explicitly")
    p.add_argument("--min-length", type=float, default=None,
                   help="override how much straight road the scenario needs")
    p.add_argument("--base", default=None, help="run a scenario YAML of your own")
    p.add_argument("--sync-mode", default=PHYSICS, choices=[PHYSICS, KINEMATIC],
                   help="how BACKGROUND actors are driven")
    p.add_argument("--ego-mode", default=PHYSICS_EGO, choices=list(EGO_MODES),
                   help="physics: IDM -> PID -> VehicleControl, ego read back "
                        "from CARLA (default). bicycle: the policy's own "
                        "kinematic model, mirrored into CARLA")
    p.add_argument("--scripted-ego", action="store_true",
                   help="no policy: drive the ego along its maneuver plan")
    p.add_argument("--desired-speed", type=float, default=None,
                   help=f"ego free-flow speed, m/s (default {IDM_V0}, "
                        f"{IDM_V0_CUTIN} for --scenario cutin, matching the "
                        "ego cruise those YAMLs are authored at)")
    p.add_argument("--no-lane-change", action="store_true",
                   help="lane-keeping ego only (the drivev2 baseline); "
                        "hard_brake and overtake are unsolvable this way")
    p.add_argument("--lane-change", action="store_true",
                   help="let the ego change lanes even in modes that default "
                        "to lane-keeping (cutin)")
    p.add_argument("--casting", dest="casting", action="store_true", default=None,
                   help="force cut-in role casting on")
    p.add_argument("--no-casting", dest="casting", action="store_false",
                   help="force cut-in role casting off")
    p.add_argument("--cruise", type=float, default=12.0)
    p.add_argument("--cutin-at", type=float, default=None, dest="cutin_at",
                   help="hold cut-in orchestration off until this sim time, "
                        "moving the merge deadline ~lc_duration later. Default "
                        "and any negative value: no delay, use the deadline "
                        "the scenario YAML authored")
    p.add_argument("--cutin-along", type=float, default=None, dest="cutin_along",
                   help="override the cut-in pin distance, m ahead of the ego "
                        "centre-to-centre (YAML default 6.0). Real CARLA bodies "
                        "are longer than the 4.5 m the scenario assumes, so the "
                        "authored pin leaves ~1.5 m bumper to bumper; anything "
                        "up to 10 m still counts as a merge and still verifies")
    p.add_argument("--policy", default=None, choices=sorted(POLICY_SHORTCUTS),
                   help="external ego_policy_v1 policy (default: highway IDM)")
    p.add_argument("--policy-request", default=None,
                   help="path to policy.json (overrides --policy)")
    p.add_argument("--policy-hz", type=float, default=20.0,
                   help="external policy decision rate in Hz (default 20)")
    p.add_argument("--video", default=None)
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--video-view", default="both", choices=["top", "chase", "both"])
    p.add_argument("--video-size", default="720x540")
    p.add_argument("--video-fps", type=float, default=30.0)
    p.add_argument("--video-top-span", type=float, default=None)
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-rendering", action="store_true",
                   help="headless CARLA; implies --no-video")
    p.add_argument("--weather", default="clear", choices=["clear", "keep"])
    p.add_argument("--no-spectator", action="store_true")
    p.add_argument("--actor-model", default="vehicle.audi.tt",
                   help="CARLA blueprint for the background actors "
                        "(default vehicle.audi.tt); falls back to the closest "
                        "body-size match if this build has no such blueprint")
    p.add_argument("--actor-color", default="200,30,30",
                   help="R,G,B paint for the background actors (default red)")
    p.add_argument("--ego-model", default=None,
                   help="CARLA blueprint for the ego (default: closest match "
                        "to its declared body)")
    p.add_argument("--ego-color", default="90,190,110",
                   help="R,G,B paint for the ego (default green, as the "
                        "scripts draw it)")
    p.add_argument("--report", default=None, help="write the JSON report here")
    p.add_argument("--verify-report", default=None,
                   help="write the run in the upstream verifier's schema here "
                        "(scripts/scenario_verify.py). Defaults to "
                        "<report>.verify.json when --report is given")
    p.add_argument("--traj-dt", type=float, default=0.05,
                   help="seconds between recorded trajectory samples "
                        "(default 0.05 = 20 Hz)")
    p.add_argument("--quiet", action="store_true")
    return p


def config_from_args(args) -> RunConfig:
    w, _, h = args.video_size.partition("x")
    return RunConfig(
        scenario=args.scenario, host=args.host, port=args.port,
        timeout=args.timeout, load_timeout=args.load_timeout,
        town=args.town, duration=args.duration,
        linger=args.linger, fixed_delta=args.fixed_delta,
        no_rendering=args.no_rendering, road_id=args.road_id,
        min_length=args.min_length, base=args.base, sync_mode=args.sync_mode,
        ego_mode=args.ego_mode, scripted_ego=args.scripted_ego,
        desired_speed=(args.desired_speed if args.desired_speed is not None
                       else (IDM_V0_CUTIN if args.scenario == sc_mod.CUTIN
                             else IDM_V0)),
        no_lane_change=args.no_lane_change,
        lane_change=args.lane_change,
        casting=args.casting, cruise=args.cruise, cutin_at=args.cutin_at,
        cutin_along=args.cutin_along,
        policy=args.policy, policy_request=args.policy_request,
        policy_hz=args.policy_hz,
        video=args.video,
        video_dir=args.video_dir, video_view=args.video_view,
        video_size=(int(w), int(h or 540)), video_fps=args.video_fps,
        video_top_span=args.video_top_span, hud=not args.no_hud,
        no_video=args.no_video or args.no_rendering, weather=args.weather,
        actor_model=args.actor_model, actor_color=args.actor_color,
        ego_model=args.ego_model, ego_color=args.ego_color,
        spectator=not args.no_spectator, report=args.report,
        verify_report=args.verify_report, traj_dt=args.traj_dt,
        verbose=not args.quiet)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    run = HighwayRun(cfg)
    try:
        run.setup()
        rep = run.run()
    finally:
        run.teardown()
    print(summarize(rep))
    if cfg.report:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.report)) or ".",
                    exist_ok=True)
        with open(cfg.report, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"  report     {cfg.report}")
    vpath = cfg.verify_report or (
        cfg.report[:-5] + ".verify.json" if (cfg.report or "").endswith(".json")
        else (cfg.report + ".verify.json" if cfg.report else None))
    if vpath:
        os.makedirs(os.path.dirname(os.path.abspath(vpath)) or ".",
                    exist_ok=True)
        with open(vpath, "w") as fh:
            json.dump(run.verify_report(), fh, indent=2)
        print(f"  verify     {vpath}")
    return 0 if rep["grade"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
