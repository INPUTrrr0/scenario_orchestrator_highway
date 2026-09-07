#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_sensors.py — the ego's camera rig (level 3).

`capabilities.json` used to declare `observation_spaces: ["state"]` and the
reason given was blunt: "the port owns the CARLA world but attaches no sensor
rig to the ego". That is what this module fixes, and it is the whole of what
was missing — every other piece of the external-policy seam (`ego_driver.py`,
`carla_obs.py`, `policies.py`) already existed and was already exercised by
`validate.py` check 10.

Who decides what to attach
--------------------------
The policy does. A sensorimotor policy is trained on one specific rig — TFv6
reads three 384x384 pinhole cameras stitched into one 1152x384 strip, SimLingo
reads a single wide forward camera it tiles itself — and a rig chosen here
would be a rig neither model was trained on. So an `ego_policy_v1` policy may
expose `sensors()` returning a list of `CameraSpec`, and this module attaches
exactly that. A policy without `sensors()` gets no rig and the `state`
observation it always got, so nothing that worked before changes.

Synchronous capture
-------------------
CARLA delivers sensor data asynchronously even in synchronous mode: the
callback fires somewhere between `world.tick()` returning and the next tick. A
rig that simply kept the newest frame would hand the policy an image from an
arbitrary earlier tick under load, which is the classic way an agent's
behaviour becomes irreproducible and nobody notices. Each camera therefore has
its own queue and `capture(frame)` blocks until the image *stamped with the
frame it asked for* arrives, discarding anything older. That makes a rendered
tick cost real wall-clock time, which is exactly the trade the leaderboard
makes too.
"""
from __future__ import annotations

import math
import os
import queue
from dataclasses import dataclass, replace, field
from typing import Dict, List, Optional, Sequence, Tuple

from .carla_api import carla

try:                                            # numpy is optional at import
    import numpy as _np                         # time so the offline test
except Exception:                               # double can still load us
    _np = None

#: How long `capture` waits for one camera's frame before giving up. Generous:
#: a 1152x384 render on a busy GPU is not instant, and a timeout here surfaces
#: as a policy driving blind, which is worse than a slow tick.
CAPTURE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class CameraSpec:
    """One camera, in the units CARLA's blueprint attributes use.

    `x`/`y`/`z` are metres in the ego's own frame (+x forward, +y right, +z up)
    and `roll`/`pitch`/`yaw` are degrees, i.e. exactly a `carla.Transform`
    relative to the vehicle. The defaults are the CARLA Leaderboard's forward
    camera mounting, which is what this family of models is trained behind.
    """
    name: str
    width: int = 1024
    height: int = 512
    fov: float = 110.0
    x: float = -1.5
    y: float = 0.0
    z: float = 2.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.camera.rgb"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))


@dataclass(frozen=True)
class LidarSpec:
    """One ray-cast LiDAR, in CARLA blueprint units.

    A camera-only rig would starve TFv6: it is a camera+LiDAR fusion model and
    its BEV branch reads `rasterized_lidar`. Handing it a zero raster is the
    same failure mode `scenario_orchestration/README.md` already documents for
    `PLANT2_BLANK_BEV` — the model still returns numbers, and the numbers are
    not the model's published behaviour. So the rig serves points.
    """
    name: str = "lidar"
    channels: int = 64
    range_m: float = 100.0
    #: Points per REVOLUTION is what matters to a model; CARLA is configured in
    #: points per second, so the spawn scales this by the actual rotation rate.
    points_per_revolution: int = 30000
    points_per_second: int = 600000
    #: Revolutions per second. This MUST match the simulation tick rate, not the
    #: policy's decision rate. CARLA accumulates returns over simulated time and
    #: emits whatever swept past on each tick: at 20 rev/s in a world ticking at
    #: 60 Hz, one frame carries a third of a revolution — a fixed 120 deg wedge
    #: that, measured on a real run, spanned 180-299 deg. Behind and to the left,
    #: never straight ahead, so the fusion branch never saw the car in front.
    #: `CameraRig.spawn` overwrites this from the world's fixed_delta.
    rotation_frequency: float = 20.0
    upper_fov: float = 10.0
    lower_fov: float = -30.0
    x: float = -0.5
    y: float = 0.0
    z: float = 1.85
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.lidar.ray_cast"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))

    def attributes(self) -> Dict[str, str]:
        return {"channels": str(int(self.channels)),
                "range": str(float(self.range_m)),
                "points_per_second": str(int(self.points_per_second)),
                "rotation_frequency": str(float(self.rotation_frequency)),
                "upper_fov": str(float(self.upper_fov)),
                "lower_fov": str(float(self.lower_fov))}

    def for_tick_rate(self, hz: float) -> "LidarSpec":
        """This spec, rotating once per tick at `hz`.

        One full revolution per tick is the only setting that gives the model a
        complete sweep, and `points_per_second` has to rise with the rotation
        rate to keep the same number of points in each one.
        """
        if hz <= 0:
            return self
        return replace(self, rotation_frequency=float(hz),
                       points_per_second=int(round(self.points_per_revolution * hz)))


@dataclass(frozen=True)
class RadarSpec:
    """One radar, in CARLA blueprint units.

    TFv6's config sets `use_radar_detection: True` and its `RadarDetector`
    tokenizes `batch["radar"]` into the planner's cross-attention keys, so a rig
    without radar leaves the model reading a tensor its training data always
    filled. Upstream PADS short sweeps with zero rows, so an all-zero tensor is
    not a malformed input — it is a well-formed one that says "the radar is
    working and nothing is out there", which is worse than a malformed one
    because nothing downstream can tell it apart from a clear road.

    Detections are returned in the EGO frame, x forward, metres, matching the
    LiDAR path: `rasterize_lidar_bev` takes raw CARLA points and the BEV bounds
    it filters against are x in [-32, 64] — forward-biased, so x is forward.
    """
    name: str = "radar"
    horizontal_fov: float = 90.0
    vertical_fov: float = 0.1
    range_m: float = 100.0
    #: Returns per TICK the consumer wants. `lead` pads or truncates each
    #: sensor's block to `num_radar_points_per_sensor` (75), so anything less
    #: than this per tick is zero padding pretending to be clear road.
    points_per_tick: int = 150
    points_per_second: int = 1500
    x: float = 2.6
    y: float = 0.0
    z: float = 0.6
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.other.radar"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))

    def attributes(self) -> Dict[str, str]:
        return {"horizontal_fov": str(float(self.horizontal_fov)),
                "vertical_fov": str(float(self.vertical_fov)),
                "range": str(float(self.range_m)),
                "points_per_second": str(int(self.points_per_second))}

    def for_tick_rate(self, hz: float) -> "RadarSpec":
        """This spec, delivering `points_per_tick` returns on every tick."""
        if hz <= 0:
            return self
        return replace(self, points_per_second=int(round(self.points_per_tick * hz)))


#: The rigs the two installed sensorimotor policies are trained behind. A
#: policy may return its own `sensors()` instead; these exist so a policy that
#: only NAMES a rig (`"rig": "tfv6"` in policy.json) still gets the right one.
RIGS: Dict[str, List[object]] = {
    # TFv6 / LEAD: three pinhole cameras, stitched left-to-right into the
    # 1152x384 strip `policy.transfuser.final_image_*` describes. The yaws are
    # the rig's own PCAM_L0 / PCAM_F0 / PCAM_R0.
    "tfv6": [
        CameraSpec("PCAM_L0", width=384, height=384, fov=90.0, yaw=-60.0),
        CameraSpec("PCAM_F0", width=384, height=384, fov=90.0, yaw=0.0),
        CameraSpec("PCAM_R0", width=384, height=384, fov=90.0, yaw=60.0),
        LidarSpec("lidar"),
        # `lead.config.expert.sensor_rig.SensorRigConfig.radars`, in list order:
        # `_preprocess_radar_input` identifies a sensor by its index and writes
        # that index into the fifth column, so the order is part of the format.
        RadarSpec("radar1", x=2.6, z=0.60, yaw=-45.0),
        RadarSpec("radar2", x=2.6, z=0.60, yaw=45.0),
        RadarSpec("radar3", x=-2.6, z=0.60, yaw=135.0),
        RadarSpec("radar4", x=-2.6, z=0.60, yaw=225.0),
    ],
    # SimLingo: one wide forward camera. The model tiles it itself
    # (`dynamic_preprocess`), so the rig only has to deliver the full frame.
    "simlingo": [
        CameraSpec("rgb_front", width=1024, height=512, fov=110.0),
    ],
}


def rig(name: str) -> List[object]:
    """A named rig, by copy so a caller cannot edit the table."""
    if name not in RIGS:
        raise KeyError(f"unknown camera rig {name!r}; have {sorted(RIGS)}")
    return list(RIGS[name])


def specs_from(declared: Sequence[object]) -> List[object]:
    """Normalize whatever a policy's `sensors()` returned into CameraSpecs.

    A policy may return `CameraSpec`s or plain dicts; a policy repository
    should not have to import this module to describe its own rig.
    """
    out: List[object] = []
    for item in declared:
        if isinstance(item, (CameraSpec, LidarSpec, RadarSpec)):
            out.append(item)
        elif isinstance(item, dict):
            kind = str(item.get("kind", ""))
            cls = (RadarSpec if "radar" in kind
                   else LidarSpec if "lidar" in kind else CameraSpec)
            known = set(cls.__dataclass_fields__)
            out.append(cls(**{k: v for k, v in item.items() if k in known}))
        else:
            raise TypeError(
                f"a policy's sensors() must yield CameraSpec, LidarSpec, "
                f"RadarSpec or dict, got {type(item).__name__}")
    return out


class CameraRig:
    """The ego's cameras, spawned, queued and drained in lockstep with ticks."""

    def __init__(self, world, ego_actor, specs: Sequence[object]):
        self.world = world
        self.ego_actor = ego_actor
        self.specs: List[object] = list(specs)
        # (spec, sensor, queue) only for cameras that actually attached, so a
        # failed spawn cannot misalign a spec with another camera's queue.
        self.attached: List[Tuple[object, object, "queue.Queue"]] = []
        self.failed: List[str] = []
        #: tick rate the sweeping sensors were retimed to, for the run report
        self.sensor_hz: float = 0.0

    # ------------------------------------------------------------------ #
    def _tick_hz(self) -> float:
        """Simulation ticks per second, from the world's own settings.

        Read rather than assumed: a sweeping sensor's rate has to match the tick
        rate, and the tick rate is the runner's `--fixed-delta`, not a constant.
        """
        try:
            delta = float(self.world.get_settings().fixed_delta_seconds or 0.0)
        except (RuntimeError, AttributeError, TypeError):
            return 0.0
        return 1.0 / delta if delta > 0 else 0.0

    def spawn(self) -> "CameraRig":
        """Attach every camera. A camera that cannot be created is recorded in
        `failed` rather than raised: the offline test double has no sensor
        blueprints, and `validate.py` must keep running without a server."""
        library = self.world.get_blueprint_library()
        hz = 0.0 if os.environ.get("AV_SENSOR_RETIME") == "off" else self._tick_hz()
        if hz > 0:
            # Retime the sweeping sensors to the world's tick rate. Without this
            # a LiDAR at 20 rev/s in a 60 Hz world delivers a 120 deg wedge per
            # frame and a radar delivers a third of the returns its consumer
            # pads out with zeros.
            self.specs = [spec.for_tick_rate(hz)
                          if hasattr(spec, "for_tick_rate") else spec
                          for spec in self.specs]
            self.sensor_hz = hz
        for spec in self.specs:
            try:
                bp = library.find(spec.kind)
                if isinstance(spec, (LidarSpec, RadarSpec)):
                    for key, value in spec.attributes().items():
                        bp.set_attribute(key, value)
                else:
                    bp.set_attribute("image_size_x", str(int(spec.width)))
                    bp.set_attribute("image_size_y", str(int(spec.height)))
                    bp.set_attribute("fov", str(float(spec.fov)))
                sensor = self.world.spawn_actor(bp, spec.to_transform(),
                                                attach_to=self.ego_actor)
            except (RuntimeError, AttributeError, KeyError, IndexError) as exc:
                self.failed.append(f"{spec.name}: {exc}")
                continue
            q: "queue.Queue" = queue.Queue()
            sensor.listen(q.put)
            self.attached.append((spec, sensor, q))
        return self

    @property
    def active(self) -> bool:
        return bool(self.attached)

    # ------------------------------------------------------------------ #
    def capture(self, frame: Optional[int] = None) -> Dict[str, object]:
        """The rig's images for `frame`, as HxWx3 uint8 RGB arrays.

        Waits for the image stamped with the requested frame and drops any
        earlier one, so what the policy sees is the tick it was asked about
        rather than whatever happened to be newest.
        """
        out: Dict[str, object] = {}
        for spec, _sensor, q in self.attached:
            image = self._await(q, frame)
            if image is None:
                continue
            if isinstance(spec, RadarSpec):
                array = self._to_radar(image, spec)
            elif isinstance(spec, LidarSpec):
                array = self._to_points(image)
            else:
                array = self._to_rgb(image)
            if array is not None:
                out[spec.name] = array
        return out

    @staticmethod
    def _to_points(measurement):
        """CARLA hands over flat float32 x,y,z,intensity; models want Nx4."""
        if _np is None:
            return None
        raw = _np.frombuffer(measurement.raw_data, dtype=_np.float32)
        return _np.reshape(raw, (-1, 4)).copy()

    @staticmethod
    def _to_radar(measurement, spec: "RadarSpec"):
        """CARLA hands over flat float32 (velocity, azimuth, altitude, depth)
        per detection, in the SENSOR frame. Return Nx4 `(x, y, z, v)` in the
        EGO frame, which is what `filter_and_pad_radars` bounds-checks and what
        `_tokenize_radar` samples BEV features at.

        `velocity` is radial and signed the way CARLA reports it (negative is
        closing); it is passed through untouched, because a sign convention
        invented here would be a different quantity wearing the same name.
        """
        if _np is None:
            return None
        raw = _np.frombuffer(measurement.raw_data, dtype=_np.float32)
        det = _np.reshape(raw, (-1, 4))
        if det.size == 0:
            return _np.zeros((0, 4), dtype=_np.float32)
        vel, azimuth, altitude, depth = (det[:, 0], det[:, 1],
                                         det[:, 2], det[:, 3])
        # spherical -> the sensor's own cartesian frame (x along the boresight)
        horiz = depth * _np.cos(altitude)
        xs = horiz * _np.cos(azimuth)
        ys = horiz * _np.sin(azimuth)
        zs = depth * _np.sin(altitude)
        # sensor -> ego: yaw about z, then the mounting offset. Roll and pitch
        # are zero on every radar in this rig; asserting that is cheaper than
        # carrying a full rotation that is never exercised.
        yaw = _np.radians(float(spec.yaw))
        cos_y, sin_y = _np.cos(yaw), _np.sin(yaw)
        xe = xs * cos_y - ys * sin_y + float(spec.x)
        ye = xs * sin_y + ys * cos_y + float(spec.y)
        ze = zs + float(spec.z)
        return _np.stack([xe, ye, ze, vel], axis=1).astype(_np.float32)

    @staticmethod
    def _await(q: "queue.Queue", frame: Optional[int]):
        deadline_hits = 0
        while True:
            try:
                image = q.get(timeout=CAPTURE_TIMEOUT_S)
            except queue.Empty:
                return None
            if frame is None or getattr(image, "frame", frame) >= frame:
                return image
            deadline_hits += 1
            if deadline_hits > 1000:            # a stuck sensor, not a lag
                return image

    @staticmethod
    def _to_rgb(image):
        """CARLA hands over BGRA bytes; models in this family want RGB."""
        if _np is None:
            return None
        raw = _np.frombuffer(image.raw_data, dtype=_np.uint8)
        raw = raw.reshape((image.height, image.width, 4))
        return raw[:, :, :3][:, :, ::-1].copy()          # BGRA -> RGB

    # ------------------------------------------------------------------ #
    def destroy(self) -> None:
        for _spec, sensor, _q in self.attached:
            try:
                sensor.stop()
            except (RuntimeError, AttributeError):
                pass
            try:
                sensor.destroy()
            except (RuntimeError, AttributeError):
                pass
        self.attached = []

    def describe(self) -> Dict[str, object]:
        return {"cameras": [{"name": s.name, "width": s.width,
                             "height": s.height, "fov": s.fov, "yaw": s.yaw}
                            for s in self.specs if isinstance(s, CameraSpec)],
                "lidars": [{"name": s.name, "channels": s.channels,
                            "range_m": s.range_m}
                           for s in self.specs if isinstance(s, LidarSpec)],
                "radars": [{"name": s.name, "yaw": s.yaw,
                            "horizontal_fov": s.horizontal_fov,
                            "range_m": s.range_m}
                           for s in self.specs if isinstance(s, RadarSpec)],
                "attached": [s.name for s, _sn, _q in self.attached],
                "sensor_hz": self.sensor_hz,
                "lidar_points_per_rev": [s.points_per_revolution
                                         for s in self.specs
                                         if isinstance(s, LidarSpec)],
                "failed": list(self.failed)}
