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
import queue
from dataclasses import dataclass, field
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
    points_per_second: int = 600000
    rotation_frequency: float = 20.0        # = 1 / leaderboard agent period
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
        if isinstance(item, (CameraSpec, LidarSpec)):
            out.append(item)
        elif isinstance(item, dict):
            cls = LidarSpec if "lidar" in str(item.get("kind", "")) else CameraSpec
            known = set(cls.__dataclass_fields__)
            out.append(cls(**{k: v for k, v in item.items() if k in known}))
        else:
            raise TypeError(
                f"a policy's sensors() must yield CameraSpec, LidarSpec or "
                f"dict, got {type(item).__name__}")
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

    # ------------------------------------------------------------------ #
    def spawn(self) -> "CameraRig":
        """Attach every camera. A camera that cannot be created is recorded in
        `failed` rather than raised: the offline test double has no sensor
        blueprints, and `validate.py` must keep running without a server."""
        library = self.world.get_blueprint_library()
        for spec in self.specs:
            try:
                bp = library.find(spec.kind)
                if isinstance(spec, LidarSpec):
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
            array = (self._to_points(image) if isinstance(spec, LidarSpec)
                     else self._to_rgb(image))
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
                "attached": [s.name for s, _sn, _q in self.attached],
                "failed": list(self.failed)}
