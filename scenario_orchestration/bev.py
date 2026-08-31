"""The BEV raster an object-centric ego policy expects, from its own renderer.

Every released PlanT 2.0 checkpoint is trained with `input_bev=True`, so its
`state` observation has to carry a bird's-eye semantic raster. That raster is the
policy repository's own representation — its class indices, its resolution, its
prebuilt town maps — so it is produced by the policy repository's own renderer
rather than re-drawn here:

    <policy repo>/carla_garage/birds_eye_view/chauffeurnet.py :: ObsManager

The raster turns out to be cheap. Every dynamic-actor mask in that renderer is
commented out, so `bev_semantic_classes` is a function of the prebuilt town
raster and the ego's pose alone — road(1), sidewalk(2), lane markings(3, 4) — and
one `cv.warpAffine` per class per step. It carries no other traffic; the scene is
described to the policy through the object tokens instead.

The raster is handed over raw. The policy's own encoder rotates, centre-crops and
colourises it, exactly as its reference agent does, so doing any of that here
would be doing it twice.

`carla_port` never imports this module: `carla_obs.ObservationBuilder` takes the
BEV source as an injected callable precisely so the port stays free of any one
policy's internals, and the coupling lives here on the contract side.

The renderer is built by `prepare()`, which `ego_driver.PolicyEgoDriver.attach()`
calls once the ego vehicle exists — during the port's `setup()`, before the
simulation loop. A missing town raster or a missing dependency therefore fails
the run where it can still be reported, rather than on the first tick.
Dumping what the policy saw
---------------------------
`$SCENARIO_ORCHESTRATION_BEV_DUMP` names a directory. When set, every raster this
source hands over is written there as `bev_<n>.npy` -- the raw class indices,
before the policy colourises them, because the raw raster is what the checkpoint
was trained against and the colourisation is the policy's own business.

Off by default and free when off. It exists because the raster is otherwise the
one input to a run that leaves no trace: produced per tick and discarded, so a run
whose driving looks wrong cannot afterwards be asked what the policy was looking
at.
"""


from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional


#: Distinguishes "not looked up yet" from "looked up, absent".
_UNSET = object()


class BevError(Exception):
    """The policy's BEV renderer could not be built or stepped."""


#: The renderer's configuration, from the reference agent's own `_init`. These
#: are not free parameters: the checkpoint was trained on this geometry.
MAP_FOLDER = "maps_2ppm_cv"
HISTORY_IDX = (-1,)
SCALE_BBOX = True
SCALE_MASK_COL = 1.0


def prepare_sys_path(repository: str) -> List[str]:
    """`PlanT/` then `carla_garage/`, the order the policy repository needs.

    Both directories ship a `config.py` and a `model.py`, and the modules import
    each other by bare name, so the order matters and is not cosmetic. This
    mirrors what the policy module's own path preparation does, and is
    idempotent with it.
    """
    wanted = [os.path.join(repository, name) for name in ("PlanT", "carla_garage")]
    present = [p for p in wanted if os.path.isdir(p)]
    for path in present:
        while path in sys.path:
            sys.path.remove(path)
    for index, path in enumerate(present):
        sys.path.insert(index, path)
    return present


class CarlaGarageBev:
    """`bev_semantic_classes` from the policy repository's own renderer."""

    def __init__(self, repository: str, map_folder: str = MAP_FOLDER):
        self.repository = str(repository)
        self.map_folder = str(map_folder)
        self.calls = 0
        self._dump = _UNSET
        self._dump_warned = False
        self.town: Optional[str] = None
        self._manager = None
        self._obs_config: Dict[str, Any] = {}
        self._paths = prepare_sys_path(self.repository)

    # ------------------------------------------------------------------ #
    def _global_config(self):
        """`carla_garage.config.GlobalConfig`, which holds the raster geometry."""
        try:
            import config                          # repository-local module
        except ImportError as exc:
            raise BevError(
                f"cannot import the policy repository's config module from "
                f"{self.repository}: {exc}. Expected "
                f"{os.path.join(self.repository, 'carla_garage', 'config.py')}"
            ) from exc
        return config.GlobalConfig()

    def prepare(self, ego_actor) -> "CarlaGarageBev":
        """Build the renderer and bind it to the ego. Idempotent."""
        if self._manager is not None:
            return self
        try:
            from birds_eye_view.chauffeurnet import ObsManager
        except ImportError as exc:
            raise BevError(
                f"cannot import the policy repository's BEV renderer from "
                f"{self.repository}: {exc}. It needs h5py, opencv and numpy in "
                "this interpreter"
            ) from exc

        cfg = self._global_config()
        self._obs_config = {
            "width_in_pixels": cfg.lidar_resolution_width,
            "pixels_ev_to_bottom": cfg.lidar_resolution_height / 2.0,
            "pixels_per_meter": cfg.pixels_per_meter_collection,
            "history_idx": list(HISTORY_IDX),
            "scale_bbox": SCALE_BBOX,
            "scale_mask_col": SCALE_MASK_COL,
            "map_folder": self.map_folder,
        }
        self.town = self._town_of(ego_actor)
        manager = ObsManager(self._obs_config, cfg)
        try:
            # `criteria_stop` feeds only the stop-sign mask, which the renderer
            # no longer draws, so there is nothing to pass.
            manager.attach_ego_vehicle(ego_actor, criteria_stop=None)
        except (FileNotFoundError, OSError) as exc:
            raise BevError(
                f"the policy repository has no prebuilt BEV raster for town "
                f"{self.town!r}: {exc}. {self.map_folder}/ ships the towns its "
                "checkpoints were trained on"
            ) from exc
        except Exception as exc:
            raise BevError(
                f"attaching the BEV renderer to the ego failed: "
                f"{type(exc).__name__}: {exc}") from exc
        self._manager = manager
        return self

    @staticmethod
    def _town_of(ego_actor) -> str:
        try:
            return str(ego_actor.get_world().get_map().name).split("/")[-1]
        except (AttributeError, RuntimeError):        # pragma: no cover
            return "?"

    # ------------------------------------------------------------------ #
    def __call__(self, ego_actor) -> Any:
        """The raw class-index raster for the ego's current pose."""
        if self._manager is None:
            self.prepare(ego_actor)
        try:
            observation = self._manager.get_observation(None)
        except Exception as exc:
            raise BevError(f"the BEV renderer failed: "
                           f"{type(exc).__name__}: {exc}") from exc
        self.calls += 1
        raster = observation["bev_semantic_classes"]
        self._write_raster(raster)
        return raster

    # ------------------------------------------------------------------ #
    def _dump_dir(self):
        """Where to write rasters, or None. Resolved once, then cached."""
        if self._dump is _UNSET:
            path = os.environ.get("SCENARIO_ORCHESTRATION_BEV_DUMP") or None
            if path:
                try:
                    os.makedirs(path, exist_ok=True)
                except OSError:
                    path = None
            self._dump = path
        return self._dump

    def _write_raster(self, raster) -> None:
        """Persist one raster. Never fails the run: this is instrumentation."""
        directory = self._dump_dir()
        if not directory:
            return
        try:
            import numpy as np
            np.save(os.path.join(directory, "bev_%05d.npy" % self.calls),
                    np.asarray(raster, dtype="uint8"))
        except Exception as exc:                       # noqa: BLE001
            if not self._dump_warned:
                self._dump_warned = True
                sys.stderr.write("[bev] could not dump raster: %s\n" % exc)

    def describe(self) -> Dict[str, Any]:
        return {"source": "carla_garage.birds_eye_view.chauffeurnet.ObsManager",
                "repository": self.repository,
                "map_folder": self.map_folder,
                "town": self.town,
                "config": dict(self._obs_config),
                "rasters": self.calls}

    def close(self) -> None:
        manager, self._manager = self._manager, None
        if manager is not None:
            try:
                manager.clean()
            except Exception:                        # pragma: no cover
                pass


def blank_allowed() -> bool:
    """Whether the policy has been told to substitute a blank raster.

    `PLANT2_BLANK_BEV` is the policy's own switch, and its own documentation is
    blunt that a blank raster makes the driving meaningless. It is honoured here
    only to the extent of not failing the run when the renderer is unavailable:
    the observation then carries no `bev` at all and the policy decides, printing
    its own warning. The run report records that it happened.
    """
    return os.environ.get("PLANT2_BLANK_BEV", "") in ("1", "true", "True")


def build(repository: str, map_folder: str = MAP_FOLDER) -> "CarlaGarageBev":
    """The BEV source for a policy checked out at `repository`."""
    return CarlaGarageBev(repository=repository, map_folder=map_folder)
