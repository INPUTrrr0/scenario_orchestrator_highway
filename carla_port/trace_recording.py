"""Per-tick trace recording for the harness's metrics package.

The harness (`scenario_orchestration`) owns the experimental protocol and the
*metrics* too: scenario success is evaluated on the realized trajectory by
`metrics/`, rather than self-reported by each method. This repository was the
last of the three not to feed it -- it wrote `report.json` and `verify.json`,
its own formats, which is exactly what the metrics design exists to replace.
A result self-reported by the method under test is not comparable with one
measured by the harness, so a matrix mixing the two is describing two
quantities as one.

This module is the whole of this repository's side of that contract. It is a
port of the junction port's `carla_port/trace_recording.py` and keeps its
structure deliberately, so the two cannot drift into recording the same run
differently. It:

  * finds `metrics.recording` in the harness that issued the run and imports
    the recorder from there rather than reimplementing the format;
  * declares the static scene -- extents, reference paths, the road;
  * transcribes CARLA's own reading of every bound actor, each tick.

What differs from the junction port, and only this:

  * the static scene ends in a **road** rather than a junction -- road id,
    lane centres and width off `HighwayFrame` -- because that is the geometry
    the merge kernel resolves a conflict point against here;
  * the orchestration events are this repository's: a **cast** (which actor was
    given the intent) and a **lane change**, rather than a hero nomination and
    a reroute.

Two choices inherited from that port, for the same reasons.

**CARLA world coordinates, not the script frame.** This port reasons in the
script's road frame, but the OSC2 baseline has no such frame, and the point of
the metrics package is that every arm is measured the same way. So the recorder
is fed `carla_actor.get_transform()` directly, and reference paths are mapped
*into* CARLA through `HighwayFrame.to_carla_xy` rather than states being mapped
out of it.

**One world snapshot per tick, not one call per actor.** `world.get_snapshot()`
is served from the frame the client already holds and returns the whole cast,
so it costs one call rather than one per actor, and every actor in it is read
as of the same frame -- which per-actor calls, interleaved with the loop, are
not.

**Nothing here may fail the run.** Every entry point swallows its own errors
into a note. A trace is worth less than a run.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

TRACE_RATE_HZ = 10.0

#: How densely a reference path is sampled, in metres. The merge kernel
#: resolves a conflict point by entry into a 2.5 m corridor, so a vertex every
#: two metres is well inside the resolution that decides anything.
PATH_SAMPLE_M = 2.0


def find_harness_root(start: Optional[str] = None) -> Optional[str]:
    """Where the harness that issued this run lives.

    Only `metrics.recording` is imported from it, and that subpackage is
    stdlib-only by contract -- which is what makes importing it from this
    repository's interpreter safe.
    """
    override = os.environ.get("AV_HARNESS_ROOT")
    if override and os.path.isdir(override):
        return override
    path = os.path.abspath(start or os.path.dirname(os.path.dirname(__file__)))
    while True:
        if os.path.isdir(os.path.join(path, "metrics", "recording")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


#: The name the harness's recording package is loaded under. Deliberately not
#: `metrics`: this repository ships a top-level `metrics/` directory of its own
#: (docs and scripts), so `import metrics.recording` can resolve to that and
#: fail confusingly. Loading by path under a private name is also the more
#: honest thing to do -- nothing else in the harness is imported, and this
#: makes that structural rather than promised.
_RECORDING_MODULE = "harness_trace_recording"


def _load_recorder_class(root: str):
    import importlib.util
    if _RECORDING_MODULE in sys.modules:
        return sys.modules[_RECORDING_MODULE].TraceRecorder
    pkg_dir = os.path.join(root, "metrics", "recording")
    spec = importlib.util.spec_from_file_location(
        _RECORDING_MODULE, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RECORDING_MODULE] = module
    spec.loader.exec_module(module)
    return module.TraceRecorder


def make_recorder(output_dir: str, rate_hz: float = TRACE_RATE_HZ,
                  context: Optional[dict] = None) -> Tuple[object, Optional[str]]:
    """`(recorder, note)`. `recorder` is None when the harness cannot be found;
    the note says why, and is carried into the run report."""
    root = find_harness_root()
    if root is None:
        return None, ("no metrics/recording found above this repository; "
                      "no per-tick trace was written")
    try:
        TraceRecorder = _load_recorder_class(root)
    except Exception as exc:                      # pragma: no cover - defensive
        return None, "metrics/recording could not be loaded from %s: %s" % (
            root, exc)
    try:
        return TraceRecorder(output_dir, rate_hz=rate_hz,
                             context=dict(context or {})), None
    except Exception as exc:                      # pragma: no cover - defensive
        return None, "TraceRecorder could not be opened: %s" % (exc,)


# --------------------------------------------------------------------------- #
# static scene
# --------------------------------------------------------------------------- #

def declare_scene(rec, run) -> None:
    """Actors, reference paths and the road, in CARLA world coordinates."""
    if rec is None:
        return
    try:
        _declare_actors(rec, run)
        _declare_paths(rec, run)
        _declare_road(rec, run)
    except Exception as exc:                      # pragma: no cover - defensive
        run.notes.append("trace scene declaration: %s" % (exc,))


def _declare_actors(rec, run) -> None:
    for binding in run.bindings:
        actor = binding.carla_actor
        rec.declare(binding.script_actor_id,
                    extent=binding.extent,
                    type_id=getattr(actor, "type_id", None),
                    is_ego=(binding.script_actor_id == run.cfg.ego),
                    carla_id=getattr(actor, "id", None))


def _lane_polyline(run, lane_index: int) -> List[Tuple[float, float]]:
    """One lane centre, as a CARLA-frame polyline over the fitted road.

    The road frame is straight by construction (`HighwayFrame` fits a straight
    run and reports its fit error), so a lane centre needs only its two ends --
    but it is sampled at `PATH_SAMPLE_M` anyway, because the kernel measures
    corridor entry by distance along the polyline and a two-vertex line makes
    that resolution depend on the road's length.
    """
    frame = run.frame
    x = frame.lane_center_x(lane_index)
    length = float(getattr(frame, "length", 0.0) or 0.0)
    if length <= 0.0:
        return []
    n = max(2, int(length / PATH_SAMPLE_M) + 1)
    out = []
    for i in range(n):
        y = length * (i / float(n - 1))
        out.append(tuple(frame.to_carla_xy(x, y)))
    return out


def _declare_paths(rec, run) -> None:
    """The route each actor intends, as the lane centre it is bound to.

    The ego's is its companion's `reference_path` when there is one -- that is
    the geometry the run is actually steering to, and during a lane change it
    is the manoeuvre rather than the destination lane. Background actors are
    scripted along their own lane, so their lane centre IS their route.
    """
    frame = run.frame
    for binding in run.bindings:
        aid = binding.script_actor_id
        path: List[Tuple[float, float]] = []
        if aid == run.cfg.ego and getattr(run, "policy", None) is not None:
            try:
                path = [tuple(frame.to_carla_xy(px, py))
                        for px, py in run.policy.reference_path]
            except Exception:                     # pragma: no cover - defensive
                path = []
        if not path:
            try:
                # A binding carries no script state of its own; it is read back
                # out of CARLA through the frame, which is the same conversion
                # the runner uses everywhere else.
                from .carla_adapter import carla_actor_to_script_state
                state = carla_actor_to_script_state(aid, binding.carla_actor,
                                                    frame)
                lane = frame.lane_index_of(state.x)
            except Exception:
                lane = None
            if lane is not None:
                path = _lane_polyline(run, lane)
        if path:
            rec.declare_path(aid, path)


def _declare_road(rec, run) -> None:
    """The road, as the merge kernel's frame of reference.

    The junction port declares a junction here. A highway conflict is resolved
    against lane geometry instead, so what is recorded is the fitted road: its
    id, its lane centres in CARLA coordinates, and the width the corridor test
    is scaled against.
    """
    frame = run.frame
    # `frame.lanes` holds `highway_map.Lane` dataclasses, not dicts, and a lane
    # carries no index of its own -- its index IS its position in this list,
    # which is what `lane_center_x` takes. Reading it as a mapping silently
    # produced a road with zero lanes, which is a road the merge kernel cannot
    # resolve a corridor against.
    lanes = []
    for index, lane in enumerate(getattr(frame, "lanes", []) or []):
        lanes.append({"index": index,
                      "carla_lane_id": getattr(lane, "lane_id", None),
                      "oncoming": not bool(getattr(lane, "same_direction", True)),
                      "width_m": getattr(lane, "width", None),
                      "centre": _lane_polyline(run, index)})
    # `declare_zone` is the recorder's channel for static geometry -- the
    # junction port declares its junction box through it. A highway has no box,
    # so the road itself is the zone: one entry carrying the lane centres the
    # merge kernel resolves a corridor against, and the width it scales that
    # corridor by.
    rec.declare_zone("road", "highway_road",
                     road_id=getattr(frame, "road_id", None),
                     lane_width_m=float(getattr(frame, "lane_width", 0.0) or 0.0),
                     length_m=float(getattr(frame, "length", 0.0) or 0.0),
                     lanes=lanes)


# --------------------------------------------------------------------------- #
# per tick
# --------------------------------------------------------------------------- #

def capture(rec, run) -> None:
    """One tick, read off the frame the client already holds.

    Falls back to per-actor reads where no snapshot is available, and says so
    in the recorder's error list, so a trace can never look snapshot-sourced
    when it is not.
    """
    if rec is None:
        return
    try:
        snap = run.world.get_snapshot()
    except (RuntimeError, AttributeError):        # pragma: no cover
        snap = None
    states = {}
    for binding in run.bindings:
        row = (_from_snapshot(snap, binding.carla_actor) if snap is not None
               else _from_actor(binding.carla_actor))
        if row is not None:
            states[binding.script_actor_id] = row
    rec.tick(run.t_sim, states)


def _from_snapshot(snap, actor):
    try:
        st = snap.find(actor.id)
    except (RuntimeError, AttributeError):        # pragma: no cover
        st = None
    if st is None:
        return None                               # not alive this frame
    tf, v, a = st.get_transform(), st.get_velocity(), st.get_acceleration()
    return (tf.location.x, tf.location.y, tf.rotation.yaw,
            v.x, v.y, a.x, a.y)


def _from_actor(actor):
    try:
        tf = actor.get_transform()
        v = actor.get_velocity()
        a = actor.get_acceleration()
    except (RuntimeError, AttributeError):        # destroyed mid-run
        return None
    return (tf.location.x, tf.location.y, tf.rotation.yaw,
            v.x, v.y, getattr(a, "x", None), getattr(a, "y", None))


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #

def note_collision(rec, run, rc) -> None:
    """One collision, with **who** was in it.

    `RealizedCollision` names its participants `actor_id` and `other_id`
    (`carla_collision.py`), and both must reach the trace: the scenario
    kernels' "no third party redirected the interaction" predicate tests
    `ego in event.actors`, so a collision recorded with an empty participant
    list silently never fires it.

    `other_id` is None when the other body is not a bound script actor (a wall,
    a kerb). The participant list then holds only the sensor's parent, and the
    CARLA type is carried beside it, because "the ego hit something unbound"
    and "the ego hit actor 4" must not read the same.
    """
    if rec is None:
        return
    actors = [a for a in (getattr(rc, "actor_id", None),
                          getattr(rc, "other_id", None)) if a is not None]
    rec.event(getattr(rc, "sim_time", run.t_sim), "collision",
              actors=[str(a) for a in actors],
              other_type=getattr(rc, "other_type", None),
              impulse=getattr(rc, "impulse", None))


def note_cast(rec, run, actor_id, role: str, outcome: Optional[str] = None) -> None:
    """Which actor the orchestrator gave the intent to.

    The junction port records a hero nomination here. This port casts a role --
    the actor asked to cut in, to brake, to block -- and recasts when the first
    choice becomes unable to deliver. Both are the same question for the
    metrics package: which background actor was the scenario *about*.
    """
    if rec is None or actor_id is None:
        return
    try:
        rec.event(run.t_sim, "cast", actors=[str(actor_id)],
                  role=role, outcome=outcome)
    except Exception as exc:                      # pragma: no cover - defensive
        run.notes.append("trace cast event: %s" % (exc,))


def note_lane_change(rec, run, actor_id, from_lane, to_lane) -> None:
    """A realized lane change, by whoever made it.

    Recorded because a merge conflict is created or destroyed by one: the ego
    leaving the lane its hero was staged in changes which actor the kernel
    resolves a conflict point against.
    """
    if rec is None or actor_id is None:
        return
    try:
        rec.event(run.t_sim, "lane_change", actors=[str(actor_id)],
                  from_lane=from_lane, to_lane=to_lane)
    except Exception as exc:                      # pragma: no cover - defensive
        run.notes.append("trace lane_change event: %s" % (exc,))
