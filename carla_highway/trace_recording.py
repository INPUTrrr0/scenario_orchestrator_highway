"""Per-tick trace recording for the harness's metrics package.

The sibling module in `third_party/orchestration/carla_port/trace_recording.py`,
with the two pieces that are a property of the *world* replaced: the reference
paths, and the zone. Everything else -- finding the harness, loading its
recorder by path under a private name, one world snapshot per tick, and never
failing the run -- is the same and is deliberately unedited.

Why a highway trace needs saying differently
--------------------------------------------
The intersection port declares a junction zone and a route per actor, because
its scenario is a junction and its actors are routed. Here:

* **The ego's reference path is the lane it starts in, straight down the road.**
  Not the trajectory it drove: a lane change is exactly what the `lane_change`
  and `overtake` families are about, and a route recovered from the driven
  trajectory would fold the manoeuvre into the reference and leave nothing to
  measure the manoeuvre against. It is declared as `source="route"` with the
  home lane's centre line.
* **The background actors' paths come from the maneuver-script oracle**, sampled
  forward over the horizon, exactly as the intersection port does. That matters
  most for the cut-in: the actor's path *does* leave its lane, and the metric's
  merge kernel finds the conflict point by asking where that path enters the
  ego's corridor. Sampling the script rather than the lane is what makes that
  point exist.
* **The zone is the road, not a box.** A highway conflict has no junction, so
  the declared zone is the fitted straight's own extent, recorded so a reader
  can tell which stretch of which road the episode happened on.

`DETAILS.md`'s assumption -- that paths are predefined and the orchestrator only
retimes along them -- is *false* for a cut-in by construction, and this is where
that shows: the holder's path is resolved after each replan, so it is sampled at
setup from the script as it then stands and recorded as `source="script"`. A
reader can hold the assumption up against the trace instead of taking it on
faith, which is the point of recording the source at all.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from .highway_map import FORWARD_HEADING

TRACE_RATE_HZ = 10.0

#: How densely the scripted paths are sampled. 0.2 s at highway speeds is about
#: 2.5 m per vertex, well under the metric's 2.5 m corridor width.
PATH_SAMPLE_DT = 0.2

#: How far along its lane the ego's reference path is drawn, each way from the
#: anchor. The fitted straight is 120-240 m, and a path shorter than the run
#: would truncate the ego's station and with it the approach window.
EGO_PATH_MARGIN_M = 40.0


def find_harness_root(start: Optional[str] = None) -> Optional[str]:
    """Where the harness that issued this run lives.

    Only `metrics.recording` is imported from it, and that subpackage is
    stdlib-only by contract -- which is what makes importing it from this
    repository's interpreter safe.
    """
    for var in ("ORCHESTRATOR_HIGHWAY_HARNESS_ROOT",
                "ORCHESTRATION_HARNESS_ROOT"):
        override = os.environ.get(var)
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
#: `metrics`: this repository puts a top-level `metrics` module on sys.path
#: (`scenario_orchestration/metrics.py`, the canonical-vocabulary projection),
#: so `import metrics.recording` resolves to that file and fails with
#: "'metrics' is not a package". Loading by path under a private name also makes
#: it structural rather than promised that nothing else in the harness is
#: imported.
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
    """Actors, reference paths and the road, in CARLA world coordinates.

    Each of the three is attempted independently. One `try` around all three
    meant the first failure silently dropped the other two: a typo in the lane
    accessor cost the run its reference paths *and* its zone, and the trace came
    out with 160 ticks and an empty scene -- which reads as a recorded run and is
    unevaluable. A trace is worth less than a run, so nothing here raises; but a
    missing piece has to be named, and the reference paths are named loudly,
    because without them the metric has no station along the route and no
    predicate is defined at all.
    """
    if rec is None:
        return
    for name, fn, consequence in (
            ("actors", _declare_actors,
             "extents and roles are missing, so clearances cannot be computed"),
            ("reference paths", _declare_paths,
             "THE TRACE CANNOT BE EVALUATED: with no route there is no station "
             "along it, so the approach window and every predicate are undefined"),
            ("road zone", _declare_road,
             "the episode's road is not recorded; the metric does not read it, "
             "so the verdict is unaffected")):
        try:
            fn(rec, run)
        except Exception as exc:                  # pragma: no cover - defensive
            run.notes.append(
                "trace scene: %s could not be declared (%s: %s) -- %s"
                % (name, type(exc).__name__, exc, consequence))


def _declare_actors(rec, run) -> None:
    for binding in run.bindings:
        actor = binding.carla_actor
        aid = binding.script_actor_id
        rec.declare(aid, extent=binding.extent,
                    type_id=getattr(actor, "type_id", None),
                    is_ego=(aid == run.cfg.ego),
                    role=(run.orchestration_roles().get(aid)),
                    carla_id=getattr(actor, "id", None))


def _declare_paths(rec, run) -> None:
    frame = run.frame
    ego_id = str(run.cfg.ego)

    # The ego: its home lane, straight, drawn past both ends of the run. The
    # lane it *started* in, not the lane it ended in -- see the module note.
    if run.policy is not None:
        x = frame.lane_center_x(run.policy.home_lane)
        y0 = -0.5 * frame.length - EGO_PATH_MARGIN_M
        y1 = 0.5 * frame.length + EGO_PATH_MARGIN_M
        # Ordered along the direction traffic in that lane actually travels, so
        # the ego's station increases as it drives. A path laid out the other
        # way makes the approach window empty and every predicate undefined.
        heading = frame.heading_of_lane(run.policy.home_lane)
        if abs(((heading - FORWARD_HEADING + 180.0) % 360.0) - 180.0) > 90.0:
            y0, y1 = y1, y0            # the lane runs the other way
        n = max(2, int(abs(y1 - y0) / 5.0) + 1)
        route = [frame.to_carla_xy(x, y0 + (y1 - y0) * k / (n - 1.0))
                 for k in range(n)]
        rec.declare_path(ego_id, route, source="route")

    # The background: sampled from the maneuver script, as it stands now.
    from carla_port.carla_sync import world_states
    horizon = float((run.cfg.duration or 0.0) + run.cfg.linger)
    n = max(2, int(horizon / PATH_SAMPLE_DT) + 1)
    tracks: Dict[str, List[Tuple[float, float]]] = {}
    for k in range(n):
        states = world_states(run.loop.sc, k * PATH_SAMPLE_DT)
        for aid, st in states.items():
            if aid == ego_id and run.policy is not None:
                continue
            pt = frame.to_carla_xy(st.x, st.y)
            track = tracks.setdefault(aid, [])
            if not track or (abs(track[-1][0] - pt[0])
                             + abs(track[-1][1] - pt[1])) > 0.05:
                track.append(pt)
    for aid, track in tracks.items():
        if len(track) >= 2:
            rec.declare_path(aid, track, source="script")


def _declare_road(rec, run) -> None:
    """The fitted straight, as the zone. A highway has no junction box.

    Recorded as a disc about the road's own anchor with a radius that covers
    the fitted stretch, plus the numbers that say which road it was. The metric
    does not read this -- both kernels derive their conflict point from the
    reference paths -- so it is documentation of where the episode happened,
    which is what makes a trace re-readable a year later.
    """
    frame = run.frame
    anchor = frame.anchor
    rec.declare_zone("road", "highway_straight",
                     center=(anchor.x, anchor.y),
                     radius=float(0.5 * frame.length),
                     lane_width_m=float(frame.lane_width),
                     num_lanes=int(frame.num_lanes),
                     length_m=float(frame.length),
                     road_id=getattr(frame, "road_id", None),
                     section_id=getattr(frame, "section_id", None),
                     theta_deg=round(math.degrees(float(frame.theta)), 3))


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


def note_collision(rec, run, rc) -> None:
    if rec is None:
        return
    actors = [a for a in (getattr(rc, "actor", None), getattr(rc, "other", None))
              if a is not None]
    rec.event(getattr(rc, "sim_time", run.t_sim), "collision", actors=actors,
              other_type=getattr(rc, "other_type", None),
              impulse=getattr(rc, "impulse", None))


def note_hero(rec, run) -> None:
    """Who the orchestrator currently casts as the conflict actor.

    The metrics package resolves the hero from geometry, not from a method's
    declaration -- that is what keeps the verdict method-independent. Recording
    the declaration anyway is what lets the two be *compared*, and here it is
    load-bearing rather than decorative: the cut-in orchestrator **recasts**
    mid-episode when its holder becomes geometrically hopeless, so a single
    terminal value would not describe the run.
    """
    if rec is None:
        return
    hero = run.loop.holder if run.loop is not None else None
    if hero == getattr(run, "_last_nominated_hero", "<unset>"):
        return
    run._last_nominated_hero = hero
    rec.event(run.t_sim, "hero_nomination",
              actors=([str(hero)] if hero is not None else []))


def note_events(rec, run, seen: int) -> int:
    """Forward the orchestrator's own event log, from `seen` onwards.

    Returns the new count. The loop's events are cast decisions, recasts,
    commits and collision-yield replans -- the interventions, in this port's
    vocabulary -- and they are what an `intervention` event in the canonical
    trace means.
    """
    if rec is None or run.loop is None:
        return seen
    events = list(getattr(run.loop, "events", ()))
    for e in events[seen:]:
        # `event_kind`, not `kind`: the recorder's own signature is
        # `event(t, kind, actors=(), **payload)`, so a payload key called `kind`
        # collides with the positional argument.
        rec.event(float(getattr(e, "t", run.t_sim)), "intervention",
                  actors=([str(e.actor)] if getattr(e, "actor", None) else []),
                  event_kind=getattr(e, "kind", None),
                  text=getattr(e, "text", None))
    return len(events)
