#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/scenarios.py — the three highway scenarios, refitted to CARLA.

The scenarios are NOT rewritten here. They are the authored YAML files that
ship with the highway orchestrator:

    scenarios/scenario_cutin.yaml
    scenarios/scenario_hard_brake.yaml
    scenarios/scenario_overtake.yaml

loaded with `se.load_scenario` and then *retargeted* onto the road that
`HighwayFrame.discover` actually found. That last step is the whole job of this
module, and it exists because the YAML geometry is synthetic:

    map: {kind: straight, num_lanes: 3, lane_width: 3.5, length: 120}

A real CARLA lane is not 3.5 m wide, and a real straight stretch is not 120 m
long. Copying the YAML coordinates over verbatim would put actors on the lane
markings, or off the end of the road. So each actor keeps its *intent* — which
lane it is in, how far along it starts, which way it faces, what it does — and
the geometry is re-derived from the frame:

    lane index (from the YAML map)  ->  frame.lane_center_x(index)
    longitudinal y                  ->  scaled if the real road is shorter
    heading                         ->  the real travel direction of that lane

Speeds and maneuver plans are untouched: they are the scenario.

The ego (actor "0") is split out of the returned scenario. `carla_highway`
drives it closed-loop with a policy, exactly as `carla_port` does, so leaving it
in the background set would double it.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .highway_map import FORWARD_HEADING, HighwayFrame
from .script_bridge import mp, se

HERE = os.path.dirname(os.path.abspath(__file__))

#: the authored YAML lives with the script layer — the repository root on this
#: branch, a vendored `highway/` subdirectory when the port is dropped into
#: another tree. Probed the same way `script_bridge.HIGHWAY` is.
from .script_bridge import HIGHWAY as _HIGHWAY          # noqa: E402
SCENARIO_DIR = os.path.join(_HIGHWAY, "scenarios")

EGO_ID = "0"

CUTIN = "cutin"
HARD_BRAKE = "hard_brake"
OVERTAKE = "overtake"
MODES = (CUTIN, HARD_BRAKE, OVERTAKE)


@dataclass(frozen=True)
class ModeSpec:
    """What a scenario needs from the road, and how it is orchestrated."""
    name: str
    yaml: str
    lanes: int
    two_way: bool
    min_length: float
    #: does the orchestrator cast cut-in / block roles in this mode?
    casting: bool
    #: may the ego change lanes on its own?
    #:
    #: Off for `cutin`: upstream drives that ego along a straight scripted
    #: speed profile, and the scenario is a test of what the ACTORS do around
    #: it. An ego that also weaves re-poses the cut-in pin every tick, the
    #: orchestrator recasts against a moving target, and the run measures the
    #: ego's lane choices instead of the casting. `--lane-change` overrides.
    ego_lane_changes: bool
    #: simulated seconds the scenario needs to resolve
    duration: float
    blurb: str
    #: the generated straight highway this mode is meant to be recorded on
    #: (`carla_highway/make_maps.py`), used by `--xodr auto`. Every mode runs
    #: on the stock towns too; this is the road that makes it legible.
    xodr: str = ""


SPECS: Dict[str, ModeSpec] = {
    CUTIN: ModeSpec(
        name=CUTIN, yaml="scenario_cutin.yaml", lanes=3, two_way=False,
        min_length=120.0, casting=True, ego_lane_changes=False,
        duration=12.0, xodr="highway_3lane",
        blurb="3-lane one-way; orchestrator casts a cut-in against the live ego"),
    HARD_BRAKE: ModeSpec(
        name=HARD_BRAKE, yaml="scenario_hard_brake.yaml", lanes=2, two_way=False,
        min_length=160.0, casting=False, ego_lane_changes=True,
        duration=14.0, xodr="highway_2lane",
        blurb="2-lane one-way; slow lead in the ego lane, squeezed merge gap"),
    OVERTAKE: ModeSpec(
        name=OVERTAKE, yaml="scenario_overtake.yaml", lanes=2, two_way=True,
        min_length=160.0, casting=False, ego_lane_changes=True,
        duration=20.0, xodr="highway_2lane_twoway",
        blurb="2-lane two-way; stopped blocker, oncoming car in the only way past"),
}


def spec(mode: str) -> ModeSpec:
    if mode not in SPECS:
        raise ValueError(f"unknown scenario mode {mode!r}; expected one of "
                         f"{', '.join(MODES)}")
    return SPECS[mode]


def scenario_path(mode: str) -> str:
    return os.path.join(SCENARIO_DIR, spec(mode).yaml)


# --------------------------------------------------------------------------- #
# Loading + retargeting
# --------------------------------------------------------------------------- #
def load(mode: str, path: Optional[str] = None) -> "se.Scenario":
    """The authored scenario, straight off disk, in its own synthetic frame."""
    return se.load_scenario(path or scenario_path(mode))


def discover_frame(world, mode: str, min_length: Optional[float] = None,
                   road_id: Optional[int] = None) -> HighwayFrame:
    """Find the CARLA road this mode needs."""
    s = spec(mode)
    return HighwayFrame.discover(world, lanes=s.lanes, two_way=s.two_way,
                                 min_length=min_length or s.min_length,
                                 road_id=road_id)


def retarget(sc: "se.Scenario", frame: HighwayFrame,
             stretch: bool = True,
             along_offset: float = 0.0) -> Tuple["se.Scenario", List[str]]:
    """Move an authored scenario onto `frame`'s real road, in place-ish.

    Returns (scenario, notes). The scenario is a deep copy; the original YAML
    object is left alone so a caller can diff the two.

    `along_offset` slides the whole scenario — every actor, the ego included —
    that many metres up the road. The lateral layout, the spacing between actors
    and every speed are untouched, so it moves WHERE the scenario happens and
    not WHAT happens.

    It exists because a stretch of road can be unusable for reasons the fitter
    cannot see. Town04's road 47 runs under an overpass between script y=-10 and
    y=+30, and while the driving there is fine, the bird's-eye camera films the
    deck instead of the cars — the ego simply vanishes from the recording for
    half the run. The deck cannot be hidden reliably: it is many meshes, and the
    ones large enough to matter are indistinguishable by label or bounding box
    from the carriageway itself. Moving the scenario is the cheap fix, and the
    road either side is 260 m of the same geometry.
    """
    sc = copy.deepcopy(sc)
    notes: List[str] = []
    src = sc.map
    n_src = max(1, int(getattr(src, "num_lanes", 1) or 1))

    if n_src != frame.num_lanes:
        notes.append(f"scenario wants {n_src} lanes, frame fitted "
                     f"{frame.num_lanes}; lanes mapped by nearest index")

    # Longitudinal: the YAML lays actors out over `src.length`. If the real
    # straight is shorter, squeeze; never stretch beyond the authored spacing,
    # because the scenarios are timing-critical (a 30 m gap at 8 m/s is the
    # scenario).
    if along_offset:
        notes.append(f"scenario slid {along_offset:+.0f} m along the road "
                     "(staging only; spacing and speeds unchanged)")
    src_len = float(getattr(src, "length", 0.0) or 0.0)
    y_scale = 1.0
    if stretch and src_len > 0 and frame.length < src_len:
        y_scale = frame.length / src_len
        notes.append(f"real straight is {frame.length:.0f} m vs the scenario's "
                     f"{src_len:.0f} m; longitudinal layout scaled x{y_scale:.2f} "
                     "(gaps and therefore timings are compressed)")

    for a in sc.actors:
        x, y, h = a.start
        # which lane did the author mean?
        idx = _nearest_lane_index(src, x, n_src)
        idx = min(idx, frame.num_lanes - 1)
        new_x = frame.lane_center_x(idx)
        new_y = y * y_scale + along_offset
        # Preserve the authored direction of travel, but express it as the real
        # lane's heading so oncoming traffic actually faces into the ego.
        authored_forward = _is_forward(h)
        lane = frame.lanes[idx] if idx < len(frame.lanes) else None
        if lane is not None and lane.same_direction != authored_forward:
            if frame.two_way_ok(authored_forward):
                alt = frame.lane_index_for_direction(authored_forward, prefer=idx)
                if alt is not None and alt != idx:
                    notes.append(f"actor {a.id}: authored lane {idx} runs the "
                                 f"wrong way on this road; moved to lane {alt}")
                    idx = alt
                    new_x = frame.lane_center_x(idx)
                    lane = frame.lanes[idx]
        if lane is not None and lane.same_direction != authored_forward:
            # No lane runs the way the author wanted. Taking the lane's own
            # direction silently is the worst option available: on `overtake`
            # it would turn the oncoming car into a car driving away, and the
            # scenario would look like it ran fine while testing nothing.
            notes.append(
                f"actor {a.id}: this road has no "
                f"{'forward' if authored_forward else 'oncoming'} lane for it; "
                "it now runs with the lane, which changes what the scenario "
                "tests")
        new_h = lane.heading if lane is not None else (
            FORWARD_HEADING if authored_forward else (FORWARD_HEADING + 180.0) % 360.0)
        a.start = (new_x, new_y, new_h)

    sc.map = frame.map_config()
    sc.simulate()
    return sc, notes


def _is_forward(heading: float) -> bool:
    """Does this authored heading mean 'with the ego'?"""
    d = (float(heading) - FORWARD_HEADING + 180.0) % 360.0 - 180.0
    return abs(d) < 90.0


def _nearest_lane_index(m, x: float, n_lanes: int) -> int:
    best, best_d = 0, float("inf")
    for i in range(n_lanes):
        d = abs(m.lane_center_x(i) - x)
        if d < best_d:
            best, best_d = i, d
    return best


# --------------------------------------------------------------------------- #
# Splitting the ego out
# --------------------------------------------------------------------------- #
def split_ego(sc: "se.Scenario", ego_id: str = EGO_ID
              ) -> Tuple["se.Actor", "se.Scenario"]:
    """(ego actor, background scenario without it).

    The background keeps the same `MapConfig` object, so the orchestrator and
    the ego policy agree about the road.
    """
    ego = next((a for a in sc.actors if str(a.id) == str(ego_id)), None)
    if ego is None:
        raise ValueError(f"scenario has no actor {ego_id!r} to use as the ego")
    bg = se.Scenario(map=sc.map,
                     actors=[a for a in sc.actors if str(a.id) != str(ego_id)],
                     pixels_per_meter=getattr(sc, "pixels_per_meter", 6.0))
    bg.simulate()
    return ego, bg


def build(world, mode: str, path: Optional[str] = None,
          min_length: Optional[float] = None, road_id: Optional[int] = None,
          frame: Optional[HighwayFrame] = None, along_offset: float = 0.0
          ) -> Tuple[HighwayFrame, "se.Actor", "se.Scenario", List[str]]:
    """Everything a run needs: (frame, ego actor, background scenario, notes)."""
    frame = frame or discover_frame(world, mode, min_length=min_length,
                                    road_id=road_id)
    authored = load(mode, path)
    fitted, notes = retarget(authored, frame, along_offset=along_offset)
    ego, bg = split_ego(fitted)
    return frame, ego, bg, notes
