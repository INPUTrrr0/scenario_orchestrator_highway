#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/make_maps.py — straight highway maps, authored not borrowed.

Why generate a map at all
-------------------------
`survey_maps` answers the first question, and the answer is a qualified yes:
the stock towns DO have a real highway, and it is Town06 road 49 — 300 m of
junction-free four-lane straight, the only one in any shipped town (the survey
is written up in `docs/HIGHWAY_MAPS.md`). Everything else is town road, which
is how `overtake` came to thread a town centre where the manoeuvre is correct
and nearly impossible to watch.

Town06 covers the same-direction scenarios well. It does not cover `overtake`,
which needs a **contraflow** lane, and the survey found no two-way
junction-free straight anywhere longer than 90 m against a scenario asking for
160 m.

So this module authors the roads instead. `client.generate_opendrive_world()`
builds a drivable world from an OpenDRIVE string with, in the API's own words,
"no graphics besides the road and sidewalks" — no buildings, no parked cars, no
cross traffic, nothing to occlude anything. For watching a lane change that is
not a downgrade; it is the point.

    python3 -m carla_highway.make_maps
    python3 -m carla_highway.runner --scenario cutin --xodr maps/highway_3lane.xodr

What it emits
-------------
One straight, flat road per lane count, plus the two-way variant `overtake`
needs:

    highway_2lane.xodr          2 lanes, one way    hard_brake
    highway_3lane.xodr          3 lanes, one way    cutin
    highway_4lane.xodr          4 lanes, one way    headroom
    highway_2lane_twoway.xodr   1 lane each way     overtake

Geometry notes that matter to the port
--------------------------------------
* **Lane sign.** In OpenDRIVE, right-hand lanes carry negative ids and run
  along +s; left-hand lanes are positive and run against it. That is exactly
  the same/oncoming split `HighwayFrame` reads off a waypoint's yaw, so a
  `<left>` block is what makes a road two-way and nothing else has to know.
* **Lane width is uniform.** `MapConfig.lane_center_x` places scenario actors
  on an evenly spaced grid and `_fit_section` rejects a section whose spacing
  is uneven. 3.5 m throughout matches every stock town, so a scenario authored
  against the towns transfers unchanged.
* **Shoulders, not walls, at the edge.** A `driving` lane flush against the
  generated boundary wall lets a body clip it. A shoulder each side gives the
  outermost lane somewhere to overhang. `_siblings` walks *past* non-driving
  lanes without counting them, so a shoulder can never be mistaken for a lane.
* **Flat and straight.** One `<line>` geometry and a zero elevation profile:
  the script frame is a straight line, so any curvature is fit error the
  scenario pays for (`MAX_LATERAL_DEV`).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

LANE_WIDTH = 3.5
SHOULDER_WIDTH = 2.0
#: 400 m is the port's own `max_length` fit cap, so the rest is headroom for
#: the actors' 40 s scripts rather than road the scenario measures on.
DEFAULT_LENGTH = 600.0
DEFAULT_SPEED_KMH = 120.0


def _lane(lane_id: int, lane_type: str, width: float,
          mark: str = "broken") -> str:
    """One `<lane>`. `mark` is the line painted on its OUTER edge."""
    if mark == "none":
        roadmark = ('          <roadMark sOffset="0.0" type="none" '
                    'weight="standard" color="standard" width="0.0" '
                    'laneChange="both"/>\n')
    else:
        roadmark = ('          <roadMark sOffset="0.0" type="%s" '
                    'weight="standard" color="standard" width="0.15" '
                    'laneChange="both"/>\n' % mark)
    return ('        <lane id="%d" type="%s" level="false">\n'
            '          <link/>\n'
            '          <width sOffset="0.0" a="%s" b="0.0" c="0.0" d="0.0"/>\n'
            '%s'
            '        </lane>\n' % (lane_id, lane_type, width, roadmark))


def build_xodr(name: str, forward_lanes: int, backward_lanes: int = 0,
               length: float = DEFAULT_LENGTH,
               lane_width: float = LANE_WIDTH,
               shoulder: float = SHOULDER_WIDTH,
               speed_kmh: float = DEFAULT_SPEED_KMH) -> str:
    """OpenDRIVE for one straight, flat road.

    `forward_lanes` run along +s (OpenDRIVE right side, negative ids);
    `backward_lanes` run against it (left side, positive ids) and are what make
    the road two-way.
    """
    if forward_lanes < 1:
        raise ValueError("a road needs at least one forward lane")
    if length <= 0.0:
        raise ValueError("length must be positive")

    # --- right side: the direction of travel, innermost (-1) outward --- #
    right: List[str] = []
    for k in range(1, forward_lanes + 1):
        # broken line between lanes, solid on the outermost lane's outer edge
        right.append(_lane(-k, "driving", lane_width,
                           "solid" if k == forward_lanes else "broken"))
    right.append(_lane(-(forward_lanes + 1), "shoulder", shoulder, "none"))

    # --- left side: contraflow, if any --- #
    left: List[str] = []
    for k in range(1, backward_lanes + 1):
        left.append(_lane(k, "driving", lane_width,
                          "solid" if k == backward_lanes else "broken"))
    if backward_lanes:
        left.append(_lane(backward_lanes + 1, "shoulder", shoulder, "none"))
    # OpenDRIVE lists left lanes outermost-first
    left_block = ("      <left>\n" + "".join(reversed(left)) + "      </left>\n"
                  if left else "")

    # the centre lane is a marking, never drivable: solid when it divides
    # opposing traffic, broken when every lane runs the same way
    centre_mark = "solid" if backward_lanes else "broken"

    return (
        '<?xml version="1.0" standalone="yes"?>\n'
        '<OpenDRIVE>\n'
        '  <header revMajor="1" revMinor="4" name="%(name)s" version="1.00"\n'
        '          north="0.0" south="0.0" east="0.0" west="0.0" vendor="carla_highway"/>\n'
        '  <road name="%(name)s" length="%(length).4f" id="1" junction="-1">\n'
        '    <type s="0.0" type="motorway">\n'
        '      <speed max="%(speed).4f" unit="m/s"/>\n'
        '    </type>\n'
        '    <planView>\n'
        '      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="%(length).4f">\n'
        '        <line/>\n'
        '      </geometry>\n'
        '    </planView>\n'
        '    <elevationProfile>\n'
        '      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>\n'
        '    </elevationProfile>\n'
        '    <lateralProfile/>\n'
        '    <lanes>\n'
        '      <laneOffset s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>\n'
        '      <laneSection s="0.0">\n'
        '%(left)s'
        '      <center>\n'
        '        <lane id="0" type="none" level="false">\n'
        '          <roadMark sOffset="0.0" type="%(centre)s" weight="standard"'
        ' color="standard" width="0.15" laneChange="both"/>\n'
        '        </lane>\n'
        '      </center>\n'
        '      <right>\n'
        '%(right)s'
        '      </right>\n'
        '      </laneSection>\n'
        '    </lanes>\n'
        '  </road>\n'
        '</OpenDRIVE>\n'
        % {"name": name, "length": length, "speed": speed_kmh / 3.6,
           "left": left_block, "centre": centre_mark, "right": "".join(right)})


#: name -> (forward lanes, backward lanes, what it is for)
PRESETS = {
    "highway_2lane": (2, 0, "hard_brake - 2 same-direction lanes"),
    "highway_3lane": (3, 0, "cutin - 3 same-direction lanes"),
    "highway_4lane": (4, 0, "headroom - 4 same-direction lanes"),
    "highway_2lane_twoway": (1, 1, "overtake - one lane each way"),
}

DEFAULT_MAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")


def write_presets(out_dir: str, length: float = DEFAULT_LENGTH,
                  lane_width: float = LANE_WIDTH,
                  only: Optional[List[str]] = None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, (fwd, bwd, blurb) in sorted(PRESETS.items()):
        if only and name not in only:
            continue
        path = os.path.join(out_dir, name + ".xodr")
        with open(path, "w") as fh:
            fh.write(build_xodr(name, fwd, bwd, length=length,
                                lane_width=lane_width))
        written.append(path)
        print("  %-28s %d+%d lanes  %.0f m   %s"
              % (os.path.basename(path), fwd, bwd, length, blurb))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_MAP_DIR,
                    help="where to write the .xodr files (default <repo>/maps)")
    ap.add_argument("--length", type=float, default=DEFAULT_LENGTH,
                    help="road length in metres (default %.0f)" % DEFAULT_LENGTH)
    ap.add_argument("--lane-width", type=float, default=LANE_WIDTH)
    ap.add_argument("--only", action="append", default=None,
                    choices=sorted(PRESETS), help="write just this preset")
    args = ap.parse_args(argv)

    print("writing straight highway maps to %s" % args.out_dir)
    paths = write_presets(args.out_dir, length=args.length,
                          lane_width=args.lane_width, only=args.only)
    if not paths:
        print("nothing written", file=sys.stderr)
        return 1
    print("\n%d map(s). Load one with:" % len(paths))
    print("  python3 -m carla_highway.runner --scenario cutin --xodr "
          + os.path.join(args.out_dir, "highway_3lane.xodr"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
