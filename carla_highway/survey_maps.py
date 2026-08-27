#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/survey_maps.py — which CARLA town has a real highway?

`HighwayFrame.discover` fits the scenario onto *whatever* road best matches,
and by default it walks straight through junctions
(`_straight_run(through_junctions=True)`). That is the right default for
getting a scenario to run anywhere — but it is why `overtake` landed on
Town04 road 3, which threads a town centre: the manoeuvre is correct and
nearly impossible to watch behind buildings, parked cars and cross traffic.

This module answers the prior question. It loads every town's OpenDRIVE
offline — no server, `carla.Map(name, xodr)` is enough — and reports, for each
lane count, the longest **junction-free** straight run in the map. A section
that qualifies with `through_junctions=False` has no cross traffic and no
intersection anywhere in the span the scenario will use.

    python3 -m carla_highway.survey_maps                  # all towns, 2/3/4 lanes
    python3 -m carla_highway.survey_maps --lanes 3 -v     # every candidate
    python3 -m carla_highway.survey_maps --two-way        # overtake's road

The result is written up in `docs/HIGHWAY_MAPS.md`; this is the tool that
produced it, so the table can be regenerated when the CARLA build changes.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from carla_port.carla_api import carla

from .highway_map import (FORWARD_HEADING, HighwayFrame, _candidate_sections,
                          _fit_section)

DEFAULT_XODR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "install", "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive")

#: Not a shipped town. `generate_opendrive_world` writes the OpenDRIVE it was
#: handed back into this directory under this name, so after any run on a
#: generated map the survey would otherwise report our own road as a discovery.
GENERATED_WRITEBACK = "OpenDriveMap"


def _towns(xodr_dir: str) -> List[str]:
    """The shipped towns, one entry per road network.

    `*_Opt` towns are the same network with the props split into loadable
    layers; the OpenDRIVE is identical, so surveying both just doubles the rows.
    """
    names = sorted(n[:-5] for n in os.listdir(xodr_dir) if n.endswith(".xodr"))
    return [n for n in names
            if not n.endswith("_Opt") and n != GENERATED_WRITEBACK]


def load_town(xodr_dir: str, town: str):
    path = os.path.join(xodr_dir, town + ".xodr")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as fh:
        return carla.Map(town, fh.read())


class _World:
    """The two methods `HighwayFrame.discover` actually calls."""

    def __init__(self, cmap):
        self._map = cmap

    def get_map(self):
        return self._map


def survey_town(cmap, lanes: int, two_way: bool = False,
                min_length: float = 60.0, max_length: float = 1000.0,
                junction_free: bool = True) -> List[dict]:
    """Every section of `cmap` that fits `lanes` lanes, longest first.

    `junction_free` is the whole point of this module: with it, the run is
    measured with `through_junctions=False`, so the reported length is road
    the scenario can use without ever crossing an intersection.
    """
    out: List[dict] = []
    for seed in _candidate_sections(cmap, seed_step=5.0, road_id=None):
        fit = _fit_section(seed, lanes=lanes, two_way=two_way,
                           min_length=min_length,
                           straight_tol_deg=4.0, walk_step=5.0,
                           max_length=max_length,
                           through_junctions=not junction_free)
        if isinstance(fit, str):
            continue
        n_same = sum(1 for ln in fit["lanes"] if ln.same_direction)
        out.append({
            "road_id": fit["road_id"],
            "section_id": fit["section_id"],
            "length": fit["length"],
            "lane_width": fit["lane_width"],
            "same": n_same,
            "oncoming": len(fit["lanes"]) - n_same,
            "theta": fit["theta"],
            "anchor": fit["anchor"],
            "warnings": fit.get("warnings") or [],
        })
    out.sort(key=lambda f: -f["length"])
    return out


def _fmt(f: dict) -> str:
    warn = ("  ! " + "; ".join(f["warnings"])) if f["warnings"] else ""
    return (f"road {f['road_id']:>5} sec {f['section_id']}  "
            f"{f['length']:7.1f} m  lw={f['lane_width']:.2f}  "
            f"same={f['same']} onc={f['oncoming']}{warn}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xodr-dir", default=DEFAULT_XODR)
    ap.add_argument("--town", action="append", default=None,
                    help="restrict to these towns (repeatable)")
    ap.add_argument("--lanes", type=int, action="append", default=None,
                    help="lane counts to survey (default 2 3 4)")
    ap.add_argument("--two-way", action="store_true",
                    help="require an oncoming lane in the window (overtake)")
    ap.add_argument("--min-length", type=float, default=60.0)
    ap.add_argument("--allow-junctions", action="store_true",
                    help="measure runs straight through intersections, as "
                         "HighwayFrame.discover does by default")
    ap.add_argument("-n", "--top", type=int, default=3,
                    help="candidates to print per town (default 3)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every qualifying section")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.xodr_dir):
        print(f"no OpenDRIVE directory at {args.xodr_dir}", file=sys.stderr)
        return 2
    towns = args.town or _towns(args.xodr_dir)
    lane_counts = args.lanes or [2, 3, 4]
    jf = not args.allow_junctions

    print(f"OpenDRIVE: {args.xodr_dir}")
    print(f"towns: {', '.join(towns)}")
    print(f"runs measured {'WITHOUT crossing junctions' if jf else 'THROUGH junctions'}"
          f"; {'two-way' if args.two_way else 'same-direction'} lanes\n")

    best: Dict[int, Tuple[float, str, dict]] = {}
    for town in towns:
        cmap = load_town(args.xodr_dir, town)
        if cmap is None:
            print(f"== {town}: no .xodr ==")
            continue
        print(f"== {town} ==")
        for lanes in lane_counts:
            fits = survey_town(cmap, lanes, two_way=args.two_way,
                               min_length=args.min_length,
                               junction_free=jf)
            if not fits:
                print(f"  {lanes} lanes: none")
                continue
            shown = fits if args.verbose else fits[:args.top]
            print(f"  {lanes} lanes: {len(fits)} section(s)")
            for f in shown:
                print(f"    {_fmt(f)}")
            top = fits[0]
            if lanes not in best or top["length"] > best[lanes][0]:
                best[lanes] = (top["length"], town, top)
        print()

    print("=" * 68)
    print("BEST PER LANE COUNT")
    print("=" * 68)
    for lanes in lane_counts:
        if lanes not in best:
            print(f"  {lanes} lanes: NOTHING in any town")
            continue
        length, town, f = best[lanes]
        print(f"  {lanes} lanes: {town:9} {_fmt(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
