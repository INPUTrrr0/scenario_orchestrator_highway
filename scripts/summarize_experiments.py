#!/usr/bin/env python3
"""Summarize experiment JSON files by intention (cut-in, block).

Reads every ``*.json`` in the experiments folder and, for each intention,
reports how many trials included that intention and how many succeeded.

Cut-in success is reported twice:
  * **recorded** — ``cutin.success == true`` from the simulator
  * **verified** — a *proper* cut-in: the performer started in a different lane,
    completed a lane change into the ego's **original** lane (world x at spawn),
    ended ahead of the ego, and the ego had not already left that lane or moved
    into the actor's spawn lane before the actor finished merging.  Trials where
    the performer already shared the ego's lane at spawn are excluded from the
    verified denominator.

Usage:
    .venv/bin/python scripts/summarize_experiments.py
    .venv/bin/python scripts/summarize_experiments.py --dir experiments
    .venv/bin/python scripts/summarize_experiments.py --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

INTENTIONS = ("cutin", "block")
LABELS = {"cutin": "cut-in", "block": "block cut-in"}

# match scenario_editor / cutin_orchestrator lane scoring
DEFAULT_LANE_WIDTH = 3.5
SPAWN_SAME_LANE_FRAC = 0.4   # spawn x within this × lane_width of ego home → same lane
IN_LANE_FRAC = 0.25          # world-x band for "in this lane" (strict)
MIN_AHEAD_M = 0.5            # along-track metres in front of ego at commit
TIME_EPS = 0.02              # seconds — ordering tolerance


def _here() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_run(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return None


def intention_cast(data: dict, key: str) -> bool:
    cast = data.get("cast") or {}
    return cast.get(key) is not None


def intention_success(data: dict, key: str) -> Optional[bool]:
    rec = data.get(key) or {}
    return rec.get("success")


def _heading_axes(hd_deg: float) -> Tuple[float, float, float, float]:
    h = math.radians(hd_deg)
    fx, fy = math.cos(h), math.sin(h)
    return fx, fy, -fy, fx


def world_to_ego_offset(ego_pose: Tuple[float, float, float],
                        wx: float, wy: float) -> Tuple[float, float]:
    """(along, lat) in the ego body frame — same convention as scenario_editor."""
    ex, ey, eh = ego_pose
    fx, fy, nx, ny = _heading_axes(eh)
    dx, dy = wx - ex, wy - ey
    return dx * fx + dy * fy, dx * nx + dy * ny


def pose_at_time(traj: List[list], t: float) -> Optional[Tuple[float, float, float]]:
    """Last sample with timestamp <= ``t``; traj rows are [t, x, y, heading, ...]."""
    if not traj:
        return None
    best = traj[0]
    for row in traj:
        if row[0] <= t + 1e-6:
            best = row
        else:
            break
    return (float(best[1]), float(best[2]), float(best[3]))


def infer_lane_width(data: dict) -> float:
    """Use stored value if present, else infer from spawn lateral spacing."""
    if "lane_width" in data:
        return float(data["lane_width"])
    ego = data.get("ego_trajectory") or []
    spawns = data.get("spawns") or {}
    if not ego or not spawns:
        return DEFAULT_LANE_WIDTH
    ex = float(ego[0][1])
    gaps = [abs(float(sp[0]) - ex) for sp in spawns.values() if abs(float(sp[0]) - ex) > 0.1]
    return min(gaps) if gaps else DEFAULT_LANE_WIDTH


def _lane_band(lw: float, frac: float = IN_LANE_FRAC) -> float:
    return frac * lw


def _in_lane_x(x: float, lane_x: float, lw: float,
               frac: float = IN_LANE_FRAC) -> bool:
    """True when world x is inside the lane centred at ``lane_x``."""
    return abs(x - lane_x) < _lane_band(lw, frac)


def _first_event(traj: List[list], t_max: float,
                 pred) -> Optional[Tuple[float, float]]:
    """First row with t <= t_max where ``pred(t, x, y, heading)`` is true."""
    for row in traj:
        t = float(row[0])
        if t > t_max + 1e-6:
            break
        if pred(t, float(row[1]), float(row[2]), float(row[3])):
            return t, float(row[1])
    return None


def verify_proper_cutin(data: dict,
                        lane_width: Optional[float] = None
                        ) -> Tuple[bool, str, Dict[str, Any]]:
    """Geometric / temporal check for a proper cut-in.

    A proper cut-in requires the actor to leave its spawn lane, merge into the
    ego's *original* lane (ego home x at session start), and finish ahead of
    the ego — while the ego is still in that lane.  It is NOT a cut-in if the
    ego lane-changes away (e.g. into the actor's spawn lane) before the actor
    completes the merge into the ego's lane.
    """
    cast = data.get("cast") or {}
    rec = data.get("cutin") or {}
    performer = rec.get("performer") or cast.get("cutin")
    if performer is None:
        return False, "no cut-in performer", {}

    lw = lane_width if lane_width is not None else infer_lane_width(data)
    spawns = data.get("spawns") or {}
    if performer not in spawns:
        return False, f"spawn missing for actor {performer}", {}

    ego_tr = data.get("ego_trajectory") or []
    if not ego_tr:
        return False, "no ego trajectory", {}

    ego_home_x = float(ego_tr[0][1])
    actor_home_x = float(spawns[performer][0])
    details: Dict[str, Any] = {
        "performer": performer,
        "lane_width": lw,
        "ego_home_x": ego_home_x,
        "actor_home_x": actor_home_x,
    }

    if _in_lane_x(actor_home_x, ego_home_x, lw, SPAWN_SAME_LANE_FRAC):
        return False, "already on ego lane at spawn", details

    if rec.get("success") is not True:
        return False, "recorded success is not true", details

    t_commit = rec.get("t_commit")
    if t_commit is None:
        return False, "no commit time", details
    t_commit = float(t_commit)
    details["t_commit"] = t_commit

    actor_tr = (data.get("actor_trajectories") or {}).get(performer)
    if not actor_tr:
        return False, "no actor trajectory", details

    # --- temporal ordering up to commit -------------------------------- #
    def actor_enters_ego_lane(t, x, y, hd):
        return _in_lane_x(x, ego_home_x, lw)

    def ego_leaves_home(t, x, y, hd):
        return not _in_lane_x(x, ego_home_x, lw)

    def ego_enters_actor_lane(t, x, y, hd):
        return _in_lane_x(x, actor_home_x, lw)

    t_actor_in = _first_event(actor_tr, t_commit, actor_enters_ego_lane)
    t_ego_out = _first_event(ego_tr, t_commit, ego_leaves_home)
    t_ego_to_actor = _first_event(ego_tr, t_commit, ego_enters_actor_lane)

    if t_actor_in:
        details["t_actor_enter_ego_lane"] = round(t_actor_in[0], 3)
    if t_ego_out:
        details["t_ego_leave_home_lane"] = round(t_ego_out[0], 3)
    if t_ego_to_actor:
        details["t_ego_enter_actor_lane"] = round(t_ego_to_actor[0], 3)

    if t_ego_out is not None:
        if t_actor_in is None or t_ego_out[0] + TIME_EPS < t_actor_in[0]:
            return False, "ego left target lane before actor merged", details

    if t_ego_to_actor is not None:
        if t_actor_in is None or t_ego_to_actor[0] + TIME_EPS < t_actor_in[0]:
            return False, "ego moved to actor lane before actor merged", details

    # --- poses at commit (world lane + longitudinal) --------------------- #
    ego_t = pose_at_time(ego_tr, t_commit)
    actor_t = pose_at_time(actor_tr, t_commit)
    if ego_t is None or actor_t is None:
        return False, "could not sample poses at commit", details

    if not _in_lane_x(ego_t[0], ego_home_x, lw):
        return False, "ego not in original lane at commit", details
    if not _in_lane_x(actor_t[0], ego_home_x, lw):
        return False, "actor not in ego lane at commit", details

    along, lat = world_to_ego_offset(ego_t, actor_t[0], actor_t[1])
    details.update({
        "ego_x_commit": round(ego_t[0], 3),
        "actor_x_commit": round(actor_t[0], 3),
        "along_commit": round(along, 3),
        "lat_commit": round(lat, 3),
    })

    if along <= MIN_AHEAD_M:
        return False, "not ahead of ego at commit", details

    # actor must have moved toward ego home from its spawn lane
    if abs(actor_t[0] - ego_home_x) >= abs(actor_home_x - ego_home_x) - 0.2:
        return False, "actor did not move into ego lane from adjacent lane", details

    return True, "proper cut-in", details


def summarize_dir(experiments_dir: str
                  ) -> Tuple[Dict[str, Dict[str, int]], List[dict]]:
    """Return ({intention: counts}, per-file records)."""
    totals = {k: {"runs": 0, "success": 0, "fail": 0, "pending": 0} for k in INTENTIONS}
    cutin_verify = {
        "proper_runs": 0,           # cast + started off ego lane
        "spawn_same_lane": 0,       # cast but already on ego lane
        "verified_success": 0,      # proper + recorded success + geometry at commit
        "recorded_not_verified": 0, # recorded success but failed geometry
    }
    records: List[dict] = []

    if not os.path.isdir(experiments_dir):
        raise SystemExit(f"not a directory: {experiments_dir}")

    paths = sorted(
        os.path.join(experiments_dir, name)
        for name in os.listdir(experiments_dir)
        if name.endswith(".json")
    )
    if not paths:
        raise SystemExit(f"no *.json files in {experiments_dir}")

    for path in paths:
        data = load_run(path)
        if data is None:
            continue
        name = os.path.basename(path)
        rec = {"path": path, "name": name}

        for key in INTENTIONS:
            if not intention_cast(data, key):
                continue
            totals[key]["runs"] += 1
            ok = intention_success(data, key)
            if ok is True:
                totals[key]["success"] += 1
            elif ok is False:
                totals[key]["fail"] += 1
            else:
                totals[key]["pending"] += 1

        if intention_cast(data, "cutin"):
            lw = infer_lane_width(data)
            performer = (data.get("cutin") or {}).get("performer") or data["cast"]["cutin"]
            spawns = data.get("spawns") or {}
            ego_tr = data.get("ego_trajectory") or []
            on_lane_at_spawn = False
            if performer in spawns and ego_tr:
                ego_home_x = float(ego_tr[0][1])
                actor_home_x = float(spawns[performer][0])
                on_lane_at_spawn = _in_lane_x(
                    actor_home_x, ego_home_x, lw, SPAWN_SAME_LANE_FRAC)
                rec["cutin_actor_home_x"] = actor_home_x
                rec["cutin_ego_home_x"] = ego_home_x

            if on_lane_at_spawn:
                cutin_verify["spawn_same_lane"] += 1
            else:
                cutin_verify["proper_runs"] += 1

            verified, reason, details = verify_proper_cutin(data, lw)
            rec["cutin_verified"] = verified
            rec["cutin_verify_reason"] = reason
            rec["cutin_verify_details"] = details

            if verified:
                cutin_verify["verified_success"] += 1
            elif intention_success(data, "cutin") is True:
                cutin_verify["recorded_not_verified"] += 1
                rec["cutin_mismatch"] = True

        records.append(rec)

    return {**totals, "cutin_verify": cutin_verify}, records


def print_report(summary: Dict[str, Any], n_files: int, verbose: bool) -> None:
    print(f"Experiment files read: {n_files}\n")

    for key in INTENTIONS:
        t = summary[key]
        label = LABELS[key]
        runs = t["runs"]
        succ = t["success"]
        if runs == 0:
            print(f"{label:14}  runs: 0  (no trials cast this intention)")
            continue
        rate = 100.0 * succ / runs
        line = f"{label:14}  recorded success: {succ:4d} / {runs:4d} runs  ({rate:5.1f}%)"
        if t["fail"]:
            line += f"  fail: {t['fail']}"
        if t["pending"]:
            line += f"  pending: {t['pending']}"
        print(line)

    cv = summary["cutin_verify"]
    if summary["cutin"]["runs"]:
        print()
        print("Cut-in verification (proper = actor merges into ego's original "
              "lane while ego stays there; ego moving to actor's lane first "
              "does not count):")
        pr = cv["proper_runs"]
        vs = cv["verified_success"]
        if pr:
            vrate = 100.0 * vs / pr
            print(f"  proper runs (off ego lane at spawn):     {pr:4d}")
            print(f"  excluded (already on ego lane at spawn): {cv['spawn_same_lane']:4d}")
            print(f"  verified success:                      {vs:4d} / {pr:4d}  ({vrate:5.1f}%)")
        else:
            print("  no proper cut-in runs (all performers started on the ego lane)")
        if cv["recorded_not_verified"]:
            print(f"  recorded success but NOT verified:       {cv['recorded_not_verified']:4d}")

    if verbose:
        print("\nPer-file cut-in verification:")
        for rec in summary.get("_records", []):
            if "cutin_verified" not in rec:
                continue
            flag = "OK" if rec["cutin_verified"] else rec["cutin_verify_reason"]
            extra = ""
            if rec.get("cutin_mismatch"):
                extra = "  ** recorded success, failed verification **"
            print(f"  {rec['name']:16}  {flag}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize cut-in / block success rates from experiment JSON runs")
    ap.add_argument(
        "--dir", default=os.path.join(_here(), "experiments"),
        help="folder containing run_*.json files (default: ../experiments)",
    )
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print per-file cut-in verification results")
    args = ap.parse_args()

    summary, records = summarize_dir(os.path.abspath(args.dir))
    summary["_records"] = records
    print_report(summary, len(records), args.verbose)


if __name__ == "__main__":
    main()
