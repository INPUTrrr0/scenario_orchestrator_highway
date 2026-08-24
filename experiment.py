#!/usr/bin/env python3
"""
experiment.py — record cut-in / block-cut-in driving trials.
===========================================================

A thin experiment harness on top of the existing scenario editor.  It does NOT
change the simulator: it builds a *randomized* scenario, drops you straight into
Drive mode (drive the ego with WASD / arrow keys), and writes a JSON result file
recording what happened.

Each trial:
  * spawns a random number of actors (1..5, uniform) at random, non-overlapping
    lane slots around the ego — all derived from a single ``--seed``;
  * casts one actor as the CUT-IN and (traffic permitting) another as the
    BLOCK, using the orchestrator's placement scores;
  * lets the orchestrator run closed-loop while you drive the ego;
  * records: the random seed, the ego trajectory, every actor trajectory, the
    spawn poses, whether the cut-in and the block-cut-in each succeeded, and
    which actor performed each.

The output is written to ``experiments/run_<seed>.json`` (override with
``--out``).  Re-running with the same seed reproduces the same spawn layout,
so the only variable across identical seeds is how you drive.

The trial ends automatically once both intents resolve (a short settle after
the cut-in and block commit) or at ``--max-time``; closing the window ends it
early.  The result JSON is written either way.

Usage:
    .venv/bin/python experiment.py                      # random seed, interactive
    .venv/bin/python experiment.py --seed 42            # fixed layout
    .venv/bin/python experiment.py --seed 42 --out out.json
    .venv/bin/python experiment.py --seed 42 --headless # no GUI, ego autopilots

Controls (interactive):
    W / Up      throttle           A / Left   steer left
    S / Down    brake              D / Right  steer right
    (close the window to end the trial early)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

# import (not run) the editor so helper modules share one module identity
import scenario_editor as se
import cutin_orchestrator as co

Pose = Tuple[float, float, float]

# default cut-in intent (ego-relative, fixed arrival time) handed to whichever
# actor the orchestrator casts; the block intent is auto-derived from the ego's
# scripted lane change (see scenario_editor.derive_block_spec).
DEFAULT_CUTIN_SPEC = {"t": 3.0, "along": 6.0, "lat": 0.0,
                      "lc_duration": 2.0, "tail": 4.0}
PALETTE = [(210, 70, 60), (80, 140, 220), (240, 175, 65), (170, 110, 220),
           (90, 190, 110), (230, 120, 180), (120, 210, 200)]

# non-overlap: two cars in the same lane must be at least this far apart (m)
MIN_LANE_GAP = 7.0
# nominal actors cruise straight for the whole trial (must exceed --max-time)
TRIAL_CRUISE_T = 60.0


# --------------------------------------------------------------------------- #
# Randomized scenario construction
# --------------------------------------------------------------------------- #
def _ego_lane_change_lat(ego: se.Actor) -> float:
    """Lateral offset of the ego's first scripted lane change (target lane for
    the block)."""
    for m in ego.maneuvers:
        if getattr(m, "type", None) == "lane_change":
            return float(getattr(m, "lateral_offset", -3.5))
    return -3.5


def _random_spawns(rng: random.Random, m, ego_start: Pose,
                   n: int) -> List[Pose]:
    """`n` non-overlapping lane slots in a band around the ego.  Different
    lanes never overlap laterally; same-lane cars are kept MIN_LANE_GAP apart."""
    lanes = list(range(max(1, m.num_lanes)))
    y_lo = max(ego_start[1] - 20.0, -m.half_length() + 6.0)
    y_hi = min(ego_start[1] + 45.0, m.half_length() - 6.0)
    occupied: List[Tuple[float, float]] = [(ego_start[0], ego_start[1])]
    out: List[Pose] = []
    for _ in range(n):
        for _try in range(300):
            lane = rng.choice(lanes)
            x = m.lane_center_x(lane)
            y = rng.uniform(y_lo, y_hi)
            if all(not (abs(x - ox) < 1.0 and abs(y - oy) < MIN_LANE_GAP)
                   for ox, oy in occupied):
                occupied.append((x, y))
                out.append((x, y, 90.0))
                break
        else:
            break   # gave up placing this one; fewer actors than requested
    return out


def build_random_scenario(base_path: str, seed: int,
                          min_actors: int = 1, max_actors: int = 5
                          ) -> Tuple[se.Scenario, dict]:
    """Load the base (map + scripted ego + intents), then replace its traffic
    with a seeded random fleet and cast the cut-in / block roles by score.

    Returns (scenario, meta) where meta carries the seed, spawn poses, casting
    and the per-actor placement scores."""
    rng = random.Random(seed)
    base = se.load_scenario(base_path)
    ego = next((a for a in base.actors if a.id == "0"), None)
    if ego is None:
        raise SystemExit(f"{base_path}: no ego actor (id 0)")

    cutin_tpl = next((dict(a.cutin) for a in base.actors if a.cutin),
                     dict(DEFAULT_CUTIN_SPEC))
    target_lat = _ego_lane_change_lat(ego)

    n = rng.randint(min_actors, max_actors)
    spawns = _random_spawns(rng, base.map, ego.start, n)

    actors: List[se.Actor] = [ego]
    for i, sp in enumerate(spawns):
        cruise = round(rng.uniform(10.5, 13.0), 2)
        actors.append(se.Actor(
            id=str(i + 1), color=PALETTE[i % len(PALETTE)],
            length=4.5, width=2.0, start=sp, cruise=cruise,
            # seed a straight cruise plan (cut-in / block solvers overwrite
            # theirs); without it a nominal actor has no maneuvers and freezes
            maneuvers=se.cruise_plan(sp, cruise, duration=TRIAL_CRUISE_T)))

    # score every actor for each role at its spawn, cast distinct best actors
    ego_pose = ego.start
    lw = base.map.lane_width
    others = actors[1:]
    cut_scores = {a.id: co.score_cutin_candidate(a.start, ego_pose, lw)
                  for a in others}
    blk_scores = {a.id: co.score_block_candidate(a.start, ego_pose, lw,
                                                 target_lat)
                  for a in others}
    cut_id = block_id = None
    ids = [a.id for a in others]
    if len(ids) == 1:
        a = ids[0]
        if cut_scores[a] >= blk_scores[a]:
            cut_id = a
        else:
            block_id = a
    elif ids:
        cut_id = max(ids, key=lambda i: cut_scores[i])
        block_id = max((i for i in ids if i != cut_id),
                       key=lambda i: blk_scores[i])
    by = {a.id: a for a in others}
    if cut_id is not None:
        by[cut_id].cutin = dict(cutin_tpl)
    if block_id is not None:
        by[block_id].block = {"along": 0.0}   # derive t/duration/lat from ego

    sc = se.Scenario(map=base.map, actors=actors,
                     pixels_per_meter=base.pixels_per_meter)
    sc.simulate()
    se.resolve_cutins(sc)
    se.resolve_blocks(sc)

    meta = {
        "seed": seed,
        "base_scenario": os.path.basename(base_path),
        "num_actors": len(others),
        "target_lat": target_lat,
        "spawns": {a.id: [round(a.start[0], 3), round(a.start[1], 3),
                          round(a.start[2], 2)] for a in others},
        "cruise": {a.id: a.cruise for a in others},
        "cast": {"cutin": cut_id, "block": block_id},
        "scores": {"cutin": {k: round(v, 4) for k, v in cut_scores.items()},
                   "block": {k: round(v, 4) for k, v in blk_scores.items()}},
    }
    return sc, meta


# --------------------------------------------------------------------------- #
# Recorder — fed one snapshot per driving frame by run_gui's on_frame hook
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self, meta: dict, record_hz: float, max_time: float,
                 settle: float = 1.0):
        self.meta = meta
        self.rec_dt = 1.0 / max(1e-3, record_hz)
        self.max_time = max_time
        self.settle = settle
        self.ego_traj: List[list] = []
        self.actor_traj: Dict[str, List[list]] = {}
        self._last_rec = -1e9
        self._last_T = 0.0
        # role assigned at build time → whether we expect a commit at all
        self.has_cutin = meta["cast"]["cutin"] is not None
        self.has_block = meta["cast"]["block"] is not None
        self.cutin = {"performer": meta["cast"]["cutin"], "outcome": None,
                      "success": None, "t_commit": None}
        self.block = {"performer": meta["cast"]["block"], "outcome": None,
                      "success": None, "t_commit": None}
        self._done_since: Optional[float] = None

    def on_frame(self, s: dict) -> bool:
        T = s["T"]
        self._last_T = T
        # sampled trajectories (decimated to record_hz)
        if T - self._last_rec >= self.rec_dt - 1e-9:
            self._last_rec = T
            ex, ey, eh, ev = s["ego"]
            self.ego_traj.append([round(T, 3), round(ex, 3), round(ey, 3),
                                  round(eh, 2), round(ev, 3)])
            for aid, (x, y, h) in s["actors"].items():
                self.actor_traj.setdefault(aid, []).append(
                    [round(T, 3), round(x, 3), round(y, 3), round(h, 2)])

        # track live holders (role may be recast mid-drive)
        if s["cutin_holder"] is not None:
            self.cutin["performer"] = s["cutin_holder"]
        if s["block_holder"] is not None:
            self.block["performer"] = s["block_holder"]

        # finalize each role once the simulator commits it
        if s["cutin_committed"] and self.cutin["t_commit"] is None:
            self.cutin["outcome"] = s["cutin_outcome"]
            self.cutin["success"] = (s["cutin_outcome"] == "merged")
            self.cutin["performer"] = s["cutin_holder"] or self.cutin["performer"]
            self.cutin["t_commit"] = round(T, 3)
        if s["block_committed"] and self.block["t_commit"] is None:
            self.block["outcome"] = s["block_outcome"]
            self.block["success"] = (s["block_outcome"] == "blocked")
            self.block["performer"] = s["block_holder"] or self.block["performer"]
            self.block["t_commit"] = round(T, 3)

        # end conditions: both intents resolved (+settle), or a hard time cap
        cut_done = (not self.has_cutin) or self.cutin["t_commit"] is not None
        blk_done = (not self.has_block) or self.block["t_commit"] is not None
        if cut_done and blk_done:
            if self._done_since is None:
                self._done_since = T
            elif T - self._done_since >= self.settle:
                return False
        if T >= self.max_time:
            return False
        return True

    def result(self) -> dict:
        return {
            **self.meta,
            "record_hz": round(1.0 / self.rec_dt, 3),
            "dt_sim": se.DT,
            "duration_recorded": round(self._last_T, 3),
            "cutin": self.cutin,
            "block": self.block,
            "trajectory_columns": {
                "ego": ["t", "x", "y", "heading_deg", "v"],
                "actor": ["t", "x", "y", "heading_deg"]},
            "ego_trajectory": self.ego_traj,
            "actor_trajectories": self.actor_traj,
        }

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.result(), f, indent=2)


# --------------------------------------------------------------------------- #
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Record cut-in / block trials")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (default: os.urandom-derived)")
    ap.add_argument("--base", default=os.path.join(
        here, "scenarios", "scenario_cutin_block.yaml"),
        help="base scenario (map + scripted ego + intents)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: experiments/run_<seed>.json)")
    ap.add_argument("--hz", type=float, default=20.0,
                    help="trajectory recording rate (Hz)")
    ap.add_argument("--max-time", type=float, default=25.0,
                    help="hard cap on trial length (s)")
    ap.add_argument("--min-actors", type=int, default=1)
    ap.add_argument("--max-actors", type=int, default=5)
    ap.add_argument("--headless", action="store_true",
                    help="no window; ego autopilots its scripted speed")
    args = ap.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    seed = args.seed if args.seed is not None \
        else int.from_bytes(os.urandom(4), "little")
    out = args.out or os.path.join(here, "experiments", f"run_{seed}.json")

    sc, meta = build_random_scenario(args.base, seed,
                                     args.min_actors, args.max_actors)
    print(f"seed={seed}  actors={meta['num_actors']}  "
          f"cast: cut-in={meta['cast']['cutin']} block={meta['cast']['block']}")

    rec = Recorder(meta, record_hz=args.hz, max_time=args.max_time)
    se.run_gui(sc, None, auto_drive=True, on_frame=rec.on_frame)
    rec.write(out)

    r = rec.result()
    print(f"cut-in : performer={r['cutin']['performer']} "
          f"outcome={r['cutin']['outcome']} success={r['cutin']['success']}")
    print(f"block  : performer={r['block']['performer']} "
          f"outcome={r['block']['outcome']} success={r['block']['success']}")
    print(f"wrote {out}  "
          f"({len(r['ego_trajectory'])} ego samples, "
          f"{len(r['actor_trajectories'])} actor tracks)")


if __name__ == "__main__":
    main()
