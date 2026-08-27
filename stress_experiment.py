#!/usr/bin/env python3
"""Record overtake / hard_brake driving trials to JSON for verification.

Loads a stress-test scenario YAML, drops you into Drive mode, and writes
trajectories plus role metadata for ``scripts/verify_run.py``.

Usage:
    .venv/bin/python stress_experiment.py scenarios/scenario_overtake.yaml
    .venv/bin/python stress_experiment.py scenarios/scenario_hard_brake.yaml \\
        --seed 1 --headless --max-time 25 --out experiments/stress/overtake_1.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import scenario_editor as se


class StressRecorder:
    def __init__(self, meta: dict, record_hz: float, max_time: float):
        self.meta = meta
        self.rec_dt = 1.0 / max(1e-3, record_hz)
        self.max_time = max_time
        self.ego_traj: List[list] = []
        self.actor_traj: Dict[str, List[list]] = {}
        self._last_rec = -1e9
        self._last_T = 0.0

    def on_frame(self, s: dict) -> bool:
        T = s["T"]
        self._last_T = T
        if T - self._last_rec >= self.rec_dt - 1e-9:
            self._last_rec = T
            ex, ey, eh, ev = s["ego"]
            self.ego_traj.append([round(T, 3), round(ex, 3), round(ey, 3),
                                  round(eh, 2), round(ev, 3)])
            for aid, (x, y, h) in s["actors"].items():
                self.actor_traj.setdefault(aid, []).append(
                    [round(T, 3), round(x, 3), round(y, 3), round(h, 2)])
        if T >= self.max_time:
            return False
        return True

    def result(self) -> dict:
        return {
            **self.meta,
            "record_hz": round(1.0 / self.rec_dt, 3),
            "dt_sim": se.DT,
            "duration_recorded": round(self._last_T, 3),
            "trajectory_columns": {
                "ego": ["t", "x", "y", "heading_deg", "v"],
                "actor": ["t", "x", "y", "heading_deg"],
            },
            "ego_trajectory": self.ego_traj,
            "actor_trajectories": self.actor_traj,
        }

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.result(), f, indent=2)


def _scenario_kind(roles: Dict[str, str]) -> str:
    vals = set(roles.values())
    if "blocker" in vals or "oncoming" in vals:
        return "overtake"
    if "slow" in vals or "adjacent" in vals:
        return "hard_brake"
    return "stress"


def _actor_cruise(a: se.Actor) -> Optional[float]:
    if a.cruise is not None:
        return float(a.cruise)
    for m in a.maneuvers:
        if getattr(m, "type", None) == "go_straight":
            return float(m.intercept)
    return None


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Record overtake / hard_brake trials")
    ap.add_argument("scenario", help="YAML scenario path")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0, help="metadata only")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    sc = se.load_scenario(args.scenario)
    ego = next(a for a in sc.actors if a.id == "0")
    others = [a for a in sc.actors if a.id != "0"]
    roles = {a.id: a.role for a in others if a.role}
    inv_roles = {v: k for k, v in roles.items()}
    kind = _scenario_kind(roles)

    base = os.path.splitext(os.path.basename(args.scenario))[0]
    out = args.out or os.path.join(here, "experiments", "stress",
                                   f"{base}_{args.seed}.json")

    meta = {
        "seed": args.seed,
        "scenario": kind,
        "base_scenario": os.path.basename(args.scenario),
        "lane_width": sc.map.lane_width,
        "roles": inv_roles,
        "actor_roles": roles,
        "spawns": {a.id: [round(a.start[0], 3), round(a.start[1], 3),
                          round(a.start[2], 2), a.length, a.width]
                   for a in others},
        "cruise": {a.id: v for a in others if (v := _actor_cruise(a)) is not None},
    }

    rec = StressRecorder(meta, record_hz=args.hz, max_time=args.max_time)
    se.run_gui(sc, None, auto_drive=True, on_frame=rec.on_frame)
    rec.write(out)
    r = rec.result()
    print(f"scenario={kind}  duration={r['duration_recorded']:.1f}s  wrote {out}")


if __name__ == "__main__":
    main()
