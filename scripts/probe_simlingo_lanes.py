#!/usr/bin/env python3
"""Ask SimLingo, in its own training vocabulary, what it sees and what it will do.

    python scripts/probe_simlingo_lanes.py <dump_dir> [--steps 60,120,200]

The question this answers is "why does simlingo not change lanes when the car in
front slows down". That splits into three separable ones, and this script asks
each of them against the *same* captured frame, so a difference in the answers
is a difference in the question and not in the situation:

  1. Can it see the adjacent lane?  Asked as DriveLM VQA, using the exact
     question strings from `data/evalset_vqa.json` — the ones the checkpoint was
     trained and evaluated on. Out-of-distribution phrasings measure prompt luck,
     not perception, so every probe here is copied verbatim.
  2. Can it execute a lane change at all?  Asked as an Action Dreaming
     instruction (`<INSTRUCTION_FOLLOWING>` + a `dreamer.json` lanechange_rel
     template), which is the interface the paper documents for this.
  3. Does it follow the route when the route changes lanes?  Asked by bending
     the observation's route into the next lane and re-reading the plan. This is
     what the privileged route planner does in training
     (`privileged_route_planner.shift_route_smoothly`), so it is the mechanism
     the checkpoint actually learned lane changes from.

Prompt shape follows `simlingo_training/dataloader/dataset_driving.py`:
driving is `<SAFETY> Current speed: ... Target waypoint: ... What should the ego
do next?`, VQA is the same head with `Q: <question>` and NO dreamer prefix, and
instructions carry `<INSTRUCTION_FOLLOWING>`. The mode prefix and the task are
the only things that move between conditions.

Nothing here is a metric of driving quality. It is a read of the model's own
beliefs and plans, printed next to each other.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from replay_obs import build  # noqa: E402  (same policy construction as replay)

# Verbatim from data/evalset_vqa.json. Do not paraphrase: the checkpoint was
# trained on these strings and the answer vocabulary is closed.
VQA = [
    ("lanes_same_dir",
     "How many lanes are there in the same direction as the ego car?"),
    ("ego_lane",
     "On which lane is the ego vehicle (left most lane of the lanes going in "
     "the same direction is indicated with 0)?"),
    ("allowed_dir",
     "In which direction is the ego car allowed to change lanes?"),
    ("need_change",
     "Does the ego vehicle need to change lanes or deviate from the lane "
     "center due to an upcoming obstruction?"),
    ("watch_left",
     "The ego vehicle wants to do a lane change to the left. Which lanes are "
     "important to watch out for?"),
    ("watch_right",
     "The ego vehicle wants to do a lane change to the right. Which lanes are "
     "important to watch out for?"),
]

# dreamer.json -> "lanechange_rel", first template, placeholders filled.
INSTR = [("instr_left", "Shift one lane to the left."),
         ("instr_right", "Shift one lane to the right.")]

LANE_WIDTH = 3.5


def load_steps(dump_dir, wanted):
    out = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "*.json"))):
        meta = json.load(open(path))
        step = int(meta.get("_step", -1))
        if wanted and step not in wanted:
            continue
        arrays = dict(np.load(path[:-5] + ".npz"))
        obs = {k: v for k, v in meta.items() if not k.startswith("_")}
        cameras = {k.split(".", 1)[1]: v for k, v in arrays.items()
                   if k.startswith("sensor.")}
        if cameras:
            obs["sensor"] = {"cameras": cameras}
        out.append((step, obs, meta))
    return out


def bend(route, width):
    """Bend a route into the neighbouring lane, the way the expert's planner does.

    `shift_route_smoothly` ramps the lateral offset over a transition rather
    than stepping it, so a step function here would be the one shape training
    never contained. A smoothstep over the sampled length reaches a full lane by
    the far point, which is what a 2 s lane change looks like at 11 m/s.
    """
    pts = [(float(p[0]), float(p[1])) for p in route]
    if not pts:
        return pts
    span = max(1e-6, pts[-1][0] - pts[0][0])
    out = []
    for x, y in pts:
        u = min(1.0, max(0.0, (x - pts[0][0]) / span))
        out.append([x, y + width * (u * u * (3 - 2 * u))])
    return out


def plan(action):
    """Lateral reach of the model's own predicted path, in metres."""
    route = action.get("route") or []
    if not route:
        return None, None
    lat = [float(p[1]) for p in route]
    end = lat[-1]
    peak = max(lat, key=abs)
    return end, peak


def run(policy, obs, mode, task):
    policy.reset()          # PID history, so steer is comparable across conditions
    policy.mode_token = mode
    policy.instruction = task
    action = policy.act(obs)
    end, peak = plan(action)
    ctrl = action.get("control") or {}
    return {"language": action.get("meta", {}).get("language"),
            "lat_end": end, "lat_peak": peak,
            "steer": ctrl.get("steer"), "throttle": ctrl.get("throttle"),
            "brake": ctrl.get("brake")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--steps", default="",
                    help="comma-separated step indices; default every dump")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()

    wanted = {int(v) for v in args.steps.split(",") if v.strip()}
    frames = load_steps(args.dump_dir, wanted)
    if not frames:
        raise SystemExit(f"no dumps in {args.dump_dir}")
    print(f"[probe] {len(frames)} frame(s) from {args.dump_dir}")

    policy = build("simlingo")
    policy.load()

    results = []
    for step, obs, meta in frames:
        speed = float((obs.get("ego") or {}).get("speed_mps") or 0.0)
        route = obs.get("route") or []
        route_lat = float(route[-1][1]) if route else float("nan")
        n_obj = len(obs.get("objects") or [])
        lead = None
        for o in (obs.get("objects") or []):
            pos = o.get("position") or [0, 0, 0]
            if abs(float(pos[1])) < 2.0 and float(pos[0]) > 0:
                if lead is None or float(pos[0]) < lead[0]:
                    lead = (float(pos[0]), float(o.get("speed_mps") or 0.0))

        print("\n" + "=" * 78)
        print(f"step {step}   speed {speed:5.2f} m/s   objects {n_obj}   "
              f"route far-lateral {route_lat:+.2f} m"
              + (f"   lead {lead[0]:.1f} m @ {lead[1]:.1f} m/s" if lead else
                 "   no lead in lane"))
        print("=" * 78)

        row = {"step": step, "speed": speed, "route_far_lateral": route_lat,
               "lead": lead, "conditions": {}}

        def record(name, mode, task, label):
            r = run(policy, obs, mode, task)
            row["conditions"][name] = dict(r, prompt_mode=mode, task=task)
            lat = ("   plan lat %+.2f m (peak %+.2f)"
                   % (r["lat_end"], r["lat_peak"])) if r["lat_end"] is not None else ""
            print(f"\n  [{label}]{lat}")
            print(f"    steer {r['steer']:+.3f}  throttle {r['throttle']:.2f}  "
                  f"brake {r['brake']:.2f}")
            print(f"    says: {r['language']}")

        record("drive", "<SAFETY>", "", "DRIVING  (what it actually does)")
        for key, question in VQA:
            record(key, "", f"Q: {question}", f"VQA {key}")
        for key, text in INSTR:
            record(key, "<INSTRUCTION_FOLLOWING>", text, f"INSTRUCTION {key}")

        # Route conditioning: same frame, route bent into each neighbour lane.
        for key, width in (("route_left", -LANE_WIDTH),
                           ("route_right", +LANE_WIDTH)):
            bent = dict(obs)
            bent["route"] = bend(route, width)
            policy.reset()
            policy.mode_token, policy.instruction = "<SAFETY>", ""
            action = policy.act(bent)
            end, peak = plan(action)
            ctrl = action.get("control") or {}
            row["conditions"][key] = {
                "language": action.get("meta", {}).get("language"),
                "lat_end": end, "lat_peak": peak, "steer": ctrl.get("steer"),
                "throttle": ctrl.get("throttle"), "brake": ctrl.get("brake"),
                "route_far_lateral": bent["route"][-1][1] if bent["route"] else None}
            row["conditions"][key]["route_far_lateral"] = (
                bent["route"][-1][1] if bent["route"] else None)
            far = bent["route"][-1][1] if bent["route"] else float("nan")
            lat = (f"   plan lat {end:+.2f} m (peak {peak:+.2f})"
                   if end is not None else "   plan: none")
            print(f"\n  [ROUTE BENT {key.split('_')[1]}  "
                  f"(route far-lateral {far:+.2f} m)]" + lat)
            print(f"    steer {ctrl.get('steer'):+.3f}  "
                  f"throttle {ctrl.get('throttle'):.2f}  brake {ctrl.get('brake'):.2f}")
            print(f"    says: {action.get('meta', {}).get('language')}")

        results.append(row)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(results, open(args.json, "w"), indent=1)
        print(f"\n[probe] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
