#!/usr/bin/env python3
"""Replay a captured observation through a policy. No CARLA, no simulation.

    python scripts/replay_obs.py <dump_dir> [--expect drive|brake] [--step N]

A dump is what `carla_port.ego_driver._dump` writes under `$AV_DUMP_OBS`: one
`.npz` of the sensor arrays and one `.json` of everything else, per decision.
This rebuilds the observation and calls `policy.act(...)` on it, so the question
"why does this policy brake here?" becomes a two-second offline call instead of
a ninety-second GPU job.

Exit status is the point: 0 when the policy's control matches `--expect`, 1 when
it does not. That makes it a red/green loop a bisect or a fix can be driven
against.
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


def load(dump_dir: str, step: int | None):
    pairs = sorted(glob.glob(os.path.join(dump_dir, "*.json")))
    if not pairs:
        raise SystemExit(f"no dumps in {dump_dir}")
    if step is not None:
        pairs = [p for p in pairs if f"step{step:04d}" in p] or pairs[:1]
    path = pairs[0]
    meta = json.load(open(path))
    arrays = dict(np.load(path[:-5] + ".npz"))
    obs = {k: v for k, v in meta.items() if not k.startswith("_")}
    cameras = {k.split(".", 1)[1]: v for k, v in arrays.items()
               if k.startswith("sensor.")}
    if cameras:
        obs["sensor"] = {"cameras": cameras}
    if "bev" in arrays:
        obs["bev"] = {"semantic_classes": arrays["bev"]}
    return meta, obs, path


def build(policy_name: str):
    """Load the policy exactly as the runner does, without a world."""
    # `scenario_orchestration/policies.py` imports `contract` bare, so its own
    # directory has to be on sys.path — the runner gets this for free from how
    # it is launched; a standalone script does not.
    sys.path.insert(0, os.path.join(ROOT, "scenario_orchestration"))
    from scenario_orchestration import policies as pol_mod
    from scenario_orchestration.contract import PolicyRequest
    from carla_highway.runner import POLICY_SHORTCUTS, REPO_ROOT
    spec = dict(POLICY_SHORTCUTS[policy_name])
    root = os.environ.get(f"{policy_name.upper()}_ROOT") or os.path.join(
        os.path.dirname(REPO_ROOT), "third_party", policy_name)
    req = PolicyRequest(**spec, parameters={"repository_path": root})
    loaded = pol_mod.load_policy(req, harness_root=None, repo_root=REPO_ROOT)
    return loaded.policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--expect", choices=["drive", "brake", "any"], default="any",
                    help="drive: throttle>0.1 and brake<0.5. brake: the reverse.")
    ap.add_argument("--swap-route-xy", action="store_true", dest="swap_route_xy",
                    help="transpose each route point from (forward, lateral) to "
                         "(lateral, forward). simlingo's target points come from "
                         "inverse_conversion_2d with a compass pre-rotated by "
                         "-90 deg, which puts FORWARD in index 1 — the opposite "
                         "of the port's convention.")
    ap.add_argument("--paint-out", default=None, dest="paint_out",
                    help="y0,y1,x0,x1 as fractions: fill that box in every "
                         "camera with the median colour of the strip just "
                         "above it. The probe for 'is the policy reacting to "
                         "the ego's own bodywork?' — the image is the only "
                         "thing that changes.")
    ap.add_argument("--route-scale", default=None, dest="route_scale",
                    help="comma-separated factors to scale the route's "
                         "longitudinal distances by, re-running act() for each. "
                         "The probe for 'is the target point too close?': the "
                         "image, the speed and everything else are held fixed "
                         "and only the route horizon moves. Straight-line "
                         "scaling only — it would distort a curved route.")
    ap.add_argument("--max-target-speed", type=float, default=None,
                    dest="max_target_speed",
                    help="fail if the policy's target speed exceeds this "
                         "(m/s). The tfv6 symptom: it targets 18.8 m/s while "
                         "closing on a slow lead in a 30 kph zone")
    ap.add_argument("--repeat", type=int, default=1,
                    help="call act() N times to check determinism")
    args = ap.parse_args()

    meta, obs, path = load(args.dump_dir, args.step)
    name = args.policy or meta.get("_policy")
    print(f"[replay] {os.path.basename(path)}  policy={name}")
    print(f"[replay] ego={obs.get('ego')}")
    route = obs.get("route") or []
    print(f"[replay] route: {len(route)} pts, first={route[0] if route else None} "
          f"last={route[-1] if route else None}")
    cams = list((obs.get('sensor') or {}).get('cameras', {}))
    print(f"[replay] sensor arrays: {cams}")
    print(f"[replay] recorded action: {meta.get('_action')}")

    policy = build(name)
    if hasattr(policy, "load"):
        policy.load()
    if hasattr(policy, "reset"):
        policy.reset()

    if args.swap_route_xy:
        obs["route"] = [[p[1], p[0]] for p in (obs.get("route") or [])]
        print(f"[replay] route transposed; index 7 now {obs['route'][7]}")

    if args.paint_out:
        y0, y1, x0, x1 = [float(v) for v in args.paint_out.split(",")]
        for name, img in ((obs.get("sensor") or {}).get("cameras") or {}).items():
            if getattr(img, "ndim", 0) != 3 or img.shape[2] != 3:
                continue
            h, w = img.shape[:2]
            r0, r1 = int(h * y0), int(h * y1)
            c0, c1 = int(w * x0), int(w * x1)
            ref = img[max(0, r0 - int(h * 0.06)):r0, c0:c1]
            fill = (np.median(ref.reshape(-1, 3), axis=0).astype(img.dtype)
                    if ref.size else np.array([160, 160, 160], dtype=img.dtype))
            img[r0:r1, c0:c1] = fill
            print(f"[replay] painted {name}[{r0}:{r1}, {c0}:{c1}] with "
                  f"RGB{tuple(int(v) for v in fill)}")

    if args.route_scale:
        import copy
        base_route = [list(p) for p in (obs.get("route") or [])]
        print(f"[replay] sweeping route scales; base target point (index 7) = "
              f"{base_route[7] if len(base_route) > 7 else None}")
        for factor in [float(v) for v in args.route_scale.split(",") if v.strip()]:
            probe = copy.deepcopy(obs)
            probe["route"] = [[p[0] * factor, p[1] * factor] for p in base_route]
            tp = probe["route"][7] if len(probe["route"]) > 7 else None
            act_out = policy.act(probe)
            c = act_out.get("control", {})
            thr = float(c.get("throttle", 0.0)); brk = float(c.get("brake", 0.0))
            lang = (act_out.get("meta") or {}).get("language")
            tgt = act_out.get("target_speed_mps")
            print(f"[replay] scale x{factor:<4} tp={tp[0]:6.1f} m -> "
                  f"throttle={thr:.2f} brake={brk:.2f} "
                  f"{'DRIVES' if (thr > 0.1 and brk < 0.5) else 'STOPPED'}"
                  + (f" tgt={float(tgt):.2f}" if tgt is not None else ""))
            if lang:
                print(f"            language: {lang!r}")
        return 0

    controls = []
    for _ in range(max(1, args.repeat)):
        action = policy.act(obs)
        c = action.get("control", {})
        controls.append((round(float(c.get("throttle", 0.0)), 4),
                         round(float(c.get("steer", 0.0)), 4),
                         round(float(c.get("brake", 0.0)), 4)))
    for i, c in enumerate(controls):
        print(f"[replay] act#{i}: throttle={c[0]} steer={c[1]} brake={c[2]}")
    meta_out = (action.get("meta") or {})
    if meta_out.get("language"):
        print(f"[replay] language: {meta_out['language']!r}")
    if action.get("target_speed_mps") is not None:
        print(f"[replay] target_speed_mps: {action['target_speed_mps']}")
    if len(set(controls)) > 1:
        print("[replay] WARNING: act() is not deterministic across calls")

    thr, _st, brk = controls[0]
    drives = thr > 0.1 and brk < 0.5
    print(f"[replay] verdict: {'DRIVES' if drives else 'STOPPED'}")

    if args.max_target_speed is not None:
        tgt = action.get("target_speed_mps")
        if tgt is None:
            print("[replay] no target_speed_mps in the action; cannot check")
            return 1
        ok = float(tgt) <= args.max_target_speed
        print(f"[replay] target speed {float(tgt):.2f} <= "
              f"{args.max_target_speed:.2f}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    if args.expect == "drive":
        ok = drives
    elif args.expect == "brake":
        ok = not drives
    else:
        return 0
    print(f"[replay] expected {args.expect}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
