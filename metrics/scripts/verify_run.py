#!/usr/bin/env python3
"""Verify one or more recorded driving trials against scenario metrics.

Reads experiment JSON (from ``experiment.py`` or ``stress_experiment.py``) and
runs the appropriate verifier from ``scenario_verify.py``.

Usage:
    .venv/bin/python metrics/scripts/verify_run.py experiments/run_42.json
    .venv/bin/python metrics/scripts/verify_run.py --dir experiments/stress -v
    .venv/bin/python metrics/scripts/verify_run.py --setup-only experiments/overtake_1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import scenario_verify as sv

detect_scenario = sv.detect_scenario
verify_hard_brake = sv.verify_hard_brake
verify_hard_brake_setup = sv.verify_hard_brake_setup
verify_overtake = sv.verify_overtake
verify_overtake_setup = sv.verify_overtake_setup
verify_proper_cutin = sv.verify_proper_cutin


def load(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify recorded scenario trajectories")
    ap.add_argument("paths", nargs="*", help="JSON run file(s)")
    ap.add_argument("--dir", help="verify every *.json in this folder")
    ap.add_argument("--setup-only", action="store_true",
                    help="only run stage/setup checks (overtake, hard_brake)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.dir:
        d = os.path.abspath(args.dir)
        paths.extend(sorted(os.path.join(d, n) for n in os.listdir(d)
                            if n.endswith(".json")))

    if not paths:
        ap.error("provide file path(s) or --dir")

    n_ok = 0
    for path in paths:
        data = load(path)
        kind = detect_scenario(data)
        if args.setup_only and kind == "overtake":
            ok, reason, det = verify_overtake_setup(data)
        elif args.setup_only and kind == "hard_brake":
            ok, reason, det = verify_hard_brake_setup(data)
        elif kind == "cutin":
            ok, reason, det = verify_proper_cutin(data)
        elif kind == "overtake":
            ok, reason, det = verify_overtake(data)
        elif kind == "hard_brake":
            ok, reason, det = verify_hard_brake(data)
        else:
            ok, reason, det = False, f"unknown scenario ({kind})", {}

        flag = "OK" if ok else "FAIL"
        print(f"{flag:4}  [{kind:10}]  {os.path.basename(path)}  {reason}")
        if args.verbose and det.get("checks"):
            bits = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in det["checks"].items())
            print(f"       checks: {bits}")
        if ok:
            n_ok += 1

    print(f"\n{ n_ok}/{len(paths)} verified")
    raise SystemExit(0 if n_ok == len(paths) else 1)


if __name__ == "__main__":
    main()
