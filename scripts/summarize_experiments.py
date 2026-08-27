#!/usr/bin/env python3
"""Summarize experiment JSON files by intention (cut-in, block).

Reads every ``*.json`` in the experiments folder and, for each intention,
reports how many trials included that intention and how many succeeded.

Cut-in success is reported twice:
  * **recorded** — ``cutin.success == true`` from the simulator
  * **verified** — a *proper* cut-in: all four checks below pass
    (see ``verify_proper_cutin``).  Trials where the performer already shared
    the ego's lane at the start of the cut-in are excluded from the verified
    denominator.

------------------------------------------------------------------------
Cut-in verifier (all four required for verified success)
------------------------------------------------------------------------
1. **Station** — at commit, the intention-owner actor is ahead of the ego
   and within ``MAX_AHEAD_M`` (10 m) along-track.
2. **Adjacent at start** — at the beginning of the cut-in (first trajectory
   samples / spawn), the actor is in an adjacent lane relative to the ego
   (not already in the ego's lane).
3. **Lane-change + ahead** — the actor performed a lane change into the
   ego's original home lane and is ahead of the ego at commit.  The ego must
   still be in that home lane when the actor merges (ego leaving first /
   moving into the actor's spawn lane first does not count).
4. **Nominal speed after** — after the lane change, the actor continues at a
   nominal driving speed: close to the ego's contemporaneous speed (post-merge
   matching) and/or its own cruise value — not abandoned / crawling.

Block success is reported from the recorded ``block.success`` field only
(no geometric re-verification yet).

Overtake / hard-brake runs (from ``stress_experiment.py``) are verified with
``scripts/verify_run.py`` or included when ``--dir`` also contains stress JSON.

Usage:
    .venv/bin/python scripts/summarize_experiments.py
    .venv/bin/python scripts/summarize_experiments.py --dir experiments
    .venv/bin/python scripts/summarize_experiments.py --verbose
    .venv/bin/python scripts/verify_run.py experiments/stress/overtake_0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import scenario_verify as sv  # noqa: E402

verify_proper_cutin = sv.verify_proper_cutin
verify_overtake = sv.verify_overtake
verify_hard_brake = sv.verify_hard_brake
detect_scenario = sv.detect_scenario
infer_lane_width = sv.infer_lane_width


INTENTIONS = ("cutin", "block")
LABELS = {"cutin": "cut-in", "block": "block cut-in"}
STRESS_SCENARIOS = ("overtake", "hard_brake")


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


def summarize_dir(experiments_dir: str
                  ) -> Tuple[Dict[str, Dict[str, int]], List[dict]]:
    """Return ({intention: counts}, per-file records)."""
    totals = {k: {"runs": 0, "success": 0, "fail": 0, "pending": 0} for k in INTENTIONS}
    cutin_verify = {
        "proper_runs": 0,           # cast + adjacent at start
        "spawn_same_lane": 0,       # cast but already on ego lane at start
        "verified_success": 0,      # all 4 checks pass
        "recorded_not_verified": 0, # recorded success but failed verification
    }
    stress_verify = {k: {"runs": 0, "verified": 0} for k in STRESS_SCENARIOS}
    records: List[dict] = []

    if not os.path.isdir(experiments_dir):
        raise SystemExit(f"not a directory: {experiments_dir}")

    paths: List[str] = []
    for root, _dirs, files in os.walk(experiments_dir):
        for name in files:
            if name.endswith(".json"):
                paths.append(os.path.join(root, name))
    paths.sort()
    if not paths:
        raise SystemExit(f"no *.json files under {experiments_dir}")

    for path in paths:
        data = load_run(path)
        if data is None:
            continue
        name = os.path.basename(path)
        rec = {"path": path, "name": name}
        kind = detect_scenario(data)

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
            verified, reason, details = verify_proper_cutin(data, lw)
            rec["cutin_verified"] = verified
            rec["cutin_verify_reason"] = reason
            rec["cutin_verify_details"] = details

            if reason == "already on ego lane at start of cut-in":
                cutin_verify["spawn_same_lane"] += 1
            else:
                cutin_verify["proper_runs"] += 1

            if verified:
                cutin_verify["verified_success"] += 1
            elif intention_success(data, "cutin") is True:
                cutin_verify["recorded_not_verified"] += 1
                rec["cutin_mismatch"] = True

        if kind in STRESS_SCENARIOS:
            stress_verify[kind]["runs"] += 1
            fn = verify_overtake if kind == "overtake" else verify_hard_brake
            ok, reason, details = fn(data)
            rec[f"{kind}_verified"] = ok
            rec[f"{kind}_verify_reason"] = reason
            rec[f"{kind}_verify_details"] = details
            if ok:
                stress_verify[kind]["verified"] += 1

        records.append(rec)

    return {**totals, "cutin_verify": cutin_verify,
            "stress_verify": stress_verify}, records


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
        print("Cut-in verification (need all 4):")
        print("  1. owner within 10 m ahead of ego at commit")
        print("  2. actor in adjacent lane at start of cut-in")
        print("  3. actor lane-changed into ego lane and is ahead")
        print("  4. actor at nominal driving speed after the merge")
        pr = cv["proper_runs"]
        vs = cv["verified_success"]
        if pr:
            vrate = 100.0 * vs / pr
            print(f"  proper runs (adjacent at start):         {pr:4d}")
            print(f"  excluded (already on ego lane at start): {cv['spawn_same_lane']:4d}")
            print(f"  verified success (4/4):                  {vs:4d} / {pr:4d}  ({vrate:5.1f}%)")
        else:
            print("  no proper cut-in runs (all performers started on the ego lane)")
        if cv["recorded_not_verified"]:
            print(f"  recorded success but NOT verified:       {cv['recorded_not_verified']:4d}")

    sv_stress = summary.get("stress_verify") or {}
    for kind in STRESS_SCENARIOS:
        st = sv_stress.get(kind) or {"runs": 0, "verified": 0}
        if not st["runs"]:
            continue
        print()
        label = kind.replace("_", " ")
        rate = 100.0 * st["verified"] / st["runs"]
        print(f"{label:14}  verified success: {st['verified']:4d} / {st['runs']:4d} runs  ({rate:5.1f}%)")
        print(f"  (see docs/SCENARIOS_AND_VALIDATION.md for criteria)")

    if verbose:
        print("\nPer-file verification:")
        for rec in summary.get("_records", []):
            kind = None
            if "cutin_verified" in rec:
                kind = "cutin"
                verified = rec["cutin_verified"]
                reason = rec["cutin_verify_reason"]
                details = rec.get("cutin_verify_details") or {}
            elif "overtake_verified" in rec:
                kind = "overtake"
                verified = rec["overtake_verified"]
                reason = rec["overtake_verify_reason"]
                details = rec.get("overtake_verify_details") or {}
            elif "hard_brake_verified" in rec:
                kind = "hard_brake"
                verified = rec["hard_brake_verified"]
                reason = rec["hard_brake_verify_reason"]
                details = rec.get("hard_brake_verify_details") or {}
            else:
                continue
            flag = "OK" if verified else reason
            extra = ""
            if rec.get("cutin_mismatch"):
                extra = "  ** recorded success, failed verification **"
            checks = details.get("checks") or {}
            if checks:
                extra += "  [" + " ".join(
                    f"{k.split('_', 1)[0]}={'Y' if v else 'n'}" for k, v in checks.items()) + "]"
            print(f"  {rec['name']:24}  [{kind}]  {flag}{extra}")


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
