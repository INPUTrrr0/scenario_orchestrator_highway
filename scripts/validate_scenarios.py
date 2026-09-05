#!/usr/bin/env python3
"""Load every scenario this run set uses and print what is in it.

Cheap pre-flight for the GPU jobs: a typo in a YAML should cost a two-minute
CPU job, not nine L40S allocations.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from carla_highway import scenarios as sc_mod  # noqa: E402

CASES = [
    ("cutin", os.path.join(ROOT, "scenarios", "scenario_cutin_single.yaml")),
    ("cutin", None),          # the four-actor original, as a control
    ("hard_brake", None),
    ("overtake", None),
]

rc = 0
for mode, path in CASES:
    label = os.path.basename(path) if path else sc_mod.spec(mode).yaml
    try:
        sc = sc_mod.load(mode, path)
        ego, bg = sc_mod.split_ego(sc)
        spec = sc_mod.spec(mode)
        cutins = [a.id for a in bg.actors if getattr(a, "cutin", None)]
        print(f"[ok] {mode:11s} {label}")
        print(f"       lanes={spec.lanes} two_way={spec.two_way} "
              f"casting={spec.casting} duration={spec.duration}s xodr={spec.xodr or '-'}")
        print(f"       ego={ego.id}  background={[a.id for a in bg.actors]}  "
              f"cutin holders={cutins}")
    except Exception as exc:                       # noqa: BLE001
        rc = 1
        print(f"[FAIL] {mode:11s} {label}: {type(exc).__name__}: {exc}")

# the generated map overtake needs
xodr = os.path.join(ROOT, "maps", sc_mod.spec("overtake").xodr + ".xodr")
print(f"[{'ok' if os.path.isfile(xodr) else 'FAIL'}] xodr {xodr}")
rc = rc or (0 if os.path.isfile(xodr) else 1)
sys.exit(rc)
