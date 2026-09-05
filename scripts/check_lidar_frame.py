#!/usr/bin/env python3
"""Does the LiDAR sweep agree with the ground-truth object positions?

CARLA's LiDAR is left-handed: many CARLA codebases flip y before rasterising.
If this port does not and `lead` expects flipped points, the BEV is mirrored
laterally and every left/right judgement the planner makes is wrong.

The dump carries both the sweep and `objects` (ego-frame ground truth), so the
question is answerable arithmetically: cluster the returns near each object's
range and see whether the lateral offset matches y or -y.
"""
import glob
import json
import os
import sys

import numpy as np

dump = sys.argv[1]
step = sys.argv[2] if len(sys.argv) > 2 else None
paths = sorted(glob.glob(os.path.join(dump, "*.json")))
if step:
    paths = [p for p in paths if f"step{int(step):04d}" in p] or paths[:1]
meta = json.load(open(paths[0]))
arr = dict(np.load(paths[0][:-5] + ".npz"))
pts = arr.get("sensor.lidar")
if pts is None:
    raise SystemExit("no lidar in this dump")

print(f"{os.path.basename(paths[0])}: {pts.shape[0]} points, columns={pts.shape[1]}")
print(f"  x {pts[:,0].min():7.1f}..{pts[:,0].max():7.1f}   "
      f"y {pts[:,1].min():7.1f}..{pts[:,1].max():7.1f}   "
      f"z {pts[:,2].min():7.1f}..{pts[:,2].max():7.1f}")

objs = meta.get("objects") or []
print("\nground-truth objects (ego frame: x forward, y right):")
for o in objs:
    print(f"  {o['type']:>4} id={o['id']} at x={o['position'][0]:6.2f} "
          f"y={o['position'][1]:6.2f}")

# For each object, take the lidar returns in a band around its longitudinal
# range and above the road, then compare their mean lateral offset with +y/-y.
above = pts[(pts[:, 2] > -1.6) & (pts[:, 2] < 0.5)]
print("\nlateral agreement (mean y of returns near each object's range):")
for o in objs:
    ox, oy = float(o["position"][0]), float(o["position"][1])
    band = above[np.abs(above[:, 0] - ox) < 2.0]
    # discard the road surface: keep the lateral cluster nearest |oy|
    cand = band[np.abs(band[:, 1]) < 8.0]
    if len(cand) < 20:
        print(f"  object {o['id']}: only {len(cand)} returns in band; inconclusive")
        continue
    # two clusters: sign-split, report whichever holds the most returns
    pos = cand[cand[:, 1] > 0][:, 1]
    neg = cand[cand[:, 1] < 0][:, 1]
    dom = "+y" if len(pos) >= len(neg) else "-y"
    m = float(np.median(pos)) if dom == "+y" else float(np.median(neg))
    verdict = "MATCHES" if (m > 0) == (oy > 0) else "MIRRORED"
    print(f"  object {o['id']} truth y={oy:+6.2f} | returns {len(pos):4d} at +y, "
          f"{len(neg):4d} at -y, dominant {dom} median {m:+6.2f}  -> {verdict}")
