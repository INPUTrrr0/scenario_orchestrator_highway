#!/usr/bin/env python3
"""How much of a revolution does one captured LiDAR frame actually cover?

CARLA accumulates LiDAR returns over simulated time and emits them on the
sensor tick. If `rotation_frequency` is lower than the world tick rate, each
frame carries only the fraction of a revolution that elapsed — a wedge, not a
sweep — and the model sees most of the scene as empty.
"""
import glob, json, os, sys
import numpy as np

dump = sys.argv[1]
for path in sorted(glob.glob(os.path.join(dump, "*.json"))):
    arr = dict(np.load(path[:-5] + ".npz"))
    pts = arr.get("sensor.lidar")
    if pts is None:
        continue
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    az = (az + 360.0) % 360.0
    hist, _ = np.histogram(az, bins=36, range=(0, 360))
    covered = int((hist > 0).sum())
    print(f"{os.path.basename(path):28s} n={pts.shape[0]:6d} "
          f"azimuth bins occupied {covered:2d}/36 = {covered/36*100:5.1f}% "
          f"span {az.min():6.1f}..{az.max():6.1f} deg")
    if path == sorted(glob.glob(os.path.join(dump, "*.json")))[0]:
        occ = "".join("#" if h else "." for h in hist)
        print(f"    azimuth occupancy (0->360 deg): {occ}")
