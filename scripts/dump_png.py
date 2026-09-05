#!/usr/bin/env python3
"""Save the camera arrays from an observation dump as PNGs, so they can be looked at."""
import glob, os, sys
import numpy as np
from PIL import Image
dump, out = sys.argv[1], sys.argv[2]
step = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
paths = sorted(glob.glob(os.path.join(dump, "*.npz")))
if step:
    paths = [p for p in paths if f"step{int(step):04d}" in p] or paths[:1]
for p in paths[:1]:
    for key, arr in np.load(p).items():
        if not key.startswith("sensor.") or arr.ndim != 3 or arr.shape[2] != 3:
            continue
        name = key.split(".", 1)[1]
        Image.fromarray(arr.astype(np.uint8)).save(os.path.join(out, f"{name}.png"))
        print(f"{name}: {arr.shape} dtype={arr.dtype} "
              f"meanRGB=({arr[...,0].mean():.1f},{arr[...,1].mean():.1f},{arr[...,2].mean():.1f})")
