#!/usr/bin/env python3
"""Where along the road is the sky above the ego lane actually clear?

Deleting the deck has failed repeatedly: it is many meshes, several of them
larger than any safe geometric guard, and hiding the ones I can identify changes
2% of pixels. So stop deleting and start choosing. Render the top view at every
station along the road and report, per station, whether the lane below is
visible — then the scenario can be placed on a clear stretch instead.

Visibility test is semantic, not eyeballed: a semantic camera at the station
reports the tag at the image centre and the fraction of the centre strip that is
Roads/RoadLines at the ego's own elevation. A deck overhead reads as Roads too,
so the discriminator is DEPTH: a depth camera gives the distance to whatever is
under the camera, which is the deck (short) or the road (full height).
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import carla                                                   # noqa: E402
from carla_highway import scenarios as sc_mod                  # noqa: E402
from carla_port.carla_video import default_top_span            # noqa: E402

OUT = "/scratch/zwang179/traffic_orchestration/install/run_output/_probe"
os.makedirs(OUT, exist_ok=True)
PORT = int(os.environ.get("AV_PORT_OVERRIDE", "2000"))
MODE = os.environ.get("PROBE_MODE", "hard_brake")
W, H, FOV = 240, 180, 90.0

client = carla.Client("127.0.0.1", PORT)
client.set_timeout(600.0)
world = client.load_world(os.environ.get("PROBE_TOWN", "Town04"))
st = world.get_settings(); st.synchronous_mode = True; st.fixed_delta_seconds = 0.05
world.apply_settings(st)

frame = sc_mod.discover_frame(world, MODE)
span = default_top_span(frame)
z_off = 0.5 * span * (W / H) / math.tan(math.radians(0.5 * FOV))
lib = world.get_blueprint_library()
print(f"mode={MODE} road={frame.road_id}.{frame.section_id} "
      f"len={frame.length:.0f} span={span:.1f} cam_z_off={z_off:.1f}")


def shoot(kind, tf, conv=None):
    bp = lib.find(kind)
    bp.set_attribute("image_size_x", str(W)); bp.set_attribute("image_size_y", str(H))
    bp.set_attribute("fov", str(FOV))
    cam = world.spawn_actor(bp, tf)
    from queue import Queue
    q = Queue(); cam.listen(q.put)
    for _ in range(3):
        world.tick()
    img = q.get(timeout=20.0); cam.stop(); cam.destroy()
    raw = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((H, W, 4))
    return raw


stations = list(range(-100, 301, 20))
tiles, rows = [], []
for sy in stations:
    loc = frame.to_carla_location(0.0, float(sy), 0.0)
    tf = carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + z_off),
                         carla.Rotation(pitch=-90.0, yaw=frame.to_carla_yaw(90.0)))
    rgb = shoot("sensor.camera.rgb", tf)[:, :, :3][:, :, ::-1].copy()
    dep = shoot("sensor.camera.depth", tf)
    # CARLA depth: normalized = (R + G*256 + B*256^2) / (256^3 - 1), metres = n * 1000
    b, g, r = dep[:, :, 0].astype(float), dep[:, :, 1].astype(float), dep[:, :, 2].astype(float)
    metres = (r + g * 256.0 + b * 65536.0) / 16777215.0 * 1000.0
    cx0, cx1 = W // 2 - 20, W // 2 + 20
    cy0, cy1 = H // 2 - 20, H // 2 + 20
    centre = metres[cy0:cy1, cx0:cx1]
    med = float(np.median(centre))
    # ground is z_off below the camera; a deck is nearer by its clearance
    clear = med > z_off - 4.0
    rows.append((sy, med, clear))
    tiles.append(rgb)
    print(f"  station y={sy:+4d}  centre depth {med:6.1f} m "
          f"(ground at {z_off:.1f})  {'CLEAR' if clear else 'BLOCKED'}")

cols = 5
rowsn = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, H * rowsn), (18, 18, 18))
for i, t in enumerate(tiles):
    sheet.paste(Image.fromarray(t), ((i % cols) * W, (i // cols) * H))
sheet.save(os.path.join(OUT, f"{MODE}_stations.png"))

clear_runs, run = [], []
for sy, _m, ok in rows:
    if ok:
        run.append(sy)
    elif run:
        clear_runs.append((run[0], run[-1])); run = []
if run:
    clear_runs.append((run[0], run[-1]))
print(f"\nclear runs (script y): {clear_runs}")
print(f"wrote {OUT}/{MODE}_stations.png")
