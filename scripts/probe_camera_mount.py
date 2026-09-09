#!/usr/bin/env python3
"""Photograph the ego's forward camera from several mounts, on the real hero body.

The adapter mounts simlingo's camera at the position its config names,
`(-1.5, 0.0, 2.0)`, and the captured frame is a third full of the ego's own
bodywork — on the audi.tt it was roof and bonnet, on the leaderboard hero it is
the cabin interior. Rather than guess a replacement, render candidates and look.
"""
import math, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import carla                                                   # noqa: E402


def _probe_out_dir() -> str:
    env = os.environ.get("AV_INSTALL") or os.environ.get("AV_ROOT")
    if env:
        base = env if env.rstrip("/").endswith("install") else os.path.join(env, "install")
        return os.path.join(base, "run_output", "_probe", "mounts")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = here
    while True:
        if os.path.isdir(os.path.join(d, "install")):
            return os.path.join(d, "install", "run_output", "_probe", "mounts")
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.join(here, "run_output", "_probe", "mounts")
        d = parent


OUT = _probe_out_dir()
os.makedirs(OUT, exist_ok=True)
PORT = int(os.environ.get("AV_PORT_OVERRIDE", "2000"))
HERO = "vehicle.lincoln.mkz_2020"
W, H, FOV = 1024, 512, 110.0
MOUNTS = [(-1.5, 2.0), (-1.5, 2.4), (0.0, 2.0), (0.8, 1.6),
          (1.3, 1.6), (2.0, 1.4), (2.4, 1.2)]

client = carla.Client("127.0.0.1", PORT); client.set_timeout(300.0)
world = client.load_world("Town04")
st = world.get_settings(); st.synchronous_mode = True; st.fixed_delta_seconds = 1/60
world.apply_settings(st)
lib = world.get_blueprint_library()

bp = lib.find(HERO)
sp = world.get_map().get_spawn_points()[0]
ego = world.spawn_actor(bp, sp)
ego.set_simulate_physics(False)
for _ in range(5):
    world.tick()
bb = ego.bounding_box
print(f"{HERO}: bbox location=({bb.location.x:.2f},{bb.location.y:.2f},{bb.location.z:.2f}) "
      f"extent=({bb.extent.x:.2f},{bb.extent.y:.2f},{bb.extent.z:.2f})  "
      f"=> length {2*bb.extent.x:.2f} m, roof ~{bb.location.z + bb.extent.z:.2f} m")

# something to look at, straight ahead
other = world.spawn_actor(lib.find("vehicle.audi.tt"),
                          carla.Transform(sp.location + sp.get_forward_vector() * 20.0,
                                          sp.rotation))
other.set_simulate_physics(False)
for _ in range(5):
    world.tick()

from queue import Queue
for x, z in MOUNTS:
    cbp = lib.find("sensor.camera.rgb")
    cbp.set_attribute("image_size_x", str(W)); cbp.set_attribute("image_size_y", str(H))
    cbp.set_attribute("fov", str(FOV))
    cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=x, y=0.0, z=z)),
                            attach_to=ego)
    q = Queue(); cam.listen(q.put)
    for _ in range(4):
        world.tick()
    img = q.get(timeout=20.0); cam.stop(); cam.destroy()
    raw = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((H, W, 4))
    rgb = raw[:, :, :3][:, :, ::-1].copy()
    # how much of the lower-centre is ego bodywork? use the strong green paint
    lower = rgb[int(H * 0.6):, int(W * 0.25):int(W * 0.75)]
    dark = float((lower.mean(axis=2) < 110).mean())
    name = f"x{x:+.1f}_z{z:.1f}".replace(".", "p")
    Image.fromarray(rgb).save(os.path.join(OUT, name + ".png"))
    print(f"  mount x={x:+5.2f} z={z:4.2f}  lower-centre dark fraction {dark:5.1%}  -> {name}.png")

other.destroy(); ego.destroy()
print(f"wrote {OUT}")
