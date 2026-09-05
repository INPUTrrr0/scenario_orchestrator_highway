#!/usr/bin/env python3
"""Pre-flight the three integration fixes without a CARLA server or a GPU.

Cheap checks for the parts that are pure arithmetic or pure string handling, so
a sign error or a missing token costs a CPU minute instead of nine L40S jobs.
Run under whichever interpreter is available; each check skips itself if its
dependencies are not importable.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name):
    def deco(fn):
        try:
            fn()
            print(f"[ok]   {name}")
        except _Skip as exc:
            print(f"[skip] {name}: {exc}")
        except Exception as exc:                       # noqa: BLE001
            FAILURES.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return fn
    return deco


class _Skip(Exception):
    pass


def close(a, b, tol=1e-3, what=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{what}: {a!r} != {b!r}")


# --------------------------------------------------------------------------- #
@check("radar: spherical sensor frame -> ego frame")
def _radar_transform():
    import numpy as np
    try:
        from carla_port import carla_sensors as cs
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_port not importable ({exc})")

    class _Meas:
        def __init__(self, rows):
            self.raw_data = np.asarray(rows, dtype=np.float32).tobytes()

    # one detection straight down the boresight, 10 m out, closing at 5 m/s
    meas = _Meas([[-5.0, 0.0, 0.0, 10.0]])          # vel, azimuth, alt, depth

    # radar2: mounted at x=+2.6, yawed +45 deg
    spec = cs.RadarSpec("radar2", x=2.6, y=0.0, z=0.60, yaw=45.0)
    out = cs.CameraRig._to_radar(meas, spec)
    assert out.shape == (1, 4), out.shape
    close(out[0, 0], 10 * math.cos(math.radians(45)) + 2.6, what="radar2 x")
    close(out[0, 1], 10 * math.sin(math.radians(45)), what="radar2 y")
    close(out[0, 2], 0.60, what="radar2 z")
    close(out[0, 3], -5.0, what="radar2 v (passed through)")

    # radar1: same mount, yawed -45 deg -> mirrored in y
    spec = cs.RadarSpec("radar1", x=2.6, y=0.0, z=0.60, yaw=-45.0)
    out = cs.CameraRig._to_radar(meas, spec)
    close(out[0, 1], -10 * math.sin(math.radians(45)), what="radar1 y")

    # radar4: rear mount, yawed 225 deg -> behind the ego
    spec = cs.RadarSpec("radar4", x=-2.6, y=0.0, z=0.60, yaw=225.0)
    out = cs.CameraRig._to_radar(meas, spec)
    assert out[0, 0] < -2.6, f"rear radar should point backwards, got x={out[0,0]}"

    # an empty sweep is a well-formed (0, 4), not a crash
    assert cs.CameraRig._to_radar(_Meas(np.zeros((0, 4))), spec).shape == (0, 4)


@check("radar: a policy dict with kind=sensor.other.radar becomes a RadarSpec")
def _radar_specs_from():
    try:
        from carla_port import carla_sensors as cs
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_port not importable ({exc})")
    specs = cs.specs_from([
        {"name": "radar1", "kind": "sensor.other.radar", "x": 2.6, "yaw": -45.0},
        {"name": "lidar", "kind": "sensor.lidar.ray_cast"},
        {"name": "cam", "width": 384, "height": 384},
    ])
    kinds = [type(s).__name__ for s in specs]
    assert kinds == ["RadarSpec", "LidarSpec", "CameraSpec"], kinds
    assert cs.RadarSpec("r").attributes()["horizontal_fov"] == "90.0"
    # the named tfv6 rig must carry four radars, in sensor-index order
    names = [s.name for s in cs.rig("tfv6") if isinstance(s, cs.RadarSpec)]
    assert names == ["radar1", "radar2", "radar3", "radar4"], names


# --------------------------------------------------------------------------- #
@check("route: reference_path ramps through a lane change instead of stepping")
def _reference_path():
    try:
        from carla_highway.highway_ego import HighwayEgoPolicy, LC_DISTANCE
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"highway_ego not importable ({exc})")

    class _Frame:
        length = 200.0
        def lane_center_x(self, i):
            return -3.5 + 3.5 * i

    class _Ego:
        x, y = 0.0, 0.0

    class _Stub:
        frame, ego = _Frame(), _Ego()
        target_lane = 1
        lc = None
        # staticmethod via the class is a plain function; assigning it to a
        # stub would rebind it as a method and pass `self` as `f`.
        _smoothstep = staticmethod(HighwayEgoPolicy._smoothstep)

    path = HighwayEgoPolicy.reference_path.fget
    stub = _Stub()

    # no manoeuvre: a straight line down the target lane, as before
    pts = path(stub)
    assert len({round(x, 6) for x, _y in pts}) == 1, "lane keeping should be straight"
    close(pts[0][0], 0.0, what="lane-keep x")

    # mid-manoeuvre: starts where the ego IS and ends at the new lane centre
    stub.lc = {"x0": 0.0, "x1": 3.5, "s": 0.0, "L": LC_DISTANCE}
    stub.target_lane = 2
    pts = path(stub)
    close(pts[0][0], 0.0, tol=0.05, what="ramp starts at the ego's own x")
    assert pts[-1][0] > 3.0, f"ramp should reach the new lane, ended at {pts[-1][0]}"
    xs = [x for x, _y in pts]
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:])), "ramp must be monotonic"
    # and it must not be a step: the first 2 m should move only a little
    assert xs[1] - xs[0] < 0.5, f"first step too abrupt: {xs[1] - xs[0]}"

    # half way through, the path continues from where the ego already is
    stub.lc = {"x0": 0.0, "x1": 3.5, "s": LC_DISTANCE / 2, "L": LC_DISTANCE}
    pts = path(stub)
    assert 1.0 < pts[0][0] < 2.5, f"half-way start looks wrong: {pts[0][0]}"


# --------------------------------------------------------------------------- #
@check("simlingo: prompt carries two <TARGET_POINT> tokens and a CoT question")
def _simlingo_prompt():
    root = os.environ.get("SIMLINGO_ROOT",
                          "/scratch/zwang179/traffic_orchestration/policies/simlingo")
    path = os.path.join(root, "scenario_orchestration", "policy.py")
    if not os.path.isfile(path):
        raise _Skip(f"{path} not found")
    src = open(path).read()
    assert 'prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."' in src, \
        "the route block is not upstream's"
    assert '"What should the ego do next?"' in src, "missing the use_cot task suffix"
    assert "placeholder_values=[{tp_id: target_np}]" in src, \
        "placeholder_values is still empty; the model would get no route"
    assert "Target point: (" not in src, "the old literal-coordinate prompt survives"
    # two tokens, because replace_placeholder_tokens writes one embedding per pair
    assert src.count("<TARGET_POINT>") >= 2


@check("simlingo: two route points come back from the port's dense route")
def _simlingo_target_points():
    root = os.environ.get("SIMLINGO_ROOT",
                          "/scratch/zwang179/traffic_orchestration/policies/simlingo")
    path = os.path.join(root, "scenario_orchestration", "policy.py")
    if not os.path.isfile(path):
        raise _Skip(f"{path} not found")
    ns = {}
    src = open(path).read()
    start = src.index("def _target_points(")
    end = src.index("def build_policy(")
    exec("from typing import Any, Dict\n" + src[start:end], ns)     # noqa: S102
    fn = ns["_target_points"]
    route = [[2.5 + i, 0.0] for i in range(20)]
    pts = fn({"route": route})
    assert len(pts) == 2, pts
    close(pts[0][0], 9.5, what="target point")
    close(pts[1][0], 21.5, what="next target point")
    # no route at all still yields two usable points rather than a crash
    assert len(fn({})) == 2
    # a short route pads by repeating its last point
    assert len(fn({"route": [[3.0, 0.0]]})) == 2


@check("tfv6: radar block is (300, 5) with the sensor index in column 5")
def _tfv6_radar_block():
    try:
        import numpy as np
        from lead.policy.transfuser.dataloader.features import (
            _preprocess_radar_input)
        from lead.config.lead_config import LeadConfig
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"lead not importable ({exc})")
    cfg = LeadConfig()
    # float16: the annotation on _preprocess_radar_input is runtime-checked
    empty = {f"radar{i}": np.zeros((0, 4), dtype=np.float16) for i in range(1, 5)}
    blocks = _preprocess_radar_input(cfg, empty)
    block = np.concatenate(blocks, axis=0)
    per = cfg.policy.transfuser.num_radar_points_per_sensor
    assert block.shape == (per * 4, 5), block.shape
    for i in range(4):
        col = block[i * per:(i + 1) * per, 4]
        assert (col == i).all(), f"sensor index column wrong for radar{i+1}"
    # a real detection inside the BEV bounds survives the filter
    one = dict(empty)
    one["radar2"] = np.array([[10.0, 5.0, 0.6, -3.0]], dtype=np.float16)
    block = np.concatenate(_preprocess_radar_input(cfg, one), axis=0)
    kept = block[block[:, 0] != 0.0]
    assert len(kept) == 1 and abs(kept[0, 0] - 10.0) < 1e-4, kept


@check("video: VisionPanel renders cameras, a BEV raster and a placeholder")
def _vision_panel():
    try:
        import numpy as np
        from carla_port.carla_video import VisionPanel
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_video not importable ({exc})")
    panel = VisionPanel(720, 540)

    # nothing at all -> a captioned placeholder, not a black rectangle
    blank = panel.render(None)
    assert blank.shape == (540, 720, 3), blank.shape
    assert blank.dtype.name == "uint8"

    # one camera (simlingo): letterboxed, aspect preserved
    cam = np.full((512, 1024, 3), 90, dtype=np.uint8)
    out = panel.render({"cameras": {"rgb_front": cam}})
    assert out.shape == (540, 720, 3)
    assert out.max() > 0, "the camera frame did not reach the panel"

    # three cameras (tfv6): stitched, still fits
    views = {n: np.full((384, 384, 3), i * 40 + 40, dtype=np.uint8)
             for i, n in enumerate(("PCAM_L0", "PCAM_F0", "PCAM_R0"))}
    out = panel.render({"cameras": views})
    assert out.shape == (540, 720, 3)

    # a BEV raster of class indices (plant2) gets colourised, not passed through
    raster = np.arange(256 * 256, dtype=np.int32).reshape(256, 256) % 11
    out = panel.render({"cameras": {}, "bev": raster})
    assert out.shape == (540, 720, 3)
    assert len({tuple(c) for c in out.reshape(-1, 3)[::997]}) > 3, \
        "the raster should colourise to several distinct classes"

    # a lidar/radar array must not be mistaken for an image
    out = panel.render({"cameras": {"lidar": np.zeros((500, 4),
                                                      dtype=np.float32)}})
    assert out.shape == (540, 720, 3)


@check("video: the occluder filter is label-scoped and always explains itself")
def _occluders():
    try:
        import carla
        from carla_port.carla_video import clear_top_view_occluders
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_video not importable ({exc})")
    if not hasattr(getattr(carla, "CityObjectLabel", None), "Bridge"):
        raise _Skip("this CARLA build has no CityObjectLabel.Bridge")

    class _V:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _Box:
        def __init__(self, loc, ext):
            self.location, self.extent = loc, ext

    class _Obj:
        def __init__(self, oid, name, loc, ext):
            self.id, self.name = oid, name
            self.bounding_box = _Box(loc, ext)

    class _World:
        """Only the Bridge label returns anything, as on a real map."""
        def __init__(self, bridges, everything=None):
            self.bridges = bridges
            self.everything = everything if everything is not None else []
            self.disabled = None
            self.asked = []
        def get_environment_objects(self, label=None):
            self.asked.append(label)
            return (self.bridges if label == carla.CityObjectLabel.Bridge
                    else self.everything)
        def enable_environment_objects(self, ids, on):   # the 0.9.16 name
            self.disabled = (set(ids), on)

    class _Frame:
        anchor = _V(0.0, 0.0, 0.0)

    # the two Town04 decks the probe found, named like road meshes
    decks = [
        _Obj(1, "Road_Road_Town04_334_SM_0", _V(3.6, 10.7, 9.6), _V(20.0, 8.0, 1.0)),
        _Obj(2, "Road_Road_Town04_335_SM_0", _V(21.4, 10.8, 10.0), _V(20.0, 8.0, 1.0)),
        # a bridge elsewhere on the map, outside the recorded footprint
        _Obj(3, "Road_Road_Town04_900_SM_0", _V(400.0, 0.0, 12.0), _V(20.0, 8.0, 1.0)),
    ]
    # the carriageway and terrain a geometry-only filter used to sweep up. These
    # are NOT labelled Bridge, so a label-scoped query must never see them.
    carriageway = [
        _Obj(90, "Road_Road_Town04_104_SM_0", _V(1.0, 2.0, 9.0), _V(50.0, 20.0, 1.0)),
        _Obj(91, "Town04_TerrainNode_3775_SM_0", _V(2.0, 1.0, 20.0), _V(80.0, 80.0, 9.0)),
    ]
    world = _World(decks, everything=decks + carriageway)
    names, why = clear_top_view_occluders(world, _Frame(), span=94.5)
    assert names == ["Road_Road_Town04_334_SM_0",
                     "Road_Road_Town04_335_SM_0"], names
    assert world.disabled == ({1, 2}, False), world.disabled
    assert world.asked == [carla.CityObjectLabel.Bridge], world.asked
    assert "hid 2" in why, why

    # nothing overhead nearby -> empty, but the reason is still reported
    world2 = _World([decks[2]])
    names, why = clear_top_view_occluders(world2, _Frame(), span=94.5)
    assert names == [] and world2.disabled is None
    assert "none overhead" in why, why

    # a failing query must say so rather than look like "nothing to hide" —
    # this exact silence is what hid the bug for two grids
    class _Broken:
        def get_environment_objects(self, label=None):
            raise RuntimeError("time-out of 20000ms while waiting for the simulator")
    names, why = clear_top_view_occluders(_Broken(), _Frame(), span=94.5)
    assert names == [] and "query failed" in why and "time-out" in why, why

    # a build with NEITHER spelling must report that, not claim success
    class _NoToggle(_World):
        enable_environment_objects = None
    nt = _NoToggle(decks)
    names, why = clear_top_view_occluders(nt, _Frame(), span=94.5)
    assert names == [] and "no enable_environment_object" in why, why

    # the client timeout is raised for the query and restored afterwards
    class _Client:
        def __init__(self):
            self.timeouts = []
        def set_timeout(self, t):
            self.timeouts.append(t)
    client = _Client()
    clear_top_view_occluders(_World(decks), _Frame(), span=94.5, client=client)
    assert client.timeouts and client.timeouts[0] >= 60.0, client.timeouts
    assert client.timeouts[-1] == 20.0, client.timeouts


@check("video: the top camera follows the ego and keeps the frame's own yaw")
def _top_follow():
    try:
        from carla_port.carla_video import CameraRig
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_video not importable ({exc})")

    class _Loc:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Tf:
        def __init__(self, loc):
            self.location = loc

    class _Ego:
        def __init__(self, x, y, z):
            self.loc = _Loc(x, y, z)
        def get_transform(self):
            return _Tf(self.loc)

    class _Cam:
        def __init__(self):
            self.placed = []
        def set_transform(self, tf):
            self.placed.append(tf)

    class _Frame:
        anchor = _Loc(0.0, 0.0, 0.0)
        length = 400.0
        num_lanes = 3
        lane_width = 3.5
        def to_carla_yaw(self, y):
            return 90.0

    rig = CameraRig.__new__(CameraRig)
    rig.frame = _Frame()
    rig.width, rig.height, rig.fov = 720, 540, 90.0
    rig.top_span = 94.5
    rig.top_follow = True
    # `__new__` skips __init__, so the instrumentation fields have to be set by
    # hand — follow() records into them on every call.
    rig.last_follow = None
    rig.follow_failures = 0
    cam = _Cam()
    rig._top_cam = cam
    ego = _Ego(120.0, -8.0, 1.0)
    rig._ego_actor = ego

    # anchored on the ego, not the road anchor
    tf = rig._top_transform(ego)
    close(tf.location.x, 120.0, what="follows the ego in x")
    close(tf.location.y, -8.0, what="follows the ego in y")
    assert tf.location.z > 40.0, tf.location.z
    close(tf.rotation.pitch, -90.0, what="looks straight down")
    close(tf.rotation.yaw, 90.0, what="keeps the frame's yaw, not the ego's")

    # without an ego it falls back to the road anchor
    tf0 = rig._top_transform(None)
    close(tf0.location.x, 0.0, what="falls back to the anchor")

    # follow() re-places the camera; disabled, it does not
    rig.follow()
    assert len(cam.placed) == 1, cam.placed
    rig.top_follow = False
    rig.follow()
    assert len(cam.placed) == 1, "follow() must be a no-op when disabled"

    # a camera that refuses set_transform must not take the run down
    class _Broken:
        def set_transform(self, tf):
            raise RuntimeError("actor destroyed")
    rig.top_follow = True
    rig._top_cam = _Broken()
    rig.follow()
    assert rig.follow_failures == 1, rig.follow_failures
    assert rig.last_follow is not None and rig.last_follow[2] is False, rig.last_follow

    # a working placement records the camera and ego side by side, so the run
    # log can show the offset instead of leaving it to be read off the video
    rig._top_cam = _Cam()
    rig.follow()
    cam_xyz, ego_xyz, ok = rig.last_follow
    assert ok is True
    close(cam_xyz[0], ego_xyz[0], what="camera sits over the ego in x")
    close(cam_xyz[1], ego_xyz[1], what="camera sits over the ego in y")


@check("scenarios: the delayed hard_brake brakes at t=3 from a nominal speed")
def _hard_brake_delayed():
    import yaml
    path = os.path.join(ROOT, "scenarios", "scenario_hard_brake_delayed.yaml")
    doc = yaml.safe_load(open(path))
    lead = next(a for a in doc["actors"] if a["id"] == 1)
    mv = lead["maneuvers"]
    assert mv[0]["type"] == "go_straight", mv[0]
    close(mv[0]["duration"], 3.0, what="nominal phase ends at t=3")
    assert mv[0]["curve"]["v0"] >= 8.0, "lead must still be moving nominally"
    assert mv[1]["type"] == "decelerate", mv[1]
    assert mv[1]["curve"]["accel"] <= -2.5, "decel must be a real brake"
    # and it must actually load through the port's scenario layer
    from carla_highway import scenarios as sc_mod
    sc = sc_mod.load("hard_brake", path)
    ego, bg = sc_mod.split_ego(sc)
    assert [a.id for a in bg.actors] == ["1", "2"], [a.id for a in bg.actors]


@check("staging: --along-offset slides every actor and nothing else")
def _along_offset():
    try:
        from carla_highway import scenarios as sc_mod
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"scenarios not importable ({exc})")

    class _Lane:
        def __init__(self, i):
            self.same_direction = True
            self.heading = 90.0

    class _Frame:
        num_lanes = 2
        length = 400.0
        lanes = [_Lane(0), _Lane(1)]
        def lane_center_x(self, i):
            return -1.75 + 3.5 * i
        def two_way_ok(self, fwd):
            return True
        def lane_index_for_direction(self, fwd, prefer=0):
            return prefer
        def map_config(self):
            return sc_mod.mp.MapConfig(kind="straight", num_lanes=2,
                                       lane_width=3.5, length=400.0)

    path = os.path.join(ROOT, "scenarios", "scenario_hard_brake_delayed.yaml")
    frame = _Frame()

    base, _n = sc_mod.retarget(sc_mod.load("hard_brake", path), frame)
    moved, notes = sc_mod.retarget(sc_mod.load("hard_brake", path), frame,
                                   along_offset=110.0)

    b = {str(a.id): a.start for a in base.actors}
    m = {str(a.id): a.start for a in moved.actors}
    assert set(b) == set(m), (sorted(b), sorted(m))
    for aid in b:
        close(m[aid][1] - b[aid][1], 110.0, what=f"actor {aid} slid")
        close(m[aid][0], b[aid][0], what=f"actor {aid} lateral unchanged")
        close(m[aid][2], b[aid][2], what=f"actor {aid} heading unchanged")

    # spacing between actors is what the scenario measures: it must not move
    ids = sorted(b)
    for i in range(len(ids) - 1):
        gap_b = b[ids[i + 1]][1] - b[ids[i]][1]
        gap_m = m[ids[i + 1]][1] - m[ids[i]][1]
        close(gap_m, gap_b, what=f"gap {ids[i]}->{ids[i+1]} unchanged")

    # and the run must land inside the measured clear stretch (y = +40..+300)
    ego_start = m["0"][1]
    assert 40.0 <= ego_start <= 120.0, f"ego starts at {ego_start}, outside the clear run"
    assert ego_start + 150.0 <= 300.0, "the run would leave the clear stretch"

    assert any("slid" in n for n in notes), notes
    # a zero offset must change nothing at all, and say nothing
    same, notes0 = sc_mod.retarget(sc_mod.load("hard_brake", path), frame,
                                   along_offset=0.0)
    assert not any("slid" in n for n in notes0), notes0


@check("sensors: sweeping sensors are retimed to the world's tick rate")
def _sensor_retiming():
    try:
        from carla_port import carla_sensors as cs
    except Exception as exc:                           # noqa: BLE001
        raise _Skip(f"carla_port not importable ({exc})")

    lid = cs.LidarSpec("lidar")
    # the shipped default is the bug: 20 rev/s against a 60 Hz world gives a
    # 120 degree wedge per frame, measured at 180-299 deg on a real run
    close(lid.rotation_frequency, 20.0, what="default rotation rate")
    at60 = lid.for_tick_rate(60.0)
    close(at60.rotation_frequency, 60.0, what="one revolution per tick")
    close(at60.points_per_second, lid.points_per_revolution * 60,
          what="density scales with rotation rate")
    # points per revolution is the invariant, and it must not drift with hz
    for hz in (20.0, 30.0, 60.0, 100.0):
        spec = lid.for_tick_rate(hz)
        close(spec.points_per_second / spec.rotation_frequency,
              lid.points_per_revolution, tol=1.0,
              what=f"points per revolution at {hz} Hz")

    rad = cs.RadarSpec("radar1")
    at60 = rad.for_tick_rate(60.0)
    close(at60.points_per_second, rad.points_per_tick * 60,
          what="radar returns per tick")
    # 75 is what lead's _preprocess_radar_input keeps per sensor; anything less
    # per tick is zero padding masquerading as a clear road
    assert at60.points_per_second / 60.0 >= 75.0, at60.points_per_second

    # a zero or unknown tick rate must leave the spec alone rather than
    # producing a zero-rate sensor
    assert lid.for_tick_rate(0.0) is lid
    assert rad.for_tick_rate(-1.0) is rad

    # the rig reads the rate off the world, and survives a world that has none
    class _Settings:
        fixed_delta_seconds = 1.0 / 60.0
    class _World:
        def get_settings(self):
            return _Settings()
    rig = cs.CameraRig.__new__(cs.CameraRig)
    rig.world = _World()
    close(rig._tick_hz(), 60.0, what="tick rate read from the world")
    class _Broken:
        def get_settings(self):
            raise RuntimeError("no settings")
    rig.world = _Broken()
    close(rig._tick_hz(), 0.0, what="unknown tick rate is 0, not a crash")


@check("simlingo: the bonnet band is cropped exactly as agent_simlingo does")
def _bonnet_crop():
    import numpy as np
    root = os.environ.get("SIMLINGO_ROOT",
                          "/scratch/zwang179/traffic_orchestration/policies/simlingo")
    path = os.path.join(root, "scenario_orchestration", "policy.py")
    if not os.path.isfile(path):
        raise _Skip(f"{path} not found")
    ns = {"np": np}
    src = open(path).read()
    start = src.index("BONNET_CROP_NUM")
    end = src.index("def _target_points(")
    exec("import numpy as np\n" + src[start:end], ns)          # noqa: S102
    crop = ns["_crop_bonnet"]

    # upstream: rgb[:int(h - (h * 4.8) // 16), :, :]
    for h, w in ((512, 1024), (384, 1152), (256, 256)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        want = int(h - (h * 4.8) // 16)
        out = crop(frame)
        assert out.shape == (want, w, 3), (h, out.shape, want)
    # the rig's own geometry: 512 -> 359 rows, 153 discarded
    assert crop(np.zeros((512, 1024, 3), np.uint8)).shape[0] == 359

    # the discarded band is the BOTTOM: mark it and prove it is gone
    frame = np.zeros((512, 1024, 3), dtype=np.uint8)
    frame[359:, :, :] = 255                      # the bonnet band
    assert crop(frame).max() == 0, "the crop kept the bonnet band"
    frame = np.zeros((512, 1024, 3), dtype=np.uint8)
    frame[:359, :, :] = 255                      # the road ahead
    assert crop(frame).min() == 255, "the crop discarded the road"

    # a degenerate frame must not produce an empty tensor
    assert crop(np.zeros((1, 4, 3), np.uint8)).shape[0] >= 1

    # and the adapter must actually call it on the way to the model
    assert "_crop_bonnet(frame)" in src, "the crop is defined but never applied"


@check("tfv6: the camera rig matches lead's own SensorRigConfig")
def _tfv6_rig():
    root = os.environ.get("TFV6_ROOT",
                          "/scratch/zwang179/traffic_orchestration/policies/tfv6")
    adapter = os.path.join(root, "scenario_orchestration", "policy.py")
    rig_cfg = os.path.join(root, "src", "lead", "config", "expert",
                           "sensor_rig_config.py")
    if not (os.path.isfile(adapter) and os.path.isfile(rig_cfg)):
        raise _Skip("tfv6 sources not found")

    import ast as _ast
    src = open(adapter).read()
    tree = _ast.parse(src)
    sensors = None
    for node in tree.body:
        # SENSORS carries a type annotation, so it is an AnnAssign, not Assign
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "") == "SENSORS":
            sensors = _ast.literal_eval(node.value)
        elif isinstance(node, _ast.Assign) and getattr(node.targets[0], "id", "") == "SENSORS":
            sensors = _ast.literal_eval(node.value)
    assert sensors, "SENSORS not found in the adapter"
    cams = {c["name"]: c for c in sensors if "camera" not in str(c.get("kind", ""))
            and c["name"].startswith("PCAM")}

    # lead's rig, entries 0-2, which TransfuserCameraConfig.input_cameras selects
    want = {
        "PCAM_L0": dict(x=0.0,  y=-0.3, z=2.25, yaw=-57.5, fov=60.0),
        "PCAM_F0": dict(x=0.25, y=0.0,  z=2.25, yaw=0.0,   fov=60.0),
        "PCAM_R0": dict(x=0.0,  y=0.3,  z=2.25, yaw=57.5,  fov=60.0),
    }
    assert set(cams) == set(want), (sorted(cams), sorted(want))
    for name, w in want.items():
        got = cams[name]
        for key, value in w.items():
            close(float(got.get(key, 0.0)), value, tol=1e-6,
                  what=f"{name}.{key}")
        assert got["width"] == got["height"] == 384, got

    # the values must still agree with lead's config file, so a change upstream
    # is caught here rather than silently diverging
    cfg = open(rig_cfg).read()
    for token in ('"pos": [0.25, 0.0, 2.25]', '"fov": 60', '"rot": [0.0, 0.0, -57.5]'):
        assert token in cfg, f"lead's rig no longer contains {token!r}"

    # and the lidar must sit above the hero's 1.49 m roof
    lidar = [c for c in sensors if "lidar" in str(c.get("kind", ""))]
    assert lidar and float(lidar[0]["z"]) >= 2.0, lidar
    # the mount that put the lens inside the cabin must not come back
    assert all(float(c.get("x", 0.0)) > -1.0 for c in sensors), \
        "a sensor is still mounted behind the cabin"


print()
if FAILURES:
    print(f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks that could run, passed")
