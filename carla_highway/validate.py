#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/validate.py — checks that need no CARLA server.

The counterpart of `carla_port/validate.py`. It leans on a fact that port could
not use: CARLA's Python API can build a `carla.Map` from an OpenDRIVE string
with no server running, so the *real* Town geometry is available offline. Map
fitting — the riskiest part of this port — is therefore validated against the
actual roads the run will use, not against a synthetic double.

    python3 -m carla_highway.validate                    # bundled towns
    python3 -m carla_highway.validate --town Town04
    python3 -m carla_highway.validate --xodr-dir /path/to/OpenDrive

Exit code is the number of failed checks.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

DEFAULT_XODR = ("/scratch/zwang179/traffic_orchestration/install/CarlaUE4/"
                "Content/Carla/Maps/OpenDrive")

_PASS, _FAIL, _SKIP = "PASS", "FAIL", "SKIP"


class Checks:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""),
              flush=True)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.add(_PASS if ok else _FAIL, name, detail)
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == _FAIL)

    def summary(self) -> str:
        n = len(self.rows)
        f = self.failed
        s = sum(1 for st, _, _ in self.rows if st == _SKIP)
        return (f"\n{n - f - s}/{n} passed"
                + (f", {f} FAILED" if f else "")
                + (f", {s} skipped" if s else ""))


class _World:
    """The only thing HighwayFrame asks of a world is get_map()."""
    def __init__(self, cmap):
        self._m = cmap

    def get_map(self):
        return self._m


def _load_town(xodr_dir: str, town: str):
    import carla
    path = os.path.join(xodr_dir, town + ".xodr")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return carla.Map(town, fh.read())


# --------------------------------------------------------------------------- #
def check_isolation(ck: Checks) -> None:
    """The two script layers must not both be loaded."""
    import carla_highway.script_bridge as sb          # noqa: F401
    bad = [n for n in sb.CONTESTED
           if n in sys.modules and
           os.path.normpath(os.path.dirname(
               getattr(sys.modules[n], "__file__", "") or "")) !=
           os.path.normpath(sb.HIGHWAY)]
    ck.check("script layer is the highway one", not bad,
             f"foreign modules: {bad}" if bad else
             f"scenario_editor from {sb.HIGHWAY}")
    ck.check("carla_port's script layer is NOT loaded",
             "carla_port.script_bridge" not in sys.modules,
             "v2/v4 would collide with highway/ on bare module names")


def check_script_layer(ck: Checks) -> None:
    from .script_bridge import DT, co, mv, se
    ck.check("script DT is 1/60 s", abs(DT - 1.0 / 60.0) < 1e-12, f"DT={DT}")
    for fn in ("solve_closed_loop_cutin", "cruise_plan", "world_to_ego_offset",
               "live_cutin_pin", "cutin_is_merged", "actor_cruise_speed"):
        if not hasattr(se, fn):
            ck.check(f"se.{fn} present", False)
            return
    ck.check("cut-in solver surface present", True,
             "solve_closed_loop_cutin, cruise_plan, live_cutin_pin, ...")
    ck.check("CutinOrchestrator.tick is importable",
             hasattr(co, "CutinOrchestrator") and
             hasattr(co.CutinOrchestrator, "tick"))
    ck.check("rebase_scenario present", hasattr(mv, "rebase_scenario"))


def check_scenarios_parse(ck: Checks) -> None:
    from . import scenarios as sc_mod
    for mode in sc_mod.MODES:
        try:
            sc = sc_mod.load(mode)
        except Exception as exc:
            ck.check(f"{mode}: YAML loads", False, f"{type(exc).__name__}: {exc}")
            continue
        ids = [str(a.id) for a in sc.actors]
        ck.check(f"{mode}: YAML loads", True,
                 f"{len(sc.actors)} actors {ids}, map "
                 f"{sc.map.kind}/{sc.map.num_lanes} lanes")
        ck.check(f"{mode}: has an ego (actor 0)", sc_mod.EGO_ID in ids)


def check_frame(ck: Checks, cmap, town: str) -> None:
    from . import scenarios as sc_mod
    from .highway_map import HighwayFrame
    world = _World(cmap)
    for mode in sc_mod.MODES:
        spec = sc_mod.spec(mode)
        try:
            frame = sc_mod.discover_frame(world, mode)
        except RuntimeError as exc:
            ck.add(_FAIL, f"{town}/{mode}: fit a road", str(exc))
            continue
        ck.check(f"{town}/{mode}: fit a road", True,
                 f"road {frame.road_id}.{frame.section_id}, "
                 f"{frame.num_lanes} lanes, {frame.length:.0f} m, "
                 f"lw={frame.lane_width:.2f}")
        ck.check(f"{town}/{mode}: enough road for the scenario",
                 frame.length >= spec.min_length - 1e-6,
                 f"{frame.length:.0f} m >= {spec.min_length:.0f} m")
        ck.check(f"{town}/{mode}: script lanes land on real lanes",
                 frame.lane_fit_error() < 0.35,
                 f"worst {frame.lane_fit_error():.3f} m")
        if spec.two_way:
            ck.check(f"{town}/{mode}: has an oncoming lane",
                     len(frame.oncoming_lanes()) >= 1,
                     f"{len(frame.oncoming_lanes())} oncoming")

        # round trip must be the identity
        worst = 0.0
        for x, y, h in [(0, 0, 90), (frame.lane_center_x(0), -40, 90),
                        (frame.lane_center_x(frame.num_lanes - 1), 50, 270),
                        (1.5, 20, 45)]:
            bx, by, bh = frame.from_carla_transform(
                frame.to_carla_transform(x, y, h))
            worst = max(worst, abs(bx - x), abs(by - y),
                        abs((bh - h + 180) % 360 - 180))
        ck.check(f"{town}/{mode}: pose round-trip is the identity",
                 worst < 1e-3, f"worst {worst:.2e}")

        # forward really is forward
        ax, ay = frame.to_carla_xy(0.0, 0.0)
        bx, by = frame.to_carla_xy(0.0, 10.0)
        fwd_yaw = math.degrees(math.atan2(by - ay, bx - ax))
        want = frame.to_carla_yaw(90.0)
        d = abs((fwd_yaw - want + 180) % 360 - 180)
        ck.check(f"{town}/{mode}: script +y is the travel direction",
                 d < 1e-3, f"{d:.2e} deg")

        # +x is the driver's right: 90 deg clockwise of forward in CARLA
        rx, ry = frame.to_carla_xy(10.0, 0.0)
        right_yaw = math.degrees(math.atan2(ry - ay, rx - ax))
        d = abs((right_yaw - (want + 90.0) + 180) % 360 - 180)
        ck.check(f"{town}/{mode}: script +x is the driver's right",
                 d < 1e-3, f"{d:.2e} deg")

        check_retarget(ck, frame, mode, town, cmap)


def check_retarget(ck: Checks, frame, mode: str, town: str, cmap) -> None:
    from . import scenarios as sc_mod
    authored = sc_mod.load(mode)
    fitted, notes = sc_mod.retarget(authored, frame)
    off_lane = []
    off_road = []
    for a in fitted.actors:
        x, y, _h = a.start
        i = frame.lane_index_of(x)
        if abs(x - frame.lane_center_x(i)) > 1e-6:
            off_lane.append(a.id)
        if not frame.on_road(x, y):
            off_road.append(a.id)
    ck.check(f"{town}/{mode}: every actor starts on a lane centre",
             not off_lane, f"off-centre: {off_lane}" if off_lane else "")
    ck.check(f"{town}/{mode}: every actor starts on the fitted road",
             not off_road, f"off-road: {off_road}" if off_road else "")

    # and on real asphalt
    worst, worst_id = 0.0, None
    for a in fitted.actors:
        x, y, _ = a.start
        loc = frame.to_carla_location(x, y)
        wp = cmap.get_waypoint(loc, project_to_road=True)
        d = math.hypot(wp.transform.location.x - loc.x,
                       wp.transform.location.y - loc.y)
        if d > worst:
            worst, worst_id = d, a.id
    ck.check(f"{town}/{mode}: every actor starts on real asphalt",
             worst < 0.60, f"worst {worst:.2f} m (actor {worst_id})")

    if mode == sc_mod.OVERTAKE:
        onc = [a.id for a in fitted.actors
               if not sc_mod._is_forward(a.start[2])]
        ck.check(f"{town}/{mode}: oncoming traffic faces the ego",
                 len(onc) >= 1, f"oncoming actors: {onc}")

    ego, bg = sc_mod.split_ego(fitted)
    ck.check(f"{town}/{mode}: ego splits out of the background",
             str(ego.id) == sc_mod.EGO_ID and
             all(str(a.id) != sc_mod.EGO_ID for a in bg.actors),
             f"{len(bg.actors)} background actors")


def check_ego_policy(ck: Checks, cmap) -> None:
    """The lane-selection layer, on the road it will actually drive."""
    from . import scenarios as sc_mod
    from .highway_ego import Ego, HighwayEgoPolicy
    world = _World(cmap)

    # --- hard_brake: a slow lead must provoke a lane change --- #
    try:
        frame, ego_a, bg, _ = sc_mod.build(world, sc_mod.HARD_BRAKE)
    except RuntimeError as exc:
        ck.add(_SKIP, "ego: hard_brake lane change", str(exc))
        return
    x, y, h = ego_a.start
    ego = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol = HighwayEgoPolicy(frame, ego, bg, atime=0.0)
    start_lane = pol.target_lane
    for k in range(400):                       # ~6.7 s at 60 Hz
        pol.atime = k * (1.0 / 60.0)
        thr, st = pol.command(now=k / 60.0, dt=1.0 / 60.0)
        pol.integrate(thr, st, 1.0 / 60.0)
    ck.check("ego/hard_brake: changes lane around the slow lead",
             pol.target_lane != start_lane or pol.n_lane_changes > 0,
             f"lane {start_lane} -> {pol.target_lane}, "
             f"{pol.n_lane_changes} change(s), reason: {pol.reason}")

    # --- the same ego with lane changes disabled must NOT change --- #
    ego2 = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol2 = HighwayEgoPolicy(frame, ego2, bg, atime=0.0, allow_lane_change=False)
    for k in range(400):
        pol2.atime = k * (1.0 / 60.0)
        thr, st = pol2.command(now=k / 60.0, dt=1.0 / 60.0)
        pol2.integrate(thr, st, 1.0 / 60.0)
    ck.check("ego: --no-lane-change really keeps the lane",
             pol2.n_lane_changes == 0 and pol2.ego.v < 12.0,
             f"{pol2.n_lane_changes} changes, slowed to {pol2.ego.v:.1f} m/s")

    # --- overtake: never pull out in front of close oncoming traffic --- #
    try:
        frame, ego_a, bg, _ = sc_mod.build(world, sc_mod.OVERTAKE)
    except RuntimeError as exc:
        ck.add(_SKIP, "ego: overtake oncoming gate", str(exc))
        return
    x, y, h = ego_a.start
    ego3 = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol3 = HighwayEgoPolicy(frame, ego3, bg, atime=0.0)
    onc_lane = frame.lane_index_for_direction(False)
    nbrs = pol3.neighbours()
    ck.check("ego/overtake: sees the oncoming car",
             any(n.oncoming for n in nbrs),
             f"{sum(1 for n in nbrs if n.oncoming)} oncoming of {len(nbrs)}")
    # with an oncoming car close, the gate must refuse; with a clear road, allow
    from .highway_ego import Neighbour
    onc_x = frame.lane_center_x(onc_lane)

    def _oncoming_at(along: float) -> Neighbour:
        return Neighbour(actor_id="x", lane=onc_lane, lat=onc_x, along=along,
                         gap=max(along - 4.5, 0.5), speed=-8.0, oncoming=True)

    close = pol3._oncoming_clear(onc_lane, [_oncoming_at(20.0)],
                                 pass_distance=30.0)
    ck.check("ego/overtake: refuses a pass into close oncoming traffic",
             not close, "20 m of closing gap is rejected")
    far = pol3._oncoming_clear(onc_lane, [_oncoming_at(400.0)],
                               pass_distance=30.0)
    ck.check("ego/overtake: accepts a pass when the road is clear", far,
             "400 m of closing gap is accepted")

    # the merge-blindness regression: a car half way through a cut-in, between
    # two lanes, must still be seen as a leader
    half_way = Neighbour(actor_id="m", lane=pol3.target_lane,
                         lat=pol3.ego.x + 1.6, along=9.0, gap=4.5,
                         speed=6.0, oncoming=False)
    gap, v_lead = pol3._leader_near(pol3.ego.x, [half_way])
    ck.check("ego: a car merging between lanes is still a leader",
             gap is not None,
             "1.6 m off the ego's line is inside IDM_LANE_TOL")


def check_closed_loop(ck: Checks, cmap) -> None:
    """Cut-in casting, driven headless off the real road."""
    from . import scenarios as sc_mod
    from .closed_loop import HighwayClosedLoop
    from .highway_ego import Ego, HighwayEgoPolicy
    world = _World(cmap)
    try:
        frame, ego_a, bg, _ = sc_mod.build(world, sc_mod.CUTIN)
    except RuntimeError as exc:
        ck.add(_SKIP, "closed loop: cut-in casting", str(exc))
        return
    loop = HighwayClosedLoop(frame, bg, sc_mod.spec(sc_mod.CUTIN))
    ck.check("closed loop: casting is on for cutin", loop.casting,
             f"spec found: {loop.orch is not None}")
    x, y, h = ego_a.start
    ego = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol = HighwayEgoPolicy(frame, ego, loop.sc, atime=0.0)
    dt = 1.0 / 60.0
    for k in range(600):                       # 10 s
        t = k * dt
        loop.tick(ego, t)
        pol.asc, pol.atime = loop.sc, loop.atime
        thr, st = pol.command(now=t, dt=dt)
        pol.integrate(thr, st, dt)
        loop.advance(dt)
    ck.check("closed loop: a cut-in actor got cast", loop.holder is not None,
             f"holder={loop.holder} roles={loop.roles}")
    ck.check("closed loop: the cut-in resolved",
             loop.outcome in ("merged", "abandoned"),
             f"outcome={loop.outcome} after {loop.n_interventions} interventions")
    ck.check("closed loop: recasts off a hopeless holder",
             loop.n_recasts > 0 or loop.outcome == "merged",
             f"{loop.n_recasts} recast(s); CutinOrchestrator alone locks its "
             f"first pick forever")
    ck.check("closed loop: it logged what it did", len(loop.events) > 0,
             f"{len(loop.events)} events; last: "
             f"{loop.events[-1].text if loop.events else '-'}")

    # a non-casting mode must leave the scripts alone
    frame2, _e2, bg2, _ = sc_mod.build(world, sc_mod.HARD_BRAKE)
    loop2 = HighwayClosedLoop(frame2, bg2, sc_mod.spec(sc_mod.HARD_BRAKE))
    ck.check("closed loop: hard_brake does NOT cast roles",
             not loop2.casting and loop2.orch is None,
             "scripted actors stay scripted")


def check_actuation(ck: Checks) -> None:
    """The shared PID/actuator, and the steering handedness."""
    from carla_port.actuation import CarlaEgoActuator, LongitudinalPID
    from .highway_ego import DELTA_MAX, V_MAX

    pid = LongitudinalPID()
    thr, brk = pid.step(12.0, 4.0, 1.0 / 60.0)
    ck.check("PID: accelerates when below target", thr > 0 and brk == 0.0,
             f"throttle={thr:.2f}")
    pid.reset()
    thr, brk = pid.step(0.0, 8.0, 1.0 / 60.0)
    ck.check("PID: a full stop is a brake command", brk == 1.0 and thr == 0.0)

    class _Wheel:
        max_steer_angle = 70.0

    class _Phys:
        wheels = [_Wheel(), _Wheel()]

    class _Veh:
        def get_physics_control(self):
            return _Phys()

    act = CarlaEgoActuator(_Veh(), delta_max=DELTA_MAX, v_max=V_MAX)
    ck.check("actuator: steer sign is flipped for CARLA", act.steer_scale < 0,
             f"scale={act.steer_scale:.4f} (script CCW -> CARLA CW)")
    ck.check("actuator: steer scale maps DELTA_MAX to the wheel limit",
             abs(abs(act.steer_scale) - math.degrees(DELTA_MAX) / 70.0) < 1e-9)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m carla_highway.validate")
    p.add_argument("--xodr-dir", default=DEFAULT_XODR)
    p.add_argument("--town", action="append", default=None,
                   help="repeatable; default Town04 then Town05")
    args = p.parse_args(argv)
    towns = args.town or ["Town04", "Town05"]

    ck = Checks()
    print("== isolation ==")
    check_isolation(ck)
    print("== script layer ==")
    check_script_layer(ck)
    print("== scenarios ==")
    check_scenarios_parse(ck)
    print("== actuation ==")
    check_actuation(ck)

    first = None
    for town in towns:
        print(f"== map fit: {town} ==")
        try:
            cmap = _load_town(args.xodr_dir, town)
        except Exception as exc:
            ck.add(_FAIL, f"{town}: OpenDRIVE loads", f"{type(exc).__name__}: {exc}")
            continue
        if cmap is None:
            ck.add(_SKIP, f"{town}: OpenDRIVE loads",
                   f"no {town}.xodr under {args.xodr_dir}")
            continue
        ck.add(_PASS, f"{town}: OpenDRIVE loads", "offline, no server needed")
        check_frame(ck, cmap, town)
        if first is None:
            first = cmap

    if first is not None:
        print("== ego policy ==")
        check_ego_policy(ck, first)
        print("== closed loop ==")
        check_closed_loop(ck, first)
    else:
        ck.add(_SKIP, "ego policy + closed loop", "no town map available")

    print(ck.summary())
    return ck.failed


if __name__ == "__main__":
    sys.exit(main())
