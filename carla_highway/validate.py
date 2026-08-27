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

    # --- MOBIL's safety criterion is what gap acceptance could not do --- #
    # A follower 7 m behind is "clear" to a fixed-window rule either way; MOBIL
    # asks how hard the merge makes it brake, so the answer must depend on how
    # fast it is closing. If both cases agree, the safety term is not wired in.
    from .highway_ego import Neighbour, MOBIL_B_SAFE
    ego_m = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol_m = HighwayEgoPolicy(frame, ego_m, bg, atime=0.0)
    cur = pol_m.target_lane
    cand = 1 if cur == 0 else 0
    cand_x = frame.lane_center_x(cand)

    def _follower(speed: float) -> Neighbour:
        # 25 m back: a fixed-window gap rule (the old LC_REAR_GAP = 6 m) calls
        # this clear at any speed. MOBIL has to disagree at one of them.
        return Neighbour(actor_id="f", lane=cand, lat=cand_x, along=-25.0,
                         gap=20.5, speed=speed, oncoming=False,
                         length=4.5, width=2.0)

    _, _, why_slow = pol_m._mobil_evaluate(cur, cand, [_follower(12.0)])
    ok_fast, _, why_fast = pol_m._mobil_evaluate(cur, cand, [_follower(30.0)])
    ck.check("ego/MOBIL: refuses a merge that would slam the new follower",
             not ok_fast and "unsafe" in why_fast,
             f"25 m back but closing at 30 m/s: {why_fast} "
             f"(limit {MOBIL_B_SAFE} m/s^2)")
    ck.check("ego/MOBIL: the same gap at matched speed is safe",
             "unsafe" not in why_slow,
             f"25 m back at 12 m/s: {why_slow} — a fixed-window gap rule "
             "cannot tell these two apart")

    # --- the keep-home bias must make coming home cheaper than leaving --- #
    # On the 3-lane cut-in road, where home is the CENTRE lane and a keep-right
    # bias would point the wrong way.
    try:
        frame3, ego_a3, bg3, _ = sc_mod.build(world, sc_mod.CUTIN)
    except RuntimeError:
        frame3 = None
    if frame3 is not None and frame3.num_lanes >= 3:
        frame, bg = frame3, bg3
        y, h = ego_a3.start[1], ego_a3.start[2]
        centre = 1
        ego_h = Ego(x=frame.lane_center_x(centre), y=y,
                    theta=math.radians(h), v=12.0)
        pol_h = HighwayEgoPolicy(frame, ego_h, bg, atime=0.0)
        pol_h.home_lane = centre
        pol_h.target_lane = 2                 # pretend we are out of position
        _, _, why_home = pol_h._mobil_evaluate(2, centre, [])
        _, _, why_away = pol_h._mobil_evaluate(centre, 2, [])
        thr_home = float(why_home.rsplit(" ", 1)[-1])
        thr_away = float(why_away.rsplit(" ", 1)[-1])
        ck.check("ego/MOBIL: keep-home bias makes returning cheaper",
                 thr_home < thr_away,
                 f"threshold home {thr_home:+.2f} vs away {thr_away:+.2f}")

    # --- the lateral profile must be speed-independent and jerk-free --- #
    ego_p = Ego(x=frame.lane_center_x(0), y=y, theta=math.radians(h), v=12.0)
    pol_p = HighwayEgoPolicy(frame, ego_p, bg, atime=0.0)
    s0, ds0, dds0 = pol_p._smoothstep(0.0)
    s1, ds1, dds1 = pol_p._smoothstep(1.0)
    ck.check("ego/MOBIL: the lane-change profile starts and ends at rest",
             abs(s0) < 1e-9 and abs(s1 - 1.0) < 1e-9
             and max(abs(ds0), abs(ds1), abs(dds0), abs(dds1)) < 1e-9,
             "quintic smoothstep: S'=S''=0 at both ends, so no steering step")
    from .highway_ego import LC_DISTANCE
    pol_p._commit(1, now=0.0)
    pol_p.advance_lane_change(LC_DISTANCE * 0.5)
    half = pol_p.lane_target_lateral()[0]
    mid = (frame.lane_center_x(0) + frame.lane_center_x(1)) / 2.0
    ck.check("ego/MOBIL: half the distance is half the lane change",
             abs(half - mid) < 1e-6,
             f"x={half:.3f} at s=LC_DISTANCE/2, lane midpoint {mid:.3f}")
    pol_p.advance_lane_change(LC_DISTANCE * 0.5 + 1e-6)
    ck.check("ego/MOBIL: the manoeuvre retires when the distance is covered",
             pol_p.lc is None, "profile cleared at s >= LC_DISTANCE")
    # a stopped ego makes no progress along the profile — correct, not a stall
    pol_p._commit(0, now=10.0)
    before = pol_p.lc["s"]
    pol_p.advance_lane_change(0.0)
    ck.check("ego/MOBIL: a stopped ego does not advance the profile",
             pol_p.lc is not None and pol_p.lc["s"] == before,
             "distance-parameterised, so braking to a crawl stops the merge")

    # --- signed speeds: an oncoming car must not read as a fast leader --- #
    ego_s = Ego(x=x, y=y, theta=math.radians(h), v=12.0)
    pol_s = HighwayEgoPolicy(frame, ego_s, bg, atime=0.0)
    a_same = pol_s._idm_accel(20.0, 12.0)
    a_head_on = pol_s._idm_accel(20.0, -12.0)
    ck.check("ego: IDM brakes harder for a head-on than for a matched leader",
             a_head_on < a_same - 1.0,
             f"20 m gap: same-direction {a_same:+.2f}, oncoming "
             f"{a_head_on:+.2f} m/s^2")

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
    # The regression the whole port turned on: a holder that has drifted into
    # the ego's lane scores zero for casting, and `cast_roles` would hand the
    # role away on the very tick the merge is detectable.
    from .closed_loop import StickyCutinOrchestrator
    from .script_bridge import ROLE_CUTIN as _RC
    ck.check("closed loop: the orchestrator is sticky on feasibility",
             isinstance(loop.orch, StickyCutinOrchestrator),
             "cast_roles alone drops a lock whose score has fallen to 0 — "
             "which is every holder that is actually merging")
    if loop.orch is not None:
        orch = loop.orch
        actors = loop.sc.actors
        keep = str(actors[0].id)
        orch.committed = False
        orch.cutin_id = keep
        orch.sticky_id = keep
        # put the "holder" exactly on the ego's line, where it scores 0
        saved = actors[0].start
        actors[0].start = (ego.x, ego.y + 6.0, saved[2] if len(saved) > 2 else 90.0)
        orch.cast(actors, ego, loop.sc.map.lane_width, sticky=True)
        actors[0].start = saved
        ck.check("closed loop: a merging holder keeps the role",
                 orch.cutin_id == keep and orch.roles.get(actors[0].id) == _RC,
                 f"holder {keep} on the ego's own line still holds the cut-in")

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
