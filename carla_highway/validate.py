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


def _find_av_root(start: Optional[str] = None) -> Optional[str]:
    env = os.environ.get("AV_ROOT")
    if env and os.path.isfile(os.path.join(env, "install", "env.sh")):
        return env
    d = os.path.abspath(start or os.path.dirname(__file__))
    while True:
        if (os.path.isfile(os.path.join(d, "install", "env.sh"))
                and os.path.isfile(os.path.join(d, "third_party", "env.sh"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _default_xodr_dir() -> str:
    """OpenDRIVE dir from AV_SERVER / CARLA_ROOT / nearest install/CarlaUE4."""
    for root_env in ("AV_SERVER", "CARLA_ROOT"):
        root = os.environ.get(root_env)
        if root:
            candidate = os.path.join(root, "Content", "Carla", "Maps", "OpenDrive")
            if os.path.isdir(candidate):
                return candidate
    av = _find_av_root(os.path.join(os.path.dirname(__file__), ".."))
    if av:
        candidate = os.path.join(
            av, "install", "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive")
        if os.path.isdir(candidate):
            return candidate
    return ""


DEFAULT_XODR = _default_xodr_dir()

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
    import cutin_director as cd
    ck.check("cutin_director is importable",
             hasattr(cd, "CutinDirector") and hasattr(cd.CutinDirector, "tick"))
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

    # --- the speed floor must not deadlock an ego stopped behind a blocker --- #
    # MOBIL_V_MIN stops the ego twitching between lanes at walking pace; on
    # `overtake` the ego ends up stopped nose-to-tail with a stopped blocker,
    # and there refusing to decide is permanent. The two cases differ by WHY
    # the ego is slow, not by how slow it is.
    from .highway_ego import BLOCKED_GAP_M, MOBIL_V_MIN
    ego_b = Ego(x=x, y=y, theta=math.radians(h), v=0.4)
    pol_b = HighwayEgoPolicy(frame, ego_b, bg, atime=0.0)
    lane_x = frame.lane_center_x(pol_b.target_lane)

    def _lead(speed: float, along: float) -> Neighbour:
        return Neighbour(actor_id="L", lane=pol_b.target_lane, lat=lane_x,
                         along=along, gap=max(along - 4.5, 0.5), speed=speed,
                         oncoming=False, length=4.5, width=2.0)

    ck.check("ego/MOBIL: a stopped blocker lifts the low-speed hold",
             pol_b._blocked([_lead(0.0, 7.0)]),
             f"stopped {7.0 - 4.5:.1f} m ahead, ego at 0.4 m/s "
             f"(floor {MOBIL_V_MIN} m/s)")
    ck.check("ego/MOBIL: moving traffic does not lift it",
             not pol_b._blocked([_lead(6.0, 7.0)]),
             "a lead still doing 6 m/s is not a deadlock")
    ck.check("ego/MOBIL: a distant stopped car does not lift it",
             not pol_b._blocked([_lead(0.0, BLOCKED_GAP_M + 20.0)]),
             f"stopped but {BLOCKED_GAP_M + 20.0:.0f} m off is not blocking us")

    # ...and having lifted it, the ego must have room to actually get out.
    from .highway_ego import BLOCKED_STANDOFF_M, LC_DISTANCE_SLOW, IDM_S0
    blocked = [_lead(0.0, 9.0)]
    a_stand = pol_b._idm_accel(pol_b._gap_to(blocked[0]), 0.0,
                               s0=BLOCKED_STANDOFF_M)
    a_jam = pol_b._idm_accel(pol_b._gap_to(blocked[0]), 0.0, s0=IDM_S0)
    ck.check("ego/MOBIL: stands off further from a blocker it may go round",
             a_stand < a_jam,
             f"4.5 m gap: {a_stand:+.2f} vs {a_jam:+.2f} m/s^2 — IDM's own "
             f"{IDM_S0} m jam distance parks it nose-to-tail, with no room "
             "left to swing out")
    pol_b.ego.v = 0.0
    pol_b.lc = None
    pol_b._commit(1 if pol_b.target_lane == 0 else 0, now=100.0)
    ck.check("ego/MOBIL: a change from a standstill uses the short profile",
             pol_b.lc is not None
             and pol_b.lc["L"] == LC_DISTANCE_SLOW
             and LC_DISTANCE_SLOW <= BLOCKED_STANDOFF_M + 2.0,
             f"L={pol_b.lc['L']} m, inside the {BLOCKED_STANDOFF_M} m standoff; "
             f"a {LC_DISTANCE:.0f} m highway profile cannot clear a car that close")
    pol_b.ego.v = 12.0
    pol_b.lc = None
    pol_b._last_change = -1e9
    pol_b._commit(pol_b.target_lane ^ 1, now=200.0)
    ck.check("ego/MOBIL: a change at speed still uses the highway profile",
             pol_b.lc is not None and pol_b.lc["L"] == LC_DISTANCE,
             f"L={pol_b.lc['L']} m at 12 m/s")

    # MOBIL must evaluate the blocker at the SAME standoff idm_control holds,
    # or the ego parks at arm's length and never decides to go round.
    ego_s = Ego(x=x, y=y, theta=math.radians(h), v=0.0)
    pol_s = HighwayEgoPolicy(frame, ego_s, bg, atime=0.0)
    stopped = [_lead(0.0, BLOCKED_STANDOFF_M + 4.5)]     # 8 m bumper to bumper
    ck.check("ego/MOBIL: idm_control and MOBIL agree on the standoff",
             pol_s._ego_s0(stopped) == BLOCKED_STANDOFF_M,
             "one question, asked once, used by both")
    other = 1 if pol_s.target_lane == 0 else 0
    ok_go, gain_go, why_go = pol_s._mobil_evaluate(pol_s.target_lane, other,
                                                   stopped)
    a_jam2 = pol_s._idm_accel(BLOCKED_STANDOFF_M, 0.0, s0=IDM_S0)
    ck.check("ego/MOBIL: a stopped blocker at the standoff is worth going round",
             ok_go,
             f"{why_go}; at IDM's jam distance the same car scores "
             f"{a_jam2:+.2f} m/s^2 and the change is refused")

    # Contraflow must be dear to enter and cheap to leave. Once past the
    # blocker both lanes are free, the incentive is a dead tie, and a tie loses
    # — so without the second half the ego stays on the wrong side of the road.
    try:
        frame_o, ego_ao, bg_o, _ = sc_mod.build(world, sc_mod.OVERTAKE)
    except RuntimeError:
        frame_o = None
    if frame_o is not None and frame_o.oncoming_lanes():
        onc = frame_o.lane_index_for_direction(False)
        home = frame_o.lane_index_for_direction(True)
        xo, yo, ho = ego_ao.start
        pol_o = HighwayEgoPolicy(frame_o,
                                 Ego(x=frame_o.lane_center_x(onc), y=yo,
                                     theta=math.radians(ho), v=12.0),
                                 bg_o, atime=0.0)
        pol_o.home_lane = home
        pol_o.target_lane = onc
        _, _, why_out = pol_o._mobil_evaluate(home, onc, [])
        _, _, why_back = pol_o._mobil_evaluate(onc, home, [])
        thr_out = float(why_out.rsplit(" ", 1)[-1])
        thr_back = float(why_back.rsplit(" ", 1)[-1])
        ck.check("ego/MOBIL: leaving the oncoming lane is cheaper than entering",
                 thr_back < 0.0 < thr_out,
                 f"threshold out {thr_out:+.2f} vs back {thr_back:+.2f} — "
                 "coming home must not have to win a tie")


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


def check_route_offer(ck: Checks, cmap) -> None:
    """Who picks the ego's lane while an external policy drives.

    The companion's MOBIL offers a lane change through the route, which is how
    a route-following policy is told to merge. A policy that picks its own lane
    gets the route held instead: on hard_brake the offer landed on top of
    idm_mobil's own change and put the ego two lanes over.
    """
    from . import scenarios as sc_mod
    from .highway_ego import Ego, HighwayEgoPolicy
    from .runner import HighwayRun, RunConfig, _chooses_own_lane, _harness_root
    world = _World(cmap)

    class _FollowsRoute:                    # tfv6, simlingo, plant2
        pass

    class _PicksLane:                       # third_party/idm's IDMMobilPolicy
        allows_lane_change = True

    idm_root = os.path.join(_harness_root() or "", "third_party", "idm")
    name = "route offer: third_party/idm's policies pick their own lane"
    if os.path.isfile(os.path.join(idm_root, "idm", "policy.py")):
        if idm_root not in sys.path:
            sys.path.insert(0, idm_root)
        from idm.policy import IDMMobilPolicy, IDMPolicy
        ck.check(name, _chooses_own_lane(IDMMobilPolicy())
                 and _chooses_own_lane(IDMPolicy()),
                 f"read from {idm_root}; the port keys on `allows_lane_change`")
    else:
        ck.add(_SKIP, name, f"no {idm_root}")
    ck.check("route offer: a policy that declares nothing follows the route",
             not _chooses_own_lane(_FollowsRoute()))

    try:
        frame, ego_a, bg, _ = sc_mod.build(world, sc_mod.HARD_BRAKE)
    except RuntimeError as exc:
        ck.add(_SKIP, "route offer: hard_brake", str(exc))
        return
    x, y, h = ego_a.start
    home_x = frame.lane_center_x(frame.lane_index_of(x))
    dt = 1.0 / 60.0

    def drive(policy):
        """~6.7 s behind the slow lead, the ego braking under IDM in its own
        lane: all that changes is what the companion does to the route."""
        run = HighwayRun.__new__(HighwayRun)
        run.cfg = RunConfig(scenario=sc_mod.HARD_BRAKE, verbose=False)
        run.notes = []
        run.policy = HighwayEgoPolicy(
            frame, Ego(x=x, y=y, theta=math.radians(h), v=12.0), bg, atime=0.0)
        if _chooses_own_lane(policy):
            run._hold_route("idm_mobil")
        ego = run.policy.ego
        for k in range(400):
            run.t_sim = run.policy.atime = k * dt
            accel = run.policy.commanded_accel(run.policy.idm_control())
            ego.v = max(0.0, ego.v + accel * dt)
            ego.y += ego.v * dt
            run._advance_route(dt)
        off = max(abs(px - home_x) for px, _py in run.policy.reference_path)
        return run, off

    run, off = drive(_FollowsRoute())
    ck.check("route offer: a route-following policy is offered the merge",
             run.policy.n_lane_changes == 1 and off > frame.lane_width / 2.0,
             f"{run.policy.n_lane_changes} route change(s); the route reaches "
             f"{off:.2f} m off the home lane")
    run, off = drive(_PicksLane())
    ck.check("route offer: held for a policy that picks its own lane",
             run.policy.n_lane_changes == 0 and off < 1e-6 and bool(run.notes),
             f"{run.policy.n_lane_changes} route change(s); the route reaches "
             f"{off:.2f} m off the home lane")

    try:
        frame_o, ego_o, bg_o, _ = sc_mod.build(world, sc_mod.OVERTAKE)
    except RuntimeError as exc:
        ck.add(_SKIP, "route offer: overtake grade", str(exc))
        return
    run = HighwayRun.__new__(HighwayRun)
    run.cfg = RunConfig(scenario=sc_mod.OVERTAKE, verbose=False)
    run.notes, run.realized, run.interactions = [], [], {}
    xo, yo, ho = ego_o.start
    run.policy = HighwayEgoPolicy(
        frame_o, Ego(x=xo, y=yo, theta=math.radians(ho), v=0.0), bg_o, atime=0.0)
    run._hold_route("idm_mobil")
    run.policy.ego.x = frame_o.lane_center_x(frame_o.lane_index_for_direction(False))
    away = run._grade()["checks"]["home"]
    run.policy.ego.x = xo
    back = run._grade()["checks"]["home"]
    ck.check("route offer: with the route held, overtake's `home` is the ego's lane",
             not away and back,
             f"home={away} in the oncoming lane, home={back} in its own; the "
             "companion stays home whatever the ego does")


def check_heading_smoothing(ck: Checks, cmap) -> None:
    """The rendered yaw must not carry the orchestrator's replan sawtooth."""
    from carla_port.carla_adapter import ScriptState
    from carla_port.carla_sync import (DT, HEADING_MOTION, HEADING_PLAN,
                                       HEADING_RATE_MAX, StateSynchronizer)

    class _Frame:                       # heading_of touches nothing else
        pass

    # A car driving straight up +y at 12 m/s whose PLAN heading is reset to the
    # nominal lane heading every 0.10 s and swung out again in between — the
    # measured shape of the cut-in actor's yaw during its merge.
    def _states(n: int):
        for k in range(n):
            phase = k % 6
            plan_h = 90.0 - 9.0 * (phase / 5.0)      # 90 -> 81, then snap back
            yield ScriptState("4", -3.5 + 0.02 * k, 12.0 * DT * k, plan_h, 12.0)

    def _yaws(mode: str):
        sync = StateSynchronizer.__new__(StateSynchronizer)
        sync.heading = mode
        sync._last_xy, sync._yaw = {}, {}
        return [sync.heading_of(st, DT) for st in _states(120)][10:]

    plan = _yaws(HEADING_PLAN)
    motion = _yaws(HEADING_MOTION)
    swing_plan = max(plan) - min(plan)
    swing_motion = max(motion) - min(motion)
    ck.check("sync: heading from motion removes the replan sawtooth",
             swing_motion < swing_plan / 5.0,
             f"peak-to-peak yaw {swing_motion:.2f} deg vs {swing_plan:.2f} deg "
             "written verbatim from the plan")

    rate = max(abs((b - a + 180.0) % 360.0 - 180.0) / DT
               for a, b in zip(motion, motion[1:]))
    ck.check("sync: the rendered yaw stays inside a vehicle's turn rate",
             rate <= HEADING_RATE_MAX + 1e-6,
             f"worst {rate:.1f} deg/s, cap {HEADING_RATE_MAX} deg/s")

    # ...and the lean must stay inside a plausible vehicle attitude even when
    # the plan itself is not drivable (a replanned cut-in slides sideways
    # faster than it drives forward).
    from carla_port.carla_sync import HEADING_MAX_SLIP
    sync = StateSynchronizer.__new__(StateSynchronizer)
    sync.heading = HEADING_MOTION
    sync._last_xy, sync._yaw = {}, {}
    crab = []
    for k in range(120):
        # 3 m/s forward, 4.5 m/s sideways: a true direction of travel ~56 deg
        # off the road, which is what the merge actually commands
        crab.append(sync.heading_of(
            ScriptState("4", 0.0 + 4.5 * DT * k, 3.0 * DT * k, 90.0, 3.0), DT))
    worst = max(abs((h - 90.0 + 180.0) % 360.0 - 180.0) for h in crab)
    ck.check("sync: an undrivable plan is not rendered as a drift",
             worst <= HEADING_MAX_SLIP + 1e-6,
             f"worst lean {worst:.1f} deg, capped at {HEADING_MAX_SLIP} deg; "
             "the raw direction of travel here is ~56 deg off the road")

    # A stopped actor has no direction of travel and must not spin.
    sync = StateSynchronizer.__new__(StateSynchronizer)
    sync.heading = HEADING_MOTION
    sync._last_xy, sync._yaw = {}, {}
    still = [sync.heading_of(ScriptState("1", 5.0, 5.0, 90.0, 0.0), DT)
             for _ in range(40)]
    ck.check("sync: a stopped actor holds its heading",
             max(still) - min(still) < 1e-9,
             "no direction of travel to read, so the last good one is held")


def check_verify_roles(ck: Checks, cmap) -> None:
    """The role labels the upstream overtake / hard_brake verifiers look up."""
    from . import scenarios as sc_mod
    from .runner import HighwayRun, RunConfig
    world = _World(cmap)
    for mode, want in ((sc_mod.OVERTAKE, {"blocker", "oncoming"}),
                       (sc_mod.HARD_BRAKE, {"slow", "adjacent"})):
        try:
            frame, ego_a, bg, _ = sc_mod.build(world, mode)
        except RuntimeError as exc:
            ck.add(_SKIP, f"verify roles: {mode}", str(exc))
            continue
        run = HighwayRun.__new__(HighwayRun)
        run.cfg = RunConfig(scenario=mode)
        run.frame = frame
        run._spawns = {sc_mod.EGO_ID: [ego_a.start[0], ego_a.start[1],
                                       ego_a.start[2], 4.5, 2.0]}
        run._authored_roles = {}
        for a in bg.actors:
            run._spawns[str(a.id)] = [a.start[0], a.start[1], a.start[2],
                                      a.length, a.width]
            if getattr(a, "role", None):
                run._authored_roles[str(a.id)] = str(a.role)
        roles = run._derive_roles()
        ck.check(f"verify roles: {mode} labels both parts",
                 set(roles.values()) == want and len(roles) == len(bg.actors),
                 f"{roles} — without these the upstream verifier refuses the "
                 "run with 'missing ... role' before checking anything")
        # and the geometric fallback must agree with what the YAML authored
        run._authored_roles = {}
        ck.check(f"verify roles: {mode} derivation agrees with the YAML",
                 run._derive_roles() == roles,
                 "the fallback reads the same roles off the spawn geometry")


def check_closed_loop(ck: Checks, cmap) -> None:
    """The cut-in director, driven headless off the real road."""
    import cutin_director as cd
    from . import scenarios as sc_mod
    from .closed_loop import HighwayClosedLoop
    from .highway_ego import Ego, HighwayEgoPolicy
    world = _World(cmap)
    try:
        frame = sc_mod.discover_frame(world, sc_mod.CUTIN)
    except RuntimeError as exc:
        ck.add(_SKIP, "closed loop: cut-in director", str(exc))
        return
    path = sc_mod.scenario_path(sc_mod.CUTIN.replace("cutin", "cutin")) \
        if False else os.path.join(os.path.dirname(sc_mod.scenario_path(sc_mod.CUTIN)),
                                   "scenario_cutin_single.yaml")
    fitted, _ = sc_mod.retarget(sc_mod.load(sc_mod.CUTIN, path), frame)
    # observed from script poses below, so the script heading model applies
    director = cd.load_director(path, fitted, heading=cd.HEADING_SCRIPT)
    ego_a, bg = sc_mod.split_ego(fitted)
    loop = HighwayClosedLoop(frame, bg, sc_mod.spec(sc_mod.CUTIN),
                             director=director)
    ck.check("closed loop: the director runs for cutin", loop.casting,
             f"spec: {director.spec.to_dict() if director else None}")
    x, y, h = ego_a.start
    ego = Ego(x=x, y=y, theta=math.radians(h), v=13.0)
    pol = HighwayEgoPolicy(frame, ego, loop.sc, atime=0.0)
    dt = 1.0 / 60.0
    for k in range(600):                       # 10 s
        t = k * dt
        loop.tick(ego, t)
        pol.asc, pol.atime = loop.sc, loop.atime
        thr, st = pol.command(now=t, dt=dt)
        pol.integrate(thr, st, dt)
        loop.advance(dt)
        states = {}
        for a in loop.sc.actors:
            i = max(0, min(int(round(loop.atime / dt)), len(a.traj) - 1))
            if a.traj:
                ax, ay, ah = a.traj[i]
                states[a.id] = (ax, ay, ah, a.speeds[i])
        loop.observe(ego, t + dt, states)
    summ = director.summary() if director else {}
    c = summ.get("cut_in") or {}
    ck.check("closed loop: an actor holds the cut-in", loop.holder is not None,
             f"holder={loop.holder} scores={loop.scores}")
    ck.check("closed loop: the trigger fired after t=3 s",
             summ.get("t_trigger") is not None and summ["t_trigger"] > 3.0,
             f"t_trigger={summ.get('t_trigger')}")
    ck.check("closed loop: the actor entered the ego's lane", bool(c),
             f"outcome={summ.get('outcome')}")
    ck.check("closed loop: gap at the cut-in instant is on target",
             bool(c) and abs(c["gap_error_m"]) <= 0.5,
             f"gap {c.get('gap_m')} m, asked {director.spec.gap_m if director else '-'}")
    ck.check("closed loop: relative speed at the cut-in instant is on target",
             bool(c) and abs(c["rel_speed_error_mps"]) <= 1.0,
             f"dv {c.get('rel_speed_mps')} m/s, asked "
             f"{director.spec.rel_speed_mps if director else '-'}")
    ck.check("closed loop: it logged what it did", len(loop.events) > 0,
             f"{len(loop.events)} events; last: "
             f"{loop.events[-1].text if loop.events else '-'}")

    # a non-casting mode must leave the scripts alone
    frame2, _e2, bg2, _ = sc_mod.build(world, sc_mod.HARD_BRAKE)
    loop2 = HighwayClosedLoop(frame2, bg2, sc_mod.spec(sc_mod.HARD_BRAKE))
    ck.check("closed loop: hard_brake does NOT cast roles",
             not loop2.casting and loop2.director is None,
             "scripted actors stay scripted")


def check_actuation(ck: Checks) -> None:
    """The built-in ego's actuator, and the steering handedness."""
    from carla_port.actuation import CarlaEgoActuator
    from .highway_ego import A_BRAKE, A_THROTTLE, DELTA_MAX, V_MAX

    class _Wheel:
        max_steer_angle = 70.0

    class _Phys:
        wheels = [_Wheel(), _Wheel()]

    class _Veh:
        def get_physics_control(self):
            return _Phys()

    class _Ego:
        v = 4.0

    class _Policy:
        ego = _Ego()

        @staticmethod
        def commanded_accel(throttle):
            return throttle * (A_THROTTLE if throttle > 0 else A_BRAKE)

    act = CarlaEgoActuator(_Veh(), delta_max=DELTA_MAX, v_max=V_MAX)
    ctl, _ref = act.control(_Policy(), 0.3, 0.0, 1.0 / 60.0)
    ck.check("actuator: accelerates when the policy asks to",
             ctl.throttle > 0 and ctl.brake == 0.0, f"throttle={ctl.throttle:.2f}")
    act = CarlaEgoActuator(_Veh(), delta_max=DELTA_MAX, v_max=V_MAX)
    _Policy.ego.v = 0.0
    ctl, _ref = act.control(_Policy(), -0.5, 0.0, 1.0 / 60.0)
    ck.check("actuator: a stopped car asked to brake is held on the brake",
             ctl.brake == 1.0 and ctl.throttle == 0.0)
    ck.check("actuator: steer sign is flipped for CARLA", act.steer_scale < 0,
             f"scale={act.steer_scale:.4f} (script CCW -> CARLA CW)")
    ck.check("actuator: steer scale maps DELTA_MAX to the wheel limit",
             abs(abs(act.steer_scale) - math.degrees(DELTA_MAX) / 70.0) < 1e-9)

    check_spawn_gear_and_tracking(ck)


def check_spawn_gear_and_tracking(ck: Checks) -> None:
    """The spawn phase, and an acceleration policy driven through the tracker."""
    from carla_port.actuation import (HOLD_THROTTLE, SPAWN_GEAR_HOLD_S,
                                      SPAWN_MAX_S, SPAWN_REV_S, SpawnGear)
    from carla_port.carla_api import carla
    from carla_port.ego_driver import PolicyEgoDriver

    class _Gear:
        def __init__(self, ratio):
            self.ratio, self.up_ratio, self.down_ratio = ratio, 0.46, 0.23

    class _Wheel:
        radius = 35.0                                   # cm, as CARLA reports
        max_steer_angle = 70.0

    class _Mkz:                                         # vehicle.lincoln.mkz_2020
        forward_gears = [_Gear(r) for r in (4.58, 2.96, 1.91, 1.45, 1.0, 0.75)]
        final_ratio, max_rpm = 3.21, 6500.0
        wheels = [_Wheel()] * 4

    class _Telemetry:
        def __init__(self, rpm):
            self.engine_rpm = rpm

    class _Veh:
        """Rolls along +x at `v`, with `vz` of vertical speed. With
        `rpm_per_step`, reports engine rpm that rises that much per step of
        full throttle, as `get_telemetry_data` does on a live server."""

        def __init__(self, v=0.0, vz=0.0, rpm_per_step=None):
            self.v, self.vz = v, vz
            self.rpm_per_step = rpm_per_step
            self.rpm = 0.0
            self.velocity_sets = []

        def get_physics_control(self):
            return _Mkz()

        def get_velocity(self):
            return carla.Vector3D(self.v, 0.0, self.vz)

        def set_target_velocity(self, vel):
            self.velocity_sets.append(vel.x)
            self.v = vel.x

        def get_telemetry_data(self):
            if self.rpm_per_step is None:
                raise AttributeError("no telemetry")
            return _Telemetry(self.rpm)

        def get_control(self):
            return carla.VehicleControl()

    gears = {v: (SpawnGear.choose(_Veh(), v) or {}).get("gear")
             for v in (0.5, 7.0, 13.0, 40.0)}
    ck.check("spawn gear: the highest gear the gearbox would keep",
             gears == {0.5: None, 7.0: 2, 13.0: 4, 40.0: 6},
             f"{gears} (m/s -> gear); first gear at 7 m/s is what dragged "
             "the IDM ego from 7 to 5 m/s")
    dt = 1.0 / 60.0

    def run(veh, demand, pedal_brake=None, steps=180):
        """(rev steps, velocity sets, manual-gear steps, plan) over `steps`."""
        sg, revving, geared = SpawnGear(), 0, 0
        for k in range(steps):
            accel = None if pedal_brake is not None else demand(k * dt)
            pedals, kw = sg.step(veh, veh.v, dt, accel=accel,
                                 brake=pedal_brake or 0.0)
            if pedals is not None and kw.get("gear") == 0:
                revving += 1
                if veh.rpm_per_step is not None and pedals[0] > 0.0:
                    veh.rpm += veh.rpm_per_step
            elif kw.get("manual_gear_shift"):
                geared += 1
        return revving, len(veh.velocity_sets), geared, sg.plan

    hold = round(SPAWN_GEAR_HOLD_S / dt)
    veh = _Veh(13.0)
    revving, sets, geared, plan = run(veh, lambda t: -3.0 if t < 1.0 else 0.2)
    ck.check("spawn phase (acceleration policy, no telemetry): driven "
             "kinematically through a hard brake, engaged once it eases",
             revving == round(1.0 / dt) and sets == revving + 1
             and geared == hold + 1 and plan["gear"] == 3
             and abs(plan["speed_mps"] - 10.0) < 0.1,
             f"{revving} rev steps, {sets} velocity sets, {geared} manual-gear "
             f"steps; engaged in gear {plan['gear']} at {plan['speed_mps']} m/s "
             "(13 m/s braked at -3 for 1 s)")
    veh = _Veh(13.0)
    revving, sets, geared, plan = run(veh, lambda t: 0.5)
    ck.check("spawn phase (no telemetry): revs for SPAWN_REV_S",
             revving == round(SPAWN_REV_S / dt) and not plan["rpm_matched"],
             f"{revving} rev steps")
    veh = _Veh(13.0, rpm_per_step=60.0)                 # gear 4, ~1650 rpm
    revving, sets, geared, plan = run(veh, lambda t: 0.5)
    ck.check("spawn phase (telemetry): engaged on the first step the engine "
             "is at the gear's rpm", plan["rpm_matched"]
             and 60.0 * (revving - 1) < 0.97 * plan["engine_rpm"] <= veh.rpm,
             f"{revving} rev steps to {veh.rpm:.0f} rpm for gear "
             f"{plan['gear']} at {plan['engine_rpm']:.0f}")
    veh = _Veh(13.0, rpm_per_step=60.0)
    revving, sets, geared, plan = run(veh, None, pedal_brake=0.4)
    ck.check("spawn phase (pedal policy): not driven, its brake applied "
             "while the engine revs", sets == 0 and plan is not None
             and plan["kinematic"] is False and revving > 0,
             f"{sets} velocity sets, {revving} rev steps")
    veh = _Veh(13.0, rpm_per_step=0.0)                  # an engine that never revs
    revving, sets, geared, plan = run(veh, lambda t: 0.5, steps=240)
    ck.check("spawn phase: engaged after SPAWN_MAX_S whatever the engine does",
             revving == round(SPAWN_MAX_S / dt) and geared == hold + 1,
             f"{revving} rev steps")
    revving, sets, geared, plan = run(_Veh(0.0, vz=-2.4), lambda t: 1.5)
    ck.check("spawn phase: a car settling onto the road is not rolling",
             plan is None and revving == 0,
             "2.4 m/s of vertical speed at spawn is left to the gearbox")

    class _LateVeh(_Veh):                   # the spawn velocity lands a tick late
        def __init__(self):
            super().__init__(0.0)
            self.calls = 0

        def get_velocity(self):
            self.calls += 1
            return carla.Vector3D(0.0 if self.calls == 1 else 13.0, 0.0, 0.0)

    sg, late = SpawnGear(), _LateVeh()
    sg.step(late, 0.0, dt, accel=0.5)
    sg.step(late, 0.0, dt, accel=0.5)
    ck.check("spawn phase: a spawn velocity that lands a tick late still "
             "gets one", sg.plan is not None and sg.plan["spawn_speed_mps"] == 13.0,
             f"plan {sg.plan}")

    class _Ego:
        v = 5.0

    class _Companion:
        ego = _Ego()

    class _NoGearbox:                       # no gear table: no spawn gear
        def get_control(self):
            return carla.VehicleControl()

    class _Ctx:
        policy = _Companion()
        ego_actor = _NoGearbox()

    drv = PolicyEgoDriver(policy=None, name="idm")
    drv.ctx = _Ctx()
    drv._command = drv._command_from({"acceleration_mps2": 1.3, "steer": 0.0})
    first = drv._vehicle_control(carla, dt, True)
    for _ in range(int(3.0 / dt)):                      # the car does not respond
        later = drv._vehicle_control(carla, dt, False)
    ck.check("acceleration policy: feedback keeps pushing a car that is not "
             "speeding up", later.throttle > first.throttle + 0.1,
             f"throttle {first.throttle:.2f} -> {later.throttle:.2f} after 3 s "
             "at +1.3 m/s^2 asked")
    drv.tracker.reset()
    _Companion.ego.v = 10.0
    drv._command = drv._command_from({"acceleration_mps2": -6.0})
    hard = drv._vehicle_control(carla, dt, True)
    ck.check("acceleration policy: a hard deceleration is a hard brake",
             hard.brake > 0.9 and hard.throttle == 0.0, f"brake={hard.brake:.2f}")
    from carla_port.actuation import AccelerationTracker
    trk, v, most = AccelerationTracker(), 12.0, 0.0
    for _ in range(60):                     # asked -3, engine braking gives -6
        thr, _brk = trk.step(-3.0, v, dt)
        most = max(most, thr)
        v = max(0.0, v - 6.0 * dt)
    ck.check("tracker: throttle against engine braking that overshoots a "
             "braking demand", most > 0.3,
             f"max throttle {most:.2f} while slowing at -6 when -3 was asked")
    trk, v = AccelerationTracker(), 12.0
    first = trk.step(-3.0, v, dt)[1]
    for _ in range(60):                     # asked -3, the car slows at -1
        _thr, brk = trk.step(-3.0, v, dt)
        v -= 1.0 * dt
    ck.check("tracker: more brake for a car slowing slower than asked",
             brk > first + 0.3, f"brake {first:.2f} -> {brk:.2f} after 1 s")
    trk.reset(prime=True)
    ck.check("tracker: a primed reset starts from the throttle that holds a "
             "speed", abs(trk.step(0.0, 10.0, dt)[0] - HOLD_THROTTLE) < 1e-6,
             f"HOLD_THROTTLE={HOLD_THROTTLE}")
    ck.check("tracker: a stopped car asked to stay stopped is held on the brake",
             trk.step(-1.0, 0.0, dt) == (0.0, 1.0))
    trk, flips, last = AccelerationTracker(), 0, None
    for k in range(120):                    # IDM near its desired speed
        trk.step(0.35 if k == 0 else (0.25 if k % 2 else 0.15), 8.0, dt)
        flips += int(last is not None and trk._regime != last)
        last = trk._regime
    ck.check("tracker: demand dithering at a threshold does not restart it",
             flips == 0, f"{flips} regime changes")
    drv._command = drv._command_from({"control": {"throttle": 0.3, "brake": 0.0,
                                                  "steer": 0.1}})
    pedals = drv._vehicle_control(carla, dt, True)
    ck.check("pedal policy: its own control is applied as given",   # float32
             abs(pedals.throttle - 0.3) < 1e-6 and abs(pedals.steer - 0.1) < 1e-6,
             f"throttle={pedals.throttle:.4f} steer={pedals.steer:.4f}")


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
        print("== route offer ==")
        check_route_offer(ck, first)
        print("== sync ==")
        check_heading_smoothing(ck, first)
        print("== verify roles ==")
        check_verify_roles(ck, first)
        print("== closed loop ==")
        check_closed_loop(ck, first)
    else:
        ck.add(_SKIP, "ego policy + closed loop", "no town map available")

    print(ck.summary())
    return ck.failed


if __name__ == "__main__":
    sys.exit(main())
