#!/usr/bin/env python3
"""Trajectory and scenario-stage verification for cut-in, overtake, hard_brake.

Used by ``summarize_experiments.py`` and ``verify_run.py``.  Each verifier
returns ``(ok, reason, details)`` where ``details["checks"]`` maps criterion
keys to booleans.

See ``docs/SCENARIOS_AND_VALIDATION.md`` for natural-language descriptions.
"""
from __future__ import annotations

import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# import rect overlap from directives (headless, no pygame)
_HERE = __file__
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from directives import rect_corners, rects_overlap  # noqa: E402

Pose = Tuple[float, float, float]

DEFAULT_LANE_WIDTH = 3.5
DEFAULT_CRUISE_MPS = 12.0
DEFAULT_VEHICLE = (4.5, 2.0)

# --- cut-in (aligned with summarize_experiments) --------------------------- #
MAX_AHEAD_M = 10.0
MIN_AHEAD_M = 0.5
ADJ_LANE_MIN_FRAC = 0.4
ADJ_LANE_MAX_FRAC = 1.6
IN_LANE_FRAC = 0.25
TIME_EPS = 0.02
POST_MERGE_WINDOW_S = 1.5
NOMINAL_SPEED_FRAC = 0.30
NOMINAL_SPEED_ABS = 3.0
NOMINAL_MIN_MPS = 6.0
MIN_POST_SAMPLES = 3

# --- overtake -------------------------------------------------------------- #
OVERTAKE_ONCOMING_REACTION_MIN_M = 12.0   # min longitudinal gap at start
OVERTAKE_ONCOMING_SPEED_MIN = 5.0
OVERTAKE_ONCOMING_SPEED_MAX = 22.0
OVERTAKE_ADJ_CLEAR_AHEAD_M = 25.0         # no third actor within this ahead

# --- hard_brake ------------------------------------------------------------ #
HARD_BRAKE_TRIGGER_GAP_M = 6.0
HARD_BRAKE_TRIGGER_TOL_M = 1.0            # band around trigger for discrete traj
HARD_BRAKE_MIN_DECEL = -2.5               # m/s² required once within trigger gap
HARD_BRAKE_PRETRIGGER_MIN_MPS = 8.0       # must still be moving before gap closes
HARD_BRAKE_ADJ_TO_LEAD_MIN_M = 2.0        # adjacent↔lead longitudinal band
HARD_BRAKE_ADJ_TO_LEAD_MAX_M = 35.0

COLLISION_PAD = 0.3


def _heading_axes(hd_deg: float) -> Tuple[float, float, float, float]:
    h = math.radians(hd_deg)
    fx, fy = math.cos(h), math.sin(h)
    return fx, fy, -fy, fx


def world_to_ego_offset(ego_pose: Pose, wx: float, wy: float) -> Tuple[float, float]:
    ex, ey, eh = ego_pose
    fx, fy, nx, ny = _heading_axes(eh)
    dx, dy = wx - ex, wy - ey
    return dx * fx + dy * fy, dx * nx + dy * ny


def pose_at_time(traj: List[list], t: float) -> Optional[Pose]:
    if not traj:
        return None
    best = traj[0]
    for row in traj:
        if row[0] <= t + 1e-6:
            best = row
        else:
            break
    return (float(best[1]), float(best[2]), float(best[3]))


def infer_lane_width(data: dict) -> float:
    if "lane_width" in data:
        return float(data["lane_width"])
    ego = data.get("ego_trajectory") or []
    spawns = data.get("spawns") or {}
    if not ego or not spawns:
        return DEFAULT_LANE_WIDTH
    ex = float(ego[0][1])
    gaps = [abs(float(sp[0]) - ex) for sp in spawns.values() if abs(float(sp[0]) - ex) > 0.1]
    return min(gaps) if gaps else DEFAULT_LANE_WIDTH


def _in_lane_x(x: float, lane_x: float, lw: float, frac: float = IN_LANE_FRAC) -> bool:
    return abs(x - lane_x) < frac * lw


def _is_adjacent_lat(lat: float, lw: float) -> bool:
    a = abs(lat)
    return ADJ_LANE_MIN_FRAC * lw <= a <= ADJ_LANE_MAX_FRAC * lw


def _same_lane_lat(lat: float, lw: float) -> bool:
    return abs(lat) < ADJ_LANE_MIN_FRAC * lw


def _speed_samples(traj: List[list], t0: float, t1: float) -> List[Tuple[float, float]]:
    """(t, v) samples in [t0, t1]."""
    out: List[Tuple[float, float]] = []
    prev = None
    for row in traj:
        t = float(row[0])
        if t < t0 - 1e-6:
            prev = row
            continue
        if t > t1 + 1e-6:
            break
        if len(row) >= 5:
            out.append((t, float(row[4])))
        elif prev is not None:
            dt = t - float(prev[0])
            if dt > 1e-3:
                ds = math.hypot(float(row[1]) - float(prev[1]),
                                float(row[2]) - float(prev[2]))
                out.append((t, ds / dt))
        prev = row
    return out


def _vehicle_dims(data: dict, aid: str) -> Tuple[float, float]:
    vd = (data.get("vehicle_dims") or {}).get(aid)
    if vd:
        return float(vd["length"]), float(vd["width"])
    sp = (data.get("spawns") or {}).get(aid)
    if sp and len(sp) >= 5:
        return float(sp[3]), float(sp[4])
    return DEFAULT_VEHICLE


def _role_id(data: dict, role: str) -> Optional[str]:
    roles = data.get("roles") or {}
    if role in roles:
        return str(roles[role])
    for aid, r in (data.get("actor_roles") or {}).items():
        if r == role:
            return str(aid)
    return None


def _longitudinal_gap_y(pose_a: Pose, pose_b: Pose) -> float:
    """Approximate along-road gap (northbound axis): positive if b is ahead of a."""
    return pose_b[1] - pose_a[1]


def _bodies_overlap(p1: Pose, dim1: Tuple[float, float],
                    p2: Pose, dim2: Tuple[float, float]) -> bool:
    l1, w1 = dim1
    l2, w2 = dim2
    pad = COLLISION_PAD
    b1 = rect_corners(p1[0], p1[1], p1[2], l1 + pad, w1 + pad)
    b2 = rect_corners(p2[0], p2[1], p2[2], l2 + pad, w2 + pad)
    return rects_overlap(b1, b2)


def _any_collision(data: dict, t_end: Optional[float] = None) -> Optional[str]:
    """Return 'ego-<id>' if ego body overlaps an actor at any sample time."""
    ego_tr = data.get("ego_trajectory") or []
    actors = data.get("actor_trajectories") or {}
    if not ego_tr:
        return None
    el, ew = _vehicle_dims(data, "0")
    for row in ego_tr:
        t = float(row[0])
        if t_end is not None and t > t_end + 1e-6:
            break
        ep = (float(row[1]), float(row[2]), float(row[3]))
        for aid, tr in actors.items():
            ap = pose_at_time(tr, t)
            if ap is None:
                continue
            al, aw = _vehicle_dims(data, aid)
            if _bodies_overlap(ep, (el, ew), ap, (al, aw)):
                return f"ego-{aid}@{t:.2f}s"
    return None


def detect_scenario(data: dict) -> str:
    if data.get("scenario"):
        return str(data["scenario"])
    if (data.get("cast") or {}).get("cutin"):
        return "cutin"
    roles = set((data.get("roles") or {}).values())
    roles |= set((data.get("actor_roles") or {}).values())
    if "blocker" in roles or "oncoming" in roles:
        return "overtake"
    if "slow" in roles or "adjacent" in roles:
        return "hard_brake"
    return "unknown"


def verify_proper_cutin(data: dict,
                        lane_width: Optional[float] = None
                        ) -> Tuple[bool, str, Dict[str, Any]]:
    """Cut-in: four checks (see docs/SCENARIOS_AND_VALIDATION.md)."""
    cast = data.get("cast") or {}
    rec = data.get("cutin") or {}
    performer = rec.get("performer") or cast.get("cutin")
    if performer is None:
        return False, "no cut-in performer", {}

    lw = lane_width if lane_width is not None else infer_lane_width(data)
    spawns = data.get("spawns") or {}
    if performer not in spawns:
        return False, f"spawn missing for actor {performer}", {}

    ego_tr = data.get("ego_trajectory") or []
    actor_tr = (data.get("actor_trajectories") or {}).get(performer)
    if not ego_tr or not actor_tr:
        return False, "missing trajectories", {}

    ego_home_x = float(ego_tr[0][1])
    actor_home_x = float(spawns[performer][0])
    cruise = float((data.get("cruise") or {}).get(performer, DEFAULT_CRUISE_MPS))
    details: Dict[str, Any] = {
        "performer": performer, "lane_width": lw, "checks": {},
        "cruise": cruise,
    }

    t_start = max(float(ego_tr[0][0]), float(actor_tr[0][0]))
    ego0 = pose_at_time(ego_tr, t_start)
    actor0 = pose_at_time(actor_tr, t_start) or (
        actor_home_x, float(spawns[performer][1]), float(spawns[performer][2]))
    _, lat0 = world_to_ego_offset(ego0, actor0[0], actor0[1])
    if not _is_adjacent_lat(lat0, lw):
        details["checks"]["2_adjacent_at_start"] = False
        if _same_lane_lat(lat0, lw):
            return False, "already on ego lane at start of cut-in", details
        return False, "not in adjacent lane at start of cut-in", details
    details["checks"]["2_adjacent_at_start"] = True

    if rec.get("success") is not True:
        return False, "recorded success is not true", details

    t_commit = rec.get("t_commit")
    if t_commit is None:
        return False, "no commit time", details
    t_commit = float(t_commit)
    details["t_commit"] = t_commit

    def actor_enters_ego_lane(t, x, y, hd):
        return _in_lane_x(x, ego_home_x, lw)

    def ego_leaves_home(t, x, y, hd):
        return not _in_lane_x(x, ego_home_x, lw)

    def ego_enters_actor_lane(t, x, y, hd):
        return _in_lane_x(x, actor_home_x, lw)

    t_actor_in = _first_event(actor_tr, t_commit, actor_enters_ego_lane)
    t_ego_out = _first_event(ego_tr, t_commit, ego_leaves_home)
    t_ego_to_actor = _first_event(ego_tr, t_commit, ego_enters_actor_lane)

    if t_actor_in is None:
        details["checks"]["3_lane_change_ahead"] = False
        return False, "actor never entered ego lane before commit", details
    if t_ego_out and t_ego_out[0] + TIME_EPS < t_actor_in[0]:
        details["checks"]["3_lane_change_ahead"] = False
        return False, "ego left target lane before actor merged", details
    if t_ego_to_actor and t_ego_to_actor[0] + TIME_EPS < t_actor_in[0]:
        details["checks"]["3_lane_change_ahead"] = False
        return False, "ego moved to actor lane before actor merged", details

    ego_t = pose_at_time(ego_tr, t_commit)
    actor_t = pose_at_time(actor_tr, t_commit)
    if not ego_t or not actor_t:
        return False, "could not sample poses at commit", details
    if not _in_lane_x(ego_t[0], ego_home_x, lw):
        details["checks"]["3_lane_change_ahead"] = False
        return False, "ego not in original lane at commit", details
    if not _in_lane_x(actor_t[0], ego_home_x, lw):
        details["checks"]["3_lane_change_ahead"] = False
        return False, "actor not in ego lane at commit", details

    along, _ = world_to_ego_offset(ego_t, actor_t[0], actor_t[1])
    details["along_commit"] = round(along, 3)
    if along <= MIN_AHEAD_M:
        details["checks"]["3_lane_change_ahead"] = False
        return False, "not ahead of ego at commit", details
    details["checks"]["3_lane_change_ahead"] = True

    if along > MAX_AHEAD_M:
        details["checks"]["1_within_10m_ahead"] = False
        return False, f"more than {MAX_AHEAD_M:.0f} m ahead of ego at commit", details
    details["checks"]["1_within_10m_ahead"] = True

    speeds = _speed_samples(actor_tr, t_commit, t_commit + POST_MERGE_WINDOW_S)
    if len(speeds) < MIN_POST_SAMPLES:
        details["checks"]["4_nominal_speed"] = False
        return False, "too few post-merge speed samples", details
    v_mean = sum(v for _, v in speeds) / len(speeds)
    ego_speeds = _speed_samples(ego_tr, t_commit, t_commit + POST_MERGE_WINDOW_S)
    ego_v_mean = (sum(v for _, v in ego_speeds) / len(ego_speeds)) if ego_speeds else None
    details["post_merge_v_mean"] = round(v_mean, 3)

    def _nom(v, ref):
        if ref <= 1e-3:
            return abs(v) <= NOMINAL_SPEED_ABS
        return abs(v - ref) <= max(NOMINAL_SPEED_ABS, NOMINAL_SPEED_FRAC * ref)

    near_ego = ego_v_mean is not None and _nom(v_mean, ego_v_mean)
    near_cruise = _nom(v_mean, cruise)
    if near_ego or (near_cruise and v_mean >= NOMINAL_MIN_MPS):
        details["checks"]["4_nominal_speed"] = True
    else:
        details["checks"]["4_nominal_speed"] = False
        return False, f"post-merge speed {v_mean:.1f} m/s not nominal", details

    return True, "proper cut-in (4/4 checks)", details


def _first_event(traj: List[list], t_max: float,
                 pred: Callable) -> Optional[Tuple[float, float]]:
    for row in traj:
        t = float(row[0])
        if t > t_max + 1e-6:
            break
        if pred(t, float(row[1]), float(row[2]), float(row[3])):
            return t, float(row[1])
    return None


def verify_overtake_setup(data: dict,
                          lane_width: Optional[float] = None
                          ) -> Tuple[bool, str, Dict[str, Any]]:
    """Overtake stage checks (1–3): valid test layout before ego drives."""
    lw = lane_width if lane_width is not None else infer_lane_width(data)
    ego_tr = data.get("ego_trajectory") or []
    spawns = data.get("spawns") or {}
    if not ego_tr or not spawns:
        return False, "missing ego trajectory or spawns", {}

    blocker = _role_id(data, "blocker")
    oncoming = _role_id(data, "oncoming")
    if not blocker or not oncoming:
        return False, "missing blocker or oncoming role", {}

    ego0 = pose_at_time(ego_tr, float(ego_tr[0][0]))
    if blocker not in spawns or oncoming not in spawns:
        return False, "spawn missing for role actors", {}

    blk_p = tuple(spawns[blocker][:3]) if len(spawns[blocker]) >= 3 else (
        float(spawns[blocker][0]), float(spawns[blocker][1]), 90.0)
    onc_p = tuple(spawns[oncoming][:3]) if len(spawns[oncoming]) >= 3 else (
        float(spawns[oncoming][0]), float(spawns[oncoming][1]), 270.0)
    ego_home_x = float(ego0[1]) if ego0 else float(ego_tr[0][1])

    details: Dict[str, Any] = {
        "blocker": blocker, "oncoming": oncoming, "checks": {},
    }

    # (1) blocker same lane, ahead of ego
    along_b, lat_b = world_to_ego_offset(ego0, blk_p[0], blk_p[1])
    details["blocker_along_start"] = round(along_b, 3)
    if not (_same_lane_lat(lat_b, lw) and along_b > MIN_AHEAD_M):
        details["checks"]["1_blocker_ahead_same_lane"] = False
        return False, "blocker not same-lane ahead of ego at start", details
    details["checks"]["1_blocker_ahead_same_lane"] = True

    # (2) oncoming on adjacent lane, reaction gap, nominal speed
    along_o, lat_o = world_to_ego_offset(ego0, onc_p[0], onc_p[1])
    details["oncoming_lat_start"] = round(lat_o, 3)
    if not _is_adjacent_lat(lat_o, lw):
        details["checks"]["2_oncoming_adjacent_reaction"] = False
        return False, "oncoming not on ego-adjacent lane at start", details

    # longitudinal separation along road (y-axis for north/south setup)
    y_gap = abs(blk_p[1] - onc_p[1])
    details["oncoming_blocker_y_gap"] = round(y_gap, 3)
    if y_gap < OVERTAKE_ONCOMING_REACTION_MIN_M:
        details["checks"]["2_oncoming_adjacent_reaction"] = False
        return False, (f"oncoming–blocker gap {y_gap:.1f} m < "
                       f"{OVERTAKE_ONCOMING_REACTION_MIN_M:.0f} m"), details

    onc_tr = (data.get("actor_trajectories") or {}).get(oncoming) or []
    speeds = _speed_samples(onc_tr, float(onc_tr[0][0]) if onc_tr else 0.0,
                            float(onc_tr[-1][0]) if onc_tr else 1.0)
    if speeds:
        v_mean = sum(v for _, v in speeds) / len(speeds)
        details["oncoming_v_mean"] = round(v_mean, 3)
        if not (OVERTAKE_ONCOMING_SPEED_MIN <= v_mean <= OVERTAKE_ONCOMING_SPEED_MAX):
            details["checks"]["2_oncoming_adjacent_reaction"] = False
            return False, f"oncoming speed {v_mean:.1f} m/s out of range", details
    details["checks"]["2_oncoming_adjacent_reaction"] = True

    # (3) no other actor blocking adjacent lane ahead of ego
    for aid, sp in spawns.items():
        if aid in (blocker, oncoming, "0"):
            continue
        px, py = float(sp[0]), float(sp[1])
        _, lat = world_to_ego_offset(ego0, px, py)
        along, _ = world_to_ego_offset(ego0, px, py)
        if _is_adjacent_lat(lat, lw) and 0 < along < OVERTAKE_ADJ_CLEAR_AHEAD_M:
            details["checks"]["3_adjacent_lane_clear"] = False
            details["blocker_actor"] = aid
            return False, f"actor {aid} blocks adjacent lane ahead of ego", details
    details["checks"]["3_adjacent_lane_clear"] = True

    return True, "overtake setup (3/3 checks)", details


def verify_overtake_outcome(data: dict,
                            lane_width: Optional[float] = None
                            ) -> Tuple[bool, str, Dict[str, Any]]:
    """Overtake outcome (4): ego passes blocker without colliding."""
    lw = lane_width if lane_width is not None else infer_lane_width(data)
    ego_tr = data.get("ego_trajectory") or []
    if not ego_tr:
        return False, "no ego trajectory", {}

    blocker = _role_id(data, "blocker")
    oncoming = _role_id(data, "oncoming")
    if not blocker:
        return False, "no blocker role", {}

    blk_tr = (data.get("actor_trajectories") or {}).get(blocker) or []
    details: Dict[str, Any] = {"checks": {}}

    t0 = float(ego_tr[0][0])
    t1 = float(ego_tr[-1][0])
    ego_s = pose_at_time(ego_tr, t0)
    ego_e = pose_at_time(ego_tr, t1)
    blk_s = pose_at_time(blk_tr, t0) if blk_tr else None
    blk_e = pose_at_time(blk_tr, t1) if blk_tr else None

    if not ego_s or not blk_s:
        return False, "could not sample start poses", details

    ego_home_x = float(ego_s[1])
    # (4a) start behind blocker
    gap_start = _longitudinal_gap_y(ego_s, blk_s)
    details["gap_start_y"] = round(gap_start, 3)
    if gap_start <= MIN_AHEAD_M:
        details["checks"]["4_start_behind_blocker"] = False
        return False, "ego not behind blocker at start", details
    details["checks"]["4_start_behind_blocker"] = True

    # (4b) lane change into adjacent (used opposite lane to pass)
    used_adjacent = False
    for row in ego_tr:
        _, lat = world_to_ego_offset(ego_s, float(row[1]), float(row[2]))
        if _is_adjacent_lat(lat, lw):
            used_adjacent = True
            break
    details["checks"]["4_lane_change"] = used_adjacent
    if not used_adjacent:
        return False, "ego never used adjacent lane to overtake", details

    # (4c) end ahead of blocker
    if blk_e and ego_e:
        gap_end = _longitudinal_gap_y(ego_e, blk_e)
        details["gap_end_y"] = round(gap_end, 3)
        if gap_end >= -MIN_AHEAD_M:
            details["checks"]["4_end_ahead_of_blocker"] = False
            return False, "ego did not finish ahead of blocker", details
    details["checks"]["4_end_ahead_of_blocker"] = True

    # (4d) no collision with blocker or oncoming
    hit = _any_collision(data)
    details["collision"] = hit
    if hit:
        details["checks"]["4_no_collision"] = False
        return False, f"collision detected ({hit})", details
    details["checks"]["4_no_collision"] = True

    return True, "overtake success (4/4 outcome checks)", details


def verify_overtake(data: dict,
                    lane_width: Optional[float] = None
                    ) -> Tuple[bool, str, Dict[str, Any]]:
    """Full overtake verification: setup (1–3) + outcome (4)."""
    ok_s, r_s, d_s = verify_overtake_setup(data, lane_width)
    ok_o, r_o, d_o = verify_overtake_outcome(data, lane_width)
    checks = {**d_s.get("checks", {}), **d_o.get("checks", {})}
    details = {**d_s, **d_o, "checks": checks,
               "setup_ok": ok_s, "outcome_ok": ok_o}
    if ok_s and ok_o:
        return True, "overtake verified (setup + outcome)", details
    if not ok_s:
        return False, f"setup: {r_s}", details
    return False, f"outcome: {r_o}", details


def verify_hard_brake_setup(data: dict,
                            lane_width: Optional[float] = None
                            ) -> Tuple[bool, str, Dict[str, Any]]:
    """Hard-brake stage checks (1–2) per docs/SCENARIOS_AND_VALIDATION.md.

    1. Braking actor same-lane ahead; brakes only once gap ≤ trigger, with
       normal deceleration.
    2. Adjacent actor within a reasonable longitudinal range of the lead.
    """
    lw = lane_width if lane_width is not None else infer_lane_width(data)
    ego_tr = data.get("ego_trajectory") or []
    spawns = data.get("spawns") or {}
    slow_id = _role_id(data, "slow")
    adj_id = _role_id(data, "adjacent")
    if not slow_id or not ego_tr:
        return False, "missing slow role or ego trajectory", {}
    if not adj_id:
        return False, "missing adjacent role", {}

    slow_tr = (data.get("actor_trajectories") or {}).get(slow_id) or []
    adj_tr = (data.get("actor_trajectories") or {}).get(adj_id) or []
    if slow_id not in spawns or adj_id not in spawns:
        return False, "spawn missing for slow/adjacent actor", {}

    t0 = float(ego_tr[0][0])
    ego0 = pose_at_time(ego_tr, t0)
    slow0 = pose_at_time(slow_tr, t0) if slow_tr else (
        float(spawns[slow_id][0]), float(spawns[slow_id][1]), 90.0)
    adj0 = pose_at_time(adj_tr, t0) if adj_tr else (
        float(spawns[adj_id][0]), float(spawns[adj_id][1]), 90.0)
    details: Dict[str, Any] = {
        "slow": slow_id, "adjacent": adj_id, "checks": {},
    }

    # (1) same-lane ahead + brake only when within HARD_BRAKE_TRIGGER_GAP_M
    along, lat = world_to_ego_offset(ego0, slow0[0], slow0[1])
    details["slow_along_start"] = round(along, 3)
    if not (_same_lane_lat(lat, lw) and along > MIN_AHEAD_M):
        details["checks"]["1_brake_when_close"] = False
        return False, "braking actor not same-lane ahead of ego at start", details

    gap_closed = False
    brake_ok = False
    prev_v: Optional[float] = None
    prev_t: Optional[float] = None
    for er, sr in _paired_samples(ego_tr, slow_tr):
        t = float(er[0])
        ex, ey, eh = float(er[1]), float(er[2]), float(er[3])
        sx, sy = float(sr[1]), float(sr[2])
        along_g, _ = world_to_ego_offset((ex, ey, eh), sx, sy)
        speeds = _speed_samples(slow_tr, max(t0, t - 0.15), t + 0.15)
        v_now = speeds[-1][1] if speeds else (
            float(sr[4]) if len(sr) >= 5 else None)

        trigger = HARD_BRAKE_TRIGGER_GAP_M + HARD_BRAKE_TRIGGER_TOL_M
        if along_g > trigger:
            # Must not already be crawling / hard-braking before the trigger.
            if v_now is not None and v_now < HARD_BRAKE_PRETRIGGER_MIN_MPS:
                details["checks"]["1_brake_when_close"] = False
                details["early_crawl_v"] = round(v_now, 3)
                details["early_crawl_gap"] = round(along_g, 3)
                return False, (
                    f"braking actor already slow ({v_now:.1f} m/s) before "
                    f"gap ≤ {HARD_BRAKE_TRIGGER_GAP_M:.0f} m"), details
            if prev_v is not None and v_now is not None and prev_t is not None:
                dt = max(t - prev_t, 1e-3)
                decel = (v_now - prev_v) / dt
                if decel <= HARD_BRAKE_MIN_DECEL:
                    details["checks"]["1_brake_when_close"] = False
                    details["early_decel"] = round(decel, 2)
                    details["early_decel_gap"] = round(along_g, 3)
                    return False, (
                        f"braking started at gap {along_g:.1f} m "
                        f"(>{HARD_BRAKE_TRIGGER_GAP_M:.0f} m)"), details
        else:
            gap_closed = True
            if prev_v is not None and v_now is not None and prev_t is not None:
                dt = max(t - prev_t, 1e-3)
                decel = (v_now - prev_v) / dt
                if decel <= HARD_BRAKE_MIN_DECEL:
                    brake_ok = True
                    details["slow_decel_at_gap"] = round(decel, 2)
                    details["brake_gap"] = round(along_g, 3)
                    break
        if v_now is not None:
            prev_v, prev_t = v_now, t

    if not gap_closed:
        details["checks"]["1_brake_when_close"] = False
        return False, (f"ego never closed within "
                       f"{HARD_BRAKE_TRIGGER_GAP_M:.0f} m of braking actor"), details
    if not brake_ok:
        details["checks"]["1_brake_when_close"] = False
        return False, (f"no normal deceleration (≤ {HARD_BRAKE_MIN_DECEL} m/s²) "
                       f"once within {HARD_BRAKE_TRIGGER_GAP_M:.0f} m"), details
    details["checks"]["1_brake_when_close"] = True

    # (2) adjacent actor near the leading (slow) vehicle
    along_a, lat_a = world_to_ego_offset(ego0, adj0[0], adj0[1])
    details["adjacent_along_start"] = round(along_a, 3)
    details["adjacent_lat_start"] = round(lat_a, 3)
    if not _is_adjacent_lat(lat_a, lw):
        details["checks"]["2_adjacent_near_lead"] = False
        return False, "adjacent actor not on ego-adjacent lane at start", details

    lead_gap = abs(_longitudinal_gap_y(slow0, adj0))
    details["adjacent_to_lead_gap"] = round(lead_gap, 3)
    if not (HARD_BRAKE_ADJ_TO_LEAD_MIN_M
            <= lead_gap
            <= HARD_BRAKE_ADJ_TO_LEAD_MAX_M):
        details["checks"]["2_adjacent_near_lead"] = False
        return False, (
            f"adjacent–lead gap {lead_gap:.1f} m outside "
            f"[{HARD_BRAKE_ADJ_TO_LEAD_MIN_M:.0f}, "
            f"{HARD_BRAKE_ADJ_TO_LEAD_MAX_M:.0f}] m"), details
    details["checks"]["2_adjacent_near_lead"] = True

    return True, "hard_brake setup (2/2 checks)", details


def verify_hard_brake_outcome(data: dict,
                              lane_width: Optional[float] = None
                              ) -> Tuple[bool, str, Dict[str, Any]]:
    """Hard-brake outcome: ego avoids both leads (no collision)."""
    _ = lane_width
    details: Dict[str, Any] = {"checks": {}}
    hit = _any_collision(data)
    details["collision"] = hit
    if hit:
        details["checks"]["3_no_collision"] = False
        return False, f"collision detected ({hit})", details
    details["checks"]["3_no_collision"] = True
    return True, "hard_brake outcome: no collision", details


def _paired_samples(ego_tr: List[list], actor_tr: List[list]):
    """Yield (ego_row, actor_row) aligned by nearest actor sample."""
    j = 0
    for er in ego_tr:
        t = float(er[0])
        while j + 1 < len(actor_tr) and float(actor_tr[j + 1][0]) <= t:
            j += 1
        if j < len(actor_tr):
            yield er, actor_tr[j]


def verify_hard_brake(data: dict,
                      lane_width: Optional[float] = None
                      ) -> Tuple[bool, str, Dict[str, Any]]:
    ok_s, r_s, d_s = verify_hard_brake_setup(data, lane_width)
    ok_o, r_o, d_o = verify_hard_brake_outcome(data, lane_width)
    checks = {**d_s.get("checks", {}), **d_o.get("checks", {})}
    details = {**d_s, **d_o, "checks": checks,
               "setup_ok": ok_s, "outcome_ok": ok_o}
    if ok_s and ok_o:
        return True, "hard_brake verified (setup + outcome)", details
    if not ok_s:
        return False, f"setup: {r_s}", details
    return False, f"outcome: {r_o}", details


def verify_run(data: dict) -> Tuple[str, bool, str, Dict[str, Any]]:
    """Auto-detect scenario and run the appropriate verifier."""
    kind = detect_scenario(data)
    lw = infer_lane_width(data)
    if kind == "cutin":
        ok, reason, det = verify_proper_cutin(data, lw)
    elif kind == "overtake":
        ok, reason, det = verify_overtake(data, lw)
    elif kind == "hard_brake":
        ok, reason, det = verify_hard_brake(data, lw)
    else:
        return kind, False, f"unknown scenario type ({kind})", {}
    return kind, ok, reason, det
