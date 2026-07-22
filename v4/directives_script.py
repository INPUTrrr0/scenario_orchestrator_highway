#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/directives_script.py — the red-light family (D1/D2/D3) and minimal causal
repair, evaluated **along the actual maneuver script** rather than v2's
constant-velocity projection.

Why this exists: v2's directive layer projects the *instantaneous* state forward
at constant velocity, so at t=0 of scenario v20 — ego crawling at 6 m/s far from
the box — it predicts no collision and reads D1/D2 false, even though the scripted
trajectories (what the animation plays, and what actually happens) do realize the
red-light collision. Here every predicate is grounded on the simulated
trajectories, so the verdict matches the realization.

Fidelity of repair. Retiming a candidate to a target speed `v'` makes *that actor*
constant-speed by construction, so its arrival at the conflict point P is exactly
`d/v'`. The repair solves `v' = d / t_mid` where `t_mid` is the midpoint of the
ego's occupancy window of P — read off the ego's scripted trajectory by sampling
(exact to the sim step) rather than from a constant-velocity closed form. The
linear solve is exact; the only approximation is v2's pre-existing *greedy*
per-directive ordering (fix D1, then D2, then D3, with a joint re-check), which is
verified here by re-simulating each trial and re-evaluating.

Result objects (Ev / Verdict / Intervention / Rep) mirror the fields the kernel
already reads from v2, so orchestrator.py swaps layers with no downstream change.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
for p in (V2, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import scenario_editor as se          # noqa: E402
import directives as dv               # noqa: E402  (geometry + recognizer reused)
import maneuvers as mv                # noqa: E402  (rebase/retime/reroute)

DT = se.DT
SWEEP_STRIDE = 1        # frame stride for body sweeps (raise for speed, e.g. 3 = 0.05s)


# --------------------------------------------------------------------------- #
# Result types (field-compatible with the v2 layer the kernel used before)
# --------------------------------------------------------------------------- #
@dataclass
class Ev:
    value: bool
    expl: str = ""


@dataclass
class Verdict:
    d1: Ev
    d2: Ev
    d3: Ev
    hero: Optional[str]
    t_star: Optional[float]
    ok: bool
    P: Optional[Tuple[float, float]] = None
    ego_window: Optional[Tuple[float, float]] = None
    d3_witness: Optional[str] = None


@dataclass
class Intervention:
    kind: str            # 'retime' | 'reroute'
    actor: str
    value: object
    cost: float
    why: str

    def __str__(self):
        v = f"{self.value:.2f} m/s" if self.kind == "retime" else str(self.value)
        return f"{self.kind}({self.actor} -> {v})  cost {self.cost:.2f}   [{self.why}]"


@dataclass
class Rep:
    feasible: bool
    interventions: List[Intervention]
    cost: float
    final: Verdict
    reason: str = ""


# --------------------------------------------------------------------------- #
# Trajectory oracle (everything grounded on the simulated script)
# --------------------------------------------------------------------------- #
def state_at(sc: se.Scenario, k: int, ego: str, signals: dict) -> dv.State:
    acts = []
    for a in sc.actors:
        kk = min(k, len(a.traj) - 1)
        x, y, h = a.traj[kk]
        acts.append(dv.ActorState(a.id, x, y, h, a.speeds[kk], a.length, a.width))
    return dv.State(dv.MapCfg(sc.map.lane_width, sc.map.arm_length), acts, ego,
                    dict(signals), "")


def _r(a: se.Actor, prm) -> float:
    return 0.5 * math.hypot(a.length, a.width) + prm.r_margin


def _body(a: se.Actor, k: int):
    kk = min(k, len(a.traj) - 1)
    x, y, h = a.traj[kk]
    return dv.rect_corners(x, y, h, a.length, a.width)


def collide_time(a: se.Actor, b: se.Actor, H: float) -> Optional[float]:
    n = min(int(H / DT) + 1, max(len(a.traj), len(b.traj)))
    for k in range(0, n, SWEEP_STRIDE):
        if dv.rects_overlap(_body(a, k), _body(b, k)):
            return k * DT
    return None


def _poly(a: se.Actor) -> Optional[dv.Path]:
    pts = []
    for p in a.traj:
        q = (p[0], p[1])
        if not pts or abs(q[0] - pts[-1][0]) + abs(q[1] - pts[-1][1]) > 1e-6:
            pts.append(q)
    return dv.Path(pts) if len(pts) >= 2 else None


def conflict_P(a: se.Actor, ego: se.Actor) -> Optional[Tuple[float, float]]:
    pa, pe = _poly(a), _poly(ego)
    if pa is None or pe is None:
        return None
    cr = dv.first_crossing(pa, pe)
    return pe.pos_at(cr[1]) if cr else None


def occ_window(a: se.Actor, P, r: float, H: float) -> Optional[Tuple[float, float]]:
    t0 = t1 = None
    for k in range(0, min(int(H / DT) + 1, len(a.traj)), SWEEP_STRIDE):
        x, y, _ = a.traj[k]
        if (x - P[0]) ** 2 + (y - P[1]) ** 2 <= r * r:
            if t0 is None:
                t0 = k * DT
            t1 = k * DT
    return (t0, t1) if t0 is not None else None


def arc_to_P(a: se.Actor, P, kstart: int = 0) -> Tuple[float, float]:
    """Remaining arclength from frame kstart to the nearest-P point of a's path,
    and the time a reaches it."""
    bk, best = kstart, 1e18
    for k in range(kstart, len(a.traj)):
        d = (a.traj[k][0] - P[0]) ** 2 + (a.traj[k][1] - P[1]) ** 2
        if d < best:
            best, bk = d, k
    s = 0.0
    for k in range(kstart, bk):
        s += math.hypot(a.traj[k + 1][0] - a.traj[k][0],
                        a.traj[k + 1][1] - a.traj[k][1])
    return s, bk * DT


def _overlap(w1, w2) -> bool:
    return max(w1[0], w2[0]) <= min(w1[1], w2[1]) + 1e-9


def collision_now(sc: se.Scenario, ego_id: str, within: float = 0.25
                  ) -> Optional[str]:
    """The other actor whose body overlaps the ego's within the next `within`
    seconds (frame 0 == now). Used to detect the *realized* collision so the
    rollout can terminate on goal-achieved rather than a stale directive verdict."""
    by = {a.id: a for a in sc.actors}
    ego = by[ego_id]
    for a in sc.actors:
        if a.id == ego_id:
            continue
        for k in range(int(within / DT) + 1):
            if dv.rects_overlap(_body(ego, k), _body(a, k)):
                return a.id
    return None


def runs_red(astate: dv.AbsState, aid: str) -> bool:
    a = astate.actors[aid]
    if a.leg is None:
        return False
    arm = dv.arm_of(a.leg)
    return (arm in "NESW" and a.region in ("approach", "intersection")
            and astate.phase(arm) == "red")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(sc: se.Scenario, ego_id: str, signals: dict, prm,
             fast: bool = False) -> Verdict:
    """`sc` must be re-based so that frame 0 == 'now'."""
    sc.simulate()
    by = {a.id: a for a in sc.actors}
    ego = by[ego_id]
    astate = dv.recognize(state_at(sc, 0, ego_id, signals), prm)
    H = prm.H
    turns = ("__keep__",) if fast else ("__keep__", "straight", "left", "right")

    # D1: some red-runner collides with the ego on the script
    cands = [a.id for a in sc.actors
             if a.id != ego_id and runs_red(astate, a.id)]
    wits = []
    for aid in cands:
        tc = collide_time(by[aid], ego, H)
        if tc is not None:
            wits.append((tc, aid))
    if wits:
        wits.sort()
        t_star, hero = wits[0]
        d1 = Ev(True, f"{hero} collides ego @ {t_star:.2f}s")
    else:
        opts = d1_options(sc, ego_id, prm, turns=turns)
        hero = opts[0][1] if opts else (cands[0] if cands else None)
        t_star, d1 = H, Ev(False, "no red-runner collides the ego on the script")

    P = conflict_P(by[hero], ego) if hero else None
    ego_win = occ_window(ego, P, _r(ego, prm), H) if P else None

    # D2: the hero can still reach P within the ego's window, throughout [0,t*]
    if hero is None or P is None or ego_win is None:
        d2 = Ev(False, "no reachable conflict for the hero")
    else:
        d2 = _eval_d2(by[hero], P, ego_win, min(t_star, H), prm)

    # D3: no third vehicle interferes (collides ego/hero, or occupies P in-window)
    d3, w_wit = _eval_d3(sc, ego_id, hero, P, ego_win, H, prm)

    ok = d1.value and d2.value and d3.value
    return Verdict(d1, d2, d3, hero, t_star if d1.value else None, ok, P,
                   ego_win, w_wit)


def _eval_d2(hero: se.Actor, P, ego_win, t_hi: float, prm) -> Ev:
    e0, e1 = ego_win
    tau = 0.0
    while tau <= min(t_hi, e1) + 1e-9:
        k = int(round(tau / DT))
        d_h, _ = arc_to_P(hero, P, min(k, len(hero.traj) - 1))
        if e1 <= tau:
            break
        if tau + d_h / prm.v_max > e1 + prm.pad:
            return Ev(False, f"reachability margin lost at t={tau:.2f}s")
        tau += prm.dt
    return Ev(True, "hero can reach the conflict throughout")


def _eval_d3(sc, ego_id, hero, P, ego_win, H, prm) -> Tuple[Ev, Optional[str]]:
    by = {a.id: a for a in sc.actors}
    ego = by[ego_id]
    for a in sc.actors:
        if a.id in (ego_id, hero):
            continue
        if collide_time(a, ego, H) is not None:
            return Ev(False, f"{a.id} collides the ego"), a.id
        if hero and collide_time(a, by[hero], H) is not None:
            return Ev(False, f"{a.id} collides the hero"), a.id
        if P and ego_win:
            ow = occ_window(a, P, _r(a, prm), H)
            if ow and _overlap(ow, ego_win):
                return Ev(False, f"{a.id} occupies the conflict in-window"), a.id
    return Ev(True, "no interference"), None


# --------------------------------------------------------------------------- #
# Repair (analytic per-directive solve on the script + verify by re-simulation)
# --------------------------------------------------------------------------- #
def _apply(trial: se.Scenario, aid: str, kind: str, value) -> None:
    by = {a.id: a for a in trial.actors}
    if aid not in by:
        return
    if kind == "retime":
        mv.retime_actor(by[aid], float(value))
    else:
        mv.reroute_actor(by[aid], str(value), trial.map.lane_width,
                         trial.map.arm_length)
    trial.simulate()


def _copy(sc: se.Scenario) -> se.Scenario:
    return mv.rebase_scenario(sc, 0.0)


def d1_options(sc: se.Scenario, ego_id: str, prm,
               turns=("__keep__", "straight", "left", "right")) -> List[tuple]:
    """Causal retimes that make some red-runner collide the ego, cheapest first:
    (cost, aid, turn, v_target, why). Solved v'=d/t_mid, verified by sweep."""
    by = {a.id: a for a in sc.actors}
    ego = by[ego_id]
    astate = dv.recognize(state_at(sc, 0, ego_id, {}), prm)
    out = []
    for a in sc.actors:
        if a.id == ego_id or not runs_red(astate, a.id):
            continue
        for turn in turns:
            trial = _copy(sc)
            if turn != "__keep__":
                _apply(trial, a.id, "reroute", turn)
            cand = {x.id: x for x in trial.actors}[a.id]
            P = conflict_P(cand, {x.id: x for x in trial.actors}[ego_id])
            if P is None:
                continue
            ego_win = occ_window({x.id: x for x in trial.actors}[ego_id], P,
                                 _r(ego, prm), prm.H)
            if ego_win is None or ego_win[1] <= 0:
                continue
            d, _ = arc_to_P(cand, P, 0)
            t_mid = (max(ego_win[0], 0.0) + ego_win[1]) / 2.0
            if t_mid <= 1e-6 or d <= 1e-6:
                continue
            v_t = d / t_mid
            if not (0.0 < v_t <= prm.v_max):
                continue
            t2 = _copy(trial)
            _apply(t2, a.id, "retime", v_t)
            cand2 = {x.id: x for x in t2.actors}[a.id]
            ego2 = {x.id: x for x in t2.actors}[ego_id]
            if collide_time(cand2, ego2, prm.H) is None:
                continue
            v_now = a.speeds[0]
            cost = prm.w_v * abs(v_t - v_now) + (0.0 if turn == "__keep__"
                                                 else prm.w_r)
            why = (f"retime {a.id} -> {v_t:.2f} m/s"
                   + ("" if turn == "__keep__" else f" + reroute {turn}"))
            out.append((cost, a.id, turn, v_t, why))
            break                       # cheapest turn for this actor
    out.sort(key=lambda o: o[0])
    return out


def _clear_interferer(sc, ego_id, hero, w, prm, fast=False) -> Optional[Intervention]:
    """Min-|Δv| retime (incl. yield) or reroute that clears interferer w."""
    by = {a.id: a for a in sc.actors}
    v_now = by[w].speeds[0]
    trials = [("retime", v) for v in (0.0, prm.v_max, max(0.0, v_now - 4))]
    if not fast:
        trials += [("retime", 2.0), ("retime", v_now + 5)]
        trials += [("reroute", t) for t in ("left", "right", "straight")]
    best = None
    for kind, val in trials:
        t = _copy(sc)
        _apply(t, w, kind, val)
        v = evaluate(t, ego_id, {}, prm)
        # cleared iff this actor no longer interferes with the (same) hero
        tb = {a.id: a for a in t.actors}
        ego = tb[ego_id]
        clash = (collide_time(tb[w], ego, prm.H) is not None
                 or (hero and hero in tb
                     and collide_time(tb[w], tb[hero], prm.H) is not None))
        if not clash:
            cost = (prm.w_v * abs(val - v_now) if kind == "retime" else prm.w_r)
            if best is None or cost < best[0]:
                best = (cost, kind, val)
    if best is None:
        return None
    cost, kind, val = best
    return Intervention(kind, w, val, cost,
                        f"clear interferer {w}")


def repair(sc: se.Scenario, ego_id: str, signals: dict, prm,
           fast: bool = False) -> Rep:
    trial = _copy(sc)
    plan: List[Intervention] = []
    turns = ("__keep__",) if fast else ("__keep__", "straight", "left", "right")
    for _ in range(4):
        v = evaluate(trial, ego_id, signals, prm, fast=fast)
        if v.ok:
            break
        if not v.d1.value:
            opts = d1_options(trial, ego_id, prm, turns=turns)
            if not opts:
                return Rep(False, plan, sum(i.cost for i in plan), v,
                           "D1 unrepairable: no red-runner can causally reach the "
                           "ego's window (the past is not available for editing)")
            cost, aid, turn, v_t, why = opts[0]
            if turn != "__keep__":
                _apply(trial, aid, "reroute", turn)
                plan.append(Intervention("reroute", aid, turn, prm.w_r, why))
            _apply(trial, aid, "retime", v_t)
            plan.append(Intervention("retime", aid, v_t,
                                     prm.w_v * abs(v_t - sc_speed(sc, aid)), why))
        elif not v.d2.value:
            # restore reachability: retime the hero toward the window midpoint
            hero = v.hero
            opts = [o for o in d1_options(trial, ego_id, prm, turns=turns)
                    if o[1] == hero]
            if not opts:
                return Rep(False, plan, sum(i.cost for i in plan), v,
                           f"D2 unrepairable for hero {hero}: committed past P")
            cost, aid, turn, v_t, why = opts[0]
            _apply(trial, aid, "retime", v_t)
            plan.append(Intervention("retime", aid, v_t,
                                     prm.w_v * abs(v_t - sc_speed(sc, aid)),
                                     "restore D2 reachability"))
        else:
            w = v.d3_witness
            if w is None:
                break
            fix = _clear_interferer(trial, ego_id, v.hero, w, prm, fast=fast)
            if fix is None:
                return Rep(False, plan, sum(i.cost for i in plan), v,
                           f"D3 unrepairable: interferer {w} cannot be cleared")
            _apply(trial, fix.actor, fix.kind, fix.value)
            plan.append(fix)
    v = evaluate(trial, ego_id, signals, prm)
    return Rep(v.ok, plan, sum(i.cost for i in plan), v,
               "" if v.ok else "unresolved after max repair passes")


def sc_speed(sc: se.Scenario, aid: str) -> float:
    for a in sc.actors:
        if a.id == aid:
            return a.speeds[0] if a.speeds else 0.0
    return 0.0
