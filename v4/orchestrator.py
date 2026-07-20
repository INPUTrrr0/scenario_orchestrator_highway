#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/orchestrator.py — headless closed-loop scenario orchestration kernel.

One shared representation: the v0/v2 **maneuver script** is ground truth (P1). Every
orchestration tick (dt_tick = 0.1 s, P4):

    1. sample an instantaneous State from the live maneuver rollout,
    2. recognize -> evaluate_family -> repair   (all from v2/directives.py),
    3. apply any user perturbation (a maneuver-script edit, may target the ego),
    4. apply the orchestrator's minimal causal intervention *if it changes a
       standing command* (retime/reroute translated back into maneuver edits),
    5. advance the clock one dt_tick.

Only **decision points** are persisted as snapshot files + tree nodes (P3): a
perturbation, an intervention that changes a standing command, a checkpoint, a
proposal, or the onset of infeasibility. NO-OP ticks are collapsed; their elapsed
time is annotated on the connecting edge. All files for one rollout live under
`sessions/<session_id>/`, each snapshot a re-based maneuver-script scenario reusing
`Scenario.to_dict()` and the v2 provenance machinery.

Headless / import-safe (no pygame). CLI runs a scripted session from scenario v20.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
if V2 not in sys.path:
    sys.path.insert(0, V2)

import scenario_editor as se          # noqa: E402  (maneuver world / ground truth)
import directives as dv               # noqa: E402  (v2 directive layer)

DT_TICK = 0.10                        # orchestration tick (P4)
BIG_DUR = 60.0                        # a "hold/long" maneuver duration cap
DEFAULT_SIGNALS = {"N": "green", "S": "green", "E": "red", "W": "red"}
DECISION_KINDS = ("start", "intervention", "perturbation",
                  "checkpoint", "proposal", "infeasible")

TURN_TYPES = se.TURN_TYPES
LONG_TYPES = se.LONGITUDINAL_TYPES


# --------------------------------------------------------------------------- #
# Maneuver-script surgery: re-basing, retime, reroute, perturbations
# --------------------------------------------------------------------------- #
def _truncate_maneuver(m: se.Maneuver, t_i: float) -> se.Maneuver:
    """The remaining tail of maneuver `m` after local time t_i, as a fresh
    maneuver starting from the pose reached at t_i (trajectory-preserving)."""
    t_i = max(0.0, min(t_i, m.duration))
    rem = m.duration - t_i
    if m.type in TURN_TYPES:
        frac_done = t_i / max(1e-6, m.duration)
        return se.Maneuver(type=m.type, radius=m.radius,
                           angle=m.angle * (1.0 - frac_done), duration=rem)
    if m.type == "stop":
        return se.Maneuver(type="stop", duration=rem)
    # longitudinal: v(t)=v0+a t -> tail has v0'=v0+a*t_i, same slope
    return se.Maneuver(type=m.type, intercept=m.velocity_at(t_i),
                       slope=m.slope, duration=rem)


def _actor_pose_speed_at(a: se.Actor, tau: float) -> Tuple[se.Pose, float]:
    k = int(round(tau / se.DT))
    k = max(0, min(k, len(a.traj) - 1)) if a.traj else 0
    if a.traj:
        return a.traj[k], a.speeds[k]
    return a.start, 0.0


def rebase_actor(a: se.Actor, tau: float) -> se.Actor:
    """Freeze the elapsed prefix: return a copy of `a` started at its pose at
    local time `tau`, carrying only its remaining maneuver suffix."""
    a.compute_schedule()
    pose, _ = _actor_pose_speed_at(a, tau)
    segs = a.maneuvers
    if tau >= a.total - 1e-9 or not segs:
        new_segs: List = []
    else:
        i = a.active_index(tau)
        t_i = tau - a.cum[i]
        head = segs[i]
        tail = ([_truncate_maneuver(head, t_i)] if isinstance(head, se.Maneuver)
                else [copy.deepcopy(head)])
        new_segs = tail + [copy.deepcopy(s) for s in segs[i + 1:]]
    return se.Actor(id=a.id, color=a.color, length=a.length, width=a.width,
                    start=pose, maneuvers=new_segs)


def rebase_scenario(sc: se.Scenario, tau: float) -> se.Scenario:
    """Re-base the whole world to local time `tau` (its new t=0)."""
    sc.simulate()
    actors = [rebase_actor(a, tau) for a in sc.actors]
    new = se.Scenario(map=se.MapConfig(sc.map.lane_width, sc.map.arm_length),
                      actors=actors, pixels_per_meter=sc.pixels_per_meter)
    new.simulate()
    return new


def _geom_distance(m: se.Maneuver) -> float:
    if m.type in TURN_TYPES:
        return m.arc_length()
    if m.type == "stop":
        return 0.0
    return m.distance(m.duration)


def retime_actor(a: se.Actor, v_target: float) -> None:
    """Adopt target speed `v_target` along the actor's existing route, in place.
    Geometry (turn radius/angle, straight distances) preserved; only speed
    changes. v_target == 0 -> yield/stop."""
    if v_target <= 1e-3:
        a.maneuvers = [se.Maneuver(type="stop", duration=BIG_DUR)]
        return
    out: List[se.Maneuver] = []
    for m in a.maneuvers:
        if not isinstance(m, se.Maneuver):
            out.append(m)                       # function segment: passthrough
            continue
        if m.type in TURN_TYPES:
            out.append(se.Maneuver(type=m.type, radius=m.radius, angle=m.angle,
                                   duration=m.arc_length() / v_target))
        else:
            dist = _geom_distance(m)
            if dist <= 1e-6:
                continue                        # drop degenerate/stop tails
            out.append(se.Maneuver(type="go_straight", intercept=v_target,
                                   slope=0.0, duration=dist / v_target))
    if not out:
        out = [se.Maneuver(type="go_straight", intercept=v_target, slope=0.0,
                           duration=BIG_DUR)]
    a.maneuvers = out


def _snap_heading(hdg: float) -> float:
    return min((0.0, 90.0, 180.0, 270.0, 360.0),
               key=lambda h: abs(((hdg - h + 180) % 360) - 180)) % 360


def _dist_to_intersection(x: float, y: float, hdg: float, lw: float) -> float:
    """Distance along an axis-aligned inbound heading to the intersection box
    entry (|x| or |y| == lane_width)."""
    h = _snap_heading(hdg)
    if h == 90:    # +y
        return max(0.0, (-lw) - y)
    if h == 270:   # -y
        return max(0.0, y - lw)
    if h == 0:     # +x
        return max(0.0, (-lw) - x)
    return max(0.0, x - lw)                      # h == 180, -x


def build_route_maneuvers(x: float, y: float, hdg: float, speed: float,
                          turn: str, lw: float, arm: float) -> List[se.Maneuver]:
    """A constant-speed maneuver suffix from (x,y,hdg): approach -> (turn) ->
    exit. `turn` in {straight,left,right}."""
    speed = max(1.0, speed)
    r = 6.0
    d_app = _dist_to_intersection(x, y, hdg, lw)
    segs: List[se.Maneuver] = []
    if turn == "straight":
        span = d_app + 2 * lw + arm + 6.0
        return [se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                            duration=span / speed)]
    if d_app > 1e-3:
        segs.append(se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                                duration=d_app / speed))
    ttype = "turn_left" if turn == "left" else "turn_right"
    tm = se.Maneuver(type=ttype, radius=r, angle=90.0, duration=1.0)
    tm.duration = tm.arc_length() / speed
    segs.append(tm)
    segs.append(se.Maneuver(type="go_straight", intercept=speed, slope=0.0,
                            duration=(arm + 6.0) / speed))
    return segs


def reroute_actor(a: se.Actor, turn: str, lw: float, arm: float) -> None:
    """Best-effort route change: rebuild the suffix to follow `turn` at the
    actor's current speed from its current pose."""
    _, v = _actor_pose_speed_at(a, 0.0)
    if v <= 1e-3 and a.maneuvers and isinstance(a.maneuvers[0], se.Maneuver):
        v = a.maneuvers[0].velocity_at(0.0)
    x, y, hdg = a.start
    a.maneuvers = build_route_maneuvers(x, y, hdg, max(v, 6.0), turn, lw, arm)


# --- leg geometry for spawning ------------------------------------------- #
LEG_SPAWN = {  # inbound leg -> (x, y, heading) at the arm end (right-hand lane)
    "SE": lambda lw, arm: (lw / 2, -arm + 2.0, 90.0),
    "EN": lambda lw, arm: (arm - 2.0, lw / 2, 180.0),
    "WS": lambda lw, arm: (-arm + 2.0, -lw / 2, 0.0),
    "NW": lambda lw, arm: (-lw / 2, arm - 2.0, 270.0),
}
PALETTE = [(90, 190, 110), (210, 70, 60), (60, 120, 210), (200, 160, 60),
           (160, 90, 200), (80, 200, 200), (150, 185, 95), (225, 105, 165)]


# --------------------------------------------------------------------------- #
# Perturbations (user maneuver-script edits; may target the ego)
# --------------------------------------------------------------------------- #
@dataclass
class Perturbation:
    kind: str                       # add_actor | set_maneuver | set_speed | remove_actor
    actor: Optional[str] = None
    leg: Optional[str] = None
    turn: Optional[str] = None      # straight | left | right
    speed: Optional[float] = None

    def summary(self) -> str:
        if self.kind == "add_actor":
            return f"add actor on {self.leg} ({self.turn}, {self.speed:g} m/s)"
        if self.kind == "set_maneuver":
            return f"set actor {self.actor} route -> {self.turn}"
        if self.kind == "set_speed":
            return f"set actor {self.actor} speed -> {self.speed:g} m/s"
        if self.kind == "remove_actor":
            return f"remove actor {self.actor}"
        return self.kind


def apply_perturbation(sc: se.Scenario, p: Perturbation) -> se.Scenario:
    """Apply a perturbation to an already-re-based scenario (returns a new sc)."""
    lw, arm = sc.map.lane_width, sc.map.arm_length
    actors = [copy.deepcopy(a) for a in sc.actors]
    by_id = {a.id: a for a in actors}
    if p.kind == "add_actor":
        nid = str(max([int(a.id) for a in actors if a.id.isdigit()] + [-1]) + 1)
        x, y, hdg = LEG_SPAWN[p.leg](lw, arm)
        color = PALETTE[len(actors) % len(PALETTE)]
        man = build_route_maneuvers(x, y, hdg, p.speed or 10.0,
                                    p.turn or "straight", lw, arm)
        actors.append(se.Actor(id=nid, color=color, length=4.5, width=2.0,
                               start=(x, y, hdg), maneuvers=man))
    elif p.kind == "remove_actor":
        actors = [a for a in actors if a.id != p.actor]
    elif p.kind == "set_maneuver" and p.actor in by_id:
        reroute_actor(by_id[p.actor], p.turn or "left", lw, arm)
    elif p.kind == "set_speed" and p.actor in by_id:
        retime_actor(by_id[p.actor], p.speed or 0.0)
    new = se.Scenario(map=se.MapConfig(lw, arm), actors=actors,
                      pixels_per_meter=sc.pixels_per_meter)
    new.simulate()
    return new


# --------------------------------------------------------------------------- #
# Session persistence (reused v2 machinery, retargeted to one rollout)
# --------------------------------------------------------------------------- #
class SessionStore:
    def __init__(self, session_dir: str, meta: dict):
        self.dir = session_dir
        os.makedirs(self.dir, exist_ok=True)
        self.prov_path = os.path.join(self.dir, "provenance.yaml")
        self.log_path = os.path.join(self.dir, "session_log.yaml")
        self.versions: List[dict] = []
        self.meta = meta

    def _next(self) -> int:
        return (max(v["version"] for v in self.versions) + 1) if self.versions else 1

    def save_snapshot(self, sc: se.Scenario, kind: str, parent: Optional[int],
                      tick: int, sim_time: float, delta: str,
                      verdict: dict) -> int:
        n = self._next()
        fname = f"snapshot_v{n}.yaml"
        with open(os.path.join(self.dir, fname), "w") as f:
            yaml.safe_dump(sc.to_dict(), f, sort_keys=False)
        self.versions.append({
            "version": n, "file": fname, "parent": parent,
            "created": datetime.now().isoformat(timespec="seconds"),
            "kind": kind, "tick": tick, "sim_time": round(sim_time, 3),
            "delta": delta, "verdict": verdict,
        })
        self._write_provenance()
        return n

    def _write_provenance(self) -> None:
        with open(self.prov_path, "w") as f:
            yaml.safe_dump({"meta": self.meta, "versions": self.versions}, f,
                           sort_keys=False)

    def log(self, tick: int, sim_time: float, kind: str, payload: dict) -> None:
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "tick": tick, "sim_time": round(sim_time, 3),
                 "kind": kind, **payload}
        with open(self.log_path, "a") as f:
            f.write(yaml.safe_dump([entry], sort_keys=False))


# --------------------------------------------------------------------------- #
# The closed-loop kernel
# --------------------------------------------------------------------------- #
def _verdict_dict(res: dv.FamilyResult) -> dict:
    return {"d1": bool(res.d1.value), "d2": bool(res.d2.value),
            "d3": bool(res.d3.value), "hero": res.hero,
            "t_star": round(res.t_star, 3) if res.t_star else None,
            "ok": bool(res.ok)}


def _plan_key(plan: List[dv.Intervention]) -> Dict[str, tuple]:
    out = {}
    for iv in plan:
        if iv.kind == "retime":
            out[iv.actor] = ("retime", round(float(iv.value), 1))
        else:
            out[iv.actor] = ("reroute", iv.value)
    return out


class Orchestrator:
    """Drives the closed loop over a maneuver-script world. Used both headless
    (run/step) and by the interactive front end (perturb/propose/checkpoint)."""

    def __init__(self, base_scenario: str, session_dir: str,
                 ego: str = "0", signals: Optional[dict] = None,
                 prm: Optional[dv.Params] = None, label: str = ""):
        self.sc = se.load_scenario(base_scenario)
        self.ego = ego
        self.signals = dict(signals or DEFAULT_SIGNALS)
        self.prm = prm or dv.Params()
        self.T = 0.0                      # session clock
        self.t_base = 0.0                 # session clock at last re-base
        self.tick = 0
        self.standing: Dict[str, tuple] = {}      # last-applied orchestrator commands
        self.infeasible = False
        meta = {"base_scenario": os.path.basename(base_scenario),
                "ego": ego, "signals": self.signals, "dt_tick": DT_TICK,
                "label": label or os.path.basename(base_scenario)}
        self.store = SessionStore(session_dir, meta)
        res = self._evaluate()
        self.parent = self.store.save_snapshot(
            self.sc, "start", None, 0, 0.0, "session start",
            _verdict_dict(res))

    # ---- directive-layer bridge (sample the maneuver world) ---- #
    def _sample_state(self) -> dv.State:
        tau = self.T - self.t_base
        self.sc.simulate()
        k = int(round(tau / se.DT))
        acts = []
        for a in self.sc.actors:
            k2 = max(0, min(k, len(a.traj) - 1))
            x, y, hdg = a.traj[k2]
            acts.append(dv.ActorState(str(a.id), x, y, hdg, a.speeds[k2],
                                      a.length, a.width))
        return dv.State(dv.MapCfg(self.sc.map.lane_width, self.sc.map.arm_length),
                        acts, self.ego, dict(self.signals),
                        f"t={self.T:.2f}s")

    def _evaluate(self) -> dv.FamilyResult:
        astate = dv.recognize(self._sample_state(), self.prm)
        return dv.evaluate_family(astate, self.prm)

    def _rebase_here(self) -> None:
        """Re-base the live world to the current tick (its new t=0)."""
        self.sc = rebase_scenario(self.sc, self.T - self.t_base)
        self.t_base = self.T

    # ---- perturbations ---- #
    def queue_perturbation(self, p: Perturbation) -> None:
        self._pending = p

    def _apply_pending(self) -> Optional[int]:
        p = getattr(self, "_pending", None)
        if p is None:
            return None
        self._pending = None
        self._rebase_here()
        self.sc = apply_perturbation(self.sc, p)
        res = self._evaluate()
        self.store.log(self.tick, self.T, "perturbation",
                       {"perturbation": p.__dict__})
        self.parent = self.store.save_snapshot(
            self.sc, "perturbation", self.parent, self.tick, self.T,
            p.summary(), _verdict_dict(res))
        return self.parent

    # ---- orchestrator ---- #
    def _apply_plan(self, plan: List[dv.Intervention]) -> None:
        self._rebase_here()
        by_id = {a.id: a for a in self.sc.actors}
        lw, arm = self.sc.map.lane_width, self.sc.map.arm_length
        for iv in plan:
            if iv.actor == self.ego or iv.actor not in by_id:
                continue                                  # never steer the ego
            if iv.kind == "retime":
                retime_actor(by_id[iv.actor], float(iv.value))
            else:
                reroute_actor(by_id[iv.actor], str(iv.value), lw, arm)
        self.sc.simulate()

    def orchestrate(self) -> Optional[int]:
        """Run repair once; persist a node only if it changes a standing
        command (else NO-OP) or on new infeasibility. Returns node version or
        None for a NO-OP."""
        astate = dv.recognize(self._sample_state(), self.prm)
        res = dv.evaluate_family(astate, self.prm)
        rr = dv.repair(astate, self.prm)

        if rr.feasible and rr.interventions:
            key = _plan_key(rr.interventions)
            key = {a: k for a, k in key.items() if a != self.ego}
            if key and key != {a: self.standing.get(a) for a in key}:
                self._apply_plan(rr.interventions)
                self.standing.update(key)
                self.infeasible = False
                res2 = self._evaluate()
                delta = "; ".join(str(iv) for iv in rr.interventions)
                self.store.log(self.tick, self.T, "intervention",
                               {"interventions": [iv.__dict__ for iv in
                                                  rr.interventions],
                                "cost": rr.cost})
                self.parent = self.store.save_snapshot(
                    self.sc, "intervention", self.parent, self.tick, self.T,
                    delta, _verdict_dict(res2))
                return self.parent
            return None                          # re-affirming standing -> NO-OP
        if not rr.feasible and not res.ok and not self.infeasible:
            self.infeasible = True
            self.store.log(self.tick, self.T, "infeasible", {"reason": rr.reason})
            self.parent = self.store.save_snapshot(
                self.sc, "infeasible", self.parent, self.tick, self.T,
                rr.reason, _verdict_dict(res))
            return self.parent
        if res.ok:
            self.infeasible = False
        return None

    # ---- checkpoint / proposal (used by the front end) ---- #
    def checkpoint(self, note: str = "") -> int:
        self._rebase_here()
        res = self._evaluate()
        self.parent = self.store.save_snapshot(
            self.sc, "checkpoint", self.parent, self.tick, self.T,
            note or f"checkpoint @ {self.T:.2f}s", _verdict_dict(res))
        return self.parent

    # ---- proposal building blocks (also used by the front end) ---- #
    def compute_repair(self) -> dv.RepairResult:
        astate = dv.recognize(self._sample_state(), self.prm)
        return dv.repair(astate, self.prm)

    def build_trial(self, rr: dv.RepairResult) -> se.Scenario:
        """A re-based copy of the live world with `rr`'s interventions applied,
        WITHOUT touching the live world (for previews/proposals)."""
        trial = rebase_scenario(self.sc, self.T - self.t_base)
        by_id = {a.id: a for a in trial.actors}
        lw, arm = trial.map.lane_width, trial.map.arm_length
        for iv in rr.interventions:
            if iv.actor == self.ego or iv.actor not in by_id:
                continue
            if iv.kind == "retime":
                retime_actor(by_id[iv.actor], float(iv.value))
            else:
                reroute_actor(by_id[iv.actor], str(iv.value), lw, arm)
        trial.simulate()
        return trial

    def save_proposal(self, rr: dv.RepairResult, trial: se.Scenario) -> int:
        res = dv.evaluate_family(dv.recognize(
            _state_of(trial, 0.0, self.ego, self.signals), self.prm), self.prm)
        delta = "PROPOSAL: " + "; ".join(str(iv) for iv in rr.interventions)
        return self.store.save_snapshot(trial, "proposal", self.parent,
                                        self.tick, self.T, delta,
                                        _verdict_dict(res))

    def accept_proposal(self, version: int, trial: se.Scenario,
                        rr: dv.RepairResult) -> None:
        """Adopt a proposal as the committed continuation."""
        self.sc = trial
        self.t_base = self.T
        for iv in rr.interventions:
            if iv.actor == self.ego:
                continue
            self.standing[iv.actor] = (("retime", round(float(iv.value), 1))
                                       if iv.kind == "retime"
                                       else ("reroute", iv.value))
        self.parent = version

    def propose(self) -> Optional[int]:
        """Materialize the orchestrator's repair as an uncommitted proposal node
        (a branch); does NOT change the live world."""
        rr = self.compute_repair()
        if not (rr.feasible and rr.interventions):
            return None
        return self.save_proposal(rr, self.build_trial(rr))

    def current(self) -> dv.FamilyResult:
        return self._evaluate()

    def load_checkpoint(self, version: int) -> None:
        """Resume from a saved snapshot (a new branch continues from it)."""
        rec = next(v for v in self.store.versions if v["version"] == version)
        self.sc = se.load_scenario(os.path.join(self.store.dir, rec["file"]))
        # rewind the clock: returning to a checkpoint explores an alternate
        # timeline *from that point*, not from wherever the last branch ended.
        self.T = float(rec["sim_time"])
        self.tick = int(rec["tick"])
        self.t_base = self.T
        self.standing = {}
        self.infeasible = False
        self.parent = version

    # ---- stepping ---- #
    def step(self) -> None:
        """One orchestration tick: perturb -> orchestrate -> advance."""
        self._apply_pending()
        self.orchestrate()
        self.T += DT_TICK
        self.tick += 1

    def run(self, max_time: float, script: Optional[Dict[int, Perturbation]] = None):
        script = script or {}
        n = int(round(max_time / DT_TICK))
        for _ in range(n):
            if self.tick in script:
                self.queue_perturbation(script[self.tick])
            self.step()


def _state_of(sc: se.Scenario, tau: float, ego: str, signals: dict) -> dv.State:
    sc.simulate()
    k = int(round(tau / se.DT))
    acts = []
    for a in sc.actors:
        k2 = max(0, min(k, len(a.traj) - 1))
        x, y, hdg = a.traj[k2]
        acts.append(dv.ActorState(str(a.id), x, y, hdg, a.speeds[k2],
                                  a.length, a.width))
    return dv.State(dv.MapCfg(sc.map.lane_width, sc.map.arm_length), acts, ego,
                    dict(signals), "")


# --------------------------------------------------------------------------- #
# CLI: run a scripted session
# --------------------------------------------------------------------------- #
def _demo_script() -> Dict[int, Perturbation]:
    # at t=1.5s the user drops an inconvenient left-turner across the hero's path
    return {
        15: Perturbation("add_actor", leg="WS", turn="left", speed=8.0),
        40: Perturbation("set_speed", actor="9", speed=3.0),
    }


def main():
    ap = argparse.ArgumentParser(description="closed-loop orchestration (headless)")
    ap.add_argument("--base", default=os.path.join(V2, "scenarios",
                                                   "scenario_v20.yaml"))
    ap.add_argument("--session", default=None, help="session id (folder name)")
    ap.add_argument("--time", type=float, default=6.5, help="rollout seconds")
    ap.add_argument("--nominal", action="store_true",
                    help="no perturbations (orchestrator maintenance only)")
    ap.add_argument("--signals", default=None,
                    help="e.g. N=green,S=green,E=red,W=red")
    args = ap.parse_args()

    sid = args.session or (datetime.now().strftime("%Y%m%dT%H%M%S")
                           + "_v20redlight")
    session_dir = os.path.join(HERE, "sessions", sid)
    signals = dv.parse_signals(args.signals) if args.signals else DEFAULT_SIGNALS

    orch = Orchestrator(args.base, session_dir, ego="0", signals=signals,
                        label="v20 red-light (closed loop)")
    script = {} if args.nominal else _demo_script()
    orch.run(args.time, script)

    nodes = orch.store.versions
    print(f"session: {session_dir}")
    print(f"{len(nodes)} decision-point nodes:")
    for v in nodes:
        vd = v["verdict"]
        badge = "".join("+" if vd[d] else "-" for d in ("d1", "d2", "d3"))
        print(f"  v{v['version']:<2} {v['kind']:<12} t={v['sim_time']:<5} "
              f"[{badge}] parent={v['parent']}  {v['delta']}")


if __name__ == "__main__":
    main()
