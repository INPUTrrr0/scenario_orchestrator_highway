#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/orchestrator.py — headless closed-loop scenario orchestration kernel.

One shared representation: the v0/v2 **maneuver script** is ground truth (P1).
Prediction and evaluation happen **along the actual script** (directives_script),
so the D1/D2/D3 verdict matches the realization the animation plays — not a
constant-velocity projection of the current instant. Every orchestration tick
(dt_tick = 0.1 s, P4):

    1. sample the world's forward realization from the live maneuver rollout,
    2. evaluate the red-light family + compute the minimal causal repair
       (directives_script: analytic speed solves verified by re-simulation),
    3. apply any user perturbation (a maneuver edit, may target the ego),
    4. apply the orchestrator's intervention *if it changes a standing command*
       (retime/reroute realized as maneuver edits),
    5. advance the clock one dt_tick.

Only **decision points** are persisted as snapshot files + tree nodes (P3);
NO-OP ticks collapse, their elapsed time annotated on the connecting edge. All
files for one rollout live under `sessions/<session_id>/`, each snapshot a
re-based maneuver-script scenario reusing `Scenario.to_dict()` and the v2
provenance machinery.

Headless / import-safe (no pygame). CLI runs a scripted session from scenario v20.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
for p in (V2, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import scenario_editor as se          # noqa: E402
import directives as dv               # noqa: E402  (Params + geometry, via ds)
import maneuvers as mv                # noqa: E402  (rebase/retime/reroute/perturb)
import directives_script as ds        # noqa: E402  (script-grounded family + repair)

from maneuvers import Perturbation    # noqa: E402  (re-exported for callers)

DT_TICK = 0.10
DEFAULT_SIGNALS = {"N": "green", "S": "green", "E": "red", "W": "red"}
DECISION_KINDS = ("start", "intervention", "perturbation", "checkpoint",
                  "proposal", "infeasible", "collision")


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

    def load(self) -> bool:
        """Load an existing provenance.yaml into this store (for resuming)."""
        if not os.path.exists(self.prov_path):
            return False
        doc = yaml.safe_load(open(self.prov_path)) or {}
        self.versions = list(doc.get("versions", []))
        self.meta = doc.get("meta", self.meta)
        return bool(self.versions)

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
            "delta": delta, "verdict": verdict})
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
# Kernel
# --------------------------------------------------------------------------- #
def _verdict_dict(v: ds.Verdict) -> dict:
    return {"d1": bool(v.d1.value), "d2": bool(v.d2.value), "d3": bool(v.d3.value),
            "hero": v.hero, "t_star": round(v.t_star, 3) if v.t_star else None,
            "ok": bool(v.ok)}


def _plan_key(plan: List[ds.Intervention], ego: str) -> Dict[str, tuple]:
    out = {}
    for iv in plan:
        if iv.actor == ego:
            continue
        out[iv.actor] = (("retime", round(float(iv.value), 1))
                         if iv.kind == "retime" else ("reroute", iv.value))
    return out


class Orchestrator:
    """Drives the closed loop over a maneuver-script world (script-grounded
    directives). Used headless (run/step) and by the interactive front end."""

    def __init__(self, base_scenario: str, session_dir: str,
                 ego: str = "0", signals: Optional[dict] = None,
                 prm=None, label: str = "", resume: bool = False):
        self.sc = se.load_scenario(base_scenario)
        self.ego = ego
        self.signals = dict(signals or DEFAULT_SIGNALS)
        self.prm = prm or dv.Params()
        self.T = 0.0
        self.t_base = 0.0
        self.tick = 0
        self.standing: Dict[str, tuple] = {}
        self.infeasible = False
        self.done = False
        self._pending: Optional[Perturbation] = None
        meta = {"base_scenario": os.path.basename(base_scenario), "ego": ego,
                "signals": self.signals, "dt_tick": DT_TICK,
                "label": label or os.path.basename(base_scenario)}
        self.store = SessionStore(session_dir, meta)
        if resume and self.store.load():
            last = self.store.versions[-1]
            self.parent = last["version"]
            self.sc = se.load_scenario(os.path.join(self.store.dir, last["file"]))
            self.T = float(last["sim_time"])
            self.tick = int(last["tick"])
            self.t_base = self.T
        else:
            v = self._evaluate()
            self.parent = self.store.save_snapshot(self.sc, "start", None, 0, 0.0,
                                                   "session start", _verdict_dict(v))

    # ---- script-grounded evaluation ---- #
    def _world_now(self) -> se.Scenario:
        return mv.rebase_scenario(self.sc, self.T - self.t_base)

    def _evaluate(self) -> ds.Verdict:
        return ds.evaluate(self._world_now(), self.ego, self.signals, self.prm)

    def current(self) -> ds.Verdict:
        return self._evaluate()

    def _rebase_here(self) -> None:
        self.sc = mv.rebase_scenario(self.sc, self.T - self.t_base)
        self.t_base = self.T

    # ---- perturbations ---- #
    def queue_perturbation(self, p: Perturbation) -> None:
        self._pending = p

    def _apply_pending(self) -> Optional[int]:
        p = self._pending
        if p is None:
            return None
        self._pending = None
        self._rebase_here()
        self.sc = mv.apply_perturbation(self.sc, p)
        v = self._evaluate()
        self.store.log(self.tick, self.T, "perturbation",
                       {"perturbation": p.__dict__})
        self.parent = self.store.save_snapshot(
            self.sc, "perturbation", self.parent, self.tick, self.T,
            p.summary(), _verdict_dict(v))
        return self.parent

    # ---- orchestrator ---- #
    def _apply_plan(self, plan: List[ds.Intervention]) -> None:
        self._rebase_here()
        by = {a.id: a for a in self.sc.actors}
        lw, arm = self.sc.map.lane_width, self.sc.map.arm_length
        for iv in plan:
            if iv.actor == self.ego or iv.actor not in by:
                continue
            if iv.kind == "retime":
                mv.retime_actor(by[iv.actor], float(iv.value))
            else:
                mv.reroute_actor(by[iv.actor], str(iv.value), lw, arm)
        self.sc.simulate()

    def orchestrate(self) -> Optional[int]:
        w = self._world_now()
        res = ds.evaluate(w, self.ego, self.signals, self.prm)
        rr = ds.repair(w, self.ego, self.signals, self.prm)

        if rr.feasible and rr.interventions:
            key = _plan_key(rr.interventions, self.ego)
            if key and key != {a: self.standing.get(a) for a in key}:
                self._apply_plan(rr.interventions)
                self.standing.update(key)
                self.infeasible = False
                v2 = self._evaluate()
                delta = "; ".join(str(iv) for iv in rr.interventions)
                self.store.log(self.tick, self.T, "intervention",
                               {"interventions": [iv.__dict__ for iv in
                                                  rr.interventions],
                                "cost": rr.cost})
                self.parent = self.store.save_snapshot(
                    self.sc, "intervention", self.parent, self.tick, self.T,
                    delta, _verdict_dict(v2))
                return self.parent
            return None
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

    # ---- proposals / checkpoints (front end) ---- #
    def compute_repair(self) -> ds.Rep:
        return ds.repair(self._world_now(), self.ego, self.signals, self.prm)

    def build_trial(self, rr: ds.Rep) -> se.Scenario:
        trial = self._world_now()
        by = {a.id: a for a in trial.actors}
        lw, arm = trial.map.lane_width, trial.map.arm_length
        for iv in rr.interventions:
            if iv.actor == self.ego or iv.actor not in by:
                continue
            if iv.kind == "retime":
                mv.retime_actor(by[iv.actor], float(iv.value))
            else:
                mv.reroute_actor(by[iv.actor], str(iv.value), lw, arm)
        trial.simulate()
        return trial

    def save_proposal(self, rr: ds.Rep, trial: se.Scenario) -> int:
        v = ds.evaluate(trial, self.ego, self.signals, self.prm)
        delta = "PROPOSAL: " + "; ".join(str(iv) for iv in rr.interventions)
        return self.store.save_snapshot(trial, "proposal", self.parent,
                                        self.tick, self.T, delta, _verdict_dict(v))

    def accept_proposal(self, version: int, trial: se.Scenario, rr: ds.Rep) -> None:
        self.sc = trial
        self.t_base = self.T
        self.standing.update(_plan_key(rr.interventions, self.ego))
        self.parent = version

    def propose(self) -> Optional[int]:
        rr = self.compute_repair()
        if not (rr.feasible and rr.interventions):
            return None
        return self.save_proposal(rr, self.build_trial(rr))

    def checkpoint(self, note: str = "", scenario: Optional[se.Scenario] = None) -> int:
        if scenario is not None:
            self.sc = scenario
            self.sc.simulate()
            self.t_base = self.T        # the edited realization's t=0 is *now*
        self._rebase_here()
        v = self._evaluate()
        self.parent = self.store.save_snapshot(
            self.sc, "checkpoint", self.parent, self.tick, self.T,
            note or f"checkpoint @ {self.T:.2f}s", _verdict_dict(v))
        return self.parent

    def commit_perturbation(self, scenario: se.Scenario, summary: str,
                            payload: Optional[dict] = None) -> int:
        """Adopt an edited realization as a perturbation node (child of the
        current node). The edited scenario is the state at the current node's
        time, so this is authoring the future from here — a perturbation."""
        self.sc = scenario
        self.sc.simulate()
        self.t_base = self.T            # the edited realization's t=0 is *now*
        self.standing = {}
        self.infeasible = False
        self.done = False
        v = self._evaluate()
        self.store.log(self.tick, self.T, "perturbation",
                       payload or {"note": summary})
        self.parent = self.store.save_snapshot(
            self.sc, "perturbation", self.parent, self.tick, self.T,
            summary, _verdict_dict(v))
        return self.parent

    def advance_to_decision(self, max_ticks: int = 400) -> Optional[int]:
        """Run the closed loop forward until the next decision-point node is
        created (orchestrator intervention, or a collision/infeasible terminal),
        or the budget runs out. Returns the new node version, else None."""
        if self.done:
            return None
        n0 = len(self.store.versions)
        for _ in range(max_ticks):
            self.step()
            if len(self.store.versions) > n0:
                return self.store.versions[-1]
            if self.done:
                return None
        return None

    def load_checkpoint(self, version: int) -> None:
        rec = next(v for v in self.store.versions if v["version"] == version)
        self.sc = se.load_scenario(os.path.join(self.store.dir, rec["file"]))
        # rewind the clock: returning to a checkpoint explores an alternate
        # timeline *from that point*, not from wherever the last branch ended.
        self.T = float(rec["sim_time"])
        self.tick = int(rec["tick"])
        self.t_base = self.T
        self.standing = {}
        self.infeasible = False
        self.done = False
        self.parent = version

    def _check_realized(self) -> Optional[int]:
        """Terminate on goal-achieved: the ego actually collides with a
        red-runner. A success terminal, distinct from `infeasible`."""
        if self.done:
            return None
        hit = ds.collision_now(self._world_now(), self.ego)
        if hit is None:
            return None
        self.done = True
        v = self._evaluate()
        self.store.log(self.tick, self.T, "collision", {"actor": hit})
        self.parent = self.store.save_snapshot(
            self.sc, "collision", self.parent, self.tick, self.T,
            f"collision realized: ego x {hit} @ {self.T:.2f}s", _verdict_dict(v))
        return self.parent

    # ---- stepping ---- #
    def step(self) -> None:
        if self.done:
            return
        self._apply_pending()
        if self._check_realized():
            return
        self.orchestrate()
        self.T += DT_TICK
        self.tick += 1

    def run(self, max_time: float, script: Optional[Dict[int, Perturbation]] = None):
        script = script or {}
        for _ in range(int(round(max_time / DT_TICK))):
            if self.done:
                break
            if self.tick in script:
                self.queue_perturbation(script[self.tick])
            self.step()


# --------------------------------------------------------------------------- #
# CLI: run a scripted session
# --------------------------------------------------------------------------- #
def _demo_script() -> Dict[int, Perturbation]:
    return {
        15: Perturbation("add_actor", leg="WS", turn="left", speed=8.0),
        40: Perturbation("set_speed", actor="9", speed=3.0),
    }


def main():
    ap = argparse.ArgumentParser(description="closed-loop orchestration (headless)")
    ap.add_argument("--base", default=os.path.join(V2, "scenarios",
                                                   "scenario_v20.yaml"))
    ap.add_argument("--session", default=None)
    ap.add_argument("--time", type=float, default=6.5)
    ap.add_argument("--nominal", action="store_true")
    ap.add_argument("--signals", default=None)
    args = ap.parse_args()

    sid = args.session or (datetime.now().strftime("%Y%m%dT%H%M%S")
                           + "_v20redlight")
    session_dir = os.path.join(HERE, "sessions", sid)
    signals = dv.parse_signals(args.signals) if args.signals else DEFAULT_SIGNALS

    orch = Orchestrator(args.base, session_dir, ego="0", signals=signals,
                        label="v20 red-light (closed loop)")
    orch.run(args.time, {} if args.nominal else _demo_script())

    print(f"session: {session_dir}")
    print(f"{len(orch.store.versions)} decision-point nodes:")
    for v in orch.store.versions:
        vd = v["verdict"]
        badge = "".join("+" if vd[d] else "-" for d in ("d1", "d2", "d3"))
        print(f"  v{v['version']:<2} {v['kind']:<12} t={v['sim_time']:<5} "
              f"[{badge}] parent={v['parent']}  {v['delta']}")


if __name__ == "__main__":
    main()
