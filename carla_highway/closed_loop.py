#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/closed_loop.py — the cut-in orchestrator, against a live CARLA ego.

The orchestration logic lives in `cutin_director.py`, which the pygame editor
drives too, so a cut-in behaves the same whether you steer the ego yourself or
a policy drives it in CARLA. This module owns only the cadence and the clock.

The clock
---------
`CutinDirector.tick` re-bases the background scenario to the current instant,
so the actors' plans restart at script time zero. The runner samples actor
states at `atime`, which this module advances by the CARLA timestep and resets
on every orchestration tick.

Casting is per mode
-------------------
Only `scenario_cutin` is orchestrated. `scenario_hard_brake` and
`scenario_overtake` are ego-policy stress tests with fully scripted actors, and
`ModeSpec.casting` carries that decision; `--casting` / `--no-casting`
overrides it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .highway_ego import Ego
from .highway_map import HighwayFrame
from .scenarios import ModeSpec
from .script_bridge import ROLE_CUTIN, ROLE_NOMINAL, se

import cutin_director as cd

#: orchestration cadence, matching v4/orchestrator.DT_TICK
DT_TICK = 0.10


@dataclass
class Event:
    """Something the orchestrator did, for the report and the HUD."""
    t: float
    kind: str
    text: str
    actor: Optional[str] = None


class HighwayClosedLoop:
    """Owns the background scenario, its clock, and the orchestration cadence."""

    def __init__(self, frame: HighwayFrame, background: "se.Scenario",
                 mode: ModeSpec, casting: Optional[bool] = None,
                 director: Optional[cd.CutinDirector] = None,
                 ego_dims: Tuple[float, float] = (4.5, 2.0)):
        self.frame = frame
        self.sc = background
        self.mode = mode
        want = mode.casting if casting is None else bool(casting)
        self.director = director if want else None
        self.casting = self.director is not None
        self.ego_dims = ego_dims
        self.atime = 0.0                  # script time since sc's frame 0
        self._next_tick = 0.0
        #: the instant the verifier keys its merge checks off: when the holder
        #: is centred in the ego's lane (lane change complete)
        self.t_commit: Optional[float] = None
        if want and director is None:
            self._no_spec = True
        else:
            self._no_spec = False

    # ------------------------------------------------------------------ #
    def advance(self, dt: float) -> None:
        """Move the script clock forward one CARLA step."""
        self.atime += dt

    def due(self, t_sim: float) -> bool:
        return t_sim >= self._next_tick

    def ego_state(self, ego: Ego) -> cd.EgoState:
        return cd.EgoState(ego.x, ego.y, math.degrees(ego.theta) % 360.0,
                           ego.v, self.ego_dims[0], self.ego_dims[1])

    def tick(self, ego: Ego, t_sim: float) -> bool:
        """One orchestration step. Returns True if the director acted."""
        if not self.due(t_sim):
            return False
        self._next_tick = t_sim + DT_TICK
        if self.director is None:
            return False
        n_before = len(self.director.events)
        self.director.tick(self.sc, self.atime, t_sim, self.ego_state(ego))
        self.atime = 0.0
        return len(self.director.events) != n_before

    def observe(self, ego: Ego, t_sim: float,
                states: Dict[str, Tuple[float, float, float, float]]) -> None:
        """Every simulation step, with the vehicles' simulated states."""
        if self.director is None:
            return
        self.director.observe(t_sim, self.ego_state(ego), states)
        if self.t_commit is None and self.director.merged_at is not None:
            self.t_commit = self.director.merged_at

    # ---- views ---- #
    @property
    def events(self) -> List[Event]:
        if self.director is None:
            return ([Event(0.0, "cast", "no `cutin:` spec in this scenario; "
                           "casting disabled")] if self._no_spec else [])
        return [Event(e["t"], e["kind"], e["text"], e.get("actor"))
                for e in self.director.events]

    @property
    def roles(self) -> Dict[str, str]:
        if self.director is None:
            return {}
        return {a.id: (ROLE_CUTIN if a.id == self.director.holder else ROLE_NOMINAL)
                for a in self.sc.actors}

    @property
    def scores(self) -> Dict[str, float]:
        return dict(self.director.scores) if self.director else {}

    @property
    def holder(self) -> Optional[str]:
        return self.director.holder if self.director else None

    @property
    def outcome(self) -> Optional[str]:
        """not_triggered | triggered_not_started | started_not_crossed |
        cut_in (None: not casting)."""
        return self.director.outcome if self.director else None

    @property
    def merged(self) -> bool:
        return bool(self.director and self.director.merged_at is not None)

    @property
    def committed(self) -> bool:
        return bool(self.director and self.director.t_trigger is not None)

    @property
    def n_recasts(self) -> int:
        return self.director.n_recasts if self.director else 0

    @property
    def n_interventions(self) -> int:
        if self.director is None:
            return 0
        return sum(1 for e in self.director.events
                   if e["kind"] in ("cast", "trigger", "lane_change"))

    @property
    def status(self) -> str:
        if self.director is None:
            return "scripted actors (no casting in this mode)"
        ev = self.director.events
        return ev[-1]["text"] if ev else "positioning for the cut-in"

    def summary(self) -> dict:
        if self.director is None:
            return {"casting": False, "holder": None, "outcome": None,
                    "committed": False, "t_commit": None, "interventions": 0,
                    "recasts": 0, "roles": {}, "scores": {},
                    "events": [e.__dict__ for e in self.events]}
        out = self.director.summary()
        out.update({"casting": True, "merged": self.merged,
                    "t_commit": (round(self.t_commit, 3)
                                 if self.t_commit is not None else None),
                    "interventions": self.n_interventions,
                    "roles": self.roles})
        return out
