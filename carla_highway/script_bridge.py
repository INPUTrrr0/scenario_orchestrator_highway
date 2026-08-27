#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway/script_bridge.py — import the highway script layer, unmodified.

This is the ONLY place that knows where the script layer lives. Import it from
here (`from carla_highway.script_bridge import se, mp, mv, dv, co`) so every
module in the port sees the *same* module objects.

On this branch the script layer is the repository root itself: `carla_port` is
a branch of

    https://github.com/INPUTrrr0/scenario_orchestrator_highway

so `scenario_editor.py`, `maps.py`, `maneuvers.py`, `directives.py` and
`cutin_orchestrator.py` sit one directory up, and nothing in them is edited by
the port. (On the older `scenario_editor` worktree these same files were
vendored under a `highway/` subdirectory; `HIGHWAY` below still names that
location, and still falls back to it, so the port runs unchanged either way.)

Why this cannot share a process with `carla_port.script_bridge`
---------------------------------------------------------------
Both script layers are *flat* module sets that claim the same bare top-level
names — `scenario_editor`, `directives`, `maneuvers`, `directives_script`,
`orchestrator` — from different files:

    v2/scenario_editor.py        (carla_port, via carla_port/script_bridge.py)
    highway/scenario_editor.py   (this port)

Python caches modules by name, so whichever is imported first wins and the
other silently gets the wrong class objects — the same identity hazard
`carla_port/script_bridge.py` documents for `v2.scenario_editor` vs
`scenario_editor`, one level up. Rather than paper over it, this module
*detects* it and fails loudly.

The two ports are therefore alternative backends over different script layers.
They share the CARLA-side mechanics that are script-agnostic
(`carla_port.carla_api`, `carla_adapter`, `carla_sync`, `carla_collision`,
`carla_video`, `carla_obs` — see the TYPE_CHECKING note in each) but never the
script modules. Nothing here imports `carla_port.script_bridge`, `carla_map`,
`carla_ego`, `closed_loop`, `scenarios` or `carla_runner`.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: where the highway script layer lives. The repository root on this branch;
#: a vendored `highway/` subdirectory when the port is dropped into another
#: tree. Probed rather than assumed so both layouts work.
_VENDORED = os.path.join(ROOT, "highway")
HIGHWAY = _VENDORED if os.path.isfile(
    os.path.join(_VENDORED, "cutin_orchestrator.py")) else ROOT

#: bare module names both script layers claim
CONTESTED = ("scenario_editor", "directives", "maneuvers", "directives_script",
             "orchestrator")


def _guard() -> None:
    """Refuse to import on top of a foreign script layer.

    A module already in sys.modules under a contested name is fine if it came
    out of HIGHWAY (a re-import of this bridge) and fatal otherwise (v2/ or
    v4/ got there first).
    """
    for name in CONTESTED:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        origin = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "") or ""))
        if origin and os.path.normpath(origin) != os.path.normpath(HIGHWAY):
            raise RuntimeError(
                f"module {name!r} is already imported from {origin!r}, not from "
                f"{HIGHWAY!r}. The highway script layer and carla_port's v2/v4 "
                "script layer claim the same bare module names and cannot share "
                "a process. Run carla_highway and carla_port in separate "
                "processes."
            )


_guard()

if HIGHWAY not in sys.path:
    sys.path.insert(0, HIGHWAY)

import scenario_editor as se          # noqa: E402  <HIGHWAY>/scenario_editor.py
import maps as mp                     # noqa: E402  <HIGHWAY>/maps.py
import maneuvers as mv                # noqa: E402  <HIGHWAY>/maneuvers.py
import directives as dv               # noqa: E402  <HIGHWAY>/directives.py
import cutin_orchestrator as co       # noqa: E402  <HIGHWAY>/cutin_orchestrator.py

DT = se.DT                            # script trajectory grid (1/60 s)

#: role names, re-exported so the port never restates the strings
ROLE_CUTIN = co.ROLE_CUTIN
ROLE_BLOCK = co.ROLE_BLOCK
ROLE_NOMINAL = co.ROLE_NOMINAL

__all__ = ["se", "mp", "mv", "dv", "co", "DT",
           "ROLE_CUTIN", "ROLE_BLOCK", "ROLE_NOMINAL", "ROOT", "HIGHWAY"]
