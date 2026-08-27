#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_highway — CARLA 0.9.16 backend for the highway maneuver-script
orchestrator (github.com/INPUTrrr0/scenario_orchestrator_highway).

    vendored highway orchestrator  +  a straight-road CARLA frame  +  an ego

Sibling of `carla_port`, which does the same job for the signalized-junction
scripts in `v2/` and `v4/`. The two share the script-agnostic CARLA mechanics
and nothing else — see `script_bridge` for why they cannot share a process.

Entry point: `python3 -m carla_highway --scenario cutin|hard_brake|overtake`.
"""
__all__ = ["runner", "scenarios", "highway_map", "highway_ego", "closed_loop"]
