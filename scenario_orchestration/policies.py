#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scenario_orchestration/policies.py — `policy.json` -> an ego this port can drive.

Two kinds of ego policy reach this method, and they are realized differently.

**Analytic** (`idm`, `idm_assertive`, `idm_conservative`, `mobil`). Fully
determined by the parameters in the request, so there is no repository to load:
the request is bound onto `carla_highway/highway_ego.py`'s own constants and the
port's native ego runs. `highway_ego` reads them as module globals on every
call, which is what makes rebinding the whole translation.

**Learned** (`simlingo`, `plant2`, `tfv6`, ...). Loaded from their own
repository through the `ego_policy_v1` entry point that repository publishes,
and driven through `carla_port.ego_driver.PolicyEgoDriver`. Nothing about a
particular policy is known here.

Why IDM and MOBIL are two policies and not one
----------------------------------------------
They are the two halves of the port's ego and they answer different questions:
IDM is longitudinal (how fast), MOBIL is lateral (which lane). Running the ego
with MOBIL off is a real and useful baseline — it is the lane-keeping driver
that `scenario_hard_brake` and `scenario_overtake` are unsolvable for, which is
the evidence that those families need a lateral decision at all. So:

    idm    IDM longitudinal, lane-keeping        (MOBIL off)
    mobil  IDM longitudinal + MOBIL lateral      (the full model-based ego)

Both are the same code with one flag between them, and reporting them as two
policies is what lets a results table show the difference.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contract import EGO_POLICY_INTERFACE, PolicyRequest

#: request parameter -> the constant it binds in `carla_highway.highway_ego`
IDM_BINDING = {
    "desired_speed_mps": "IDM_V0",
    "max_accel_mps2": "IDM_A",
    "comfort_decel_mps2": "IDM_B",
    "min_gap_m": "IDM_S0",
    "time_headway_s": "IDM_T",
    # Not declared by any harness policy config, but part of IDM and accepted
    # so an experiment can sweep it without a code change here.
    "acceleration_exponent": "IDM_DELTA",
}

#: request parameter -> the MOBIL constant it binds
MOBIL_BINDING = {
    "politeness": "MOBIL_P",
    "switch_threshold_mps2": "MOBIL_A_THR",
    "safe_decel_mps2": "MOBIL_B_SAFE",
    "min_interval_s": "MOBIL_MIN_INTERVAL",
    "min_speed_mps": "MOBIL_V_MIN",
    "settle_tolerance_m": "MOBIL_SETTLE_TOL",
    "keep_home_bias_mps2": "MOBIL_BIAS_HOME",
    "oncoming_bias_mps2": "MOBIL_BIAS_ONCOMING",
}

#: Policy names realized analytically, and whether each drives MOBIL.
ANALYTIC_POLICIES = {
    "idm": False,
    "idm_assertive": False,
    "idm_conservative": False,
    "mobil": True,
}


class PolicyTranslationError(Exception):
    """The request cannot be realized by this method."""


@dataclass
class AnalyticBinding:
    """The result of binding a request onto the port's own ego."""
    policy: str
    lane_changes: bool
    applied: Dict[str, float] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    desired_speed: Optional[float] = None

    def metadata(self) -> Dict[str, Any]:
        meta = {
            "policy": self.policy,
            "realized_by": "carla_highway.highway_ego.HighwayEgoPolicy",
            "actuates": "control",
            "lateral_model": "MOBIL" if self.lane_changes else "lane-keeping",
            "constants": dict(self.applied),
            "ignored_parameters": list(self.ignored),
            "notes": list(self.notes),
        }
        if self.desired_speed is not None:
            meta["desired_speed_mps"] = round(self.desired_speed, 3)
        return meta


def is_analytic(request: PolicyRequest) -> bool:
    """Whether this request is one the port realizes with its own ego.

    Decided from the declaration, not just the name: a policy calling itself
    `idm` but wanting sensor input or emitting trajectories is not the analytic
    ego this port has, and quietly running IDM under its name would be worse
    than reporting that it cannot be run.
    """
    if request.interface != EGO_POLICY_INTERFACE:
        return False
    if request.observation_space != "state" or request.action_space != "control":
        return False
    name = (request.name or "").lower()
    if name in ANALYTIC_POLICIES:
        return True
    return (("idm" in name or "mobil" in name)
            and not request.requires and not request.checkpoint)


def wants_lane_changes(request: PolicyRequest) -> bool:
    """Does this analytic request drive MOBIL, or keep its lane?"""
    name = (request.name or "").lower()
    if name in ANALYTIC_POLICIES:
        return ANALYTIC_POLICIES[name]
    return "mobil" in name


def bind_analytic(request: PolicyRequest) -> AnalyticBinding:
    """Bind an analytic request onto `carla_highway.highway_ego`'s constants.

    Applied to the module object rather than to a copy, because the ego's
    `_idm_accel` and its MOBIL incentive both resolve these as module globals
    on every call. Rebinding them is therefore the whole translation, and it
    leaves the ego's code untouched.
    """
    import carla_highway.highway_ego as hw_ego

    binding = AnalyticBinding(policy=request.name or "idm",
                              lane_changes=wants_lane_changes(request))
    both = dict(IDM_BINDING)
    both.update(MOBIL_BINDING)
    for key, value in (request.parameters or {}).items():
        target = both.get(key)
        if target is None:
            binding.ignored.append(key)
            continue
        if not hasattr(hw_ego, target):
            binding.ignored.append(key)
            binding.notes.append(
                f"parameter {key!r} maps to {target}, which this ego does not define")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            binding.ignored.append(key)
            binding.notes.append(f"parameter {key}={value!r} is not numeric")
            continue
        setattr(hw_ego, target, number)
        binding.applied[target] = number

    if not binding.lane_changes:
        mobil_params = [k for k in (request.parameters or {}) if k in MOBIL_BINDING]
        if mobil_params:
            binding.notes.append(
                "MOBIL parameters were bound but this policy keeps its lane, so "
                "they have no effect: " + ", ".join(sorted(mobil_params)))

    v0 = binding.applied.get("IDM_V0", hw_ego.IDM_V0)
    if v0 > hw_ego.V_MAX:
        # The ego's actuation envelope, not its IDM law: the bicycle model and
        # the longitudinal PID both clamp to V_MAX, so a desired speed above it
        # is simply unreachable. Reported rather than silently raised, because
        # V_MAX is the vehicle model and the request only describes the driver.
        binding.notes.append(
            f"desired_speed_mps {v0:.2f} exceeds the ego's V_MAX "
            f"{hw_ego.V_MAX:.2f} m/s; the ego cannot reach it")
    if binding.ignored:
        binding.notes.append("parameters with no equivalent were ignored: "
                             + ", ".join(sorted(binding.ignored)))
    binding.desired_speed = float(v0)
    return binding


# --------------------------------------------------------------------------- #
# External ego_policy_v1 repositories
# --------------------------------------------------------------------------- #
@dataclass
class LoadedPolicy:
    name: str
    policy: Any
    repository: str
    resolution: str
    request: PolicyRequest

    def metadata(self) -> Dict[str, Any]:
        meta = {"policy": self.name, "repository": self.repository,
                "resolution": self.resolution,
                "interface": self.request.interface,
                "observation_space": self.request.observation_space,
                "action_space": self.request.action_space}
        own = getattr(self.policy, "metadata", None)
        if callable(own):
            try:
                meta["policy_metadata"] = own()
            except Exception as exc:            # diagnostics only
                meta["policy_metadata_error"] = f"{type(exc).__name__}: {exc}"
        return meta


def resolve_repository(request: PolicyRequest, harness_root: Optional[str],
                       repo_root: str) -> tuple[str, str]:
    """Where this policy's repository is, and how that was decided.

    `$<POLICY>_ROOT` wins: it describes the machine rather than the experiment,
    which is what makes a policy findable on a cluster without a harness
    checkout beside it.
    """
    env_key = f"{(request.name or '').upper().replace('-', '_')}_ROOT"
    env_root = os.environ.get(env_key)
    if env_root:
        return env_root, f"${env_key}"
    declared = request.repository or ""
    for base, how in ((harness_root, "harness root"),
                      (os.path.dirname(repo_root), "sibling of this repository")):
        if not base:
            continue
        candidate = os.path.join(base, declared)
        if os.path.isdir(candidate):
            return candidate, f"{how} + policy.repository"
    raise PolicyTranslationError(
        f"cannot locate the repository for policy {request.name!r}: "
        f"{declared!r} was not found under the harness root or beside this "
        f"repository, and ${env_key} is not set")


def load_policy(request: PolicyRequest, harness_root: Optional[str],
                repo_root: str) -> LoadedPolicy:
    """Import a policy repository's `ego_policy_v1` entry point and build it."""
    repository, resolution = resolve_repository(request, harness_root, repo_root)
    entry = os.path.join(repository, request.entry_point or
                         "scenario_orchestration/policy.py")
    if not os.path.isfile(entry):
        raise PolicyTranslationError(
            f"policy {request.name!r} declares entry point {request.entry_point!r}, "
            f"which does not exist at {entry}")
    spec = importlib.util.spec_from_file_location(
        f"_policy_{request.name}", entry)
    if spec is None or spec.loader is None:            # pragma: no cover
        raise PolicyTranslationError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    # The policy's own repository must be importable while it loads.
    for path in (os.path.dirname(entry), repository):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    factory = None
    for attr in ("build_policy", "make_policy", "load_policy", "Policy"):
        factory = getattr(module, attr, None)
        if callable(factory):
            break
    if factory is None:
        raise PolicyTranslationError(
            f"{entry} exposes no build_policy()/make_policy() factory; "
            "ego_policy_v1 requires one")
    policy = factory(request.to_dict() if hasattr(request, "to_dict") else request)
    loader = getattr(policy, "load", None)
    if callable(loader):
        policy = loader() or policy
    return LoadedPolicy(name=request.name or "policy", policy=policy,
                        repository=repository, resolution=resolution,
                        request=request)


def describe_unsupported(request: PolicyRequest) -> str:
    """Why a request this method cannot run cannot be run."""
    return (f"policy {request.name!r} declares interface {request.interface!r}, "
            f"observation space {request.observation_space!r} and action space "
            f"{request.action_space!r}, and names no entry point to load. This "
            f"method realizes {sorted(ANALYTIC_POLICIES)} analytically and any "
            f"other ego_policy_v1 policy from its own repository.")
