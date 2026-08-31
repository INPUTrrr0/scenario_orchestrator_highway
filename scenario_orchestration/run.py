#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scenario_orchestration/run.py — this repository's harness entry point.

The evaluation harness (github.com/friedeggs/scenario_orchestration) reaches
every method the same way: one subprocess per experiment cell, two JSON
documents in, one JSON document out.

    python scenario_orchestration/run.py \
        --scenario-request request.json \
        --policy-request  policy.json \
        --output-dir      results/raw/<experiment_id>

    -> <output-dir>/method_result.json   {"status", "metrics", "trace_path"}
       <output-dir>/trace.json           what the orchestrator decided, per tick

Nothing is imported from the harness: `contract.py` next to this file is this
repository's own copy of the two document schemas, which is what lets the
method use whatever Python, CUDA and simulator version it needs.

What this method realizes
-------------------------
The three straight-road scenario families, as `carla_highway`'s three modes:

    cut_in       -> `cutin`        an adjacent actor is cast and merges into the
                                   ego lane inside its safety envelope
    lane_change  -> `hard_brake`   a slow lead in the ego's lane with traffic
                                   squeezing the adjacent one
    overtake     -> `overtake`     a stopped blocker with oncoming traffic
                                   constraining the only way past

`lane_change` maps to the mode this repository calls `hard_brake` because that
is the same situation under this repository's own name for it, and it satisfies
all three of the family's declared preconditions. The junction families
(`red_light`, `left_turn`, `right_turn`) belong to a different method; this
repository's `carla_port/` carries only the script-agnostic CARLA mechanics
that `carla_highway` imports, not the junction port.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (HERE, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from contract import (METHOD_RESULT_FILE, TRACE_FILE, MethodResult,  # noqa: E402
                      PolicyRequest, ScenarioRequest, failure, write_json)

#: Scenario family -> the mode `carla_highway/scenarios.py` realizes it as.
#: `capabilities.json` declares the keys of this mapping and nothing else.
FAMILY_TO_MODE = {
    "cut_in": "cutin",
    "lane_change": "hard_brake",
    "overtake": "overtake",
}

#: Said in the run notes so a reader of the results does not have to work out
#: why a family is running under a different name.
FAMILY_NOTE = {
    "lane_change": "realized by this repository's `hard_brake` mode: a 4 m/s "
                   "lead in the ego lane with 10.5 m/s traffic squeezing the "
                   "adjacent one, which is this family's situation under the "
                   "name this repository gave it",
}

#: Towns whose road fit has actually been measured. Town04 is the only stock
#: town where all three families fit at their authored scale: it has both the
#: multi-lane highway and a two-way road, which `overtake` requires. A town
#: that is not listed still runs -- the fitter is generic -- and the report's
#: `frame` block carries the lane fit error it achieved.
MEASURED_TOWNS = ("Town04", "Town05")

#: Every knob this entry point accepts, as (name, type, default). Read from the
#: request's parameter blocks, or from `ORCHESTRATION_<NAME>` in the environment.
SETTINGS: Dict[str, Tuple[type, Any]] = {
    # CARLA connection. Machine-level, so usually environment rather than config.
    "carla_host": (str, "127.0.0.1"),
    "carla_port": (int, 2000),
    "carla_timeout": (float, 20.0),
    "carla_load_timeout": (float, 180.0),
    # Road.
    "town": (str, "Town04"),
    "road_id": (int, None),
    "min_length": (float, None),
    "xodr": (str, None),                   # a purpose-built road instead
    # Simulation.
    "fixed_delta": (float, None),          # None = the script's own DT (1/60 s)
    "sync_mode": (str, "physics"),
    "no_rendering": (bool, False),
    "linger": (float, 1.0),
    "z_offset": (float, 0.10),
    "reground_every": (int, 0),
    # Orchestration.
    "casting": (bool, None),
    "cutin_at": (float, None),
    "cruise": (float, 12.0),
    "interaction_gap_m": (float, 4.0),
    "lane_conflict_m": (float, 2.2),
    "base_scenario": (str, None),
    "ego": (str, "0"),
    # Ego policy.
    "policy_hz": (float, 20.0),
    "harness_root": (str, None),
    # Artifacts. Recording is off by default: it costs two rendered cameras per
    # tick and needs ffmpeg, which a matrix sweep does not want.
    "record_video": (bool, False),
    "video_view": (str, "both"),
    "video_fps": (float, 30.0),
}

#: Request parameters that describe the scenario rather than configure the run.
DESCRIPTIVE = frozenset({"town"})

#: Files this method writes beyond method_result.json.
NATIVE_REPORT_FILE = "orchestration_report.json"


def _coerce(name: str, kind: type, value: Any) -> Any:
    if value is None:
        return None
    if kind is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"setting {name}={value!r} is not a {kind.__name__}") from exc


class Settings:
    """The resolved configuration, and where each value came from."""

    def __init__(self, request: ScenarioRequest):
        declared = request.settings()
        self.values: Dict[str, Any] = {}
        self.sources: Dict[str, str] = {}
        for name, (kind, default) in SETTINGS.items():
            env = os.environ.get(f"ORCHESTRATION_{name.upper()}")
            if env is not None:
                self.values[name] = _coerce(name, kind, env)
                self.sources[name] = f"$ORCHESTRATION_{name.upper()}"
            elif name in declared:
                self.values[name] = _coerce(name, kind, declared[name])
                self.sources[name] = "request"
            else:
                self.values[name] = default
                self.sources[name] = "default"
        for name, alias in (("carla_host", "CARLA_HOST"),
                            ("carla_port", "CARLA_PORT"),
                            ("carla_timeout", "CARLA_TIMEOUT")):
            if self.sources[name] == "default" and os.environ.get(alias):
                self.values[name] = _coerce(name, SETTINGS[name][0],
                                            os.environ[alias])
                self.sources[name] = f"${alias}"
        self.declared = dict(declared)
        self.unknown = sorted(k for k in declared
                              if k not in SETTINGS and k not in DESCRIPTIVE)

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def describe(self) -> Dict[str, Any]:
        return {k: {"value": v, "from": self.sources[k]}
                for k, v in sorted(self.values.items())}


def build_config(request: ScenarioRequest, settings: Settings,
                 output_dir: str, notes: List[str]):
    """`request.json` -> `carla_highway`'s RunConfig.

    Anything the harness states wins over this method's own default: the
    horizon becomes the run duration, and the seed and town come from the
    request.
    """
    from carla_highway import scenarios as hw_scenarios
    from carla_highway.closed_loop import DT_TICK
    from carla_highway.runner import RunConfig
    from carla_highway.script_bridge import DT

    family = request.scenario_family
    mode = FAMILY_TO_MODE[family]
    if family in FAMILY_NOTE:
        notes.append(FAMILY_NOTE[family])

    town = settings["town"]
    if town and town not in MEASURED_TOWNS and not settings["xodr"]:
        notes.append(
            f"town {town!r} is not one whose road fit this method has measured "
            f"({', '.join(MEASURED_TOWNS)}); the fitter still runs, and the "
            "report's frame block carries the lane fit error it achieved")

    horizon = float(request.evaluation.horizon_s)
    spec = hw_scenarios.spec(mode)
    tuned = float(getattr(spec, "duration", horizon) or horizon)
    if horizon < tuned:
        notes.append(
            f"the family's horizon is {horizon:g}s and the {mode!r} mode is "
            f"tuned to resolve in {tuned:g}s; the horizon is honoured, so the "
            "interaction may not have happened yet when the episode ends")

    fixed_delta = settings["fixed_delta"] or DT
    tick_hz = float(request.evaluation.tick_rate_hz or 0.0)
    if tick_hz > 0 and abs(1.0 / tick_hz - DT_TICK) > 1e-6:
        notes.append(
            f"the evaluation protocol asks for {tick_hz:g} Hz orchestration; "
            f"this method orchestrates at {1.0 / DT_TICK:g} Hz "
            "(carla_highway/closed_loop.DT_TICK) and the requested rate was "
            "not applied")

    cfg = RunConfig(
        scenario=mode,
        host=settings["carla_host"], port=settings["carla_port"],
        timeout=settings["carla_timeout"],
        load_timeout=settings["carla_load_timeout"],
        town=town, duration=horizon, fixed_delta=fixed_delta,
        no_rendering=bool(settings["no_rendering"]),
        road_id=settings["road_id"], min_length=settings["min_length"],
        base=settings["base_scenario"], sync_mode=settings["sync_mode"],
        z_offset=settings["z_offset"], reground_every=settings["reground_every"],
        ego=settings["ego"], ego_mode="physics",
        casting=settings["casting"], cruise=settings["cruise"],
        cutin_at=settings["cutin_at"],
        interaction_gap_m=settings["interaction_gap_m"],
        lane_conflict_m=settings["lane_conflict_m"],
        linger=settings["linger"], spectator=False, verbose=True,
        no_video=not bool(settings["record_video"]),
        video_dir=os.path.join(output_dir, "video"),
        video_view=settings["video_view"], video_fps=settings["video_fps"],
        report=None, policy_hz=float(settings["policy_hz"]),
    )
    if settings["xodr"]:
        cfg.xodr = settings["xodr"]
    return cfg


def build_ego(policy: PolicyRequest, cfg, settings: Settings,
              notes: List[str]) -> Tuple[Optional[Any], Dict[str, Any]]:
    """`policy.json` -> either a binding onto this port's ego, or a loaded one."""
    import policies

    if policies.is_analytic(policy):
        binding = policies.bind_analytic(policy)
        notes.extend(binding.notes)
        cfg.no_lane_change = not binding.lane_changes
        cfg.lane_change = binding.lane_changes
        if binding.desired_speed is not None:
            cfg.desired_speed = float(binding.desired_speed)
        print(f"[run] ego policy {binding.policy!r} realized analytically "
              f"({'IDM + MOBIL' if binding.lane_changes else 'IDM, lane-keeping'})"
              + (": " + ", ".join(f"{k}={v:g}"
                                  for k, v in sorted(binding.applied.items()))
                 if binding.applied else ""), flush=True)
        return None, binding.metadata()

    if not policy.entry_point:
        raise policies.PolicyTranslationError(policies.describe_unsupported(policy))

    loaded = policies.load_policy(policy, harness_root=settings["harness_root"],
                                  repo_root=REPO_ROOT)
    print(f"[run] ego policy {loaded.name!r} loaded from {loaded.repository} "
          f"({loaded.resolution})", flush=True)
    meta = loaded.metadata()
    meta.update({"realized_by": "carla_port.ego_driver.PolicyEgoDriver",
                 "actuates": "control",
                 "declared_action_space": policy.action_space})
    if policy.action_space != "control":
        notes.append(
            f"ego policy {loaded.name!r} declares action space "
            f"{policy.action_space!r}; it is actuated through the 'control' "
            "block its own controllers return, so the controller is the policy's")
    return loaded, meta


def write_trace(output_dir: str, request: ScenarioRequest,
                policy: PolicyRequest, report: Dict[str, Any], run) -> str:
    """The per-run trace: what the orchestrator decided, step by step."""
    trace = {
        "experiment_id": request.experiment_id,
        "scenario_family": request.scenario_family,
        "native_id": request.native_id,
        "algorithm": request.algorithm,
        "policy": policy.name,
        "seed": request.seed,
        "horizon_s": request.evaluation.horizon_s,
        "target_event": request.evaluation.target_event,
        "sim_time_s": report.get("sim_time"),
        "frame": report.get("frame"),
        "orchestration": report.get("orchestration"),
        "interactions": report.get("interactions"),
        "ego": report.get("ego"),
        "ego_lane_track": report.get("ego_lane_track"),
        "realized_collisions": report.get("realized_collisions"),
        "grade": report.get("grade"),
        "trajectories": report.get("trajectories"),
        "notes": report.get("notes"),
    }
    driver = getattr(run, "ego_driver", None)
    if driver is not None and getattr(driver, "actions", None):
        trace["ego_policy_actions"] = list(driver.actions)
    write_json(os.path.join(output_dir, TRACE_FILE), trace)
    return TRACE_FILE


def _salvage(run, notes: List[str]) -> Dict[str, Any]:
    """The native report from a run that died, if it can still be produced."""
    try:
        return run.report()
    except Exception as exc:                # noqa: BLE001 - diagnostics only
        notes.append(f"no partial report could be produced: "
                     f"{type(exc).__name__}: {exc}")
        return {}


def _is_declined(exc: BaseException) -> bool:
    """Did the method decline the job, rather than break on it?

    A road the scenario does not fit on is a `failure` — this method genuinely
    cannot run that cell — while an unexpected exception is an `error`.
    """
    return isinstance(exc, (RuntimeError, ValueError)) and any(
        s in str(exc).lower() for s in
        ("no straight section", "no lane", "cannot", "unknown scenario mode"))


def execute(request: ScenarioRequest, policy: PolicyRequest,
            output_dir: str) -> MethodResult:
    """Translate, run, and project the report onto the canonical vocabulary."""
    import metrics as metrics_module

    notes: List[str] = []
    settings = Settings(request)
    if settings.unknown:
        notes.append("request parameters this method does not recognize were "
                     "ignored: " + ", ".join(settings.unknown))

    if request.scenario_family not in FAMILY_TO_MODE:
        return failure(
            f"scenario family {request.scenario_family!r} has no implementation "
            f"in this repository; it realizes {sorted(FAMILY_TO_MODE)} (see "
            f"scenario_orchestration/capabilities.json).",
            settings=settings.describe())

    from carla_highway.runner import HighwayRun, summarize

    cfg = build_config(request, settings, output_dir, notes)
    loaded, policy_meta = build_ego(policy, cfg, settings, notes)

    run = HighwayRun(cfg)
    if loaded is not None:
        cfg.policy = cfg.policy_request = None
        run.external_policy = loaded

    started = time.time()
    report: Dict[str, Any] = {}
    crashed: Optional[BaseException] = None
    try:
        run.setup()
        report = run.run()
    except Exception as exc:                # noqa: BLE001 - reported, not swallowed
        # A run that died at 15 s of a 20 s horizon still measured 15 s, and
        # throwing that away leaves nothing to debug from. It is NOT reported
        # as metrics -- a truncated episode's numbers are not the episode's --
        # but it is reported.
        crashed = exc
        traceback.print_exc()
        report = _salvage(run, notes)
    finally:
        try:
            run.teardown()
        except Exception as exc:                     # pragma: no cover
            notes.append(f"teardown: {type(exc).__name__}: {exc}")

    if crashed is None:
        print(summarize(report), flush=True)
        canonical, method_metrics = metrics_module.canonical(
            report, request, ego=cfg.ego)
        status, reason = "success", None
    else:
        canonical, method_metrics = {}, {"partial_report": report}
        status = "failure" if _is_declined(crashed) else "error"
        reason = (str(crashed) if _is_declined(crashed) else
                  "".join(traceback.format_exception_only(
                      type(crashed), crashed)).strip())
        method_metrics["traceback"] = traceback.format_exc().splitlines()[-12:]

    method_metrics.update({
        "wall_time_s": round(time.time() - started, 3),
        "policy": policy_meta,
        "settings": settings.describe(),
        "run_notes": list(notes) + list(report.get("notes") or []),
    })

    trace = write_trace(output_dir, request, policy, report, run)
    write_json(os.path.join(output_dir, NATIVE_REPORT_FILE), report)
    return MethodResult(status=status, metrics=canonical, reason=reason,
                        method_metrics=method_metrics, trace_path=trace)


def _check_environment(request: ScenarioRequest) -> None:
    """Cross-check the environment the harness exports against the request."""
    pairs = (("SCENARIO_ORCHESTRATION_EXPERIMENT_ID", request.experiment_id),
             ("SCENARIO_ORCHESTRATION_SEED", str(request.seed)))
    for name, expected in pairs:
        actual = os.environ.get(name)
        if actual is not None and expected and actual != expected:
            print(f"[run] warning: {name}={actual!r} disagrees with the "
                  f"request's {expected!r}; the request wins", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scenario_orchestration/run.py",
        description="Run one experiment cell of the highway orchestrator.")
    parser.add_argument("--scenario-request", required=True,
                        help="path to the harness's request.json")
    parser.add_argument("--policy-request", required=True,
                        help="path to the harness's policy.json")
    parser.add_argument("--output-dir", required=True,
                        help="where method_result.json and trace.json go")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    request = ScenarioRequest.from_json(args.scenario_request)
    policy = PolicyRequest.from_json(args.policy_request)
    _check_environment(request)

    try:
        result = execute(request, policy, output_dir)
    except Exception as exc:                # noqa: BLE001 - always report
        traceback.print_exc()
        result = MethodResult(
            status="error",
            reason="".join(traceback.format_exception_only(
                type(exc), exc)).strip(),
            method_metrics={"traceback":
                            traceback.format_exc().splitlines()[-12:]})

    write_json(os.path.join(output_dir, METHOD_RESULT_FILE), result.to_dict())
    print(f"[run] {result.status}: {os.path.join(output_dir, METHOD_RESULT_FILE)}",
          flush=True)
    # The harness reads method_result.json; a non-zero exit would be reported as
    # a failure regardless of what that document says, so a reported outcome
    # exits zero and only an unwritable output directory is a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
