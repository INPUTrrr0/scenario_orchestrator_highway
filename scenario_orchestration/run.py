#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The `scenario_orchestration` execution-method entry point for this repository.

    python scenario_orchestration/run.py \\
        --scenario-request /path/to/request.json \\
        --policy-request  /path/to/policy.json \\
        --output-dir      /path/to/results/raw/<experiment_id>

This is the whole of the method side of the contract (`third_party/README.md`,
DESIGN.md section 5). It imports nothing from the harness: the two request
documents are parsed by local mirrors in `contract.py`, and the one document it
writes back is `method_result.json`. It is deliberately the same shape as the
sibling file in `third_party/orchestration` -- same three mappings, same
failure-reporting rule, same settings mechanism -- because the two methods are
meant to be answering the same request the same way, and a difference between
them should be a difference in the method rather than in the plumbing.

What it does, and nothing more
------------------------------
Translate, execute, report. The closed-loop orchestration itself -- role
casting, the ego-relative cut-in pin, the rebase/retime/replan cycle, the
actor-actor collision-yield directive -- stays upstream in `cutin_orchestrator`
and the maneuver script, and the CARLA backend stays in `carla_highway/`, both
reached exactly as a human would from the command line.

    request.json  ->  carla_highway.runner.RunConfig
    policy.json   ->  the highway ego's IDM constants, or an ego_policy_v1 object
    report        ->  canonical metrics + method_metrics

The three mappings
------------------
**Scenario family.** The port realizes three highway modes, and the mapping onto
the harness's family names is one-to-one but not name-for-name:

    cut_in      -> cutin        3-lane one-way; the orchestrator CASTS a cut-in
                                and replans it against the live ego
    lane_change -> hard_brake   2-lane one-way; a slow lead in the ego's lane
                                and a squeezed merge gap in the next one --
                                which is `IMPLEMENTATION.md` section 5.3's
                                `lane_change` (merge) scenario exactly: "slow
                                lead constrains the ego lane while an adjacent
                                vehicle constrains the alternative lane"
    overtake    -> overtake     2-lane two-way; a stopped blocker and an
                                oncoming car in the only way past

`lane_change -> hard_brake` is the one worth checking rather than trusting: the
port's mode is named for what the *lead* does and the family for what the *ego*
must decide, and they are the same scenario seen from the two ends.

**Ego policy.** Two cases, both in `policies.py`: the analytic IDM family binds
onto `carla_highway/highway_ego.py`'s own IDM constants, and anything else is
loaded from its own repository through `ego_policy_v1` and drives the ego
through `carla_port/ego_driver.py`.

**Metrics.** `metrics.py`, and it says there why this method's self-reported
`scenario_realized` is deliberately weaker than its own per-mode grade. The
harness's own verdict comes from `metrics/` over the per-tick trace and lands in
`metrics_v2.json` beside these numbers rather than over them.

Configuration
-------------
Anything the port needs that the contract does not name is read from the
request's two parameter blocks -- `implementation.parameters` (per family, from
`scenarios/<family>/implementations.yaml`) and `parameters` (per method, from
`configs/algorithm/orchestrator_highway.yaml`) -- with the per-family block
winning. Every key is also overridable from the environment as
`ORCHESTRATOR_HIGHWAY_<KEY>`, which is what makes a sweep possible without
editing the harness's configs. `SETTINGS` below is the complete list.

Failure reporting
-----------------
A report is always written, and the process exits 0 whenever one was written, so
the harness reads this method's own status and reason rather than inferring one
from an exit code. A crash before the report can be written is the only case
that exits non-zero.
"""

from __future__ import annotations

import argparse
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

#: Scenario family -> the port's scenario mode. `capabilities.json` declares the
#: keys of this mapping and nothing else.
FAMILY_TO_MODE = {
    "cut_in": "cutin",
    "lane_change": "hard_brake",
    "overtake": "overtake",
}

#: Every knob this entry point accepts, as (name, type, default). Read from the
#: request's parameter blocks, or from `ORCHESTRATOR_HIGHWAY_<NAME>` in the
#: environment.
SETTINGS: Dict[str, Tuple[type, Any]] = {
    # CARLA connection. Machine-level, so usually environment rather than config.
    "carla_host": (str, "127.0.0.1"),
    "carla_port": (int, 2000),
    "carla_timeout": (float, 20.0),
    "carla_load_timeout": (float, 180.0),
    # Map and road.
    "town": (str, "Town04"),
    "road_id": (int, None),
    "min_length": (float, None),
    #: An OpenDRIVE straight generated by `carla_highway.make_maps`, instead of
    #: a stock town. "auto" picks the one authored for the mode. Left off by
    #: default: the stock towns are what the other two methods run on, and a
    #: generated world would make the comparison a comparison of roads.
    "xodr": (str, None),
    # Simulation.
    "fixed_delta": (float, None),          # None = the script's own DT (1/60 s)
    "sync_mode": (str, "physics"),         # how BACKGROUND actors are driven
    "no_rendering": (bool, False),
    "linger": (float, 1.0),
    "z_offset": (float, 0.10),
    "reground_every": (int, 0),
    # Orchestration.
    "casting": (bool, None),               # None = the mode's own default
    "cruise": (float, 12.0),
    "cutin_at": (float, None),
    "cutin_along": (float, None),
    #: Where the orchestrator aims the conflict, in seconds of margin at the
    #: point the two bodies have to share road. 0.0 is a collision. The mapping
    #: is per mode and is documented on `carla_highway.runner.RunConfig`; it is
    #: absolute for `cutin` and an offset for the two scripted modes, and the
    #: run report says which it was.
    "ttc_target": (float, None),
    # Ego.
    "desired_speed": (float, None),        # None = the mode's own free-flow
    "ego_mode": (str, "physics"),
    "no_lane_change": (bool, False),
    "lane_change": (bool, False),
    "policy_hz": (float, 20.0),
    "bev_map_folder": (str, "maps_2ppm_cv"),
    "keep_actions": (int, 0),
    "harness_root": (str, None),           # for resolving a policy's repository
    # Artifacts.
    "record_video": (bool, False),
    "video_view": (str, "both"),
    "video_fps": (float, 30.0),
    #: Per-tick trace (states.jsonl + scene.json) for the harness's metrics
    #: package. On by default: unlike video it costs no rendering, and the
    #: harness's scenario-success metric cannot be computed without it.
    "record_trace": (bool, True),
    "trace_rate_hz": (float, 10.0),
}

#: Request parameters that describe the scenario rather than configure the run.
#: Recognized so they are not reported as unrecognized on every single run.
DESCRIPTIVE = frozenset({"highway", "prompt", "scenario", "road"})

#: Files this method writes into --output-dir, beyond method_result.json.
NATIVE_REPORT_FILE = "highway_report.json"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
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

    ENV_PREFIX = "ORCHESTRATOR_HIGHWAY_"

    def __init__(self, request: ScenarioRequest):
        declared = request.settings()
        self.values: Dict[str, Any] = {}
        self.sources: Dict[str, str] = {}
        for name, (kind, default) in SETTINGS.items():
            env = os.environ.get(self.ENV_PREFIX + name.upper())
            if env is not None and env != "":
                self.values[name] = _coerce(name, kind, env)
                self.sources[name] = "$" + self.ENV_PREFIX + name.upper()
            elif name in declared:
                self.values[name] = _coerce(name, kind, declared[name])
                self.sources[name] = "request"
            else:
                self.values[name] = default
                self.sources[name] = "default"
        # Machine-level aliases, for a runner started by hand next to a server.
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
        value = self.values.get(name, default)
        return default if value is None else value

    def describe(self) -> Dict[str, Any]:
        return {"values": dict(self.values), "sources": dict(self.sources),
                "unrecognized_request_parameters": list(self.unknown)}


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
def build_config(request: ScenarioRequest, settings: Settings,
                 output_dir: str, notes: List[str]):
    """`request.json` -> the port's RunConfig."""
    from carla_highway import scenarios as sc_mod
    from carla_highway.highway_ego import IDM_V0, IDM_V0_CUTIN
    from carla_highway.runner import RunConfig
    from carla_highway.script_bridge import DT

    mode = FAMILY_TO_MODE[request.scenario_family]
    spec = sc_mod.spec(mode)

    horizon = float(request.evaluation.horizon_s or 0.0)
    if horizon <= 0:
        horizon = spec.duration
        notes.append(f"the protocol declared no horizon; the mode's own "
                     f"{spec.duration:g} s is used")
    elif horizon < spec.duration:
        # Reported rather than silently honoured: these scenarios are
        # timing-critical, and cutting the episode short can end it before the
        # conflict resolves, which reads as a policy that never got past the
        # blocker rather than as a horizon that was too short.
        notes.append(
            f"the protocol asks for a {horizon:g} s horizon; mode {mode!r} needs "
            f"{spec.duration:g} s to resolve, and the run is truncated")

    tick_hz = float(request.evaluation.tick_rate_hz or 0.0)
    fixed_delta = settings["fixed_delta"] or DT
    if tick_hz > 0 and abs(1.0 / tick_hz - 0.10) > 1e-6:
        notes.append(
            f"the protocol asks for {tick_hz:g} Hz orchestration; this method "
            "orchestrates at 10 Hz (carla_highway.closed_loop.DT_TICK) and the "
            "requested rate was not applied")

    desired = settings["desired_speed"]
    if desired is None:
        desired = IDM_V0_CUTIN if mode == sc_mod.CUTIN else IDM_V0

    if request.native_id and request.native_id != mode:
        notes.append(f"native_id {request.native_id!r} differs from the mode "
                     f"name; the mode run is {mode!r}")

    return RunConfig(
        scenario=mode,
        host=settings["carla_host"], port=settings["carla_port"],
        timeout=settings["carla_timeout"],
        load_timeout=settings["carla_load_timeout"],
        town=settings["town"], xodr=settings["xodr"],
        duration=horizon, linger=settings["linger"], fixed_delta=fixed_delta,
        no_rendering=bool(settings["no_rendering"]),
        road_id=settings["road_id"], min_length=settings["min_length"],
        sync_mode=settings["sync_mode"], z_offset=settings["z_offset"],
        reground_every=settings["reground_every"],
        ego_mode=settings["ego_mode"], scripted_ego=False,
        desired_speed=desired,
        no_lane_change=bool(settings["no_lane_change"]),
        lane_change=bool(settings["lane_change"]),
        ego_policy=True,
        casting=settings["casting"], cruise=settings["cruise"],
        cutin_at=settings["cutin_at"], cutin_along=settings["cutin_along"],
        ttc_target=settings["ttc_target"],
        video=None, video_dir=os.path.join(output_dir, "video"),
        video_view=settings["video_view"], video_fps=settings["video_fps"],
        no_video=not bool(settings["record_video"]),
        # The spectator camera is a courtesy to a human watching one run; on a
        # sweep it is a per-tick RPC for nobody.
        spectator=False,
        report=os.path.join(output_dir, NATIVE_REPORT_FILE),
        verify_report=os.path.join(output_dir, "highway_report.verify.json"),
        trace_dir=(output_dir if settings["record_trace"] else None),
        trace_rate_hz=(tick_hz if tick_hz > 0 else settings["trace_rate_hz"]),
        verbose=True,
    )


def build_ego_driver(policy: PolicyRequest, settings: Settings,
                     notes: List[str]) -> Tuple[Optional[Any], Dict[str, Any]]:
    """`policy.json` -> either a parameter binding or an external ego driver.

    Returns (loaded, metadata). A None `loaded` means the ego is the highway
    port's own IDM with the request's parameters bound onto it; the port then
    runs exactly as it does from the command line.
    """
    import policies

    if policies.is_analytic_idm(policy):
        binding = policies.bind_idm(policy)
        notes.extend(binding.notes)
        print(f"[run] ego policy {binding.policy!r} realized analytically: "
              + ", ".join(f"{k}={v:g}" for k, v in sorted(binding.applied.items())),
              flush=True)
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
            f"{policy.action_space!r}; it is actuated through the 'control' block "
            "its own controllers return, so the controller is the policy's")
    return loaded, meta


def build_bev(loaded, policy: PolicyRequest, settings: Settings,
              notes: List[str]):
    """The BEV raster source an object-centric policy needs, if it needs one.

    Identical to the sibling port's, including the reason: every released
    PlanT 2.0 checkpoint is trained with `input_bev=True`, so a checkpoint handed
    nothing is not a result. `PLANT2_BLANK_BEV` is the escape hatch and it says
    in the report that the run's driving is not meaningful.

    A `sensor` policy is skipped outright. It reads the rig
    (`ego_driver._build_rig`), never the raster, and building one anyway would
    make an unrelated dependency of the policy repository -- h5py, and a town
    raster that ships only for some maps -- a hard prerequisite for running a
    camera policy that has no use for either.
    """
    if policy.observation_space == "sensor":
        return None

    import bev as bev_module

    if bev_module.blank_allowed():
        notes.append(
            "PLANT2_BLANK_BEV is set, so no BEV raster is supplied and the policy "
            "substitutes a blank one; driving behaviour from this run is not "
            "meaningful and its metrics should not be reported as results")
        return None
    return bev_module.build(loaded.repository,
                            map_folder=settings["bev_map_folder"])


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
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
            f"scenario_orchestration/capabilities.json). The intersection "
            f"families are the sibling port's, third_party/orchestration.",
            settings=settings.describe())

    from carla_highway.runner import HighwayRun, summarize
    from carla_port.ego_driver import PolicyEgoDriver

    cfg = build_config(request, settings, output_dir, notes)
    loaded, policy_meta = build_ego_driver(policy, settings, notes)

    run = HighwayRun(cfg)
    if loaded is not None:
        # The port loads an external policy itself from `cfg.policy_request`,
        # but that path re-reads the document and re-resolves the repository.
        # This one is already resolved, so the driver is injected instead --
        # which is also what lets the BEV source be attached, since the port's
        # own path does not build one.
        run.ego_driver = PolicyEgoDriver(
            loaded.policy, name=loaded.name, hz=float(settings["policy_hz"]),
            bev=build_bev(loaded, policy, settings, notes),
            keep_actions=int(settings["keep_actions"]))
        run._external_policy_name = loaded.name

    started = time.time()
    report: Dict[str, Any] = {}
    crashed: Optional[BaseException] = None
    try:
        run.setup()
        report = run.run()
    except Exception as exc:                # noqa: BLE001 - reported, not swallowed
        crashed = exc
        traceback.print_exc()
        report = _salvage(run, notes)
    finally:
        try:
            run.teardown()
        except Exception as exc:                     # pragma: no cover
            notes.append(f"teardown: {type(exc).__name__}: {exc}")

    if crashed is None:
        try:
            print(summarize(report), flush=True)
        except Exception:                            # pragma: no cover
            pass
        canonical, method_metrics = metrics_module.canonical(
            report, request, ego=str(cfg.ego))
        status, reason = "success", None
    else:
        # No canonical metrics from a truncated episode: `scenario_realized`
        # would be false because the run died, not because the scenario failed,
        # and a value like that aggregated into a table is worse than a gap.
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
    if run.ego_driver is not None:
        method_metrics["ego_driver_actions"] = list(run.ego_driver.actions)

    trace = write_trace(output_dir, request, policy, report, run)
    write_json(os.path.join(output_dir, NATIVE_REPORT_FILE), report)

    return MethodResult(status=status, metrics=canonical, reason=reason,
                        method_metrics=method_metrics, trace_path=trace)


def _salvage(run, notes: List[str]) -> Dict[str, Any]:
    """The native report from a run that died, if it can still be produced.

    `report()` reads the frame, the loop and the interactions, all of which
    exist as soon as `setup()` has got past the road fit. A crash earlier than
    that leaves genuinely nothing to salvage, and says so.
    """
    try:
        return run.report()
    except Exception as exc:                # noqa: BLE001 - diagnostics only
        notes.append(f"no partial report could be produced: "
                     f"{type(exc).__name__}: {exc}")
        return {}


def write_trace(output_dir: str, request: ScenarioRequest,
                policy: PolicyRequest, report: Dict[str, Any], run) -> str:
    """The per-run trace: what the orchestrator decided, step by step.

    `trace.json` is the method's own narrative -- the cast, the recasts, the
    commit, every orchestration event with its timestamp -- and `states.jsonl`
    beside it is the canonical per-tick rollout the harness's metrics package
    evaluates. This points at the second rather than duplicating it.
    """
    orch = dict(report.get("orchestration") or {})
    trace = {
        "experiment_id": request.experiment_id,
        "scenario_family": request.scenario_family,
        "native_id": request.native_id,
        "algorithm": request.algorithm,
        "policy": policy.name,
        "seed": request.seed,
        "horizon_s": request.evaluation.horizon_s,
        "target_event": request.evaluation.target_event,
        "mode": report.get("scenario"),
        "sim_time_s": report.get("duration"),
        "frame": report.get("frame"),
        "orchestration": orch,
        "interventions": orch.get("events"),
        "cast": {"holder": orch.get("holder"), "roles": orch.get("roles"),
                 "scores": orch.get("scores"),
                 "outcome": orch.get("outcome"),
                 "t_commit": orch.get("t_commit")},
        "interactions": report.get("interactions"),
        "realized_collisions": report.get("realized_collisions"),
        "ego_collisions": report.get("ego_collisions"),
        "ego": report.get("ego"),
        "ego_driver": report.get("ego_driver"),
        "grade": report.get("grade"),
        "ttc_target": report.get("ttc_target"),
        "trace_v2": report.get("trace_v2"),
        "notes": report.get("notes"),
    }
    driver = getattr(run, "ego_driver", None)
    if driver is not None and getattr(driver, "actions", None):
        trace["ego_policy_actions"] = list(driver.actions)
    write_json(os.path.join(output_dir, TRACE_FILE), trace)
    return TRACE_FILE


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the highway orchestrator as a scenario_orchestration "
                    "execution method.")
    parser.add_argument("--scenario-request", required=True,
                        help="path to the harness's request.json")
    parser.add_argument("--policy-request", required=True,
                        help="path to the harness's policy.json")
    parser.add_argument("--output-dir", required=True,
                        help="where method_result.json and the trace are written")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        request = ScenarioRequest.from_json(args.scenario_request)
        policy = PolicyRequest.from_json(args.policy_request)
    except (OSError, ValueError) as exc:
        print(f"[run] cannot read the request documents: {exc}", file=sys.stderr)
        return 2

    _check_environment(request)
    print(f"[run] {request.experiment_id or '<unnamed>'}: family "
          f"{request.scenario_family!r}, policy {policy.name!r}, seed "
          f"{request.seed}", flush=True)

    try:
        result = execute(request, policy, output_dir)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:      # noqa: BLE001 - every failure is reportable
        print(traceback.format_exc(), file=sys.stderr)
        if _is_declined(exc):
            result = failure(str(exc), status="failure")
        else:
            detail = "".join(
                traceback.format_exception_only(type(exc), exc)).strip()
            result = failure(detail, status="error",
                             traceback=traceback.format_exc().splitlines()[-12:])

    result.write(output_dir)
    print(f"[run] {result.status}: wrote {METHOD_RESULT_FILE} to {output_dir}",
          flush=True)
    if result.reason:
        print(f"[run] reason: {result.reason}", flush=True)
    return 0


def _is_declined(exc: BaseException) -> bool:
    """Whether a failure is "this cell cannot be run here" rather than a bug.

    A reason is reported either way; the distinction is only whether `error`
    (something broke) or `failure` (something could not be run) is the truthful
    status. `incompatible` is never ours to report -- that is the harness's
    verdict, and a method that took it would be deciding what its own absence
    means for someone else's matrix.
    """
    declined: List[type] = [ImportError]
    for module_name, attribute in (("policies", "PolicyTranslationError"),
                                   ("bev", "BevError")):
        try:
            declined.append(getattr(__import__(module_name), attribute))
        except (ImportError, AttributeError):    # pragma: no cover
            pass
    return isinstance(exc, tuple(declined))


def _check_environment(request: ScenarioRequest) -> None:
    """Cross-check the environment the harness exports against the request."""
    pairs = (("SCENARIO_ORCHESTRATION_EXPERIMENT_ID", request.experiment_id),
             ("SCENARIO_ORCHESTRATION_SEED", str(request.seed)))
    for name, expected in pairs:
        actual = os.environ.get(name)
        if actual is not None and expected and actual != expected:
            print(f"[run] warning: {name}={actual!r} disagrees with the "
                  f"request's {expected!r}; the request wins", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
