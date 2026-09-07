"""Translating a `PolicyRequest` into this repository's own policy plumbing.

DESIGN.md section 6: "A method adapter is responsible for translating a
standardized policy request into the policy interface required by that method."
That translation is this module, and it has exactly two cases.

Analytic policies (`idm`, `idm_assertive`, `idm_conservative`)
-------------------------------------------------------------
These are fully determined by the five numbers in the request, and the port
already drives its ego with an IDM: `carla_highway/highway_ego.py`'s, which is
drivev2's IDM law carried onto this branch under the same constant names
(`IDM_V0`, `IDM_A`, ...). So the translation is a parameter binding, not a new
policy — the request's `desired_speed_mps` becomes `IDM_V0`, and so on for the
other four. `HighwayEgoPolicy` resolves those names as module globals on every
call, so rebinding them is the whole translation and the ego stays the port's
own.

This file is the sibling of `third_party/orchestration`'s and differs from it in
exactly that one target module: the intersection port binds onto `v4/drivev2`,
this one onto `carla_highway.highway_ego`. Everything below -- resolving a
policy repository, loading its `policy.py`, the `ego_policy_v1` contract -- is
identical and deliberately unedited, because the point of the harness's `M + P`
integration cost is that the two ports answer the same request the same way.

External policies (`plant2`, and anything else speaking `ego_policy_v1`)
-----------------------------------------------------------------------
These live in their own repository and expose `scenario_orchestration/policy.py`.
The contract for that file is one factory and one step:

    build_policy(request: dict) -> policy
    policy.act(observation: dict) -> action

`build_policy` is called with the parsed `policy.json` verbatim. `act` is called
with a `state` observation (`carla_port/carla_obs.py`) and must return a mapping;
this port actuates the action's `control` block (`carla_port/ego_driver.py`).
`load()`, `reset()`, `close()` and `metadata()` are used when the policy has
them and skipped when it does not. Anything missing is reported by name rather
than as an AttributeError from inside a simulation loop.

The policy module is imported in-process. Both sides need CARLA's Python API and
the policy's own inference stack in one interpreter, which is what the project's
`carla-ubuntu20-plant2` Apptainer image provides (`apptainer shell --nvccli`).
A policy whose environment genuinely cannot be shared can still be driven over
its own `--serve` pipe; that is a transport this module does not need to own.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from contract import EGO_POLICY_INTERFACE, PolicyRequest

#: The factory `policy.py` must expose.
FACTORY = "build_policy"

#: The step method the returned object must expose.
STEP = "act"

#: The analytic IDM family, and how each declared parameter binds onto
#: `carla_highway/highway_ego.py`'s own IDM constants. Those names are the
#: target because the highway ego reads them by name off its own module on every
#: call, exactly as drivev2 does in the sibling port.
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

#: Policy names this repository realizes with the highway ego's own IDM.
IDM_POLICIES = ("idm", "idm_assertive", "idm_conservative")


class PolicyTranslationError(Exception):
    """The declared ego policy cannot be realized by this repository."""


# --------------------------------------------------------------------------- #
# The analytic IDM family
# --------------------------------------------------------------------------- #
@dataclass
class IdmBinding:
    """The result of binding a request onto the highway ego's IDM constants."""

    policy: str
    applied: Dict[str, float] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def metadata(self) -> Dict[str, Any]:
        return {"policy": self.policy,
                "realized_by": "carla_highway.highway_ego.HighwayEgoPolicy",
                "actuates": "control", "idm_constants": dict(self.applied),
                "ignored_parameters": list(self.ignored),
                "notes": list(self.notes)}


def bind_idm(request: PolicyRequest) -> IdmBinding:
    """Bind an analytic IDM request onto the highway ego's own IDM constants.

    Applied to the `highway_ego` module object rather than to a copy of the
    policy: the ego resolves `IDM_V0` and friends as module globals on every
    call, precisely so the IDM law stays the port's. Rebinding them is therefore
    the whole translation, and nothing in `carla_highway/` is edited.

    `IDM_V0_CUTIN` is bound too when a desired speed is given. The port carries
    a separate free-flow speed for the `cutin` mode -- the authored YAML runs its
    ego at 13 m/s rather than 12, and the cut-in pin was solved against that --
    and leaving it at its default would mean the request's desired speed silently
    did not apply to one of the three modes.
    """
    import carla_highway.highway_ego as dv2

    binding = IdmBinding(policy=request.name or "idm")
    for key, value in request.parameters.items():
        target = IDM_BINDING.get(key)
        if target is None:
            binding.ignored.append(key)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            binding.ignored.append(key)
            binding.notes.append(f"parameter {key}={value!r} is not numeric")
            continue
        setattr(dv2, target, number)
        binding.applied[target] = number

    v0 = binding.applied.get("IDM_V0", dv2.IDM_V0)
    if "IDM_V0" in binding.applied:
        dv2.IDM_V0_CUTIN = v0
        binding.applied["IDM_V0_CUTIN"] = v0
    if v0 > dv2.V_MAX:
        # The port's actuation envelope, not its IDM law: the bicycle model and
        # the longitudinal PID both clamp to V_MAX, so a desired speed above it
        # is simply unreachable. Reported rather than silently raised, because
        # V_MAX is the vehicle model and the request only describes the driver.
        binding.notes.append(
            f"desired_speed_mps {v0:.2f} exceeds the port's V_MAX "
            f"{dv2.V_MAX:.2f} m/s; the ego cannot reach it")
    if binding.ignored:
        binding.notes.append(
            "parameters with no IDM equivalent were ignored: "
            + ", ".join(sorted(binding.ignored)))
    return binding


# --------------------------------------------------------------------------- #
# External ego_policy_v1 repositories
# --------------------------------------------------------------------------- #
@dataclass
class LoadedPolicy:
    """An external policy object plus where it came from."""

    policy: Any
    name: str
    repository: str
    entry_point: str
    resolution: str            # how the repository path was found

    def metadata(self) -> Dict[str, Any]:
        return {"policy": self.name, "repository": self.repository,
                "entry_point": self.entry_point,
                "repository_resolved_by": self.resolution}


def resolve_repository(request: PolicyRequest, harness_root: Optional[str],
                       repo_root: str) -> Tuple[str, str]:
    """Where the policy repository actually is, and how we decided.

    The declared `repository` (`third_party/plant2`) is relative to the harness
    root and is the answer once the submodule is initialized, which is the
    expected steady state. Until then, or when the policy is checked out beside
    this repository rather than inside the harness, the earlier candidates
    apply. Every path tried is reported so a failure names them all.
    """
    candidates: List[Tuple[str, str]] = []

    override = request.parameters.get("repository_path")
    if override:
        candidates.append((str(override), "policy.parameters.repository_path"))

    env_name = f"{(request.name or 'policy').upper()}_ROOT"
    for name in (env_name, "EGO_POLICY_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append((value, f"${name}"))

    if request.repository:
        declared = str(request.repository)
        if os.path.isabs(declared):
            candidates.append((declared, "policy.repository"))
        else:
            if harness_root:
                candidates.append((os.path.join(harness_root, declared),
                                   "policy.repository (harness root)"))
            candidates.append((os.path.join(repo_root, declared),
                               "policy.repository (this repository)"))

    if request.name:
        # A sibling checkout: ../<name>, relative to this repository.
        candidates.append((os.path.join(os.path.dirname(repo_root), request.name),
                           "sibling checkout"))

    entry = request.entry_point or "scenario_orchestration/policy.py"
    tried: List[str] = []
    for path, how in candidates:
        absolute = os.path.abspath(os.path.expanduser(path))
        tried.append(f"{absolute} ({how})")
        if os.path.isfile(os.path.join(absolute, entry)):
            return absolute, how

    raise PolicyTranslationError(
        f"ego policy {request.name!r} declares {entry!r} but no checkout "
        f"exposing it was found. Initialize the submodule the harness declares "
        f"(git submodule update --init {request.repository}), or point "
        f"policy.parameters.repository_path / ${env_name} at the checkout. "
        f"Tried: " + "; ".join(tried))


#: Request keys whose value is a path when it is one. `checkpoint` is in the
#: contract; the rest are the conventional names a policy repository uses for a
#: file inside its checkpoint, and are only rewritten when the rewrite lands on
#: something that exists.
PATH_PARAMETERS = ("weights", "checkpoint", "config", "config_path",
                   "model_path", "weights_path")


def _absolutize(payload: dict, root: Optional[str]) -> dict:
    """Make a policy request's relative checkpoint paths absolute.

    The harness writes them relative to ITS root -- `checkpoint:
    third_party/checkpoints/tfv6/tfv6_resnet34` -- because that is where the
    declaration lives and where an operator reads it. The policy resolves them
    against the working directory, which for a method subprocess is the METHOD's
    repository. Those two only ever agreed by accident, and when they disagree
    the policy reports a missing checkpoint that is sitting right there.

    The same rewrite the OSC2 runner's bridge does
    (`osc2carla_policy_bridge.PolicyBridge._absolutize`), for the same reason and
    on the same keys, so a policy behaves identically under both methods.

    Rewritten only when the rewrite lands on something that exists, so a
    genuinely missing checkpoint still fails naming the path the harness asked
    for rather than one invented here.
    """
    if not root or not os.path.isdir(root):
        return payload

    def resolve(value):
        if not isinstance(value, str) or not value or os.path.isabs(value):
            return value
        candidate = os.path.join(root, value)
        if not os.path.exists(candidate):
            return value
        return os.path.abspath(candidate)

    payload["checkpoint"] = resolve(payload.get("checkpoint"))
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        for key in PATH_PARAMETERS:
            if key in parameters:
                parameters[key] = resolve(parameters[key])
    return payload


def _carla_python_api() -> Optional[str]:
    """`$CARLA_ROOT/PythonAPI/carla`, when that is a real directory.

    `CARLA_ROOT` is upstream's own name for this and describes the machine, so
    it is read from the environment rather than declared in a config.
    """
    root = (os.environ.get("CARLA_ROOT") or "").strip()
    if not root:
        return None
    api = os.path.join(root, "PythonAPI", "carla")
    return api if os.path.isdir(os.path.join(api, "agents")) else None


def load_module(repository: str, entry_point: str, name: str):
    """Import a policy repository's `policy.py` from its path.

    Imported by path rather than by module name: the file is the contract, the
    repository is not required to be an installable package, and two policies
    may each ship a `policy.py`. Its own directory goes on `sys.path` first,
    because a policy module may import its repository's siblings by bare name.

    Three paths, matching what the OSC2 runner's bridge puts there
    (`osc2carla_policy_bridge._load_module`), because the same three policies
    are loaded through both and neither should have to be told twice:

      the repository ROOT   a vision policy's adapter imports its own package by
                            bare name -- `lead.inference...` for TFv6 -- and the
                            entry point's directory is one level below it.
      the entry point's dir the adapter's own siblings.
      CARLA's PythonAPI     `$CARLA_ROOT/PythonAPI/carla` carries the `agents`
                            package (GlobalRoutePlanner, RoadOption) that ships
                            with the SERVER and not with the pip wheel. Every
                            route-conditioned policy needs it and the installed
                            ones disagree about whose job it is to find: SimLingo
                            adds it itself, TFv6 expects a symlink in its own
                            checkout. Doing it here means neither has to be right.
    """
    path = os.path.join(repository, entry_point)
    directory = os.path.dirname(path)
    for candidate in (repository, directory, _carla_python_api()):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
    module_name = f"ego_policy_{name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:      # pragma: no cover - unreadable file
        raise PolicyTranslationError(f"cannot load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PolicyTranslationError(
            f"importing {path} failed: {type(exc).__name__}: {exc}. The policy's "
            "own inference stack has to be importable in this interpreter; the "
            "project's carla-ubuntu20-plant2 Apptainer image provides both it "
            "and the CARLA API (apptainer shell --nvccli)") from exc
    return module


#: What this port can put in front of a policy. `state` is object-centric scene,
#: route and neighbours (`carla_obs`); `sensor` is whatever rig the policy
#: declares through `sensors()`, attached by `ego_driver._build_rig`.
OBSERVATION_SPACES = ("state", "sensor")


def load_policy(request: PolicyRequest, harness_root: Optional[str],
                repo_root: str) -> LoadedPolicy:
    """Build an external `ego_policy_v1` policy from its repository.

    Checks the declaration first, because saying no before importing torch is
    cheaper for everyone. This port serves `state` always and `sensor` when the
    policy asks for a rig (`ego_driver._build_rig` -> `carla_sensors`), so the
    only observation spaces refused here are the ones nothing can produce.
    """
    if request.interface != EGO_POLICY_INTERFACE:
        raise PolicyTranslationError(
            f"ego policy {request.name!r} declares interface "
            f"{request.interface!r}; this repository speaks "
            f"{EGO_POLICY_INTERFACE!r}")
    if request.observation_space not in OBSERVATION_SPACES:
        raise PolicyTranslationError(
            f"ego policy {request.name!r} needs {request.observation_space!r} "
            f"observations; this port provides {sorted(OBSERVATION_SPACES)}")

    entry_point = request.entry_point or "scenario_orchestration/policy.py"
    repository, how = resolve_repository(request, harness_root, repo_root)
    module = load_module(repository, entry_point, request.name or "policy")

    factory = getattr(module, FACTORY, None)
    if not callable(factory):
        public = ", ".join(sorted(n for n in dir(module)
                                  if not n.startswith("_"))) or "nothing public"
        raise PolicyTranslationError(
            f"{os.path.join(repository, entry_point)} exposes no callable "
            f"{FACTORY}(request); ego_policy_v1 requires "
            f"{FACTORY}(request: dict) -> policy. Found: {public}")
    try:
        policy = factory(_absolutize(request.to_dict(), harness_root))
    except Exception as exc:
        raise PolicyTranslationError(
            f"{FACTORY}() of ego policy {request.name!r} raised "
            f"{type(exc).__name__}: {exc}") from exc
    if policy is None:
        raise PolicyTranslationError(
            f"{FACTORY}() of ego policy {request.name!r} returned None")

    step = getattr(policy, STEP, None)
    if not callable(step):
        raise PolicyTranslationError(
            f"ego policy {request.name!r} built a "
            f"{type(policy).__name__} with no callable {STEP}(observation); "
            f"ego_policy_v1 requires {STEP}(observation: dict) -> action")

    return LoadedPolicy(policy=policy, name=request.name or "policy",
                        repository=repository, entry_point=entry_point,
                        resolution=how)


# --------------------------------------------------------------------------- #
# Which case applies
# --------------------------------------------------------------------------- #
def is_analytic_idm(request: PolicyRequest) -> bool:
    """Whether the request is one this repository realizes with drivev2's IDM.

    Decided from the declaration, not just the name: a policy that names itself
    `idm` but wants sensor input or emits trajectories is not the analytic IDM
    the port has, and quietly running IDM under its name would be worse than
    reporting that it cannot be run.
    """
    if request.interface != EGO_POLICY_INTERFACE:
        return False
    if request.observation_space != "state" or request.action_space != "control":
        return False
    name = (request.name or "").lower()
    if name in IDM_POLICIES:
        return True
    # A variant not in the list above is still the analytic IDM if it says it is
    # one and needs nothing beyond its own parameters. The entry point is no
    # help in deciding: every policy config declares one, including the analytic
    # ones, because a policy repository may also want to be loadable directly.
    return ("idm" in name and not request.requires and not request.checkpoint)


def describe_unsupported(request: PolicyRequest) -> str:
    """Why a policy cannot be run here, in terms a report can quote."""
    reasons: List[str] = []
    if request.interface != EGO_POLICY_INTERFACE:
        reasons.append(f"interface {request.interface!r} is not "
                       f"{EGO_POLICY_INTERFACE!r}")
    if request.observation_space not in OBSERVATION_SPACES:
        reasons.append(
            f"observation space {request.observation_space!r} is not provided by "
            f"this port, which serves {sorted(OBSERVATION_SPACES)}")
    if not request.entry_point:
        reasons.append(
            "the policy declares no entry point, so it can neither be realized "
            "analytically (it is not the IDM family) nor loaded as an "
            "ego_policy_v1 repository")
    return (f"ego policy {request.name!r} cannot be realized by this "
            f"repository: " + "; ".join(reasons or ["no reason determined"]))
