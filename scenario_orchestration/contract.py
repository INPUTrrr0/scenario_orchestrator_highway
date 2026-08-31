"""The serialized `scenario_orchestration` contract, mirrored locally.

`third_party/README.md` and `DESIGN.md` section 5 make the integration boundary
a subprocess and two JSON documents. The documents are the contract — not a
shared Python class — so they are re-declared here rather than imported:

    request.json   -> ScenarioRequest
    policy.json    -> PolicyRequest
    method_result.json <- MethodResult

Nothing in this package imports `scenario_orchestration`, which is what lets
this repository pin its own Python, CARLA and CUDA versions. Unknown keys are
ignored so the harness can grow either document without breaking the runner, and
missing optional keys fall back to the documented defaults.

The statuses a method may report itself are success / failure / timeout / error.
`incompatible` is the harness's verdict to make, never ours: a method that
declines a cell reports `failure` with a reason, and the harness decides what
that means for the matrix.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

SCENARIO_SCHEMA_VERSION = "1.0.0"
POLICY_SCHEMA_VERSION = "1.0.0"

#: The standardized ego-policy interface (`contracts/policy.py`).
EGO_POLICY_INTERFACE = "ego_policy_v1"

#: What the harness reads back out of `--output-dir`.
METHOD_RESULT_FILE = "method_result.json"
TRACE_FILE = "trace.json"

#: Statuses a method is allowed to report for itself.
METHOD_STATUSES = ("success", "failure", "timeout", "error")

#: Canonical metric names (`contracts/metrics.py`). Anything else a method
#: reports is preserved by the harness under `method_metrics`, so this list is
#: only here to keep our own naming honest.
CANONICAL_METRICS = (
    "scenario_success",
    "scenario_realized",
    "collision",
    "near_collision",
    "time_to_event",
    "intervention_cost",
    "scenario_duration",
)


def _known(cls, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------- #
# request.json
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationProtocol:
    """How the family is scored, independent of who executes it."""

    horizon_s: float = 20.0
    target_event: str = ""
    success_criteria: List[str] = field(default_factory=list)
    tick_rate_hz: float = 10.0

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "EvaluationProtocol":
        data = _known(cls, dict(payload or {}))
        data["success_criteria"] = list(data.get("success_criteria") or [])
        data["horizon_s"] = float(data.get("horizon_s") or 20.0)
        data["tick_rate_hz"] = float(data.get("tick_rate_hz") or 10.0)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"horizon_s": self.horizon_s, "target_event": self.target_event,
                "success_criteria": list(self.success_criteria),
                "tick_rate_hz": self.tick_rate_hz}


@dataclass(frozen=True)
class ScenarioImplementation:
    """This method's own native realization of the family (DESIGN.md section 7).

    The harness always sends this block, falling back to the family name as the
    native id, so the repository side never has to special-case its absence.
    """

    family: str = ""
    method: str = ""
    native_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ScenarioImplementation":
        data = _known(cls, dict(payload or {}))
        data["parameters"] = dict(data.get("parameters") or {})
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "method": self.method,
                "native_id": self.native_id, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class ScenarioRequest:
    """`request.json`: the whole scenario-side contract."""

    experiment_id: str = ""
    scenario_family: str = ""
    semantic_id: str = ""
    algorithm: str = ""
    seed: int = 0
    schema_version: str = SCENARIO_SCHEMA_VERSION
    evaluation: EvaluationProtocol = field(default_factory=EvaluationProtocol)
    implementation: Optional[ScenarioImplementation] = None
    metrics: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ScenarioRequest":
        data = _known(cls, dict(payload or {}))
        data["evaluation"] = EvaluationProtocol.from_dict(data.get("evaluation"))
        implementation = data.get("implementation")
        data["implementation"] = (ScenarioImplementation.from_dict(implementation)
                                 if implementation else None)
        data["metrics"] = list(data.get("metrics") or [])
        data["parameters"] = dict(data.get("parameters") or {})
        data["seed"] = int(data.get("seed") or 0)
        data["output_dir"] = (None if data.get("output_dir") is None
                              else str(data["output_dir"]))
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ScenarioRequest":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "scenario_family": self.scenario_family,
                "semantic_id": self.semantic_id,
                "algorithm": self.algorithm,
                "seed": self.seed,
                "evaluation": self.evaluation.to_dict(),
                "implementation": (self.implementation.to_dict()
                                   if self.implementation else None),
                "metrics": list(self.metrics),
                "parameters": dict(self.parameters),
                "output_dir": self.output_dir}

    # -- convenience -------------------------------------------------------
    @property
    def native_id(self) -> str:
        """The method's own name for this scenario, or the family name."""
        if self.implementation and self.implementation.native_id:
            return self.implementation.native_id
        return self.scenario_family

    def setting(self, key: str, default: Any = None) -> Any:
        """One knob, wherever the harness put it.

        Two blocks can carry method configuration, and both are legitimate:
        `implementation.parameters` is per-family (`scenarios/<family>/
        implementations.yaml`) and `parameters` is per-method
        (`configs/algorithm/orchestration.yaml`). The per-family block is the
        more specific of the two, so it wins.
        """
        implementation = (self.implementation.parameters
                          if self.implementation else {})
        if key in implementation:
            return implementation[key]
        if key in self.parameters:
            return self.parameters[key]
        return default

    def settings(self) -> Dict[str, Any]:
        """Both configuration blocks merged, per-family winning."""
        merged = dict(self.parameters)
        if self.implementation:
            merged.update(self.implementation.parameters)
        return merged


# --------------------------------------------------------------------------- #
# policy.json
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyRequest:
    """`policy.json`: the ego policy in method-agnostic terms.

    Translating this into the port's own policy plumbing is this repository's
    job — that is what keeps integration cost at M + P rather than M x P
    (DESIGN.md section 6). See `policies.py`.
    """

    name: str = ""
    interface: str = EGO_POLICY_INTERFACE
    implementation: str = ""
    observation_space: str = "state"
    action_space: str = "control"
    seed: int = 0
    schema_version: str = POLICY_SCHEMA_VERSION
    experiment_id: str = ""
    repository: Optional[str] = None
    entry_point: Optional[str] = None
    checkpoint: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "PolicyRequest":
        data = _known(cls, dict(payload or {}))
        data["parameters"] = dict(data.get("parameters") or {})
        data["requires"] = list(data.get("requires") or [])
        data["seed"] = int(data.get("seed") or 0)
        data.setdefault("interface", EGO_POLICY_INTERFACE)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "PolicyRequest":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "name": self.name,
                "interface": self.interface,
                "implementation": self.implementation,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "seed": self.seed,
                "repository": self.repository,
                "entry_point": self.entry_point,
                "checkpoint": self.checkpoint,
                "parameters": dict(self.parameters),
                "requires": list(self.requires)}


# --------------------------------------------------------------------------- #
# method_result.json
# --------------------------------------------------------------------------- #
@dataclass
class MethodResult:
    """What this repository writes into `--output-dir`.

    `metrics` may mix canonical names with anything else; the harness splits
    them and preserves the rest under `method_metrics`. Everything outside the
    reserved keys is preserved too, but naming it explicitly is clearer than
    relying on that.
    """

    status: str = "success"
    metrics: Dict[str, Any] = field(default_factory=dict)
    method_metrics: Dict[str, Any] = field(default_factory=dict)
    trace_path: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in METHOD_STATUSES:
            raise ValueError(
                f"status {self.status!r} is not one of {METHOD_STATUSES}; "
                "'incompatible' is the harness's verdict, never a method's")

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status,
                "metrics": dict(self.metrics),
                "method_metrics": dict(self.method_metrics),
                "trace_path": self.trace_path,
                "reason": self.reason}

    def write(self, output_dir: str) -> str:
        return write_json(os.path.join(output_dir, METHOD_RESULT_FILE),
                          self.to_dict())


def failure(reason: str, status: str = "failure", **method_metrics: Any
            ) -> MethodResult:
    """A reported failure with an explanation the harness can quote."""
    return MethodResult(status=status, reason=reason,
                        method_metrics=dict(method_metrics))
