# `scenario_orchestration/` — the evaluation harness contract

This directory is how the
[evaluation harness](https://github.com/friedeggs/scenario_orchestration)
reaches this repository. It is the *only* interface: the harness never imports
anything else here, and nothing here imports the harness.

```bash
python scenario_orchestration/run.py \
    --scenario-request request.json \
    --policy-request  policy.json \
    --output-dir      results/raw/<experiment_id>
```

writes `method_result.json` (status + canonical metrics) and `trace.json` (what
the orchestrator decided, step by step) into the output directory.

| file | role |
|---|---|
| `run.py` | the entry point: translate the request, run, project, report |
| `contract.py` | this repository's own copy of the two document schemas |
| `metrics.py` | the report -> the harness's canonical metric vocabulary |
| `policies.py` | `policy.json` -> an ego: analytic, or loaded from its own repo |
| `capabilities.json` | what this method supports, reconciled by `validate` |

## Families

| family | mode | the target interaction |
|---|---|---|
| `cut_in` | `cutin` | a cast actor merges into the ego lane inside its safety envelope |
| `lane_change` | `hard_brake` | a slow lead in the ego lane, traffic squeezing the adjacent one |
| `overtake` | `overtake` | a stopped blocker, oncoming traffic constraining the only way past |

`lane_change` runs the mode this repository calls `hard_brake`: the same
situation under this repository's own name for it, satisfying all three of the
family's declared preconditions. The signalised-junction families belong to a
different method — `carla_port/` here carries only the script-agnostic CARLA
mechanics `carla_highway` imports, not the junction port.

## Ego policies

**Analytic** — realized natively by binding the request onto
`carla_highway/highway_ego.py`'s own constants, which it reads as module
globals on every call:

| policy | ego |
|---|---|
| `idm`, `idm_assertive`, `idm_conservative` | IDM longitudinal, **lane-keeping** |
| `mobil` | IDM longitudinal + **MOBIL** lateral |

Those two are the same code with one flag between them, and they are separate
policies on purpose: the lane-keeping one is the baseline that makes
`lane_change` and `overtake` unsolvable, which is the evidence those families
need a lateral decision at all.

**Learned** — any other `ego_policy_v1` policy, loaded from its own repository
and driven through `carla_port.ego_driver.PolicyEgoDriver`. Note that the
method steps the policy **in process**, so the interpreter running `run.py`
needs both CARLA's Python API and that policy's inference stack. Point the
harness at one with `SCENARIO_ORCHESTRATION_PYTHON`, or give the method a
`launcher` in its algorithm config.

## `scenario_realized` is not `scenario_success`

`scenario_realized` answers **did the scenario happen**, not did the ego cope.
A cut-in that merges into an ego which then hits it is realized *and* a
collision; an ego that never got past a blocker in a genuinely constrained
overtake window is a realized scenario the policy failed. The port's own
pass/fail verdict asks the second question and is reported separately as
`method_metrics.port_verdict`.

One consequence worth knowing: the cut-in's interaction test needs a *lateral*
gate as well as a distance one. An actor spawned one lane over is already
inside a 4 m band on distance alone, so a band tested on distance reported the
target interaction at t=0 — before any merge had begun.
