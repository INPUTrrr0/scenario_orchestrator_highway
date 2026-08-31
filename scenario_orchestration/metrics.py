"""The highway run report, in the harness's canonical metric vocabulary.

`contracts/metrics.py` defines seven canonical names. `carla_highway.runner`'s
report is much richer, so this module is a projection: it maps what the port
measured onto the canonical names, keeps everything else under
`method_metrics`, and says what the judgement is wherever the mapping is a
judgement rather than a lookup.

scenario_realized, and why this method's own answer is the weaker one
--------------------------------------------------------------------
The port grades each mode in the terms it was written in
(`carla_highway/README.md` section 6):

    cutin        the cut-in **merged** (not abandoned) and nothing hit the ego
    hard_brake   the ego got **past the slow lead** without a collision
    overtake     the ego got **past the blocker**, **returned to its lane**,
                 and hit nothing

Every one of those is a claim about *the ego*, not about the scenario. They are
`scenario_success` shaped -- did the whole episode come out the way the author
intended -- and using them as `scenario_realized` would repeat the defect
experiment 001 named on both other arms: a competent ego that declines the
conflict looks like an orchestration failure. So:

    scenario_realized   the orchestrator committed the conflict the mode is
                        about, whether or not the ego then survived it. For
                        `cutin` that is the merge being *performed* (the
                        orchestrator cast a holder and the holder completed its
                        lane change into the ego's lane); for `hard_brake` and
                        `overtake` the conflict is scripted rather than cast, so
                        it is that the conflict actor was placed and reached the
                        ego's road -- which the port answers as having recorded a
                        finite minimum gap to it.
    scenario_success    the port's own per-mode grade, unchanged and reported
                        under its own name.

This is still self-reported, and the harness does not have to believe any of it.
`metrics/` evaluates realization from the recorded trajectory
(`states.jsonl` + `scene.json`, written by `carla_highway/trace_recording.py`)
and writes its verdict to `metrics_v2.json` beside these numbers rather than
over them. What this module owes the harness is a number that is not *actively
misleading* in the table, and the port's own grade would have been.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from contract import ScenarioRequest

#: Success criteria this method knows how to evaluate, and what each reads.
CRITERIA = {
    "scenario_realized": lambda m: bool(m.get("scenario_realized")),
    "no_collision": lambda m: not bool(m.get("collision")),
    "collision": lambda m: bool(m.get("collision")),
    "near_collision": lambda m: bool(m.get("near_collision")),
}

#: Which background actor each mode's conflict is, when the scenario names it
#: rather than the orchestrator casting it. Both are authored as actor "1" in
#: the YAML and both carry a `role:` tag, which `runner._derive_roles` passes
#: through; the id is the fallback for a scenario that omits the tag.
CONFLICT_ROLE = {"hard_brake": "slow", "overtake": "blocker"}
CONFLICT_ID = "1"


def canonical(report: Mapping[str, Any], request: ScenarioRequest, ego: str
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`(canonical metrics, method metrics)` for one run."""
    mode = str(report.get("scenario") or "")
    grade = dict(report.get("grade") or {})
    orch = dict(report.get("orchestration") or {})
    hits = ego_collisions(report, ego)

    realized, realized_why = _realized(mode, report, orch)
    collision = bool(hits)
    gap = _min_hero_gap(mode, report, orch)

    metrics: Dict[str, Any] = {
        "scenario_realized": realized,
        "collision": collision,
        # A hazard delivered and survived. Independent of the port's own grade,
        # so a near miss it did not credit still shows up.
        "near_collision": bool(realized and not collision
                              and gap is not None and gap <= NEAR_GAP_M),
        "time_to_event": _time_to_event(mode, report, orch, hits),
        "scenario_duration": _num(report.get("duration")),
        # `HighwayClosedLoop.summary()` counts the orchestrator's own
        # interventions -- rebases, retimes, collision-yield replans -- which is
        # the same quantity the sibling port reports under this name.
        "intervention_cost": _num(orch.get("interventions")),
    }
    criteria_detail = {}
    success = _success(request, metrics, criteria_detail)
    if success is not None:
        metrics["scenario_success"] = success

    method = _method_metrics(report, request, criteria_detail, realized_why,
                             gap, hits)
    return metrics, method


#: Below this centre-to-centre gap the hero and the ego were close enough for
#: the episode to count as a near miss rather than as traffic. The port reports
#: bumper-adjusted minimum gaps per actor, so this is compared against those.
#: 4.0 m is the intersection port's own `interaction_gap_m`, reused so the two
#: arms' self-reported `near_collision` means the same thing.
NEAR_GAP_M = 4.0


def ego_collisions(report: Mapping[str, Any], ego: str) -> List[Dict[str, Any]]:
    """Realized CARLA collisions the ego was party to.

    `carla_highway.runner` stringifies its collision records, and already
    separates the ego's from the rest, so the ego's list is taken from the
    report rather than re-derived.
    """
    return [{"event": str(c)} for c in (report.get("ego_collisions") or [])]


def _realized(mode: str, report: Mapping[str, Any], orch: Mapping[str, Any]):
    """`(realized, why)` -- the orchestrator committed the mode's conflict."""
    if mode == "cutin":
        outcome = orch.get("outcome")
        performer = orch.get("holder")
        realized = outcome == "merged"
        return realized, (
            "cutin: the orchestrator's cast holder %s its lane change into the "
            "ego's lane (outcome=%r, performer=%r). Not the port's grade, which "
            "also requires the ego to have survived it."
            % ("completed" if realized else "did not complete", outcome,
               performer))
    role = CONFLICT_ROLE.get(mode)
    gap = _min_hero_gap(mode, report, orch)
    realized = gap is not None
    return realized, (
        "%s: the conflict is scripted rather than cast, so realization is that "
        "the %s actor was placed and came within measuring distance of the ego "
        "(min_gap=%s). Not the port's grade, which asks whether the ego got "
        "past it." % (mode or "?", role or CONFLICT_ID,
                      "None" if gap is None else "%.2f m" % gap))


def _conflict_actor(mode: str, report: Mapping[str, Any],
                    orch: Mapping[str, Any]) -> Optional[str]:
    if mode == "cutin":
        if orch.get("holder"):
            return str(orch["holder"])
        return None
    role = CONFLICT_ROLE.get(mode)
    # The roles the port derived, which live in its orchestration block rather
    # than at the top of the report (`HighwayClosedLoop.summary`).
    roles = orch.get("roles") or {}
    for aid, tag in roles.items():
        if role and str(tag) == role:
            return str(aid)
    return CONFLICT_ID if CONFLICT_ID in (report.get("interactions") or {}) \
        else None


def _min_hero_gap(mode: str, report: Mapping[str, Any],
                  orch: Mapping[str, Any]) -> Optional[float]:
    aid = _conflict_actor(mode, report, orch)
    if aid is None:
        return None
    entry = (report.get("interactions") or {}).get(aid) or {}
    return _num(entry.get("min_gap"))


def _time_to_event(mode: str, report: Mapping[str, Any],
                   orch: Mapping[str, Any], hits: List[Dict[str, Any]]):
    """When the conflict happened, in simulated seconds.

    The commit instant when the orchestrator recorded one -- that is the moment
    the scenario became a conflict -- and `None` otherwise. Deliberately not the
    first collision time: `time_to_event` is about the scenario's event, and a
    collision is the ego's outcome, which `collision` already reports.
    """
    if orch.get("t_commit") is not None:
        return _num(orch["t_commit"])
    return None


def _success(request: ScenarioRequest, metrics: Mapping[str, Any],
             detail: Dict[str, Any]) -> Optional[bool]:
    """The protocol's own success criteria, evaluated on the canonical metrics.

    The port's per-mode grade is reported separately under
    `method_metrics.port_grade`; this is the harness's declared conjunction, so
    a matrix that changes `success_criteria` changes this number and not the
    port's.
    """
    wanted = list(request.evaluation.success_criteria or [])
    if not wanted:
        return None
    ok = True
    for name in wanted:
        fn = CRITERIA.get(name)
        if fn is None:
            detail[name] = "not evaluable by this method"
            return None
        value = bool(fn(metrics))
        detail[name] = value
        ok = ok and value
    return ok


def _method_metrics(report: Mapping[str, Any], request: ScenarioRequest,
                    criteria_detail: Dict[str, Any], realized_why: str,
                    gap: Optional[float], hits: List[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """Everything the port measured that has no canonical name.

    Whole-report passthrough rather than a cherry-picked list. Experiment 003
    had to correct itself because `_method_metrics` in the sibling port selected
    the keys it forwarded and the swept parameter was not among them, so the
    value reached the run report and stopped there. Forwarding the report
    entire means a knob added later is recorded without anyone remembering to
    add it here.
    """
    out = {k: v for k, v in report.items()
           if k not in ("grade", "realized_collisions", "ego_collisions")}
    out.update({
        "port_grade": dict(report.get("grade") or {}),
        "scenario_realized_basis": realized_why,
        "min_hero_gap_m": gap,
        "near_gap_threshold_m": NEAR_GAP_M,
        "ego_collisions": hits,
        "realized_collisions": [str(c) for c in
                                (report.get("realized_collisions") or [])],
        "success_criteria": dict(criteria_detail),
    })
    return out


def _num(value):
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
