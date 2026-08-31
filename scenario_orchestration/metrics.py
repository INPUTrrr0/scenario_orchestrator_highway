#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scenario_orchestration/metrics.py — this method's report -> canonical metrics.

The harness owns the family definition and the evaluation protocol
(`configs/scenario/<family>.yaml`); this repository owns its native realization
and the projection between them. `carla_highway/runner.py` records only facts —
per-actor closest approach and when it happened, which lane the ego occupied
when — and every judgement about whether those facts constitute a family's
target interaction is made here, from the sentence the harness wrote in
`target_interaction`.

The distinction the whole file turns on:

    `scenario_realized` answers "did the scenario happen".
    It does NOT answer "did the ego cope".

A cut-in that merges into an ego which then hits it is realized AND a
collision. An ego that never got past a blocker in a genuinely constrained
overtake window is a realized scenario the policy failed. Keeping those apart
is the point of having both metrics; the port's own pass/fail verdict asks the
second question and is reported separately as `method_metrics.port_verdict`.
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

#: The families this repository realizes.
FAMILIES = ("cut_in", "lane_change", "overtake")

#: How near traffic in the target lane has to come for that lane to count as
#: "occupied" in the lane_change family. Wider than the interaction band, which
#: measures a conflict: this measures a lane that is not simply free, and a car
#: two seconds away at 12 m/s already denies a comfortable merge.
TARGET_LANE_OCCUPIED_M = 25.0


def ego_collisions(report: Mapping[str, Any], ego: str) -> List[Dict[str, Any]]:
    """Realized CARLA collisions the ego was party to.

    Both directions count: the sensor that fires is the one on the actor whose
    body registered the hit, and being rammed is still a collision for the ego.

    Everything else is reported but never graded. Background actors collide for
    a mundane reason — the scripts run far longer than the fitted straight, so
    an actor that outlives the scenario drives off the end of the fit and into
    the scenery, long after the part being measured.
    """
    out = []
    for c in report.get("realized_collisions") or []:
        if not isinstance(c, Mapping):
            continue
        if c.get("actor") == ego or c.get("other") == ego:
            out.append(dict(c))
    return out


# --------------------------------------------------------------------------- #
# Turning each family's `target_interaction` sentence into a predicate
# --------------------------------------------------------------------------- #
def _same_lane_leads(interactions: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """Actors that began ahead of the ego, in its lane, travelling with it."""
    out = [(aid, it) for aid, it in interactions.items()
           if it.get("same_lane_at_start") and not it.get("oncoming")
           and float(it.get("start_along") or 0.0) > 0.0]
    out.sort(key=lambda kv: float(kv[1].get("start_along") or 0.0))
    return out


def _other_lane_actors(interactions: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    return [(aid, it) for aid, it in interactions.items()
            if not it.get("same_lane_at_start") and not it.get("oncoming")]


def _oncoming_actors(interactions: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    return [(aid, it) for aid, it in interactions.items() if it.get("oncoming")]


def _used_opposing_lane(report: Mapping[str, Any]) -> Optional[float]:
    """When the ego first entered a lane that runs against it, if it did."""
    opposing = {int(l["index"]) for l in (report.get("lanes") or [])
                if l.get("oncoming")}
    for entry in report.get("ego_lane_track") or []:
        try:
            t, lane = float(entry[0]), int(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lane in opposing:
            return t
    return None


def target_interaction(family: str, report: Mapping[str, Any]
                       ) -> Tuple[Optional[str], bool, Optional[float], Dict[str, Any]]:
    """(hero, delivered, first_time, detail) for one family."""
    interactions = dict(report.get("interactions") or {})
    orchestration = dict(report.get("orchestration") or {})
    detail: Dict[str, Any] = {"family": family}

    if family == "cut_in":
        # "an adjacent actor cuts into the ego lane inside the ego safety
        # envelope" -- both halves are required. The merge has to complete, and
        # it has to complete close AND in the ego's lane: `t_conflict` carries
        # the lateral gate, where `t_close` would fire at spawn for any actor
        # sitting one lane over.
        hero = orchestration.get("holder")
        outcome = orchestration.get("outcome")
        it = interactions.get(str(hero)) if hero is not None else None
        merged = outcome == "merged"
        inside = bool(it and it.get("t_conflict") is not None)
        detail.update({"holder": hero, "outcome": outcome, "merged": merged,
                       "entered_ego_lane_inside_band": inside,
                       "t_close_ignoring_lane": (it or {}).get("t_close")})
        return hero, (merged and inside), (it or {}).get("t_conflict"), detail

    if family == "lane_change":
        # "the ego vehicle is required to change lane into occupied traffic" --
        # the ego must actually be forced (its own lane blocked inside the
        # band) AND the lane it would move into must be occupied. Whether the
        # ego completed the change is its performance, not the realization.
        leads = _same_lane_leads(interactions)
        hero, lead = (leads[0] if leads else (None, {}))
        blocked_at = lead.get("t_conflict") if lead else None
        contenders = [
            (aid, it) for aid, it in _other_lane_actors(interactions)
            if it.get("min_gap") is not None
            and float(it["min_gap"]) <= TARGET_LANE_OCCUPIED_M
        ]
        detail.update({"blocking_lead": hero, "blocked_at": blocked_at,
                       "target_lane_occupied_by": [aid for aid, _ in contenders],
                       "target_lane_window_m": TARGET_LANE_OCCUPIED_M})
        return hero, bool(blocked_at is not None and contenders), blocked_at, detail

    if family == "overtake":
        # "the ego overtake manoeuvre interacts with oncoming traffic" -- the
        # manoeuvre has to be forced (a lead in the ego's lane inside the band,
        # or the ego actually out in the opposing lane) and oncoming traffic
        # has to constrain it. WAITING for the oncoming car counts: it is the
        # oncoming car that made the ego wait.
        leads = _same_lane_leads(interactions)
        blk, lead = (leads[0] if leads else (None, {}))
        opposed_at = _used_opposing_lane(report)
        forced = bool((lead.get("t_conflict") if lead else None) is not None
                      or opposed_at is not None)
        oncoming = [(aid, it) for aid, it in _oncoming_actors(interactions)
                    if it.get("t_close") is not None]
        hero, first = ((oncoming[0][0], oncoming[0][1].get("t_close"))
                       if oncoming else (None, None))
        detail.update({"blocking_lead": blk,
                       "blocked_at": lead.get("t_conflict") if lead else None,
                       "entered_opposing_lane_at": opposed_at,
                       "constrained_by": [aid for aid, _ in oncoming]})
        return hero, bool(forced and oncoming), first, detail

    detail["note"] = f"family {family!r} is not realized by this repository"
    return None, False, None, detail


def _success(request: ScenarioRequest, metrics: Mapping[str, Any]
             ) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Evaluate the family's own declared success criteria.

    The protocol comes from the harness, so the criteria are evaluated as
    declared rather than hard-coded. A criterion this method cannot read is
    reported by name instead of being quietly treated as satisfied -- a success
    flag computed from half a protocol would be worse than none.
    """
    declared = list(request.evaluation.success_criteria)
    if not declared:
        return None, {"declared": [], "note": "the family declares no criteria"}
    evaluated: Dict[str, bool] = {}
    unknown: List[str] = []
    for name in declared:
        predicate = CRITERIA.get(str(name))
        if predicate is None:
            unknown.append(str(name))
            continue
        evaluated[str(name)] = bool(predicate(metrics))
    detail: Dict[str, Any] = {"declared": declared, "evaluated": evaluated}
    if unknown:
        detail["unevaluated"] = unknown
        detail["note"] = ("this method cannot evaluate " + ", ".join(unknown)
                          + "; scenario_success is withheld rather than guessed")
        return None, detail
    return all(evaluated.values()), detail


def canonical(report: Mapping[str, Any], request: ScenarioRequest, ego: str
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(canonical metrics, method-specific metrics) for one run.

    Only metrics that were actually measured are reported. A metric with no
    value -- `time_to_event` when no event happened -- is left out rather than
    reported as zero, because zero is a time and "did not happen" is not.
    """
    family = str(request.scenario_family)
    hits = ego_collisions(report, ego)
    collision = bool(hits)
    hero, delivered, first_time, detail = target_interaction(family, report)

    metrics: Dict[str, Any] = {
        "scenario_realized": delivered,
        "collision": collision,
        "near_collision": delivered and not collision,
        "intervention_cost": float(report.get("intervention_cost") or 0.0),
        "scenario_duration": float(report.get("sim_time") or 0.0),
    }
    if first_time is None and hits:
        first_time = min(float(h["sim_time"]) for h in hits)
    if first_time is not None:
        metrics["time_to_event"] = float(first_time)

    success, criteria_detail = _success(request, metrics)
    if success is not None:
        metrics["scenario_success"] = success

    grade = dict(report.get("grade") or {})
    method_metrics: Dict[str, Any] = {
        "target_interaction": detail,
        "hero": hero,
        "success_criteria": criteria_detail,
        # The port's own verdict, which asks whether the EGO coped -- a
        # different question from whether the scenario happened.
        "port_verdict": {"success": grade.get("success"),
                         "checks": grade.get("checks"),
                         "detail": grade.get("detail")},
        "orchestration": report.get("orchestration"),
        "ego": report.get("ego"),
        "frame": report.get("frame"),
        "interactions": report.get("interactions"),
        "ego_collisions": hits,
        "collisions": report.get("realized_collisions"),
        "interaction_gap_m": report.get("interaction_gap_m"),
        "lane_conflict_m": report.get("lane_conflict_m"),
        "notes": report.get("notes"),
    }
    for key in ("video", "trajectories", "body_extents"):
        if report.get(key):
            method_metrics[key] = report[key]
    return metrics, method_metrics
