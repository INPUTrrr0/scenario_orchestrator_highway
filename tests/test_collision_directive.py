#!/usr/bin/env python3
"""Collision-avoidance directive for the cut-in orchestrator."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scenario_editor as se
import cutin_orchestrator as co


def _actor(aid, x, y, hd=90.0, cutin=None, block=None, cruise=12.0):
    a = se.Actor(id=str(aid), color=(200, 80, 80), length=4.5, width=2.0,
                 start=(x, y, hd), cruise=cruise, cutin=cutin, block=block,
                 maneuvers=se.cruise_plan((x, y, hd), cruise))
    return a


def test_overlap_detected():
    a = _actor("1", 0.0, 0.0)
    b = _actor("2", 0.0, 2.0)   # overlapping longitudinally
    hits = co.predicted_collisions(
        [a, b], lambda actor, t: actor.start, horizon=0.2)
    assert hits and hits[0][:2] == ("1", "2"), hits


def test_separated_not_detected():
    a = _actor("1", 0.0, 0.0)
    b = _actor("2", 0.0, 20.0)
    hits = co.predicted_collisions(
        [a, b], lambda actor, t: actor.start, horizon=0.2)
    assert hits == [], hits


def test_owner_outranks_nominal():
    owner = _actor("2", -3.5, 0.0, cutin={"t": 3.0, "along": 6.0, "lat": 0.0})
    other = _actor("1", 0.0, 8.0)
    scores = {"1": {co.ROLE_CUTIN: 0.9, co.ROLE_BLOCK: 0.9},
              "2": {co.ROLE_CUTIN: 0.1, co.ROLE_BLOCK: 0.0}}
    assert co.action_priority(owner, scores) > co.action_priority(other, scores)


def test_probability_breaks_owner_tie():
    cut = _actor("2", -3.5, 0.0, cutin={"t": 3.0})
    blk = _actor("1", 0.0, 8.0, block={"along": 0.0})
    scores = {"1": {co.ROLE_CUTIN: 0.03, co.ROLE_BLOCK: 0.01},
              "2": {co.ROLE_CUTIN: 0.45, co.ROLE_BLOCK: 0.03}}
    assert co.action_priority(cut, scores) > co.action_priority(blk, scores)


def test_ahead_interferer_is_sped_up():
    priv = _actor("2", -3.5, -50.0, cutin={"t": 3.0})
    inter = _actor("1", 0.0, -30.0, block={"along": 0.0})
    v = co.yield_speed(priv, inter, priv.start, inter.start, v_priv=12.0, t_hit=1.0)
    assert v > 12.0, v


def test_seed4_yields_actor1_to_cutin_owner():
    """Reproduce experiment --seed 4: actor 2 cuts in, actor 1 occupies the
    target lane ahead and would close the gap by slowing (block hold)."""
    ego = se.Actor(id="0", color=(90, 190, 110), length=4.5, width=2.0,
                   start=(0.0, -50.0, 90.0),
                   maneuvers=[se.Maneuver(type="go_straight", duration=5.0,
                                          intercept=12.0)])
    a1 = _actor("1", 0.0, -30.5, block={"along": 0.0, "t": 4.0,
                                        "duration": 4.0, "lat": -3.5},
                cruise=11.5)
    a2 = _actor("2", -3.5, -50.7,
                cutin={"t": 3.0, "along": 6.0, "lat": 0.0, "lc_duration": 2.0},
                cruise=12.8)
    # closed-loop-ish plans: a2 lane-changes into ego's lane, a1 drops back
    a2.maneuvers = se.solve_closed_loop_cutin(
        a2.start, 0.0, -44.0, 3.0, 12.0, lc_duration=2.0, tail=8.0)
    a1.maneuvers = se.cruise_plan(a1.start, 7.0)   # block dropping back
    sc = se.Scenario(map=se.MapConfig(kind="straight", num_lanes=3,
                                      lane_width=3.5, length=120),
                     actors=[ego, a1, a2])
    sc.simulate()
    scores = {"1": {co.ROLE_CUTIN: 0.03, co.ROLE_BLOCK: 0.01},
              "2": {co.ROLE_CUTIN: 0.45, co.ROLE_BLOCK: 0.03}}
    yields = co.resolve_actor_collisions(
        sc.actors, lambda a, t: a.pose_at_time(t), scores)
    assert yields, "expected a predicted collision on seed-4 geometry"
    inter, v, priv, t_hit = yields[0]
    assert priv.id == "2" and inter.id == "1", (priv.id, inter.id)
    assert v > co.actor_plan_speed(a1), (v, co.actor_plan_speed(a1))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                fails += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("ALL TESTS PASSED" if not fails else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)
