#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless tests for the v2 directive layer."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import directives as dv


def famres(state, prm):
    return dv.evaluate_family(dv.recognize(state, prm), prm)


def test_recognition():
    prm = dv.Params()
    st = dv.mkstate("rec", [("0", 1.75, -30, 90, 12),      # SE approach
                            ("1", 30, 1.75, 180, 10),      # EN approach
                            ("2", -8, 1.75, 180, 12),      # WN exit
                            ("3", 3.0, -2.6, 120, 8)])     # inside box (SE right-ish?)
    a = dv.recognize(st, prm).actors
    assert a["0"].region == "approach" and a["0"].leg == "SE"
    assert a["1"].region == "approach" and a["1"].leg == "EN" and not a["1"].committed
    assert a["2"].region == "exit" and a["2"].leg == "WN" and a["2"].committed
    assert a["3"].region == "intersection" and a["3"].committed
    # mid-left-turn recognition: point on SE-left arc
    p = dv.route_path(st.map, "SE", "left", prm)
    s = dv.stopline_s(st.map, p, "SE") + 3.0
    (x, y), h = p.pos_at(s), p.heading_at(s)
    st2 = dv.mkstate("turn", [("0", 1.75, -30, 90, 12), ("1", x, y, h, 8)])
    a1 = dv.recognize(st2, prm).actors["1"]
    assert a1.region == "intersection" and (a1.leg, a1.turn) == ("SE", "left"), \
        (a1.leg, a1.turn)


def test_profile():
    p = dv.Profile.retime(4.0, 10.0, 2.0)   # ramp 3 s, then plateau
    assert abs(p.v_at(1.5) - 7.0) < 1e-9
    assert abs(p.s_at(3.0) - 21.0) < 1e-9
    assert abs(p.arrival(21.0) - 3.0) < 1e-6
    assert abs(p.arrival(31.0) - 4.0) < 1e-6
    stop = dv.Profile.retime(10.0, 0.0, 2.0)  # brakes to standstill after 25 m
    assert stop.arrival(24.0) is not None and stop.arrival(26.0) is None
    jump = dv.Profile.retime(4.0, 10.0, None)
    assert abs(jump.arrival(20.0) - 2.0) < 1e-9


def test_s1_all_true():
    prm = dv.Params()
    res = famres(dv.demo_states()[0], prm)
    assert res.ok and res.hero == "1"
    assert abs(res.t_star - 2.65) < 0.15, res.t_star
    # dense verification: bodies really overlap
    astate = dv.recognize(dv.demo_states()[0], prm)
    assert dv.verify_collision(astate, prm, {}, "1", res.t_star) is not None


def test_s2_retime():
    prm = dv.Params()
    st = dv.demo_states()[1]
    res = famres(st, prm)
    assert not res.d1.value and res.hero == "1"
    rr = dv.repair(dv.recognize(st, prm), prm)
    assert rr.feasible and len(rr.interventions) == 1
    iv = rr.interventions[0]
    assert iv.kind == "retime" and iv.actor == "1"
    assert abs(iv.value - 10.68) < 0.4, iv.value
    assert rr.final.ok


def test_s3_witness_switch():
    prm = dv.Params()
    st = dv.demo_states()[2]
    res = famres(st, prm)
    assert not res.d1.value
    rr = dv.repair(dv.recognize(st, prm), prm)
    assert rr.feasible and rr.final.hero == "2", rr.final.hero
    assert all(iv.actor == "2" for iv in rr.interventions)


def test_s4_interference():
    prm = dv.Params()
    st = dv.demo_states()[3]
    res = famres(st, prm)
    assert res.d1.value and res.hero == "1"
    assert not res.d3.value and res.d3.witness.get("w") == "2"
    rr = dv.repair(dv.recognize(st, prm), prm)
    assert rr.feasible and rr.final.ok
    assert any(iv.actor == "2" for iv in rr.interventions)


def test_s5_crossing_interferer():
    prm = dv.Params()
    st = dv.demo_states()[4]
    res = famres(st, prm)
    assert res.d1.value and not res.d3.value
    rr = dv.repair(dv.recognize(st, prm), prm)
    assert rr.feasible and rr.final.ok


def test_s6_infeasible():
    prm = dv.Params()
    rr = dv.repair(dv.recognize(dv.demo_states()[5], prm), prm)
    assert not rr.feasible and "red" in rr.reason


def test_ramp_committed_overshoot():
    """Jump model: hero can always delay -> collidable at t=0.
    Ramp model: hero too fast/close to stop -> committed -> not collidable."""
    st = dv.mkstate("committed", [("0", 1.75, -30, 90, 12), ("1", 8.0, 1.75, 180, 18.0)])
    jump, ramp = dv.Params(ramp=None), dv.Params(ramp=3.0)
    cj = dv.Ctx(dv.recognize(st, jump), dv.Evolution(dv.recognize(st, jump), jump),
                0.0, jump, {"ego": "0", "hero": "1"})
    cr = dv.Ctx(dv.recognize(st, ramp), dv.Evolution(dv.recognize(st, ramp), ramp),
                0.0, ramp, {"ego": "0", "hero": "1"})
    okj, _ = dv.collidable(cj, "1")
    okr, _ = dv.collidable(cr, "1")
    assert okj and not okr, (okj, okr)


def test_signals_explicit():
    prm = dv.Params()
    st = dv.demo_states()[0]
    st.signals = {"N": "green", "S": "green", "E": "green", "W": "green"}
    res = famres(st, prm)
    assert not res.d1.value  # nobody runs a red if every arm is green


def test_adapter():
    prm = dv.Params()
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "v1", "scenarios", "scenario_v1.yaml")
    st = dv.state_from_scenario(path, 1.0, None)
    assert len(st.actors) >= 2
    res = famres(st, prm)  # must evaluate without error
    assert res.d1 is not None and res.d2 is not None and res.d3 is not None


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
