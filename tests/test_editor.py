"""Headless tests for the scenario editor kernel (no display needed).

Run from anywhere: python3 tests/test_editor.py
"""
import importlib.util
import math
import os
import sys

SE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SE_DIR not in sys.path:
    sys.path.insert(0, SE_DIR)
spec = importlib.util.spec_from_file_location("se", os.path.join(SE_DIR, "scenario_editor.py"))
se = importlib.util.module_from_spec(spec)
sys.modules["se"] = se
spec.loader.exec_module(se)

SAMPLE = os.path.join(SE_DIR, "scenarios", "scenario_v1.yaml")
failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


def min_dist(sc):
    n = min(len(a.traj) for a in sc.actors)
    md, mt = 1e9, 0.0
    for k in range(n):
        x0, y0, _ = sc.actors[0].traj[k]
        x1, y1, _ = sc.actors[1].traj[k]
        d = math.hypot(x1 - x0, y1 - y0)
        if d < md:
            md, mt = d, k * se.DT
    return md, mt


# ---- 1. sample scenario collides ----
sc = se.load_scenario(SAMPLE)
md, mt = min_dist(sc)
check("sample collision", md < 1.0, f"min center dist {md:.3f} m at t={mt:.2f}s")

# ---- 2. collision holds for ANY start positions along the legs ----
ok = True
for ego_y in (-38.0, -25.0, -15.0, -8.0):
    for hero_x in (38.0, 28.0, 12.0, 6.0):
        s2 = se.load_scenario(SAMPLE)
        s2.actors[0].start = (1.75, ego_y, 90.0)
        s2.actors[1].start = (hero_x, 1.75, 180.0)
        s2.simulate()
        d, t = min_dist(s2)
        if d >= 1.0:
            ok = False
            print(f"   miss: ego_y={ego_y} hero_x={hero_x} -> {d:.3f} m at {t:.2f}s")
check("collision independent of starts", ok, "16 start combinations")

# ---- 3. v0 parity: scripted actor == closed-form chaining ----
m1 = se.Maneuver(type="go_straight", duration=2.0, intercept=10.0)
m2 = se.Maneuver(type="turn_left", duration=2.0, radius=6.0, angle=90.0)
m3 = se.Maneuver(type="decelerate", duration=2.0, intercept=m2.exit_speed(), slope=-2.0)
a = se.Actor(id="9", color=(1, 2, 3), length=4.5, width=2.0,
             start=(0.0, -20.0, 90.0), maneuvers=[m1, m2, m3])
s3 = se.Scenario(map=se.MapConfig(), actors=[a])
s3.simulate()
p0 = a.start
e1 = m1.end_pose(p0)
e2 = m2.end_pose(e1)
ok = True
for t, want in [(1.0, m1.pose_at(p0, 1.0)), (3.0, m2.pose_at(e1, 1.0)),
                (5.5, m3.pose_at(e2, 1.5)), (6.0, m3.pose_at(e2, 2.0))]:
    got = a.pose_at_time(t)
    err = max(abs(g - w) for g, w in zip(got, want))
    if err > 1e-9:
        ok = False
        print(f"   parity err at t={t}: {err:.2e}  got={got} want={want}")
check("v0 closed-form parity", ok)

# ---- 4. yaw output: pursuit via bearing() closes distance ----
target = se.Actor(id="0", color=(0, 0, 0), length=4.5, width=2.0,
                  start=(0.0, 0.0, 90.0),
                  maneuvers=[se.Maneuver(type="go_straight", duration=10.0,
                                         intercept=3.0)])
fn = se.Function(duration=10.0, nodes=[
    se.Node(id="n0", kind="obs", source="self", field="pos"),
    se.Node(id="n1", kind="obs", source="0", field="pos"),
    se.Node(id="n2", kind="op", op="bearing", inputs=["n0", "n1"]),
    se.Node(id="n3", kind="const", value=8.0),
], out_speed="n3", out_yaw="n2")
chaser = se.Actor(id="1", color=(0, 0, 0), length=4.5, width=2.0,
                  start=(30.0, 25.0, 180.0), maneuvers=[fn])
s4 = se.Scenario(map=se.MapConfig(), actors=[target, chaser])
s4.simulate()
d0 = math.hypot(30.0, 25.0)
dT = math.hypot(s4.actors[1].traj[-1][0] - s4.actors[0].traj[-1][0],
                s4.actors[1].traj[-1][1] - s4.actors[0].traj[-1][1])
check("yaw pursuit closes distance", dT < 2.0 < d0, f"{d0:.1f} m -> {dT:.2f} m")

# ---- 5. unbound / incomplete graph holds entry state ----
fn2 = se.Function(duration=4.0, nodes=[])
a5 = se.Actor(id="2", color=(0, 0, 0), length=4.5, width=2.0,
              start=(5.0, 5.0, 45.0),
              maneuvers=[se.Maneuver(type="go_straight", duration=2.0,
                                     intercept=6.0), fn2])
s5 = se.Scenario(map=se.MapConfig(), actors=[a5])
s5.simulate()
p2, p6 = a5.pose_at_time(2.0), a5.pose_at_time(6.0)
travelled = math.hypot(p6[0] - p2[0], p6[1] - p2[1])
check("unbound function holds entry speed+heading",
      abs(travelled - 6.0 * 4.0) < 0.5 and abs(p6[2] - 45.0) < 1e-6,
      f"travelled {travelled:.2f} m (expect ~24), heading {p6[2]:.1f}")

# ---- 6. graph analysis helpers ----
f6 = se.Function(duration=1.0, nodes=[
    se.Node(id="a", kind="const", value=1.0),
    se.Node(id="b", kind="obs", source="self", field="pos"),
    se.Node(id="c", kind="op", op="if", inputs=["a", "b"]),   # incomplete
])
check("type: op with T resolves from inputs",
      se.node_value_type(f6, "c") == se.POINT)
check("incomplete op detected", not se.op_is_complete(f6, f6.node("c")))
check("slot 3 of if accepts point", se.slot_accepts(f6, f6.node("c"), 2, se.POINT))
check("slot 3 of if rejects scalar", not se.slot_accepts(f6, f6.node("c"), 2, se.SCALAR))
check("compatible ops (S,S)",
      set(se.compatible_ops(f6, [se.SCALAR, se.SCALAR])) >=
      {"add", "sub", "mul", "div", "min", "max", "lt", "gt", "if"})
check("compatible ops (P,P) includes dist/bearing/midpoint",
      {"dist", "bearing", "midpoint"} <=
      set(se.compatible_ops(f6, [se.POINT, se.POINT])))
check("cycle detection",
      se.would_cycle(f6, "c", "a") is False      # a into c: fine (a is upstream)
      and se.would_cycle(f6, "a", "c") is True   # c into a: a feeds c -> cycle
      and se.would_cycle(f6, "c", "c") is True)  # self-loop
f6.nodes.append(se.Node(id="d", kind="op", op="x", inputs=["b"]))
check("downstream delete set", se.downstream_ids(f6, "b") == {"b", "c", "d"})

# ---- 7. save -> reload round trip preserves the function ----
import tempfile
import yaml
sc7 = se.load_scenario(SAMPLE)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "rt.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(sc7.to_dict(), f, sort_keys=False)
    sc7b = se.validate_scenario(p)
    d1, _ = min_dist(sc7)
    d2, _ = min_dist(sc7b)
    check("round-trip validate + same behavior", abs(d1 - d2) < 1e-9,
          f"{d1:.4f} vs {d2:.4f}")


# ---- 8. validator rejects bad graphs ----
def expect_invalid(name, mutate):
    raw = yaml.safe_load(open(SAMPLE))
    mutate(raw)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        p = f.name
    try:
        se.validate_scenario(p)
        check(name, False, "accepted!")
    except ValueError as e:
        check(name, True, str(e)[:60])
    finally:
        os.unlink(p)


expect_invalid("rejects point-bound OUT",
               lambda r: r["actors"][1]["maneuvers"][0]["out"].update(speed="n1"))
expect_invalid("rejects missing input",
               lambda r: r["actors"][1]["maneuvers"][0]["nodes"][4].update(inputs=["zz", "n2"]))
expect_invalid("rejects incomplete op",
               lambda r: r["actors"][1]["maneuvers"][0]["nodes"][4].update(inputs=["n1"]))
expect_invalid("rejects unknown observed actor",
               lambda r: r["actors"][1]["maneuvers"][0]["nodes"][0].update(source="42"))
expect_invalid("rejects cycle",
               lambda r: r["actors"][1]["maneuvers"][0]["nodes"][6].update(inputs=["n7", "n5"]))

print()
print("ALL PASS" if not failures else f"FAILURES: {failures}")
sys.exit(1 if failures else 0)
