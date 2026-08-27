# `carla_highway` — CARLA 0.9.16 backend for the highway orchestrator

This package puts the **highway** orchestrator —
[INPUTrrr0/scenario_orchestrator_highway](https://github.com/INPUTrrr0/scenario_orchestrator_highway),
i.e. the repository this branch lives in — into CARLA 0.9.16. The script layer
is the repository root: `../scenario_editor.py`, `../maps.py`,
`../maneuvers.py`, `../directives.py`, `../cutin_orchestrator.py`. None of it
is edited by the port; [`script_bridge.py`](script_bridge.py) is the only
module that knows where it lives.

`../carla_port/` alongside is the script-agnostic half of the sibling
signalized-junction port (cameras, video, collision sensors, actuation, the
CARLA API shim). Only the modules `carla_highway` actually imports were
carried over onto this branch; the junction script layer (`v2/`, `v4/`) and the
modules that need it were not.

```
vendored highway orchestrator  +  a straight-road CARLA frame  +  an ego policy
```

Three scenarios, the ones the upstream repo ships as its stress tests:

| mode | road it needs | what happens |
|---|---|---|
| `cutin` | 3 same-direction lanes, 120 m | the orchestrator **casts** one of four actors as the cut-in and replans it against the live ego every 0.1 s; it recasts when the holder becomes hopeless |
| `hard_brake` | 2 same-direction lanes, 160 m | a 4 m/s lead in the ego's lane, a 10.5 m/s lead in the next one squeezing the merge gap |
| `overtake` | 2 lanes, **two-way**, 160 m | a blocker brakes to a stop in the ego's lane; the only way past is the **oncoming** lane, with a car coming |

---

## 1. Running it

```bash
python3 -m carla_highway --scenario cutin --town Town04
```

from the repository root, with CARLA 0.9.16's Python API importable, a server
running, `Pillow` for the HUD and `ffmpeg` on PATH. Useful flags:

```
--scenario MODE      cutin | hard_brake | overtake   (default cutin)
--town NAME          default Town04: it has both the multi-lane highway and a
                     two-way road, so all three modes fit at authored scale
--road-id N          pick the road section explicitly instead of fitting
--min-length M       override how much straight road the scenario needs
--ego-mode MODE      physics: IDM -> PID -> VehicleControl, ego read back from
                     CARLA (default).  bicycle: the policy's own kinematic
                     model, mirrored in with set_transform
--scripted-ego       no policy; drive the ego along its maneuver plan
--no-lane-change     lane-keeping ego only — the drivev2 baseline (see §4)
--desired-speed M/S  ego free-flow speed (default 12.0)
--casting / --no-casting     force role casting on or off
--sync-mode physics|kinematic       how BACKGROUND actors are driven
--duration SECONDS   simulated seconds (default 12)
--report out.json    machine-readable run report
--video PATH / --video-dir DIR / --video-view top|chase|both / --no-video
```

Validation, **no server needed**:

```bash
python3 -m carla_highway.validate
python3 -m carla_highway.validate --town Town04 --town Town05
```

CARLA's Python API can build a `carla.Map` straight from an OpenDRIVE string, so
the map fit — the riskiest part of this port — is checked against the *real*
Town geometry offline rather than against a synthetic double. 92 checks.

---

## 2. Why this is a separate package

`carla_port` and `carla_highway` **cannot run in the same process**, and the
reason is worth stating plainly because it drove most of the design.

Both script layers are *flat* module sets that claim the same bare top-level
names from different files:

| bare name | `carla_port` | `carla_highway` |
|---|---|---|
| `scenario_editor` | `v2/scenario_editor.py` | `<repo root>/scenario_editor.py` |
| `directives`, `maneuvers`, `directives_script`, `orchestrator` | `v2/`, `v4/` | `<repo root>/` |

Python caches modules by name, so whichever imports first wins and the other
silently gets the wrong classes — the same identity hazard
`carla_port/script_bridge.py` documents one level up for `v2.scenario_editor`
vs `scenario_editor`. Rather than paper over it,
[`script_bridge.py`](script_bridge.py) *detects* it and raises.

So the two ports are alternative backends over different script layers. They
share the CARLA-side mechanics and nothing else.

### What is shared, not duplicated

Making the sharing possible took four small changes in `carla_port`, none of
which alter its behaviour:

* `carla_adapter`, `carla_sync`, `carla_video`, `carla_obs` and `ego_driver`
  imported the frame and script types **only for annotations** — and every one
  of those modules has `from __future__ import annotations`, so under PEP 563
  they were never evaluated. They moved under `if TYPE_CHECKING:`, which makes
  the modules frame-agnostic. `carla_sync`'s one real use, `DT = se.DT`, became
  the literal `1/60` both script layers define.
* `carla_collision` was already clean.
* The longitudinal PID and the `carla.VehicleControl` conversion moved out of
  `carla_ego.py` into [`carla_port/actuation.py`](../carla_port/actuation.py),
  parameterized by the vehicle limits. Both ports actuate through **one**
  controller. `ego_driver.py` already argues that re-deriving a controller
  "would silently change the numbers while still calling them the policy's";
  that applies just as much across two ports of one control law.
* `carla_video`'s `Hud` gained optional `line3` / `badges` / `hero_label`
  overrides so a backend whose world is not a junction can fill the same two
  rows with its own terms. Defaults are unchanged, so `carla_port` still
  renders `signals ...` and `D1..D4`.

### Reused, not reimplemented

Role casting is **not** rewritten here. `CutinOrchestrator.tick` upstream
already rebases, re-casts, replans the holder toward the ego-relative pin,
applies the actor–actor collision-yield directive and re-simulates — and it
touches pygame only in `draw_panel`, which nothing here calls. So
[`closed_loop.py`](closed_loop.py) owns the *cadence and the clock*, not the
policy, and the orchestration logic stays upstream where highway runs and CARLA
runs can be compared against each other.

---

## 3. Fitting the road

[`highway_map.py`](highway_map.py) is the counterpart of `carla_port`'s
`carla_map.py`. Where that fits one signalized junction, this fits one straight
multi-lane stretch, because that is the world the highway scripts live in:

```python
MapConfig(kind="straight", num_lanes=N, lane_width=w, length=L)
lane_center_x(i) = -N*w/2 + w/2 + i*w
```

`HighwayFrame` reuses `IntersectionFrame`'s coordinate convention *exactly* —
anchor, `theta`, the y-mirror, `yaw = theta - heading` — so the two frames are
interchangeable everywhere the shared mechanics are used. Forward
(script heading 90) maps to the ego lane's travel direction `D`, so
**`theta = D + 90`**, which also puts script `+x` 90 degrees clockwise of travel
in CARLA's left-handed frame, i.e. on the driver's right, as the scripts assume.

`lane_width` is measured from real lane-centre spacing rather than taken from
`Waypoint.lane_width`, for the same reason `carla_map` does it: the scripts
place actors at `lane_center_x(i)` and those places have to be real lanes.

Two findings shaped `discover()`:

**Runs must continue through junctions.** Stopping the walk at
`is_junction` makes every town look like it has no straight longer than ~20 m,
along roads that visibly run for hundreds. At a fork the walk now takes the
branch whose heading best matches the start heading — driving straight on.
`through_junctions=False` restores the strict behaviour.

**Straightness is a lateral budget, not a heading tolerance.** A 4-degree
per-step heading tolerance integrates to metres over a 200 m fit. It passed
every per-step check on Town05 while placing actors **1.46 m** off the real lane
centre — half a car outside the lane. The binding criterion is now cumulative
perpendicular deviation from the straight axis, capped at `MAX_LATERAL_DEV`
(0.35 m), which is exactly the quantity that breaks a scenario. Worst actor
placement across Town04 and Town05 is now **0.04 m**.

What the towns actually offer (from `validate.py`):

| town | 3-lane one-way | 2-lane one-way | 2-lane two-way |
|---|---|---|---|
| Town04 | 390 m | 390 m | 170 m |
| Town05 | 160 m | 250 m | 250 m |
| Town03 / Town10HD | — / — | 190 m / 130 m | 200 m / 130 m |

Town04 is the default: it is the only installed town where all three modes fit
at their authored scale. Where a real straight is shorter than the YAML's
`length`, [`scenarios.py`](scenarios.py) scales the *longitudinal* layout to fit
and says so in the run notes — the scenarios are timing-critical, so that
compression changes their difficulty and should not pass silently.

### Retargeting, not rewriting

The scenarios are the authored YAML files, loaded with `se.load_scenario`. Each
actor keeps its **intent** — which lane, how far along, which way it faces, what
it does — and only the geometry is re-derived from the frame. Speeds and
maneuver plans are untouched: they *are* the scenario.

---

## 4. The ego

This is the one place the port writes something `carla_port` deliberately does
not, and it is worth being explicit about why.

`carla_port` subclasses `v4/drivev2.py`'s `Drive` and inherits `idm_control` /
`path_steer` / `autonomous_control` unmodified. Neither half of that is
available here:

1. **Import identity.** `drivev2` does `import scenario_editor as se`, and that
   bare name belongs to the *highway* layer in this process (§2). Importing it
   would bind drivev2 to the wrong script layer.

2. **A fixed reference path cannot do these scenarios.** drivev2 builds
   `_ref_path` once from the junction turn geometry and pure-pursues it forever.
   That is right when the route is decided before the run. But `hard_brake` asks
   the ego to *change lanes* around a slow lead, and `overtake` asks it to use
   the **oncoming** lane to get around a stopped car. IDM is longitudinal only:
   on a fixed path the ego would brake and sit behind the blocker forever, and
   both scenarios would grade as a failure of the harness rather than of the
   policy under test.

So [`highway_ego.py`](highway_ego.py) keeps drivev2's laws — IDM longitudinally
(same constants, quoted from `v4/drivev2.py`), pure pursuit laterally — and adds
a **lane-selection layer**: gap acceptance over the adjacent lanes, preferring
same-direction lanes, treating the oncoming lane as a last resort behind an
explicit time-to-collision gate, and returning to the home lane once it is
clear. On a straight road the reference path is the line `x = lane centre`, so
the lookahead point is analytic and drivev2's polyline search disappears.

### What the baseline actually does

Five defects in this layer only showed up once it drove a real car, and they are
worth knowing about because they are the failure modes any lane-selection policy
has to get right:

* **A merging car belongs to no lane.** Matching leaders by lane index makes a
  car half way through a cut-in invisible until it has finished merging — the
  ego drove into it at 2.8 s. Leaders are matched by lateral distance
  (`IDM_LANE_TOL`), which is what drivev2 does and why its comment says it
  "naturally picks up a right-turn merge target".
* **Steering needs speed; a blocked lane forbids speed.** Stopped behind the
  stopped blocker with a clear lane alongside, the ego could never move again:
  the bicycle model's yaw rate is proportional to `v`, and IDM holds a stopped
  car at its jam distance. A bounded creep breaks the deadlock, allowed only
  while committed to a lane that is clear.
* **Going home must not undo the reason you left.** Testing the home lane on
  distance alone is satisfied by the very obstacle being avoided, so the ego
  pulled out to overtake and merged straight back in behind the blocker. The
  return decision now uses the same IDM comparison as the decision to leave.
* **A pass takes as long as it takes to get past the blocker.** Estimating it
  as `distance / ground speed` assumes the obstacle is parked: it predicted a
  2.8 s pass that really took 5.5 s and left 4 m to an oncoming car. The
  divisor is now the closing speed on the blocker, floored so a same-speed
  blocker does not make every overtake look infinite.
* **Body clearance is a box separation**, not `hypot() - radius`, which called a
  clean lane-apart pass a zero gap.

With the honest pass estimate the baseline **declines** `overtake`'s early
window — the one the scenario describes as "a tight gap in front of the oncoming
car" — and takes the other route it offers: wait, let the oncoming car through,
then go around. Committing to the early window needs a policy that anticipates
the blocker is about to stop, which is exactly the kind of judgement an external
`ego_policy_v1` is there to supply.

It is a **baseline to stress, not a driver to be proud of** — deliberately small
and legible. Two escape hatches:

* `--no-lane-change` restores pure lane-keeping IDM, the drivev2 behaviour. Use
  it to confirm that `hard_brake` and `overtake` are unsolvable that way (the
  ego slows to the lead's speed and stays there); the validation suite asserts
  exactly that.
* `--ego-policy` routes through `carla_port.ego_driver.PolicyEgoDriver`, which
  speaks `scenario_orchestration`'s `ego_policy_v1`. Bringing your own policy
  is the point; this class is what runs when you do not.

---

## 5. The loop

CARLA runs synchronously at `fixed_delta_seconds = se.DT = 1/60 s`, the scripts'
own trajectory step, so sampled states land exactly on grid frames. Per step:

```
A. read the ego back from CARLA      (CARLA integrated it; CARLA is authoritative)
B. orchestrate when due              (every DT_TICK = 0.10 s, cutin mode only)
C. sample the script world at the current script time
D. write the BACKGROUND actors in    (the ego is driven, not placed)
E. actuate the ego                   (policy -> PID -> VehicleControl)
F. world.tick()
G. collect collisions, grade, capture a video frame
```

`CutinOrchestrator.tick` returns a scenario rebased to *now*, so script-local
time restarts at zero on every orchestration tick; `closed_loop.py` resets its
clock accordingly. That is the same contract `carla_sync.py` describes for
`Orchestrator._rebase_here`, reached by a different route.

Only `cutin` is orchestrated. `hard_brake` and `overtake` are ego-policy stress
tests whose actors are fully scripted **on purpose** — cast roles or yields
would change the very timings they were tuned around. `--casting` overrides it.

### Recasting, which does not come for free

`CutinOrchestrator.cast` is sticky on *identity*: once `cutin_id` is set it
re-locks the same actor every tick until the cut-in commits. The rule that moves
the role lives one level up, in `scenario_editor.py`'s `cast_cutin_roles`, which
the pygame editor calls and a headless port does not. Without it the first pick
holds the role forever — on `scenario_cutin` the opening tie went to actor 2,
which stayed cast at a candidate score of **0.016** while actor 4 sat at
**0.64** and the cut-in ran out its deadline.

`closed_loop.py` ports that rule, with upstream's own predicates
(`live_cutin_feasible`, `live_cutin_required_speed`). The important half of it is
easy to get backwards:

> stickiness is on **feasibility**, not score.

Mid-chase the holder drifts toward the ego's lane, which tanks its *candidate*
score — that is progress, not failure. So the role moves only when the holder
can no longer reach the pin by the deadline, and then it goes to the best-scoring
actor that still can.

---

## 6. Grading

Each mode is graded in the terms it was written in, and the verdict goes to the
HUD, the summary line and the JSON report:

| mode | success |
|---|---|
| `cutin` | the cut-in **merged** (not abandoned) and nothing hit the ego |
| `hard_brake` | the ego got **past the slow lead** without a collision |
| `overtake` | the ego got **past the blocker**, **returned to its lane**, and hit nothing |

The report also carries the fitted frame (road, lanes, `lane_fit_error`), the
ego's lane changes and whether it used the oncoming lane, per-actor minimum gap
and pass/no-pass, every orchestration event with its timestamp, and all realized
CARLA collisions.

---

## 7. What the runs show

Town04, `--ego-mode physics`, the five runs in one job:

| run | verdict | what happened |
|---|---|---|
| `cutin` | merged, **contact** | role recast to actor 4, cut-in merges; the merge grazes the ego |
| `hard_brake` | **PASS** | 2 lane changes, 133 m, closest pass 1.6 m |
| `overtake` | **PASS** | waits out the oncoming car, creeps past the blocker, returns to lane |
| `cutin --scripted-ego` | merged, **contact** | same recast to actor 4, same contact |
| `hard_brake --no-lane-change` | **FAIL** | "stuck behind the slow lead" |

Two of those rows are the point of the table.

**`hard_brake --no-lane-change` failing is the evidence for §4.** With pure
lane-keeping IDM — drivev2's behaviour — the ego closes on the 4 m/s lead,
matches its speed and sits there for the whole run. The scenario is not
solvable without a lateral decision, so a port that only inherited drivev2's
policy would have reported the harness failing rather than the policy.

**`cutin --scripted-ego` reproducing the contact is what clears the ego
policy.** That run drives the authored 13 → 4 m/s profile the YAML was written
around, so the casting runs under upstream's own conditions — and it recasts to
actor 4 and merges, exactly as the upstream README describes. It also makes
contact, just like the policy ego. The cause is geometry, not driving:

```
cut-in pin is along=6.0 m centre-to-centre; with the spawned bodies
(3.7 m ego, up to 5.2 m actor) that is 1.55 m bumper to bumper
```

`along` is centre to centre, and `spawn_bindings(adopt_carla_extents=True)`
replaces the script's nominal 4.5 × 2.0 m bodies with the real bounding boxes.
In the script world nothing enforces the result — pygame has no physics, so a
pin that overlaps is simply drawn overlapping — while CARLA's collision sensors
do. `runner.py` reports this clearance at setup and flags it when it is under
`CUTIN_TIGHT_CLEARANCE`. Widen `along` in the scenario YAML to give the merge
room; the port deliberately does not edit the authored scenario to hide it.

The background collision in `overtake` (`2 x static.vegetation @ 19.75s`) is
the oncoming car outliving the scenario and driving off the end of the fitted
straight, well after the measured part. It is reported, not graded — see §6.

---

## 8. Layout

| file | owns |
|---|---|
| `script_bridge.py` | the only import site for the script layer; the collision guard (§2) |
| `highway_map.py` | `HighwayFrame`: fitting the straight road, the coordinate conversion |
| `scenarios.py` | the three modes; loading the authored YAML and retargeting it |
| `highway_ego.py` | the ego: IDM + pure pursuit + lane selection (§4) |
| `closed_loop.py` | orchestration cadence and clock around `CutinOrchestrator` |
| `runner.py` | connection, sync mode, spawning, the loop, grading, reporting. **Entry point** |
| `validate.py` | the offline checks, against real OpenDRIVE geometry |

Nothing in the script layer is edited, exactly as `carla_port` leaves `v2/` and `v4/`
alone. `carla_port` itself is imported only for modules that touch no script
layer: `carla_api`, `actuation`, `carla_adapter`, `carla_sync`,
`carla_collision`, `carla_video`, `carla_obs`, `ego_driver`.

Outputs land in `carla_highway/outputs/` (mp4) and `carla_highway/reports/`
(JSON); both are gitignored.
