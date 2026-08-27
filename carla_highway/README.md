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

So [`highway_ego.py`](highway_ego.py) restates the laws instead of importing
them, quoting the values from the reference implementations:

| law | what it decides | reference |
|---|---|---|
| **IDM** | speed | `v4/drivev2.py` |
| **MOBIL** | which lane | the highway `drivev2.py` of PR #1, *"Model-based ego for drivev2.py: IDM speed control + MOBIL lane changes"* |
| lateral profile + inverse bicycle model | the steering that realises a change | same PR |

### From gap acceptance to MOBIL

The first pass of this file predates PR #1. It paired IDM with a hand-rolled
gap-acceptance rule — accept a lane if a fixed window ahead (`LC_FRONT_GAP` 8 m)
and behind (`LC_REAR_GAP` 6 m) is empty; among the acceptable ones take
whichever IDM likes best — and tracked the chosen lane centre with pure pursuit.
MOBIL replaces the decision and the profile replaces the tracker. Three things
change, none cosmetic:

1. **The safety criterion acquires units.** Gap acceptance asked *"is 6 m behind
   me empty?"*. MOBIL asks *"how hard would merging make that car brake?"* and
   refuses above `MOBIL_B_SAFE` (4 m/s²). Those are not the same question: 25 m
   of gap is comfortable behind a follower matching your speed and a collision
   behind one closing at 30 m/s, and only the second question separates them.
   The validation suite pins exactly that pair.
2. **The gain is polite.** It counts the acceleration inflicted on the follower
   left behind and the follower merged in front of, weighted by `MOBIL_P`, not
   only the ego's own gain. Background actors here never actually react, so
   politeness models a courtesy the traffic will not reciprocate — the safety
   term is what protects them.
3. **The steering survives braking.** The lateral profile advances with
   *distance travelled*, not wall-clock time (`LC_DISTANCE`, 30 m). This is the
   one that matters most on `cutin`: IDM brakes the ego to a crawl mid-change,
   and a time-parameterised tracker then demands the same sideways displacement
   with no forward speed left to make it with — the heading blows up and the
   wheel saturates. On distance, curvature works out to `d·S''(f)/L²`, entirely
   speed-independent, and the lateral motion simply stops when the car does.
   The profile is the quintic smoothstep rather than the script layer's cubic,
   because here the second derivative *is* the steering command and the cubic
   has `S''(0) = 6`: it would demand a step onto full lock at the start of every
   change.

Two deliberate deviations from upstream, both because this port drives roads
upstream's scenarios do not have:

* **Keep-*home* rather than keep-*right* bias.** Same mechanism — an asymmetric
  switching threshold — keyed to the lane the scenario spawned the ego in
  instead of to +x. They agree whenever home is the rightmost lane. Where they
  differ is `cutin`, which authors the ego into the **centre** lane of three: a
  keep-right bias would walk it out of the lane the cut-in was solved against
  before the orchestrator ever casts a role.
* **A contraflow veto is kept** (`_oncoming_clear`). Upstream argues, correctly,
  that with signed velocities an oncoming car is just a leader with a negative
  `v` and IDM's `dv` term reads the true closing speed — no special case needed.
  But MOBIL's incentive term is an *instantaneous* acceleration comparison, and
  a head-on closing at 25 m/s from 60 m away barely dents it while being exactly
  the thing that kills you. `overtake` is the only mode where this fires.

Signed velocities came across as well, and they matter beyond contraflow:
`Actor.speeds` is an unsigned magnitude along each actor's *own* heading, so
before projection an oncoming car and a car driving away both read `+8`, and
IDM's `dv = v_ego - v_lead` reports `+4` for a pair closing at 20 m/s — the ego
accelerates at a car coming straight for it. So does lane membership by **body
overlap** instead of nearest centre, which keeps a merging car in *both* lanes
it straddles rather than teleporting it across the line in one frame.

### What the baseline actually does

Five defects in this layer only showed up once it drove a real car, and they are
worth knowing about because they are the failure modes any lane-selection policy
has to get right:

* **A merging car belongs to no lane.** Matching leaders by lane index makes a
  car half way through a cut-in invisible until it has finished merging — the
  ego drove into it at 2.8 s. Leaders are matched by lateral distance
  (`IDM_LANE_TOL`), which is what drivev2 does and why its comment says it
  "naturally picks up a right-turn merge target". MOBIL's body-overlap lane
  membership is the same insight applied to the *decision* rather than the
  car-following: the lane a merger is vacating must not read as empty while its
  body is still sitting in it.
* **Steering needs speed; a blocked lane forbids speed.** Stopped behind the
  stopped blocker with a clear lane alongside, the ego could never move again:
  the bicycle model's yaw rate is proportional to `v`, and IDM holds a stopped
  car at its jam distance. A bounded creep breaks the deadlock, allowed only
  while committed to a lane that is clear.
* **Going home must not undo the reason you left.** Testing the home lane on
  distance alone is satisfied by the very obstacle being avoided, so the ego
  pulled out to overtake and merged straight back in behind the blocker. The
  explicit "return home" branch that fixed it is gone now: MOBIL's keep-home
  bias is the same idea expressed as a threshold, and the return is evaluated
  by the identical acceleration comparison as the departure by construction.
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

Getting that rule *half* right is worse than not having it, and the first CARLA
cut-in runs found both halves the hard way. Every run — the MOBIL ego, the
lane-keeping ego, and the scripted ego reproducing upstream's own conditions —
reported `abandoned`, without exception.

**Feasibility and eligibility are different tests, and the recast has to respect
both.** `live_cutin_feasible` asks whether an actor could still reach the pin by
the deadline; a car coming up from behind can. But `cast_roles` will only cast
an actor that is `cutin_eligible` — adjacent lane **and already ahead** — and
only honours a lock whose score is above zero. Handing the role to a feasible
but ineligible actor produces a two-tick oscillation at 10 Hz: the recast sets
`cutin_id`, the next `cast` refuses the lock and falls to "no viable candidate",
the tick after re-picks the same hopeless holder. That was **51–57 interventions
inside four seconds**, never committing. `_recast_if_hopeless` now requires a
non-zero score of its replacement, and otherwise leaves the holder to abandon on
time — which is what upstream does.

**A merging holder scores zero, and `cast_roles` will take the role off it on
the very tick the merge becomes detectable.** `cutin_is_merged` needs the actor
within 0.5 m of a pin whose lateral offset is 0 — on the ego's own line;
`cutin_adjacent` needs it at least 0.4 lane widths away from that line. The two
cannot hold at once, `tick` casts before it plans, and so
`apply_closed_loop_cutin` — the only thing that can ever return `"merged"` — is
never called for the actor that just merged. `scenario_editor.py`'s casting does
not have this problem because it identifies the holder by *which actor carries
the spec*. [`StickyCutinOrchestrator`](closed_loop.py) applies the same rule to
`CutinOrchestrator` by overriding `cast` alone: the holder keeps the role while
`_recast_if_hopeless` judges it still able to make the pin, and everything else
is cast exactly as before. With it, all three egos merge.

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

### Grading with the upstream verifier instead

That grade is asked in CARLA's terms — did the collision sensor fire, did the
orchestrator declare a merge. The repository's own verifier
([`scripts/scenario_verify.py`](../scripts/scenario_verify.py), described in
[`docs/SCENARIOS_AND_VALIDATION.md`](../docs/SCENARIOS_AND_VALIDATION.md)) asks
a different and stricter question: it does **not** trust the orchestrator's
verdict, and re-derives all four cut-in criteria from the recorded
*trajectories*.

So every run also writes a second file in that verifier's schema:

```bash
python3 -m carla_highway.runner --scenario cutin --report out/run.json
# -> out/run.json          the port's own report
# -> out/run.verify.json   the same run, in the upstream schema
python3 scripts/verify_run.py out/run.verify.json -v
```

`--verify-report PATH` names it explicitly; `--traj-dt` sets the sampling
period (default 0.05 s — the verifier takes the last sample at or before a
query time, so this is the timing resolution of every check it makes).

No coordinate conversion is involved, and that is not luck: the script frame
this port works in *is* the frame the scenarios are authored in — `+y` along the
road, lanes separated in `x`, headings in degrees. The poses are read back from
CARLA rather than taken from the commanded states, because the point of
verifying a CARLA run is to check what the simulator actually did with the plan.

Two things the cut-in verifier does that the port's own grade does not, and both
are worth knowing before reading a result:

* **It fails a run where the ego left its starting lane before the actor
  merged** — `ego left target lane before actor merged`, and separately `ego not
  in original lane at commit`. A cut-in is defined relative to the lane the ego
  was in. This is the criterion an ego with MOBIL lane changes runs straight
  into (§7).
* **It does not check for collisions at all.** `verify_proper_cutin` never calls
  `_any_collision` — only the overtake verifier does. A cut-in can therefore
  verify 4/4 while the two bodies are overlapping in CARLA, which is precisely
  what the authored 6 m pin does against real CARLA vehicle extents. Read the
  port's `ego_collisions` alongside it; `--cutin-along` exists for this.

---

## 7. What the runs show

### The cut-in, three egos

Town04 road 40, `--ego-mode physics --duration 14 --cutin-along 9`:

| ego | port verdict | upstream verifier | what happened |
|---|---|---|---|
| `--lane-change` (IDM + **MOBIL**, §4) | **PASS** merged, no contact | **FAIL** `ego left target lane before actor merged` | MOBIL leaves the centre lane at t=1.3 s; the role is recast 2 → 4 at t=3.2 s and actor 4 merges at t=4.28 s, 8.8 m ahead and 0.2 m off the ego's line |
| default (IDM, lane-keeping) | **PASS** merged, no contact | **OK 4/4** | same recast, merge at t=3.87 s, 6.2 m ahead |
| `--scripted-ego` (the authored 13 → 4 m/s profile) | **PASS** merged, no contact | **OK 4/4** | actor 2 holds the role throughout and merges at t=5.48 s, 10.0 m ahead |

The third row is the fidelity check: it drives the ego the YAML was authored
around, so the casting runs under upstream's own conditions.

### The MOBIL ego cannot pass the cut-in verifier, and that is a real finding

Its merge is geometrically correct — 8.8 m ahead of the ego, 0.2 m off its
line, comfortably inside the 10 m station window, no contact. It fails because
`verify_proper_cutin` measures everything against `ego_trajectory[0][1]`: the
`x` the ego started at, held fixed for the whole run. An ego that changes lanes
therefore trips two of its rules by construction —

```
t_ego_out = _first_event(ego_tr, t_commit, ego_leaves_home)   # -> "ego left
                                                              #  target lane"
if not _in_lane_x(ego_t[0], ego_home_x, lw):                  # -> "ego not in
                                                              #  original lane"
```

— no matter where the cut-in actually landed relative to it.

`docs/SCENARIOS_AND_VALIDATION.md` describes the opposite intent in as many
words: *"Ego on lane 1, actor 1 on lane 2 … then Ego merges to lane 1. Actor 1
is no longer valid to perform the intention. For the scenario to be successful,
the orchestrator should choose actor 2."* That is a scenario in which the ego
changes lanes and the orchestrator recasts — precisely what the MOBIL run does,
and precisely what the implementation cannot score. Reading criterion 1
("**ahead of the ego** and within 10 m") and criterion 3 ("merges into the
**ego's lane**") against the ego's *live* lane rather than its spawn lane would
close the gap; that is the verifier's call, and the port does not presume it.

Until then, `--lane-change` and the verifier are asking different questions of
the same run, and the port's own grade (§6) and the report's `ego.lane_changes`
are what distinguish them.

### The stress scenarios

| run | verdict | what happened |
|---|---|---|
| `hard_brake` | **PASS** | 2 lane changes, 133 m, closest pass 1.6 m |
| `overtake` | **PASS** | waits out the oncoming car, creeps past the blocker, returns to lane |
| `hard_brake --no-lane-change` | **FAIL** | "stuck behind the slow lead" |

**`hard_brake --no-lane-change` failing is the evidence for §4.**

 With pure
lane-keeping IDM — drivev2's original behaviour — the ego closes on the 4 m/s lead,
matches its speed and sits there for the whole run. The scenario is not
solvable without a lateral decision, so a port that only inherited drivev2's
policy would have reported the harness failing rather than the policy.

### The pin is authored against bodies CARLA does not spawn

At the authored `along: 6.0` every ego made **contact** with the merging actor,
including the scripted one. The cause is geometry, not driving:

```
cut-in pin is along=6.0 m centre-to-centre; with the spawned bodies
(3.7 m ego, up to 5.2 m actor) that is 1.55 m bumper to bumper
```

`along` is centre to centre, and `spawn_bindings(adopt_carla_extents=True)`
replaces the script's nominal 4.5 × 2.0 m bodies with the real bounding boxes.
In the script world nothing enforces the result — pygame has no physics, so a
pin that overlaps is simply drawn overlapping — while CARLA's collision sensors
do. `runner.py` reports this clearance at setup and flags it when it is under
`CUTIN_TIGHT_CLEARANCE`. The port does not edit the authored scenario to hide
it; `--cutin-along` overrides the pin from the command line instead, and the
runs above use `--cutin-along 9` — 4.55 m bumper to bumper, and still inside
the 10 m window that both `se.cutin_is_merged` and the verifier check. Note
that the cut-in verifier would not have caught this either way: it never calls
`_any_collision` (§6).

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
