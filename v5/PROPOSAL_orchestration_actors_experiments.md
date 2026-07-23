# Design proposal: robust orchestration, an actor execution layer, and the experiment suite

This proposes solutions for the three task areas, grounded in the current code
(`v4/orchestrator.py`, `v4/drive.py`, `v4/directives_script.py`, the v2 kernel in
`v2/scenario_editor.py` / `v2/directives.py`, and the unbuilt actor design in
`v3/DESIGN.md`). It ends with a substrate recommendation and a phased plan.

The three tasks are coupled by one loop that already exists in `drive.py`: every
few frames the orchestrator queries costed actions from the actors (`d1_options`)
and commits the cheapest that forces a collision. Task 1 makes that loop *reliably*
win; task 2 makes the actors it queries *well-behaved* (no overlap, on-road, gap-
keeping) and *fast*; task 3 wraps the whole thing in a portable benchmark. The
current `d1_options` / `repair` machinery and the `evaluate` D1/D2/D3 family are the
seams every proposal below plugs into, so nothing here is a rewrite.

---

## 0. Where the current system stands

The orchestrator's job is to make some background actor collide with the ego,
preferably a side (T-bone) impact of a red-light runner. The state of play:

The **headless kernel** (`orchestrator.py`) drives a maneuver-script world and, each
tick, evaluates the red-light family and applies the *minimal causal repair*
(`ds.repair`) as retime/reroute edits to non-ego actors. It is a *repairer*: it acts
only when a standing command needs changing, and it is built to keep a scripted
scenario collidable.

The **playable loop** (`drive.py`) is the harder problem and the one the task
targets: a human (or policy) drives the ego with a bicycle model, and every
`ORCH_EVERY = 3` frames `Drive.orchestrate()` calls `ds.d1_options(..., verify=False)`
to pick a single **pursuer** — an uncommitted red-runner — and retimes it to
`v' = d/t_mid` so it arrives at the conflict point `P` in the middle of the ego's
predicted occupancy window. The pursuer shadows the ego: slow to wait, speed to
catch.

Why it **fails on some seeds** (the reported bug):

1. **Single pursuer, greedy, memoryless.** `d1_options` returns the cheapest single
   actor; `orchestrate()` keeps one `pursuer`. If that actor becomes uncommitted-
   infeasible (crosses `P`, or its route no longer intersects the ego's after the
   ego turns/stops), there is no fallback to a second actor in the *same* tick, and
   no coordination between two actors to cover both escape directions.
2. **Prediction/actuation mismatch.** The ego prediction in `_world()` is a *single
   constant-speed straight segment* over the horizon (`go_straight` at current `v`).
   A human who accelerates, brakes, or turns invalidates `P` and the window; the
   re-aim only corrects at 10 Hz and always one step behind. On seeds where the ego
   maneuvers hard, the pursuer chronically aims at a stale `P`.
3. **`v' = d/t_mid` is a point solution, not a capture guarantee.** It centers the
   pursuer on the *current* window midpoint but does not certify the collision is
   *unavoidable*: a decelerating ego shrinks and shifts the window faster than a
   `v_max`-capped pursuer can re-close, so the pursuer overshoots `P` ahead of the
   ego and misses behind.
4. **No T-bone shaping.** Any collision counts (`collision_now` is body-overlap);
   nothing biases toward a *lateral/head-on-T* geometry, so the wins that do happen
   are often glancing/rear.
5. **Actors are not defensive or well-behaved** (task 2), so "the ego loses" is
   currently an artifact of dumb traffic, not a certified critical scenario.

The fixes below keep the exact interfaces (`evaluate`, `d1_options`, `repair`,
`retime_actor`/`reroute_actor`, the rebase machinery) and change the *decision
logic* on top of them.

---

## 1. Orchestration algorithm (task 1)

**Goal.** At each tick: (a) query every actor for the cost of the actions available
to it, (b) choose actions for *all* actors that minimize a global objective whose
minimizer is a T-bone on the ego. Reliably win; optionally real-time.

### 1.1 The objective

Define per-actor the current-tick action set `A_i` (from task 2 / `d1_options`), and
let a *joint plan* be one action per non-ego actor. The orchestrator minimizes

```
J(plan) =  w_miss · MISS(plan)                 # will the ego be hit at all?
         + w_geom · GEOM(plan)                  # is the hit a side/T-bone?
         + w_time · TIMING(plan)                # is the hit unavoidable for the ego?
         + w_act  · Σ_i cost_i(action_i)        # action effort (reuse w_v, w_r)
         + w_beh  · Σ_i illegal_i(action_i)     # off-road / overlap / gap penalties
```

with terms grounded on the rollout the actors produce (task 2), not a closed form:

- **`MISS`** — 0 if some actor's swept body overlaps the ego within the horizon
  (reuse `collide_time` / `collision_now`), else the minimum body-to-body distance
  at closest approach along the predicted rollout. This is the dominant term; its
  gradient (via finite differences over candidate speeds) is what the actors' cost
  curves already approximate through `v' = d/t_mid`.
- **`GEOM`** — the T-bone shaper. At the predicted contact frame, take the angle
  `θ` between ego heading and impactor heading. Reward `θ ≈ 90°` (T-bone) or `180°`
  (head-on): `GEOM = min(|θ−90°|, |θ−180°|) / 90°`. Because red-runners on the
  crossing arms (`EN`/`WS` vs. the ego's `SE`) approach perpendicular to the ego by
  construction, a red-runner hero already scores well here — `GEOM` mainly breaks
  ties toward the crossing actor over a same-lane rear-ender.
- **`TIMING`** — the unavoidability term, and the real fix for the seeds that fail.
  Instead of centering on the window midpoint, score how deep the impactor enters
  the ego's **point-of-no-escape (PONE) window** — the interval during which the
  ego, braking at `b_comf` from its current speed, can no longer clear `P`
  (`d_ego < v_ego²/(2·b_comf)` + reaction, exactly the v3/DESIGN §1.5 / P12 tube
  boundary). `TIMING = 0` when the impactor's occupancy of `P` lands inside the PONE
  window; grows as it drifts toward the escapable edges. Minimizing this selects the
  timing that traps a *defensive* ego, which is what makes the win robust to the ego
  braking or accelerating.
- **`Σ cost_i`** — reuse the existing `w_v·|Δv| + w_r` from `Params`.
- **`Σ illegal_i`** — large penalty so the chosen actions stay on-road, non-
  overlapping, gap-respecting (defined by task 2). This keeps the orchestrator from
  "cheating" by teleporting an actor into the ego.

### 1.2 The per-tick loop (query → choose → commit)

```
orchestrate(world, ego_pred):
    # 1. QUERY: each actor returns its costed candidate actions for this tick
    cand[i] = actor_i.candidates(obs_i)          # task 2: [(action, cost, rollout_stub)]

    # 2. SCORE candidate heroes against the ego's PONE window
    #    (reuse d1_options geometry: conflict_P, occ_window, arc_to_P)
    for i, a in cand:
        P_i        = conflict_P(a.rollout, ego_pred)          # crossing point
        pone_i     = pone_window(ego_pred, P_i, b_comf)       # NEW: unavoidability interval
        occ_i      = occ_window(a.rollout, P_i, r_i)          # when i occupies P
        score_i(a) = w_miss·MISS + w_geom·GEOM + w_time·gap(occ_i, pone_i) + w_act·a.cost

    # 3. CHOOSE: assignment, not a single pick
    hero, a*   = argmin_i,a score_i(a)                        # primary impactor
    blockers   = greedy_min_cost actions that clear D3 interferers of hero
    escapes    = for each ego escape corridor not covered by hero,
                 assign the cheapest secondary actor to cover it (see 1.3)
    plan       = {hero: a*} ∪ blockers ∪ escapes

    # 4. COMMIT the joint plan via existing edits, re-simulate, persist decision node
    apply_plan(plan)                                          # retime/reroute per actor
```

This is the same shape as `Drive.orchestrate()` today, with three changes: the pick
becomes a **scored assignment over all actors** rather than one pursuer; the score
uses the **PONE window** rather than the raw midpoint; and it adds **escape-corridor
coverage** so a maneuvering ego cannot slip the trap.

### 1.3 Why this reliably wins (escape-corridor coverage)

The ego at any instant has a small, enumerable set of escapes: *proceed* (clear `P`
before the impactor), *stop short* (brake before `P`), and — with steering —
*divert* left/right. A single pursuer can deny at most one. The robust construction
is a **pursuit assignment**:

- **Primary hero** on the ego's dominant crossing arm, timed into the PONE window so
  *proceed* and *stop-short* are both losing (arrive-together if ego proceeds; the
  hero's own following/gap logic lets it creep into `P` if the ego stops — the
  "yielding blocker becomes the collision" case from v3/DESIGN S8).
- **Secondary actor(s)** from the other crossing arm (`WS` if hero is on `EN`)
  assigned to cover the diversion corridor, so a steering escape drives into the
  second impactor. On a 4-arm intersection two well-placed crossers cover the plane.

Formally this is a **pursuit-evasion capture** on path coordinates (v3/DESIGN P12):
because actors ride fixed paths and both players are effectively bang-bang under
accel/jerk limits, "can the ego escape all assigned pursuers?" reduces to interval
arithmetic on arrival/clearance times — the capture tube. When the tube is non-empty
the assignment is a *certificate* the ego cannot escape; when it is empty from the
current state the orchestrator reports `INFEASIBLE: ego escapes` (a real verdict, not
a silent miss) and the scenario is flagged rather than scored as a loss. This is the
single most important robustness change: **the algorithm knows when a win is
guaranteed and only claims it then.**

### 1.4 Real-time variant

The current loop is already near real-time via `verify=False` in `d1_options` and
`SWEEP_STRIDE = 4`. To keep the richer objective real-time:

- **Two-rate scheduling.** Run the full scored assignment at ~10 Hz (`ORCH_EVERY`),
  but between assignments run only a cheap **pursuer tracker**: hold the committed
  hero and re-solve its single speed against the freshly measured ego window (one
  `occ_window` + one `arc_to_P`, no scenario copy — the `__keep__` path is already
  pure geometry on cached trajectories).
- **PONE window is closed-form** (a double-integrator bound), so `TIMING` costs O(1).
- **Candidate pruning.** Only crossing-arm red-runners are hero candidates; cap the
  per-actor action set to {keep, brake-to-wait, accel-to-catch, ±one reroute}. This
  is a handful of rollouts per tick, matching v3/DESIGN's "<200 rollouts/decision"
  target, comfortably real-time in pure Python on the demo map.
- **Vectorize the sweep.** The body-overlap sweep in `collide_time`/`occ_window` is
  the hot loop; a numpy batch over frames (all candidates at once) is the documented
  fallback if profiling demands it, single-file friendly.

### 1.5 Test protocol ("the ego always loses")

Reuse and extend `drive.py --headless`, which already records ego commands and can
auto-drive:

1. **Adversarial ego policies, not just the default auto-drive.** Add scripted
   evaders as `policy(self, t)` callbacks: constant-cruise, hard-brake-on-threat,
   full-throttle-bolt, swerve-left/right, and a reactive TTC-braking driver (a
   defensive baseline). These are the ego policies task 3 needs anyway.
2. **Seed sweep.** Run every ego policy across, say, seeds 0–199 headless. Record
   for each rollout: collided? contact angle `θ`; T-bone iff `θ∈[60°,120°]∪[150°,180°]`;
   ticks-to-collision; whether the capture tube was non-empty at commit.
3. **Pass criteria.** (a) *Win rate* = fraction with a collision, target 100% on
   seeds the tube certifies as capturable and correctly `INFEASIBLE` (no false claim)
   on the rest; (b) *T-bone rate* among wins; (c) no `illegal` actor states in the
   winning rollout (task 2). Emit a CSV + a summary table; regressions are visible
   per seed.
4. **Golden replays.** Persist the `commands.json` of a few decisive rollouts and
   the session provenance so a fixed ego trace re-collides after code changes — a
   cheap regression suite over the orchestrator.

---

## 2. Actor modelling execution layer (task 2)

**Goal.** Per orchestration call, each actor takes an actor-centric observation and
returns a set of *candidate actions with costs*; actors must not overlap each other,
must stay on-road, and must maintain a desired gap. Also compute the high-level
route(s) — the lane centerlines — for the map. Optionally very fast.

The current actors are constant-speed maneuver scripts (`build_route_maneuvers`) with
**no interaction**: they can overlap, drift off geometry after a retime, and never
keep a gap. v3/DESIGN.md already specifies the fix in full (IDM + pure-pursuit over
masks and gap acceptance); this section makes it concrete as an *execution layer* and
maps it onto the existing simulate loop.

### 2.1 Lane centerlines / route graph (compute once per map)

The map is a 4-arm intersection (`lane_width`, `arm_length`; arms N/S/E/W). Build a
**route library** once:

- For each inbound leg `∈ {SE, EN, WS, NW}` and each `turn ∈ {straight, left, right}`,
  the centerline polyline = inbound-lane centerline → tangent fillet arc through the
  box → outbound-lane centerline, using the exact geometry already in
  `mv.build_route_maneuvers` (`_dist_to_intersection`, the `r=6` fillet). Wrap each as
  a `dv.Path` (arc-length param, `pos_at`, `nearest` are already implemented in
  `v2/directives.py`).
- Cache `{(leg, turn): Path}` on the map. This is the actor's **plan** (v3/DESIGN
  §1.2) and the substrate for on-road checks, conflict points, and gap measurement.
  (`Path` already provides `pos_at`, `heading_at`, `project`, `min_dist_to`.)
- Conflict points between every pair of routes are precomputed with `dv.first_crossing`
  (already used by `conflict_P`). This is the data the orchestrator's `P_i` reads.

Twelve routes, computed once — negligible cost, and it directly supplies "the high-
level route(s) of the actor for a map."

### 2.2 Actor-centric observation

A fixed egocentric feature vector (v3/DESIGN §1.4 `Obs`), cheap to fill from the
previous tick's world snapshot (the sim already passes `snap`):

```
Obs_i = ( v_i,                          # own speed
          v_des_i,                      # desired speed (param)
          s_to_stopline,                # arclength to own stop line (from Path)
          lead_gap, lead_closing,       # nearest actor ahead in own corridor (or ∞)
          conflict_gaps[],              # per conflict point on the plan: gap to crossing traffic
          kappa_ahead,                  # path curvature ahead (for speed-in-turn)
          go_flag )                     # gap-acceptance latch
```

Corridor membership = lateral offset from the actor's `Path` below a threshold; lead
selection = smallest positive along-path delta among actors in the corridor. All from
`Path.project` / `Path.min_dist_to`, already implemented.

### 2.3 Candidate actions with costs

Each actor exposes `candidates(Obs) -> [(action, cost)]`. Actions are the intervention
vocabulary the orchestrator already speaks, so nothing downstream changes:

| Action | Meaning | Cost (reuse `Params`) |
|---|---|---|
| `keep` | hold current `v_des`/plan | 0 |
| `retime(v')` | new target speed along the plan | `w_v·|v' − v|` |
| `reroute(turn)` | replan from current pose (uncommitted only) | `w_r` |
| `brake_to_wait` | `retime(0)` at the yield/stop line | `w_v·v` |
| `accel_to_catch` | `retime(v_max)` | `w_v·(v_max − v)` |

The **cost is the orchestrator's query answer**; the candidate speeds come from the
same `v' = d/t_mid` geometry (`d1_options`) plus the PONE-window target of §1.1. The
key upgrade over today: the *rollout stub* returned with each candidate is produced by
the interaction-aware motion layer (§2.4), so the cost reflects what the actor's own
braking/following will actually do — closing v2's constant-velocity gap
(`f3910e3 "use maneuvers instead of constant velocity"` was the first step; this is
the completion).

### 2.4 Motion layer: no overlap, on-road, gap-keeping (fast, procedural)

Drop the constant-speed integrator's role of "final motion" into the v3/DESIGN P3a
**surrogate**, behind the `MotionPolicy` interface so a learned policy can replace it
later without touching the directive layer:

- **Longitudinal: IDM** against the tightest active constraint (lead vehicle, stop
  point, or free road). This gives **gap maintenance** (`T_headway`, `s0`) and
  **no rear-overlap** for free — an actor physically cannot close its headway below
  `s0`. Interactions are the source of the emergent `blocks`/yield behavior the
  directive family already looks for.
- **Lateral: pure-pursuit** tracking the actor's `Path`. This keeps actors **on-road
  by construction** — motion is generated *along the centerline*, so a retime changes
  speed only, never geometry (fixing the "drift off after retime" failure). The
  road/off-road check becomes a cheap assertion (lateral offset ≤ lane half-width),
  used as the `illegal_i` penalty in §1.1.
- **Cross-actor overlap** (not just rear) is prevented by adding the gap-acceptance
  rule at conflict points: an actor yields (`go = gap > gap_accept`) unless its mask
  says otherwise. The hero (`signals: off`) runs its red but *still brakes for a car
  in its own lane* — the compositional violation v2 could not express.
- **Limits:** `|a| ≤ a_max`, jerk `≤ 5 m/s³` (v3/DESIGN), so rollouts are smooth and
  the bang-bang capture-tube analysis of §1.3 is valid.

**Integration with the existing simulator.** `v2/scenario_editor.py` already supports
a per-actor `Function` segment evaluated against the *previous* tick's world snapshot
(order-independent, `simulate()` at `DT = 1/60`). The execution layer is exactly a
built-in `MotionPolicy` occupying that reactive slot — so it plugs into the existing
`_step`/`simulate` loop with no new engine. Determinism (v3/DESIGN P4) keeps rollouts
reproducible for the tests in §1.5.

**Speed.** The whole layer is closed-form per tick (IDM + pure-pursuit are a few flops);
"very fast" is met without vectorization. If the seed sweep needs it, batch the
rollout across actors in numpy (state arrays, one update kernel) — still single-file.

### 2.5 Test protocol ("actors behave properly")

Assertions over headless rollouts across the seed sweep: (1) **no overlap** — no two
actor bodies intersect at any frame (body-sweep, `SWEEP_STRIDE=1` in test); (2)
**on-road** — every actor's lateral offset from its `Path` ≤ lane half-width + margin
for all frames; (3) **gap** — minimum time-headway to lead ≥ a floor (e.g. 0.5·
`T_headway`) except during the intended (orchestrated) collision; (4) **liveness** —
uncommitted actors reach their goal leg absent intervention. Report per-seed pass/fail;
these become CI assertions.

### 2.6 Substrate recommendation (v3 agentic vs. v4 maneuver-script)

**Recommendation: realize the v3 agentic execution layer, but incrementally, behind
the interfaces v4 already uses — do not fork the repo.**

Reasoning. Task 2's three defects (overlap, off-road, no gap) are *definitional
consequences* of the constant-speed maneuver model: it has no interaction term, so no
amount of patching on top yields gap-keeping without effectively reimplementing IDM.
v3/DESIGN already worked the design out completely (IDM + pure-pursuit + masks + gap
acceptance + the capture tube that also underwrites task 1's robustness). Extending v4
would mean building that same machinery ad hoc and *still* lacking the tube that makes
the orchestrator provably win. So the substrate should be v3's actor model.

The cost of "realize v3" is mitigated because the seams already exist:

- The **directive layer is unchanged** — `evaluate`/`repair`/`d1_options` keep their
  signatures; only the atoms re-ground on rollouts (v3/DESIGN §5).
- The **simulator is unchanged** — the surrogate policy is a `Function`-slot reactive
  segment in the current `simulate()` loop.
- v3/DESIGN ships an **adapter** that imports v4/v2 files (synthesizing goals/masks),
  and an **S10 regression** where all-override puppets reproduce v2 exactly — so v4
  scenarios and the existing sessions keep working, and the migration is verifiable
  step by step.

Concretely: keep v4 running as the regression baseline, implement v3's actor model +
rollout engine as the new execution layer, route `drive.py`'s orchestrator through it,
and retire the constant-speed path only once the S10 regression and the §2.5 behavior
tests pass. This gets task 2's correctness *and* task 1's guarantee from one body of
work, rather than two partial ones.

---

## 3. Experiments (task 3)

**Goal.** Stand up the experiment harness: integrate baselines, implement ego
policies, make methods portable to a common environment/configuration, and set up ~4
additional SafeBench scenarios.

### 3.1 Common environment & configuration (do this first)

Everything else depends on a single environment contract, so factor it out before
integrating anything:

- **`ScenarioEnv`** — a thin gym-like façade over the existing rollout: `reset(config)
  → obs`, `step(ego_action) → obs, done, info`, where `info` carries the D1/D2/D3
  verdict, collision/contact-angle, and the capture-tube certificate. `drive.py`'s loop
  becomes one consumer of this env (interactive), the headless sweep another. This is
  the "portable to a common environment" deliverable.
- **One config schema** (YAML, extends the v3 state schema): `map`, `signals`, ego
  spawn + policy, background-actor spawns/goals/masks/params, orchestrator method +
  weights (`w_miss…`, `w_v`, `w_r`), seed, horizon. Every method and scenario is a
  config; results are reproducible from `(config, seed)`. This subsumes the ad-hoc
  constants scattered in `drive.py` (`VIEW`, `V_MAX`, `SIGNALS`, spawn ranges).
- **Runner + metrics.** A `run(config) → record` that logs collision rate, T-bone
  rate, contact angle, min-TTC, time-to-collision, intervention cost, tube feasibility,
  and behavior-legality (§2.5) — one row per `(method, scenario, ego_policy, seed)`.

### 3.2 Ego policies (the evaluee)

Implement a ladder from trivial to defensive, all behind one `EgoPolicy.act(obs) →
(throttle, steer)` interface (reuse `drive.py`'s bicycle model):

- **Scripted** — cruise, hard-brake-on-threat, bolt, swerve (the §1.5 evaders).
- **Rule-based defensive** — TTC/IDM braking + lane-keeping: the honest "safe driver"
  baseline the orchestrator must still defeat when the tube says it can.
- **Learned slot** — the same `MotionPolicy`/`EgoPolicy` interface accepts a trained
  net (behavior-cloned or RL) later; ship the interface now, a numpy MLP stub as proof
  (v3/DESIGN Q1), full training out of scope for setup.

Ego defensiveness is the crux (v3/DESIGN P5/Q4): a defensive ego makes "the ego loses"
a *real* claim, so the defensive rule-based policy is the primary evaluee, scripted
ones are stress corners.

### 3.3 Baseline orchestration methods (what we compare against)

Integrate baselines behind an `Orchestrator.plan(world, ego_pred) → joint_plan`
interface so all are swappable in a config:

- **No-op / nominal** — background traffic runs its own logic, no orchestration (lower
  bound on collision rate; sanity check that scenarios aren't trivially deadly).
- **v2/v4 repairer** (`ds.repair`) — the current minimal-causal-repair kernel, as the
  in-repo baseline.
- **Single-pursuer** (today's `drive.py` `orchestrate`) — the greedy heuristic, to
  quantify the lift from the §1 assignment + PONE objective.
- **SafeBench-style learning-to-collide baselines** — the standard SafeBench
  adversarial-scenario generators (e.g. learning-based / adversarial-perturbation /
  optimization-based scenario search). Integrate them as `Orchestrator` implementations
  operating on the *same* `ScenarioEnv`, so the comparison is apples-to-apples. (These
  come from the SafeBench platform; see §3.5 note on verifying the current scenario set
  and baseline list before locking the suite.)
- **Ours** — the §1 scored-assignment orchestrator with the capture-tube certificate.

Comparison metrics: collision/T-bone rate, unavoidability (tube-certified fraction),
intervention cost (realism — cheaper = less contrived), and behavior-legality.

### 3.4 The ~four additional SafeBench scenarios

The repo currently implements one family (signalized-intersection red-light T-bone).
SafeBench's routing-level scenarios are defined by the safety-critical *interaction
type*; port four more that the intersection substrate + agentic actors already support,
each as a config with its own hero-selection and conflict geometry:

1. **Unprotected left turn across oncoming** — ego turns left, adversary comes straight
   on the opposing arm; hero = oncoming through-vehicle; conflict at the box. Exercises
   gap-acceptance and the PONE window on a turning ego.
2. **Crossing with occluded/late-appearing vehicle** — adversary enters from a crossing
   arm with a delayed/occluded reveal (spawn timing as the perturbation), forcing the
   late-detection response; tests unavoidability under reduced ego reaction distance.
3. **Lead-vehicle sudden brake / cut-in (car-following)** — adversary ahead in the
   ego's lane brakes hard or cuts in; hero = lead; the rear-end/longitudinal case,
   directly exercising the IDM gap layer and a non-T-bone contact geometry (a useful
   negative for the `GEOM` term).
4. **Right-turn conflict / merging** — adversary and ego contend for the same outbound
   lane on a turn; a merging-gap conflict rather than a crossing one.

(A fifth natural one: pedestrian/VRU crossing — deferred unless the map gains a
crosswalk/VRU actor type, since it needs a non-vehicle body model.)

Each scenario reuses the same env, actor layer, ego policies, and metrics; only the
spawn config, signal phase, and hero-selection predicate change. This is the payoff of
§3.1: a new scenario is a new YAML, not new code.

### 3.5 One verification step before locking the suite

SafeBench's exact current scenario taxonomy, the canonical baseline list, and the
routing/perception-split conventions have evolved across releases. Before freezing
§3.3–§3.4, confirm the four scenarios and the baseline set against the current
SafeBench release/paper so terminology and metrics line up with what reviewers expect.
I can pull that and reconcile it with this plan on request.

---

## 4. Phased plan

1. **Env + config + metrics (§3.1).** Factor `ScenarioEnv` out of `drive.py`; one YAML
   schema; the runner. Nothing else is portable without this.
2. **Actor execution layer (§2).** Route library/centerlines; `Obs`; IDM + pure-pursuit
   surrogate behind `MotionPolicy`; behavior tests (§2.5). Fixes task 2; unblocks the
   defensive ego and the capture tube.
3. **Orchestrator (§1).** PONE window; scored assignment + escape-corridor coverage;
   capture-tube certificate + `INFEASIBLE: ego escapes`; two-rate real-time scheduling.
   Test via the seed sweep (§1.5). Fixes task 1.
4. **Experiments (§3.2–§3.4).** Ego-policy ladder; baseline integration (incl.
   SafeBench methods); the four extra scenarios; the comparison tables.
5. **Verification (§3.5 + throughout).** S10 v2-regression, behavior assertions, golden
   replays, and the SafeBench reconciliation.

The through-line: one env, one actor model, one orchestrator objective, evaluated the
same way across scenarios — so each task's deliverable is reused by the next rather
than rebuilt.
