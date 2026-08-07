# Scenario Editor v3 — Design: Directives over Agentic Actors (draft for review)

v3 replaces v2's kinematic actor model with **agentic actors**: each actor has its
own internal logic — goals and a planned route, tactical rules (follow, yield, obey
signals) that can be *individually violated*, and a motion layer designed as a slot
for a **deep policy**. The v2 directive layer (the red-light family D1–D3, the modal
DSL, causal minimal intervention) survives with its meaning intact; what is rebuilt
is everything underneath it: prediction, the branch set behind ◇, the intervention
vocabulary, and recognition.

Deliverable: `v3/directives.py` — headless, import-safe, single file, `pyyaml` only
(P-scope, per review). Decisions are marked **P#**; §11 lists the items adopted on
recommendation and still open for review.

---

## 0. What v2 assumed and v3 replaces

| | v2 | v3 |
|---|---|---|
| Actor | pose + speed | pose + speed + **goal, plan, parameters, violation masks** |
| Behavior | constant velocity along recognized route | closed-loop policy: tactical rules → motion policy |
| Prediction | closed-form interval algebra | fixed-step rollout of all policies |
| ◇ branch set | hero's route × target speed | **intervention-reachable** evolutions (§4) |
| Intervention | `retime` / `reroute` (direct control) | nudge → mask flip → override ladder (§6), effects *mediated by the actor's own logic* |
| Recognition | guesses lane, route, role | goals/plans are **state**; recognition computes only derived quantities (§7) |
| Interference | detected by formula (`blocks`, car-following bound) | **emergent** — background actors avoid collisions themselves; detected on the rollout |

The deepest change is the last column of the intervention row: v2 could *set* a
speed; v3 can only *ask* — change what an actor wants, what it ignores, or (at high
cost) take the wheel — and the effect on the trajectory is whatever the actor's
logic does with it. Intervention search therefore becomes search over re-rollouts
(§6), seeded by v2's window algebra where it still applies.

## 1. Actor model

### 1.1 Actor state (P1 — goals and plans are state, not inference)

Per review: the scenario state carries each actor's internals as ground truth. The
orchestrator authors the world; it does not have to guess it.

```yaml
map:     {lane_width: 3.5, arm_length: 60}
signals: {N: green, S: green, E: red, W: red}     # static over the horizon (P8)
ego: 0
actors:
  - id: 0
    pose:   {x: 1.75, y: -30.0, heading: 90, speed: 12.0}
    dims:   {length: 4.5, width: 2.0}
    goal:   {turn: straight}                       # left | straight | right
    params: {v_des: 12.0, T_headway: 1.5, s0: 2.0, a_max: 2.5, b_comf: 3.0,
             gap_accept: 4.0}                      # defaults; all optional
    masks:  {signals: on, follow: on, gap: on}     # on = rule active (compliant)
  - id: 1
    pose:   {x: 40.0, y: 1.75, heading: 180, speed: 13.0}
    goal:   {turn: straight}
    masks:  {signals: off, follow: on, gap: on}    # a red-light runner that still
                                                   # brakes for cars ahead
```

The **plan** — the concrete path polyline through the intersection — is derived from
`goal` + current pose using v2's route geometry (inbound leg → tangent fillet →
outbound leg) and cached on the actor; it is *the actor's own plan*, not a
recognizer's fit. A goal change (authoring edit or intervention) triggers a replan
from the current pose, legal only while **uncommitted** (before the plan-divergence
point at the stop line — v2's commitment rule, unchanged).

### 1.2 Layer 1 — goals and route planning

`goal.turn` selects the outbound leg under right-hand traffic; the planner emits the
path and the arc-length metadata the tactical layer needs (distance to stop line,
to each conflict point, to exit). Actors that have exited hold v2's straight-to-map-
edge continuation.

### 1.3 Layer 2 — tactical rules, gated by violation masks (P2)

Three rule modules, each computing a **constraint**, each gated by its mask; a
masked rule emits nothing. The layer's output is a `TacticalCmd`:

```
TacticalCmd = (path, v_target, stop_s | None, go: bool)
```

| Rule | Mask | Constraint emitted |
|---|---|---|
| **Signal compliance** | `signals` | if own approach phase ∈ {red} and not past stop line: `stop_s` = stop line |
| **Car-following** | `follow` | nearest lead vehicle in the plan corridor → gap/closing-speed pair passed to the motion layer (IDM-style interaction term); "slow down to avoid actors in front" |
| **Gap acceptance** | `gap` | at each conflict point where this actor must yield (turning across oncoming traffic; entering on a permissive phase): compute the gap to conflicting traffic; `go = gap > gap_accept`; while `¬go`, `stop_s` = yield line. "Speed up if there is a safe gap" is the motion layer recovering `v_des` the instant `go` flips true |

Constraints compose by tightness: effective stop point = nearest active `stop_s`;
`v_target = v_des` always (rules bound, they don't retarget). Masks make violation
*compositional*: v3's canonical hero has `signals: off` with `follow`/`gap` still
`on` — it runs the red **and** still brakes behind a slow lead, which v2's
constant-velocity hero could not express (and which creates the new demo states of
§9).

### 1.4 Layer 3 — motion policy: the deep slot (P3)

The motion layer is a fixed interface; "realistic motion ultimately comes from a
deep policy" is honored by making the *contract* the design object:

```python
class MotionPolicy(Protocol):
    def act(self, obs: Obs, cmd: TacticalCmd) -> Control   # Control(accel, kappa)
```

`Obs` is a fixed, egocentric feature vector — own speed; `v_des`; distance to
effective stop point; lead gap and closing speed (or sentinels); path curvature
ahead; `go` flag — the same vector whether the consumer is the shipped surrogate or
a trained network.

**P3a — v3 ships a procedural surrogate behind the interface** (adopted on
recommendation, §11): IDM longitudinal control against the effective constraint
(lead vehicle, stop point, or free road), pure-pursuit lateral tracking of the plan,
accel and jerk limits (`|a| ≤ a_max`, `|ȧ| ≤ 5 m/s³`). A learned policy drops into
the slot later without touching the directive layer.

**P4 — policies are deterministic in v3.** Rollouts are reproducible; evaluation
is a function of state. Stochastic policies (temperature > 0) are a noted
extension — evaluation would become quantile/CVaR over seeds, and `Eval.confidence`
regains a probabilistic meaning.

### 1.5 The ego is an agent too (P5)

The ego runs the same three-layer logic (masks all `on`, never intervened on). This
has a consequence worth stating loudly: **a defensive ego tries to avoid D1's
collision.** In v2 the ego rolled forward blindly; in v3 the ego's car-following
will brake when the hero cuts across, so D1 = "some vehicle should collide with the
ego" is only satisfiable when the geometry and timing *defeat the ego's avoidance* —
the hero must enter the conflict point inside the ego's point-of-no-escape window
(`d_ego < v_ego²/(2·b_comf)` plus reaction). This is a feature, not a bug: a
scenario satisfying v3-D1 is a *genuinely critical* scenario, and the intervention
search (§6) is now searching for exactly the timing that makes the collision
unavoidable. An inattentive-ego profile (ego masks configured `off` in the state)
remains available as an *authoring* choice; it is never an intervention.

## 2. Simulation & prediction (rollout engine)

**P6 — fixed-step synchronous rollout.** `dt_sim = 0.1 s`; each tick, every actor
evaluates tactical rules and its motion policy against the *previous* tick's world
(v1's convention — order-independent), then integrates (bicycle/unicycle update on
`(accel, kappa)`). Horizon `H = 15 s`. The **nominal evolution ξ\*** is the rollout
of all actors' policies from the given state with no interventions. Temporal
operators sample ξ\* at `dt_eval = 0.25 s`, as in v2.

Rollouts are the unit of cost. They are memoized by (state-hash, intervention-set);
a full evaluation of the family is a handful of rollouts, an intervention search a
few dozen (budget: designed to stay under ~2000 rollouts ≈ seconds in pure Python
for the demo map; §11-Q5).

## 3. Logic: the DSL under rollouts

The v2 AST (`Atom, Not, And, Or, F, G, Dia, Exists`) is reused unchanged in form.
Two semantic upgrades:

**P7 — atoms return signed robustness margins**, not just booleans:
`ρ(collide) = r_conflict − min separation` (positive inside), `ρ(runs_red)` =
stop-line overshoot at red, etc. Connectives take min/max (STL-style), `F`/`G` take
max/min over samples, `Exists` max over bindings, `◇` max over the sampled branch
set. Verdicts remain `ρ > 0`; the margins exist to *rank witnesses and drive the
intervention search* — with closed forms gone, ρ is what bisection optimizes.
`Eval(value, ρ, witness, t_fail, explanation)` extends v2's result type
(`confidence` is retired with the recognizer, §7).

**P8 — signal phases are static over the horizon**, as in v2. Compliant actors
facing red simply wait — which is realistic, and the source of the yield-deadlock
interference in §9. Phase schedules (fixed-time cycling) are a schema-compatible
extension: `signals: {E: {phase: red, t_green: 8.0}}`.

## 4. ◇: intervention-reachability (P9, adopted on recommendation)

In v2, ◇ ranged over the hero's own admissible controls. In v3 an actor's "own
choices" and "what the orchestrator can cause" have come apart; ◇ follows the
orchestrator, preserving the property that made D2 load-bearing:

> `◇φ` holds at `(ξ, t)` iff some **admissible intervention** ι applied from the
> concrete state reached at time t makes φ hold on the re-rolled evolution.

For D2 the intervention set is **hero-scoped** (v2-Q3's analogue): nudges, mask
flips, or override targeting the hero only. So D2 = `G[≤t*] ◇ F[≤H] collide(hero,
ego)` again reads: *at every instant, the orchestrator can still cause the
collision* — D1 remains causally repairable, and D2 going tight is the actionable
early warning. A behavioral-uncertainty modality (◇ᵇ over plausible parameter
perturbations — "what might they do on their own") is deferred; §11-Q2.

**P12 — D2's ◇ is evaluated as membership in a backward reachable tube, not by
per-sample search.** With jerk-limited controls and route choice, `◇ F collide` has
no one-line test — v2's `d_h/v_max ≤ e₁` was the constant-velocity degenerate case.
Instead, once per (hero-candidate, route), compute the **capture tube**: the set of
hero states from which some admissible control forces collision with the ego
*despite the ego's avoidance* — a pursuit-evasion capture basin, because the v3 ego
defends itself (P5). Then:

- **D2 = tube membership along ξ\*.** Sample the nominal trajectory and test
  whether the hero is still inside the tube; `t_fail` = first exit. The tube is
  computed once per evaluation — the `G[≤t*] ◇` nesting no longer re-solves a
  reachability problem at every sample.
- **Structure keeps it semi-analytic.** Actors ride fixed paths, so the game
  reduces to path coordinates — hero `(s_h, v_h)` vs ego `(s_e, v_e)`, collision ⟺
  occupancies of P overlap — and both players' extremal strategies are **bang-bang**
  (max-brake / max-accel under `a_max`, jerk limit). The tube boundary is therefore
  interval arithmetic on jerk-limited arrival/clearance times: v2's window algebra
  with double-integrator bounds replacing constant speed, not a dense HJ grid solve.
  Route choice while uncommitted = a finite union of per-route tubes. A discretized
  backward BFS over the 4D path-coordinate grid is the documented fallback if
  richer dynamics (true off-path lane changes) ever break this monotone structure;
  it is not in v3 core.
- **What the tube does and doesn't answer.** The ego plays its comfort envelope
  (`b_comf`, `a_max`) as the evader, so tube membership is *robust*-sufficient for
  ◇ at the override rung (collision forcible whatever the ego does within its
  envelope) and near-tight against the surrogate ego, which brakes at that
  envelope. For rungs 1–2 the mediated dynamics reach only a subset of the
  override's, so the tube is a *necessary* condition — it prunes; re-rollouts
  confirm. A tube exit along ξ\* is itself confirmed by a small rollout grid before
  D2 is reported failed (no false alarms from envelope conservatism).
- **The tube is policy-free.** It depends on dynamic limits only, never on the
  motion policy — it survives any drop-in network unchanged (P3), which is exactly
  the division of labor: tubes answer *what is forcible*, rollouts answer *what the
  logic actually does*. The point-of-no-escape window of P5/S9 is the tube boundary
  read in ego coordinates.

## 5. The family, re-grounded

D1–D3 are textually identical to v2 (§0 there); their atoms re-ground on rollouts:

- `collide(a, b)` — oriented-rectangle overlap, now checked on ξ\* samples (the
  closed-form occupancy-interval test survives as the *seed* for search, §6).
- `runs_red(v)` — v's approach is red **∧** v's rollout crosses the stop line during
  red. Note this is now a *behavioral* fact: it is true of an actor whose `signals`
  mask is off (or whose brake cannot stop it in time), rather than an assumption
  about a trajectory.
- `interferes(w, hero, ego)` — same derived atom (`F collide(w,·) ∨ blocks ∨
  occupies_conflict`), but `blocks` is now read off the rollout: w blocks the hero
  iff the hero's *own car-following* holds it below the speed D1 needs (the
  constant-speed bound of v2-§5 is gone; the actor logic computes the truth).
  Interference by *yielding* appears for free: a compliant turner waiting in the
  hero's lane is a `blocks` witness (§9-S8).

`hero` binding: still D1's existential witness; candidates ranked by
intervention cost-to-collidability (as v2), computed with the §6 seed.

## 6. Interventions: the ladder (P10, adopted on recommendation)

All interventions are **causal** (v2-Q5 verbatim): forward-in-time edits from the
evaluation instant, never to history, never to the ego. What changed is that edits
now target the actor's *logic inputs*, and their effect is mediated — hence found by
search over re-rollouts, not solved in closed form.

| Rung | Intervention | Semantics | Cost |
|---|---|---|---|
| 1 | `nudge_speed(v, v_des')` | change desired speed; the policy tracks it subject to its own rules | `1·\|v_des'−v_des\|` per m/s |
| 1 | `nudge_gap(v, g')` | change gap-acceptance threshold (patience) | `2` |
| 1 | `regoal(v, turn')` | change goal → replan; only while uncommitted | `5` |
| 2 | `flip_mask(v, rule)` | toggle one violation mask (make a violator, or make one comply) | `3` per rule |
| 3 | `override(v, profile)` | seize direct control: actor becomes a v2-style kinematic puppet on a commanded speed profile; its logic is suspended | `20 + 1·\|Δv\|` |

Rung 1 keeps every actor in-character; rung 2 changes character; rung 3 abandons
the actor model (and makes v2 a degenerate case of v3 — an all-override
intervention set with trivial policies reproduces v2 exactly, which is the
regression baseline for tests).

**P11 — repair = tube projection as seed + rollout refinement + joint re-check.**
The tube (P12) hands the D1/D2 repair its geometry: the minimal *override* repair
is literally the **projection of the hero's state onto the tube** — minimality free
from the geometry — and nudge seeds are read off the tube boundary in `v_des`
coordinates (the speed that re-enters the tube, ignoring interactions), then
refined by bisection on true re-rolled robustness (the actor's own braking,
following, and jerk limits bend the answer away from the seed). D3 repairs, which
target non-hero interferers outside the tube's scope, seed from v2-style window
algebra on the rollout-derived occupancies. Per-directive
repairs then a joint re-evaluation, ≤ 3 passes, exactly v2-P6's loop. Every repair
verifies mask side effects — e.g. flipping the hero's `follow` mask to fix a slow
lead (D3) may just cause the *wrong* collision (hero rear-ends the lead before
reaching ego); the re-roll catches this, the ranking discards it. `INFEASIBLE`
reporting gains a new reason alongside v2's: **"ego escapes under all admissible
interventions"** (the defensive ego of P5 cannot be trapped from this state).

## 7. Recognition, reduced

With goals, plans, parameters, and masks in the state (P1), the v2 recognizer's
inference duties disappear. What remains is a slim derived layer computed per
rollout sample: region classification (approach / box / exited), arc-length
positions along plans, conflict points between plan pairs, commitment status,
lead-vehicle relations. No confidences; a derived fact is exact given the state.
The one place inference survives: the **adapter** (§8) imports legacy files that
lack internals and synthesizes them — those synthesized fields are flagged as
assumptions in the report, exactly as v2 flagged assumed signals.

## 8. Inputs, adapter, CLI

`python3 v3/directives.py --demo` — built-in states. `--state FILE.yaml [--time T]
[--signals ...]` accepts three schemas, distinguished automatically:

- **v3 state file** (§1.1): evaluated as-is.
- **v2 state file** (pose+speed only): internals synthesized — `goal` from v2 route
  recognition (straight default), `v_des` = current speed, masks from consistency
  (an actor that must cross on red to be D1's witness gets `signals: off`); each
  synthesis flagged.
- **v0/v1 scenario file**: `state_from_scenario(path, T)` samples the script, then
  the v2-file synthesis path. A v1 function-graph actor (e.g. the
  `v_hero = v_ego·d_h/d_e` controller) is imported as its sampled pose+speed;
  reproducing it as a *policy* is out of scope (function graphs as custom policies:
  §11-Q6).

Report format follows v2: state summary (goals, masks, any synthesized fields),
per-directive `Eval` with ρ and witness, interventions with costs and the
post-intervention all-green re-evaluation.

## 9. Demonstration states

v2's S1–S6 carry over (their expected verdicts unchanged — with masks set so the
hero is a red-light runner). New states exercise what only agentic actors express:

| | State | Expected |
|---|---|---|
| S7 | hero (`signals: off, follow: on`) behind a slow lead on `EN` | D1 ✗ — hero's *own logic* brakes it out of ego's window → repair compares `nudge_speed(lead)` vs `flip_mask(hero, follow)`; re-roll rejects the mask flip if it causes hero–lead collision first |
| S8 | compliant left-turner waiting (gap acceptance) in hero's lane | D3 ✗ (`blocks`, emergent from yielding) → `nudge_gap` so it accepts and clears, or `regoal` it |
| S9 | defensive ego: hero timed so ego can comfortably brake | D1 ✗ though paths conflict → repair retimes hero to arrive inside ego's point-of-no-escape window; if no such timing exists, `INFEASIBLE: ego escapes` |
| S10 | v2 regression: all actors overridden puppets | verdicts and repairs match v2's S1–S6 outputs |

## 10. File structure (`v3/directives.py`)

1. Map & route geometry (reused from v2, unchanged).
2. Actor model: schema, params, masks; tactical rules; `Obs`/`TacticalCmd`/
   `MotionPolicy` interface; the surrogate policy.
3. Rollout engine (P6) + memoization.
4. Derived-state layer (§7).
5. Predicate library: rollout-backed atoms with robustness (P7).
6. Capture tubes: per-(candidate, route) jerk-limited arrival-interval algebra,
   pursuit-evasion against the ego envelope (P12).
7. Logic DSL: v2 AST, ◇ re-scoped to intervention-reachability (P9), D2 via tube
   membership with rollout confirmation.
8. Family definition (unchanged text from v2 §0).
9. Intervention ladder + repair search (P10, P11: tube-projection seeds).
10. Adapter (three schemas), demo states, CLI, headless tests (incl. S10 v2
    regression).

Editor integration remains deferred (v2 §9 hooks unchanged, plus one addition:
per-actor mask toggles as authoring UI).

## 11. Open items

Adopted on recommendation, awaiting confirmation:

- **Q1 — deep policy: interface + surrogate (P3a).** Alternative: embed a tiny
  numpy MLP behavior-cloned from the surrogate to prove the slot with real weights.
- **Q2 — ◇ = intervention-reachable (P9).** The behavioral-uncertainty modality
  ◇ᵇ is deferred; if both are wanted, D2 uses ◇ⁱ and family authors may use either.
- **Q3 — ladder includes `override` (P10).** Dropping rung 3 makes the system
  purer (everything through actor logic) but INFEASIBLE more common, and loses the
  v2-regression construction (S10).

Genuinely open:

- **Q4 — ego defensiveness default (P5).** Full logic is the default here; confirm
  that D1-as-unavoidable-collision is the intended (stronger) reading, and that
  inattentive-ego profiles are authoring-only.
- **Q5 — rollout budget.** With D2 tube-based (P12), rollouts are spent only on
  ξ\*, tube-exit confirmation, and repair refinement — target well under ~200 per
  repair in pure Python; numpy vectorization is the fallback (still single-file).
- **Q6 — v1 function graphs as custom policies.** Excluded from v3 core; they
  could later implement `MotionPolicy` (graph output → `Control`), unifying v1 and
  v3. Worth a note in the editor-integration plan.
