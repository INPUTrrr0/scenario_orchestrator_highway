# Scenario Editor v2 — Design: Declarative Directive Layer (draft for review)

v2 adds a **directive layer**: scenario *families* defined by declarative modal-logic
directives, plus the machinery to (a) **abstract** a concrete scenario state, (b)
**evaluate** the directives against that state *and its predicted evolution*, and (c)
compute a **minimal intervention** when a directive is false. First family:
**red light violation**.

Deliverable: `v2/directives.py` — a headless, import-safe single file (`pyyaml` only;
no pygame). The v1 editor (`v2/scenario_editor.py`, currently an unmodified copy) is
untouched; integration hooks are listed in §9. Decisions are marked **P#**; the four
formerly-open questions were resolved as proposed (§10).

---

## 0. The family: informal → formal

```
RedLightViolation
  D1  some vehicle should collide with the ego                    (goal)
  D2  at all times, some vehicle could collide with the ego       (maintain reachability)
  D3  other vehicles would not interfere with the scenario        (non-interference)
```

The arrows in the informal spec read as a refinement chain: D1 states the goal; D2
keeps the goal *reachable at every instant* (robust to timing drift); D3 protects the
realization from third parties. Formalized in the language of §2:

```
D1:  ∃v ∈ V∖{ego} :  runs_red(v)  ∧  F[≤H]  collide(v, ego)
D2:  G[≤t*]  ◇ F[≤H]  collide(hero, ego)
D3:  G[≤t*]  ¬ ∃w ∈ V∖{ego, hero} :  interferes(w, hero, ego)
```

where `hero` is the **witness** of D1's existential (or the best candidate if D1 is
false), `H` is the prediction horizon (default 15 s), and `t*` is the planned
collision time under D1 (falling back to `H` when D1 fails). `runs_red` makes the
family a *red light* violation rather than a generic collision; without it D1–D3 are
family-agnostic (see Q1).

Note the three distinct modalities behind the English: *should* = obligation over the
**nominal** predicted evolution (F on the prediction); *could* = possibility over
**admissible alternative** evolutions (◇ over a branch set); *would* = what holds
along the evolutions that realize the scenario (evaluated on the nominal realization;
stronger reading in Q2).

## 1. Scenario state & evolution model

**Concrete state** — an instant, not a script (contrast v0/v1, where a scenario *is*
its maneuver script):

```yaml
map:     {lane_width: 3.5, arm_length: 60}     # v0 map model, 8 legs
signals: {N: green, S: green, E: red, W: red}  # per-arm inbound phase (optional, Q1)
ego: 0
actors:                                        # pose + speed only
  - {id: 0, x: 1.75, y: -30.0, heading: 90,  speed: 12.0, length: 4.5, width: 2.0}
  - {id: 1, x: 40.0, y: 1.75,  heading: 180, speed: 13.0, length: 4.5, width: 2.0}
```

An adapter `state_from_scenario(path, T)` samples a v0/v1 scenario file at clock `T`
(via the existing simulation) into this form, so directive evaluation composes with
the editor's saved scenarios.

**P1 — Evolution model M (piecewise-constant, continuous velocity; C¹ paths).**
Speed profiles along any evolution are **piecewise constant: plateaus joined by
constant-acceleration ramps** — continuous overall, with discontinuities only in
*acceleration* (instantaneous changes in `a`, i.e. unbounded jerk; `|a| ≤ a_max`,
default 3 m/s²). Speeds never jump. Actors follow their lane centerlines; paths are
polyline+arc: inbound leg → intersection → outbound leg, where turns are circular
fillets **tangent to both centerlines** — so the velocity *direction* is continuous
too. After exiting, actors continue straight to the map edge and hold there. The
**nominal evolution ξ\*** is the degenerate case: each actor holds its current speed
(a single plateau) along its *recognized* route (§4).

**Branching.** Nondeterminism is (a) the **route choice** at the intersection: an
actor on an inbound leg has routes `{left, straight, right}` (each mapping to the
correct outbound leg under right-hand traffic); and (b) for controllable vehicles,
the speed profile. The **admissible branch set Γ(x, v)** for a controllable vehicle
`v` allows re-choosing `v`'s route and steering to a **target speed
`v' ∈ [0, v_max]`** (default `v_max = 20 m/s`), realized admissibly as *one ramp at
`|a| ≤ a_max` from the current speed, then a plateau* — the modality never teleports
a speed. All other actors stay on nominal. Γ is what `◇` quantifies over; only the
hero is treated as controllable (Q3).

## 2. Formal language

**P2 — A small modal-temporal logic, embedded as a Python combinator DSL.**

```
φ ::= atom(args…)                     # grounded predicate (§3)
    | ¬φ | φ ∧ φ | φ ∨ φ
    | G[≤τ] φ | F[≤τ] φ              # temporal, along the current evolution
    | ◇ φ | □ φ                       # modal, over the branch set Γ
    | ∃ v ∈ V∖R : φ(v)                # role-quantification (R = bound roles)
```

**Semantics.** A model is a pointed structure `(x, M)`: concrete state + evolution
model. Formulas are evaluated at a *time point along an evolution*, starting at
`(ξ*, t=0)`:

- `atom` — looked up in the predicate library against the **abstract state at that
  time point** (§4 machinery re-abstracts the predicted concrete state).
- `F[≤τ] φ` / `G[≤τ] φ` — ∃/∀ over sampled time points `t' ∈ [t, t+τ]` of the
  current evolution (dt = 0.25 s), with closed forms used where available (§5).
- `◇ φ` — true at `(ξ, t)` iff φ holds at `(ξ', t)` for **some** ξ' ∈ Γ(x_t): the
  evolution is re-branched from the concrete state reached at time t. `□` is dual.
- `∃ v : φ(v)` — disjunction over concrete vehicles; the satisfying binding is
  reported as the **witness** (this is how `hero` gets bound).

In the DSL:

```python
V, W = Var("v"), Var("w")
D1 = Exists(V, RunsRed(V) & F(Collide(V, EGO), le=H))
D2 = G(Dia(F(Collide(HERO, EGO), le=H)), le=T_STAR)
D3 = G(~Exists(W, Interferes(W, HERO, EGO)), le=T_STAR)
FAMILY = Family("red_light_violation", d1=D1, d2=D2, d3=D3)
```

Evaluation returns a structured result, not a bare bool:
`Eval(value, witness, t_fail, explanation, confidence)` — `t_fail` is the first time
point at which a G-obligation broke (drives intervention), `explanation` names the
failing atom, `confidence` is the min recognition confidence of the atoms used (§4).

## 3. Predicate library (grounding)

**P3 — atoms are named functions over the abstract state**, registered in a library;
directives may only mention registered atoms (unknown atoms are a load-time error).

| Atom | Meaning (all w.r.t. abstract state at the evaluation time point) |
|---|---|
| `collide(a, b)` | oriented body rectangles of a and b overlap |
| `runs_red(v)` | v's approach signal is red ∧ v's route crosses the stop line during the prediction (i.e. v does not stop) |
| `on_leg(v, l)` / `in_intersection(v)` / `exited(v)` | region classification |
| `approaching(v)` | inbound, before its stop line |
| `paths_conflict(v, u)` | v's and u's predicted paths intersect → conflict point P(v,u) |
| `tt_conflict(v, u)` | signed time to P(v,u): occupancy interval `[(d−ℓ/2)/s, (d+ℓ/2)/s]` |
| `blocks(w, v)` | w is in v's lane corridor ahead of v with a gap/speed that caps v below the speed D1 needs |
| `occupies_conflict(w, v, u)` | w's predicted occupancy of a disc around P(v,u) overlaps the collision window |
| `interferes(w, hero, ego)` | `F[≤t*] (collide(w,ego) ∨ collide(w,hero)) ∨ blocks(w,hero) ∨ occupies_conflict(w,hero,ego)` (a derived atom) |

Collision under constant velocities has a closed form: paths conflict at P; each
vehicle occupies P during an interval; `collide` ⟺ the intervals overlap (checked
exactly, with the dense rectangle test as fallback for in-intersection geometry).

## 4. Recognition machinery (concrete → abstract)

**P4 — a `Recognizer` producing an `AbstractState`,** since directives cannot be
queried against raw poses. Per actor:

- **Lane assignment** — snap to the nearest leg centerline with heading agreement
  (≤ 30°); confidence decays with lateral offset and heading error. Region =
  approach / intersection box / exit.
- **Route estimate** — on an approach: `straight` unless the state says otherwise
  (no blinkers in this world), reported with low confidence; inside the
  intersection: fit the observed pose against the straight path and both fillet
  arcs, pick the best-fitting route with confidence from residual.
- **Role recognition** — `ego` is given by the state (`ego:` key). `hero` is not
  given: it is *recognized* as D1's witness — the evaluator tries candidates in
  order of increasing "collidability cost" (how small an intervention would make D1
  true for them), so the witness is stable and the same ranking is reused by the
  intervention search.
- **Signal phase** — read from `signals:` if present; otherwise *assumed*
  family-consistent (ego's arm green, conflicting arms red) and flagged as an
  assumption in the output (Q1).
- **Derived quantities** — conflict points between route pairs, distances-to-go
  along paths, occupancy intervals.

Abstraction is applied to the *initial* concrete state and, during temporal
evaluation, to each predicted concrete state (cheap: lane/route are propagated along
the evolution rather than re-fit, so confidence is inherited).

## 5. Evaluating the family (closed forms)

Under P1 the interesting atoms reduce to interval arithmetic on arrival times. With
ego's occupancy of P being `[e₀, e₁]` and hero at distance `d_h(t)` from P at
evaluation time t:

- **D1** — collide ⟺ hero's occupancy `[h₀, h₁]` overlaps `[e₀, e₁]`; evaluated per
  candidate v and per route of v (disjunction), nominal speeds. `t*` = midpoint of
  the overlap.
- **D2** — under ramp+plateau controls, hero's *reachable arrival times* at P from
  the state at t form an interval `[t_min(t), t_max(t)]`: `t_min` = max-accel ramp
  to `v_max` then hold; `t_max = ∞` iff hero can still stop short of P
  (`d_h ≥ v_h²/(2·a_max)`), else the max-braking arrival bound. `◇ F collide` at t
  ⟺ ego's window is not over (`e₁(t) > 0`), hero has not passed P, and
  `[t_min(t), t_max(t)] ∩ [e₀(t), e₁(t)] ≠ ∅`. Still closed-form interval
  arithmetic; D2 = G of that test, checked densely over `[0, t*]` with the first
  violation time reported.
- **D3** — per third vehicle w: closed-form window overlap for
  `occupies_conflict`, first-collision check against ego/hero, and a constant-speed
  car-following bound for `blocks` (w slower than hero's required speed with gap
  closing to < `g_min` before `t*`).

## 6. Minimal intervention (causal)

**P5 — interventions are *causal*: control edits applied from the evaluation
instant forward, never edits to the state itself.** An intervention may not change
the past — nor the present, discontinuously: every actor keeps its pose, speed, and
history; what changes is its **future control**, executed under P1 kinematics.
Never applied to the ego (Q3). Per non-ego actor:

| Intervention | Semantics | Cost |
|---|---|---|
| `retime(v, v')` | ramp at `\|a\| ≤ a_max` from current speed to target `v' ∈ [0, v_max]`, then plateau (`v' = 0` = yield/stop) | `w_v·\|v'−v\|`, w_v = 1 /(m/s) |
| `reroute(v, r)` | re-choose route — only while v is **uncommitted** (before its path diverges at the stop line) | `w_r = 5` |

Total cost = Σ over edited actors; **minimal intervention = argmin cost s.t. all
three directives evaluate true from the current state under the amended controls.**
State edits — `Δs` spawn slides, instantaneous `Δv`, add/remove actor — are *not*
interventions: they rewrite history. They remain available as **authoring edits**
in the editor at design time (t = 0, where no past exists yet), outside this
machinery.

The causal restriction is what makes D2 load-bearing: a causal `retime` repair of
D1 exists for a candidate at time t **iff D2 holds for it at t** — the repair must
place the candidate's arrival inside ego's window, and `[t_min, t_max]` (§5) is
exactly what `retime` can reach. D2 is the invariant that keeps D1 causally
repairable; that is why the family orders D1 → D2 → D3, and why an orchestrator
should act when D2 gets *tight*, not when D1 finally breaks.

**P6 — repair algorithm: analytic per-directive repair + joint re-check** (approx.
minimal; exact minimality would need a joint search — see Q4):

1. Evaluate D1–D3; if all true, return ∅.
2. **D1 repair** — per hero candidate (ranked by §4): solve the ramp+plateau
   kinematics for the target `v'` placing arrival at P nearest to the midpoint of
   ego's window (a quadratic in the ramp time; feasible iff
   `[t_min, t_max] ∩ [e₀, e₁] ≠ ∅` — exactly D2 for that candidate). Else
   `reroute` if uncommitted. Take the cheapest feasible candidate.
3. **D2 repair** — if the D2 margin is predicted to cross zero at a future t < t*,
   `retime` *now* to restore slack — acting early is cheaper than acting late,
   which is D2's whole point.
4. **D3 repair** — per interferer: the min-`|v'−v|` `retime` that moves its
   occupancy off the conflict window / opens the corridor gap (interval arithmetic
   again; includes `v' = 0`, i.e. make it yield), else `reroute` if uncommitted.
5. Re-evaluate all directives under the amended controls; iterate (≤ 3 passes).
   Report interventions, total cost, and the post-intervention evaluation. If
   infeasible — hero committed past P, no uncommitted candidate, interferer
   unclearable — report `INFEASIBLE` with the reason: **the past is not available
   for editing.**

## 7. Demonstration

`python3 v2/directives.py --demo` evaluates the built-in demo states. A single input
is given with `--state FILE.yaml [--time T] [--signals N=green,E=red,...]`, where
FILE is either format, distinguished by schema:

- **State file** (§1 schema: actors carry `x/y/heading/speed`) — evaluated as-is;
  `--time` is rejected (a state has no clock).
- **Scenario file** (v0/v1 schema: actors carry `start` + `maneuvers`) — passed
  through `state_from_scenario(path, T)`: the editor's simulation is run headlessly
  and sampled at clock `T` (default 0, wrapped mod the loop period) to yield
  poses+speeds. Scenario files carry no signal phases, so phases come from
  `--signals` or the family-consistent assumption (Q1), flagged as assumed.

For either format the tool prints, per state: the abstract-state summary (lanes, routes, roles, assumed
signals), each directive's `Eval` (value, witness, t_fail, explanation), and — where
something is false — the intervention, its cost, and the re-evaluation showing all
green. Demo states:

| | State | Expected |
|---|---|---|
| S1 | hero westbound on `EN` (red), on collision course with northbound ego | D1 ✓ D2 ✓ D3 ✓ |
| S2 | hero too slow — misses ego's window | D1 ✗ → `retime` hero |
| S3 | hero already through the intersection | D2 ✗ at t=0; hero committed → `reroute` another uncommitted candidate, else `INFEASIBLE` |
| S4 | slow lead vehicle ahead of hero on `EN` | D3 ✗ (`blocks`) → `retime` lead |
| S5 | third vehicle's path crosses P during the collision window | D3 ✗ (`occupies_conflict`) → `retime` it clear of the window |
| S6 | no vehicle on any conflicting red approach | D1 ✗, no witness → `reroute` an uncommitted vehicle, else `INFEASIBLE` |

S1 is also exercised via the adapter on `scenarios/scenario_v1.yaml` (the v1 worked
example — whose function graph `v_hero = v_ego·d_h/d_e` is, pleasingly, exactly a
controller that *maintains D2 invariantly*).

## 8. File structure (`v2/directives.py`)

1. State & map model (reuses v0 conventions; no pygame import).
2. Evolution model: routes, fillet geometry, constant-velocity rollout, Γ.
3. Recognizer → `AbstractState`.
4. Predicate library (registry of atoms + closed forms).
5. Logic DSL: AST combinators, evaluator (temporal sampling + modal branching),
   `Eval` results.
6. Family definition (§0) — the only place the red-light family is spelled out.
7. Intervention search (§6).
8. Demo states, adapter from editor scenarios, CLI (`--demo`, `--state`), tests
   runnable headless.

## 9. Editor integration (deferred, not in this deliverable)

Hooks kept in mind: a directive verdict strip in the GUI (three lamps, re-evaluated
on every edit); "apply intervention" as an edit-log action; family files under
`scenarios/families/*.yaml`. None of this blocks the headless deliverable.

## 10. Resolved questions

- **Q1 — signals: explicit.** Signal phases live in the state (assumed
  family-consistently when absent, flagged as an assumption); D1 keeps the
  `runs_red(hero)` conjunct.
- **Q2 — D3 strength: nominal.** "Would not interfere" is evaluated along the
  nominal collision-realizing evolution (the `□`-over-branches strengthening noted
  as a possible later flag).
- **Q3 — control scope: as proposed.** `◇` branches only over the hero's route and
  constant speed; interventions may edit any non-ego actor, never the ego.
- **Q4 — minimality: greedy analytic.** Closed-form per-directive repairs with a
  joint re-check, always verified by re-evaluation; approximately minimal.
- **Q5 — causality (added in review).** Interventions never change the past:
  causal, forward-in-time control edits only (`retime`, `reroute`), executed under
  P1 kinematics. State edits are authoring operations belonging to the editor, not
  the directive machinery.
