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

**P1 — Evolution model M (piecewise constant velocity, C¹ paths).** Each actor
follows its lane centerline at its **current speed, held constant** for the whole
prediction. Paths are polyline+arc: inbound leg → intersection → outbound leg, where
turns are circular fillets **tangent to both centerlines**, traversed at the same
speed. Speed is constant and the velocity *direction* is continuous through tangent
arcs, so the velocity profile has no jump discontinuities — the "continuous"
strengthening is satisfied. After exiting, actors continue straight to the map edge
and hold there.

**Branching.** The only nondeterminism is the **route choice** at the intersection:
an actor on an inbound leg has routes `{left, straight, right}` (each mapping to the
correct outbound leg under right-hand traffic). An evolution is a route assignment
per actor; the **nominal evolution ξ\*** takes each actor's *recognized* route (§4).
The **admissible branch set Γ(x, v)** for a controllable vehicle `v` additionally
allows re-choosing `v`'s route and re-choosing its constant speed `v' ∈ (0, v_max]`
(default `v_max = 20 m/s`), with all other actors on nominal. Γ is what `◇`
quantifies over; only the hero is treated as controllable (Q3).

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
- **D2** — `◇ F collide` at time t ⟺ hero has not passed P, ego's window is not
  over (`e₁(t) > 0`), and `∃ v' ∈ (0, v_max]: d_h(t)/v' ∈ [e₀(t), e₁(t)]` ⟺
  `d_h(t)/v_max ≤ e₁(t)`. So D2 = G of a one-line inequality — checked densely over
  `[0, t*]` and reported with the first violation time.
- **D3** — per third vehicle w: closed-form window overlap for
  `occupies_conflict`, first-collision check against ego/hero, and a constant-speed
  car-following bound for `blocks` (w slower than hero's required speed with gap
  closing to < `g_min` before `t*`).

## 6. Minimal intervention

**P5 — intervention space: edits to the *state*, never to the ego** (the ego is the
system under test; Q3). Per non-ego actor:

| Knob | Cost |
|---|---|
| `Δv` — change constant speed (result ∈ (0, v_max]) | `w_v·|Δv|`, w_v = 1 /(m/s) |
| `Δs` — slide along its lane (spawn shift) | `w_s·|Δs|`, w_s = 0.5 /m |
| `route(v) ← r` | `w_r = 5` |
| `remove(v)` (last resort) | `w_x = 20` |

Total cost = Σ over edited actors; **minimal intervention = argmin cost s.t. all
three directives evaluate true on the edited state.**

**P6 — repair algorithm: analytic per-directive repair + joint re-check** (approx.
minimal; exact minimality would need a joint search — see Q4):

1. Evaluate D1–D3; if all true, return ∅.
2. **D1 repair** — for each hero candidate (ranked by §4): the constant speed that
   centers hero's occupancy on ego's is `v* = d_h/((e₀+e₁)/2)` — closed form. If
   `v* > v_max` or hero is past P, fall back to `Δs` (slide back along the lane,
   also closed form) or a route change; take the cheapest feasible candidate.
3. **D2 repair** — usually implied by D1 under constant velocity; when it fails on
   the margin (`d_h(t)/v_max > e₁(t)` for some t < t*), shift `Δs` to restore slack.
4. **D3 repair** — per interferer: min `|Δv|` or `|Δs|` that clears the conflict
   window / opens the corridor gap (interval arithmetic again), else reroute, else
   remove. Cheapest option per interferer.
5. Re-evaluate all directives on the edited state; iterate (≤ 3 passes). Report the
   intervention list, total cost, and the post-intervention evaluation. If still
   infeasible (e.g. no candidate vehicle exists at all), report `INFEASIBLE` with
   the reason rather than inventing actors.

## 7. Demonstration

`python3 v2/directives.py --demo` (also `--state FILE.yaml` for a single state)
prints, per state: the abstract-state summary (lanes, routes, roles, assumed
signals), each directive's `Eval` (value, witness, t_fail, explanation), and — where
something is false — the intervention, its cost, and the re-evaluation showing all
green. Demo states:

| | State | Expected |
|---|---|---|
| S1 | hero westbound on `EN` (red), on collision course with northbound ego | D1 ✓ D2 ✓ D3 ✓ |
| S2 | hero too slow — misses ego's window | D1 ✗ → `Δv` on hero |
| S3 | hero already through the intersection | D2 ✗ at t=0 → `Δs` (or reroute) |
| S4 | slow lead vehicle ahead of hero on `EN` | D3 ✗ (`blocks`) → `Δv`/`Δs` on lead |
| S5 | third vehicle's path crosses P during the collision window | D3 ✗ (`occupies_conflict`) → `Δs` |
| S6 | no vehicle on any conflicting red approach | D1 ✗, no cheap witness → route change or `INFEASIBLE` |

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
