# Scenario Editor v1 — Design Document (finalized)

v1 adds **Functions**: reactive primitives alongside v0's maneuver primitives.
A Function maps *observations* (of other agents and the map) to *actions*, edited in a
circuit-design-style node-graph GUI in the same subwindow as the maneuver
timing-curve editor.

Everything from v0 (looping clock, versioned saves, provenance, edit log) is retained.
This doc covers only what changes.

**Status:** all decisions (D1–D8) agreed and implemented. Rev 3 added time
observables (D5) and made **dropdowns replace click-to-cycle everywhere** (op
selection, value selection, and the segment `type` button in the original UI).

---

## Design decisions

**D1 (revised) — Selectable output attribute: speed or yaw.** The function editor has
two fixed sink nodes: **OUT:speed** (m/s, clamped [0, 60]) and **OUT:yaw** (absolute
heading, degrees CCW from East). Wiring a scalar into a sink makes the function drive
that attribute; an unbound attribute *holds its entry value* (speed carried in from
the previous segment, or the entry heading). Either or both may be bound. Motion
during a function segment is a kinematic unicycle: each tick, heading and speed are
taken from the sinks (or held), and position integrates forward. With only OUT:speed
bound the actor moves in a straight line at the commanded speed; with OUT:yaw bound
it can steer (e.g. `bearing(self.pos, actor0.pos)` = pursuit).

**D2 (revised) — Edge-carried operations; multi-input instantiation.** The user
creates *value nodes*; dragging node→node creates an operation with those inputs and a
materialized *output node*. Your question — how are ops with more than two inputs
instantiated — is answered by rule 3:

1. **Drag A → B** (both existing nodes): creates a binary op with inputs (A, B).
2. **Drag A → empty canvas**: creates a unary op on A.
3. **Drag X → an existing op node**: if the op's signature has an unfilled slot of
   X's type, X fills the first such slot; otherwise the wire is refused with a status
   message. This is how ops grow beyond two inputs.
4. **Click the op label**: opens a dropdown of operations whose signature is
   consistent with the currently wired inputs. Switching *to* a higher-arity op
   (e.g. binary `mul` → ternary `if`) is allowed: missing slots render as hollow red
   ports and the node reads "incomplete" until filled via rule 3. An incomplete
   graph makes the segment hold its entry state, and is flagged by `--validate`.

So the practical route to `if(cond, then, else)`: drag cond→then (binary op), switch
it to `if` via the dropdown, drag the else node onto the op. Input order is shown as numbered slots; a
small ⇄ badge on 2-input ops of asymmetric signature (`sub`, `div`, `lt`, `gt`) swaps
their inputs. Output nodes are connectable, so graphs compose. DAG only — a wire that
would create a cycle is refused.

**D3 (agreed) — Tick-based simulation (fixed dt = 1/60 s) with cached trajectory.**
Reactive actors need feedback, so v1 simulates forward in ticks and caches all poses
(+speeds) over one loop period, recomputing after every edit (hundreds of ticks —
instant). Scrubbing/looping reads the cache; the UI feels identical to v0. Functions
read the **previous tick's** world state (synchronous update), so per-tick evaluation
order across actors doesn't matter. Note: because a function segment's end pose is
emergent, downstream segments' start poses now come from this simulation rather than
closed-form chaining — same continuity guarantee, computed numerically.

**D4 (resolved by D7) — Loop period.** Function segments carry an explicit `duration`
like any maneuver, so v0 scheduling is unchanged: actor total = Σ segment durations,
period = max total, finished actors hold their final pose. No scenario-level fallback
needed.

**D5 (revised) — Observation catalog.** A value node selects via dropdown:
- `const` — user-typed number (click the value to edit, v0 field UX);
- `time.global` — current global clock T within the loop (s);
- `time.segment` — time since this function segment started (s);
- `self.speed`, `self.pos`, `self.heading` — the owning actor's own state;
- `actor<k>.speed`, `actor<k>.pos`, `actor<k>.heading` — any other actor (dynamic,
  observed per tick);
- `map.<named point>` — static points: intersection center plus the four
  lane-centerline crossing points `cross_NE/NW/SE/SW` (e.g. `cross_NE` = (+1.75, +1.75));
- `map.custom` — a static point with user-typed `(x, y)` (click-to-place on the BEV
  deferred to v2).

**D6 (revised) — Types and operation catalog, now with conditionals.** Two value
types: **Scalar** and **Point**. Booleans are scalars: 0 = false, nonzero = true;
predicates return 1.0/0.0.

| Signature | Operations |
|---|---|
| (Scalar, Scalar) → Scalar | `add`, `sub`, `mul`, `div`, `min`, `max`, `lt`, `gt` |
| Scalar → Scalar | `neg`, `abs`, `not` |
| (Scalar, T, T) → T, T ∈ {Scalar, Point} | `if` (cond nonzero → 2nd input, else 3rd) |
| (Point, Point) → Scalar | `dist`, `bearing` (heading from 1st to 2nd, deg) |
| (Point, Point) → Point | `midpoint` |
| Point → Scalar | `x`, `y`, `mag` |

Wires are type-checked; the op dropdown lists only signature-compatible ops. `div`
guards the denominator with ε = 1e-6. `bearing` exists mainly to make OUT:yaw useful
(pursuit/aim-at-point behaviors).

**D7 (revised) — Functions are sequence items.** An actor has one sequential list of
**segments**, each either a maneuver (v0 types) or a **function**; they never overlap.
`function` is simply a 7th segment type: the header `type:` button now opens a
dropdown over `go_straight, turn_left, turn_right, accelerate, decelerate, stop,
function` (it no longer cycles).
A function segment has: `duration`, the node graph, and its OUT bindings. Chaining is
as in v0: the next segment starts at the pose actually reached (for function segments,
the simulated end pose), and longitudinal segments seed `v0` from the incoming speed
(for function segments, the last commanded speed). The subwindow therefore needs no
tabs — prev/next steps through segments, and the panel shows the curve editor or the
node-graph editor according to the current segment's type.

**D8 (agreed) — Layout.** Window stays 1100 px wide. Subwindow 250 → 400 px; BEV
canvas 720 → 540 px; `pixels_per_meter` ≥ 6.5 (map crops rather than shrinks). The v1
sample scenario uses `arm_length: 40` so the whole map **including spawn points** fits
on-screen. Larger legacy maps load fine but their outer arms crop.

---

## 1. Data model

```
Actor:   ... (v0 fields), segments[]           # YAML key stays `maneuvers`
Segment: Maneuver (v0, unchanged) | Function
Function: duration, nodes[], out {speed: node_id|null, yaw: node_id|null}
Node:    id, kind, pos (x, y editor coords)
  kind=obs:       source ('self' | actor id), field ('speed'|'pos'|'heading')
  kind=const:     value
  kind=map_point: name (named catalog) | point [x, y]
  kind=op:        op, inputs [node ids, in slot order]
```

Edges are implied by op nodes' `inputs`. Node `pos` persists the editor layout.

## 2. Evaluation & motion

Per tick, topological evaluation of the DAG against the previous tick's world state.
Function-segment motion (unicycle):

```
v  = eval(out.speed)  if bound else  held entry speed      (clamped to [0, 60])
h  = eval(out.yaw)    if bound else  held heading
x += v·cos(h)·dt ;  y += v·sin(h)·dt
```

An incomplete/unbound graph holds entry speed and heading (straight line).

## 3. Simulation (revised)

Fixed-step simulator, dt = 1/60 s, run over the loop period after load and after every
edit, producing per-actor pose+speed arrays. The interactive loop and `--capture` read
the cache by index; the time scrubber snaps to the nearest tick. Pure-maneuver actors
reproduce v0 trajectories exactly (same `pose_at` math, sampled per tick).

## 4. GUI

### Layout
```
+---------------------------------------------------------+
| T=[..]  Reset [Play] Save   +Actor -Actor                |  56 px
+---------------------------------------------------------+
|                 BEV (cropped, ppm >= 6.5)                |  540 px
+---------------------------------------------------------+
| Actor <id>  segment i/n   [type][prev][next][+seg][-seg] |  400 px
|   curve editor (maneuver)  OR  node-graph (function)     |
+---------------------------------------------------------+
```

### Function editor (shown when the current segment is a function)
- **Canvas:** node area filling the subwindow. Nodes are rounded boxes labelled with
  their value (`actor 0 . speed`, `const 1.5`, `map cross_NE`, `dist`, ...); op nodes
  draw incoming edges with the op label at the junction and numbered input slots.
  Sinks **OUT:speed** and **OUT:yaw** are fixed at the right edge; `duration` remains
  a plain field.
- **Create node:** double-click empty canvas → new value node (default `const 1.0`).
- **Choose value:** click a value node → dropdown of the observation catalog (D5);
  click a const's number / custom point's coords → type, Enter commits.
- **Wire / ops / multi-input:** rules 1–4 of D2. Scalar → OUT sink binds the function.
- **Delete:** select node + Delete removes it and everything downstream (status
  message names the count). ESC deselects.
- **Move:** drag a node body; positions saved.
- Graph edits are logged to `edit_history.yaml` (`action: fn_add_node | fn_set_value |
  fn_wire | fn_set_op | fn_swap_inputs | fn_delete_node | fn_bind_out | fn_unbind_out |
  fn_move_node`, the last logged once on drag release, like `move_actor`). Const-value
  and `duration` edits use the v0 parameter-edit log format.

### Dropdown widget
Generic single-column popup anchored to the clicked element: hover highlight, click
selects, ESC/click-outside cancels, scrolls if taller than the subwindow. **All
multi-choice selection uses dropdowns — nothing click-cycles**: value-node selection,
op selection, and the segment `type:` button in the subwindow header.

## 5. Worked example — guaranteed perpendicular collision

Ego (actor 0) northbound on `SE` with ordinary maneuvers; hero (actor 1) westbound on
`EN` with a single function segment (OUT:speed bound, OUT:yaw unbound → heading stays
180°, i.e. straight down its lane). Lane centerlines cross at `P = cross_NE =
(1.75, 1.75)`.

```
v_hero = speed(ego) * ( dist(self.pos, P) / dist(ego.pos, P) )
```

Why it always collides: the rule keeps d_hero/d_ego constant, so d_hero → 0 exactly
when d_ego → 0 — both reach P at the same instant, for **any** starting positions
along the two legs and any ego speed profile. Ships as `scenarios/scenario_v1.yaml`.

## 6. YAML schema additions

```yaml
actors:
  - id: 1
    ...
    maneuvers:                       # mixed segment list (key name kept from v0)
      - type: function
        duration: 12.0
        out: {speed: n7}             # yaw omitted -> held
        nodes:
          - {id: n0, kind: obs, source: "0",  field: speed, pos: [40, 60]}
          - {id: n1, kind: obs, source: self, field: pos,   pos: [40, 130]}
          - {id: n2, kind: map_point, name: cross_NE,       pos: [40, 200]}
          - {id: n3, kind: obs, source: "0",  field: pos,   pos: [40, 270]}
          - {id: n4, kind: op, op: dist, inputs: [n1, n2],  pos: [200, 165]}
          - {id: n5, kind: op, op: dist, inputs: [n3, n2],  pos: [200, 235]}
          - {id: n6, kind: op, op: div,  inputs: [n4, n5],  pos: [340, 200]}
          - {id: n7, kind: op, op: mul,  inputs: [n0, n6],  pos: [470, 130]}
```

`--validate` additionally checks: node ids unique, inputs resolve, graph acyclic, op
signatures complete and type-correct, observed actor ids exist, OUT bindings are
Scalar. v0 files (no function segments) remain valid; `regen_state.sh` unchanged.

## 7. Single-file structure changes (`scenario_editor.py`)

1. Dataclasses: + `Node`, `Function`; segment lists hold both kinds.
2. Loader/validator: parse + validate function segments and graphs.
3. Function runtime: type checking, topo sort, per-tick evaluation, unicycle motion.
4. **Simulator: closed-form → fixed-step with trajectory cache** (biggest change;
   also supplies downstream segments' start poses and incoming speeds).
5. Renderer/UI: node-graph editor, dropdown widget (replacing all click-cycling,
   incl. the `type:` button), `function` as a segment type, new layout.
6. Persistence: new edit-log actions; schema additions.

Still a single file, `pygame` + `pyyaml` only; simulation/evaluation import-safe for
headless testing. In addition to v0's `--validate` and `--capture`, a `--snapshot
OUT.png [--select ID --time T]` flag renders one paused frame headlessly (used to
verify the editors without a display).
