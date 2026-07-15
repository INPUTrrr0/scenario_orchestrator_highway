# Scenario Editor — Design Document (finalized)

A single-file Python (pygame) tool to load, visualize, and edit orchestrated driving scenarios at a 4-way intersection.

**Resolved decisions**
- **Motion model:** chained path primitives — each maneuver is a geometric segment, chained end-to-end at the actual reached pose (no teleports).
- **Timing curve:** straight-line maneuvers use a linear *velocity-vs-time* curve (`v0` + `accel·t`); distance is emergent (∫v dt), so there is no `length` input. Turns are geometry (`radius`, `angle`) traversed over `duration`.
- **Scheduling:** each actor runs its maneuvers sequentially; all actors run in parallel on one **global looping clock** (D2). An actor that finishes its sequence **holds its final pose** until the loop restarts.
- **Lane changes dropped** (D3).
- **Curve editor:** one maneuver at a time — the maneuver active at the paused timestamp (D4).
- **Saving/versioning:** edited scenarios save to a dedicated `scenarios/` folder with the next available version number; a **branching provenance graph** is tracked in metadata (D5).
- **Coordinates:** graph/Cartesian, `y` up (D6).

---

## 1. World frame & conventions
- Units: meters. Origin `(0,0)` at intersection center.
- `x` = East, `y` = North (graph coords, y-up). Heading in degrees, `0°`=East, `90°`=North, CCW positive.
- Screen mapping flips `y` and scales by `pixels_per_meter`.

## 2. Data model

```
MapConfig: lane_width, arm_length          # single carriageway N-S and E-W
Actor:  id, color, length, width, start{x,y,heading}, maneuvers[]
Maneuver: type, duration, curve{v0, accel}   (longitudinal) | radius, angle (turns)
```

There is **no `length` parameter**. Distance is a derived quantity, not an input: for straight-line motion it is the integral of the velocity curve over the duration; for a turn it is the arc `radius · angle`. `duration` (seconds) and speed (m/s) are the two independent inputs; distance = ∫speed dt falls out of them.

### Maneuver catalog

| Type | Defined by | Motion |
|------|-----------|--------|
| `go_straight` | velocity curve, `accel = 0` | forward at constant speed `v0`; heading fixed |
| `accelerate` | velocity curve, `accel > 0` | forward, speeding up from `v0` |
| `decelerate` | velocity curve, `accel < 0` | forward, slowing down from `v0` (never reverses) |
| `turn_left` | `radius`, `angle`(°, def 90), `duration` | CCW arc, sweeps full `angle` over `duration`, heading += |
| `turn_right` | `radius`, `angle`, `duration` | CW arc, heading −= |
| `stop` | `duration` | holds pose for `duration` |

`curve_kind` is `velocity` for the three longitudinal types (editable velocity-vs-time plot) and `none` for turns/stop (geometry fields only).

## 3. Motion

Local time `t ∈ [0, duration]`.

- **Longitudinal** (`go_straight`/`accelerate`/`decelerate`): velocity `v(t) = v0 + accel·t`, clamped so it never goes negative; distance `s(t) = v0·t + ½·accel·t²`; pose = start + heading·`s(t)`. Distance is emergent — the maneuver can never "finish early and hold." `go_straight` is simply `accel = 0`.
- **Turns:** `frac = clamp(t/duration, 0, 1)`; the actor sweeps `angle·frac` along an arc of the given `radius`. Always completes exactly the specified angle at `t = duration`.
- **stop:** stationary for `duration`.

Editable per maneuver: **v0** (speed), **accel** (for accel/decel), **radius**/**angle** (turns), **duration**. `Maneuver.exit_speed()` gives the path speed at the end, used to seed the next maneuver's `v0` so speed is continuous across boundaries.

## 4. Simulation & scheduling
- Per actor, cumulative maneuver start times; `total` = Σ durations. Global loop period = `max(total)` over actors.
- At clock `T` (mod period): if `T ≥ actor.total`, hold final pose; else locate active maneuver `i`, `t = T − cum[i]`, pose = `maneuver_i.pose_at(start_pose_i, t)`.
- **Chaining:** maneuver `i+1` starts at maneuver `i`'s end pose `pose_at(start, duration)`. Longitudinal distance is emergent and turns always complete their angle, so segments join continuously.
- Paths are precomputed at load and after each edit.

## 5. Map & leg naming
One N-S carriageway and one E-W carriageway crossing at the origin (each bidirectional, width `2·lane_width`). Rendered: grass background, asphalt, dashed center line, solid edge lines, stop lines. No lane-change lanes.

**Legs (8).** Each arm has two lanes, named `[arm][side]`: the first letter is the arm (N/E/S/W), the second the side. Ordered counter-clockwise from `EN`: `EN, NE, NW, WN, WS, SW, SE, ES`. Inbound (toward-center) lanes follow right-hand traffic:
- `SE` = South arm, east lane (x=+1.75) — inbound **northbound**.
- `EN` = East arm, north lane (y=+1.75) — inbound **westbound**.
- `WS` = West arm, south lane (y=−1.75) — inbound **eastbound**.
- `NW` = North arm, west lane (x=−1.75) — inbound **southbound**.

The default sample places the ego (ID 0, green) on `SE` and actor 1 (red) on `EN`, both going straight through so their paths cross near the center.

## 6. GUI layout
```
+---------------------------------------------------------+
| T=[1.23]/8.0s  Reset [▶/⏸] Save     +Actor  -Actor      |  top bar
+---------------------------------------------------------+
|                                                         |
|                BEV visualization (map + actors)         |  center canvas
|            actors clickable & selectable when paused    |
|                                                         |
+---------------------------------------------------------+
|  Actor <ID> — <maneuver> [i/n]              < prev next>|  timing-curve subwindow
|  velocity (m/s)    ____----      • current-time marker  |  (stacked BELOW the BEV,
|      |         __----                                   |   always visible; editor
|      |____---------------------  time                   |   shown when paused + an
|   v0:[ ]  accel:[ ]  duration:[ ]                       |   actor is selected)
+---------------------------------------------------------+
```
- Top bar: editable current-time field `T=[..]` (click and type a time to scrub; auto-pauses), centered Play/Pause toggle (infinite loop when playing) plus Reset/Save, and **+Actor / -Actor** for adding/removing actors (`-Actor` enabled only with a selection; `a` and `Delete` are keyboard shortcuts).
- BEV: each actor a colored rectangle with numeric ID label and heading indicator; selected actor outlined.
- The timing-curve subwindow is stacked **below** the BEV (fixed region, no overlap; the window height is `topbar + canvas + subwindow`).
- Selection only while paused: click an actor → outline + open the subwindow titled with its ID.
- Subwindow shows the maneuver active at the paused time (prev/next steps through the list). Longitudinal maneuvers show a velocity-vs-time plot editable via text fields **and** draggable endpoints (left → `v0`, right → `accel`; `go_straight` has only the left/`v0` handle since it is flat). Turns/stop show geometry fields (`radius`/`angle`/`duration`) instead of a plot. Every commit re-precomputes the path and is logged.
- **Add actor:** spawns a new actor with the next integer ID on the next inbound leg (cycling SE/EN/WS/NW) with a default straight-through maneuver. **Remove actor:** drops the selected actor. Both are recorded in the edit log as structural entries (`action: add_actor|remove_actor`).
- **Spawn editing (drag in BEV):** the selected actor shows a highlighted **start marker** at its start pose plus a **rotation handle**. Drag the marker body to move the spawn `(x, y)` (the whole trajectory shifts rigidly); drag the handle to set the start heading. Grabbing either jumps the clock to `T=0` so the marker and the animated body coincide. Committed on release and logged (`action: move_actor` with old/new start).
- **Maneuver list editing:** `+ mvr` inserts a `go_straight` (continuing at the current speed) after the current maneuver; `- mvr` deletes the current one (kept ≥ 1). Editable params depend on type: `v0`/`duration` for `go_straight`, `v0`/`accel`/`duration` for accel/decel, `radius`/`angle`/`duration` for turns.
- **Change maneuver type:** the `type: <t>` button (top-right header row with prev/next/+mvr/-mvr) cycles `go_straight → turn_left → turn_right → accelerate → decelerate → stop`. Turn defaults are filled (radius 5, angle 90). Longitudinal types are seeded from the **incoming speed** `v_in` (previous maneuver's exit speed, or 10 m/s for the first): `v0 = v_in`, with `accel = 0` (go_straight) / `+2` (accelerate) / `−2` (decelerate). Logged as `action: set_maneuver_type`. (Reordering maneuvers remains YAML-only by design.)

**Speed continuity.** Longitudinal maneuvers inherit the incoming speed as `v0`, so an `accelerate` inserted after a moving maneuver builds from the current speed instead of snapping to 0 (which used to read as a *deceleration* at the boundary). `Maneuver.exit_speed()` computes the carried-over speed: `v0 + accel·duration` for longitudinal, `arc_length / duration` for turns, 0 for `stop`.

## 7. Persistence, versioning & provenance (D5)

All scenario artifacts live under a `scenarios/` folder:
```
scenarios/
  scenario_v1.yaml          # base
  scenario_v2.yaml          # saved edit (parent v1)
  scenario_v3.yaml          # branch (also parent v1)
  provenance.yaml           # version graph
  edit_history.yaml         # append-only edit log
```

**Save:** writes `scenario_v{N}.yaml` where `N` = next available global version number (max existing + 1, so branches never collide), then appends a node to the provenance graph with `parent` = the version currently loaded. The saved version becomes the new working base; to branch, reload an earlier version and save again (it gets a fresh N with that earlier parent).

**`provenance.yaml`:**
```yaml
versions:
  - {version: 1, file: scenario_v1.yaml, parent: null, created: <iso>}
  - {version: 2, file: scenario_v2.yaml, parent: 1,    created: <iso>}
  - {version: 3, file: scenario_v3.yaml, parent: 1,    created: <iso>}   # branch
```

**`edit_history.yaml`** (append-only; parameter edits and structural add/remove):
```yaml
- timestamp: 2026-07-14T15:04:05
  base_version: 1
  actor_id: "0"
  maneuver_index: 1
  maneuver_type: turn_left
  parameter: slope
  old_value: 0.33
  new_value: 0.50
- timestamp: 2026-07-14T15:05:10
  base_version: 1
  action: add_actor
  actor_id: "2"
```

**Format drift & regeneration.** The persisted files (`edit_history.yaml`, `provenance.yaml`, saved `scenario_v*.yaml`) can fall out of step as the editor's format evolves. Two tools keep them consistent:
- `scenario_editor.py --validate <file>` loads a scenario against the *current* format and exits 0 (OK) or 1 (with the reason). No GUI.
- `regen_state.sh [scenarios_dir]` regenerates state: resets `edit_history.yaml`, validates each `scenario_v*.yaml` and **moves** (never deletes) incompatible ones to `scenarios/incompatible/`, then rebuilds `provenance.yaml` to reference only surviving files (parent links preserved where the parent still exists). `scenario_v1.yaml` is the source of truth; everything else is regenerable.

## 8. YAML scenario schema
```yaml
map: {lane_width: 3.5, arm_length: 60}
render: {pixels_per_meter: 6}
actors:
  - id: 0
    color: [90, 190, 110]
    length: 4.5
    width: 2.0
    start: {x: 1.75, y: -58, heading: 90}     # SE leg, northbound
    maneuvers:
      - {type: go_straight, duration: 4.0, curve: {v0: 14.5, accel: 0.0}}
      - {type: turn_left,   radius: 6, angle: 90, duration: 2.5}
  - id: 1
    color: [60, 120, 210]
    length: 4.5
    width: 2.0
    start: {x: 58, y: 1.75, heading: 180}     # EN leg, westbound
    maneuvers:
      - {type: go_straight, duration: 2.5, curve: {v0: 12.0, accel: 0.0}}
      - {type: decelerate,  duration: 3.0, curve: {v0: 12.0, accel: -4.0}}  # 12 → 0 m/s
      - {type: stop,        duration: 2.0}
      - {type: accelerate,  duration: 3.0, curve: {v0: 0.0, accel: 3.0}}    # 0 → 9 m/s
```
`length` (actor body length) and `width` describe the drawn rectangle; they are unrelated to distance travelled.

## 9. Single-file structure (`scenario_editor.py`)
Dependencies: `pygame`, `pyyaml`.
1. Dataclasses — `Maneuver`, `Actor`, `MapConfig`, `Scenario`.
2. Loader/validator (YAML → objects).
3. Geometry — per-type `pose_at(start, t)` (time-based); per-actor path precompute (chained by end pose).
4. Simulator — global clock → per-actor pose; loop + end-hold.
5. Renderer — map, actors, top bar, curve subwindow.
6. UI/input — play/pause, actor hit-test, field editing, drag handles, prev/next.
7. Persistence — versioned save, provenance graph, edit logger.
8. Main loop (fixed timestep). Geometry/simulation are import-safe (no display) for headless testing.
