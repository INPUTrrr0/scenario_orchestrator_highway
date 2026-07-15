# Scenario Editor — Design Document (finalized)

A single-file Python (pygame) tool to load, visualize, and edit orchestrated driving scenarios at a 4-way intersection.

**Resolved decisions**
- **Motion model:** chained path primitives — each maneuver is a geometric segment, chained end-to-end at the actual reached pose (no teleports).
- **Timing curve:** linear, but its meaning is **per-maneuver** (D1): geometric maneuvers use *progress-fraction vs time*; speed maneuvers expose the physically relevant quantity, e.g. *velocity vs time* for accelerate/decelerate.
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
Maneuver: type, duration, curve{slope, intercept}, <geometry params>
```

### Maneuver catalog

| Type | Curve meaning | Geometry / params | Effect |
|------|---------------|-------------------|--------|
| `go_straight` | progress vs time | `length` | forward, heading fixed |
| `turn_left` | progress vs time | `radius`, `angle`(°, def 90) | CCW arc, heading += |
| `turn_right` | progress vs time | `radius`, `angle` | CW arc, heading −= |
| `accelerate` | **velocity vs time** | `length` | forward; velocity = `intercept + slope·t` (slope>0) |
| `decelerate` | **velocity vs time** | `length` | forward; velocity = `intercept + slope·t` (slope<0) |
| `stop` | — (hold) | — | holds pose for `duration` |

Each maneuver reports a `curve_kind` (`progress` or `velocity`) that determines both how the curve maps to motion and the y-axis label in the editor.

## 3. Timing curve → motion

Local time `t ∈ [0, duration]`, `curve(t) = slope·t + intercept`.

- **progress kind:** `u = clamp(curve(t), 0, 1)`; pose = `pose_along_segment(u)`.
- **velocity kind:** `v(t) = curve(t)` (m/s); distance `s(t) = intercept·t + ½·slope·t²`; `u = clamp(s/length, 0, 1)`. This yields real intra-segment speed change even though the curve is linear (linear in velocity). `intercept` = initial velocity, `slope` = acceleration.
- **stop:** `u` held; actor stationary for `duration`.

Editable per maneuver: **slope**, **intercept**, **duration**.

## 4. Simulation & scheduling
- Per actor, cumulative maneuver start times; `total` = Σ durations. Global loop period = `max(total)` over actors.
- At clock `T` (mod period): if `T ≥ actor.total`, hold final pose; else locate active maneuver `i`, `t = T − cum[i]`, `u = progress_i(t)`, pose from the chained segment.
- **Chaining:** maneuver `i+1` starts at maneuver `i`'s *actually reached* end pose `pose_at(u_end)` where `u_end = progress_i(duration_i)` — avoids discontinuities when a curve doesn't complete a segment.
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
|  velocity/progress  ____----     • current-time marker  |  (stacked BELOW the BEV,
|      |         __----                                   |   always visible; editor
|      |____---------------------  time                   |   shown when paused + an
|   slope:[ ] intercept:[ ] duration:[ ]                  |   actor is selected)
+---------------------------------------------------------+
```
- Top bar: editable current-time field `T=[..]` (click and type a time to scrub; auto-pauses), centered Play/Pause toggle (infinite loop when playing) plus Reset/Save, and **+Actor / -Actor** for adding/removing actors (`-Actor` enabled only with a selection; `a` and `Delete` are keyboard shortcuts).
- BEV: each actor a colored rectangle with numeric ID label and heading indicator; selected actor outlined.
- The timing-curve subwindow is stacked **below** the BEV (fixed region, no overlap; the window height is `topbar + canvas + subwindow`).
- Selection only while paused: click an actor → outline + open the subwindow titled with its ID.
- Subwindow shows the maneuver active at the paused time (prev/next steps through the list). Y-axis label switches between "progress" and "velocity (m/s)" by `curve_kind`. Edit via text fields **and** draggable plot endpoints (left endpoint → intercept, right endpoint → end value → slope). Every commit re-precomputes the path and is logged.
- **Add actor:** spawns a new actor with the next integer ID on the next inbound leg (cycling SE/EN/WS/NW) with a default straight-through maneuver. **Remove actor:** drops the selected actor. Both are recorded in the edit log as structural entries (`action: add_actor|remove_actor`).
- **Spawn editing (drag in BEV):** the selected actor shows a highlighted **start marker** at its start pose plus a **rotation handle**. Drag the marker body to move the spawn `(x, y)` (the whole trajectory shifts rigidly); drag the handle to set the start heading. Committed on release and logged (`action: move_actor` with old/new start).
- **Maneuver list editing:** `+ mvr` inserts a default `go_straight` after the current maneuver; `- mvr` deletes the current one (kept ≥ 1). Per maneuver, geometry params are editable alongside the timing curve — `length` for straight/accel/decel, or `radius` + `angle` for turns — as text fields. (Reordering and changing a maneuver's type remain YAML-only by design.)

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
  - id: car_A
    color: [210, 70, 60]
    length: 4.5
    width: 2.0
    start: {x: -55, y: -1.75, heading: 0}     # eastbound
    maneuvers:
      - {type: go_straight, length: 35, duration: 3.0, curve: {slope: 0.333, intercept: 0.0}}
      - {type: turn_left,   radius: 6, angle: 90, duration: 2.5, curve: {slope: 0.4, intercept: 0.0}}
      - {type: go_straight, length: 40, duration: 3.5, curve: {slope: 0.286, intercept: 0.0}}
  - id: car_B
    color: [60, 120, 210]
    length: 4.5
    width: 2.0
    start: {x: 1.75, y: 55, heading: 270}     # southbound
    maneuvers:
      - {type: go_straight, length: 25, duration: 2.5, curve: {slope: 0.4, intercept: 0.0}}
      - {type: decelerate,  length: 12, duration: 3.0, curve: {slope: -2.0, intercept: 8.0}}  # v: 8→2 m/s
      - {type: stop,        duration: 2.0, curve: {slope: 0.0, intercept: 0.0}}
      - {type: accelerate,  length: 30, duration: 3.0, curve: {slope: 2.0, intercept: 0.0}}   # v: 0→6 m/s
```

## 9. Single-file structure (`scenario_editor.py`)
Dependencies: `pygame`, `pyyaml`.
1. Dataclasses — `Maneuver`, `Actor`, `MapConfig`, `Scenario`.
2. Loader/validator (YAML → objects).
3. Geometry — per-type `pose_at(u)` + arc length; per-actor path precompute (chained by reached pose).
4. Simulator — global clock → per-actor pose; loop + end-hold.
5. Renderer — map, actors, top bar, curve subwindow.
6. UI/input — play/pause, actor hit-test, field editing, drag handles, prev/next.
7. Persistence — versioned save, provenance graph, edit logger.
8. Main loop (fixed timestep). Geometry/simulation are import-safe (no display) for headless testing.
