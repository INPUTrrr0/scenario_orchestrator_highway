# Scenario Editor v4 — Design: Closed-Loop Scenario Orchestration (draft for review)

v4 turns the v2 directive layer from a **single-scene** repair tool into a
**closed-loop orchestrator**. v2 answered *"given this instant, what minimal
intervention makes the red-light family hold?"* — one shot, one scene. v4 asks that
question **at every tick of a running rollout**: it steps the world forward, lets the
user inject **perturbations** as it runs, and lets the orchestrator **continuously
intervene** to keep the scenario on its intended course. The whole session — every
tick, every perturbation, every intervention, every checkpoint — is recorded as an
interactive **game tree** with per-node playback of the scenario realization, in the
spirit of `v0/outputs/scenario_tree.html`.

v3 is **out of scope**: v4 builds on the v2 kinematic model (constant-velocity,
route-based prediction; `retime`/`reroute` interventions), not v3's agentic actors.

Deliverables (§10): `v4/orchestrator.py` (headless closed-loop kernel, imports both
`v2/scenario_editor.py` and `v2/directives.py`), `v4/session_editor.py` (interactive
pygame front end), and `v4/build_tree.py` (session → colored HTML game tree). All
design forks raised at review are **resolved** in §11; decisions are marked **P#**.

---

## 0. From single-scene intervention to closed-loop rollout

| | v2 (single scene) | v4 (closed loop) |
|---|---|---|
| Input | one `State` at one instant | a base maneuver-script scenario + a live session of edits |
| What runs | `recognize → evaluate_family → repair`, once | the same triad **every tick**; only *decision points* are persisted |
| Time | frozen (prediction only) | advancing one `dt_tick` (0.1 s) at a time |
| Intervention | proposed, rendered as a compare video | **applied** as a maneuver-script edit and carried forward |
| User role | authors the scene at t=0 | injects **perturbations** (incl. on the ego) at any tick while it runs |
| Output | `S*_compare.mp4` per demo state | a session folder of **snapshots** + a colored game tree |
| Files | `scenario_v{N}.yaml` (independent scenarios) | `snapshot_v{N}.yaml` (realizations *within one* rollout) |

## 1. One representation: the maneuver script (P1)

**P1 — the maneuver script is the single ground-truth representation** shared by the
base scenario, user perturbations, and orchestrator interventions. There is no second
world model. The v2 instantaneous `State` is used only as a *transient sample* that
the directive layer reads each tick; it is never authoritative and never persisted as
truth.

Concretely, the rollout is a v0/v2 `Scenario` (actors = `start` pose + a `maneuvers[]`
list) advancing on the global looping clock (`Scenario.simulate()`). Everything that
changes the future — a perturbation or an intervention — is a **maneuver-script edit**,
so the whole session stays expressible in one language and reuses the v0/v2
`Scenario`, `Actor`, `Maneuver`, and `Persistence` code directly.

The bridge to the directive layer is v2's existing adapter, run every tick:

```
state  = <sample poses+speeds from the live Scenario at clock T>   # like state_from_scenario
astate = recognize(state, prm)                                     # v2
result = evaluate_family(astate, prm)                              # v2  (D1,D2,D3, hero, t*)
rr     = repair(astate, prm)                                       # v2  minimal causal repair
```

`repair` returns v2 `Intervention(kind, actor, value, why, cost)` in state/route terms
(`retime` → target speed; `reroute` → route turn). §4 translates those back into
maneuver-script edits, closing the loop in the one representation.

## 2. The closed-loop kernel

One **orchestration tick** = `dt_tick` = **0.1 s** (P4). At tick clock `T`:

```
while running:
    sc.simulate();  state = sample(sc, T)          # maneuver world → instantaneous state
    astate = recognize(state, prm)
    result = evaluate_family(astate, prm)

    # (a) user perturbation queued for this tick?  → maneuver-script edit (§3)
    if pending_perturbation:
        sc = apply_perturbation(sc, T, pending)     # add actor / rewrite a suffix (may be the ego)
        emit_node(kind="perturbation", ...)         # DECISION POINT → snapshot + tree node

    # (b) orchestrator runs EVERY tick, but only a DECISION POINT is persisted (P3)
    rr = repair(astate, prm)
    if rr.feasible and rr.interventions and changes_standing_controls(rr):
        sc = apply_interventions(sc, T, rr)         # translate retime/reroute → maneuvers (§4)
        emit_node(kind="intervention", delta=rr.interventions)   # DECISION POINT
    elif not rr.feasible and not result.ok and newly_infeasible():
        emit_node(kind="infeasible", reason=rr.reason)           # DECISION POINT
    # else: NO-OP — orchestrator re-affirmed existing controls; nothing persisted, no node

    # (c) advance the clock one dt_tick along the (possibly edited) maneuver world
    T += dt_tick
```

Ordering is **perturb → orchestrate → advance**: the user perturbs the future, the
orchestrator gets exactly one tick to react, then the world moves. This is v2's causal
contract — edits touch only the future; the elapsed prefix is frozen (§4).

**Decision points only (P3).** The orchestrator *runs* every 0.1 s tick, but a tick is
only recorded — as a snapshot file **and** a tree node — when it is a **decision
point**: the committed world actually changes. That means a user perturbation, an
orchestrator intervention that *changes a standing command* (not merely re-affirms the
current target speed/route), a checkpoint, a proposal, or the onset of infeasibility.
The long runs of NO-OP ticks in between — where the orchestrator confirms the family
still holds under the current controls — are **collapsed**: no file, no node. The tree
and the session folder therefore contain only the moments where something was decided.

**Freezing time.** `running` is user-controlled. When frozen, the clock stops
advancing but recognition/evaluation still run, so the user can *explore the current
timeline* (scrub the forward realization) without committing ticks (§7).

**Re-basing (freeze the past).** When any edit is applied at `T`, each actor is
re-based to its pose at `T` (new `start`) with its remaining maneuver suffix; elapsed
maneuvers become history and are dropped from the live suffix. A snapshot is therefore
a self-contained scenario whose t=0 is the tick's `T` (§6).

## 3. Perturbations (user edits, in maneuver-script terms)

A **perturbation** is a user maneuver-script edit injected mid-rollout — the user
playing adversary against the orchestrator. Because it is a maneuver edit on the
shared representation, it uses the v0/v2 editor's own vocabulary:

| Perturbation | Maneuver-script effect | From the brief |
|---|---|---|
| `add_actor(leg, maneuvers)` | spawn a new actor at a leg with a maneuver list (default straight-through) | "adding an actor" |
| `set_maneuver(aid, turn)` | replace the actor's upcoming maneuver with a turn (`turn_left`/`turn_right`) | "setting its future maneuver to be an inconvenient turn" |
| `set_speed(aid, v0)` | edit the velocity curve of the current/next longitudinal maneuver | slow a lead vehicle to block the hero |
| `remove_actor(aid)` | despawn a non-ego actor | |

`set_maneuver` to an "inconvenient turn" is the headline case: it turns a background
vehicle into an **interferer** (a `blocks` / `occupies_conflict` violator of D3), or
removes the hero's collision course (a D1/D2 violator) — exactly the failures v2's
`repair` was built to clear. The orchestrator then has one tick to respond.

**Every perturbation is recorded** in the append-only session log (§6) with its tick
and payload, whether or not the orchestrator reacts, so the session's full adversarial
history is replayable.

Perturbations differ from interventions by *who* and *what*: perturbations are **user**
edits that may add/remove actors and reshape routes, **including the ego** (P2 — the
player may perturb or directly control the ego); interventions are **orchestrator**
edits computed by `repair`, restricted to `retime`/`reroute` on **non-ego** actors
(v2's rule — the orchestrator plays around the ego, never steers it). So the ego is
the one actor only the player may move: the orchestrator must keep the family holding
against whatever the player does to it. This distinction drives the tree coloring (§8).

## 4. The orchestrator (always running, persisted only at decision points)

**P3 — the orchestrator runs `repair` every 0.1 s tick, but records a node only when
it decides something.** Running continuously is what lets it act when v2's D2 margin
gets *tight* rather than when D1 finally breaks (v2 §6) — the concrete demonstration
that D2 is load-bearing. But re-affirming an already-standing command is not a
decision, so it is collapsed (§2b). Each tick:

1. `rr = repair(astate, prm)` — v2's greedy analytic repair (per-directive fixes +
   joint re-check), returning `Intervention`s or an INFEASIBLE reason.
2. **If the result changes a standing command**, translate & apply each intervention as
   a maneuver-script edit from `T` forward, on the re-based suffix (§2), and emit an
   `intervention` node (`delta` = the interventions):
   - `retime(a, v')` → set the actor's next longitudinal maneuver to `go_straight` at
     `v0 = v'` (under `--ramp`, an `accelerate`/`decelerate` ramp to `v'` then hold).
   - `reroute(a, turn)` → change the upcoming turn maneuver's type (only while the
     actor is uncommitted — before its stop line, matching v2).
3. If the result only re-affirms the current controls, the tick is a **NO-OP**: applied
   silently, no file, no node.
4. If INFEASIBLE while the family is broken (and not already flagged), emit an
   `infeasible` node with v2's reason ("the past is not available for editing").

**Manual proposal.** While frozen, the user can ask the orchestrator to *propose*
without committing (§7): it runs `repair` and materializes the post-intervention
realization as a `kind = proposal` branch — the tree equivalent of v2's
nominal-vs-intervention compare. The user then accepts (commits the maneuver edits) or
discards.

## 5. Snapshots & the game tree

**A tree node is emitted only at a decision point** (P3); NO-OP ticks are collapsed
(§2b, §4.3). Node kinds and how they color (§8):

| `kind` | Emitted when |
|---|---|
| `start` | seeding from v20 (root) |
| `intervention` | orchestrator committed a `retime`/`reroute` that changed a standing command |
| `perturbation` | user injected a maneuver edit (incl. on the ego) |
| `checkpoint` | user saved the current realization (§7) |
| `proposal` | orchestrator proposal, uncommitted |
| `infeasible` | `repair` found no causal fix while the family was broken |

**Tree structure = provenance graph** (reusing v2's `Persistence` in spirit): each
snapshot is a "version"; `parent` = the snapshot it advanced from. Branching arises
exactly as in the editor:

- **Linear progress** — consecutive ticks: each node's parent is the previous tick's.
- **Returning to a checkpoint** — the user reloads an earlier snapshot and continues;
  the next tick takes that snapshot as parent → a **branch** (a new timeline from a
  shared past). This is the "return to different saved checkpoints" requirement.
- **Proposals** — a `proposal` node hangs off the current node as a sibling to the
  eventual committed continuation; accepting it makes it the committed child.

**Depth is bounded by decisions, not ticks.** Because NO-OP ticks are never persisted
(§2b), a timeline has one node per decision point — typically a handful per rollout,
not ~65. Each edge is annotated with the elapsed sim time it spans (the collapsed
NO-OP run), so the tick gap between decisions is still visible without a node per tick.
The `scenario_tree.html` collapse/expand per subtree and lazy media loading are kept
for large branching sessions.

## 6. Session persistence (reused machinery, retargeted)

**P5 — reuse the v0/v2 `Persistence` class (versioned save + provenance graph +
append-only log), retargeted so all files belong to one rollout.** Two changes only:

1. **Folder = session id, not `scenarios/`.** The directory is
   `v4/sessions/<session_id>/`, `<session_id>` a timestamped slug
   (e.g. `20260720T144000_v20redlight`). Everything for one rollout lives here.
2. **Files are snapshots, not scenarios.** `save_version` writes `snapshot_v{N}.yaml`
   — a plain v0/v2 **maneuver-script scenario** (via `Scenario.to_dict()`), re-based to
   the tick's `T`, so it round-trips through `load_scenario` and renders forward with
   the existing pipeline. `_next_number`, `provenance.yaml`, and parent-linking are
   unchanged; `provenance.yaml` gains `kind` + `tick` + `sim_time` + a one-line `delta`
   per node (for coloring/labels). `edit_history.yaml` becomes `session_log.yaml` — the
   append-only record of every perturbation and intervention with `tick` and payload.

| v0/v2 concept | v4 counterpart |
|---|---|
| `scenarios/` folder | `sessions/<session_id>/` folder |
| `scenario_v{N}.yaml` (a scenario) | `snapshot_v{N}.yaml` (a maneuver-script realization within the rollout) |
| `provenance.yaml` (version DAG) | same file + `kind`/`tick`/`delta` per node → the game tree |
| `edit_history.yaml` (edit log) | `session_log.yaml` (perturbations + interventions) |
| `save_version(scenario)` | `save_snapshot(scenario, meta)` |
| parent = currently loaded version | parent = current decision-point node |

**`provenance.yaml` node** (schema addition):

```yaml
versions:
  - version: 34
    file: snapshot_v34.yaml         # a re-based maneuver-script scenario
    parent: 33
    created: '2026-07-20T14:41:00'
    kind: intervention              # start|nominal|intervention|perturbation|checkpoint|proposal|infeasible
    tick: 34
    sim_time: 3.40
    delta: "retime 9 -> 6.2 m/s  (clear occupies_conflict)"
    verdict: {d1: true, d2: true, d3: true, hero: '1', t_star: 4.1}
```

Because each snapshot is an ordinary scenario file, the whole session is inspectable,
diffable, and re-renderable with existing v0/v2 tools.

## 7. Interactive front end (`v4/session_editor.py`)

A **version of the pygame editor** — reusing v2 `scenario_editor.py`'s renderer, actor
hit-testing, spawn/drag handles, and top-bar widgets, plus `render_demo.py`'s
`draw_map`/`draw_actor` so the live view and the tree clips look identical — re-skinned
to drive the closed loop instead of editing a single scenario.

**Layout.**

```
+-----------------------------------------------------------------------+
| t=3.40s  [▶/⏸ run]  [. step]   Perturb: +Actor  Turn  Speed  -Actor   |  top bar
|          D1:PASS  D2:PASS  D3:FAIL   hero=1  t*=4.10   D2 slack=0.7s   |  verdict strip
+-----------------------------------------------------------------------+
|                                                                       |
|                 BEV: map + actors (live rollout)                      |  canvas
|        (frozen: actors clickable; forward realization ghosted)        |
|                                                                       |
+-----------------------------------------------------------------------+
|  Checkpoints:  [C0 start] [C1 3.4s] [C2* branch]     Propose  Accept  |  session bar
+-----------------------------------------------------------------------+
```

**Controls / keybindings.**

- `Space` — **run/freeze** the closed loop (freeze = "freeze time", per the brief).
- `.` — single-**step** one tick (only while frozen).
- While **frozen**: `←/→` scrub the forward realization (explore the current timeline
  as a prediction, no ticks committed); click an actor to inspect its recognized
  route/region/speed and which D-atoms it participates in.
- `a` / `t` / `s` / `Delete` — inject **perturbations** `add_actor` /
  `set_maneuver` (cycle the selected actor's upcoming turn) / `set_speed` /
  `remove_actor`; spawn/drag handles reused for pose. Each queues for the next tick
  (§2a) and is logged.
- `k` — **save checkpoint** (freeze + `save_snapshot(kind="checkpoint")`).
- `l` — **load checkpoint** (pick from the session bar; the loop resumes from that
  snapshot → a new branch on the next tick).
- `p` — **propose** (orchestrator computes a repair, shows the post-intervention
  realization as a ghost + a `proposal` node); `Enter` accepts, `Esc` discards.
- The **verdict strip** shows live D1/D2/D3 lamps, hero, `t*`, and D2 slack, re-run
  every tick — realizing v2's deferred "editor integration" hook.

Live-view flashes match the tree colors (§8): perturbation ticks flash amber,
intervention ticks flash cyan.

## 8. The HTML game tree (`v4/build_tree.py` → `session.html`)

Reuses `v0/outputs/scenario_tree.html` almost wholesale — the tidy upside-down layout,
collapse/expand, embedded-provenance fallback, and per-node looping media. Changes:

1. **Media per node = the realization at that decision point.** For each snapshot,
   render its **forward realization** — the re-based scenario rolled forward under its
   current maneuver world for a fixed horizon (default 6 s, or to loop end) — to a
   looping `snapshot_v{N}.mp4` via the `render_demo.py` frame pipeline (`draw_map` +
   `draw_actor` + PIL + ffmpeg). Playing a node thus plays *the scenario realization
   at that decision point* (P6: fixed-horizon forward clip, so even a single-tick node
   shows meaningful motion; consecutive `nominal` nodes with identical suffixes reuse a
   cached clip to bound render cost).

2. **Node & edge coloring by `kind`** — the "different colors for perturbations vs
   interventions" requirement:

   | `kind` | color | meaning |
   |---|---|---|
   | `start` | slate/neutral | seed state |
   | `intervention` | **cyan** `#38b2c6` | orchestrator correction |
   | `perturbation` | **amber** `#e6a23c` | user adversarial edit (incl. on the ego) |
   | `checkpoint` | **gold border** | user-saved realization |
   | `proposal` | **violet, dashed** | uncommitted orchestrator proposal |
   | `infeasible` | **red** | repair impossible (causal limit hit) |

   The edge into a node is colored by that node's kind, so a glance shows where the
   user pushed (amber) and where the orchestrator pushed back (cyan); the edge also
   carries the elapsed sim time of the collapsed NO-OP run it spans. A **legend** is
   added to the header.

3. **Node label** — `v{N}` + `kind` + `tick`/`sim_time`, plus for
   perturbations/interventions the one-line `delta` and a `D1D2D3` verdict badge.

## 9. Base scenario: v20 (red-light violation in populated traffic)

The seed is a **version of `v2/scenarios/scenario_v20.yaml`** — the red-light
violation in populated traffic: ego (id 0, green) northbound on `SE`, a red-running
approach on `EN`, plus background vehicles (turners, a stopped queue on the W arm,
crossing traffic). It is loaded with `load_scenario` as the tick-0 maneuver world;
family-consistent signals (`E,W: red`; `N,S: green`) are attached for the directive
layer. From there the world advances under the shared maneuver representation.

The nominal (no-perturbation) session should show the orchestrator **maintaining** the
red-light collision course as the populated traffic drifts — keeping D2 slack positive
with gentle `retime`s, clearing incidental D3 interferers — a continuous demonstration
that the family holds across the whole rollout, not just at t=0. The user then perturbs
(drop an inconvenient turner in front of the hero) and watches the orchestrator
recover, all captured in the tree.

## 10. File structure & deliverables

```
v4/
  DESIGN.md               # this document
  orchestrator.py         # headless closed-loop kernel: tick loop, sample→recognize→repair,
                          #   maneuver-edit translation, snapshot I/O. Imports v2 scenario_editor
                          #   + directives + a retargeted Persistence (§6). CLI: run a scripted session.
  session_editor.py       # interactive pygame front end (§7); kernel import-safe, display optional
  build_tree.py           # session folder → snapshot_v{N}.mp4 (render_demo pipeline) + session.html
  sessions/
    <session_id>/
      snapshot_v1.yaml … snapshot_v{N}.yaml   # re-based maneuver-script realizations
      provenance.yaml     # game tree (+kind/tick/delta per node)
      session_log.yaml    # append-only perturbations + interventions
      session.html        # colored, playable game tree
      snapshot_v*.mp4     # per-node forward-realization clips
```

`orchestrator.py` is headless and import-safe (no pygame at import) so a session can
be scripted and turned into a tree with no display; `session_editor.py` adds the GUI.

## 11. Resolved questions (from review)

- **Q1 — representation → RESOLVED (P1).** One shared representation: the **maneuver
  script** is ground truth. Directives evaluate on states sampled from the maneuver
  rollout each tick; orchestrator `retime`/`reroute` and user perturbations are both
  applied as maneuver-script edits; snapshots are plain maneuver-script scenario files
  (reusing `Scenario`/`Persistence`).
- **Q2 — perturbation scope → RESOLVED.** The **player may perturb or control the
  ego** (P2); the orchestrator's `repair` interventions remain non-ego (v2's rule), so
  the ego is the actor only the player moves and the orchestrator must accommodate.
- **Q3 — node granularity → RESOLVED (P3).** The orchestrator runs every 0.1 s tick
  but a node/snapshot is recorded **only at decision points** (perturbation,
  intervention that changes a standing command, checkpoint, proposal, infeasible).
  NO-OP ticks are collapsed — no file, no node — and their elapsed time is annotated on
  the connecting edge.
- **Q4 — tick length → RESOLVED (P4).** `dt_tick = 0.1 s`.
- **Q5 — node playback → RESOLVED (P6).** Fixed-horizon (~6 s) forward realization
  from each node, cached across identical consecutive suffixes.
- **Q6 — tree media → RESOLVED.** Pre-rendered looping **mp4** per node via the
  `render_demo` pipeline, for exact visual parity with the editor and
  `scenario_tree.html`.
```
