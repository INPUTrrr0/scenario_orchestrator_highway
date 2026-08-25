## Directory Structure

| Module | Role |
|--------|------|
| `scenario_editor.py` | Sim + pygame editor (maneuvers, Functions, straight multi-lane map, `lane_change`) |
| `maps.py` | Road layout (`MapConfig`) + pygame map drawing (straight / intersection) |
| `directives.py` | D1/D2/D3 (single-scene) |
| `directives_script.py` | Script-grounded evaluate/repair for closed-loop |
| `maneuvers.py` | Rebase / retime / reroute / perturbations |
| `orchestrator.py` | Headless closed-loop session runner |
| `cutin_orchestrator.py` | Cut-in orchestrator: role casting + closed-loop cut-in (drive the ego, side panel shows cast roles) |
| `drive.py` | Drive the ego (`--mode intersection` or `--mode cutin`) |
| `scenarios/` | `scenario_cutin.yaml`, `scenario_cutin_block.yaml`, `scenario_overtake.yaml`, `scenario_hard_brake.yaml`, `scenario_v20.yaml`, … |
| `docs/` | Design notes from former versions |
| `v5/` | Untouched proposal |

## Quick start

```bash
.venv/bin/python scenario_editor.py scenarios/scenario_cutin.yaml
.venv/bin/python scenario_editor.py scenarios/scenario_cutin.yaml --validate

.venv/bin/python drive.py --mode cutin
.venv/bin/python drive.py --mode cutin --headless --duration 8

.venv/bin/python orchestrator.py --nominal --session smoke

# cut-in orchestrator: random fleet, role casting, you drive the ego
.venv/bin/python cutin_orchestrator.py --seed 1 --actors 4

# ego-policy stress tests (you drive; scripts only set the stage)
.venv/bin/python scenario_editor.py scenarios/scenario_overtake.yaml
.venv/bin/python scenario_editor.py scenarios/scenario_hard_brake.yaml
```

## Maneuver types

`go_straight`, `lane_change` (+left / −right via `lateral_offset`), `turn_left`, `turn_right`,
`accelerate`, `decelerate`, `stop`, and `function` (node graphs). The editor type control is a **dropdown**.

## Cut-in target pin

An actor can carry a `cutin` spec instead of hand-written maneuvers:

```yaml
cutin: {t: 3.0, along: 6.0, lat: 0.0, lc_duration: 2.0, tail: 4.0}
```

**Scripted Play:** at time `t`, complete the lane change `along` metres ahead
of the ego (and `lat` to its left). Offsets are ego-relative — change the
ego's speed and the world place moves with the ego.

**Drive mode (closed-loop):** the pin is glued to the live ego and moves as
you throttle/brake/steer. The actor continuously rebases and replans toward
the *current* pin until it merges, **or abandons the cut-in once past `t`**
and cruises straight if the merge never happened. `t` is a hard deadline.

In the editor: pause and **drag the yellow pin** to set `t` + (along, lat);
press **Drive** / **F** to take the ego and watch the actor chase the moving
pin.

## Role casting (orchestrator panel)

When a scenario has a `cutin` or `block` spec and several actors (see
`scenario_cutin.yaml` or the combined `scenario_cutin_block.yaml`), the
editor shows an **orchestrator** card in
the top left: an *intention matrix* with one row per actor and one column
per intention — **none | cut-in | block**. The cell of the actor's assigned
intention gets a green light; unassigned cells stay hollow. Each cut-in /
block cell also shows the actor's live candidate score as a percentage —
how well placed it is to perform that intention *right now* (recomputed
every frame against the on-screen poses, so you can watch the next-best
candidate rise as the holder's score falls). The percentages are
display-only; casting still uses the score + feasibility logic below.
The orchestrator scores every non-ego
actor (adjacent lane, ~14 m ahead of the ego is ideal) and hands the `cutin`
spec to the best one; the previous holder goes back to cruising. Recasting
happens when you edit spawns (with 1.25× hysteresis) and, while driving, when
the current holder becomes geometrically hopeless (ego passed it or it fell
far behind). The scoring lives in `cutin_orchestrator.py` and is shared with
the standalone session.

### Collision directive (yield / replan)

While driving, the orchestrator also scans ~3 s of planned trajectories for
**actor–actor body overlap** (oriented rectangles, with a small safety pad).
On a predicted hit it replans the *lower-priority* actor and leaves the
owner's plan alone:

1. **Action owner first** — whoever currently holds `cutin` or `block`
   outranks nominal traffic.
2. **Then probability** — among owners (or among nominals), the panel
   placement score of the held / best action breaks the tie.

The yielder is sped *forward* if it is ahead of the owner (clear the merge
slot and keep a ~2 m bumper gap) or slowed if it is behind.  Self-governed
actors and an in-progress cut-in are never the ones rewritten.

This is what unblocks `experiment.py --seed 4`: actor 2 owns the cut-in, actor
1 sits in the target lane ahead, so actor 1 is pushed forward and actor 2
keeps the lane change.


### Mixed autonomy

Each panel row has a governance dropdown next to the actor's name
(clickable even while driving):

- **fully autonomous** (`auto`, default) — the actor has no independent
  plan; it follows whatever the orchestrator casts (cut-in or nominal
  cruising).
- **fully self-governed** (`self`) — the actor executes only user-issued
  intents (edited segments, a dragged cut-in pin); the orchestrator never
  conscripts it and never recasts a cut-in away from it.

The mode persists in the YAML as `autonomy: self` on the actor (omitted
when autonomous).

## Ego-policy stress tests

Two scenarios where the *actors* are fully scripted and the ego script is a
plain constant-speed profile — it encodes **no policy**. Press **Drive / F**
and drive the ego yourself (or hook up a policy under test); the scripts
only set the stage, so nothing assumes what the ego will do.

* **`scenario_overtake.yaml`** — the ego must go around a blocking object
  using the opposite lane while dealing with oncoming traffic, without
  colliding with either car. The blocker (red) drives ahead in the ego's
  lane, brakes hard at t≈2.5 s and stays stopped; an oncoming car (blue)
  runs southbound in the opposite lane and reaches the stopped car around
  t≈8 s. Overtake early (tight gap in front of the oncoming car) or brake
  and go around after it passes — both windows are deliberately tight.
* **`scenario_hard_brake.yaml`** — the ego should lane-change to evade a
  slow lead (red, 4 m/s, ~30 m ahead: ~3 s to contact at cruise speed)
  while a normal-speed lead (amber, 10.5 m/s) in the adjacent lane squeezes
  the merge gap. The ego must avoid hitting both cars while overtaking.

```bash
.venv/bin/python scenario_editor.py scenarios/scenario_overtake.yaml
.venv/bin/python scenario_editor.py scenarios/scenario_hard_brake.yaml
```

## Experiment harness (recorded trials)

`experiment.py` is a thin wrapper over the editor for collecting labelled
driving trials — it does **not** modify the simulator (it only uses two new
opt-in `run_gui` hooks, `auto_drive` and `on_frame`).  Each trial builds a
*randomized* scenario from a single seed, drops you straight into Drive mode,
and writes a JSON result.

```bash
# interactive: drive the ego (WASD / arrows); the trial auto-ends once the
# cut-in and block resolve (or --max-time), and the result is saved
.venv/bin/python experiment.py --seed 42
# no window; the ego autopilots its scripted speed (handy for batch runs)
.venv/bin/python experiment.py --seed 42 --headless --out experiments/run_42.json
```

Per trial, seeded from `--seed`:

- a random **number of actors** (1–5, uniform) is spawned at random,
  **non-overlapping** lane slots in a band around the ego (same-lane cars are
  kept ≥ 7 m apart; different lanes never overlap);
- one actor is cast as the **cut-in** and (traffic permitting) another as the
  **block**, by the orchestrator's placement scores;
- the orchestrator runs closed-loop while you drive.

The layout (spawns + casting) is fully determined by the seed, so the same
seed reproduces the same scene — the only variable is how you drive.  The
output JSON records:

- `seed`, `base_scenario`, `num_actors`, `spawns`, `cruise`, `cast`, `scores`;
- `cutin` / `block`: `{performer, outcome, success, t_commit}` — success means
  the cut-in **merged** / the block **denied the merge** (see the block grading
  above);
- `ego_trajectory` (`[t, x, y, heading_deg, v]`) and `actor_trajectories`
  (`id -> [t, x, y, heading_deg]`), sampled at `--hz` (default 20 Hz).

Results default to `experiments/run_<seed>.json`.  Flags: `--base`,
`--hz`, `--max-time`, `--min-actors`, `--max-actors`, `--headless`.
