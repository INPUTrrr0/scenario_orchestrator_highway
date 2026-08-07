# Scenario Editor (flattened)

Unified package merging former **v0–v4** features. **`v5/` is untouched** (proposal only).

## What's here

| Module | Role |
|--------|------|
| `scenario_editor.py` | Sim + pygame editor (maneuvers, Functions, straight multi-lane map, `lane_change`) |
| `maps.py` | Road layout (`MapConfig`) + pygame map drawing (straight / intersection) |
| `directives.py` | Red-light family D1/D2/D3 (single-scene) |
| `directives_script.py` | Script-grounded evaluate/repair for closed-loop |
| `maneuvers.py` | Rebase / retime / reroute / perturbations |
| `orchestrator.py` | Headless closed-loop session runner |
| `cutin_orchestrator.py` | Cut-in orchestrator: role casting + closed-loop cut-in (drive the ego, side panel shows cast roles) |
| `drive.py` | Drive the ego (`--mode intersection` or `--mode cutin`) |
| `scenarios/` | `scenario_v20.yaml`, `scenario_cutin.yaml`, `scenario_v1.yaml` |
| `docs/` | Design notes from former versions |
| `v5/` | Untouched proposal |
| `archive/` | Former `v0`–`v4` trees; `session_line/` holds parked session UI / build_tree |

## Quick start

```bash
.venv/bin/python scenario_editor.py scenarios/scenario_cutin.yaml
.venv/bin/python scenario_editor.py scenarios/scenario_cutin.yaml --validate

.venv/bin/python drive.py --mode cutin
.venv/bin/python drive.py --mode cutin --headless --duration 8

.venv/bin/python orchestrator.py --nominal --session smoke

# cut-in orchestrator: random fleet, role casting, you drive the ego
.venv/bin/python cutin_orchestrator.py --seed 1 --actors 4
```

Session editor UI / build_tree / render_demo live under `archive/session_line/` (parked).

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

When a scenario has a `cutin` spec and several actors (see
`scenario_cutin.yaml`), the editor shows an **orchestrator** card in the top
left: one row per actor with a round dot — **green** = cast as the cut-in,
**grey** = nominal (cruise straight). The orchestrator scores every non-ego
actor (adjacent lane, ~14 m ahead of the ego is ideal) and hands the `cutin`
spec to the best one; the previous holder goes back to cruising. Recasting
happens when you edit spawns (with 1.25× hysteresis) and, while driving, when
the current holder becomes geometrically hopeless (ego passed it or it fell
far behind). The scoring lives in `cutin_orchestrator.py` and is shared with
the standalone session.
