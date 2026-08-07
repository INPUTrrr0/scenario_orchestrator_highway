# v4 — Closed-Loop Scenario Orchestration

Continuously evaluate the red-light family (D1/D2/D3 from v2) and intervene while a
scenario **rolls out**, letting you inject perturbations live and recording the whole
session as a colored, playable **game tree**. See `DESIGN.md` for the full design.

The maneuver script is the single ground truth; the directive layer predicts **along
the actual script** (`directives_script.py`), so the verdict matches what the
animation plays. Interventions and perturbations are maneuver-script edits. Only
decision points are saved; each is a re-based scenario snapshot under
`sessions/<id>/`.

## Files

- `orchestrator.py` — headless closed-loop kernel (tick loop, snapshots, provenance).
- `directives_script.py` — script-grounded D1/D2/D3 + minimal causal repair.
- `maneuvers.py` — shared maneuver surgery: rebase, retime, reroute, perturbations.
- `session_editor.py` — interactive pygame front end.
- `build_tree.py` — session folder → per-node mp4 clips + `session.html` game tree.
- `sessions/<id>/` — one rollout: `snapshot_v*.yaml`, `provenance.yaml`,
  `session_log.yaml`, `snapshot_v*.mp4`, `session.html`.

Requires `pyyaml`, `pygame` (editor), and `Pillow` + `ffmpeg` (tree rendering).
Imports the v2 `scenario_editor.py`, `directives.py`, and `render_demo.py`.

## Run a scripted session (headless)

```bash
# nominal: no perturbations — orchestrator maintains, collision realized
python3 orchestrator.py --nominal --session my_run

# with the built-in demo perturbations (add a turner, slow an actor)
python3 orchestrator.py --session my_run

# options: --base <scenario.yaml>  --time <seconds>  --signals N=green,E=red,...
```

## Build the game tree

```bash
python3 build_tree.py sessions/my_run          # renders mp4s + session.html
python3 build_tree.py sessions/my_run --no-render   # rebuild HTML only
# then open sessions/my_run/session.html   (open from the folder so clips load)
```

Node/edge colors: slate = start, **amber = user perturbation**,
**cyan = orchestrator intervention**, gold = checkpoint, violet = proposal,
green = collision realized, red = infeasible. Edges annotate the elapsed time of the
collapsed NO-OP run between decisions.

## Interactive editor

```bash
python3 session_editor.py --session my_session
```

| Key | Action |
|---|---|
| `Space` | run / freeze the loop (freeze time) |
| `.` | single step (while frozen) |
| `← / →` | scrub the forward realization (while frozen) |
| click | select an actor |
| `a` | add actor (cycles spawn leg SE→EN→WS→NW) |
| `t` | set selected actor's route (straight→left→right) — may be the ego |
| `s` / `S` | slow selected actor to 3 m/s / speed it to 12 m/s |
| `Del` | remove selected actor |
| `k` / `l` | save checkpoint / load next checkpoint (branch from it) |
| `p` → `Enter`/`Esc` | propose intervention → accept / discard |
| `q` | quit |

Perturbations may target the ego; the orchestrator's interventions never do. After a
session, run `build_tree.py` on its folder to view the tree.
