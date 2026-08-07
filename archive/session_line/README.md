# Session editor UI line (parked)

Moved here 2026-08-06. Interactive session UI + tree rendering only — not the
orchestrator kernel or directives (those stay live at package root).

| File | Role |
|------|------|
| `session_editor.py` | Interactive orchestration UI |
| `realization_editor.py` | Embeddable scenario pane used by the session editor |
| `build_tree.py` | Session → HTML game tree |
| `render_demo.py` | Offline mp4 render helpers (used by `build_tree`) |
| `sessions/` | Saved rollouts (`my_session`, `smoke_flat`, …) |

These still import live root modules (`scenario_editor`, `maps`, `directives*`,
`maneuvers`, `orchestrator`). To run later:

```bash
cd archive/session_line
PYTHONPATH=../.. ../../.venv/bin/python session_editor.py --session my_session
```
