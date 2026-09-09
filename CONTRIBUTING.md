# Contributing

## Branch model

| Branch | Owns | Does not own |
|--------|------|--------------|
| `main` | Simulator orchestrator algorithm, scenarios, simulator tests, metrics/verifiers | CARLA imports, policy submodules, Slurm GPU launchers |
| `carla_port_highway_mobil` | Everything on `main`, plus `carla_highway/`, shared `carla_port/`, maps, `policies/`, harness adapter, recording scripts | Junction editor history (`v0`–`v5`) |

Develop algorithm changes on `main`. After simulator CI passes, GitHub Actions opens a sync PR into `carla_port_highway_mobil`. Merge that PR only after CARLA/offline adapter checks still pass.

Collaborators reproduce CARLA experiments from the **exact commit SHA** pinned by `friedeggs/scenario_orchestration` at `third_party/orchestrator_highway`, not from floating branch tips.

## What may be pushed

Push only source, configuration, fixtures, and docs that are needed to reproduce a working result.

Do **not** push:

- run outputs, videos, session dumps, caches
- local virtualenvs, checkpoints, Hugging Face caches
- exploratory one-off scripts that are not part of the supported entrypoints
- absolute cluster paths that only work on one machine (prefer `AV_ROOT` / relative paths)

## Local checks before push

```bash
# simulator (both branches)
python -m pytest tests/test_directives.py tests/test_collision_directive.py -q
python tests/test_editor.py

# CARLA branch only — offline, no live server
python -m carla_highway.validate
```
