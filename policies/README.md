# `policies/` — the ego policies under test

Four ego policies, as standalone runnable checkouts. This is the single home for
the repositories; the shared install machinery (interpreters, venvs, downloaded
weights, caches, build scripts) stays in `../third_party/`, and the harness
reaches these through symlinks.

```
policies/
├── simlingo/   RenzKa/simlingo         @ 743b243   CVPR'25 VLA, CARLA
├── tfv6/       kesai-labs/lead         @ 197fb5dd  TransFuser v6, CARLA
├── ltfv6/      autonomousvision/lead   @ 730bc1a2  cvpr2026 branch — NAVSIM, see below
└── plant2 ->   ../scenario_orchestrator_meta_repo/third_party/plant2
                friedeggs/plant2        @ 928b919   PlanT 2.0, CARLA
```

Source `../third_party/env.sh` before running anything: it exports
`SIMLINGO_ROOT`, `TFV6_ROOT`, `PLANT2_ROOT` and `LTFV6_ROOT`, which is what
`scenario_orchestration/policies.py`'s `resolve_repository()` honours, plus the
cache redirections that keep model downloads out of `$HOME`.

## Why the venvs did not move

`third_party/venvs/` stayed put deliberately. A virtualenv bakes absolute paths
into its shebangs, its `pyvenv.cfg`, and — for anything installed with
`pip install -e` — into a generated finder module. `tfv6` is installed that way,
and its `__editable___lead_1_5_0_finder.py` carried about 130 absolute paths
into the old location; those were rewritten as part of the move, and the
post-move verification below is what confirms it worked. Moving the venvs
themselves would have meant rebuilding them.

## Why `plant2` is a symlink and the others are not

`plant2` is a **git submodule of the harness** (`.gitmodules` declares
`third_party/plant2`). Moving its checkout out from under git and leaving a
symlink in its place turns a gitlink into a symlink and permanently dirties
`git status` in the harness. So the real checkout stays where git put it and
`policies/plant2` links to it.

`simlingo` and `tfv6` are **not** submodules — the harness ships them as empty
placeholder directories — so those are real directories here, and the harness's
`third_party/{simlingo,tfv6}` are the symlinks. One copy of each on disk either
way; the arrow just points whichever direction keeps git honest.

## LTFv6

`LTFv6` is **Latent TransFuser v6** — the camera-only variant of TFv6, with the
LiDAR branch replaced by a positional encoding (`docs/source/transfuser_versions.md`).
The weights here are `ln2697/tfv6_navsim`, the **NAVSIM** checkpoint accompanying
the LEAD paper.

The checkout is pinned to a **different branch** from `tfv6`, and that is the
whole reason it is a separate checkout:

| | branch | commit | has NAVSIM workspace |
|---|---|---|---|
| `policies/tfv6` | `main` | `197fb5dd` (v1.5.0) | no |
| `policies/ltfv6` | `cvpr2026` | `730bc1a2` | yes — `3rd_party/navsim_workspace/navsim{v1.1,v2.2}` |

`main` contains no NAVSIM code at all; `cvpr2026` contains 372 NAVSIM files,
including the two prepared workspaces the model card names as *"the only
configurations we have validated end-to-end against the reported numbers"*.
Sharing one checkout between TFv6 and LTFv6 would therefore have meant choosing
one branch and losing the other.

### What this means for running it

* **Its own benchmark is NAVSIM, not CARLA.** The validated path is
  `3rd_party/navsim_workspace/navsimv{1.1,2.2}`, which needs the NAVSIM/nuPlan
  data. The HF repo ships 191 sample files (`data/*/transfuser_{feature,target}.gz`)
  plus `example.ipynb`, which is enough for a smoke test but not for a benchmark
  number.
* **`ltfv6.py` is standalone** (108 KB, the whole model definition), so the
  checkpoint can be stepped without the repository — see `requirements.txt`
  next to it.
* **Coordinate frame is a trap.** The model card is explicit: it was trained in
  CARLA's left-handed frame (x-forward, **y-right**), not the ISO 8855
  convention NAVSIM/nuPlan use. Waypoints and headings out of `ltfv6.py` need
  `y -> -y`, `yaw -> -yaw` before anything downstream consumes them. The
  prepared workspaces already do this; a hand-rolled integration must.

Weights live in `../third_party/checkpoints/ltfv6/` (307 MB, `model_0060.pth`).

## Verification

```bash
sbatch ../third_party/sbatch_verify.sh     # load checkpoint + one forward pass
```
