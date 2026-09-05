#!/bin/bash -l
#SBATCH --job-name=av-probe-sl
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/zwang179/traffic_orchestration/third_party/logs/%x-%j.out
# Ask simlingo, in its own training vocabulary, what it sees and what it plans.
# No CARLA server: this replays observations already captured by _dump.
#   sbatch sbatch_probe_simlingo.sh <dump_dir> [--steps 60,120] [--json out.json]
set -u
DUMP="${1:?usage: sbatch_probe_simlingo.sh <dump_dir> [args...]}"; shift
AV_ROOT="/scratch/zwang179/traffic_orchestration"
ORCH="${AV_ROOT}/scenario_orchestration_repo/third_party/orchestrator_highway"
source "${AV_ROOT}/install/env.sh"; source "${AV_ROOT}/third_party/env.sh"
av_clean_python_env
FF=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
export PATH="${FF}:${PATH}"
source "${AV_VENV}/bin/activate"
PY="${AV_VENV}/bin/python"
export PYTHONPATH="${ORCH}:${SIMLINGO_ROOT}:${SIMLINGO_VENV}/lib/python3.8/site-packages:${AV_SERVER}/PythonAPI/carla:${PYTHONPATH:-}"
export CARLA_ROOT="${AV_SERVER}"; export SDL_VIDEODRIVER=dummy
# hydra.utils.to_absolute_path resolves `pretrained/<variant>` against the CWD,
# so this has to run where the runner runs.
cd "${SIMLINGO_ROOT}"
exec "${PY}" "${ORCH}/scripts/probe_simlingo_lanes.py" "${DUMP}" "$@"
