#!/bin/bash -l
#SBATCH --job-name=av-replay
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:25:00
#SBATCH --output=%x-%j.out
# Offline replay of a captured observation. No CARLA server.
#   sbatch sbatch_replay.sh <simlingo|tfv6|plant2> <dump_dir> [extra args...]
set -u
POLICY="${1:?policy}"; DUMP="${2:?dump dir}"; shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
ORCH="${SCRIPT_DIR}"
av_clean_python_env
FF=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin
if [ "${POLICY}" = "simlingo" ]; then
    module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
    export PATH="${FF}:${PATH}"; source "${AV_VENV}/bin/activate"
    PY="${AV_VENV}/bin/python"; EXTRA="${SIMLINGO_ROOT}:${SIMLINGO_VENV}/lib/python3.8/site-packages"
else
    export PATH="${FF}:${PATH}"
    V="$(echo "${POLICY}" | tr 'a-z' 'A-Z')_VENV"; R="$(echo "${POLICY}" | tr 'a-z' 'A-Z')_ROOT"
    source "${!V}/bin/activate"; PY="${!V}/bin/python"; EXTRA="${!R}"
    if [ "${POLICY}" = "tfv6" ]; then
        LB="${!R}/3rd_party/leaderboard/${TFV6_LEADERBOARD:-standard}"
        EXTRA="${EXTRA}:${LB}/leaderboard:${LB}/scenario_runner"
    fi
fi
export PYTHONPATH="${ORCH}:${EXTRA}:${AV_SERVER}/PythonAPI/carla:${PYTHONPATH:-}"
export CARLA_ROOT="${AV_SERVER}"; export SDL_VIDEODRIVER=dummy
# simlingo's adapter resolves `pretrained/<variant>/conversation.py` against the
# CWD (hydra.utils.to_absolute_path), so the replay has to run where the runner
# runs: the policy repository root.
R_VAR="$(echo "${POLICY}" | tr 'a-z' 'A-Z')_ROOT"
cd "${!R_VAR}"
exec "${PY}" "${ORCH}/scripts/replay_obs.py" "${DUMP}" --policy "${POLICY}" "$@"
