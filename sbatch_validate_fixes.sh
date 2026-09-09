#!/bin/bash -l
#SBATCH --job-name=av-validate-fixes
#SBATCH --account=aip-cmaddis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=%x-%j.out
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
ORCH="${SCRIPT_DIR}"
CHECK="${ORCH}/scripts/validate_fixes.py"
RC=0

echo "############ py3.8 CARLA venv (carla_port, highway_ego, simlingo adapter)"
av_clean_python_env
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate"
PYTHONPATH="${ORCH}:${AV_SERVER}/PythonAPI/carla" "${AV_VENV}/bin/python" "${CHECK}" || RC=1
deactivate 2>/dev/null || true

echo
echo "############ py3.10 tfv6 venv (lead's radar preprocessing)"
av_clean_python_env
source "${TFV6_VENV}/bin/activate"
LB="${TFV6_ROOT}/3rd_party/leaderboard/${TFV6_LEADERBOARD:-standard}"
PYTHONPATH="${ORCH}:${TFV6_ROOT}:${LB}/leaderboard:${LB}/scenario_runner" \
    "${TFV6_VENV}/bin/python" "${CHECK}" || RC=1

exit "${RC}"
