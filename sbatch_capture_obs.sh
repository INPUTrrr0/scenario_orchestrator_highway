#!/bin/bash
#SBATCH --job-name=av-capture-obs
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=%x-%j.out
# Capture the observations at each policy's failure moment, for offline replay.
#   sbatch sbatch_capture_obs.sh simlingo cutin      0,1,2,5,10
#   sbatch sbatch_capture_obs.sh tfv6 hard_brake     60,70,80,85,90,95,100
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
POLICY="${1:?policy}"; SCEN="${2:?scenario}"; STEPS="${3:?steps}"
export AV_DUMP_OBS="${AV_INSTALL:-${AV_ROOT}/install}/run_output/_obs/${POLICY}_${SCEN}"
export AV_DUMP_STEPS="${STEPS}"
rm -rf "${AV_DUMP_OBS}"; mkdir -p "${AV_DUMP_OBS}"
echo "[capture] dumping steps ${STEPS} to ${AV_DUMP_OBS}"
exec "${SCRIPT_DIR}/run_record_policy.sh" "${POLICY}" "${SCEN}"
