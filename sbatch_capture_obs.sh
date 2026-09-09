#!/bin/bash
#SBATCH --job-name=av-capture-obs
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/zwang179/traffic_orchestration/third_party/logs/%x-%j.out
# Capture the observations at each policy's failure moment, for offline replay.
#   sbatch sbatch_capture_obs.sh simlingo cutin      0,1,2,5,10
#   sbatch sbatch_capture_obs.sh tfv6 hard_brake     60,70,80,85,90,95,100
set -uo pipefail
POLICY="${1:?policy}"; SCEN="${2:?scenario}"; STEPS="${3:?steps}"
export AV_DUMP_OBS="/scratch/zwang179/traffic_orchestration/install/run_output/_obs/${POLICY}_${SCEN}"
export AV_DUMP_STEPS="${STEPS}"
rm -rf "${AV_DUMP_OBS}"; mkdir -p "${AV_DUMP_OBS}"
echo "[capture] dumping steps ${STEPS} to ${AV_DUMP_OBS}"
exec /scratch/zwang179/traffic_orchestration/scenario_orchestrator_meta_repo/third_party/orchestrator_highway/run_record_policy.sh "${POLICY}" "${SCEN}"
