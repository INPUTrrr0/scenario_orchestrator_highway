#!/bin/bash
#SBATCH --job-name=av-cutin-policy
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/zwang179/traffic_orchestration/third_party/logs/%x-%j.out
# Record the Town04 cut-in with one learned ego policy.
#   sbatch --job-name=av-cutin-plant2 sbatch_cutin_policy.sh plant2
set -uo pipefail
exec /scratch/zwang179/traffic_orchestration/scenario_orchestrator_meta_repo/third_party/orchestrator_highway/run_record_cutin_policy.sh "${1:?usage: sbatch_cutin_policy.sh simlingo|tfv6|plant2}"
