#!/bin/bash
#SBATCH --job-name=av-policy
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/zwang179/traffic_orchestration/third_party/logs/%x-%j.out
# Record one highway scenario with one learned ego policy.
#   sbatch --job-name=av-cutin-plant2 sbatch_policy.sh plant2 cutin
#   sbatch --job-name=av-overtake-tfv6 sbatch_policy.sh tfv6 overtake
set -uo pipefail
exec /scratch/zwang179/traffic_orchestration/scenario_orchestrator_meta_repo/third_party/orchestrator_highway/run_record_policy.sh \
    "${1:?usage: sbatch_policy.sh <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>}" \
    "${2:?usage: sbatch_policy.sh <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>}"
