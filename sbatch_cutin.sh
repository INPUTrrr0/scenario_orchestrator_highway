#!/bin/bash -l
#SBATCH --job-name=highway-cutin
#SBATCH --account=aip-six
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/zwang179/traffic_orchestration/install/sbatch-cutin-%j.out
#SBATCH --error=/scratch/zwang179/traffic_orchestration/install/sbatch-cutin-%j.err

exec /scratch/zwang179/traffic_orchestration/scenario_orchestrator_highway/run_record_cutin.sh
