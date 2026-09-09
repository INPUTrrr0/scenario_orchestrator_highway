#!/bin/bash -l
#SBATCH --job-name=av-validate-scenarios
#SBATCH --account=aip-cmaddis
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=%x-%j.out
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
av_clean_python_env
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate"
exec "${AV_VENV}/bin/python" "${SCRIPT_DIR}/scripts/validate_scenarios.py"
