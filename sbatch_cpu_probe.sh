#!/bin/bash -l
#SBATCH --job-name=av-cpu-probe
#SBATCH --account=aip-cmaddis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=%x-%j.out
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate"
exec "${AV_VENV}/bin/python" "$@"
