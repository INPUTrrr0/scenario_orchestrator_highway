#!/bin/bash
#SBATCH --job-name=av-policy
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Record one highway scenario with one learned ego policy.
#   sbatch --job-name=av-cutin-plant2 sbatch_policy.sh plant2 cutin
#   sbatch --job-name=av-overtake-tfv6 sbatch_policy.sh tfv6 overtake
# Slurm writes %x-%j.out into the directory from which you submit.
set -uo pipefail
# Slurm COPIES the batch script to /var/spool/slurmd/job<N>/slurm_script before
# running it, so ${BASH_SOURCE[0]} is the spool copy and its dirname is not this
# repository -- `sbatch sbatch_policy.sh ...` failed with "No such file or
# directory" on run_record_policy.sh for exactly that reason. $SLURM_SUBMIT_DIR
# is where the submit happened, which is this directory in the documented usage;
# it is checked rather than trusted, and the BASH_SOURCE path still answers when
# the script is run directly from a shell.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "${_here}/run_record_policy.sh" ]; then
    SCRIPT_DIR="${_here}"
elif [ -n "${AV_HIGHWAY_ROOT:-}" ] && [ -x "${AV_HIGHWAY_ROOT}/run_record_policy.sh" ]; then
    SCRIPT_DIR="${AV_HIGHWAY_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -x "${SLURM_SUBMIT_DIR}/run_record_policy.sh" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "cannot locate run_record_policy.sh: submit from the repository, or" >&2
    echo "set \$AV_HIGHWAY_ROOT to it" >&2
    exit 66
fi
exec "${SCRIPT_DIR}/run_record_policy.sh" \
    "${1:?usage: sbatch_policy.sh <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>}" \
    "${2:?usage: sbatch_policy.sh <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>}"
