#!/bin/bash -l
#SBATCH --job-name=av-probe-occluders
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0:25:00
#SBATCH --output=%x-%j.out
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
ORCH="${SCRIPT_DIR}"
PORT=$(( 2000 + (${SLURM_JOB_ID:-0} % 200) * 4 ))
export AV_PORT_OVERRIDE="${PORT}"
av_clean_python_env
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate"
cd "${AV_RUN}"
nohup "${AV_SERVER}/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port="${PORT}" -nosound -quality-level=Low \
    >"${AV_INSTALL}/carla_server_probe_occluders.log" 2>&1 &
SERVER_PID=$!
trap 'kill -9 ${SERVER_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 90); do
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$" && break
    sleep 2
done
export PYTHONPATH="${ORCH}:${AV_SERVER}/PythonAPI/carla"
"${AV_VENV}/bin/python" "${ORCH}/scripts/probe_camera_mount.py" || true
