#!/bin/bash -l
#SBATCH --job-name=av-b2d-simlingo
#SBATCH --account=aip-cmaddis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Run SimLingo through ITS OWN Bench2Drive eval, the way the paper does.
#
# This is the differential test for the crawl: the same checkpoint, the same
# agent file, the same leaderboard — only the harness differs. If it drives here
# and crawls in our port, the defect is in our observation pipeline. If it
# crawls here too, it is the checkpoint or this environment.
#
#   sbatch sbatch_bench2drive.sh bench2drive_160
set -u
ROUTE_NAME="${1:-bench2drive_160}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"
REPO="${SIMLINGO_ROOT}"
CKPT="${AV_CKPT}/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
ROUTE="${REPO}/leaderboard/data/bench2drive_split/${ROUTE_NAME}.xml"
OUT="${AV_RUN}/_b2d/${ROUTE_NAME}"
rm -rf "${OUT}"; mkdir -p "${OUT}"
PORT=$(( 2000 + (${SLURM_JOB_ID:-0} % 200) * 4 ))
TM_PORT=$(( PORT + 2 ))

av_clean_python_env
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
export PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin:${PATH}"
source "${AV_VENV}/bin/activate"
PY="${AV_VENV}/bin/python"
SIMLINGO_SITE="${SIMLINGO_VENV}/lib/python3.8/site-packages"

export CARLA_ROOT="${AV_SERVER}"
export SCENARIO_RUNNER_ROOT="${REPO}/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="${REPO}/Bench2Drive/leaderboard"
export PYTHONPATH="${REPO}:${REPO}/team_code:${SIMLINGO_SITE}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${CARLA_ROOT}/PythonAPI/carla"
export SAVE_PATH="${OUT}/viz"
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "${SAVE_PATH}"

echo "[b2d] route=${ROUTE_NAME} town=$(grep -ho 'town="[^"]*"' "${ROUTE}" | head -1)"
echo "[b2d] checkpoint=${CKPT}"
echo "[b2d] port=${PORT}"

# NO CARLA server here on purpose: Bench2Drive's leaderboard_evaluator starts
# and owns one itself (it launched on -carla-rpc-port=2115 when we passed
# --port=2112). Running a second instance on the same GPU just competes for
# memory, and peeking at it is what made an earlier, healthy run look stalled.
( while true; do echo "[hb] $(date +%H:%M:%S) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 30; done ) &
HB=$!
trap 'kill -9 ${HB} 2>/dev/null || true; pkill -u "$USER" -f CarlaUE4-Linux-Shipping 2>/dev/null || true' EXIT

cd "${REPO}"
"${PY}" -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --routes="${ROUTE}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${OUT}/result.json" \
    --timeout=600 \
    --agent="${REPO}/team_code/agent_simlingo.py" \
    --agent-config="${CKPT}" \
    --traffic-manager-seed=1 \
    --port="${PORT}" \
    --traffic-manager-port="${TM_PORT}" 2>&1
echo "[b2d] evaluator rc=$?"
[ -f "${OUT}/result.json" ] && "${PY}" -c "
import json;d=json.load(open('${OUT}/result.json'))
for r in d.get('_checkpoint',{}).get('records',[]):
    s=r.get('scores',{})
    print('status:', r.get('status'))
    print('driving score:', s.get('score_composed'), 'route completion:', s.get('score_route'), 'infractions:', s.get('score_penalty'))
    print('infractions:', {k:v for k,v in (r.get('infractions') or {}).items() if v})
"
