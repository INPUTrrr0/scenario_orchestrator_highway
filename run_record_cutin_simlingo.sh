#!/bin/bash -l
# Boot CARLA and record cut-in with SimLingo as ego (orchestrator unchanged).
#
# Uses carla-venv2 for CARLA 0.9.16 + SimLingo's site-packages for torch/VLA.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=av_env.sh
source "${SCRIPT_DIR}/av_env.sh"
source_av_env "${SCRIPT_DIR}"

LOG="${AV_INSTALL}/carla_server_cutin_simlingo.log"
PORT="${AV_PORT:-2000}"
OUT_DIR="${AV_RUN}/cutin_simlingo"
VIDEO="${OUT_DIR}/cutin_simlingo.mp4"
REPORT="${OUT_DIR}/cutin_report.json"
PY="${AV_VENV}/bin/python"
SIMLINGO_SITE="${SIMLINGO_VENV}/lib/python3.8/site-packages"
mkdir -p "${AV_RUN}" "${OUT_DIR}"

echo "[host] $(hostname)"
echo "[gpu]  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "[out]  ${OUT_DIR}"
echo "[py]   ${PY} (CARLA 0.9.16) + SimLingo site-packages"

av_clean_python_env
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
export PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin:${PATH}"
source "${AV_VENV}/bin/activate"

pkill -u "$USER" -f CarlaUE4-Linux-Shipping 2>/dev/null || true
sleep 3

cd "${AV_RUN}"
nohup "${AV_SERVER}/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port="${PORT}" -nosound -quality-level=Low \
    >"${LOG}" 2>&1 &
SERVER_PID=$!

LISTENING=0
for i in $(seq 1 90); do
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$" && LISTENING=1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -80 "${LOG}"; exit 2; }
    sleep 2
done
[ "${LISTENING}" -eq 1 ] || { tail -120 "${LOG}"; kill -9 "${SERVER_PID}" 2>/dev/null; exit 3; }

"${PY}" - <<PY || { kill -9 "${SERVER_PID}" 2>/dev/null; exit 4; }
import carla, time, sys
c = carla.Client('127.0.0.1', int("${PORT}")); c.set_timeout(60.0)
for _ in range(24):
    try:
        print('[ready]', c.get_world().get_map().name); sys.exit(0)
    except RuntimeError:
        time.sleep(5)
sys.exit(1)
PY

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONPATH="${SCRIPT_DIR}:${SIMLINGO_ROOT}:${SIMLINGO_SITE}:${AV_SERVER}/PythonAPI/carla:${PYTHONPATH:-}"
export CARLA_ROOT="${AV_SERVER}"
export SDL_VIDEODRIVER=dummy
FFMPEG_DIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin"

echo "[step] cut-in + SimLingo ego"
cd "${SIMLINGO_ROOT}"
env PATH="${FFMPEG_DIR}:${PATH}" \
    PYTHONPATH="${PYTHONPATH}" \
    CARLA_ROOT="${CARLA_ROOT}" \
    SDL_VIDEODRIVER="${SDL_VIDEODRIVER}" \
    "${PY}" -m carla_highway \
    --scenario cutin \
    --town Town04 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --ego-mode physics \
    --policy simlingo \
    --video "${VIDEO}" \
    --video-view both \
    --cutin-along 9 \
    --report "${REPORT}" \
    --verify-report "${OUT_DIR}/cutin_verify.json" \
    2>&1 | tee "${OUT_DIR}/record.log"
RC=${PIPESTATUS[0]}
echo "[step] runner rc=${RC}"

if [ -f "${VIDEO}" ]; then
    echo "[done] video: ${VIDEO}"
    ls -lh "${VIDEO}"
fi
exit "${RC}"
