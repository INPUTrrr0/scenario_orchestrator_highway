#!/bin/bash -l
# Boot CARLA and record the highway cut-in scenario (ego policy + orchestrator).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../install/env.sh"

LOG="${AV_INSTALL}/carla_server_cutin.log"
PORT="${AV_PORT}"
OUT_DIR="${AV_RUN}/cutin_demo"
VIDEO="${OUT_DIR}/cutin_ego_orchestrator.mp4"
REPORT="${OUT_DIR}/cutin_report.json"
mkdir -p "${AV_RUN}" "${OUT_DIR}"

echo "[host] $(hostname)"
echo "[gpu]  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "[out]  ${OUT_DIR}"

module --force purge >/dev/null 2>&1
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate" 2>/dev/null || true

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

"${AV_VENV}/bin/python" - <<PY || { kill -9 "${SERVER_PID}" 2>/dev/null; exit 4; }
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

export PYTHONPATH="${SCRIPT_DIR}:${AV_SERVER}/PythonAPI/carla:${PYTHONPATH:-}"
FFMPEG_DIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin"

echo "[step] running cut-in (ego policy + orchestrator)"
cd "${SCRIPT_DIR}"
env PATH="${FFMPEG_DIR}:${PATH}" \
"${AV_VENV}/bin/python" -m carla_highway \
    --scenario cutin \
    --town Town04 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --ego-mode physics \
    --video "${VIDEO}" \
    --video-view both \
    --cutin-at 6 \
    --report "${REPORT}" \
    2>&1 | tee "${OUT_DIR}/record.log"
RC=${PIPESTATUS[0]}
echo "[step] runner rc=${RC}"

if [ -f "${VIDEO}" ]; then
    echo "[done] video: ${VIDEO}"
    ls -lh "${VIDEO}"
fi
exit "${RC}"
