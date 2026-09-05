#!/bin/bash -l
# Boot CARLA and record the Town04 cut-in with a learned ego policy.
#
#   ./run_record_cutin_policy.sh simlingo|tfv6|plant2
#
# The orchestrator is unchanged between policies on purpose: the same scenario,
# the same casting, the same cut-in geometry, so the three videos differ only in
# who is driving the ego. This generalises run_record_cutin_simlingo.sh, which
# hardcoded simlingo.
#
# Interpreter, per policy
# -----------------------
# The runner drives CARLA in-process, so the CARLA python API and the policy's
# inference stack have to be importable in ONE interpreter, and the policies do
# not agree on a Python:
#
#   simlingo   py3.8  -> run the py3.8 CARLA venv, append simlingo's site-packages
#   tfv6       py3.10 -> run the policy's own venv, which carries a carla wheel
#   plant2     py3.10 -> ditto
#
set -u
POLICY="${1:-}"
case "${POLICY}" in
    simlingo|tfv6|plant2) ;;
    *) echo "usage: $0 simlingo|tfv6|plant2" >&2; exit 64 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV_ROOT="/scratch/zwang179/traffic_orchestration"
source "${AV_ROOT}/install/env.sh"
source "${AV_ROOT}/third_party/env.sh"

LOG="${AV_INSTALL}/carla_server_cutin_${POLICY}.log"
# Two of these can land on the same node, and a CARLA server is addressed by
# port. Derive the port from the job so concurrent runs cannot collide, and
# never blanket-kill CarlaUE4 (that would take out the other job's server).
PORT="${AV_PORT_OVERRIDE:-$(( 2000 + (${SLURM_JOB_ID:-0} % 200) * 4 ))}"
OUT_DIR="${AV_RUN}/cutin_${POLICY}"
VIDEO="${OUT_DIR}/cutin_${POLICY}.mp4"
REPORT="${OUT_DIR}/cutin_report.json"
mkdir -p "${AV_RUN}" "${OUT_DIR}"

FFMPEG_DIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin"
CARLA_PY_API="${AV_SERVER}/PythonAPI/carla"

echo "[host]   $(hostname)"
echo "[gpu]    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "[policy] ${POLICY}"
echo "[out]    ${OUT_DIR}"

av_clean_python_env

# ---------------------------------------------------------------- interpreter
if [ "${POLICY}" = "simlingo" ]; then
    # py3.8: the CARLA venv is the base and simlingo is appended to it.
    module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
    export PATH="${FFMPEG_DIR}:${PATH}"
    source "${AV_VENV}/bin/activate"
    PY="${AV_VENV}/bin/python"
    POLICY_ROOT="${SIMLINGO_ROOT}"
    EXTRA_PATH="${SIMLINGO_ROOT}:${SIMLINGO_VENV}/lib/python3.8/site-packages"
else
    # py3.10: the policy venv is the base and already carries carla 0.9.16.
    export PATH="${FFMPEG_DIR}:${PATH}"
    VENV_VAR="$(echo "${POLICY}" | tr 'a-z' 'A-Z')_VENV"
    ROOT_VAR="$(echo "${POLICY}" | tr 'a-z' 'A-Z')_ROOT"
    source "${!VENV_VAR}/bin/activate"
    PY="${!VENV_VAR}/bin/python"
    POLICY_ROOT="${!ROOT_VAR}"
    EXTRA_PATH="${POLICY_ROOT}"
    if [ "${POLICY}" = "tfv6" ]; then
        # The adapter's controller subclasses TransfuserAgent to reuse its
        # control selection, and that class descends from the CARLA leaderboard's
        # AutonomousAgent, so `leaderboard` and `srunner` have to be importable
        # even though no leaderboard runs here. They are vendored, not installed:
        # 3rd_party/leaderboard/<variant>/ ships a leaderboard and a
        # scenario_runner side by side. `standard` is upstream CARLA's.
        LB="${POLICY_ROOT}/3rd_party/leaderboard/${TFV6_LEADERBOARD:-standard}"
        EXTRA_PATH="${EXTRA_PATH}:${LB}/leaderboard:${LB}/scenario_runner"
    fi
fi
echo "[py]     ${PY}"

# ---------------------------------------------------------------- carla server
if ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
    echo "[fail] port ${PORT} is already in use" >&2; exit 1
fi

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

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

PYTHONPATH="${SCRIPT_DIR}:${EXTRA_PATH}:${CARLA_PY_API}:${PYTHONPATH:-}"
export PYTHONPATH
export CARLA_ROOT="${AV_SERVER}"
export SDL_VIDEODRIVER=dummy

"${PY}" - <<PY || { echo "[fail] server never answered"; exit 4; }
import carla, time, sys
c = carla.Client('127.0.0.1', int("${PORT}")); c.set_timeout(60.0)
for _ in range(24):
    try:
        print('[ready]', c.get_world().get_map().name); sys.exit(0)
    except RuntimeError:
        time.sleep(5)
sys.exit(1)
PY

# ---------------------------------------------------------------- the run
echo "[step] cut-in + ${POLICY} ego"
cd "${POLICY_ROOT}"
env PATH="${PATH}" PYTHONPATH="${PYTHONPATH}" CARLA_ROOT="${CARLA_ROOT}" \
    SDL_VIDEODRIVER="${SDL_VIDEODRIVER}" \
    "${PY}" -m carla_highway \
    --scenario cutin \
    --town Town04 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --ego-mode physics \
    --policy "${POLICY}" \
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
