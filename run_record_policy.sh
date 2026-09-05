#!/bin/bash -l
# Boot CARLA and record ONE highway scenario with ONE learned ego policy.
#
#   ./run_record_policy.sh <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>
#
# Generalises run_record_cutin_policy.sh, which hardcoded `--scenario cutin`.
# The interpreter selection, the port derivation and the CARLA boot are
# unchanged from that script; only the scenario block below is new.
#
# Why --along-offset 110 on Town04
# --------------------------------
# Road 47 passes under an overpass between script y=-10 and y=+30. Driving there
# is fine; FILMING there is not — the bird's-eye camera looks down on the deck
# and the ego is invisible for half the run, which is what a depth sweep of the
# road showed (clear runs: y=-100..-20 and y=+40..+300). Sliding the scenario
# +110 m puts the whole run inside the 260 m clear stretch: the hard_brake ego
# starts at y=+50 and finishes near +190. Spacing, speeds and lateral layout are
# untouched, so this changes where the run is filmed and not what it measures.
# Town01 `overtake` needs no offset; its view is already clear.
#
# Scenario block
# --------------
#   cutin       scenarios/scenario_cutin_single.yaml on Town04. The single-car
#               cut-in: ego + one actor, which holds the cut-in role for the
#               whole run (no nominal traffic, so nothing to recast to). Both
#               drive nominally until t=3 (`--cutin-at 3`); the
#               `--cutin-along 9` pin is kept from the earlier recordings so the
#               videos are comparable.
#   hard_brake  scenarios/scenario_hard_brake_delayed.yaml on Town04. The lead
#               drives nominally for three seconds, THEN brakes to 4 m/s, with a
#               normal-speed car squeezing the merge gap in the next lane.
#   overtake    the authored scenario on Town01 road 8 — the stock two-way
#               straight (310 m junction-free, honest 4.00 m lanes) that the map
#               survey picks for this scenario (docs/HIGHWAY_MAPS.md). It has to
#               be a real town rather than the generated two-way map: plant2 is
#               a privileged BEV policy and renders from a prebuilt raster in
#               `carla_garage/birds_eye_view/maps_2ppm_cv/<Town>.h5`, which ships
#               only for the stock towns. A generated world reports its map name
#               as `OpenDriveMap`, there is no raster for it, and the run dies in
#               `bev.prepare` before the first tick.
#
#               Set AV_OVERTAKE_MAP=generated to use maps/highway_2lane_twoway.xodr
#               instead (600 m, nothing occluding the manoeuvre). simlingo and
#               tfv6 are camera policies and run there fine; plant2 cannot.
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
SCENARIO="${2:-cutin}"
case "${POLICY}" in
    simlingo|tfv6|plant2) ;;
    *) echo "usage: $0 <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>" >&2; exit 64 ;;
esac
case "${SCENARIO}" in
    cutin|hard_brake|overtake) ;;
    *) echo "usage: $0 <simlingo|tfv6|plant2> <cutin|hard_brake|overtake>" >&2; exit 64 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV_ROOT="/scratch/zwang179/traffic_orchestration"
source "${AV_ROOT}/install/env.sh"
source "${AV_ROOT}/third_party/env.sh"

# ---------------------------------------------------------------- scenario
# TAG names the run directory; it differs from SCENARIO for cutin so the
# single-car recordings never overwrite the earlier four-actor ones.
SCEN_ARGS=()
case "${SCENARIO}" in
    cutin)
        TAG="cutin_single"
        # --cutin-at 3: the orchestrator stays asleep until t=3, so ego and
        # actor both drive nominally first and the merge is something that
        # HAPPENS rather than something already underway at frame 0. The
        # authored deadline (`cutin: {t: 6.0}`) is unchanged, so the actor has
        # three seconds to make the pin.
        SCEN_ARGS=(--scenario cutin
                   --base "${SCRIPT_DIR}/scenarios/scenario_cutin_single.yaml"
                   --town Town04
                   --cutin-at 3
                   --cutin-along 9
                   --along-offset 110)
        ;;
    hard_brake)
        TAG="hard_brake"
        # The delayed variant: the lead cruises at 11 m/s for three seconds and
        # only then brakes to 4 m/s, so the run shows nominal driving, the
        # brake, and the ego's reaction — rather than opening with a car already
        # crawling. See the header of the YAML for its disagreement with
        # scenario_verify.py's proximity-triggered setup metric.
        SCEN_ARGS=(--scenario hard_brake
                   --base "${SCRIPT_DIR}/scenarios/scenario_hard_brake_delayed.yaml"
                   --town Town04
                   --along-offset 110)
        ;;
    overtake)
        if [ "${AV_OVERTAKE_MAP:-stock}" = "generated" ]; then
            TAG="overtake_gen"
            SCEN_ARGS=(--scenario overtake --xodr auto)
        else
            TAG="overtake"
            SCEN_ARGS=(--scenario overtake --town Town01 --road-id 8)
        fi
        ;;
esac

LOG="${AV_INSTALL}/carla_server_${TAG}_${POLICY}.log"
# Two of these can land on the same node, and a CARLA server is addressed by
# port. Derive the port from the job so concurrent runs cannot collide, and
# never blanket-kill CarlaUE4 (that would take out the other job's server).
PORT="${AV_PORT_OVERRIDE:-$(( 2000 + (${SLURM_JOB_ID:-0} % 200) * 4 ))}"
OUT_DIR="${AV_RUN}/${TAG}_${POLICY}"
VIDEO="${OUT_DIR}/${TAG}_${POLICY}.mp4"
REPORT="${OUT_DIR}/report.json"
mkdir -p "${AV_RUN}" "${OUT_DIR}"

FFMPEG_DIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin"
CARLA_PY_API="${AV_SERVER}/PythonAPI/carla"

echo "[host]     $(hostname)"
echo "[gpu]      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "[policy]   ${POLICY}"
echo "[scenario] ${SCENARIO} (tag ${TAG})"
echo "[out]      ${OUT_DIR}"

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
echo "[py]       ${PY}"

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
echo "[step] ${SCENARIO} + ${POLICY} ego"
cd "${POLICY_ROOT}"
env PATH="${PATH}" PYTHONPATH="${PYTHONPATH}" CARLA_ROOT="${CARLA_ROOT}" \
    SDL_VIDEODRIVER="${SDL_VIDEODRIVER}" \
    "${PY}" -m carla_highway \
    "${SCEN_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --ego-mode physics \
    ${AV_EGO_MODEL:+--ego-model "${AV_EGO_MODEL}"} \
    ${AV_TRAFFIC_LIGHTS:+--traffic-lights "${AV_TRAFFIC_LIGHTS}"} \
    --policy "${POLICY}" \
    --video "${VIDEO}" \
    --video-view both \
    --report "${REPORT}" \
    --verify-report "${OUT_DIR}/verify.json" \
    2>&1 | tee "${OUT_DIR}/record.log"
RC=${PIPESTATUS[0]}
echo "[step] runner rc=${RC}"

if [ -f "${VIDEO}" ]; then
    echo "[done] video: ${VIDEO}"
    ls -lh "${VIDEO}"
fi
exit "${RC}"
