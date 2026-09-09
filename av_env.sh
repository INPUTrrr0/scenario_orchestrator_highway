# Resolve the traffic_orchestration workspace root and source its env files.
#
# The root is the directory that contains both `install/env.sh` and
# `third_party/env.sh`. That works for both layouts:
#   <root>/scenario_orchestrator_highway/
#   <root>/scenario_orchestrator_meta_repo/third_party/orchestrator_highway/
#
# Override with AV_ROOT when the layout differs. Source this file after setting
# SCRIPT_DIR to the caller's directory (or pass the start directory as $1).

resolve_av_root() {
    local start="${1:-${SCRIPT_DIR:-$(pwd)}}"
    if [ -n "${AV_ROOT:-}" ] && [ -f "${AV_ROOT}/install/env.sh" ]; then
        printf '%s\n' "${AV_ROOT}"
        return 0
    fi
    local d
    d="$(cd "${start}" && pwd)"
    while [ "${d}" != "/" ]; do
        if [ -f "${d}/install/env.sh" ] && [ -f "${d}/third_party/env.sh" ]; then
            printf '%s\n' "${d}"
            return 0
        fi
        d="$(dirname "${d}")"
    done
    return 1
}

source_av_env() {
    local start="${1:-${SCRIPT_DIR:-$(pwd)}}"
    AV_ROOT="$(resolve_av_root "${start}")" || {
        echo "AV_ROOT not found from ${start}." >&2
        echo "Set AV_ROOT to the tree that contains install/env.sh and third_party/env.sh." >&2
        return 1
    }
    export AV_ROOT
    # shellcheck disable=SC1090,SC1091
    source "${AV_ROOT}/install/env.sh"
    # shellcheck disable=SC1090,SC1091
    source "${AV_ROOT}/third_party/env.sh"
}
