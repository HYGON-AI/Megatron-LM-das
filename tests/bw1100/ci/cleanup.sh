#!/usr/bin/env bash
# Clean up processes and temp dirs owned by the current CI run
# (patterned after verl-das tests/hcu/ci/cleanup.sh):
#   - only kills processes tagged with DAS_HCU_CI_RUN_ID (verified via
#     /proc/<pid>/environ), never unrelated ones
#   - temp dir restricted to DAS_HCU_CI_TMP_ROOT with path-safety checks
set -euo pipefail

run_id="${DAS_HCU_CI_RUN_ID:-}"
tmp_root="${DAS_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/das-hcu-ci}"

if [[ -z "${run_id}" ]]; then
    echo "No DAS_HCU_CI_RUN_ID is set; skipping owned PID and temporary directory cleanup."
    exit 0
fi
if [[ ! "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: unsafe DAS_HCU_CI_RUN_ID: ${run_id}" >&2
    exit 1
fi

tmp_root="$(realpath -m -- "${tmp_root}")"
if [[ "${tmp_root}" == "/" ]]; then
    echo "ERROR: refusing to use the filesystem root as DAS_HCU_CI_TMP_ROOT" >&2
    exit 1
fi
run_dir="${tmp_root}/${run_id}"
case "${run_dir}" in
    "${tmp_root}"/*) ;;
    *)
        echo "ERROR: refusing to clean path outside the HCU CI temporary root: ${run_dir}" >&2
        exit 1
        ;;
esac

owns_pid() {
    local candidate_pid="$1"
    [[ -r "/proc/${candidate_pid}/environ" ]] &&
        tr '\0' '\n' < "/proc/${candidate_pid}/environ" |
        grep -Fqx "DAS_HCU_CI_RUN_ID=${run_id}"
}

if [[ -d "${run_dir}/pids" ]]; then
    shopt -s nullglob
    for pid_file in "${run_dir}"/pids/*.pid; do
        pid="$(tr -d '[:space:]' < "${pid_file}")"
        if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        if ! owns_pid "${pid}"; then
            echo "Skipping PID ${pid}: process is not owned by HCU CI run ${run_id}."
            continue
        fi
        kill -TERM "${pid}" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null && owns_pid "${pid}"; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    shopt -u nullglob
fi

if [[ -d "${run_dir}" ]]; then
    rm -rf -- "${run_dir}"
fi

echo "Cleaned resources owned by HCU CI run ${run_id}."
