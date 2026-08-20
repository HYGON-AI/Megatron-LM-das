#!/usr/bin/env bash
# Restore runner workspace ownership (patterned after verl-das):
# files written as root inside the job container are chown'ed back to the
# runner user via a short docker run, so later checkouts do not fail.
#
# Required env: GITHUB_WORKSPACE / RUNNER_TEMP / DAS_HCU_CI_IMAGE
set -euo pipefail

if [[ -z "${GITHUB_WORKSPACE:-}" || -z "${RUNNER_TEMP:-}" || -z "${DAS_HCU_CI_IMAGE:-}" ]]; then
    echo "ERROR: GITHUB_WORKSPACE, RUNNER_TEMP, and DAS_HCU_CI_IMAGE are required" >&2
    exit 1
fi

workspace="$(realpath -m -- "${GITHUB_WORKSPACE}")"
runner_temp="$(realpath -m -- "${RUNNER_TEMP}")"
work_root="$(dirname -- "${runner_temp}")"

case "${workspace}" in
    "${work_root}"/*/*) ;;
    *)
        echo "ERROR: refusing to change ownership outside the runner work root: ${workspace}" >&2
        exit 1
        ;;
esac

owner="$(stat -c '%u:%g' -- "${runner_temp}")"
if [[ ! "${owner}" =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
    echo "ERROR: unsafe runner ownership derived from ${runner_temp}: ${owner}" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is required to restore runner workspace ownership" >&2
    exit 1
fi

docker run --rm \
    --volume "${workspace}:/workspace" \
    "${DAS_HCU_CI_IMAGE}" \
    chown -R -- "${owner}" /workspace
echo "Restored ${workspace} ownership to ${owner}."
