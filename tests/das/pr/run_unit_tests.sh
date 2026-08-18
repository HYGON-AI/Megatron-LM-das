#!/usr/bin/env bash
# PR unit test entry (runs inside run-test action, after prepare_workspace):
#   env: BUCKET / N_REPEAT / DAS_HCU_CI_LOG_DIR / MEGATRON_PATH / DTK_ENV
#        DAS_COVERAGE_DISABLED (set by check_environment.sh)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

if [[ -z "${BUCKET:-}" ]]; then
    echo "ERROR: BUCKET is required" >&2
    exit 1
fi

# Install missing test deps with pinned versions as a fallback for older images.
# New CI images bake them in; this path still requires container network access.
ensure_test_deps() {
    local missing=()
    python3 -c "import pytest_mock" >/dev/null 2>&1 || missing+=("pytest-mock==3.14.0")
    python3 -c "import coverage" >/dev/null 2>&1 || missing+=("coverage==7.6.1")
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Installing missing test deps: ${missing[*]}"
        if ! pip install --quiet "${missing[@]}"; then
            echo "WARN: pip install failed; tests may fail without: ${missing[*]}" >&2
        fi
    fi

    if ! python3 -c "import pytest_mock" >/dev/null 2>&1; then
        echo "ERROR: pytest-mock is required after dependency preparation" >&2
        return 1
    fi

    # check_environment.sh may have disabled coverage before the fallback
    # installation. Re-evaluate the final environment instead of retaining a
    # stale DAS_COVERAGE_DISABLED=1 in this step.
    if python3 -c "import coverage" >/dev/null 2>&1; then
        export DAS_COVERAGE_DISABLED=0
    else
        echo "WARN: coverage is unavailable; falling back to plain pytest" >&2
        export DAS_COVERAGE_DISABLED=1
    fi
}
ensure_test_deps

mkdir -p "${DAS_HCU_CI_LOG_DIR}"

bash tests/unit_tests/run_ci_test.sh \
    --bucket "${BUCKET}" \
    --unit-test-repeat "${N_REPEAT:-1}" \
    --log-dir "${DAS_HCU_CI_LOG_DIR}"
