#!/usr/bin/env bash
# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
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

# Install missing test deps with pinned versions (container needs network):
#   - pytest-mock missing -> mocker-fixture cases fail honestly
#   - coverage missing   -> falls back to plain pytest (DAS_COVERAGE_DISABLED)
ensure_test_deps() {
    local missing=()
    python3 -c "import pytest_mock" >/dev/null 2>&1 || missing+=("pytest-mock==3.14.0")
    python3 -c "import coverage" >/dev/null 2>&1 || missing+=("coverage==7.6.1")
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Installing missing test deps: ${missing[*]}"
        if ! pip install --quiet "${missing[@]}"; then
            echo "WARN: pip install failed; tests may fail without: ${missing[*]}" >&2
            for dep in "${missing[@]}"; do
                [[ "${dep}" == coverage* ]] && export DAS_COVERAGE_DISABLED=1
            done
        fi
    fi
}
ensure_test_deps

mkdir -p "${DAS_HCU_CI_LOG_DIR}"

bash tests/unit_tests/run_ci_test.sh \
    --bucket "${BUCKET}" \
    --unit-test-repeat "${N_REPEAT:-1}" \
    --log-dir "${DAS_HCU_CI_LOG_DIR}"
