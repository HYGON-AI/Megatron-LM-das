#!/usr/bin/env bash
# PR unit test entry (runs inside run-test action, after prepare_workspace):
#   env: BUCKET / N_REPEAT / DAS_HCU_CI_LOG_DIR / MEGATRON_PATH / DTK_ENV
#        DAS_COVERAGE_DISABLED (set by check_environment.sh)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

# --- c211-i108431 read-only HCU device-identity diagnostic (transient; read-only) ---
{
  echo "C211_I108431_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
  echo "C211_HOSTNAME=$(hostname 2>/dev/null || true)"
  echo "C211_ID=$(id 2>/dev/null || true)"
  echo "C211_UNAME=$(uname -a 2>/dev/null || true)"
  echo "C211_RUNNER_NAME=${RUNNER_NAME:-} C211_GITHUB_JOB=${GITHUB_JOB:-} C211_RUN_ID=${GITHUB_RUN_ID:-}"
  echo "--- C211 hy-smi --showuniqueid ---"
  if command -v hy-smi >/dev/null 2>&1; then hy-smi --showuniqueid 2>&1 || true; else echo "C211_NO_HY_SMI"; fi
  echo "--- C211 hy-smi -a ---"
  if command -v hy-smi >/dev/null 2>&1; then hy-smi -a 2>&1 | head -150 || true; fi
  echo "--- C211 sysfs device identity ---"
  for f in /sys/class/drm/card*/device/unique_id /sys/class/drm/card*/device/device /sys/class/drm/card*/device/vendor /sys/class/drm/card*/device/serial; do
    if [ -e "$f" ]; then printf 'C211_SYSFS %s=%s\n' "$f" "$(cat "$f" 2>/dev/null | tr '\n' ' ' || true)"; fi
  done
  echo "--- C211 sysfs drm ---"
  ls -l /sys/class/drm 2>/dev/null | head -24 || true
  echo "--- C211 devnodes ---"
  ls -l /dev/kfd /dev/mkfd /dev/dri 2>/dev/null || true
  echo "--- C211 lspci ---"
  if command -v lspci >/dev/null 2>&1; then lspci -nn 2>/dev/null | grep -i -E 'hygon|1d94' || echo "C211_NO_LSPCI_MATCH"; else echo "C211_NO_LSPCI"; fi
  echo "C211_I108431_END"
} || true
# --- end c211-i108431 block ---


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
