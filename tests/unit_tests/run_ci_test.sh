#!/bin/bash
set -euo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --bucket BUCKET [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

export UNIT_TEST_MODE=1

# DTK environment (env-driven, default /opt/dtk; skip when absent)
DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
if [ -n "$DTK_ENV" ] && [ -f "$DTK_ENV" ]; then
    # shellcheck disable=SC1090
    source "$DTK_ENV"
fi

# Conda activation only when CONDA_HOME is set (e.g. /opt/conda)
if [ -n "${CONDA_HOME:-}" ] && [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-base}"
fi

MEGATRON_PATH="${MEGATRON_PATH:-}"
if [ -z "$MEGATRON_PATH" ]; then
    echo "Error: MEGATRON_PATH is required (path to megatron checkout)" >&2
    exit 1
fi
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"

if [ -d "${MEGATRON_PATH}/tests" ]; then
    if [ -e "${MEGATRON_PATH}/tests_bak" ]; then
        echo "Error: stale Megatron-LM/tests_bak exists; refusing to overwrite it" >&2
        exit 1
    fi
    mv -- "${MEGATRON_PATH}/tests" "${MEGATRON_PATH}/tests_bak"
    # restore the submodule dir on success/failure/cancel to keep the
    # persistent runner workspace clean
    trap 'if [ -d "${MEGATRON_PATH}/tests_bak" ]; then mv -- "${MEGATRON_PATH}/tests_bak" "${MEGATRON_PATH}/tests"; fi' EXIT
fi

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_PATH}/../../"

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR="$(pwd)/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --help)
        usage
        ;;
    --bucket)
        BUCKET="$2"
        shift 2
        ;;
    --unit-test-repeat)
        UNIT_TEST_REPEAT="$2"
        shift 2
        ;;
    --unit-test-timeout)
        UNIT_TEST_TIMEOUT="$2"
        shift 2
        ;;
    --log-dir)
        LOG_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate BUCKET
if [[ -z "${BUCKET:-}" ]]; then
    echo "Error: BUCKET is required"
    usage
fi

# Validate LOG_DIR
if [[ -z "${LOG_DIR:-}" ]]; then
    echo "Error: LOG_DIR is required"
    usage
else
    mkdir -p "${LOG_DIR}"
fi

if [[ ! "${UNIT_TEST_REPEAT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --unit-test-repeat must be a positive integer" >&2
    exit 1
fi
if [[ ! "${UNIT_TEST_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --unit-test-timeout must be a positive integer" >&2
    exit 1
fi

# Set default timeout if not specified
if [[ "$UNIT_TEST_TIMEOUT" == "10" ]]; then
    UNIT_TEST_TIMEOUT=$((10 * UNIT_TEST_REPEAT))
fi

export BUCKET

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --node_rank "${NODE_RANK}"
    --log-dir "${LOG_DIR}"
    --tee "0:3"
    --redirects "3"
)

# Reduce memory usage by NCCL
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0
export ONE_LOGGER_JOB_CATEGORY=test

# Coverage is optional: DAS_COVERAGE_DISABLED=1 falls back to plain pytest.
COVERAGE_DISABLED="${DAS_COVERAGE_DISABLED:-0}"
TEST_TARGET="$(printf '%s\n' "${BUCKET}" | sed 's|/\*\*/\*\.py$||')"

for ((iteration = 1; iteration <= UNIT_TEST_REPEAT; iteration++)); do
    echo "Running unit test (${iteration}/${UNIT_TEST_REPEAT})."
    CMD=(python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")
    if [ "$COVERAGE_DISABLED" == "1" ]; then
        CMD+=(
            -m pytest
            -xvs
            "${TEST_TARGET}"
        )
    else
        # parallel mode: each worker (rank) writes its own .coverage.* file,
        # avoiding 8-process concurrent writes to one data-file; combine later
        CMD+=(
            -m coverage run
            --parallel-mode
            --source=hcu_megatron/core
            -m pytest
            -xvs
            "${TEST_TARGET}"
        )
    fi
    "${CMD[@]}"
done

if [ "$COVERAGE_DISABLED" != "1" ]; then
    # combine merges all .coverage.* files
    coverage combine -q
fi

if [ -d "${MEGATRON_PATH}/tests_bak" ]; then
    mv -- "${MEGATRON_PATH}/tests_bak" "${MEGATRON_PATH}/tests"
fi
