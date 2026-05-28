#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --bucket BUCKET [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH/../../

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs

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
    mkdir -p $LOG_DIR
fi

# Set default timeout if not specified
if [[ "$UNIT_TEST_TIMEOUT" == "10" ]]; then
    UNIT_TEST_TIMEOUT=$((10 * UNIT_TEST_REPEAT))
fi

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
TEST_PATH=$( dirname $( dirname ${CURRENT_DIR}))

cd $TEST_PATH

export BUCKET

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_NODEID:-${SLURM_NODEID:-0}}
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
    --log-dir $LOG_DIR
    --tee "0:3"
    --redirects "3"
)

# Reduce memory usage by NCCL
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0
export ONE_LOGGER_JOB_CATEGORY=test

for i in $(seq $UNIT_TEST_REPEAT); do
    echo "Running unit test."
    CMD=$(echo python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} \
        -m coverage run \
        --data-file=.coverage.unit_tests \
        --source=megatron/core \
        -m pytest \
        -xvs \
        $(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||'))
    eval "$CMD"

done

coverage combine -q
