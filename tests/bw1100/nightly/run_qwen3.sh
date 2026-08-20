#!/usr/bin/env bash
# Nightly Qwen3-8B entry (runs inside the run-test action, after
# prepare_workspace has been sourced by the action):
#   - launches the repo example via mpirun (the example's default backend;
#     the workflow must NOT set LAUNCH_BACKEND=torchrun for this to apply)
#   - sources requirements/env.sh, which provides CUDA_DEVICE_MAX_CONNECTIONS=1
#     and the NCCL/ROCSHMEM tuning the example expects
#   - overrides socket ifnames for the container (eth0, not eth2)
#   - writes the dataset index cache to a writable location instead of the
#     read-only asset mount
#   env: DATA_PATH / TOKENIZER_MODEL_PATH / CHECKPOINT_PATH / TRAIN_ITERS /
#        TRAINING_MODE / DATA_CACHE_PATH / DTK_ENV
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

for var in DATA_PATH TOKENIZER_MODEL_PATH CHECKPOINT_PATH TRAIN_ITERS; do
    [[ -n "${!var:-}" ]] || { echo "ERROR: ${var} is required" >&2; exit 1; }
done
TRAINING_MODE="${TRAINING_MODE:-pretrain}"

# requirements/env.sh is the example's own env (NCCL_ALGO, GLOO/NCCL ifnames,
# HSA_FORCE_FINE_GRAIN_PCIE, CUDA_DEVICE_MAX_CONNECTIONS=1, ...). It derives
# MEGATRON_PATH from $0, which resolves to "/" when env.sh is sourced from
# another script, so restore it afterwards. MEGATRON_PATH is this repo root
# (that is where the entry scripts live); 3rdparty/* is only on PYTHONPATH,
# which prepare_workspace.sh owns.
source "${repo_root}/requirements/env.sh"
export MEGATRON_PATH="${repo_root}"

DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
[[ -f "${DTK_ENV}" ]] && source "${DTK_ENV}"

# Container has a single eth0; env.sh hardcodes eth2 for physical hosts.
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# Index cache location for pretrain. The asset mount is read-only, and the
# example defaults the cache next to the data files, so redirect it.
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${CHECKPOINT_PATH}/dataset-cache}"
mkdir -p "${DATA_CACHE_PATH}"

LOG_DIR="${DAS_HCU_CI_LOG_DIR:-ci-logs}"
mkdir -p "${LOG_DIR}"

HOST="$(hostname)"
HOSTFILE="/tmp/hostfile-${DAS_HCU_CI_RUN_ID:-$$}"
echo "${HOST} slots=8" > "${HOSTFILE}"

DIST_URL="${DIST_URL:-localhost}"
DIST_PORT="${DIST_PORT:-11452}"

echo "=== nightly launch ==="
echo "backend=mpirun hosts=${HOST} np=8 mode=${TRAINING_MODE} iters=${TRAIN_ITERS}"
echo "data=${DATA_PATH}"
echo "cache=${DATA_CACHE_PATH}"

# mpirun runs the train script once per rank; each rank picks up its
# OMPI_COMM_WORLD_* env vars inside the script. --bind-to none keeps MPI off
# the cores so launch_with_binding.sh can apply the NUMA mapping. The CI copy
# of the train script lives here so nightly-only args (--data_cache_path) do
# not touch the shared example.
mpirun -np 8 --hostfile "${HOSTFILE}" --allow-run-as-root --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    bash -c "
        source ${repo_root}/requirements/env.sh
        source ${DTK_ENV}
        export MEGATRON_PATH=${repo_root}
        export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
        cd ${repo_root}/tests/bw1100/nightly
        bash train_qwen3_8B.sh ${DIST_URL} ${DIST_PORT} \
            --data_path=${DATA_PATH} \
            --launch_backend=mpirun \
            --tokenizer_path=${TOKENIZER_MODEL_PATH} \
            --checkpoint_path=${CHECKPOINT_PATH} \
            --launch_with_binding=${repo_root}/requirements/launch_with_binding.sh \
            --train_iters=${TRAIN_ITERS} \
            --training_mode=${TRAINING_MODE} \
            --data_cache_path=${DATA_CACHE_PATH}
    " 2>&1 | tee "${LOG_DIR}/train_qwen3_8B.log"
exit "${PIPESTATUS[0]}"
