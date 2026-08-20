#!/usr/bin/env bash
# Nightly Qwen3-VL-8B SFT entry (runs inside the run-test action, after
# prepare_workspace has been sourced by the action):
#   - launches the repo VL example via torchrun (the example's torchrun
#     branch; the workflow must NOT set LAUNCH_BACKEND for this to apply)
#   - sources requirements/env.sh, which provides CUDA_DEVICE_MAX_CONNECTIONS=1
#     and the NCCL/ROCSHMEM tuning the example expects
#   - overrides socket ifnames for the container (eth0, not eth2)
#   env: DATA_PATH / TOKENIZER_MODEL_PATH / CHECKPOINT_PATH / TRAIN_ITERS /
#        DTK_ENV
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

for var in DATA_PATH TOKENIZER_MODEL_PATH CHECKPOINT_PATH TRAIN_ITERS; do
    [[ -n "${!var:-}" ]] || { echo "ERROR: ${var} is required" >&2; exit 1; }
done

# requirements/env.sh is the example's own env (NCCL_ALGO, GLOO/NCCL ifnames,
# HSA_FORCE_FINE_GRAIN_PCIE, CUDA_DEVICE_MAX_CONNECTIONS=1, ...). It derives
# MEGATRON_PATH from $0, which resolves to "/" when env.sh is sourced from
# another script, so restore it afterwards. MEGATRON_PATH is this repo root:
# the VL entry ${MEGATRON_PATH}/pretrain_vlm.py must be this repo one (it
# registers --vlm-data-config-path / --model-arch / --processor-path via
# hcu_megatron), not the upstream copy under 3rdparty/. PYTHONPATH is owned by
# prepare_workspace.sh; env.sh only prepends to it, so it is left alone.
source "${repo_root}/requirements/env.sh"
export MEGATRON_PATH="${repo_root}"

DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
[[ -f "${DTK_ENV}" ]] && source "${DTK_ENV}"

# Container has a single eth0; env.sh hardcodes eth2 for physical hosts.
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

LOG_DIR="${DAS_HCU_CI_LOG_DIR:-ci-logs}"
mkdir -p "${LOG_DIR}"

# The example resolves the rank from the launcher's env (OMPI_* under mpirun).
# Under torchrun those are absent, so the single-node layout has to be exported
# explicitly -- the example hard-fails on an unset NODE_RANK.
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

echo "=== nightly VL launch ==="
echo "backend=torchrun nodes=${NNODES} gpus=${GPUS_PER_NODE} iters=${TRAIN_ITERS}"
echo "data_config=${DATA_PATH}"

cd "${repo_root}/tests/bw1100/nightly"
bash train_qwen3vl_8B.sh \
    "${MASTER_ADDR}" "${MASTER_PORT}" \
    --data_path=${DATA_PATH} \
    --launch_backend=torchrun \
    --tokenizer_path=${TOKENIZER_MODEL_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --launch_with_binding=${repo_root}/requirements/launch_with_binding.sh \
    --train_iters=${TRAIN_ITERS} 2>&1 | tee "${LOG_DIR}/train_qwen3vl_8B.log"
exit "${PIPESTATUS[0]}"
