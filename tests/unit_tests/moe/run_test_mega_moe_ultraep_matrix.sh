#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node EP8 runner for the MegaMoE x UltraEP matrix test.
#
#   bash tests/unit_tests/moe/run_mega_moe_ultraep_matrix.sh
#   MEGAMOE_MATRIX_SEQ_LEN=2048 bash tests/unit_tests/moe/run_mega_moe_ultraep_matrix.sh
#   MEGAMOE_MATRIX_VARIANTS=baseline,megamoe_ultraep bash tests/.../run_mega_moe_ultraep_matrix.sh
#
# Extra arguments are forwarded to pytest, e.g. `-k test_performance`.

set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CURRENT_DIR}/../../.." && pwd)"

DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
if [[ -f "${DTK_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${DTK_ENV}"
fi
# NOTE: env.sh derives its own CURRENT_DIR/MEGATRON_PATH from "$0" and exports
# them, so sourcing it from here clobbers those names and points PYTHONPATH at
# the wrong tree. Everything below therefore uses REPO_ROOT and rebuilds
# PYTHONPATH afterwards.
# shellcheck disable=SC1091
source "${REPO_ROOT}/requirements/env.sh"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

# Topology / rocSHMEM settings UltraEP needs inside one node. The test also
# sets these itself, this keeps a manual `torchrun ...` invocation consistent.
export HSA_USE_SVM=0
export MAX_NUM_NVL_PEERS=8
export ROCSHMEM_BACKEND=ipc
export ROCSHMEM_HEAP_SIZE="${MEGAMOE_MATRIX_ROCSHMEM_HEAP_SIZE:-2147483648}"
unset ROCSHMEM_GDA_PROVIDER || true
export GPU_MAX_HW_QUEUES=10
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-12377}"

cd "${REPO_ROOT}"
# -p no:cacheprovider: eight ranks share one working tree and would otherwise
# race writing .pytest_cache.
exec python3 -m torch.distributed.run \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --master-port="${MASTER_PORT}" \
    -m pytest -s -vv -p no:cacheprovider \
    tests/unit_tests/moe/test_mega_moe_ultraep_matrix.py "$@"
