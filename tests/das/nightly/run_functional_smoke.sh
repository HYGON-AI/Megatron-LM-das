#!/usr/bin/env bash
# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Nightly / L1 functional smoke entry (runs inside run-test action):
#   1. synthetic data (gen_toy_data.sh)
#   2. tiny GPT end-to-end training (pretrain_gpt.py, DAS entry point)
#
# Env:
#   DAS_SMOKE_TRAIN_ITERS   training iterations (L1=5, nightly=20)
#   DAS_TRANSFORMER_IMPL    transformer impl (default transformer_engine)
#   DAS_TOY_VOCAB_SIZE      synthetic vocab size (default 256)
#   DAS_TOY_NUM_DOCS        synthetic document count (default 2000)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

mkdir -p "${DAS_HCU_CI_LOG_DIR}"

toy_root="${DAS_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/das-hcu-ci}/${DAS_HCU_CI_RUN_ID}"
mkdir -p "${toy_root}"

# 1) synthetic data
bash tests/das/ci/gen_toy_data.sh \
    --output-prefix "${toy_root}/toy" \
    --num-docs "${DAS_TOY_NUM_DOCS:-2000}"

# 2) tiny model training smoke (default transformer_engine, matching production;
#    the `local` branch in gpt_builders.py currently passes enable_hyper_connection,
#    which this pinned megatron version does not support)
# NOTE: das parse_args branches on LAUNCH_BACKEND (default mpirun overwrites
# RANK with the --rank default of -1), so torchrun launches MUST set
# LAUNCH_BACKEND=torchrun. This applies to your own training scripts too.
LAUNCH_BACKEND=torchrun python3 -m torch.distributed.run \
    --nproc_per_node "${GPUS_PER_NODE:-8}" \
    --nnodes 1 \
    --master_addr localhost \
    --master_port "${DAS_SMOKE_MASTER_PORT:-29500}" \
    pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 4 \
    --seq-length 64 \
    --max-position-embeddings 64 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters "${DAS_SMOKE_TRAIN_ITERS:-5}" \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 1 \
    --eval-global-batch-size 32 \
    --data-path "${toy_root}/toy" \
    --tokenizer-type NullTokenizer \
    --vocab-size "${DAS_TOY_VOCAB_SIZE:-256}" \
    --bf16 \
    --transformer-impl "${DAS_TRANSFORMER_IMPL:-transformer_engine}" \
    --use-mcore-models \
    --no-load-optim \
    --no-load-rng \
    --lr 1.0e-4 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --min-lr 1.0e-5 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --distributed-backend nccl \
    --data-cache-path "${toy_root}/cache" \
    --num-workers 0 \
    2>&1 | tee -a "${DAS_HCU_CI_LOG_DIR}/pretrain.log"

echo "Functional smoke passed (iters=${DAS_SMOKE_TRAIN_ITERS:-5}, impl=${DAS_TRANSFORMER_IMPL:-transformer_engine})."
