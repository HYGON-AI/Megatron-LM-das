#!/usr/bin/env bash
# Nightly Qwen3-8B training validation with real, read-only assets.
# NIGHTLY_MODE=pretrain consumes an indexed .bin/.idx dataset.
# NIGHTLY_MODE=sft consumes train.jsonl and valid.jsonl.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

mode="${NIGHTLY_MODE:?NIGHTLY_MODE must be pretrain or sft}"
asset_root="${DAS_HCU_ASSET_ROOT:?DAS_HCU_ASSET_ROOT must point to the mounted nightly assets}"

if [[ "${mode}" != "pretrain" && "${mode}" != "sft" ]]; then
    echo "ERROR: unsupported NIGHTLY_MODE: ${mode}" >&2
    exit 1
fi
if [[ ! -d "${asset_root}" ]]; then
    echo "ERROR: nightly asset root does not exist: ${asset_root}" >&2
    exit 1
fi

resolve_model_path() {
    if [[ -n "${DAS_QWEN3_8B_MODEL_PATH:-}" ]]; then
        printf '%s\n' "${DAS_QWEN3_8B_MODEL_PATH}"
        return
    fi

    local config
    local -a candidates=()
    while IFS= read -r -d '' config; do
        if python3 - "${config}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
is_qwen3_8b = (
    config.get("model_type") == "qwen3"
    and config.get("hidden_size") == 4096
    and config.get("num_hidden_layers") == 36
)
raise SystemExit(0 if is_qwen3_8b else 1)
PY
        then
            candidates+=("$(dirname "${config}")")
        fi
    done < <(find "${asset_root}" -maxdepth 4 -type f -name config.json -print0)

    if [[ ${#candidates[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one Qwen3-8B model under ${asset_root}; found ${#candidates[@]}." >&2
        printf '  %s\n' "${candidates[@]:-Set DAS_QWEN3_8B_MODEL_PATH explicitly.}" >&2
        return 1
    fi
    printf '%s\n' "${candidates[0]}"
}

resolve_pretrain_prefix() {
    if [[ -n "${DAS_QWEN3_PRETRAIN_DATA_PATH:-}" ]]; then
        local configured="${DAS_QWEN3_PRETRAIN_DATA_PATH}"
        configured="${configured%.bin}"
        configured="${configured%.idx}"
        printf '%s\n' "${configured}"
        return
    fi

    local index_file prefix
    local -a candidates=()
    while IFS= read -r -d '' index_file; do
        prefix="${index_file%.idx}"
        if [[ -f "${prefix}.bin" ]]; then
            candidates+=("${prefix}")
        fi
    done < <(find "${asset_root}" -maxdepth 5 -type f -name '*.idx' -print0)

    if [[ ${#candidates[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one indexed pretraining dataset under ${asset_root}; found ${#candidates[@]}." >&2
        printf '  %s\n' "${candidates[@]:-Set DAS_QWEN3_PRETRAIN_DATA_PATH explicitly.}" >&2
        return 1
    fi
    printf '%s\n' "${candidates[0]}"
}

resolve_sft_dir() {
    if [[ -n "${DAS_QWEN3_SFT_DATA_PATH:-}" ]]; then
        printf '%s\n' "${DAS_QWEN3_SFT_DATA_PATH}"
        return
    fi

    local train_file data_dir
    local -a candidates=()
    while IFS= read -r -d '' train_file; do
        data_dir="$(dirname "${train_file}")"
        if [[ -f "${data_dir}/valid.jsonl" ]]; then
            candidates+=("${data_dir}")
        fi
    done < <(find "${asset_root}" -maxdepth 5 -type f -name train.jsonl -print0)

    if [[ ${#candidates[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one SFT dataset directory under ${asset_root}; found ${#candidates[@]}." >&2
        printf '  %s\n' "${candidates[@]:-Set DAS_QWEN3_SFT_DATA_PATH explicitly.}" >&2
        return 1
    fi
    printf '%s\n' "${candidates[0]}"
}

model_path="$(resolve_model_path)"
if [[ ! -f "${model_path}/config.json" ]]; then
    echo "ERROR: Qwen3-8B config.json is missing from ${model_path}" >&2
    exit 1
fi

mkdir -p "${DAS_HCU_CI_LOG_DIR}"

common_args=(
    --use-bridge
    --bridge-hf-model "${model_path}"
    --load-weights
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --num-query-groups 8
    --group-query-attention
    --seq-length "${DAS_QWEN3_SEQ_LENGTH:-1024}"
    --max-position-embeddings 40960
    --swiglu
    --qk-layernorm
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 1000000
    --untie-embeddings-and-output-weights
    --transformer-impl transformer_engine
    --use-mcore-models
    --tensor-model-parallel-size "${DAS_QWEN3_TP:-2}"
    --pipeline-model-parallel-size "${DAS_QWEN3_PP:-2}"
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --micro-batch-size "${DAS_QWEN3_MICRO_BATCH_SIZE:-1}"
    --global-batch-size "${DAS_QWEN3_GLOBAL_BATCH_SIZE:-8}"
    --train-iters "${DAS_QWEN3_TRAIN_ITERS:-5}"
    --lr 3.0e-5
    --lr-decay-style constant
    --min-lr 3.0e-5
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --ckpt-format torch
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn
    --no-load-optim
    --no-load-rng
    --eval-iters 0
    --log-interval 1
    --num-workers 2
)

data_args=()
if [[ "${mode}" == "pretrain" ]]; then
    pretrain_prefix="$(resolve_pretrain_prefix)"
    if [[ ! -f "${pretrain_prefix}.bin" || ! -f "${pretrain_prefix}.idx" ]]; then
        echo "ERROR: pretraining data requires ${pretrain_prefix}.bin and ${pretrain_prefix}.idx" >&2
        exit 1
    fi
    data_args=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${model_path}"
        --data-path "${pretrain_prefix}"
        --split "949,50,1"
    )
else
    sft_dir="$(resolve_sft_dir)"
    if [[ ! -f "${sft_dir}/train.jsonl" || ! -f "${sft_dir}/valid.jsonl" ]]; then
        echo "ERROR: SFT data requires train.jsonl and valid.jsonl under ${sft_dir}" >&2
        exit 1
    fi
    data_args=(
        --sft
        --sft-tokenizer-prompt-format default
        --tokenizer-type SFTTokenizer
        --tokenizer-model "${model_path}"
        --no-create-attention-mask-in-dataloader
        --train-data-path "${sft_dir}/train.jsonl"
        --valid-data-path "${sft_dir}/valid.jsonl"
    )
fi

echo "Running Qwen3-8B ${mode} nightly validation for ${DAS_QWEN3_TRAIN_ITERS:-5} iterations."

LAUNCH_BACKEND=torchrun python3 -m torch.distributed.run \
    --nproc_per_node "${GPUS_PER_NODE:-8}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "${MASTER_ADDR:-localhost}" \
    --master_port "${MASTER_PORT:-29500}" \
    pretrain_gpt.py \
    "${common_args[@]}" \
    "${data_args[@]}"
