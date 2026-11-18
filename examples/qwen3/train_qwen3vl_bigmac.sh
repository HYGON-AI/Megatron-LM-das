#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
HF_MODEL_PATH="${HF_MODEL_PATH:-<HF_MODEL_PATH>}"
DATASET_PATH="${DATASET_PATH:-<VLM_CONFIG_JSON>}"
MCORE_CKPT_PATH="${MCORE_CKPT_PATH:-}"
SAVE_PATH="${SAVE_PATH:-}"
LOAD_PATH="${LOAD_PATH:-}"
PROFILING="${PROFILING:-none}"
REPRODUCE=0

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
bool_value() {
    case "${2,,}" in
        1|true|yes|on) printf '1' ;;
        0|false|no|off) printf '0' ;;
        *) fail "$1 must be 0/1 or false/true" ;;
    esac
}
for option in "$@"; do
    case "$option" in
        --tokenizer_path=*) HF_MODEL_PATH="${option#*=}" ;;
        --data_path=*) DATASET_PATH="${option#*=}" ;;
        --checkpoint_path=*) MCORE_CKPT_PATH="${option#*=}" ;;
        --save_path=*) SAVE_PATH="${option#*=}" ;;
        --load_path=*) LOAD_PATH="${option#*=}" ;;
        --profiling=*) PROFILING="${option#*=}" ;;
        --reproduce) REPRODUCE=1 ;;
        --launch_backend=torchrun) ;;
        --help|-h)
            printf '%s\n' \
                'HF_MODEL_PATH=./models/Qwen3-VL DATASET_PATH=./data/vlm-config.json bash examples/qwen3/train_qwen3vl_bigmac.sh' \
                'USE_PP_EXECUTOR=0 USE_BIGMAC_STYLE=0 selects native scheduling.' \
                'SAVE_PATH enables saving; LOAD_PATH enables checkpoint loading; LOAD_WEIGHTS=1 loads HF weights.' \
                'SAVE_CHECKPOINT=1 uses MCORE_CKPT_PATH or OUTPUT_BASEPATH/checkpoint.' \
                'DRY_RUN=1 prints the command without launching training.'
            exit 0 ;;
        *) fail "Unknown option: $option" ;;
    esac
done

export USE_PP_EXECUTOR="${USE_PP_EXECUTOR:-1}"
export USE_BIGMAC_STYLE="${USE_BIGMAC_STYLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export MEGATRON_LAUNCH_BACKEND=torchrun
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/3rdparty/Megatron-LM:${REPO_ROOT}/3rdparty/Megatron-Bridge/src${PYTHONPATH:+:${PYTHONPATH}}"
export OUTPUT_BASEPATH="${OUTPUT_BASEPATH:-./outputs/qwen3vl_bigmac}"
export TENSORBOARD_PATH="${TENSORBOARD_PATH:-${OUTPUT_BASEPATH}/tensorboard}"
export PP_PROFILER_OUTPUT_BASEPATH="${PP_PROFILER_OUTPUT_BASEPATH:-${REPO_ROOT}/outputs/pp_profiler/qwen3vl_bigmac}"
export MEGATRON_PP_EXECUTOR_TRACE_DIR="${MEGATRON_PP_EXECUTOR_TRACE_DIR:-${PP_PROFILER_OUTPUT_BASEPATH}/traces}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6202}"
TP="${TP:-2}"
PP="${PP:-4}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SP="$(bool_value SP "${SP:-0}")"
DRY_RUN="$(bool_value DRY_RUN "${DRY_RUN:-0}")"
LOAD_WEIGHTS="$(bool_value LOAD_WEIGHTS "${LOAD_WEIGHTS:-0}")"
SAVE_CHECKPOINT="$(bool_value SAVE_CHECKPOINT "${SAVE_CHECKPOINT:-0}")"
SAVE_HF_WEIGHTS="$(bool_value SAVE_HF_WEIGHTS "${SAVE_HF_WEIGHTS:-0}")"
SAVE_HF_WEIGHTS_DISTRIBUTED="$(bool_value SAVE_HF_WEIGHTS_DISTRIBUTED "${SAVE_HF_WEIGHTS_DISTRIBUTED:-0}")"

for name in NNODES GPUS_PER_NODE TP PP CP MBS GBS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ && "$NODE_RANK" -lt "$NNODES" ]] || fail 'NODE_RANK must be in [0, NNODES)'
if [[ "$NNODES" -gt 1 && ( "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.0.0.1 ) ]]; then
    fail 'Set MASTER_ADDR to the rendezvous host for multi-node training'
fi
if [[ "$SAVE_CHECKPOINT" == 1 && -z "$SAVE_PATH" ]]; then
    SAVE_PATH="${MCORE_CKPT_PATH:-${OUTPUT_BASEPATH}/checkpoint}"
fi
if [[ -n "$MCORE_CKPT_PATH" && -z "$SAVE_PATH" ]]; then
    printf '%s\n' 'MCORE_CKPT_PATH alone does not enable saving; set SAVE_CHECKPOINT=1 or SAVE_PATH.' >&2
fi
[[ "$SAVE_HF_WEIGHTS" == 0 || -n "$SAVE_PATH" ]] || fail 'SAVE_HF_WEIGHTS requires SAVE_PATH or SAVE_CHECKPOINT=1'
[[ "$SAVE_HF_WEIGHTS_DISTRIBUTED" == 0 || "$SAVE_HF_WEIGHTS" == 1 ]] || fail 'SAVE_HF_WEIGHTS_DISTRIBUTED requires SAVE_HF_WEIGHTS=1'

if [[ "$DRY_RUN" == 0 ]]; then
    [[ -f "${HF_MODEL_PATH}/config.json" ]] || fail 'Set HF_MODEL_PATH to a local Qwen3-VL model directory containing config.json'
    [[ -f "$DATASET_PATH" ]] || fail 'Set DATASET_PATH to a DAS VLM JSON config referencing JSONL data, not a WebDataset directory'
    [[ -z "$LOAD_PATH" || -d "$LOAD_PATH" ]] || fail "Checkpoint directory not found: $LOAD_PATH"
    [[ -d "${REPO_ROOT}/3rdparty/Megatron-LM/megatron" ]] || fail 'Missing 3rdparty/Megatron-LM checkout'
    [[ -d "${REPO_ROOT}/3rdparty/Megatron-Bridge/src/megatron/bridge" ]] || fail 'Missing 3rdparty/Megatron-Bridge checkout'
fi

MODEL_SHAPE_ARGS=()
if [[ -f "${HF_MODEL_PATH}/config.json" ]]; then
    CONFIG_ARGS=(--pipeline-parallel-size "$PP")
    if [[ -n "${MP_PP0_LAYERS:-}" ]]; then
        CONFIG_ARGS+=(--first-pipeline-num-layers "$MP_PP0_LAYERS")
    fi
    HF_MODEL_ARGS=$("$PYTHON" "${REPO_ROOT}/hcu_megatron/core/pipeline_parallel/bigmac/config.py" "$HF_MODEL_PATH" "${CONFIG_ARGS[@]}")
    read -r -a MODEL_SHAPE_ARGS <<< "$HF_MODEL_ARGS"
else
    MODEL_SHAPE_ARGS=('<HF_MODEL_ARGS_FROM_CONFIG>')
    printf '%s\n' 'DRY_RUN: HF structure/MoE parameters remain placeholders until HF_MODEL_PATH is provided.' >&2
fi

ARGS=(
    --seq-length "${SEQ_LEN:-2048}"
    "${MODEL_SHAPE_ARGS[@]}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-262144}"
    --normalization RMSNorm
    --use-bridge --bridge-hf-model "$HF_MODEL_PATH"
    --transformer-impl transformer_engine --use-mcore-models
    --micro-batch-size "$MBS" --global-batch-size "$GBS"
    --train-iters "$TRAIN_ITERS"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --adam-beta1 0.9 --adam-beta2 0.95
    --init-method-std "${INIT_METHOD_STD:-0.02}" --clip-grad 1.0
    --bf16 --disable-bias-linear --attention-dropout 0 --hidden-dropout 0 --swiglu
    --lr "${LR:-1.0e-5}" --min-lr "${MIN_LR:-1.0e-6}" --lr-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS:-0}" --lr-decay-iters "${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
    --ckpt-format torch --ddp-average-in-collective --overlap-grad-reduce
    --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP" --use-distributed-optimizer
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$HF_MODEL_PATH"
    --bigmac-data-compat
    --dataloader-type external --vlm-data-config-path "$DATASET_PATH"
    --model-arch qwen3vl --processor-path "$HF_MODEL_PATH" --split 949,50,1
    --log-throughput --log-interval 1
    --eval-iters "${EVAL_ITERS:-0}" --eval-interval "${EVAL_INTERVAL:-1000}"
    --save-interval "${SAVE_INTERVAL:-1000}" --tensorboard-dir "$TENSORBOARD_PATH"
)
if [[ "$(bool_value USE_FLASH_ATTN "${USE_FLASH_ATTN:-0}")" == 1 ]]; then
    ARGS+=(--use-flash-attn)
fi
ATTENTION_BACKEND="${ATTENTION_BACKEND-flash}"
[[ -z "$ATTENTION_BACKEND" ]] || ARGS+=(--attention-backend "$ATTENTION_BACKEND")
case "${RECOMPUTE_GRANULARITY:-selective}" in
    none) ;;
    selective) ARGS+=(--recompute-granularity selective --recompute-modules core_attn) ;;
    *) fail 'RECOMPUTE_GRANULARITY must be none or selective' ;;
esac
if [[ " ${MODEL_SHAPE_ARGS[*]} " == *' --num-experts '* ]]; then
    ARGS+=(--expert-model-parallel-size "${EP:-1}" --expert-tensor-parallel-size "${ETP:-1}"
        --moe-grouped-gemm --moe-token-dispatcher-type alltoall
        --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.001)
    if [[ "$TP" -gt 1 && "$SP" == 0 ]]; then
        printf '%s\n' "[Qwen3-VL config] MoE with TP=${TP} requires SP=1; enabling sequence parallelism." >&2
        SP=1
    fi
fi
[[ "$SP" == 0 ]] || ARGS+=(--sequence-parallel)
[[ "$LOAD_WEIGHTS" == 0 ]] || ARGS+=(--load-weights)
[[ -z "$SAVE_PATH" ]] || ARGS+=(--save "$SAVE_PATH")
[[ -z "$LOAD_PATH" ]] || ARGS+=(--load "$LOAD_PATH")
[[ "$SAVE_HF_WEIGHTS" == 0 ]] || ARGS+=(--save-hf-weights)
[[ "$SAVE_HF_WEIGHTS_DISTRIBUTED" == 0 ]] || ARGS+=(--save-hf-weights-distributed)
if [[ "$REPRODUCE" == 1 ]]; then
    NUM_WORKERS=0
    export MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC=1 ROCBLAS_ATOMICS_MOD=0
    export MIOPEN_DEBUG_CONV_FFT=0 MIOPEN_DEBUG_CONV_DIRECT=0 MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_WINOGRAD=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    ARGS+=(--reproduce)
fi
ARGS+=(--num-workers "$NUM_WORKERS")

case "$PROFILING" in
    none) ;;
    torch|hip)
        read -r -a PROFILE_RANKS <<< "${PROFILE_RANKS:-0}"
        ARGS+=(--profile --profile-ranks "${PROFILE_RANKS[@]}"
            --profile-step-start "${PROFILE_STEP_START:-3}" --profile-step-end "${PROFILE_STEP_END:-4}")
        if [[ "$PROFILING" == torch ]]; then
            ARGS+=(--use-pytorch-profiler --profile-dir "${OUTPUT_BASEPATH}/torch_prof_tp${TP}-pp${PP}-cp${CP}")
        else
            ARGS+=(--use-hip-profiler)
        fi ;;
    *) fail 'PROFILING must be none, torch or hip' ;;
esac
COMMAND=("$PYTHON" -m torch.distributed.run
    --nnodes "$NNODES" --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"
    --nproc_per_node "$GPUS_PER_NODE"
    "${REPO_ROOT}/pretrain_vlm.py" "${ARGS[@]}")
if [[ "$PROFILING" == hip ]]; then
    COMMAND=(hipprof -d "${OUTPUT_BASEPATH}/hip_prof_data" --hip-trace --trace-off "${COMMAND[@]}")
fi
printf '%q ' "${COMMAND[@]}"
printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0
if [[ "$PROFILING" == hip ]]; then
    mkdir -p "${OUTPUT_BASEPATH}/hip_prof_data"
fi
exec "${COMMAND[@]}"

