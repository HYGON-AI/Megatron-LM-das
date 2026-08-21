#!/bin/bash
# CI-maintained copy of examples/qwen3/train_qwen3_8B.sh. Kept separate so
# nightly can pass --data_cache_path without touching the shared example.
# Re-sync from examples/ when upstream example changes.
INITIALIZATION_ARGS=( --num-workers 2)

for para in "$@"
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --launch_backend* ]];then
        launch_backend=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    elif [[ $para == --training_mode* ]];then
        training_mode=${para#*=}
    elif [[ $para == --train_iters* ]];then
        train_iters=${para#*=}
    elif [[ $para == --data_cache_path* ]];then
        data_cache_path=${para#*=}
    elif [[ $para == --reproduce* ]];then
        INITIALIZATION_ARGS=( --reproduce --num-workers 0)
        export MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC=1  # miopen 确定算法打开
        export ROCBLAS_ATOMICS_MOD=0                     # rocblas 关闭原子操作
        # 关闭miopen中的atomic操作算法, 只保留gemm算法
        export MIOPEN_DEBUG_CONV_FFT=0
        export MIOPEN_DEBUG_CONV_DIRECT=0
        export MIOPEN_DEBUG_CONV_GEMM=1
        export MIOPEN_DEBUG_CONV_WINOGRAD=0
        export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    fi
done

# data path
DATA_PATH=${data_path:-${DATA_PATH:-}}
TOKENIZER_MODEL_PATH=${tokenizer_path:-${TOKENIZER_MODEL_PATH:-}}
CHECKPOINT_PATH=${checkpoint_path:-${CHECKPOINT_PATH:-/tmp/qwen3-8b}}
TRAINING_MODE=${training_mode:-${TRAINING_MODE:-pretrain}}
LOAD_HF_WEIGHTS=${LOAD_HF_WEIGHTS:-false}

if [[ -z "${DATA_PATH}" || -z "${TOKENIZER_MODEL_PATH}" ]]; then
    echo "DATA_PATH and TOKENIZER_MODEL_PATH must be set"
    exit 1
fi
if [[ "${TRAINING_MODE}" != "pretrain" && "${TRAINING_MODE}" != "sft" ]]; then
    echo "TRAINING_MODE must be pretrain or sft"
    exit 1
fi
# 运行环境参数
DIST_URL=${1:-${DIST_URL:-localhost}}
DIST_PORT=${2:-${DIST_PORT:-29500}}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export LAUNCH_BACKEND=${launch_backend:-${LAUNCH_BACKEND:-mpirun}}

MASTER_ADDR=${MASTER_ADDR:-loadlhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
NODE_RANK=${NODE_RANK:?"NODE_RANK must be set (or run under mpirun which provides OMPI_COMM_WORLD_RANK/PMI_RANK)"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Respect an externally provided MEGATRON_PATH (CI runs this copy from
# tests/bw1100/nightly, where the default computation resolves too shallow).
MEGATRON_PATH=${MEGATRON_PATH:-$( dirname $( dirname ${CURRENT_DIR}))}

# default env
export GPU_MAX_HW_QUEUES=4

# split hyperparameters
TP=2
PP=2
CP=1

# batch hyperparameters
MBS=1
GBS=32

# seq hyperparameters
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=40960

# train iteration hyperparameters
TRAIN_ITERS=${train_iters:-${TRAIN_ITERS:-500}}
LR_WARMUP_ITERS=1
if [[ ! "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_ITERS must be a positive integer"
    exit 1
fi

# Optional writable location for dataset index caches. Defaults to a
# cache/ dir next to the data files, which CI mounts read-only.
DATA_CACHE_PATH=${data_cache_path:-${DATA_CACHE_PATH:-}}

MPI_DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)
TORCH_DISTRIBUTED_ARGS=(
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --nproc_per_node $GPUS_PER_NODE
)

BRIDGE_ARGS=()
if [[ "${LOAD_HF_WEIGHTS}" == "true" ]]; then
    BRIDGE_ARGS=(
        --use-bridge
        --bridge-hf-model ${TOKENIZER_MODEL_PATH}
        --load-weights
    )
fi

GPT_MODEL_ARGS=(
    ${BRIDGE_ARGS[@]}
    --seq-length ${SEQ_LEN}
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 12288 
    --num-attention-heads 32
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-query-groups 8
    --group-query-attention

    --swiglu
    --qk-layernorm
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models 
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --lr 3.0e-5 
    --lr-decay-style cosine 
    --min-lr 3.0e-6
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --ckpt-format torch
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn

    # --fp8-format hybrid
    # --fp8-recipe tensorwise
    # --fp8-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --use-distributed-optimizer 
    --sequence-parallel
)

if [[ "${TRAINING_MODE}" == "pretrain" ]]; then
    DATA_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${TOKENIZER_MODEL_PATH}
        --data-path ${DATA_PATH}
        --split 949,50,1
    )
    if [[ -n "${DATA_CACHE_PATH}" ]]; then
        DATA_ARGS+=( --data-cache-path ${DATA_CACHE_PATH} )
    fi
else
    DATA_ARGS=(
        --sft
        --sft-tokenizer-prompt-format default
        --tokenizer-type SFTTokenizer
        --tokenizer-model ${TOKENIZER_MODEL_PATH}
        --no-create-attention-mask-in-dataloader
        --train-data-path ${DATA_PATH}/train.jsonl
        --valid-data-path ${DATA_PATH}/valid.jsonl
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 5
    --log-interval 1
    --save-interval 1000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" 
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 4
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_qwen3_8B_tp${TP}-pp${PP}-cp${CP}
    --use-pytorch-profiler
    --pytorch-profiler-collect-callstack
    --record-memory-history
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

if [[ "$LAUNCH_BACKEND" == "mpirun" ]]; then
    APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${MPI_DISTRIBUTED_ARGS[@]} \
        ${INITIALIZATION_ARGS[@]} \
        ${FP8_PARALLEL_ARGS[@]} \
        "
elif [[ "$LAUNCH_BACKEND" == "torchrun" ]]; then
    APP="torchrun ${TORCH_DISTRIBUTED_ARGS[@]} \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${INITIALIZATION_ARGS[@]} \
        ${FP8_PARALLEL_ARGS[@]} \
        "
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi
if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

#for hygon cpu
if [[ "$LAUNCH_BACKEND" == "mpirun" ]]; then
    ${launch_with_binding} ${LOCAL_RANK} ${APP}
elif [[ "$LAUNCH_BACKEND" == "torchrun" ]]; then
    echo ${APP}
    ${APP}
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi
