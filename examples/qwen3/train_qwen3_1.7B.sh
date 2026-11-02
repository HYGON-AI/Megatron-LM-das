#!/bin/bash

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
    elif [[ $para == --reproduce* ]];then
        # 统计每一层的前反向参数和梯度均值
        INITIALIZATION_ARGS=( --reproduce --num-workers 0 --seed 1234 --dataloader-type single --finetune)
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
CHECKPOINT_PATH=${checkpoint_path:-${CHECKPOINT_PATH:-/tmp/qwen3-1.7b}}
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
export MEGATRON_LAUNCH_BACKEND=${launch_backend:--mpirun}

MASTER_ADDR=${MASTER_ADDR:-loadlhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
NODE_RANK=${NODE_RANK:?"NODE_RANK must be set (or run under mpirun which provides OMPI_COMM_WORLD_RANK/PMI_RANK)"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# default env
export GPU_MAX_HW_QUEUES=4

# split hyperparameters
# Qwen3-1.7B (28 layers, hidden 2048) 全参数量很小，8*64G 单卡即可放下
# （bf16权重+梯度+分布式优化器状态 估算每卡 ~10GB），TP=1/PP=1 把 8 张卡
# 全部留给数据并行（DP=8），吞吐和显存都最有富余。
TP=1
PP=1
CP=1

# batch hyperparameters
MBS=1
GBS=32

# seq hyperparameters
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=40960

# train iteration hyperparameters
TRAIN_ITERS=${train_iters:-${TRAIN_ITERS:-1500}}
LR_WARMUP_ITERS=1
if [[ ! "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_ITERS must be a positive integer"
    exit 1
fi

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


GPT_MODEL_ARGS=(
    --seq-length ${SEQ_LEN}
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-query-groups 8
    --group-query-attention

    --swiglu
    --qk-layernorm
    --normalization RMSNorm
    --position-embedding-type rope
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
    --rotary-base 1000000
    --lr 3.0e-5
    --lr-decay-style cosine
    --min-lr 3.0e-6
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --ckpt-format torch
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn

    # --fp8-format hybrid
    # --fp8-recipe blockwise # tensorwise
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
else
    DATA_ARGS=(
        --sft
        --sft-tokenizer-prompt-format default
        --tokenizer-type SFTTokenizer
        --tokenizer-model ${TOKENIZER_MODEL_PATH}
        --no-create-attention-mask-in-dataloader
        --train-data-path ${DATA_PATH}/train.jsonl
        --valid-data-path ${DATA_PATH}/valid.jsonl
        --finetune
    )
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 5
    --log-interval 1
    --save-interval 50
    --eval-interval 1000
    # --save $CHECKPOINT_PATH
    # --no-save-optim
    # --no-save-rng
    # --load $CHECKPOINT_PATH
    # --no-load-optim
    # --no-load-rng
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_qwen3_1.7B_tp${TP}-pp${PP}-cp${CP}
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

if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
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
elif [[ "$MEGATRON_LAUNCH_BACKEND" == "torchrun" ]]; then
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
if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
    ${launch_with_binding} ${LOCAL_RANK} ${APP}
elif [[ "$MEGATRON_LAUNCH_BACKEND" == "torchrun" ]]; then
    echo ${APP}
    ${APP}
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi
