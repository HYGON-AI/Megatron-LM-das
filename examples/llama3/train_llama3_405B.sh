#!/bin/bash

INITIALIZATION_ARGS=( --num-workers 2)

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --launch_backend* ]];then
        launch_backend=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
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
DATA_PATH=${data_path}
TOKENIZER_MODEL_PATH=${tokenizer_path}
CHECKPOINT_PATH=${checkpoint_path}

# 运行环境参数
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export LAUNCH_BACKEND=${launch_backend:-"mpirun"}
MASTER_ADDR=${MASTER_ADDR:-loadlhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# default env
export GPU_MAX_HW_QUEUES=4

# split hyperparameters
TP=8
PP=16
CP=2
FIRST_PP_LAYERS=7
LAST_PP_LAYERS=7

# batch hyperparameters
MBS=1
GBS=1152

# seq hyperparameters
SEQ_LEN=8192
MAX_POSITION_EMBEDDINGS=131072

# train iteration hyperparameters
TRAIN_ITERS=10000
LR_WARMUP_ITERS=8000

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
    --num-layers 126
    --hidden-size 16384
    --ffn-hidden-size 53248
    --num-attention-heads 128
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --group-query-attention
    --num-query-groups 8
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
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --lr 8.0e-5 
    --lr-decay-style cosine 
    --min-lr 8.0e-8
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --ckpt-format torch
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --sequence-parallel
    --pipeline-model-parallel-size ${PP}
    --decoder-first-pipeline-num-layers ${FIRST_PP_LAYERS}
    --decoder-last-pipeline-num-layers ${LAST_PP_LAYERS}
    --context-parallel-size ${CP}
    --use-distributed-optimizer 
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH} 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 5
    --log-interval 1
    --save-interval 1000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" 
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_llama3_405B_tp${TP}-pp${PP}-cp${CP}
    --use-pytorch-profiler
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
