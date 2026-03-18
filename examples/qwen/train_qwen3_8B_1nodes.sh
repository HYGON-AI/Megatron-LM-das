#!/bin/bash

INITIALIZATION_ARGS=( --num-workers 2)

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
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
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# default env
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10 #10 # 4 # 20



DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

GPT_MODEL_ARGS=(
    --seq-length 32768 
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 12288 
    --num-attention-heads 32
    --max-position-embeddings 40960
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
    --micro-batch-size 1
    --global-batch-size 32
    --train-iters 50
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --rotary-base 1000000
    --lr 3.0e-5 
    --lr-decay-style cosine 
    --min-lr 3.0e-6
    --lr-warmup-iters 1
    --ckpt-format torch
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn

)


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --context-parallel-size 2
    --use-distributed-optimizer 
    --sequence-parallel
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
    --profile-ranks 0 4
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_qwen_cp2_qknorm
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${FP8_PARALLEL_ARGS[@]} \
    "

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

#for hygon cpu
${launch_with_binding} ${LOCAL_RANK} ${APP}