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

# default env
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH
export TRITON_HOME=/tmp

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

GPT_MODEL_ARGS=(
    --seq-length 8192
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144 
    --moe-ffn-hidden-size 768
    --num-attention-heads 32
    --max-position-embeddings 262144
    --num-query-groups 4
    --group-query-attention
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --kv-channels 128
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models 
    --micro-batch-size 1
    --global-batch-size 128
    --train-iters 100000
    --weight-decay 0.01 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008 
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --qk-layernorm
    --rotary-base 1000000
    --lr 1.0e-5
    --lr-decay-style cosine 
    --min-lr 1.0e-6
    --lr-warmup-iters 100
    --lr-decay-iters 99900
    --ckpt-format torch_dist
    --ddp-average-in-collective
    --overlap-grad-reduce
    --overlap-param-gather
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --main-params-dtype fp16
    --enable-cuda-graph
    --te-rng-tracker
    --disable-msc
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-expert-capacity-factor 1
    --moe-pad-expert-input-to-capacity
    --moe-permute-fusion
    --moe-grouped-gemm
    --moe-layer-freq '([1]*48)'
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer 
    --sequence-parallel
)

DATA_ARGS=(
    --tokenizer-type SFTTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 949,50,1
    --sft
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 10
    --log-interval 1
    --save-interval 1000000 
    --eval-interval 10000
    --save Qwen3_30B_A3B_sft
    --load $CHECKPOINT_PATH
    --ckpt-fully-parallel-load
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --tensorboard-dir ./tensorboard
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 5
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_qwen3_30B_A3B_tp4-pp2-ep8-etp1-cp1_sft
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 5
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
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
