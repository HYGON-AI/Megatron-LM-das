#!/bin/bash

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
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
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# enable BatchLinear
export GROUPED_GEMM_BatchLinear=1

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 65536
    --num-layers 4
    --hidden-size 6144
    --ffn-hidden-size 16384
    --num-attention-heads 48
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --ckpt-format torch
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-expert-capacity-factor 0.5
    --moe-pad-expert-input-to-capacity
    #--moe-grouped-gemm
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 10
    --lr-decay-iters 10000
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --bf16
    --overlap-param-gather
    --overlap-grad-reduce
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters -1 \
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_mixtral8x22B_1nodes_tp2-pp1-ep8-etp1-cp1
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x22B"}
    )
fi

APP="python3 -u ${MEGATRON_PATH}/pretrain_gpt.py \
    ${DISTRIBUTED_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    "

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

#for hygon cpu
${MEGATRON_PATH}/requirements/launch_with_binding.sh ${LOCAL_RANK} ${APP}