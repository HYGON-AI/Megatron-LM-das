#!/bin/bash

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
export GPU_MAX_HW_QUEUES=6

# int8_simulation_fp8
export NVTE_INT8_SIM_FP8_TENSORWISE=1
export NVTE_DISABLE_NVRTC=1
export NVTE_INT8_SIM_FP8=1

num_layers=12
num_expert=1024
TP=2
PP=4
EP=512
ETP=4
CP=1
DP=$((${WORLD_SIZE} / ${TP} / ${PP} / ${CP}))
EDP=$((${WORLD_SIZE} / ${PP} / ${EP} / ${ETP}))
GBS=$((64 * ${DP}))
LR=9.36e-05
MIN_LR=9.36e-06
TRAIN_ITERS=10

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
    --disable-gloo-process-groups
    --distributed-timeout-minutes 30
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 32768
    --num-layers ${num_layers}
    --hidden-size 8192
    --ffn-hidden-size 33144
    --num-attention-heads 64
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --no-check-for-nan-in-loss-and-grad
    --fp8-format hybrid
    --fp8-recipe tensorwise 
    --fp8-param-gather
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --group-query-attention
    --num-query-groups 64
    --manual-gc
    --manual-gc-interval 5
    --use-quantize-comm
    --use-intra-ep
    --overlap-param-gather
    --overlap-grad-reduce
    --swiglu
)

MOE_ARGS=(
    --num-experts ${num_expert}
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-expert-capacity-factor 1
    --moe-pad-expert-input-to-capacity
    --moe-permute-fusion
    --moe-grouped-gemm
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 98,2,0
)

TRAINING_ARGS=(
    --train-iters ${TRAIN_ITERS}
    --micro-batch-size 1
    --global-batch-size ${GBS}
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-warmup-init ${MIN_LR}
    --lr-warmup-fraction 0.01
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size ${CP}
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-throughput
    --log-interval 1
    --save-interval 100000
    --eval-interval 10000
    --eval-iters -1
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
    --no-save-optim
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_aibenchmark_3072nodes_tp${TP}-pp${PP}-ep${EP}-etp${ETP}-cp${CP}
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
        --wandb-project ${WANDB_PROJECT:-"GPT"}
        --wandb-exp-name ${WANDB_NAME:-"GPT_567B"}
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
${launch_with_binding} ${LOCAL_RANK} ${APP}