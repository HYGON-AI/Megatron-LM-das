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
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10
export NVTE_OVERLAP_GRAD_REDUCE=1
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# int8_simulation_fp8
export NVTE_INT8_SIM_FP8_TENSORWISE=1
export NVTE_DISABLE_NVRTC=1
export NVTE_INT8_SIM_FP8=1

# triton cache dir
export TRITON_HOME=/tmp

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 32768
    --num-layers 8
    --hidden-size 8192
    --ffn-hidden-size 32768
    --num-attention-heads 64
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --ckpt-format torch
    --use-quantize-comm
    --schedule-method interleaved_1f1b
    --overlap-moe-expert-parallel-comm
    --delay-wgrad-compute
    --group-query-attention
    --num-query-groups 8
    --fp8-format hybrid
    --fp8-recipe tensorwise 
    --fp8-param-gather
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-router-group-topk 2
    --moe-router-num-groups 8
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-grouped-gemm
    --moe-expert-capacity-factor 1
    --moe-pad-expert-input-to-capacity
    --moe-router-dtype fp32
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 98,2,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4096
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
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 2
    --context-parallel-size 1
    --num-layers-per-virtual-pipeline-stage 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-throughput \
    --log-interval 1 \
    --save-interval 100000 \
    --eval-interval 10000 \
    --eval-iters 5 \
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 6 8 32
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_aibenchmark_128nodes_tp4-pp4-ep32-etp2-cp1-vp2
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 6 8 32
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