#!/bin/bash

for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# Runs Mixtral 8x22B model
source /opt/dtk/env.sh

# default env
DIST_URL=${1}
DIST_PORT=25900
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10

# nccl env
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=7
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_TOPO_FILE="./topo-input.xml"

# enable BatchLinear
export GROUPED_GEMM_BatchLinear=1

# data path
CHECKPOINT_PATH="path to CKPT" 
TOKENIZER_MODEL="path to tokenizer.model"
DATA_PATH="path to my-mixtral_text_document"

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
    --num-layers 56
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
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 10
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-param-gather
    --overlap-grad-reduce
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_mixtral8x22B_8nodes_tp4-pp8-ep8-ep_tp1-cp1
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
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
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
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
case ${LOCAL_RANK} in
[0])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
[4])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=4 --membind=4 ${APP}
  ;;
[5])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=5 --membind=5 ${APP}
  ;;
[6])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=6 --membind=6 ${APP}
  ;;
[7])
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  #numactl --cpunodebind=7 --membind=7 ${APP}
  ;;
esac

