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
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# enable BatchLinear
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export NVTE_OVERLAP_GRAD_REDUCE=1
export TRITON_HOME=/tmp

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
    --distributed-timeout-minutes 60
    --distributed-backend nccl
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 16
    --moe-layer-freq 1
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --rotary-base 10000
    --use-flash-attn
    --multi-latent-attention
    --enable-experimental
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 20
    --no-create-attention-mask-in-dataloader
    --kv-channels 128
    --make-vocab-size-divisible-by 3232
    --qk-layernorm
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope attn
    --te-rng-tracker
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

MOE_ARGS=(
    --num-experts 256
    --moe-aux-loss-coeff 1e-4
    # --moe-enable-deepep
    # --moe-deepep-num-sms 48
    --moe-token-dispatcher-type alltoall # flex
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 8
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-fusion
    --moe-router-force-load-balancing
    --moe-permute-fusion
    --moe-grouped-gemm
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 949,50,1
    --num-workers 6
    --no-mmap-bin-files
)

TRAINING_ARGS=(
    --train-iters 10
    --micro-batch-size 1
    --global-batch-size 256
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --lr-warmup-init 3.9e-7
    --lr-warmup-fraction 0.01
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --overlap-param-gather
    --overlap-grad-reduce
    # --num-layers-per-virtual-pipeline-stage 2
    # --overlap-moe-expert-parallel-comm
    # --schedule-method dualpipev
    # --delay-wgrad-compute
    # --use-quantize-comm
)

LOGGING_ARGS=(
    --log-throughput
    --log-interval 1
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --logging-level 40
    --save-interval 10000
    --eval-interval 200
    --eval-iters -1
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_deepseek671B_32nodes
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
        --wandb-project ${WANDB_PROJECT:-"DeepseekV3"}
        --wandb-exp-name ${WANDB_NAME:-"DeepseekV3_671B"}
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