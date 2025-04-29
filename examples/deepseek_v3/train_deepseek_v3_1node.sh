#!/bin/bash
for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
        # export GPU_FLUSH_ON_EXECUTION=1
        # export HIP_DIRECT_DISPATCH=0
    fi
done
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $CURRENT_DIR
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10

export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=7
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_TOPO_FILE="/public/home/yuguo/check/rccl-tests-0204/topo-input.xml" #"your topo file"
export GLOG_minloglevel=3
export GROUPED_GEMM_BatchLinear=1
export LD_LIBRARY_PATH=/public/home/yuguo/data/rocblas-install-0224/lib:$LD_LIBRARY_PATH

LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

### BASE CONFIG ###
MODEL_SIZE=A37B
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=4096
PR=bf16
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=1
PP=2
CP=1
EP=4
SP=true
DO=true
FL=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=none
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=500
DATASET_PATH=${MEGATRON_PATH}/deepseekv3_dataset/mmap_deepseekv3_datasets_text_document #"your data path"
VALID_DATASET_PATH=${MEGATRON_PATH}/deepseekv3_dataset/mmap_deepseekv3_datasets_text_document #"your data path"
PRETRAIN_CHECKPOINT_PATH=${MEGATRON_PATH}/deepseekv3_dataset #"your model path"

# the following two values will not be used when SFT is true
TRAIN_TOKENS=100000000
WARMUP_TOKENS=10000
###############################

OUTPUT_BASEPATH=./output
### OTHERS ###

if [ $FL = true ]; then
    :
    #exit -1
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=" \
        --attention-backend fused
    "
fi

if [ $MODEL_SIZE = A37B ]; then
    TRAIN_ITERS=2
    HIDDEN_SIZE=7168
    NUM_ATTENTION_HEADS=128
    NUM_LAYERS=2
    INTERMEDIATE_SIZE=18432
    MOE_INTERMEDIATE_SIZE=2048
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
    EXTRA_VOCAB_SIZE=467
    Q_LORA_RANK=1536
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=8 #256
    ROUTER_TOPK=8
    NUM_SHARED_EXPERTS=1
    RMS_NORM_EPS=1e-6

    moe_options=" \
        --moe-grouped-gemm \
        --moe-expert-capacity-factor 1 \
        --moe-pad-expert-input-to-capacity \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-model-parallel-size ${EP} \
        --expert-tensor-parallel-size 1 \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq ([0]*0+[1]*2) \
        --q-lora-rank ${Q_LORA_RANK} \
        --kv-lora-rank ${KV_LORA_RANK} \
        --qk-head-dim ${QK_NOPE_HEAD_DIM} \
        --qk-pos-emb-head-dim  ${QK_ROPE_HEAD_DIM} \
        --v-head-dim ${V_HEAD_DIM} \
        --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
        "

fi

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ -z ${MP_VP} ]; then
    vp_option=""
else
    vp_option=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 
if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_option=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_option=" \
                    "
fi


if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
                    "
fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_option=" \
            --tokenizer-model $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=${24}
    LR_WARMUP_ITERS=${25}
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-deepseek-v3"
else
    # TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    PREFIX="pretrain-mcore-deepseek-v3"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
else 
    dataset_options=" \
        --data-path ${DATASET_PATH} \
        --split 99,1,0"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    echo "Currently MLA do not support THD format attention, thus sequence packing can not be used..."
    packing_options=""
else
    packing_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

megatron_options="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 5 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --tokenizer-type DeepSeekV2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --no-save-optim \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --ckpt-format torch \
        --transformer-impl transformer_engine \
        --use-rope-scaling \
        --multi-latent-attention \
        --mtp-num-layers 1 \
        --use-mcore-models \
        "

TORCH_PROFIE_ARGS="  \
    --profile \
    --profile-ranks 0 1 2 3 4 5 6 7 \
    --profile-step-start 3 \
    --profile-step-end 4 \
    --profile-dir torch_prof_data_16nodes_dcu \
    --use-pytorch-profiler \
"

HIP_PROFIE_ARGS="  \
    --profile \
    --profile-ranks 0 1 2 3 4 5 6 7 \
    --profile-step-start 4 \
    --profile-step-end 5 \
    --use-hip-profiler \
"

APP="python3 -u ${MEGATRON_PATH}/pretrain_gpt.py
     ${megatron_options} \
     ${dataset_options} \
     ${pr_options} \
     ${load_option} \
     ${activation_checkpoint_options} \
     ${do_option} \
     ${sp_option} \
     ${moe_options} \
     ${offload_option} \
     ${sft_options} \
     ${vp_option} \
     ${packing_options} \
     ${uneven_split_option} \
     ${attn_backend_option} \
     ${comm_overlap_option} \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --local-rank ${LOCAL_RANK} \
    --dist-url tcp://${1}:25900 \
    "

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

case ${LOCAL_RANK} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[4])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[5])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[6])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
[7])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ${APP}
  ;;
esac
