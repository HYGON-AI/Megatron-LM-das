#!/bin/bash
# set -eux

for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))


#default env
#export FLASH_ATTENTION_PRINT_PARAM=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_P2P_LEVEL=PXB # SYS
export GPU_MAX_HW_QUEUES=10
#export HIP_ALLOC_INITIALIZE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# nccl env
export NCCL_ALGO=Ring
export NCCL_NCHANNELS_PER_PEER=16
export NCCL_MIN_NCHANNELS=32 # 20
export NCCL_MAX_NCHANNELS=32 # 20
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=7
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_TOPO_FILE="/public/home/wangxj/Projects/rccl-test/rccl-tests-0204/topo-input.xml"
export GLOG_minloglevel=3 # 打印error级别的nccl日志
source /opt/dtk/env.sh

# hipblaslt库
export LD_LIBRARY_PATH=/public/home/wangxj/Downloads/blas/hipblaslt-install-dtk-25.04-0212/lib:$LD_LIBRARY_PATH 

# rocblas
export LD_LIBRARY_PATH=/public/home/wangxj/Downloads/blas/rocblas-install-0227/lib:$LD_LIBRARY_PATH

# torch控制多流转单流
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1 

#增加编译缓存
export cache_size_limit=64

# CHECKPOINT_PATH=./Llama-2-7b-hf-to-meg-tp1-pp2 #CHECKPOINT_PATH=./tmp_7b # 
SAVE_PATH=./tmp_7b
TENSORBOARD_LOGS_PATH=./tmp_7b  #$2 #<Specify path>
DATA_PATH="/public/home/wangxj/Downloads/datasets/oscar-1GB/oscar-1GB-llama2_text_document" #<Specify path and file prefix>_text_document

GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008 
    --num-attention-heads 32
    --max-position-embeddings 4096

    --normalization RMSNorm 
    --position-embedding-type rope # none # 
    --untie-embeddings-and-output-weights # 分开处理embed和输出权重, 增加灵活性
)

export NVTE_FLASH_ATTN=1 # 走cutlass
# export NVTE_FLASH_ATTN_TRITON=1 # 走triton_fa
# --transformer-impl transformer_engine # 走core用这两组参数
    # --use-mcore-models
    # --transformer-impl local # 走legacy用这两组参数
    # --use-legacy-models 
TRAINING_ARGS=(
    --transformer-impl local # 走legacy用这两组参数
    --use-legacy-models 
    --micro-batch-size 1
    --global-batch-size 256 #256 #240 #60 #512 #64
    --train-iters 50
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    # --fp16 # 开启fp16需要指定loss-scale
    # --loss-scale 1024
    --use-distributed-optimizer 
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    # --no-gradient-accumulation-fusion
    --swiglu
    --lr 3.0e-5 
    --lr-decay-style cosine 
    --min-lr 3.0e-6
    --lr-warmup-iters 1
    --ckpt-format torch
    --ddp-average-in-collective # 在dp阶段通信中, 梯度或参数将被直接平均, 而不是先求和(到一个设备)再平均
    # --recompute-granularity full # 开启重计算降低显存增加耗时
    # --recompute-num-layers 5 #0 #
    # --recompute-method block
    --overlap-grad-reduce # 重叠ddp grad reduce
    # --tp-comm-overlap # tensor parallel comm和gemm重叠, 优化项未适配
    # --tp-comm-overlap-rs-dgrad # reduce-scatter和dgrad gemm重叠
    --use-flash-attn
)
# 使用torch fa的环境变量
# export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
# export TORCHINDUCTOR_BENCHMARK_FUSION=1
# export TORCHINDUCTOR_BENCHMARK_MULTI_TEMPLATES=1
# export TORCHINDUCTOR_MAX_AUTOTUNE=1
# export TORCHINDUCTOR_CACHE_DIR=./cache
# --use-flash-attn-cutlass # cutlass fa
# --use-flash-attn-triton # triton fa
# --use-flash-attn-torch # torch fa

MODEL_PARALLEL_ARGS=(
    --sequence-parallel
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 2
  # --context-parallel-size 2
  # --num-layers-per-virtual-pipeline-stage 4
  # --microbatch-group-size-per-virtual-pipeline-stage 1
  # --no-overlap-p2p-communication # 开启后
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --seq-length 4096 #4096
    --split 949,50,1
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model /public/home/wangxj/Downloads/model_weights/llama2_7b_hf/tokenizer.model
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 50
    --log-interval 1
    --save-interval 1000 
    --eval-interval 1000 
    --save $SAVE_PATH 
    --load $SAVE_PATH 
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# FINETUNE_ARGS=(
#     # --finetune
#     # --pretrained-checkpoint $CHECKPOINT_PATH
#     --load $CHECKPOINT_PATH
#     --no-load-optim
#     --no-load-rng
# )

PROFILE_ARGS=(
    --profile
    --profile-step-start 4
    --profile-step-end 5
    --use-pytorch-profiler
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-dir prof_data
)

RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
DIST_URL=${1}
DIST_PORT=34577

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${DISTRIBUTED_ARGS[@]} \
        
"
# 开启profile
# ${PROFILE_ARGS[@]} \

# export HIP_VISIBLE_DEVICES=0,7 #  # 4,5,6,7 #,
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #  # 4,5,6,7 #,
# export CUDA_VISIBLE_DEVICES=4,5,6,7 # 0,1,2,3,
# ${APP}
case ${LOCAL_RANK} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  numactl --cpunodebind=3 --membind=3 ${APP}
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[4])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  numactl --cpunodebind=4 --membind=4 ${APP}
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[5])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  numactl --cpunodebind=5 --membind=5 ${APP}
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[6])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  numactl --cpunodebind=6 --membind=6 ${APP}
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[7])
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  numactl --cpunodebind=7 --membind=7 ${APP}
  # hipprof --hip-trace --trace-off numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
esac