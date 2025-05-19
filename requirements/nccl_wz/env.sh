# nccl env
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=7
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_TOPO_FILE=${MEGATRON_PATH}/requirements/nccl_wz/topo-input.xml