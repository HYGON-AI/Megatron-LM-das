# These variables should not be modified.
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0

# These variables should be modified according to the environment of the machine you are using.
# Please choose one from [wz, 508-shca, 508-mlnx].

# wz
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_TOPO_FILE=${MEGATRON_PATH}/requirements/topo-input.xml
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=10737418240
export ROCSHMEM_TOPO_FILE_FORCE=${MEGATRON_PATH}/requirements/topo.config

# 508-shca
# module load app/rccl/shca_rdma_plugins/v8 
# module load app/rccl/tests 
# module load app/rccl/topos/default
# module load mpi/openmpi/5.0.3/gcc-8.5.0/shca_ucx-1.15.0
# export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
# export NCCL_PXN_DISABLE=0
# export RCCL_PXN_GPU_BALANCE=1
# export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"
# export NCCL_PLUGIN_P2P=ib
# export NCCL_SOCKET_IFNAME=ib0
# export SHCA_DEBUG_MASK=0
# export SHCA_CMR_LOG_LEVEL=1
# export SHCA_SHUT_UP_FWB=0
# export NCCL_NET_PLUGIN=shca
# export UCX_IB_NUM_PATHS=1

# 508-mlnx
# module load app/rccl/tests
# module load app/rccl/topos/default
# module load mpi/hpcx/2.18.0/gcc-8.5.0/mlnx
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
# export NCCL_PXN_DISABLE=0
# export NCCL_NCHANNELS_PER_PEER=32
# export NCCL_MIN_P2P_NCHANNELS=32
# export NCCL_MAX_P2P_NCHANNELS=32
# export RCCL_P2P_XHCL_CHANNEL_NUM=30
