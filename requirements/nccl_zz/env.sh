# nccl env
module load compiler/dtk/25.04.1
module load app/rccl/shca_rdma_plugins/v8 
module load app/rccl/tests 
module load app/rccl/topos/shca 
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca_ucx-1.15.0

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NCHANNELS_PER_PEER=16
export NCCL_MIN_P2P_NCHANNELS=16
export NCCL_MAX_P2P_NCHANNELS=16
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
export NCCL_IB_PCI_RELAXED_ORDERING=0
export NCCL_PLUGIN_P2P=ucx
export NCCL_SOCKET_IFNAME=eno1
export SHCA_DEBUG_MASK=0
export SHCA_CMR_LOG_LEVEL=1
export SHCA_SHUT_UP_FWB=1
export SHCA_UCT_CQ_SIZE_INC=5
export UCX_RNDV_PUT_FORCE_FLUSH=y
export NCCL_PXN_DISABLE=1
export NCCL_NET_PLUGIN=shca