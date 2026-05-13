# These variables should not be modified.
# CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONWARNINGS=ignore
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0

export TRITON_HOME=/tmp/wangxj3
# export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# These variables should be modified according to the environment of the machine you are using.
# 811 节点配置
module purge
# module unuse /public/software/modules

# module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/shca_rdma_plugins/v8
# module load app/rccl/tests  


export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
export NCCL_PXN_DISABLE=0
export RCCL_PXN_GPU_BALANCE=1
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=eno1
export SHCA_DEBUG_MASK=0
export SHCA_CMR_LOG_LEVEL=1
export NCCL_NET_PLUGIN=shca
export UCX_IB_NUM_PATHS=1
export C10D_USE_IPV6=0
export C10D_SOCKET_IFNAME=eno1

