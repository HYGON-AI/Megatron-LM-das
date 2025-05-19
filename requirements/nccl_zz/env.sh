# nccl env
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
export NCCL_TOPO_FILE=${MEGATRON_PATH}/requirements/nccl_zz/topo-input.xml
export NCCL_IB_PCI_RELAXED_ORDERING=0
export NCCL_PLUGIN_P2P=ucx
export NCCL_PXN_DISABLE=0
export NCCL_SOCKET_IFNAME=eno1
export LD_LIBRARY_PATH=${MEGATRON_PATH}/requirements/nccl_zz/lib-v8:$LD_LIBRARY_PATH