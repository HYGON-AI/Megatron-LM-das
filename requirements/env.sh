# These variables should not be modified.
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TORCH_CPP_LOG_LEVEL=fatal
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=Ring
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0

export PYTHONPATH=${MEGATRON_PATH}/3rdparty/Megatron-LM:${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src:$PYTHONPATH
export TRITON_HOME=/tmp

# These variables should be modified according to the environment of the machine you are using.
# Auto-detect the IB HCA family and pick the matching profile. Force a specific
# profile by exporting IB_TYPE=mlnx or IB_TYPE=shca before sourcing this file.

# Auto-detect the host's default-route NIC and use it for NCCL/Gloo's socket
# (non-IB) transport; different nodes can have different interface names.
# Force a specific interface by exporting NCCL_SOCKET_IFNAME and/or
# GLOO_SOCKET_IFNAME before sourcing this file.
if [ -z "${NCCL_SOCKET_IFNAME:-}" ] || [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
  # Prefer /proc/net/route: no extra tool dependency, and the bench container
  # shares the host netns (--network=host) so it reflects the real route.
  _default_ifname=$(awk '$2=="00000000" && $8=="00000000" {print $1; exit}' /proc/net/route 2>/dev/null)
  # Fall back to net-tools' route (iproute2's `ip` isn't installed in this image).
  [ -n "${_default_ifname}" ] || _default_ifname=$(route -n 2>/dev/null | awk '$1=="0.0.0.0" && $3=="0.0.0.0" {print $8; exit}')
  if [ -n "${_default_ifname}" ]; then
    : "${NCCL_SOCKET_IFNAME:=${_default_ifname}}"
    : "${GLOO_SOCKET_IFNAME:=${_default_ifname}}"
    export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
  else
    echo "[env.sh] warning: could not detect default-route interface; NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME left unset" >&2
  fi
  unset _default_ifname
else
  export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
fi

# export GLOO_SOCKET_IFNAME=ens69f0
# export NCCL_SOCKET_IFNAME=ens69f0

export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_HEAP_SIZE=10737418240

if [ -z "${IB_TYPE:-}" ]; then
  if [ -d /sys/class/infiniband ]; then
    _ib_devices=$(ls /sys/class/infiniband 2>/dev/null)
    case "$_ib_devices" in
      *mlx*) IB_TYPE=mlnx ;;
      *shca*) IB_TYPE=shca ;;
      *) IB_TYPE=mlnx
         echo "[env.sh] warning: no mlx*/shca* device under /sys/class/infiniband; defaulting IB_TYPE=mlnx" >&2 ;;
    esac
    unset _ib_devices
  else
    IB_TYPE=mlnx
    echo "[env.sh] warning: /sys/class/infiniband missing; defaulting IB_TYPE=mlnx" >&2
  fi
fi
export IB_TYPE

case "$IB_TYPE" in
  mlnx)
    # Auto-detect which mlx5 ports are actually up via `ibstatus` and build
    # NCCL_IB_HCA from only the ACTIVE + LinkUp ones (down/disabled ports
    # differ per node and per cluster, so hardcoding one list doesn't travel;
    # see the ROCE_LINK/ROCE_LEAVES DOWN ports seen in platform/resource
    # checks). Force a specific value by exporting NCCL_IB_HCA before
    # sourcing this file.
    if [ -z "${NCCL_IB_HCA:-}" ]; then
      _detected_hca=""
      if command -v ibstatus >/dev/null 2>&1; then
        _detected_hca=$(ibstatus 2>/dev/null | tr -d "'" | awk '
          /^Infiniband device/ { dev=$3; port=$5 }
          /^[ \t]*state:/      { active = ($0 ~ /ACTIVE/) }
          /^[ \t]*phys state:/ { if (active && $0 ~ /LinkUp/) { if (out != "") out = out ","; out = out dev ":" port } }
          END { print out }
        ')
      fi
      if [ -n "${_detected_hca}" ]; then
        NCCL_IB_HCA=${_detected_hca}
      else
        NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1
        echo "[env.sh] warning: ibstatus unavailable or found no ACTIVE+LinkUp mlx5 port; falling back to hardcoded NCCL_IB_HCA=${NCCL_IB_HCA}" >&2
      fi
      unset _detected_hca
    fi
    export NCCL_IB_HCA
    # export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
    ;;
  shca)
    export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
    export NCCL_PLUGIN_P2P=ib
    export NCCL_NET_PLUGIN=/opt/rccl-rdma-sharp-plugins/lib/librccl-net-shca.so
    export NCCL_TOPO_FILE=/usr/local/built-in-508-topo-input-tj-default.xml
    export RCCL_PXN_GPU_BALANCE=1
    # Per-machine tuning (e.g. 508-shca); enable as needed:
    export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"
    export SHCA_DEBUG_MASK=0
    export SHCA_CMR_LOG_LEVEL=1
    export SHCA_SHUT_UP_FWB=0
    export UCX_IB_NUM_PATHS=1
    export RCCL_P2P_XHCL_CHANNEL_NUM=30
    ;;
  *)
    echo "[env.sh] error: unknown IB_TYPE='$IB_TYPE' (expected mlnx or shca)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac