#!/bin/bash

LOCAL_RANK=$1
shift

gpu_map=(0 1 2 3 4 5 6 7)
numa_map=(0 1 2 3 4 5 6 7)

GPU_ID=${gpu_map[$LOCAL_RANK]}
NUMA_ID=${numa_map[$LOCAL_RANK]}

export HIP_VISIBLE_DEVICES=${GPU_ID}
numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} "$@"