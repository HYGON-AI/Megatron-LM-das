#!/bin/bash

# wz
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
numa_map=(0 1 2 3 4 5 6 7)

# 508
# export HIP_VISIBLE_DEVICES=0,1,2,3,5,4,7,6
# numa_map=(0 3 2 1 7 4 5 6)

LOCAL_RANK=$1
shift

NUMA_ID=${numa_map[$LOCAL_RANK]}
numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} "$@"
