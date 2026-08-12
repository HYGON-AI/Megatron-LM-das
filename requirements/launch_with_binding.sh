#!/bin/bash

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
numa_map=(0 1 2 3 4 5 6 7)

LOCAL_RANK=$1
shift

NUMA_ID=${numa_map[$LOCAL_RANK]}
numactl --cpunodebind=${NUMA_ID} --preferred=${NUMA_ID} "$@"
