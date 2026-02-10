#!/bin/bash

# wz
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
numa_map=(0 1 2 3 4 5 6 7)

# 508 天龙
# export HIP_VISIBLE_DEVICES=0,1,5,4,2,3,7,6
# numa_map=(0 1 5 4 2 3 7 6)

# 508 mlnx
# export HIP_VISIBLE_DEVICES=0,1,2,3,5,4,7,6
# corenum=`cat /proc/cpuinfo| grep "cpu cores" | uniq | awk '{print $4}'`
# if [ "${corenum}" -eq "64" ]
# then
#         numa_map=(0 3 2 1 7 4 5 6)
# else
#         numa_map=(0 2 3 5 8 6 11 9)
# fi

LOCAL_RANK=$1
shift

NUMA_ID=${numa_map[$LOCAL_RANK]}
numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} "$@"