# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch

from megatron.core.dist_checkpointing.mapping import is_main_replica


def _compute_shards_access(rank_sharding):
    shard_access_cnt = torch.zeros(
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device="cpu"
    )
    for rank, sharding in rank_sharding:
        if is_main_replica(sharding.replica_id):
            if 'norm' in sharding.key:
                shard_access_cnt[sharding.local_chunk_offset_in_global()] = 1
            else:
                shard_access_cnt[sharding.local_chunk_offset_in_global()] += 1
    return shard_access_cnt
