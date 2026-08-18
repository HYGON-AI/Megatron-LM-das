# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Utilities for exchanging data between ranks."""

import logging
from collections import defaultdict
from typing import Optional, Set

import torch

from megatron.core.utils import get_pg_size
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, is_main_replica, ReplicaId
from megatron.core.dist_checkpointing.utils import _sharded_tensor_shard_id, _ShardId
from megatron.core.dist_checkpointing.exchange_utils import ShardDistribution, _shard_size, distribute_shards_to_ranks


logger = logging.getLogger(__name__)


def is_main_replica_norm(replica_id: ReplicaId):
    if isinstance(replica_id, int):
        return replica_id == 0
    return len(replica_id) > 0 and replica_id[-1] == 0


def determine_main_replica_uniform_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    ignore_groups: bool = False,
) -> Optional[ShardDistribution]:
    """Computes the save distribution.

    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.

    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to compute the distribution of
        parallelization_group (ProcessGroup): distribution will be computed
            within this process group
        ignore_groups (bool, optional): whether the distribution defines groups.
            This option is primarily used during loading, as it ensures that all replicas,
            including non-main ones, are loaded by this parallelization group
            Defaults to False.

    Returns (ShardDistribution, optional): distribution that can be used to apply the
        parallelization. Returns None if the process_group is trivial (1 rank)

    """
    if parallelization_group is None:
        parallelization_group = torch.distributed.group.WORLD
    group_size = get_pg_size(group=parallelization_group)
    if group_size <= 1:
        return
    local_shards = list(
        sh_base
        for sh_base in nested_values(sharded_state_dict)
        if isinstance(sh_base, ShardedTensor)
    )
    local_shards_no_data = [ten.without_data() for ten in local_shards]

    all_shards = [None] * get_pg_size(group=parallelization_group)
    torch.distributed.all_gather_object(
        all_shards, local_shards_no_data, group=parallelization_group
    )

    shard_to_ranks = defaultdict(list)
    shard_to_size = {}
    shard_to_metadata = {}
    group_has_main_replica: Set[_ShardId] = set()
    group_has_non_main_replica: Set[_ShardId] = set()

    for rank, rank_shards in enumerate(all_shards):
        for sh_ten in rank_shards:
            shard_id = _sharded_tensor_shard_id(sh_ten)
            shard_to_ranks[shard_id].append(rank)
            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _shard_size(sh_ten)
                shard_to_metadata[shard_id] = sh_ten
            if 'norm' in shard_id[0]:
                if is_main_replica_norm(sh_ten.replica_id):
                    group_has_main_replica.add(shard_id)
                else:
                    group_has_non_main_replica.add(shard_id)
            else:
                if is_main_replica(sh_ten.replica_id):
                    group_has_main_replica.add(shard_id)
                else:
                    group_has_non_main_replica.add(shard_id)

    # we always include all main replicas, and non-main only if `ignore_groups`
    shards_in_this_group: Set[_ShardId] = group_has_main_replica
    if ignore_groups:
        shards_in_this_group = shards_in_this_group | group_has_non_main_replica
    # cross-parallel-group references are empty if `not ignore_groups`,
    # otherwise it's `group_has_non_main_replica - group_has_main_replica`
    cross_parallelization_group_loads = shards_in_this_group - group_has_main_replica

    # Filter out shards that don't belong to this group
    shard_to_ranks = {k: v for k, v in shard_to_ranks.items() if k in shards_in_this_group}

    shard_to_saving_rank = distribute_shards_to_ranks(
        shard_to_ranks, shard_to_size, len(all_shards), cross_parallelization_group_loads
    )

    return ShardDistribution(
        shard_to_saving_rank, shards_in_this_group, shard_to_metadata, shard_to_ranks
    )
