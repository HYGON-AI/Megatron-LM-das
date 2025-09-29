import logging

import numpy as np
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import is_main_replica


logger = logging.getLogger(__name__)


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


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            continue
        if all_slices and 'norm' in sharding.key:
            continue

        all_slices.append((sharding.flattened_range.start, sharding.flattened_range.stop))

    starts, stops = map(np.asarray, zip(*sorted(all_slices)))
    expected_size = np.product(local_shape)
    if starts[0] != 0 or stops[-1] != expected_size or not np.all(starts[1:] == stops[:-1]):
        raise CheckpointingException(
            f"Flattened ranges dont cover the whole shard {tensors_by_shard[0]} of size {expected_size}. Ranges: {(starts, stops)}"
        )
