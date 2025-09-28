# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import Metadata

from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_inplace,
    extract_matching_values,
    merge,
    nested_values,
)
from megatron.core.dist_checkpointing.exchange_utils import (
    ShardDistribution,
    determine_main_replica_uniform_distribution,
    exchange_by_distribution,
    exchange_loaded_objects_gather_object,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, StateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    SaveShardedStrategy,
)
from megatron.core.dist_checkpointing.utils import (
    _sharded_object_id,
    _sharded_tensor_shard_id,
    _ShardId,
    debug_time,
)
from megatron.core.dist_checkpointing.validation import (
    determine_global_metadata,
    validate_sharding_integrity,
)
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.core.dist_checkpointing.strategies.fully_parallel import distribute_main_replicas_with_precomputed_distribution

logger = logging.getLogger(__name__)

T = TypeVar('T', ShardedObject, ShardedTensor)
class FullyParallelLoadStrategyWrapper(LoadShardedStrategy):
    def apply_loading_parallelization(
        self, sharded_state_dict: ShardedStateDict
    ) -> Optional[ShardDistribution]:
        """Distributes the load across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of loads among the ranks.
        Marks ShardedTensors to be loaded by the current rank with replica_id 0
        (and others with non 0 values).

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the loading

        Returns:
            ShardDistribution (optional): the computed loading distribution
        """
        if self.do_cache_distribution and self.cached_distribution is not None:
            logger.debug(f'Apply *cached* load parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            logger.debug(f'Apply load parallelization')
            precomputed_distribution = determine_main_replica_uniform_distribution(
                sharded_state_dict, self.parallelization_group
            )

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution

        return precomputed_distribution