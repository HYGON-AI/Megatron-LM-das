# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import logging
from typing import Optional

from megatron.core.dist_checkpointing.exchange_utils import (
    ShardDistribution,
    determine_main_replica_uniform_distribution,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.fully_parallel import distribute_main_replicas_with_precomputed_distribution

logger = logging.getLogger(__name__)


class FullyParallelLoadStrategyWrapper():
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
