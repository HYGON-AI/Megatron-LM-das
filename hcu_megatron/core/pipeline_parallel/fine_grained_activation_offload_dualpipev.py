# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    debug_rank,
    GPUTensorPool,
    ChunkOffloadHandler,
)


def get_backward_chunk_order(num_microbatches):
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    order = list() # 0: first chunk; 1: second chunk
    for _ in range(pp_size - pp_rank - 1):   # 1b1w1f
        order.append(1)
    num_second_chunks = pp_size - pp_rank - 1

    num_remain_backward = 2 * num_microbatches - (pp_size - pp_rank - 1)
    next_backward_chunk = 1
    for _ in range(num_remain_backward):
        order.append(next_backward_chunk)
        num_second_chunks += next_backward_chunk

        next_backward_chunk = 1 - next_backward_chunk
        if (
            next_backward_chunk == 1
            and num_second_chunks == num_microbatches
        ):
            next_backward_chunk = 0

    return order


class PipelineOffloadManagerDualpipeV():
    """
    Singleton manager for coordinating activation offloading across pipeline stages.
    Manages chunk handlers, synchronizes GPU-CPU transfers,
    and handles virtual pipeline parallelism.
    """

    def __init__(self):
        """Initialize the manager with queues and dedicated CUDA streams."""
        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        # Shared CPU tensor pool for all chunks to improve reuse efficiency
        self._cpu_tensor_pool = GPUTensorPool(device="cpu", pin_memory=True)

        # Whether the manager is in warmup phase.
        self._is_warmup = True
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each forward pass.
        self._cached_chunks_forward = []
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each backward pass.
        self._cached_chunks_backward = [None] * 2 * get_num_microbatches()
        # Index of the current backward chunk in the cached chunks backward.
        self._cached_chunks_index_backward = 0
        # Index of the current forward chunk in the cached chunks forward.
        self._cached_chunks_index_forward = 0

        self.do_offload = True

        # Do not offload the last X groups so that the reloading won't block the computing stream.
        self._offload_margin = 0
        # Sometimes we need to delay the offloading and launch it later.
        # The delayed offload groups are stored in a queue.
        self._delayed_offload_groups = []
        # chunk order for backward pass
        self._backward_chunk_order = get_backward_chunk_order(get_num_microbatches())
        self.reset()

    def push(self, handler, is_first_chunk=True):
        """Add a chunk handler to the backward queue."""
        debug_rank(f"pushing handler {handler}")
        target_chunk_idx = 0 if is_first_chunk else 1
        if self._is_warmup:
            for idx, chunk_idx in enumerate(self._backward_chunk_order):
                if chunk_idx == target_chunk_idx and self._cached_chunks_backward[idx] is None:
                    self._cached_chunks_backward[idx] = handler
                    break

    def init_model_chunk_offload_handler(
        self,
        is_first_chunk,
        min_offloaded_tensor_size=1024 * 1024,
        max_inflight_offloads=None,
    ):
        """
        Initialize a chunk offload handler for a model chunk (microbatch).

        Args:
            is_first_chunk: self-explained
            min_offloaded_tensor_size: Minimum tensor size (in elements) to offload
            max_inflight_offloads: If set, cap pending offloads per group name before main
                wait_event; see ``fine_grained_offloading_max_inflight_offloads`` on
                ``TransformerConfig``.
        """
        if not self._is_warmup:
            return

        # Use shared CPU tensor pool for better reuse across chunks
        cur_chunk = ChunkOffloadHandler(
            min_offloaded_tensor_size,
            self._cpu_tensor_pool,
            max_inflight_offloads=max_inflight_offloads,
        )
        debug_rank(f"init_model_chunk_offload_handler {cur_chunk}")
        self.push(cur_chunk, is_first_chunk=is_first_chunk)
        self._cur_forward_chunk = cur_chunk
        self._cached_chunks_forward.append(cur_chunk)


class FineGrainedActivationOffloadingInterface:
    """Interface for fine-grained activation offloading."""

    @staticmethod
    def init_chunk_handler(
        is_first_chunk, min_offloaded_tensor_size, max_inflight_offloads=None
    ):
        """Initialize the chunk handler, called at the start of a microbatch forward pass."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
        PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
            is_first_chunk,
            min_offloaded_tensor_size,
            max_inflight_offloads=max_inflight_offloads,
        )
