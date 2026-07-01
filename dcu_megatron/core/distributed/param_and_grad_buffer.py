# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
from torch.distributed import _coalescing_manager

from megatron.core.utils import log_single_rank
from megatron.core.distributed.param_and_grad_buffer import BufferType, shard_buffer
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import reduce_scatter_with_fp32_accumulation
from megatron.core.distributed.param_and_grad_buffer import dist_reduce_scatter_func
from megatron.training.global_vars import get_args
from megatron.training import get_timers

logger = logging.getLogger(__name__)


class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.

    Args:
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in _ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in _ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger _ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
        bucket_id: Index of bucket in buffer.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
        components: Optional[List[Tuple[torch.nn.Parameter, int, torch.Size]]] = None,
    ):
        self.params_list = params
        self.params = set(params)
        # Make sure there are no duplicate params.
        assert len(self.params_list) == len(self.params)
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        if components is not None:
            self.components = components
        else:
            self.components = []


class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    Provides functionality to register when params in the bucket group have grads ready to be
    synced; an asynchronous communication call is automatically launched when _all_ params in
    the bucket group have grads ready.

    Args:
        buckets: A list of buckets.
        ddp_config: DistributedDataParallel config object.
        collective_group: intra_distributed_optimizer_instance_group if using distributed
            optimizer, data_parallel_group if not.
        collective_group_size: World size using the intra data-parallel group.
    """

    def __init__(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config
        self.timers = get_timers()

        if self.ddp_config.use_distributed_optimizer:
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        else:
            self.data_parallel_group = collective_group

        # State for bookkeeping: params is the set of parameters this bucket group is
        # responsible for, param_to_bucket maps params to the corresponding bucket.
        self.param_to_bucket = {}
        self.params = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group = None

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_distributed_optimizer_instance_group = None
            self.communication_stream = None
            assert (
                not self.ddp_config.reduce_scatter_with_fp32_accumulation
            ), "RS w/ FP32 accumulation not supported with num_distributed_optimizer_instances > 1"

        global dist_reduce_scatter_func
        if self.ddp_config.reduce_scatter_with_fp32_accumulation:
            dist_reduce_scatter_func = reduce_scatter_with_fp32_accumulation
            log_single_rank(
                logger,
                logging.INFO,
                "Using reduce_scatter_with_fp32_accumulation as reduce-scatter implementation",
            )

        # per_param_grad_ready_counts is a dict mapping parameters to number of times
        # `register_grad_ready` is called for that parameter *when
        # self.is_last_microbatch is True*. Should be 1 for most params but could be greater
        # than 1 if control flow passes through the same parameter multiple times. We lazily
        # populate this in the first batch, hence the .is_first_batch attribute.
        # When overlap_grad_reduce is True, communication (all-reduce or reduce-scatter)
        # is issued when per_param_grad_ready_counts equals golden_per_param_grad_ready_counts.
        # In other words, communication is dispatched as soon as all gradients in this bucket
        # are *ready*, as marked by the backward hook.
        # The set of keys in per_param_grad_ready_counts should be equal to `params`.
        self.golden_per_param_grad_ready_counts = {}
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
        self.is_first_batch = True

        # Other metadata to keep track of collectives.
        self.param_gather_handle = None
        self.param_gather_dispatched = False
        self.grad_reduce_handle = None

        # Each time a local shard is created from bucket.param_data or bucket.grad_data, it
        # introduces some CPU overheads. We use these two lists to cache the created local
        # shards to avoid unnecessary CPU operations. This does not increase GPU memory usage
        # because it only saves a slice view, which shares the same memory with bucket.param_data
        # or bucket.grad_data.
        self.cached_param_buffer_shard_list = [None] * len(self.buckets)
        self.cached_grad_buffer_shard_list = [None] * len(self.buckets)

        self.initial_error = False
        self.max_rank_error = 0.0
        self.min_rank_error = 0.0

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        args = get_args()
        if self.is_first_batch and self.grad_reduce_handle is not None:
            # Make this start_grad_sync call a no-op if in first batch and collective has
            # already been dispatched.
            return

        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # We use the following stream synchronization for the gradient reduction
        # within and across DistOpt instances.

        # Compute Stream: -------------Gradient compute-------------------
        # Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
        # NCCL Stream:          -------RS------     -------AR------

        # Use async communications only when overlap_grad_reduce is True.
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )
        if (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        ):
            # Assign a communication stream if we have multiple DistOpt instances and we
            # need to overlap communication.
            stream_context = torch.cuda.stream(self.communication_stream)

            # The RS/AR communication stream needs to wait for the default stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.default_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        # Coalesce communication kernels across buckets in the bucket group.
        if args.enable_dynamic_grad_comp and args.compressor is not None:
            # Coalesce communication kernels across buckets in the bucket group.
            compressed_data_list = []
            if args.overlap_grad_reduce and args.all_reduce_time:
                self.timers('DP_time', log_level=0).start()
            with stream_context:
                for bucket in self.buckets:
                    for_P, for_Q, metadata = args.compressor.compress_bucket(bucket)
                    compressed_data_list.append((bucket, for_P, for_Q, metadata))

            with _coalescing_manager(communication_group, async_ops=async_op) as cm:
                for _, for_P, _, _ in compressed_data_list:
                    torch.distributed.all_reduce(for_P, op=reduce_op, group=communication_group, async_op=async_op)
                for _, _, for_Q, _ in compressed_data_list:
                    torch.distributed.all_reduce(for_Q, op=reduce_op, group=communication_group, async_op=async_op)

            if not async_op:
                for bucket, for_P, for_Q, metadata in compressed_data_list:
                    args.compressor.decompress_bucket(bucket, for_P, for_Q, metadata)
            else:
                self._pending_compressed_data = compressed_data_list

        else:
            if args.enable_dynamic_grad_comp:
                if args.overlap_grad_reduce and args.all_reduce_time:
                    self.timers('DP_time', log_level=0).start()

            grad_reduce_handle = None
            with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
                for idx, bucket in enumerate(self.buckets):
                    if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
                        if self.cached_grad_buffer_shard_list[idx] is None:
                            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                                bucket.grad_data, self.intra_distributed_optimizer_instance_size
                            )
                        local_data_view = self.cached_grad_buffer_shard_list[idx][
                            self.intra_distributed_optimizer_instance_rank
                        ]
                        group_size = torch.distributed.get_world_size(group=communication_group)
                        if group_size > 1:
                            grad_reduce_handle = dist_reduce_scatter_func(
                                local_data_view,
                                bucket.grad_data,
                                op=reduce_op,
                                group=communication_group,
                                async_op=async_op,
                            )
                    else:
                        if torch.distributed.get_rank() == 0 and force_all_reduce:
                            logger.info(
                                f"Performing reduction using all_reduce because {force_all_reduce=}"
                            )
                        torch.distributed.all_reduce(
                            bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
                        )
        if args.enable_dynamic_grad_comp:
            if args.overlap_grad_reduce and args.all_reduce_time:
                self.timers('DP_time').stop()

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            # Create a new coalescing manager for the inter-instance all-reduce.
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]

                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        if async_op:
            if self.ddp_config.reduce_scatter_with_fp32_accumulation and not force_all_reduce:
                assert (
                    len(self.buckets) == 1
                ), "Only 1 bucket supported with reduce_scatter_with_fp32_accumulation=True"
                # torch.distributed._coalescing_manager does not correctly handle calling our custom
                # collective handle's .wait() method, so we take matters into our own hands here.
                assert grad_reduce_handle is not None
                self.grad_reduce_handle = grad_reduce_handle
            else:
                self.grad_reduce_handle = cm
        else:
            # When using `_coalescing_manager`, even if a synchronous op (async_op=False) is used,
            # `cm` is not None, which is different from when `_coalescing_manager` is not used in
            # which case the torch.distributed._reduce_scatter_base() will return None. In order to
            # maintain consistency with prior code, we need to manually set communication handle to
            # None.
            self.grad_reduce_handle = None

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        args = get_args()

        self.param_gather_dispatched = False
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            return
        # If first batch, start asynchronous communication here. register_grad_ready() launches
        # asynchronous communication only once self.golden_per_param_grad_ready_counts is
        # populated at the end of this first batch.
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.default_stream().wait_stream(self.communication_stream)
            return
        
        if self.grad_reduce_handle is None:
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )

        if args.enable_dynamic_grad_comp:
            if (args.compressor is not None and
                    hasattr(self, '_pending_compressed_data') and
                    self._pending_compressed_data is not None):
                self.grad_reduce_handle.wait()
                self.grad_reduce_handle = None
                for bucket, for_P, for_Q, metadata in self._pending_compressed_data:
                    args.compressor.decompress_bucket(bucket, for_P, for_Q, metadata)
            else:
                self.grad_reduce_handle.wait()
                self.grad_reduce_handle = None
        else:
            self.grad_reduce_handle.wait()
            self.grad_reduce_handle = None


class _ParamAndGradBuffer:

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ) -> _ParamAndGradBucket:
        """
        Helper function that creates a new bucket. Also updates param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global _ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )

        args = get_args()
        if args.enable_dynamic_grad_comp:
            components = []
            offset_in_bucket = 0
            for param in bucket_params:
                param_numel = param.numel()
                param_shape = param.shape
                components.append((param, offset_in_bucket, param_shape))
                offset_in_bucket += param_numel
        else:
            components = None
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
            components=components,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket
