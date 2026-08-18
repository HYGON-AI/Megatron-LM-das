# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import logging
from contextlib import nullcontext
from functools import wraps
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed import _coalescing_manager

from megatron.core.distributed.param_and_grad_buffer import (
    BufferType,
    _ParamAndGradBucket,
    shard_buffer,
)
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import dist_reduce_scatter_func
from megatron.training.global_vars import get_args
from megatron.training import get_timers

from hcu_megatron.training.arguments import get_adaptor_args

logger = logging.getLogger(__name__)


def _param_and_grad_bucket_init_wrapper(_param_and_grad_bucket_init_func):
    @wraps(_param_and_grad_bucket_init_func)
    def wrapper(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
        param_index_map: Dict[torch.nn.Parameter, tuple],
        params_with_extra_main_grads: List[torch.nn.Parameter],
        components: Optional[List[Tuple[torch.nn.Parameter, int, torch.Size]]] = None,
    ):
        _param_and_grad_bucket_init_func(
            self,
            params,
            param_data,
            grad_data,
            offset,
            numel_unpadded,
            gradient_scaling_factor,
            bucket_id,
            param_index_map,
            params_with_extra_main_grads,
        )

        if components is not None:
            self.components = components
        else:
            self.components = []

    return wrapper


def _param_and_grad_bucket_group_init_wrapper(_param_and_grad_bucket_group_init_func):
    @wraps(_param_and_grad_bucket_group_init_func)
    def wrapper(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ):
        _param_and_grad_bucket_group_init_func(
            self,
            buckets,
            ddp_config,
            collective_group,
            collective_group_size,
        )
        self.timers = get_timers()

    return wrapper


class _ParamAndGradBucketGroup:

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        args = get_args()
        adaptor_args = get_adaptor_args()
        if self.is_first_batch and self.grad_reduce_handle is not None:
            # Make this start_grad_sync call a no-op if in first batch and collective has
            # already been dispatched.
            return

        # Drain the predecessor bucket group's reduce-scatter before allocating ours. Only
        # linked under reduce_scatter_with_fp32_accumulation, which holds an intermediate
        # all-to-all output tensor pinned until .wait() runs. We only drain when the
        # predecessor has actually been dispatched this iteration (grad_reduce_handle set):
        # backward param ordering does not always match bucket linkage order (e.g. NVFP4
        # bucket layouts), so the predecessor may not have fired yet when we arrive here.
        # In that case the predecessor will dispatch and drain on its own once its params
        # become ready. The end-of-step finalize loop still catches any bucket that
        # neither a successor nor itself drained.
        if (
            self.previous_grad_reduce_bucket_group is not None
            and self.previous_grad_reduce_bucket_group.grad_reduce_handle is not None
        ):
            self.previous_grad_reduce_bucket_group.finish_grad_sync(
                force_all_reduce=force_all_reduce
            )

        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        # Copy accumulated .main_grad into communication buffer before collective if
        # .main_grad is not in .grad_data already (e.g., because we want to do local
        # gradient accumulation in a higher precision).
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                    param.main_grad_copy_in_grad_buffer.copy_(param.main_grad)

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

            # The RS/AR communication stream needs to wait for the current stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.current_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        # Coalesce communication kernels across buckets in the bucket group.
        if adaptor_args.enable_dynamic_grad_comp and args.compressor is not None:
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
            if adaptor_args.enable_dynamic_grad_comp:
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
        if adaptor_args.enable_dynamic_grad_comp:
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
        adaptor_args = get_adaptor_args()

        self.param_gather_dispatched = False
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            self._copy_back_extra_main_grads()
            return
        if self.grad_reduce_finished:
            return
        # If first batch, start asynchronous communication here. register_grad_ready() launches
        # asynchronous communication only once self.golden_per_param_grad_ready_counts is
        # populated at the end of this first batch.
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.current_stream().wait_stream(self.communication_stream)
            self._copy_back_extra_main_grads()
            self.grad_reduce_finished = True
            return

        if self.grad_reduce_handle is None:
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )

        if adaptor_args.enable_dynamic_grad_comp:
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

        self._copy_back_extra_main_grads()
        self.grad_reduce_finished = True


class _ParamAndGradBuffer:

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
        bucket_params_with_extra_main_grads: List[torch.Tensor],
        nvfp4_packed_start_index: int = None,
        nvfp4_packed_end_index: int = None,
    ) -> _ParamAndGradBucket:
        """
        Helper function that creates a new bucket. Also updates param->bucket mapping.

        For NVFP4 buffers, nvfp4_packed_start_index and nvfp4_packed_end_index
        are provided separately because the param buffer uses packed numel while
        the grad buffer uses full numel.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]
        if nvfp4_packed_start_index is not None:
            assert (
                nvfp4_packed_start_index,
                nvfp4_packed_end_index,
            ) == self.nvfp4_packed_bucket_indices[bucket_id]

        # Get appropriate view into global _ParamAndGradBuffer.
        # For NVFP4, param buffer uses packed offsets; otherwise same as start/end.
        bucketed_param_data = None
        if self.param_data is not None:
            if nvfp4_packed_start_index is not None:
                assert nvfp4_packed_end_index is not None
                bucketed_param_data = self._get(
                    torch.Size([nvfp4_packed_end_index - nvfp4_packed_start_index]),
                    nvfp4_packed_start_index,
                    buffer_type=BufferType.PARAM,
                )
            else:
                bucketed_param_data = self._get(
                    torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
                )
        # Grad buffer always uses full-numel offsets.
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )

        adaptor_args = get_adaptor_args()
        if adaptor_args.enable_dynamic_grad_comp:
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
            param_index_map=self.param_index_map,
            params_with_extra_main_grads=bucket_params_with_extra_main_grads,
            components=components,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket
