# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Callable

import torch

from megatron.core.models.common.utils import TransformerLayerNode as MegatronCoreTransformerLayerNode
from megatron.core.pipeline_parallel.utils import make_viewless

try:
    from transformer_engine.pytorch.ep import is_symm_backed
except ImportError:
    is_symm_backed = None

from hcu_megatron.training.arguments import get_adaptor_args


class TransformerLayerNode(MegatronCoreTransformerLayerNode):
    """Base class for transformer layer computation nodes.

    This class provides common functionality for different types of
    transformer layer nodes (attention, MLP, etc.)
    """

    def forward(self, inputs=(), stream_wait_event=None, stream_record_event=None, is_recompute=False):
        """Schedule node forward"""
        self.is_recompute = is_recompute
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        output = self._forward(
                *inputs,
                stream_wait_event=stream_wait_event,
                stream_record_event=stream_record_event,
                is_recompute=is_recompute,
            )
        if self.is_layer_last_node:
            self._post_forward_hook()
        return output

    def _forward(self, *inputs, stream_wait_event=None, stream_record_event=None, is_recompute=False):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} forward"):
            if stream_wait_event is not None:
                stream_wait_event.wait(self.stream)

            self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
            for i, input in enumerate(self.inputs):
                if input is not None:
                    input.requires_grad = inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data, is_recompute=is_recompute)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple([make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data])

            if not get_adaptor_args().integrate_recompute_to_ep_comm_overlap or is_recompute:
                self.outputs = data

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            for input in inputs:
                if input is not None:
                    input.record_stream(self.stream)
                    # Skip symmetric-memory (zero-copy EP) buffers
                    if not (
                        self.ncclep_zero_copy
                        and is_symm_backed is not None
                        and is_symm_backed(input)
                    ):
                        input.untyped_storage().resize_(0)

        return data

    def forward_impl(self, *args, is_recompute=False):
        """Calls the submodule as the forward pass."""
        return self.submodule(self, *args, is_recompute=is_recompute)

    def backward(self, *output_grad, stream_wait_event=None, stream_record_event=None):
        """Execute backward pass and corresponding hooks."""
        grads = super(MegatronCoreTransformerLayerNode, self).backward(
            *output_grad,
            stream_wait_event=stream_wait_event,
            stream_record_event=stream_record_event,
        )
        if not self.delay_wgrad_compute and self.is_layer_first_node:
            self._post_backward_hook()
        return grads

    def detach(self, t):
        """Detaches a tensor and stores it for backward computation."""
        if get_adaptor_args().integrate_recompute_to_ep_comm_overlap and not self.is_recompute:
            return t
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        if not get_adaptor_args().integrate_recompute_to_ep_comm_overlap or self.is_recompute:
            self.before_detached = self.before_detached + (t,)
            self.detached = self.detached + (detached,)
        return detached
