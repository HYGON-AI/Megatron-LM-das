# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Callable, Optional

import torch

from megatron.core.pipeline_parallel.utils import make_viewless


def set_ideal_affinity_for_current_gpu():
    pass


class NoopScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained (e.g., dense layer doesn't need
    moe_dispatch and moe_combine). It simply returns its inputs unchanged
    in both forward and backward passes.
    """

    def forward(self, inputs, stream_wait_event=None, stream_record_event=None, is_recompute=False):
        """Passes through inputs unchanged in the forward pass."""
        return inputs

    def backward(self, outgrads, stream_wait_event=None, stream_record_event=None):
        """Passes through gradients unchanged in the backward pass."""
        return outgrads


class ScheduleNode():
    """Base node for fine-grained scheduling.

    This class represents a computational node in the pipeline schedule.
    It handles the execution of forward and backward operations on a stream.
    """

    def __init__(
        self,
        forward_func: Callable,
        stream: torch.cuda.Stream,
        event: torch.cuda.Event,
        backward_func: Optional[Callable] = None,
        free_input: bool = False,
        name: str = "schedule_node",
    ):
        """Initialize a schedule node.

        Args:
            forward_func (callable): Function to execute during the forward pass.
            stream (Callable): Func that returns CUDA stream for computation.
                This can be either a 'compute' stream or a 'communicate' stream.
                - 'compute' stream: Used for computational nodes like attention and experts.
                - 'communicate' stream: Used for nodes that handle token communication,
                  such as token dispatch and combine operations in MoE layers.
            event (torch.cuda.Event): The CUDA event used for synchronization. Each
                microbatch within a model chunk shares the same event, which is used
                to manage dependencies between nodes operating on different streams.
            backward_func (callable, optional): Function for the backward pass.
            free_input (bool): Flag to indicate if the input should be freed after the
                forward pass.
            name (str): Name of the node for debugging purposes.
        """
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self.default_backward_func
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.inputs = None
        self.outputs = None
        self.is_recompute = False

    def forward(self, inputs=(), stream_wait_event=None, stream_record_event=None):
        """Schedule node forward"""
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(
                *inputs,
                stream_wait_event=stream_wait_event,
                stream_record_event=stream_record_event,
            )

    def _forward(self, *inputs, stream_wait_event=None, stream_record_event=None):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} forward"):
            if stream_wait_event is not None:
                stream_wait_event.wait(self.stream)

            self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
            for i, input in enumerate(self.inputs):
                if input is not None:
                    # requires_grad is set to true for post_process module, otherwise
                    #  backward will raise error when recomputation is enabled.
                    input.requires_grad = True if self.name == "post_process" else inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple([make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data])

            self.outputs = data

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            for input in inputs:
                if input is not None:
                    input.record_stream(self.stream)
                    input.untyped_storage().resize_(0)

        return self.outputs

    def get_output(self):
        """Get the forward output"""
        return self.outputs

    def backward(self, output_grad, stream_wait_event=None, stream_record_event=None):
        """Schedule node backward"""
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(
                *output_grad,
                stream_wait_event=stream_wait_event,
                stream_record_event=stream_record_event,
            )

    def _backward(self, *output_grad, stream_wait_event=None, stream_record_event=None):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} backward"):
            if stream_wait_event is not None:
                stream_wait_event.wait(self.stream)

            outputs = self.outputs
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            assert len(outputs) == len(output_grad), (
                f"{len(outputs)} of {type(outputs[0])} is not equal to "
                f"{len(output_grad)} of {type(output_grad[0])}"
            )
            output_grad = self.backward_func(outputs, output_grad)

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

        # output_grad maybe from another stream
        if output_grad:
            for g in output_grad:
                if g is not None:
                    g.record_stream(self.stream)

        grads = self.get_grad()
        self._release_state()

        return grads

    def _release_state(self):
        """Clear the state of the node"""
        self.inputs = None
        self.outputs = None
        del self.forward_func
        del self.backward_func


LM_HEAD_RES_REDUCE_STREAM = None

def get_lm_head_res_reduce_stream():
    global LM_HEAD_RES_REDUCE_STREAM
    return LM_HEAD_RES_REDUCE_STREAM


def set_lm_head_res_reduce_stream(stream=None):
    global LM_HEAD_RES_REDUCE_STREAM
    if LM_HEAD_RES_REDUCE_STREAM is not None:
        return

    if stream is None:
        stream = torch.cuda.Stream(device="cuda")

    LM_HEAD_RES_REDUCE_STREAM = stream
