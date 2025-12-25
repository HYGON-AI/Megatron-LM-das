# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch.autograd import Variable

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import stream_acquire_context, make_viewless
from megatron.core.pipeline_parallel.utils import ScheduleNode as MegatronCoreScheduleNode


class ScheduleNode(MegatronCoreScheduleNode):
    """Base node for fine-grained scheduling.

    This class represents a computational node in the pipeline schedule.
    It handles the execution of forward and backward operations on a stream.
    """
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
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} forward")
            with torch.cuda.stream(self.stream):
                if stream_wait_event is not None:
                    stream_wait_event.wait(self.stream)

                self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
                for i, input in enumerate(self.inputs):
                    if input is not None:
                        input.requires_grad = inputs[i].requires_grad

                data = tuple(self.inputs)
                data = self.forward_func(*data)

                if not isinstance(data, tuple):
                    data = make_viewless(data)
                else:
                    data = tuple(
                        [make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data]
                    )

                self.output = data

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

            torch.cuda.nvtx.range_pop()

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            for input in inputs:
                if input is not None:
                    input.record_stream(self.stream)
                    input.untyped_storage().resize_(0)

        return self.output

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
        with stream_acquire_context(self.stream, self.event):
            torch.cuda.nvtx.range_push(f"{self.name} backward")
            with torch.cuda.stream(self.stream):
                if stream_wait_event is not None:
                    stream_wait_event.wait(self.stream)

                outputs = self.output
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                assert len(outputs) == len(output_grad), (
                    f"{len(outputs)} of {type(outputs[0])} is not equal to "
                    f"{len(output_grad)} of {type(output_grad[0])}"
                )
                output_grad = self.backward_func(outputs, output_grad)

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

            torch.cuda.nvtx.range_pop()

        # output_grad maybe from another stream
        if output_grad:
            for g in output_grad:
                if g is not None:
                    g.record_stream(self.stream)

        grads = self.get_grad()
        self._release_state()

        return grads


class VppContextManager:
    """A reusable context manager for switch vpp stage"""

    def __init__(self, vpp_rank):
        self.vpp_rank = vpp_rank

    def __enter__(self):
        self.origin_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.vpp_rank)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.origin_vpp_rank)
