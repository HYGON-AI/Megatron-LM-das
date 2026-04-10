import torch

from megatron.core.pipeline_parallel.utils import stream_acquire_context, make_viewless
from megatron.core.pipeline_parallel.utils import ScheduleNode as MegatronCoreScheduleNode


class NoopScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained (e.g., dense layer doesn't need
    moe_dispatch and moe_combine). It simply returns its inputs unchanged
    in both forward and backward passes.
    """

    def forward(self, inputs, stream_wait_event=None, stream_record_event=None):
        """Passes through inputs unchanged in the forward pass."""
        return inputs

    def backward(self, outgrads, stream_wait_event=None, stream_record_event=None):
        """Passes through gradients unchanged in the backward pass."""
        return outgrads


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
                    # Manually trigger the memory release of dgrad tensor
                    # to avoid delayed garbage collection. If
                    # delay_grads_release is True, dgrad is last used in
                    # wgrad compute and skip the release here.
                    if self.manual_release_grads and not self.delay_grads_release:
                        g.untyped_storage().resize_(0)

        grads = self.get_grad()
        self._release_state()

        return grads
