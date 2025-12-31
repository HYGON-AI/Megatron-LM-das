import contextlib

import torch

try:
    import transformer_engine  # pylint: disable=unused-import
    from transformer_engine.pytorch.distributed import activation_recompute_forward
    from transformer_engine.pytorch.fp8 import fp8_autocast

    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False

from megatron.core.tensor_parallel.random import (
    _fork_rng,
    _get_all_rng_states,
    _set_all_rng_states,
)
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputFunction as MegatronCoreCheckpointWithoutOutputFunction


class CheckpointWithoutOutputFunction(MegatronCoreCheckpointWithoutOutputFunction):
    """
    Checkpoint Function Helper for CheckpointWithouOutput.
    Save context for recompute.
    """

    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        # Get the inputs from the context instead of the saved tensors
        # because the saved tensors are already cached by the recomputation. (by the activation reloading? dongcl)
        # This is to avoid double-reloading the inputs in CPU offloading scenario.
        inputs = ctx.inputs
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput(object):
    def checkpoint(self, run_function, *args):
        """Checkpoint function."""
        self.run_function = run_function

        self.rng_states = _get_all_rng_states()

        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)
        return outputs

    def _recompute(self, _):
        """Used as a hook to recompute the output."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        with _fork_rng():
            _set_all_rng_states(*self.rng_states)

            if self.fp8:
                recompute_ctx = activation_recompute_forward(
                    activation_recompute=True, recompute_phase=True
                )
                fp8_ctx = fp8_autocast(enabled=self.ctx.fp8, fp8_recipe=self.ctx.fp8_recipe)
            else:
                recompute_ctx = contextlib.nullcontext()
                fp8_ctx = contextlib.nullcontext()

            inputs = self.ctx.saved_tensors

            # do not know why, if saved_tensors is handled by saved_tensor_hook, grad of inputs will be None (not nan)
            # detach it to bypass
            def detach(t):
                if isinstance(t, torch.Tensor):
                    requires_grad = t.requires_grad
                    t = t.detach()
                    t.requires_grad_(requires_grad)
                return t

            inputs = tuple(detach(t) for t in inputs)
            with torch.enable_grad(), fp8_ctx, recompute_ctx:
                outputs = self.run_function(*inputs)

        self.run_function = None
        self.rng_states = None

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # restore the recomputed memory without changing the metadata
        with torch.no_grad():
            for output, recomputation_output in zip(self.outputs, outputs):
                output_size = recomputation_output.untyped_storage().size()
                output.untyped_storage().resize_(output_size)
                output.untyped_storage().copy_(recomputation_output.untyped_storage())

        self.ctx.outputs = outputs
        self.ctx.inputs = inputs
        self.outputs = None
        self.ctx = None
