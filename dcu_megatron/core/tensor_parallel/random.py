from functools import wraps
from typing import List, Union
import torch
from torch.utils.checkpoint import _get_autocast_kwargs
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from torch.utils.checkpoint import detach_variable

from dcu_megatron.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
import contextlib

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
    _set_cuda_rng_state,
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

class RngStateContext:
    """Random number generator state context."""
    def __init__(self, cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
        self.fwd_cpu_rng_state = cpu_rng_state
        self.fwd_cuda_rng_state = cuda_rng_state
        self.fwd_cuda_rng_state_tracker = cuda_rng_state_tracker


def checkpoint_wrapper(checkpoint):
    @wraps(checkpoint)
    def wrapper(function, distribute_saved_activations, *args):
        # Use the original checkpoint logic when riPipe is disabled.
        if not get_pipeline_checkpoint_manager().open_ri_pipe:
            return checkpoint(function, distribute_saved_activations, *args)

        # Execute the function directly when recomputation is disabled.
        if not get_pipeline_checkpoint_manager().chunk_do_recompute:
            return function(*args)

        if distribute_saved_activations:
            raise RuntimeError("no distributed")

        # _get_autocast_kwargs has different signatures across PyTorch releases.
        # Prefer the newer device-aware form, but fall back to the older no-arg API.
        try:
            device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device='cuda')
        except TypeError:
            device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

        # Save RNG state from the forward pass.
        fwd_rng_state = RngStateContext(torch.get_rng_state(), torch.cuda.get_rng_state(), get_cuda_rng_tracker().get_states())

        # Storage for tensors captured by saved_tensors_hooks.
        storage: List[Union[torch.Tensor, None]] = []
        counter = 0

        def pack(x):
            nonlocal counter
            counter += 1
            return counter - 1

        def early_unpack():
            """
            Function that triggers recomputation ahead of backward.
            """
            def inner_pack(inner):
                storage.append(inner.detach())
                return None

            def inner_unpack(packed):
                raise RuntimeError("You are calling backwards on a tensor that is never exposed. Please open an issue.")

            # Save the current RNG state.
            bwd_cpu_rng_state = torch.get_rng_state()
            bwd_cuda_rng_state = torch.cuda.get_rng_state()
            bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

            # Restore the RNG state from the forward pass.
            torch.set_rng_state(fwd_rng_state.fwd_cpu_rng_state)
            _set_cuda_rng_state(fwd_rng_state.fwd_cuda_rng_state, device=torch.cuda.current_device())
            get_cuda_rng_tracker().set_states(fwd_rng_state.fwd_cuda_rng_state_tracker)

            # Run recomputation.
            with torch.enable_grad(), \
                    torch.amp.autocast('cuda', **device_autocast_kwargs) if device_autocast_kwargs else contextlib.nullcontext(), \
                    torch.amp.autocast('cpu', **cpu_autocast_kwargs) if cpu_autocast_kwargs else contextlib.nullcontext(), \
                    torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                _unused = function(*args)

            # Restore the current RNG state.
            torch.set_rng_state(bwd_cpu_rng_state)
            _set_cuda_rng_state(bwd_cuda_rng_state, device=torch.cuda.current_device())
            get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        # Queue early_unpack when advance recomputation is enabled.
        if get_pipeline_checkpoint_manager().do_pre_recompute:
            get_pipeline_checkpoint_manager().add_recompute(early_unpack)

        def unpack(x):
            """
            Unpack tensors and restore them during backward.
            """
            if len(storage) == 0:
                if get_pipeline_checkpoint_manager().do_pre_recompute:
                    raise RuntimeError(f"rank-{torch.distributed.get_rank()}: recompute is not done")

                def inner_pack(inner):
                    storage.append(inner.detach())
                    return None

                def inner_unpack(packed):
                    raise RuntimeError(
                        "You are calling backwards on a tensor that is never exposed. Please open an issue.")

                # Save the current RNG state.
                bwd_cpu_rng_state = torch.get_rng_state()
                bwd_cuda_rng_state = torch.cuda.get_rng_state()
                bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

                # Restore the RNG state from the forward pass.
                torch.set_rng_state(fwd_rng_state.fwd_cpu_rng_state)
                _set_cuda_rng_state(fwd_rng_state.fwd_cuda_rng_state, device=torch.cuda.current_device())
                get_cuda_rng_tracker().set_states(fwd_rng_state.fwd_cuda_rng_state_tracker)

                # Run recomputation.
                with torch.enable_grad(), \
                        torch.amp.autocast('cuda', **device_autocast_kwargs) if device_autocast_kwargs else contextlib.nullcontext(), \
                        torch.amp.autocast('cpu', **cpu_autocast_kwargs) if cpu_autocast_kwargs else contextlib.nullcontext(), \
                        torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args)

                # Restore the current RNG state.
                torch.set_rng_state(bwd_cpu_rng_state)
                _set_cuda_rng_state(bwd_cuda_rng_state, device=torch.cuda.current_device())
                get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

            r = storage[x]
            storage[x] = None
            return r

        # Pack and unpack tensors with saved_tensors_hooks.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = function(*args)
        return output

    return wrapper
