import contextlib
from functools import wraps
from typing import List, Union

import torch
from torch.utils.checkpoint import _get_autocast_kwargs

from megatron.core.tensor_parallel.random import _set_cuda_rng_state, get_cuda_rng_tracker

from dcu_megatron.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager


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
