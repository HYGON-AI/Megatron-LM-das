# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
from collections.abc import Callable
from functools import wraps
from typing import List, TypeVar, Union
from typing_extensions import TypeVarTuple, Unpack

import torch
from torch.utils.checkpoint import _get_autocast_kwargs

from megatron.core.tensor_parallel.random import _set_cuda_rng_state, get_cuda_rng_tracker, _get_all_rng_states, CheckpointWithoutOutputFunction

from hcu_megatron.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager


_R = TypeVar('_R')
_Ts = TypeVarTuple('_Ts')


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


class CheckpointWithoutOutput(object):
    """
    Checkpoint a model or part of the model and release the output.

    For the normal 'checkpoint` function, the outputs of it may be saved by the following
    modules for their backward computation. However, the output of the checkpointed function is
    re-generated at recomputation, so the output store is not technically needed. This method can
    manually discard the output in the forward pass and restore it by recomputation in the
    backward pass to reduce the memory usage.

    Due to the reason above, to save memory with this method, the caller should make sure that the
    discarded output tensors are directly saved in the following modules for backward computation.
    """

    def __init__(self, fp8=False, ckpt_manager=None):
        """
        Initialize CheckpointWithoutOutput.

        Args:
            fp8: Whether to use FP8 mode. Defaults to False.
            ckpt_manager: Optional CheckpointManager instance. When provided,
                         checkpoint() will auto-register to the manager, and
                         discard_output_and_register_recompute() will only discard
                         output without registering individual hooks.
        """
        self.fp8 = bool(fp8)
        self.ckpt_manager = ckpt_manager
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function: Callable[[Unpack[_Ts]], _R], *args: Unpack[_Ts]) -> _R:
        """
        Checkpoint function.

        If ckpt_manager was provided during initialization, this checkpoint
        will be automatically registered to the manager after execution.
        """

        # If in cuda graph warmup, disable checkpointing, as 'discard_output_and_register_recompute'
        # may be called in a separate graph warmup.
        from megatron.core.transformer.cuda_graphs import is_graph_warmup

        if is_graph_warmup():
            return run_function(*args)

        self.run_function = run_function

        self.rng_states = _get_all_rng_states()

        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)

        # Auto-register to manager if provided
        if self.ckpt_manager is not None:
            self.ckpt_manager.add_checkpoint(self)

        return outputs


class CheckpointManager:
    """
    Manages multiple CheckpointWithoutOutput objects within a TransformerBlock
    cross layer recomputations, enabling unified recomputation during backward pass.
    This is particularly useful for scenarios where multiple checkpoint operations have
    sequential dependencies (i.e., the output of one checkpoint is the input of the next).

    Usage:
        ckptManager = CheckpointManager()
        ckpt_function = CheckpointWithoutOutput(ckpt_manager=ckptManager)
        ckpt_function.checkpoint(run_function, *args)
        # other checkpointed operations
        ckpt_manager.discard_all_outputs_and_register_unified_recompute(final_output)
    """

    def __init__(self):
        self.checkpoints = []
        # Set by TransformerBlock before each layer forward.
        # When True, the layer should keep block-boundary output uncheckpointed.
        self.is_last_layer_in_recompute_block = False

    def add_checkpoint(self, ckpt):
        """Add a checkpoint to the manager."""
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        if not isinstance(ckpt, CheckpointWithoutOutput):
            raise TypeError("Expected CheckpointWithoutOutput object")
        if ckpt.outputs is None:
            raise ValueError("CheckpointWithoutOutput must call checkpoint() before adding")
        self.checkpoints.append(ckpt)

    def discard_all_outputs_and_register_unified_recompute(self, hook_tensor):
        """Discard all checkpoint outputs to save memory and register unified recompute hook."""
        for ckpt in self.checkpoints:
            for output in ckpt.outputs:
                output.untyped_storage().resize_(0)

        # Register unified recompute hook
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._unified_recompute_hook)

    def _unified_recompute_hook(self, grad_output):
        for ckpt in self.checkpoints:
            # Call _recompute for each checkpoint in forward order
            # The _recompute method will restore the output tensor storage
            ckpt._recompute(None)
