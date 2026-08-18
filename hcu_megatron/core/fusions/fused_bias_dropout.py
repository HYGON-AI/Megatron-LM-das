# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from megatron.core.jit import jit_fuser
from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
    bias_dropout_add_unfused,
)


if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointManager


def get_bias_dropout_add(
    training, fused, mhc_recompute_manager: Optional['CheckpointManager'] = None
):
    """
    Get the bias-dropout-add function.

    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: Optional CheckpointManager for checkpoint management.
            When provided, the returned function will wrap the BDA operation with
            CheckpointWithoutOutput for memory-efficient recomputation.

    Returns:
        A callable that performs bias-dropout-add operation.
    """
    if mhc_recompute_manager is not None:
        # Return a checkpointed version that handles tuple unpacking internally
        return _get_checkpointed_bda(training, fused, mhc_recompute_manager)

    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
    

def _get_checkpointed_bda(training, fused, mhc_recompute_manager: 'CheckpointManager'):
    """
    Create a checkpointed bias-dropout-add function.

    This function handles:
    1. Tuple unpacking for x_with_bias (required because save_for_backward can't save tuples)
    2. Non-tensor arguments like dropout probability (handled by CheckpointWithoutOutput)
    3. Auto-registration to the CheckpointManager

    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: CheckpointManager for checkpoint management.

    Returns:
        A callable that performs checkpointed bias-dropout-add operation.
    """
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    # Get the underlying BDA function
    if fused:
        if training:
            bda_func = bias_dropout_add_fused_train
        else:
            bda_func = bias_dropout_add_fused_inference
    else:
        bda_func = bias_dropout_add_unfused(training)

    def _checkpointed_bda(x_with_bias, residual, prob):
        """
        Checkpointed BDA that handles tuple unpacking internally.

        Args:
            x_with_bias: Either a tuple (x, bias) or a single tensor x.
            residual: Residual tensor.
            prob: Dropout probability.

        Returns:
            Output tensor after bias-dropout-add.
        """
        # Create checkpoint with manager
        ckpt = CheckpointWithoutOutput(ckpt_manager=mhc_recompute_manager)

        # Handle case where x_with_bias might be a single tensor (e.g., from IdentityOp)
        if isinstance(x_with_bias, tuple):
            x, bias = x_with_bias
        else:
            x = x_with_bias
            bias = None

        # Wrapper function that re-packs the tuple for the actual BDA function
        def _bda_wrapper(output, bias, res, dropout):
            return bda_func((output, bias), res, dropout)

        # Call checkpoint with unpacked arguments
        result = ckpt.checkpoint(_bda_wrapper, x, bias, residual, prob)

        # No-op when manager is set - manager handles all discarding uniformly
        ckpt.discard_output_and_register_recompute(result)

        return result

    return _checkpointed_bda
