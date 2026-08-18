# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.jit import jit_fuser


@jit_fuser
def _fused_rmsnorm_silu_gate(
    x: Tensor, gate: Tensor, weight: Tensor, eps: float, zero_centered_gamma: bool
) -> Tensor:
    """Fuse RMSNorm and output gate for the common GDN RMSNorm+silu path."""
    x_dtype = x.dtype
    x = x.reshape(-1, x.shape[-1])
    gate = gate.reshape(-1, gate.shape[-1])

    x_float = x.float()
    rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    norm_weight = weight.float()
    if zero_centered_gamma:
        norm_weight = norm_weight + 1.0
    y = x_float * rms * norm_weight
    y = y * F.silu(gate.float())
    return y.to(x_dtype)


class GatedDeltaNet:
    def _apply_gated_norm(self, x, gate):
        if self._can_use_fused_gated_rmsnorm():
            return _fused_rmsnorm_silu_gate(
                x,
                gate,
                self.out_norm.weight,
                float(self.out_norm.eps),
                self.config.layernorm_zero_centered_gamma,
            )
        return self._apply_gated_norm_fallback(x, gate)

    @jit_fuser
    def _apply_gated_norm_fallback(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)

        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    def _can_use_fused_gated_rmsnorm(self):
        return (
            self.config.normalization == "RMSNorm"
            and self.activation in ["silu", "swish"]
            and hasattr(self.out_norm, "weight")
            and hasattr(self.out_norm, "eps")
        )
