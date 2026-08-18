# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core.transformer.spec_utils import build_module

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointManager

from megatron.core.utils import nvtx_range_pop, nvtx_range_push

_HYPERCONNECTION_LOGGING_TRACKER = {}


def get_hyperconnection_logging_tracker():
    """Return the moe layer wise tracker."""
    global _HYPERCONNECTION_LOGGING_TRACKER
    return _HYPERCONNECTION_LOGGING_TRACKER


@torch.compile
def _sinkhorn_iterations(input_logits: Tensor, num_iterations: int, eps: float) -> Tensor:
    row_max = input_logits.max(dim=-1, keepdim=True).values
    M = torch.exp(input_logits - row_max)
    for _ in range(num_iterations):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=eps)
    return M


class SinkhornKnopp(torch.autograd.Function):
    """Sinkhorn-Knopp projection to doubly stochastic matrix.

    This is an autograd.Function because the iterative forward is re-executed
    during backward (under torch.enable_grad) so that PyTorch's autograd can
    differentiate through it without storing all intermediate iteration states.
    """

    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Run Sinkhorn iterations and save inputs for backward recomputation."""
        M = _sinkhorn_iterations(input_logits, num_iterations, eps)
        ctx.save_for_backward(input_logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return M

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Recompute forward under enable_grad and back-propagate."""
        (input_logits,) = ctx.saved_tensors
        with torch.enable_grad():
            logits = input_logits.detach().requires_grad_(True)
            M = _sinkhorn_iterations(logits, ctx.num_iterations, ctx.eps)
            M.backward(grad_output)
        return logits.grad, None, None


def native_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Native Sinkhorn-Knopp (autograd.Function wrapper)."""
    return SinkhornKnopp.apply(input_logits, num_iterations, eps)


@torch.compile
def native_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Native n-stream weighted aggregation: out = sum_j(h_pre_j * x_j)."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


@torch.compile
def native_h_post_bda(
        h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Native H_res @ residual + H_post * (x [+ bias])."""
    s, b, n, C = original_residual.shape
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, C)
    mixed = torch.bmm(h_res_batched, residual_batched).view(s, b, n, C)
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)  # h_post.unsqueeze(-1) [s, b, n, 1]; x.unsqueeze(2) [s, b, 1, C]
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


@torch.compile
def native_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Native fused projection + RMS normalization."""
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = 1.0 / v
    return proj, r


# ============================================================================
# HyperConnectionModule
# ============================================================================


# TODO: keep hyper connection in fp32 computation
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int, hc_type: str = 'attn', ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.mhc_use_tilekernels = config.mhc_use_tilekernels
        self.block_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon

        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values
        # - H_res: n^2 values (before Sinkhorn projection)
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, self.n * self.n + 2 * self.n, bias=False
        )

        if self.mhc_use_tilekernels:
            assert not config.use_vwn, "mhc_use_tilekernels does not support VWN"
            assert not config.mhc_lite, "mhc_use_tilekernels does not support mhc_lite"
            assert not config.mhc_hres_vwnstyle, "mhc_use_tilekernels does not support mhc_hres_vwnstyle"
            assert not config.use_mhc_svd, "mhc_use_tilekernels does not support use_mhc_svd"
            assert config.mhc_fuse_h_post_compute, (
                "mhc_use_tilekernels requires --mhc-fuse-h-post-compute (TileKernels "
                "always fuses H_res @ residual + H_post * x)."
            )
            assert config.num_residual_streams == 4, (
                "TileKernels' mhc_post backward is hard-coded for mhc_mult=4"
            )
            assert config.mhc_tau == 1.0, (
                "TileKernels' sinkhorn kernel has no tau argument (equivalent to tau=1.0). "
                "Use --mhc-tau 1.0 with --mhc-use-tilekernels."
            )

            self.hc_type = hc_type
            self.is_mtp = False
            mhc_mult = self.n
            mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
            self.mhc_mult3 = mhc_mult3
            self.log_amax_per_step = config.mhc_log_amax_per_step

            # Learnable parameters (TileKernels layout).
            self.scale = nn.Parameter(torch.full((3,), 0.01, dtype=torch.float32))
            self.base = nn.Parameter(torch.zeros(mhc_mult3, dtype=torch.float32))

            self.norm_weight = nn.Parameter(
                torch.ones(mhc_mult * self.hidden_size, dtype=torch.float32)
            )

            self._init_base()

            # TileKernels mhc_pre / mhc_post hyperparameters.
            self.pre_eps = 1e-6
            self.post_mult_value = 2.0
            self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.sinkhorn_eps = 1e-6
            self.norm_eps = config.layernorm_epsilon
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            if self.tp_size > 1:
                for param in [self.scale, self.base, self.norm_weight]:
                    setattr(param, 'allreduce', True)
                    setattr(param, 'sequence_parallel', config.sequence_parallel)

        else:
            init_alpha = config.mhc_init_gating_factor
            # Learnable scaling factors (Eq. 5 in paper)
            self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
            self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
            self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))
            # Static bias terms
            self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))

            self._sinkhorn_op = native_sinkhorn
            self._h_aggregate_op = native_h_aggregate
            self._h_post_bda_op = native_h_post_bda
            self._proj_rms_op = native_proj_rms

            self._init_weights()

    def _init_base(self) -> None:
        nn.init.zeros_(self.mapping_proj.weight)
        n = self.n
        init_h_pre = torch.full((n,), -8.0)
        ln = self.layer_number
        if self.config.mhc_init_hpre_use_module_layer:
            ln = ln * 2 + int(self.hc_type == 'mlp')
        init_h_pre[ln % n] = 0.0

        init_h_post = torch.zeros(n)

        if self.config.mhc_tau == 1:
            init_h_res = torch.full((n, n), -8.0)
            init_h_res.fill_diagonal_(0.0)
        else:
            init_h_res = torch.zeros(n, n)
            init_h_res.fill_diagonal_(1.0)

        with torch.no_grad():
            self.base.copy_(
                torch.cat([init_h_pre, init_h_post, init_h_res.view(-1)], dim=0)
            )

    def _maintain_float32_params(self) -> None:
        self.scale.data = self.scale.data.float()
        self.base.data = self.base.data.float()
        self.norm_weight.data = self.norm_weight.data.float()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Projection + RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        x_2d = x.reshape(s * b, nC)
        proj, r = self._proj_rms_op(x_2d, self.mapping_proj.weight, self.norm_eps)
        return proj.view(s, b, -1), r.view(s, b, 1)

    @torch.compile
    def _compute_h(self, proj: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.

        Args:
            proj: [s, b, n^2 + 2n] - projected hidden states
            r: [s, b, 1] - scaling factors

        Returns:
            h_pre: [s, b, n] - aggregation weights
            h_post: [s, b, n] - expansion weights
            h_res: [s, b, n^2] - residual mixing logits
        """
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )  # 总长度: n + n + n^2 = n^2 + 2n
        h = r * proj * alpha_ + self.bias  # [s, b, n^2+2n]
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid()  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n: 2 * self.n].sigmoid() * 2  # [s, b, n]
        h_res = h[..., 2 * self.n:]
        return h_pre, h_post, h_res

    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.

        Reference: Eq. (5) and (8) in mHC paper

        Args:
            x: [s, b, n*C] - n-stream hidden states

        Returns:
            h_pre: [s, b, n] - aggregation weights (sigmoid activated)
            h_post: [s, b, n] - expansion weights (2*sigmoid activated)
            h_res: [s, b, n, n] - residual mixing matrix (doubly stochastic)
        """
        s, b, _ = x.shape
        with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
            proj, r = self._projection_and_get_norm(x)
        with torch.cuda.nvtx.range("HyperConnection::compute_h"):
            h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = self._sinkhorn_op(
            h_res.view(s, b, self.n, self.n), self.sinkhorn_iterations, self.norm_eps
        )  # [s, b, n, n]

        return h_pre, h_post, h_res

    @torch.compile
    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.

        Computes: H_post^T @ x

        Args:
            x: Input tensor, can be either:
               - [s, b, C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape

        if x.dim() == 1:
            # x is bias with shape [C], need to broadcast to [s, b, 1, C]
            C = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)
        else:
            # x is [s, b, C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(2)  # [s, b, 1, C]

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        # Using broadcast multiply instead of einsum
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(s, b, n * C)

    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
            self,
            x_with_bias: Tuple[Tensor, Optional[Tensor]],
            h_post: Tensor,
            manager: Optional['CheckpointManager'] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.

        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional CheckpointManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            # Checkpoint _apply_h_post to discard the output
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, x, h_post
            )

            # Checkpoint _apply_h_post for bias if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post(x, h_post)
            bias_out = self._apply_h_post(bias, h_post) if bias is not None else None

        return x_out, bias_out

    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights

        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size
        x_streams = x.view(s, b, self.n, C)
        return self._h_aggregate_op(x_streams, h_pre)

    @torch.compile
    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.

        Computes: H_res.T @ residual

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """

        """
                输入 residual: [s, b, n*C]
               |
               ├─→ view(s, b, n, C) 
               |        |
               |        └─→ [s, b, n, C]
               |                |
               |                └─→ view(s*b, n, C)
               |                         |
        输入 h_res: [s, b, n, n]         |
               |                         |
               └─→ view(s*b, n, n)       |
                        |                |
                        └────→ bmm ──────┘
                                 |
                            [s*b, n, C]
                                 |
                            view(s, b, n*C)
                                 |
                           输出: [s, b, n*C]
        """
        s, b, _ = residual.shape
        n = self.n
        C = self.hidden_size

        # Reshape for bmm: [s, b, n, n] -> [s*b, n, n]
        h_res_batched = h_res.view(s * b, n, n)
        # [s, b, n*C] -> [s, b, n, C] -> [s*b, n, C]
        residual_batched = residual.view(s, b, n, C).view(s * b, n, C)

        # Batch matrix multiply: [s*b, n, n] @ [s*b, n, C] -> [s*b, n, C]
        mixed = torch.bmm(h_res_batched, residual_batched)

        return mixed.view(s, b, n * C)

    def forward(
            self, hidden_states: Tensor, mhc_recompute_manager: Optional['CheckpointManager'] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: Optional CheckpointManager for checkpoint management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(hidden_states, mhc_recompute_manager)
        elif self.mhc_use_tilekernels:
            return self._tilelang_forward(hidden_states)
        else:
            return self._forward_normal(hidden_states)

    def _forward_normal(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)

        # Aggregate for layer input
        with torch.cuda.nvtx.range("HyperConnection::aggregate"):
            aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post

    def _tilelang_forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run TileKernels' `mhc_pre`.

        Returns a flat 3-tuple so `CheckpointWithoutOutput` can iterate storages:
            layer_input: [s, b, C] aggregated input for the downstream sublayer.
            post_mix:    [s, b, n, 1]  (occupies the original `h_res_or_residual` slot)
            comb_mix:    [s, b, n, n]  (occupies the original `h_post` slot)
        """
        from tile_kernels.modeling.mhc import mhc_pre
        s, b, nC = hidden_states.shape
        residual_tk = hidden_states.view(s, b, self.n, self.block_size)

        # TileKernels' _MHCFnNormwMerge backward writes directly into .main_grad
        # and returns None for fn/norm_weight when those attributes exist. The
        # mapping weight already flows through .float(), which creates a fresh
        # tensor when params are bf16. norm_weight is stored in fp32, so .float()
        # may be a no-op; explicitly clone it to keep autograd materializing
        # param.grad for Megatron DDP overlap hooks.
        norm_weight = self.norm_weight.clone() if getattr(self.norm_weight, 'main_grad',
                                                          None) is not None else self.norm_weight

        nvtx_range_push("HyperConnectionTKModule.forward.mhc_pre")

        layer_input, (post_mix, comb_mix) = mhc_pre(
            residual_tk,
            self.mapping_proj.weight.float(),
            self.scale,
            self.base,
            norm_eps=self.norm_eps,
            norm_weight=norm_weight,
            mhc_mult=self.n,
            post_mult_value=self.post_mult_value,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            sinkhorn_repeat=self.sinkhorn_iterations,
        )
        nvtx_range_pop("HyperConnectionTKModule.forward.mhc_pre")

        self._log_stats(comb_mix)

        return layer_input, comb_mix, post_mix

    def _log_stats(self, comb_mix: Tensor) -> None:
        """Periodic amax / doubly-stochastic / scale logging, shared with the
        non-TK HyperConnectionModule via `get_hyperconnection_logging_tracker`.
        """
        from megatron.training import get_args
        from hcu_megatron.core.transformer.hyper_connection import (
            get_hyperconnection_logging_tracker,
        )
        if self.is_mtp or not torch.is_grad_enabled():
            return
        if (get_args().curr_iteration + 1) % self.log_amax_per_step != 0:
            return

        tracker = get_hyperconnection_logging_tracker()
        n_total = self.config.num_layers + (self.config.mtp_num_layers or 0)
        if 'hc_forward_amax' not in tracker:
            for key in (
                    'hc_forward_amax',
                    'hc_backward_amax',
                    'hc_eye_dist',
                    'hc_not_ds_ratio',
                    'hc_alpha_pre',
                    'hc_alpha_post',
                    'hc_alpha_res',
            ):
                tracker[key] = torch.zeros(n_total * 2, device=comb_mix.device)

        idx = (self.layer_number - 1) * 2 + int(self.hc_type == 'mlp')
        with torch.no_grad():
            h_res = comb_mix  # [s, b, n, n]
            col_sum = h_res.sum(-1)
            row_sum = h_res.sum(-2)
            tracker['hc_forward_amax'][idx] = torch.maximum(
                tracker['hc_forward_amax'][idx], row_sum.amax()
            )
            tracker['hc_backward_amax'][idx] = torch.maximum(
                tracker['hc_backward_amax'][idx], col_sum.amax()
            )
            eye = torch.eye(self.n, device=h_res.device, dtype=h_res.dtype)
            tracker['hc_eye_dist'][idx] = (h_res - eye).norm(dim=(-2, -1)).mean()

            ones = torch.ones(self.n, device=h_res.device, dtype=h_res.dtype)
            row_err = (row_sum - ones).abs().max(dim=-1).values
            col_err = (col_sum - ones).abs().max(dim=-1).values
            tracker['hc_not_ds_ratio'][idx] = (
                ((row_err > 1e-2) | (col_err > 1e-2)).float().mean()
            )

            scale = self.scale.data.float()
            tracker['hc_alpha_pre'][idx] = scale[0]
            tracker['hc_alpha_post'][idx] = scale[1]
            tracker['hc_alpha_res'][idx] = scale[2]

    def _forward_with_checkpoint(
            self, hidden_states: Tensor, manager: 'CheckpointManager'
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.

        compute_mappings is called directly (not checkpointed) since its outputs
        (h_pre, h_post, h_res) are needed downstream. Only aggregate is wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            manager: CheckpointManager for unified recomputation

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        if self.mhc_use_tilekernels:
            raise ValueError("mhc_use_tilekernels does not support the use of recompute.")
        else:
            h_pre, h_post, h_res = self.compute_mappings(hidden_states)

            # Checkpoint aggregate - auto-registers to manager
            aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self.aggregate, hidden_states, h_pre
            )

        return aggregated, h_res, h_post

    # ==================== Block-level utilities ====================

    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
        return expanded.view(s, b, n * C)

    @staticmethod
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.mean(dim=2)
        return contracted

    # ==================== Fused kernel placeholder ====================

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
            self,
            h_res: Tensor,
            original_residual: Tensor,
            h_post: Tensor,
            layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
            dropout_prob: float,
            training: bool,
            fused: bool,
            manager: Optional['CheckpointManager'] = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.

        The computation flow is:
            1. mixed = H_res.T @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states (before H_res applied)
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional CheckpointManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.

        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None:
            if self.mhc_use_tilekernels:
                raise ValueError("mhc_use_tilekernels does not support the use of recompute.")
            else:
                return self._fused_h_res_h_post_bda_with_checkpoint(
                    h_res,
                    original_residual,
                    h_post,
                    layer_output_with_bias,
                    dropout_prob,
                    training,
                    fused,
                    manager,
                )
        elif self.mhc_use_tilekernels:
            return self.apply_h_post_fuse(
                layer_output_with_bias,
                original_residual,
                h_post,
                h_res,
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
            )

    def _fused_h_res_h_post_bda_native(
            self,
            h_res: Tensor,
            original_residual: Tensor,
            h_post: Tensor,
            layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
            dropout_prob: float,
            training: bool,
            fused: bool,
    ) -> Tensor:
        """
        h_res, h_post and bda.

        When dropout is zero (or inference), uses a single fused/reference kernel
        for H_res @ residual + H_post * (x + bias). Falls back to unfused
        implementation when dropout is needed.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation

        Returns:
            output: [s, b, n*C] - final output
        """
        x, bias = layer_output_with_bias

        if dropout_prob == 0.0 or not training:
            s, b, _ = original_residual.shape
            n = self.n
            C = self.hidden_size
            orig_reshaped = original_residual.view(s, b, n, C)
            output = self._h_post_bda_op(h_res, orig_reshaped, h_post, x, bias)
            return output.view(s, b, n * C)

        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
            mixed = self.apply_h_res(h_res, original_residual)
        with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
            x_expanded = self._apply_h_post(x, h_post)
            bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None
        bda_func = get_bias_dropout_add(training, fused)
        with torch.cuda.nvtx.range("HyperConnection::bda"):
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
        return output

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_with_checkpoint")
    def _fused_h_res_h_post_bda_with_checkpoint(
            self,
            h_res: Tensor,
            original_residual: Tensor,
            h_post: Tensor,
            layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
            dropout_prob: float,
            training: bool,
            fused: bool,
            manager: 'CheckpointManager',
    ) -> Tensor:
        """
        Checkpointed variant of _fused_h_res_h_post_bda_native.

        Wraps compute in CheckpointWithoutOutput for activation memory savings.
        Cannot reuse _native directly because checkpoint requires all args to be
        positional Tensors; tuple/Optional/scalar args are unpacked or captured
        via closure instead.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: CheckpointManager for checkpoint management

        Returns:
            output: [s, b, n*C] - final output
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        x, bias = layer_output_with_bias
        n = self.n
        C = self.hidden_size

        # Fast path: no dropout — use fused/reference h_post_bda kernel (same as _native)
        if dropout_prob == 0.0 or not training:

            def _fused_wrapper(h_res, original_residual, h_post, x, *optional_bias):
                s, b, _ = original_residual.shape
                orig_reshaped = original_residual.view(s, b, n, C)
                b_arg = optional_bias[0] if optional_bias else None
                return self._h_post_bda_op(h_res, orig_reshaped, h_post, x, b_arg).view(s, b, n * C)

            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
            if bias is not None:
                output = ckpt.checkpoint(_fused_wrapper, h_res, original_residual, h_post, x, bias)
            else:
                output = ckpt.checkpoint(_fused_wrapper, h_res, original_residual, h_post, x)

        # Slow path: dropout required — fused kernel does not support dropout,
        # fall back to sequential apply_h_res + apply_h_post + bda
        else:
            from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

            bda_func = get_bias_dropout_add(training, fused)
            has_bias = bias is not None

            def _native_wrapper(h_res, original_residual, h_post, x, *optional_bias):
                with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
                    mixed = self.apply_h_res(h_res, original_residual)
                with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
                    x_expanded = self._apply_h_post(x, h_post)
                    if has_bias:
                        bias_expanded = self._apply_h_post(optional_bias[0], h_post)
                    else:
                        bias_expanded = None
                with torch.cuda.nvtx.range("HyperConnection::bda"):
                    output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
                return output

            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
            if has_bias:
                output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x, bias)
            else:
                output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x)

        return output

    def apply_h_post_fuse(
            self,
            x_with_bias: Tuple[Tensor, Optional[Tensor]],
            residual: Tensor,
            post_mix: Tensor,
            comb_mix: Tensor,
    ) -> Tensor:
        """Fused `H_res @ residual + H_post * x` via TileKernels' `mhc_post`."""
        from tile_kernels.modeling.mhc.ops import mhc_post

        x, bias = x_with_bias
        assert bias is None, "HyperConnectionTKModule requires bias=None"

        s, b, C = x.shape
        residual_tk = residual.view(s, b, self.n, C)
        nvtx_range_push("HyperConnectionTKModule.apply_h_post_fuse.mhc_post")
        out = mhc_post(x.contiguous(), residual_tk, post_mix, comb_mix)
        nvtx_range_pop("HyperConnectionTKModule.apply_h_post_fuse.mhc_post")
        return out.reshape(s, b, self.n * C)

    def apply_h_post_fuse_with_checkpoint(
            self,
            x_with_bias: Tuple[Tensor, Optional[Tensor]],
            residual: Tensor,
            post_mix: Tensor,
            comb_mix: Tensor,
            manager: Optional['CheckpointManager'] = None,
    ) -> Tensor:
        """
        Checkpointed variant with optional fallback to direct execution.

        Args:
            x_with_bias: Tuple of (x, bias) - bias must be None for fused path
            residual: [s, b, n*C] - n-stream hidden states
            post_mix: [s, b, n] - post-mixing weights (H_post)
            comb_mix: [s, b, n, n] - combination mixing matrix (H_res)
            manager: CheckpointManager for checkpoint management (required if checkpointing)
            force_checkpoint: If True, always use checkpoint; if False, use direct when manager is None
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        x, bias = x_with_bias
        assert bias is None, "HyperConnectionTKModule requires bias=None"

        def _compute(x, post_mix, comb_mix):
            from tile_kernels.modeling.mhc.ops import mhc_post
            from megatron.core.utils import nvtx_range_push, nvtx_range_pop

            s, b, C = x.shape
            n = self.n
            residual_tk = residual.view(s, b, n, C)

            nvtx_range_push("HyperConnectionTKModule.apply_h_post_fuse.mhc_post")
            out = mhc_post(x.contiguous(), residual_tk, post_mix, comb_mix)
            nvtx_range_pop("HyperConnectionTKModule.apply_h_post_fuse.mhc_post")

            return out.reshape(s, b, n * C)

        # 决定是否使用 checkpoint
        use_checkpoint = manager is not None

        if use_checkpoint:
            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)

            output = ckpt.checkpoint(_compute, x, post_mix, comb_mix)
        else:
            # 直接执行
            output = _compute(x, post_mix, comb_mix)

        return output


# ==================== Checkpoint utilities for mHC ====================


class HyperConnectionCheckpoint:
    """
    Checkpoint utility for mHC intermediate activations.

    Implements the paper's "recomputing strategy" to reduce memory footprint
    by discarding intermediate n-stream activations and recomputing on-the-fly.
    """

    @staticmethod
    def compute_optimal_block_size(num_layers: int, num_streams: int) -> int:
        """
        Compute optimal recomputation block size.

        From paper Eq. (20): L_r^* ≈ sqrt(nL/(n+2))

        Args:
            num_layers: Total number of transformer layers
            num_streams: Number of residual streams (n)

        Returns:
            block_size: Optimal block size for checkpointing
        """
        block_size = int(math.sqrt(num_streams * num_layers / (num_streams + 2)))
        return max(1, block_size)
