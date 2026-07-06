from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import (
    causal_conv1d_fn,
    chunk_gated_delta_rule,
    l2norm,
    torch_chunk_gated_delta_rule,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push


@jit_fuser
def _fused_g_and_beta(
    alpha: Tensor, beta: Tensor, A_log: Tensor, dt_bias: Tensor
) -> Tuple[Tensor, Tensor]:
    """Fuse GDN decay and beta transforms into one JIT-compilable elementwise region."""
    g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)
    beta = beta.sigmoid()
    return g, beta


@jit_fuser
def _expand_qk_heads(query: Tensor, key: Tensor, repeat_factor: int) -> Tuple[Tensor, Tensor]:
    """Expand Q/K heads in one JIT-compilable region when value heads outnumber key heads."""
    query = query.repeat_interleave(repeat_factor, dim=2)
    key = key.repeat_interleave(repeat_factor, dim=2)
    return query, key


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
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if packed_seq_params is not None:
            # TODO: support packed sequence
            raise NotImplementedError("GDN does not support packed sequence for now.")

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                self.v_dim // self.tp_size,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        # Convolution on qkv
        qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix="conv1d")
        if (causal_conv1d_fn is None) or self.config.deterministic_mode:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        nvtx_range_pop(suffix="conv1d")

        # Split qkv into query, key, and value
        qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.tp_size, self.qk_dim // self.tp_size, self.v_dim // self.tp_size],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        qk_head_repeat_factor = self.num_value_heads // self.num_key_heads
        if qk_head_repeat_factor > 1:
            query, key = _expand_qk_heads(query, key, qk_head_repeat_factor)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g, beta = _fused_g_and_beta(alpha, beta, self.A_log, self.dt_bias)
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        if self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

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
