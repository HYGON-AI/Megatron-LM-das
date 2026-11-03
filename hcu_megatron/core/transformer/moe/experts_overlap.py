# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/baidu-baige/Loong-Megatron
"""
Ported from:
  /public/home/chenyashu/code/LoongForge/LoongForge-master/third_party/
  Loong-Megatron/megatron/core/transformer/moe/experts.py:1000-1174
  (TEGroupedMLP.forward_with_dispatch_overlap method)

Modifications from Loong:
  - Turned from bound method into module-level function so we can patch it
    into upstream TEGroupedMLP without modifying the class (matches how
    hcu_megatron's patch mechanism works).
  - Uses `_grouped_mlp_get_expert_weights` / `_grouped_mlp_set_expert_weights`
    helpers from hcu_megatron.core.transformer.moe.experts (also ports of
    Loong's TEGroupedMLP methods).
  - Uses `experts._apply_activation(...)` via getattr to reuse the private
    activation helper defined in upstream MegatronCoreTEGroupedMLP.

Time-space schedule:
  ─────────────────────────────────────────────────────────────────────────
  Default stream (home compute path):
    [get home fc1 w] → [fc1_preprocess] → [home fc1 GEMM]
                                             ↓ activation
                                          [wait fc1_event] → [set echo fc1 w]
                                          [echo fc1 GEMM] → [act]
    [get home fc2 w] → [fc2_preprocess] → [home fc2 GEMM]
                                             ↓
                                          [wait fc2_event] → [set echo fc2 w]
                                          [echo fc2 GEMM]
    → [cat home_out + echo_out]

  moe_a2a_stream (dispatch path):
    [wait default] → [fc1 A2A] → [record fc1_event]
    [wait default] → [fc2 A2A] → [record fc2_event]
  ─────────────────────────────────────────────────────────────────────────

The overlap is real: fc1 A2A runs on the side stream while home fc1 GEMM
runs on the default stream. Same for fc2.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from contextlib import nullcontext
import torch.distributed as dist
from torch.autograd.function import once_differentiable

class _LayerBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *values):
        ctx.set_materialize_grads(False)
        return tuple(values)

    @staticmethod
    def backward(ctx, *gradients):
        # One node joins the token and home-weight gradient paths. Do not
        # release the preceding layer while this layer's weight A2A is pending.
        return tuple(gradients)

def _inactive_home_weight(weight, fused):
    # TE versions that inspect weights[0].requires_grad need a trainable first
    # slot even when its m_split is zero. Never let this GEMM overwrite the real
    # home main_grad (especially on the first microbatch).
    placeholder = Parameter(weight.detach(), requires_grad=True)
    if fused:
        placeholder.main_grad = torch.zeros_like(weight.main_grad)
        placeholder.grad_added_to_main_grad = False
        placeholder.zero_out_wgrad = True
    return placeholder

def _exchange(output, input_, group, output_splits, input_splits):
    work = dist.all_to_all_single(
        output, input_.contiguous(), output_split_sizes=output_splits,
        input_split_sizes=input_splits, group=group, async_op=True,
    )
    work.wait()


class _WeightExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, group, send_splits, recv_splits, stream, ticket):
        # apply() must be called OUTSIDE the side-stream context. Its backward
        # is then scheduled on the same stream as the token backward.
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        output = packed.new_empty((sum(recv_splits), packed.shape[1]))
        if packed.is_cuda:
            stream.wait_stream(torch.cuda.current_stream(packed.device))
            packed.record_stream(stream)
            output.record_stream(stream)
        with torch.cuda.stream(stream) if packed.is_cuda else nullcontext():
            _exchange(output, packed, group, recv_splits, send_splits)
            if packed.is_cuda:
                ticket['done'] = torch.cuda.Event()
                ticket['done'].record(stream)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        result = grad.new_empty((sum(ctx.send_splits), grad.shape[1]))
        _exchange(result, grad, ctx.group, ctx.send_splits, ctx.recv_splits)
        return result, None, None, None, None, None


class _AfterEarlierWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, later_input, earlier_result):
        ctx.set_materialize_grads(False)
        return later_input

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None

def _start_weight_dispatch(dispatcher, metadata, preprocessed, stream, earlier=None):
    start = getattr(dispatcher, 'expert_dispatch_overlap_start', None)
    if start is not None:
        return start(metadata, preprocessed, stream, earlier)
    packed = preprocessed['permuted_weights']
    if earlier is not None:
        packed = _AfterEarlierWeight.apply(packed, earlier)
    ticket = {}
    received = _WeightExchange.apply(
        packed, dispatcher.ep_group, metadata.input_splits,
        metadata.output_splits, stream, ticket,
    )
    return received, ticket

def _finish_weight_dispatch(dispatcher, metadata, preprocessed, received, ticket):
    finish = getattr(dispatcher, 'expert_dispatch_overlap_finish', None)
    if finish is not None:
        return finish(metadata, preprocessed, received, ticket)
    from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
    if 'done' in ticket:
        torch.cuda.current_stream(received.device).wait_event(ticket['done'])
    received, _ = sort_chunks_by_idxs(
        received, metadata.input_chunks_per_rank,
        dispatcher.sort_input_by_local_experts, fused=False,
    )
    return [w.reshape(preprocessed['weight_shape']) if w.numel() else w
            for w in torch.split(received, metadata.has_experts_per_slot, dim=0)]




def _apply_activation(experts, fc1_output, bias_parallel, permuted_probs):
    """Apply activation function (and optional bias) to FC1 output.

    Ported from Loong TEGroupedMLP._apply_activation (Ref/experts.py:932-998).
    Standalone function because upstream 0.18.2 TEGroupedMLP doesn't have this method.
    """
    from megatron.core.activations import squared_relu
    from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
    from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
    from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl

    config = experts.config
    activation_func = experts.activation_func

    if getattr(config, 'use_te_activation_func', False):
        if bias_parallel is not None:
            fc1_output = fc1_output + bias_parallel
        fc1_output = activation_func(fc1_output)
        if permuted_probs is not None:
            original_dtype = fc1_output.dtype
            fc1_output = (fc1_output * permuted_probs).to(original_dtype)
    elif config.bias_activation_fusion:
        if activation_func == F.silu and config.gated_linear_unit:
            fc1_output = weighted_bias_swiglu_impl(
                fc1_output, bias_parallel, permuted_probs,
                config.activation_func_fp8_input_store,
            )
        elif activation_func == quick_gelu and config.gated_linear_unit:
            fc1_output = weighted_bias_quick_geglu_impl(
                fc1_output, bias_parallel, permuted_probs,
                config.activation_func_fp8_input_store,
                config.glu_linear_offset,
                config.activation_func_clamp_value,
            )
        else:
            raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
    elif activation_func == squared_relu and getattr(config, 'use_fused_weighted_squared_relu', False):
        assert bias_parallel is None
        fc1_output = weighted_squared_relu_impl(fc1_output, permuted_probs)
    else:
        if config.gated_linear_unit:
            def glu(x):
                x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                val = getattr(config, 'activation_func_clamp_value', None)
                if val is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                return config.activation_func(x_glu) * (
                    x_linear + getattr(config, 'glu_linear_offset', 0)
                )
            fc1_output = glu(fc1_output)
        else:
            fc1_output = activation_func(fc1_output)
        original_dtype = fc1_output.dtype
        fc1_output = (fc1_output * permuted_probs).to(original_dtype)
    return fc1_output


def _grouped_mlp_forward_with_dispatch_overlap(
    experts,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    num_home_experts: int,
    home_expert_indices: List[int],
    echo_expert_indices: List[int],
    expert_dispatcher,
    fc1_dispatch_metadata,
    fc2_dispatch_metadata,
    a2a_stream: "torch.cuda.Stream",
    fc1_event: "torch.cuda.Event",
    fc2_event: "torch.cuda.Event",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Home GEMM overlap; backward orders FC2 weights -> FC1 weights -> tokens.

    fc1_event/fc2_event remain accepted for caller compatibility.
    Completion events are owned by the exchange tickets instead.
    """
    from hcu_megatron.core.transformer.moe.experts import (
        _grouped_mlp_get_expert_weights,
        _grouped_mlp_set_expert_weights,
        echo_dispatch_home_weights,
    )

    # ── Sanity checks ──────────────────────────────────────────────────────
    if getattr(experts, 'offload_expert_fc1', False) or getattr(experts, 'offload_moe_act', False):
        raise RuntimeError(
            "forward_with_dispatch_overlap does not support fine-grained activation offloading."
        )
    if getattr(experts, 'activation_recompute', False):
        raise RuntimeError(
            "forward_with_dispatch_overlap does not support moe_act activation recompute."
        )

    fused = experts.config.gradient_accumulation_fusion
    if not echo_expert_indices:
        raise RuntimeError('Echo overlap requires at least one local echo slot')

    # Join both weight-gradient paths before releasing token gradients to the
    # upstream dispatcher. Keep original home Parameters installed in the module.
    dispatch_fc1_weights = _grouped_mlp_get_expert_weights(experts, 'fc1', home_expert_indices)
    dispatch_fc2_weights = _grouped_mlp_get_expert_weights(experts, 'fc2', home_expert_indices)

    dispatch_fc1_weights = echo_dispatch_home_weights(dispatch_fc1_weights, fused)
    dispatch_fc2_weights = echo_dispatch_home_weights(dispatch_fc2_weights, fused)
    boundary = _LayerBoundary.apply(
        permuted_local_hidden_states, permuted_probs,
        *dispatch_fc1_weights, *dispatch_fc2_weights,
    )
    permuted_local_hidden_states, permuted_probs = boundary[:2]
    n_home = len(home_expert_indices)
    dispatch_fc1_weights = boundary[2:2 + n_home]
    dispatch_fc2_weights = boundary[2 + n_home:]

    # ── Preparation ────────────────────────────────────────────────────────
    tpe_list = tokens_per_expert.long().cpu().tolist()
    num_echo_experts = len(echo_expert_indices)
    received_num_tokens = permuted_local_hidden_states.shape[0]
    permuted_local_hidden_states = permuted_local_hidden_states.contiguous()

    # moe_received_token_capacity is Loong-specific; upstream 0.18.2 lacks it.
    _received_capacity = getattr(experts.config, 'moe_received_token_capacity', None)
    if _received_capacity is not None:
        actual_total = sum(tpe_list)
        permuted_local_hidden_states = permuted_local_hidden_states[:actual_total]
        permuted_probs = permuted_probs[:actual_total]

    # Split tokens / probs into home and echo halves (zero-copy views).
    home_token_count = sum(tpe_list[:num_home_experts])
    home_tokens = permuted_local_hidden_states[:home_token_count]
    echo_tokens = permuted_local_hidden_states[home_token_count:]
    home_tpe = tpe_list[:num_home_experts]
    echo_tpe = tpe_list[num_home_experts:]
    home_probs = permuted_probs[:home_token_count].unsqueeze(-1)
    echo_probs = permuted_probs[home_token_count:].unsqueeze(-1)

    if getattr(experts.config, 'moe_apply_probs_on_input', False):
        assert experts.config.moe_router_topk == 1, \
            "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
        home_tokens = (home_probs * home_tokens).to(home_tokens.dtype)
        home_probs = torch.ones_like(home_probs)
        if echo_tokens.numel() > 0:
            echo_tokens = (echo_probs * echo_tokens).to(echo_tokens.dtype)
            echo_probs = torch.ones_like(echo_probs)

    # TE GroupedLinear requires exactly num_local_experts entries in m_splits.
    # Home-only call: real counts for home slots, 0 for echo slots (kernel skips m=0).
    # Echo-only call: 0 for home slots, real counts for echo slots.
    home_tpe_padded = home_tpe + [0] * num_echo_experts
    echo_tpe_padded = [0] * num_home_experts + echo_tpe

    # Lazy-init placeholder tensors for echo weight slots (once per module lifetime).
    if echo_expert_indices and not hasattr(experts, '_echo_fc1_placeholders'):
        # Compute weight shape from home slot weight0.
        ref_w1 = getattr(experts.linear_fc1, 'weight0')
        fc1_shape = ref_w1.shape
        experts._echo_fc1_placeholders = [
            torch.empty(fc1_shape, device=ref_w1.device, dtype=ref_w1.dtype)
            for _ in range(num_echo_experts)
        ]
        ref_w2 = getattr(experts.linear_fc2, 'weight0')
        fc2_shape = ref_w2.shape
        experts._echo_fc2_placeholders = [
            torch.empty(fc2_shape, device=ref_w2.device, dtype=ref_w2.dtype)
            for _ in range(num_echo_experts)
        ]

    # ══ FC1: dispatch on stream A  ∥  home GEMM on default stream ══════════
    if echo_expert_indices:
        _grouped_mlp_set_expert_weights(
            experts, "fc1", experts._echo_fc1_placeholders, echo_expert_indices
        )
    home_fc1_weights = _grouped_mlp_get_expert_weights(experts, "fc1", home_expert_indices)

    # Preprocess on default stream — avoids HBM contention with dispatch kernel.
    fc1_preprocessed = expert_dispatcher.expert_dispatch_preprocess(
        fc1_dispatch_metadata, *dispatch_fc1_weights
    )

    # Launch FC1 A2A on stream A.
    received_fc1_weights, fc1_ticket = _start_weight_dispatch(
        expert_dispatcher, fc1_dispatch_metadata, fc1_preprocessed, a2a_stream,
    )

    # Home FC1 GEMM on default stream (overlaps with A2A above).
    home_fc1_out, home_bias1 = experts.linear_fc1(home_tokens, home_tpe_padded)
    home_act = _apply_activation(experts, home_fc1_out, home_bias1, home_probs)

    # Echo FC1: wait for A2A, install dispatched weights, run GEMM.
    if echo_expert_indices:
        dispatched_fc1_weights = _finish_weight_dispatch(
            expert_dispatcher, fc1_dispatch_metadata, fc1_preprocessed,
            received_fc1_weights, fc1_ticket,
        )
        _grouped_mlp_set_expert_weights(
            experts, "fc1", list(dispatched_fc1_weights), echo_expert_indices
        )
        # Replace home slots with no-grad Parameters to skip wgrad alloc in echo backward.
        _grouped_mlp_set_expert_weights(
            experts, "fc1",
            [_inactive_home_weight(w, fused) for w in home_fc1_weights],
            home_expert_indices,
        )
        echo_fc1_out, echo_bias1 = experts.linear_fc1(echo_tokens, echo_tpe_padded)
        # Restore original Parameters so the optimizer can update them.
        _grouped_mlp_set_expert_weights(
            experts, "fc1", home_fc1_weights, home_expert_indices
        )
        echo_act = _apply_activation(experts, echo_fc1_out, echo_bias1, echo_probs)

    # ══ FC2: dispatch on stream A  ∥  home GEMM on default stream ══════════
    if echo_expert_indices:
        _grouped_mlp_set_expert_weights(
            experts, "fc2", experts._echo_fc2_placeholders, echo_expert_indices
        )
    home_fc2_weights = _grouped_mlp_get_expert_weights(experts, "fc2", home_expert_indices)

    fc2_preprocessed = expert_dispatcher.expert_dispatch_preprocess(
        fc2_dispatch_metadata, *dispatch_fc2_weights
    )

    # Reverse dependency: FC2 weight gradient exchange precedes FC1.
    received_fc2_weights, fc2_ticket = _start_weight_dispatch(
        expert_dispatcher, fc2_dispatch_metadata, fc2_preprocessed,
        a2a_stream, received_fc1_weights,
    )

    home_fc2_out, home_output_bias = experts.linear_fc2(home_act, home_tpe_padded)

    # Echo FC2: wait for A2A, install dispatched weights, run GEMM.
    if echo_expert_indices:
        dispatched_fc2_weights = _finish_weight_dispatch(
            expert_dispatcher, fc2_dispatch_metadata, fc2_preprocessed,
            received_fc2_weights, fc2_ticket,
        )
        _grouped_mlp_set_expert_weights(
            experts, "fc2", list(dispatched_fc2_weights), echo_expert_indices
        )
        _grouped_mlp_set_expert_weights(
            experts, "fc2",
            [_inactive_home_weight(w, fused) for w in home_fc2_weights],
            home_expert_indices,
        )
        echo_fc2_out, echo_output_bias = experts.linear_fc2(echo_act, echo_tpe_padded)
        _grouped_mlp_set_expert_weights(
            experts, "fc2", home_fc2_weights, home_expert_indices
        )

    # Concatenate home + echo outputs.
    if echo_expert_indices:
        if home_output_bias is not None or echo_output_bias is not None:
            raise RuntimeError(
                "forward_with_dispatch_overlap does not support add_bias_linear=True."
            )
        output = torch.cat([home_fc2_out, echo_fc2_out], dim=0)
    else:
        if home_output_bias is not None:
            raise RuntimeError(
                "forward_with_dispatch_overlap does not support add_bias_linear=True."
            )
        output = home_fc2_out

    # Pad to received_num_tokens if capacity is set.
    if _received_capacity is not None:
        pad_len = received_num_tokens - output.shape[0]
        if pad_len > 0:
            output = torch.cat(
                [output, torch.zeros(pad_len, output.shape[1],
                                     dtype=output.dtype, device=output.device)],
                dim=0,
            )

    output_bias = None
    return output, output_bias
