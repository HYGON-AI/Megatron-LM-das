# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Optional, Tuple, Union
from functools import wraps

import torch
from torch import Tensor
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    deprecate_inference_params,
    is_fa_min_version,
    is_using_quantization_scales,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

from megatron.core.transformer.attention import HAVE_FA3, HAVE_FA4

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

try:
    import megatron.core.models.common.embeddings.yarn_rotary_pos_embedding as _yarn_mod
except ImportError:
    pass

from hcu_megatron.training.arguments import get_adaptor_args


def _yarn_get_concentration_factor_from_config(config):
    yarn_scaling = getattr(config, "yarn_rotary_scaling_factor", None)
    if yarn_scaling is not None:
        return _yarn_mod._yarn_get_concentration_factor(
            yarn_scaling,
            getattr(config, "yarn_mscale", None),
            getattr(config, "yarn_mscale_all_dim", None),
        )
    return 1.0


def attention_init_wrapper(attention_init_func):
    @wraps(attention_init_func)
    def wrapper(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        attention_type,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        attention_init_func(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        if get_adaptor_args().pipe_sp_splits != 1:
            self.core_attention_flash = FlashSeqSelfAttention(
                causal=True, softmax_scale=self.config.softmax_scale, attention_dropout=self.config.attention_dropout
            )

    return wrapper


def kv_sp_flash_func(q, k, v, kv_cache, softmax_scale, causal):
    out = FlashAttnVarlenFunc.apply(q, k, v, kv_cache, 0.0, causal, softmax_scale)
    return out


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            kv_cache,
            dropout_p,
            causal,
            softmax_scale,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        batch_size = q.shape[0]
        if 'k_cache' in kv_cache:
            k_cache, v_cache = kv_cache['k_cache'], kv_cache['v_cache']
            offset = k_cache.shape[1]
            k_whole = torch.cat([k_cache, k], dim=1).contiguous()
            v_whole = torch.cat([v_cache, v], dim=1).contiguous()
        else:
            offset = 0
            k_whole = k
            v_whole = v
        kv_cache['k_cache'], kv_cache['v_cache'] = k_whole, v_whole
        seqlen_k = k_whole.shape[1]
        seqlen_q = q.shape[1]
        ctx._seqlen_k = seqlen_k
        ctx._seqlen_q = seqlen_q
        ctx._offset = offset
        q, k_whole, v_whole = [rearrange(x, 'b s ... -> (b s) ...').contiguous() for x in [q, k_whole, v_whole]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device="cuda")

        q = q.contiguous()
        k_whole = k_whole.contiguous()
        v_whole = v_whole.contiguous()
        out, q, k_whole, v_whole, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k_whole,
            v_whole,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale,
            block_table=None,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_softmax=False,
            softcap=0,
        )
        ctx._kv_cache = kv_cache
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state,

        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        k_whole = ctx._kv_cache['k_cache']
        v_whole = ctx._kv_cache['v_cache']
        batch_size = q.size(0) // ctx._seqlen_q
        pk = k_whole[:, :ctx._seqlen_k]
        pv = v_whole[:, :ctx._seqlen_k].contiguous()
        ctx._kv_cache['k_cache'], ctx._kv_cache['v_cache'] = pk[:, :-ctx._seqlen_q], pv[:, :-ctx._seqlen_q]
        pk, pv = [rearrange(x, 'b s ... -> (b s) ...') for x in [pk, pv]]
        pk = pk.contiguous()
        pv = pv.contiguous()
        q = q.contiguous()

        last_idx = not 'k_grad' in ctx._kv_cache
        dq, dk, dv = torch.empty_like(q), torch.empty_like(pk), torch.empty_like(pv)
        _flash_attn_varlen_backward(
            dout,
            q,
            pk,
            pv,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx._seqlen_q,
            ctx._seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            rng_state=rng_state,
            softcap=0,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        dq, dk, dv = [rearrange(x, '(b s) ... -> b s ...', b=batch_size) for x in [dq, dk, dv]]
        dk = dk.contiguous()
        dv = dv.contiguous()
        dq = dq.contiguous()
        if not last_idx:
            k_grad_p, v_grad_p = ctx._kv_cache['k_grad'], ctx._kv_cache['v_grad']
            dk += k_grad_p
            dv += v_grad_p
        ctx._kv_cache['k_grad'], ctx._kv_cache['v_grad'] = dk[:, :ctx._offset], dv[:, :ctx._offset]
        dk = dk[:, ctx._offset:ctx._seqlen_k]
        dv = dv[:, ctx._offset:ctx._seqlen_k]

        return dq, dk, dv, None, None, None, None, None, None, None


class FlashSeqSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()

        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.kv_cache = {}

    def forward(self, q, k, v, micro_sp_idx):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))
        args = get_adaptor_args()
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        if torch.is_tensor(micro_sp_idx):
            micro_sp_idx = micro_sp_idx.item()
        # micro_batch_id = args.schedule_info['micro_seq_id'] // args.pipe_sp_splits
        if args.pipe_sp_splits != 1:
            if micro_sp_idx == 0:
                self.kv_cache = {}
            output = kv_sp_flash_func(q, k, v, self.kv_cache, self.softmax_scale, True)
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            return output.contiguous()


class Attention():
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        micro_sp_idx=None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert (
                HAVE_FA4 or HAVE_FA3 or is_fa_min_version("2.7.3")
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        is_inference_mode = InferenceMode.is_active()
        # is_using_flash_decode - True is we are using the static inference engine with flash decode
        is_using_flash_decode = is_inference_mode and self.config.flash_decode
        # is_using_flashinfer_rope - True if we are using the dynamic inference engine
        # with flashinfer fused rope
        is_using_flashinfer_rope = (
            is_inference_mode
            and inference_context is not None
            and not inference_context.is_static_batching()
            and inference_context.use_flashinfer_fused_rope
        )
        if is_using_flash_decode or is_using_flashinfer_rope:
            # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        split_qkv = (self.attention_type == "cross") or not all(
            [
                not self.config.test_mode,
                self.config.fused_single_qkv_rope,
                inference_context is None,
                packed_seq_params is None,
                (
                    rotary_pos_emb is not None
                    and rotary_pos_emb[0] is not None
                    and rotary_pos_emb[1] is not None
                ),
                not self.config.flash_decode,
                HAVE_FUSED_QKV_ROPE,
                self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
                self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
            ]
        )
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        if self.attention_type != "cross":
            assert not (
                self.config.fused_single_qkv_rope and split_qkv
            ), "fused_single_qkv_rope requested but not available/supported for the config."

        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            qkv_output = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                split_qkv=split_qkv,
                output_gate=self.config.attention_output_gate,
            )
        if self.offload_qkv_linear:
            # `qkv_output` may be a tuple; commit supports tuple/list and will keep structure.
            qkv_output = off_interface.group_commit(
                qkv_output, name="qkv_linear", forced_released_tensors=[]
            )
        attn_mask_type = self.attn_mask_type
        block_table = None
        gate = None
        if split_qkv:
            if self.config.attention_output_gate:
                query, key, value, gate = qkv_output
            else:
                query, key, value = qkv_output
            mixed_qkv = qkv_split_arg_list = None
        else:
            assert (
                not self.config.attention_output_gate
            ), "attention_output_gate is not supported for unsplit mixed_qkv tensor."
            mixed_qkv, qkv_split_arg_list = qkv_output
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and InferenceMode.is_active()
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = apply_module(self.linear_proj)(context_layer)
            return output, bias

        if (
            in_decode_mode
            and self.config.cuda_graph_impl == "local"
            and inference_context.is_static_batching()
        ):
            raise ValueError(f"CUDA graphs must use flash decode with static batching!")

        if split_qkv:
            query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
                self._adjust_key_value_for_inference(
                    inference_context,
                    query,
                    key,
                    value,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    rotary_pos_cos_sin,
                    sequence_len_offset,
                )
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        args = get_adaptor_args()
        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (
            not self.config.flash_decode or inference_context is None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            from hcu_megatron.core.pipeline_parallel.seq1f1b.sp_utils import get_splits
            if args.pipe_sp_splits != 1:
                if args.pipe_sp_strategy == "uniform_comp":
                    splits = get_splits()
                    start = sum(splits[:micro_sp_idx])
                    end = sum(splits[:micro_sp_idx + 1])
                elif args.pipe_sp_strategy == "average":
                    start = (micro_sp_idx * q_pos_emb.size(0)) // args.pipe_sp_splits
                    end = ((micro_sp_idx + 1) * q_pos_emb.size(0)) // args.pipe_sp_splits
                q_pos_emb = q_pos_emb[start:end]
                k_pos_emb = k_pos_emb[start:end]

            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if split_qkv:
                if q_pos_emb is not None:
                    # TODO VIJAY: simplify
                    if inference_context is None or inference_context.is_static_batching():
                        query = apply_rotary_pos_emb(
                            query,
                            q_pos_emb,
                            config=self.config,
                            cu_seqlens=cu_seqlens_q,
                            mscale=_yarn_get_concentration_factor_from_config(self.config),
                            cp_group=self.pg_collection.cp,
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                        )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
            else:
                query, key, value = apply_fused_qkv_rotary_pos_emb(
                    mixed_qkv, q_pos_emb, k_pos_emb, qkv_split_arg_list
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                if args.pipe_sp_splits != 1:
                    q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                               for x in (query, key, value)]
                    core_attn_out = self.core_attention_flash(q, k, v, micro_sp_idx)
                    core_attn_out = rearrange(core_attn_out, 'b s h d -> s b (h d)').contiguous()
                else:
                    # Static batching attention kernel.
                    with off_interface(
                        self.offload_core_attention and self.training, query, "core_attn"
                    ) as query:
                        core_attn_out = apply_module(self.core_attention)(
                            query,
                            key,
                            value,
                            attention_mask,
                            attn_mask_type=attn_mask_type,
                            attention_bias=attention_bias,
                            packed_seq_params=packed_seq_params,
                        )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                    inference_context.is_decode_only(),
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

                # Clear the outputs for padding tokens when using quantization scales
                # to avoid corrupting amax calculations
                if is_using_quantization_scales(self.config):
                    core_attn_out[inference_context.padding_slice] = 0.0

            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # Output gate
        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._apply_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="linear_proj")
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(
                output, name="attn_proj", forced_released_tensors=[core_attn_out]
            )
        nvtx_range_pop(suffix="linear_proj")

        return output, bias

    def compute_qkv(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context=None,  # pylint: disable=unused-arguments
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params=None,  # pylint: disable=unused-arguments
        position_ids=None,       # pylint: disable=unused-arguments
        *,
        inference_params=None,   # pylint: disable=unused-arguments
    ):
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert (
                HAVE_FA4 or HAVE_FA3 or is_fa_min_version("2.7.3")
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        is_inference_mode = InferenceMode.is_active()
        # is_using_flash_decode - True is we are using the static inference engine with flash decode
        is_using_flash_decode = is_inference_mode and self.config.flash_decode
        # is_using_flashinfer_rope - True if we are using the dynamic inference engine
        # with flashinfer fused rope
        is_using_flashinfer_rope = (
            is_inference_mode
            and inference_context is not None
            and not inference_context.is_static_batching()
            and inference_context.use_flashinfer_fused_rope
        )
        if is_using_flash_decode or is_using_flashinfer_rope:
            # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        self.split_qkv = (self.attention_type == "cross") or not all(
            [
                not self.config.test_mode,
                self.config.fused_single_qkv_rope,
                inference_context is None,
                packed_seq_params is None,
                (
                    rotary_pos_emb is not None
                    and rotary_pos_emb[0] is not None
                    and rotary_pos_emb[1] is not None
                ),
                not self.config.flash_decode,
                HAVE_FUSED_QKV_ROPE,
                self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
                self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
            ]
        )
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        if self.attention_type != "cross":
            assert not (
                self.config.fused_single_qkv_rope and self.split_qkv
            ), "fused_single_qkv_rope requested but not available/supported for the config."

        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            qkv_output = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                split_qkv=self.split_qkv,
                output_gate=self.config.attention_output_gate,
            )
        if self.offload_qkv_linear:
            # `qkv_output` may be a tuple; commit supports tuple/list and will keep structure.
            qkv_output = off_interface.group_commit(
                qkv_output, name="qkv_linear", forced_released_tensors=[]
            )
        nvtx_range_pop(suffix="qkv")

        return qkv_output

    def compute_attn(
        self,
        qkv_output,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        gate = None
        if self.split_qkv:
            if self.config.attention_output_gate:
                query, key, value, gate = qkv_output
            else:
                query, key, value = qkv_output
            mixed_qkv = qkv_split_arg_list = None
        else:
            assert (
                not self.config.attention_output_gate
            ), "attention_output_gate is not supported for unsplit mixed_qkv tensor."
            mixed_qkv, qkv_split_arg_list = qkv_output

        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert (
                HAVE_FA4 or HAVE_FA3 or is_fa_min_version("2.7.3")
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and InferenceMode.is_active()
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = apply_module(self.linear_proj)(context_layer)
            return output, bias

        if (
            in_decode_mode
            and self.config.cuda_graph_impl == "local"
            and inference_context.is_static_batching()
        ):
            raise ValueError(f"CUDA graphs must use flash decode with static batching!")

        attn_mask_type = self.attn_mask_type
        block_table = None
        if self.split_qkv:
            query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
                self._adjust_key_value_for_inference(
                    inference_context,
                    query,
                    key,
                    value,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    rotary_pos_cos_sin,
                    sequence_len_offset,
                )
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (
            not self.config.flash_decode or inference_context is None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if self.split_qkv:
                if q_pos_emb is not None:
                    # TODO VIJAY: simplify
                    if inference_context is None or inference_context.is_static_batching():
                        query = apply_rotary_pos_emb(
                            query,
                            q_pos_emb,
                            config=self.config,
                            cu_seqlens=cu_seqlens_q,
                            mscale=_yarn_get_concentration_factor_from_config(self.config),
                            cp_group=self.pg_collection.cp,
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                        )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
            else:
                query, key, value = apply_fused_qkv_rotary_pos_emb(
                    mixed_qkv, q_pos_emb, k_pos_emb, qkv_split_arg_list
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                with off_interface(
                    self.offload_core_attention and self.training, query, "core_attn"
                ) as query:
                    core_attn_out = apply_module(self.core_attention)(
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                    inference_context.is_decode_only(),
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

                # Clear the outputs for padding tokens when using quantization scales
                # to avoid corrupting amax calculations
                if is_using_quantization_scales(self.config):
                    core_attn_out[inference_context.padding_slice] = 0.0

            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # Output gate
        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._apply_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        return core_attn_out

    def compute_proj(self, core_attn_out):
        # =================
        # Output. [sq, b, h]
        # =================

        nvtx_range_push(suffix="linear_proj")
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(
                output, name="attn_proj", forced_released_tensors=[core_attn_out]
            )
        nvtx_range_pop(suffix="linear_proj")

        return output, bias
