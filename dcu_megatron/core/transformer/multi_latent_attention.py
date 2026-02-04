import torch

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.core.utils import deprecate_inference_params

from dcu_megatron.core.pipeline_parallel import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from .utils import get_delay_release_qkv_linear_tensor


class MultiLatentAttention():
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        micro_sp_idx=None,  # pylint: disable=unused-argument
    ):
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        if self.offload_qkv_linear:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="qkv_linear")
        with get_fine_grained_offloading_context(self.offload_qkv_linear):
            query, key, value = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        if self.offload_qkv_linear:
            delay_release_module = "qkv_linear" if get_delay_release_qkv_linear_tensor() else None
            query, key, value = fine_grained_offloading_group_commit(
                query, key, value,
                name="qkv_linear",
                forced_released_tensors=[hidden_states],
                delay_release_module=delay_release_module
            )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                if self.offload_core_attention and self.training:
                    query = fine_grained_offloading_group_start(query, name="core_attn")

                with get_fine_grained_offloading_context(self.offload_core_attention):
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                    )

                if self.offload_core_attention and self.training:
                    (core_attn_out,) = fine_grained_offloading_group_commit(
                        core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                    )
            elif self.cache_mla_latents:
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
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias


class MLASelfAttention():
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def compute_qkv(
        self,
        hidden_states,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,  # pylint: disable=unused-arguments
        rotary_pos_cos=None,  # pylint: disable=unused-arguments
        rotary_pos_sin=None,  # pylint: disable=unused-arguments
        packed_seq_params=None,
        position_ids=None,
        *,
        inference_params=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        if self.offload_qkv_linear:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="qkv_linear")
        with get_fine_grained_offloading_context(self.offload_qkv_linear):
            query, key, value = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        if self.offload_qkv_linear:
            delay_release_module = "qkv_linear" if get_delay_release_qkv_linear_tensor() else None
            query, key, value = fine_grained_offloading_group_commit(
                query, key, value,
                name="qkv_linear",
                forced_released_tensors=[hidden_states],
                delay_release_module=delay_release_module
            )

        return query, key, value

    def compute_attn(
        self,
        qkv_output,
        attention_mask,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        query, key, value = qkv_output

        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                if self.offload_core_attention and self.training:
                    query = fine_grained_offloading_group_start(query, name="core_attn")

                with get_fine_grained_offloading_context(self.offload_core_attention):
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                    )

                if self.offload_core_attention and self.training:
                    (core_attn_out,) = fine_grained_offloading_group_commit(
                        core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                    )
            elif self.cache_mla_latents:
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
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        return core_attn_out

    def compute_proj(self, core_attn_out):
        # =================
        # Output. [sq, b, h]
        # =================
        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )
        return output, bias
