import contextlib

from megatron.core.utils import deprecate_inference_params

from dcu_megatron.core.transformer import (
    PipelineOffloadManager,
    group_prefetch_offload_start,
    group_prefetch_offload_commit,
)


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
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
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
            offload_context = contextlib.nullcontext()
            if self.offload_core_attention and self.training:
                query = group_prefetch_offload_start(query, name="core_attn")
                offload_context = PipelineOffloadManager.get_instance()

            with offload_context:
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                )

            if self.offload_core_attention and self.training:
                core_attn_out, = group_prefetch_offload_commit(core_attn_out, name="core_attn", release_tensors=[query, key, value])
                offload_context = contextlib.nullcontext()

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
        offload_context = contextlib.nullcontext()
        if self.offload_attn_proj:
            core_attn_out = group_prefetch_offload_start(core_attn_out, name="attn_proj")
            offload_context = PipelineOffloadManager.get_instance()
        with offload_context:
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = group_prefetch_offload_commit(output, bias, name="attn_proj", release_tensors=[core_attn_out])
            offload_context = contextlib.nullcontext()

        return output, bias


class MLASelfAttention():
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def backward_dw(self):
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()
        self.linear_proj.backward_dw()

    def compute_qkv(
        self,
        hidden_states,
        key_value_states=None,
        inference_context=None,
        packed_seq_params=None,
        position_ids=None,
        *,
        inference_params=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        return query, key, value

    def compute_attn(
        self,
        query,
        key,
        value,
        attention_mask,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):

        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
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
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )

        if packed_seq_params is not None:
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

        output, bias = self.linear_proj(core_attn_out)
        return output, bias
