import torch
import torch.nn.functional as F
import transformer_engine
from contextlib import nullcontext
from megatron.core import mpu, tensor_parallel
from functools import wraps
from megatron import core
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType
from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.utils import deprecate_inference_params
from megatron.legacy.model.enums import AttnType
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.legacy.model.module import MegatronModule
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.legacy.model.transformer import bias_dropout_add_fused_inference, bias_dropout_add_fused_train, get_bias_dropout_add
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None
try:
    from einops import rearrange
except ImportError:
    rearrange = None

try: # 使用定长fa
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def parallel_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        args = get_args()
        if args.swiglu:
            @torch.compile(mode="max-autotune-no-cudagraphs")
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu

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
        # from megatron.core.pipeline_parallel.schedules import print_log
        # print_rank(k_whole.shape, 7)
        # print_rank(f"seqlen: {ctx._seqlen_k}", 7)
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


class FlashFixedSelfAttention(torch.nn.Module):
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
        assert flash_attn_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

        self.flash_attn_func = flash_attn_func

    def forward(self, q, k, v, micro_sp_idx=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        output = self.flash_attn_func(q, k, v, dropout_p=self.dropout_p, softmax_scale=self.softmax_scale, causal=self.causal)
        # [b,s,a,dim]
        return output

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
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, micro_sp_idx):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))
        args = get_args()
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

def parallel_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        if self.use_flash_attn:
            if args.pipe_sp_splits != 1:
                self.core_attention_flash = FlashSeqSelfAttention(
                    causal=True, attention_dropout=self.config.attention_dropout
                )
            else:
                self.core_attention_flash = FlashFixedSelfAttention(
                    causal=True, attention_dropout=self.config.attention_dropout
                )

    return wrapper


class ParallelAttentionPatch(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    # query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head) =>
    # query_layer = query_layer.contiguous().view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_context=None,
                rotary_pos_emb=None, *, inference_params=None, micro_sp_idx=None):
        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if inference_context:
            if self.layer_number not in inference_context.key_value_memory_dict:
                inf_max_seq_len = inference_context.max_sequence_length
                inf_max_batch_size = inference_context.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)

                inference_context.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
                is_first_step = True
            else:
                inference_key_memory, inference_value_memory = \
                    inference_context.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================
        if self.attention_type == AttnType.self_attn:

            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer,
            key_layer,
            value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)

            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
            query_layer = query_layer.contiguous().view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
            value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)

        if inference_context:
            batch_start = inference_context.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_context.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0), ("Current sequence length is "
            "longer than expected maximum sequence length! Increase inference_max_seq_length.")
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]


            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding
                    # (only the last token in the sequence)
                    q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
                else:
                    # In the first forward pass of inference,
                    # we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire
                    # prefix + to-be-generated output so
                    # we slice to just the prefix.
                    q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
        args = get_args()
        from dcu_megatron.core.pipeline_parallel.seq1f1b.sp_utils import get_splits
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if args.pipe_sp_splits != 1:
                if args.pipe_sp_strategy == "uniform_comp":
                    splits = get_splits()
                    start = sum(splits[:micro_sp_idx])
                    end = sum(splits[:micro_sp_idx+1])
                elif args.pipe_sp_strategy == "average":
                    start = (micro_sp_idx * q_pos_emb.size(0)) // args.pipe_sp_splits
                    end = ((micro_sp_idx + 1) * q_pos_emb.size(0)) // args.pipe_sp_splits
                q_pos_emb = q_pos_emb[start:end]
                k_pos_emb = k_pos_emb[start:end]

            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb,self.config)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb,self.config)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if args.pipe_sp_splits != 1:
            assert args.use_flash_attn, "FlashAttention is required for microbatching in sequence"

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v, micro_sp_idx)
            else:
                context_layer = self.core_attention_flash(q, k, v, micro_sp_idx)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias

class ParallelTransformerLayerPatch(MegatronModule):
    """A single transformer layer.

        Transformer layer takes input with size [s, b, h] and returns an
        output of the same size.
    """
    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_context=None,
                rotary_pos_emb=None,
                micro_sp_idx=None,
                *,
                inference_params=None):

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Update the params in case the retro param changes during inference
        # TODO: better redesign with inference param
        args = get_args()
        if args.retro_add_retriever:
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = args.retro_chunk_length
            self.retro_retrieved_length = \
                args.retro_num_retrieved_chunks * args.retro_chunk_length

        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)

        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                norm_output,
                attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb, micro_sp_idx=micro_sp_idx)

        # Residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                norm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            norm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        norm_output = self.post_attention_norm(norm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            norm_input, norm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    norm_input,
                    norm_output,
                    bias_dropout_add_func)
        elif self.layer_type == LayerType.retro_encoder:
            norm_input, norm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    norm_input,
                    norm_output,
                    bias_dropout_add_func)
        elif self.layer_type in (LayerType.retro_decoder,
                                 LayerType.retro_decoder_with_retriever):
            retriever_output, norm_input, norm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    norm_input,
                    norm_output,
                    inference_context,
                    bias_dropout_add_func)
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        mlp_output, mlp_bias = self.mlp(norm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(mlp_output,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            return output, retriever_output
        else:
            return output



class ParallelTransformerPatch(MegatronModule):

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              rotary_pos_emb, is_first_microbatch, micro_sp_idx=None):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        te_forward_kwargs = {}
        if self.transformer_impl == 'transformer_engine':
            te_forward_kwargs['is_first_microbatch'] = is_first_microbatch
            if self.transformer_engine_v_0_10:
                te_forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                if self.transformer_impl == 'transformer_engine':
                    hidden_states = transformer_engine.pytorch.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        tensor_parallel.get_cuda_rng_tracker,
                        mpu.get_tensor_model_parallel_group(),
                        hidden_states, attention_mask, encoder_output,
                        enc_dec_attn_mask, **te_forward_kwargs)
                else:
                    micro_sp_idx = torch.tensor([micro_sp_idx], device='cuda', dtype=torch.int32)
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask,
                        encoder_output, enc_dec_attn_mask,
                        None, None, None, None, rotary_pos_emb, micro_sp_idx)

                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = transformer_engine.pytorch.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            tensor_parallel.get_cuda_rng_tracker,
                            mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        micro_sp_idx = torch.tensor([micro_sp_idx], device='cuda', dtype=torch.int32)
                        hidden_states = tensor_parallel.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb, micro_sp_idx)
                else:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        micro_sp_idx = torch.tensor([micro_sp_idx], device='cuda', dtype=torch.int32)
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb, micro_sp_idx)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states


    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_context=None,
                rotary_pos_emb=None,
                *,
                inference_params=None,
                micro_sp_idx=None):
        # hidden_states: [s, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Checks.
        if inference_context:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Forward layers.
        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                if self.recompute_granularity == 'full':
                    hidden_states = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               rotary_pos_emb,
                                                               is_first_microbatch,
                                                               micro_sp_idx)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                    }
                    if self.transformer_impl == 'local':
                        forward_kwargs['inference_context'] = inference_context
                    else:
                        forward_kwargs['inference_params'] = inference_context

                    if self.transformer_impl == 'transformer_engine':
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
                        if self.transformer_engine_v_0_10:
                            forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                    else:
                        forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                        forward_kwargs['retriever_input'] = retriever_input
                        forward_kwargs['retriever_output'] = retriever_output
                        forward_kwargs['retriever_attn_mask'] = retriever_attn_mask
                        forward_kwargs['micro_sp_idx'] = micro_sp_idx

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)

                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                        # First Retro decoder layer returns both hidden_states
                        # and retriever_output. Make retriever_output available
                        # to subsequence Retro layers.
                        if isinstance(hidden_states, tuple):
                            assert len(hidden_states) == 2
                            hidden_states, retriever_output = hidden_states
                            forward_kwargs["retriever_output"] = retriever_output

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states