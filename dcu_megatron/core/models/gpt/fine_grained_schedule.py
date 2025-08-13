import contextlib
import weakref
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor

from megatron.core.transformer.module import float16_to_fp32
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.utils import deprecate_inference_params

from dcu_megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState
from dcu_megatron.core.pipeline_parallel.combined_1f1b import (
    AbstractSchedulePlan,
    ScheduleNode,
    get_com_stream,
    get_comp_stream,
    make_viewless,
)


def weak_method(method):
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


class PreProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self):

        gpt_model = self.gpt_model
        decoder_input = self.model_chunk_state.decoder_input
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        inference_context = self.model_chunk_state.inference_context
        inference_params = self.model_chunk_state.inference_params
        packed_seq_params = self.model_chunk_state.packed_seq_params

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif gpt_model.pre_process:
            decoder_input = gpt_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            # TODO(dongcl)
            decoder_input = gpt_model.decoder.input_tensor

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if gpt_model.position_embedding_type == 'rope' and not gpt_model.config.multi_latent_attention:
            if not gpt_model.training and gpt_model.config.flash_decode and inference_context:
                assert (
                    inference_context.is_static_batching()
                ), "GPTModel currently only supports static inference batching."
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = gpt_model.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    gpt_model.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = gpt_model.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, gpt_model.decoder, decoder_input, gpt_model.config, packed_seq_params
                )
                rotary_pos_emb = gpt_model.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        elif gpt_model.position_embedding_type == 'mrope' and not gpt_model.config.multi_latent_attention:
            if gpt_model.training or not gpt_model.config.flash_decode:
                rotary_pos_emb = gpt_model.rotary_pos_emb(position_ids, gpt_model.mrope_section)
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            (gpt_model.config.enable_cuda_graph or gpt_model.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_context
            and inference_context.is_static_batching()
            and not gpt_model.training
        ):
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * inference_context.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # saved for later use
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):
        gpt_model = self.gpt_model
        
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        labels = self.model_chunk_state.labels
        loss_mask = self.model_chunk_state.loss_mask
        attention_mask = self.model_chunk_state.attention_mask
        decoder_input = self.model_chunk_state.decoder_input
        inference_params= self.model_chunk_state.inference_params
        rotary_pos_emb = self.model_chunk_state.rotary_pos_emb
        rotary_pos_cos = self.model_chunk_state.rotary_pos_cos
        rotary_pos_sin = self.model_chunk_state.rotary_pos_sin
        packed_seq_params = self.model_chunk_state.packed_seq_params
        extra_block_kwargs = self.model_chunk_state.extra_block_kwargs
        sequence_len_offset = self.model_chunk_state.sequence_len_offset
        runtime_gather_output = self.model_chunk_state.runtime_gather_output
        inference_context = self.model_chunk_state.inference_context
    
        # Final layer norm.
        if gpt_model.decoder.final_layernorm is not None:
            hidden_states = gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = transformer_layer.make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not gpt_model.decoder.pre_process and len(gpt_model.decoder.layers) == 0 and not gpt_model.decoder.final_layernorm:
            hidden_states = hidden_states.clone()

        # logits and loss
        output_weight = None
        if gpt_model.share_embeddings_and_output_weights:
            output_weight = gpt_model.shared_embedding_or_output_weight()

        if gpt_model.mtp_process:
            hidden_states = gpt_model.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=gpt_model.embedding,
                output_layer=gpt_model.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                compute_language_model_loss=gpt_model.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if (
            gpt_model.mtp_process is not None
            and getattr(gpt_model.decoder, "main_final_layernorm", None) is not None
        ):
            # move block main model final norms here
            hidden_states = gpt_model.decoder.main_final_layernorm(hidden_states)

        if not gpt_model.post_process:
            return hidden_states

        if (
            not gpt_model.training
            and inference_context is not None
            and inference_context.materialize_only_last_token_logits
        ):
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [1, B, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = gpt_model.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if has_config_logger_enabled(gpt_model.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(gpt_model.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = gpt_model.compute_language_model_loss(labels, logits)

        return float16_to_fp32(loss)


class TransformerLayerNode(ScheduleNode):

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False):
        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_inputs=free_inputs,
        )
        # layer state
        self.common_state = common_state
        # model chunk state
        self.chunk_state = chunk_state
        self.layer = layer
        self.detached = tuple()
        self.before_detached = tuple()

    def detach(self, t):
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def backward_impl(self, outputs, output_grad):
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self.before_detached = None
        self.detached = None
        # return grads for record stream
        return grads


class MoeAttnNode(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_context = self.chunk_state.inference_context
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset
        inference_params = self.chunk_state.inference_params

        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            (
                hidden_states,
                shared_expert_output,
                tokens_per_expert,
                permutated_local_input_tokens,
                probs,
            ) = self.layer._submodule_attention_router_shared_expert_compound_forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )
        self.common_state.tokens_per_expert = tokens_per_expert

        # detached here
        self.common_state.probs = self.detach(probs)
        self.common_state.residual = self.detach(hidden_states)
        if self.layer.mlp.use_shared_expert and not self.layer.mlp.shared_expert_overlap:
            self.common_state.shared_expert_output = self.detach(shared_expert_output)

        return permutated_local_input_tokens

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_router_compound_dw()


class MoeAttnQKVNode(TransformerLayerNode):
    def forward_impl(self, hidden_states):
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            (
                attention_residual,
                query,
                key,
                value
            ) = self.layer._submodule_attention_preprocess_forward(
                hidden_states,
            )
        self.common_state.attention_residual = self.detach(attention_residual)
        return query, key, value

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_qkv_dw()


class MoeCoreAttnNode(TransformerLayerNode):
    def forward_impl(self, query, key, value):
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        inference_context = self.chunk_state.inference_context
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset
        inference_params = self.chunk_state.inference_params
        attention_mask = self.chunk_state.attention_mask
        attention_bias = self.chunk_state.attention_bias
        packed_seq_params = self.chunk_state.packed_seq_params

        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            core_attn_out = self.layer._submodule_attention_core_attn_forward(
                query,
                key,
                value,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        return core_attn_out


class MoeAttnPostNode(TransformerLayerNode):

    def forward_impl(self, core_attn_out):
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            (
                hidden_states,
                shared_expert_output,
                tokens_per_expert,
                permutated_local_input_tokens,
                probs,
            ) = self.layer._submodule_attention_proj_router_shared_expert_compound_forward(
                self.common_state.attention_residual,
                core_attn_out,
            )

        self.common_state.attention_residual = None
        self.common_state.tokens_per_expert = tokens_per_expert

        # detached here
        self.common_state.probs = self.detach(probs)
        self.common_state.residual = self.detach(hidden_states)
        if self.layer.mlp.use_shared_expert and not self.layer.mlp.shared_expert_overlap:
            self.common_state.shared_expert_output = self.detach(shared_expert_output)

        return permutated_local_input_tokens

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_proj_router_shared_expert_compound_dw()


class MoeDispatchNode(TransformerLayerNode):

    def forward_impl(self, permutated_local_input_tokens):
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            tokens_per_expert, global_input_tokens = token_dispatcher.dispatch_all_to_all(
                self.common_state.tokens_per_expert, permutated_local_input_tokens
            )
            # release tensor not used by backward
            # inputs.untyped_storage().resize_(0)
        self.common_state.tokens_per_expert = tokens_per_expert

        return global_input_tokens


class MoeMlPNode(TransformerLayerNode):
    def forward_impl(self, global_input_tokens):
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            expert_output, mlp_bias = self.layer._submodule_routed_experts_forward(
                self.common_state.tokens_per_expert, global_input_tokens
            )
            assert mlp_bias is None

        return expert_output

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_routed_experts_dw()


class MoeCombineNode(TransformerLayerNode):
    def forward_impl(self, expert_output):
        # TODO(lhb): if dw use grad of residual and probs, necessary synchronization should be add
        residual = self.common_state.residual
        token_dispatcher = self.layer.mlp.token_dispatcher

        shared_expert_output = None
        if self.layer.mlp.use_shared_expert and not self.layer.mlp.shared_expert_overlap:
            shared_expert_output = self.common_state.shared_expert_output

        with token_dispatcher.per_batch_state_context(self.common_state):
            permutated_local_input_tokens = token_dispatcher.combine_all_to_all(
                expert_output
            )
            output = self.layer._submodule_post_combine_forward(
                permutated_local_input_tokens, shared_expert_output, None, residual
            )
        cur_stream = torch.cuda.current_stream()
        self.common_state.residual.record_stream(cur_stream)
        self.common_state.probs.record_stream(cur_stream)
        if self.layer.mlp.use_shared_expert and not self.layer.mlp.shared_expert_overlap:
            self.common_state.shared_expert_output.record_stream(cur_stream)

        self.common_state.residual = None
        self.common_state.probs = None
        self.common_state.shared_expert_output = None
        return output


class DenseAttnNode(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_context = self.chunk_state.inference_context
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset
        inference_params = self.chunk_state.inference_params

        hidden_states = self.layer._submodule_attention_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            inference_context,
            packed_seq_params,
            sequence_len_offset,
            inference_params=inference_params,
        )
        return hidden_states

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_dw()


class FakeScheduleNode:

    def forward(self, inputs):
        return inputs

    def backward(self, outgrads):
        return outgrads


class DenseMlpNode(TransformerLayerNode):
    def forward_impl(self, hidden_states):
        return self.layer._submodule_dense_forward(hidden_states)

    def dw(self):
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_mlp_dw()


def build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream):
    common_state = TransformerLayerState()
    attn = DenseAttnNode(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = FakeScheduleNode()
    mlp = DenseMlpNode(chunk_state, common_state, layer, comp_stream, event)
    mlp.name = "mlp"
    combine = FakeScheduleNode()
    return TransformerLayerSchedulePlan(attn, dispatch, mlp, combine)


def build_layer_schedule_plan(layer, event, chunk_state, comp_stream, com_stream):
    if not isinstance(layer.mlp, MoELayer):
        return build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream)
    common_state = TransformerLayerState()

    attn_pre = MoeAttnQKVNode(chunk_state, common_state, layer, comp_stream, event)
    attn_pre.name = "attn_qkv"

    core_attn = MoeCoreAttnNode(chunk_state, common_state, layer, comp_stream, event)
    core_attn.name = "core_attn"

    attn_post = MoeAttnPostNode(chunk_state, common_state, layer, comp_stream, event)
    attn_post.name = "attn_post"

    dispatch = MoeDispatchNode(chunk_state, common_state, layer, com_stream, event, True)
    dispatch.name = "dispatch"

    mlp = MoeMlPNode(chunk_state, common_state, layer, comp_stream, event, True)
    mlp.name = "mlp"

    combine = MoeCombineNode(chunk_state, common_state, layer, com_stream, event, True)
    combine.name = "combine"

    return TransformerLayerSchedulePlan(attn_pre, core_attn, attn_post, dispatch, mlp, combine)


class TransformerLayerState(MoEAlltoAllPerBatchState):
    pass


class ModelChunkSate:
    pass


class TransformerLayerSchedulePlan:


    def __init__(self, attn_pre, core_attn, attn_post, dispatch, mlp, combine):
        self.attn_pre = attn_pre
        self.core_attn = core_attn
        self.attn_post = attn_post
        self.dispatch = dispatch
        self.mlp = mlp
        self.combine = combine


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    def __init__(self):
        super().__init__()
        self._pre_process = None
        self._post_process = None
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []
        self._event = torch.cuda.Event()

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        block_level_wgrad_compute=False,
    ):

        return schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad=grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
            block_level_wgrad_compute=block_level_wgrad_compute,
        )

    @property
    def event(self):
        return self._event

    def record_current_stream(self):
        stream = torch.cuda.current_stream()
        self.event.record(stream)

    def wait_current_stream(self):
        stream = torch.cuda.current_stream()
        self.event.wait(stream)

    @property
    def pre_process(self):
        return self._pre_process

    @pre_process.setter
    def pre_process(self, value):
        self._pre_process = value

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, value):
        self._post_process = value

    def get_layer(self, i):
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def num_layers(self):
        return len(self._transformer_layers)

    def add_layer(self, layer):
        self._transformer_layers.append(layer)

    @property
    def state(self):
        return self._model_chunk_state
    
    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        self._pre_process.model_chunk_state = None
        self._pre_process = None

        if self._post_process is not None:
            self._post_process.model_chunk_state = None
            self._post_process = None

# F_DISPATCH_B_MLP_SYNC_EVENT = torch.cuda.Event()
F_DISPATCH_B_MLP_SYNC_EVENT = None
B_MLP_B_DISPATCH_SYNC_EVENT = torch.cuda.Event()

F_ATTN_PRE_B_COMBINE_SYNC_EVENT = torch.cuda.Event()
B_COMBINE_F_ATTN_POST_SYNC_EVENT = None
# B_COMBINE_F_ATTN_POST_SYNC_EVENT = torch.cuda.Event()
B_ATTN_POST_F_COMBINE_SYNC_EVENT = torch.cuda.Event()


def schedule_layer_1f1b(
    f_layer,
    b_layer,
    f_input=None,
    b_grad=None,
    f_context=None,
    b_context=None,
    is_last_layer_in_bwd=False,
    block_level_wgrad_compute=False,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    is_sync_1f1b = f_layer is not None and b_layer is not None


    f_attn_pre_b_combine_sync_event = F_ATTN_PRE_B_COMBINE_SYNC_EVENT if is_sync_1f1b else None
    if f_layer is not None:
        with f_context:
            f_input = f_layer.attn_pre.forward(
                f_input,
                stream_record_event=f_attn_pre_b_combine_sync_event,
            )

    b_combine_f_attn_post_sync_event = B_COMBINE_F_ATTN_POST_SYNC_EVENT if is_sync_1f1b else None
    if b_layer is not None:
        with b_context:
            b_grad = b_layer.combine.backward(
                b_grad,
                stream_wait_event=f_attn_pre_b_combine_sync_event,
                stream_record_event=b_combine_f_attn_post_sync_event,
            )

    f_dispatch_b_mlp_sync_event = F_DISPATCH_B_MLP_SYNC_EVENT if is_sync_1f1b else None
    if f_layer is not None:
        with f_context:
            f_input = f_layer.core_attn.forward(f_input)
            f_input = f_layer.attn_post.forward(
                f_input,
                stream_wait_event=b_combine_f_attn_post_sync_event,
            )
            f_input = f_layer.dispatch.forward(
                f_input,
                stream_record_event=f_dispatch_b_mlp_sync_event,
            )

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.mlp.backward(
                b_grad,
                stream_wait_event=f_dispatch_b_mlp_sync_event,
            )

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.dispatch.backward(b_grad)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.mlp.forward(f_input)

    b_attn_post_f_combine_sync_event = B_ATTN_POST_F_COMBINE_SYNC_EVENT if is_sync_1f1b else None
    if b_layer is not None:
        with b_context:
            if not block_level_wgrad_compute:
                b_layer.mlp.dw()
            b_grad = b_layer.attn_post.backward(
                b_grad,
                stream_record_event=b_attn_post_f_combine_sync_event,
            )

    if f_layer is not None:
        with f_context:
            f_input = f_layer.combine.forward(
                f_input,
                stream_wait_event=b_attn_post_f_combine_sync_event,
            )

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.core_attn.backward(b_grad)
            b_grad = b_layer.attn_pre.backward(b_grad)

    if b_layer is not None and not block_level_wgrad_compute and not is_last_layer_in_bwd:
        with b_context:
            b_layer.attn_pre.dw()
            b_layer.attn_post.dw()

    return f_input, b_grad


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    block_level_wgrad_compute=False,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    f_input = None
    if f_schedule_plan:
        # pp output send/receive sync
        if pre_forward is not None:
            with f_context:
                pre_forward()
                del pre_forward
        f_schedule_plan.record_current_stream()
        f_input = f_schedule_plan.pre_process.forward()

    if b_schedule_plan:
        b_schedule_plan.record_current_stream()

        if pre_backward is not None:
            # pp grad send receive sync here, safe for now, maybe not safe in the future
            with torch.cuda.stream(get_com_stream()):
                b_schedule_plan.wait_current_stream()
                with b_context:
                    pre_backward()
                    del pre_backward
                b_schedule_plan.record_current_stream()

        assert grad is not None
        if b_schedule_plan.post_process is not None:
            with b_context:
                grad = b_schedule_plan.post_process.backward(grad)

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        f_input, grad = schedule_layer_1f1b(
            f_layer,
            b_layer,
            f_input=f_input,
            b_grad=grad,
            f_context=f_context,
            b_context=b_context,
            is_last_layer_in_bwd=(i == b_num_layers - 1),
            block_level_wgrad_compute=block_level_wgrad_compute,
        )
        torch.cuda.nvtx.range_pop()

    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            _, grad, = schedule_layer_1f1b(
                None,
                b_layer,
                b_grad=grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
                block_level_wgrad_compute=block_level_wgrad_compute
            )
            torch.cuda.nvtx.range_pop()

    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, _ = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

    # output pp send receive, overlapped with attn backward
    if f_schedule_plan is not None and post_forward is not None:
        with f_context:
            # The last submodule(layer_pre_forward) is running in the communication stream,
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_com_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input)
                del post_forward

    # pp grad send / receive, overlapped with attn dw of cur micro-batch and forward attn of next micro-batch
    if b_schedule_plan is not None and post_backward is not None:
        with b_context:
            b_schedule_plan.wait_current_stream()
            post_backward(grad)
            del post_backward

    # Delay the last attn_dw in backward pass (attn_dw of the first layer)
    # for overlapping with the p2p comm
    if not block_level_wgrad_compute and b_num_layers > 0:
        with b_context:
            b_schedule_plan.get_layer(0).attn_pre.dw()
            b_schedule_plan.get_layer(0).attn_post.dw()

    with f_context:
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)
    with b_context:
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(grad)

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()
    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()

    # Release reference as early as possible, this helps avoid memory leak.
    if b_schedule_plan is not None:
        b_schedule_plan.release_state()

    if b_num_layers and block_level_wgrad_compute:
        def chunk_backward_dw():
            with b_context:
                for i in range(b_num_layers):
                    b_layer = b_schedule_plan.get_layer(i)
                    b_layer.attn_pre.dw()
                    b_layer.attn_post.dw()
                    b_layer.mlp.dw()
        return f_input, chunk_backward_dw

    return f_input, None


def build_model_chunk_schedule_plan(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None
):

    comp_stream = get_comp_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state
    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.decoder_input = decoder_input
    state.labels = labels
    state.inference_context = inference_context
    state.packed_seq_params = packed_seq_params
    state.extra_block_kwargs = extra_block_kwargs
    state.runtime_gather_output = runtime_gather_output
    state.inference_params = inference_params
    state.loss_mask = loss_mask
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, state, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:
        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    return model_chunk_schedule_plan
