from typing import Any, Optional, Tuple, Union

from torch import Tensor

from megatron.training import get_args
from megatron.core import tensor_parallel, parallel_state
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronCoreTransformerLayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)


def get_transformer_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    args = get_args()
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                middle_pipeline_stages *= 2

            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if middle_pipeline_stages > 0:
                num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
            else:
                num_layers_per_pipeline_rank = 0

            middle_pipeline_rank = (
                pipeline_rank
                if config.num_layers_in_first_pipeline_stage is None
                else pipeline_rank - 1
            )

            if not getattr(args, 'dualpipev_first_chunk', True):
                middle_pipeline_rank = (
                    config.pipeline_model_parallel_size
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.pipeline_model_parallel_size - 1
                ) + (config.pipeline_model_parallel_size - (pipeline_rank + 1))

            if getattr(args, 'dualpipev_first_chunk', True) and pipeline_rank == 0:
                    offset = 0
            else:
                offset = (
                    middle_pipeline_rank * num_layers_per_pipeline_rank
                ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

            if getattr(args, 'dualpipev_first_chunk', True):
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = num_layers - (pipeline_rank + 1) * num_layers_per_pipeline_rank

            # Reduce the offset of embedding layer from the total layer number
            if config.account_for_embedding_in_pipeline_split:
                if not parallel_state.is_pipeline_first_stage():
                    offset -= 1
                elif not getattr(args, 'dualpipev_first_chunk', True):
                    offset -= 1
    else:
        offset = 0
    return offset


class TransformerLayer(MegatronCoreTransformerLayer):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):

        if (
            not isinstance(self.mlp, MoELayer)
            or not isinstance(self.mlp.token_dispatcher, MoEAlltoAllTokenDispatcher)
        ):
            return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    inference_params=inference_params,
                )

        (
            hidden_states,
            pre_mlp_layernorm_output,
            permutated_local_input_tokens,
            probs,
        ) = self._submodule_attention_router_compound_forward(
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

        dispatched_input, probs = self._submodule_dispatch_forward(
            permutated_local_input_tokens,
            probs，
        )

        expert_output, shared_expert_output, mlp_bias = self._submodule_moe_forward(
            dispatched_input,
            probs,
            pre_mlp_layernorm_output
        )

        expert_output = self._submodule_combine_forward(expert_output)

        output = self._submodule_post_combine_forward(
            expert_output,
            shared_expert_output,
            mlp_bias,
            hidden_states
        )

        return output, None

    def _submodule_attention_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        # todo
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        return hidden_states

    def _submodule_attention_preprocess_forward(
        self,
        hidden_states: Tensor,
    ):
        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        query, key, value = self.self_attention.compute_qkv(
            input_layernorm_output,
        )

        return residual, query, key, value

    def _submodule_attention_core_attn_forward(
        self,
        query,
        key,
        value,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        core_attn_out = self.self_attention.compute_attn(
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

    def _submodule_attention_postprocess_forward(
        self,
        residual,
        core_attn_out,
    ):
        attention_output_with_bias = self.self_attention.compute_proj(
            core_attn_out
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        return hidden_states

    def _submodule_attention_postprocess_router_compound_forward(
        self,
        residual,
        core_attn_out,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states = self._submodule_attention_postprocess_forward(
            residual,
            core_attn_out,
        )

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        permutated_local_input_tokens, probs, pre_mlp_layernorm_output = self.mlp.router_and_preprocess(pre_mlp_layernorm_output)
        outputs = [
            hidden_states，
            permutated_local_input_tokens,
            probs,
            pre_mlp_layernorm_output,
        ]
        return tuple(outputs)

    def _submodule_router_forward(
        self,
        pre_mlp_layernorm_output
    ):
        permutated_local_input_tokens, probs, pre_mlp_layernorm_output = self.mlp.router_and_preprocess(pre_mlp_layernorm_output)

        return tuple(outputs)

    def _submodule_attention_router_compound_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states = self._submodule_attention_forward(
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

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        (
            permutated_local_input_tokens,
            probs,
            pre_mlp_layernorm_output,
        ) = self._submodule_router_forward(pre_mlp_layernorm_output)

        outputs = [
            hidden_states,
            pre_mlp_layernorm_output,
            permutated_local_input_tokens,
            probs,
        ]
        return tuple(outputs)

    def _submodule_attention_shared_expert_compound_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states = self._submodule_attention_forward(
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

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        shared_expert_output = self._submodule_shared_expert_forward(pre_mlp_layernorm_output)

        outputs = [
            hidden_states,
            shared_expert_output,
            pre_mlp_layernorm_output,
        ]
        return tuple(outputs)

    def _submodule_attention_router_shared_expert_compound_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Performs a combined forward pass that includes self-attention, MLP routing and shared-experts logic.
        """
        (
            hidden_states,
            pre_mlp_layernorm_output,
            permutated_local_input_tokens,
            probs,
        ) = self._submodule_attention_router_compound_forward(
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

        shared_expert_output = self._submodule_shared_expert_forward(pre_mlp_layernorm_output)

        outputs = [
            hidden_states,
            shared_expert_output,
            permutated_local_input_tokens,
            probs,
        ]
        return tuple(outputs)

    def _submodule_attention_proj_router_shared_expert_compound_forward(
        self,
        residual,
        core_attn_out,
    ):
        """
        Performs a combined forward pass that includes self-attention, MLP routing and shared-experts logic.
        """
        (
            hidden_states，
            permutated_local_input_tokens,
            probs,
            pre_mlp_layernorm_output,
        ) = self._submodule_attention_postprocess_router_compound_forward(
            residual,
            core_attn_out,
        )

        shared_expert_output = self._submodule_shared_expert_forward(pre_mlp_layernorm_output)

        outputs = [
            hidden_states,
            shared_expert_output,
            permutated_local_input_tokens,
            probs,
        ]
        return tuple(outputs)

    def _submodule_shared_expert_forward(self, pre_mlp_layernorm_output):
        """
        Performs a forward pass for shared experts.
        """
        shared_expert_output = None
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            shared_expert_output = self.mlp.shared_experts(pre_mlp_layernorm_output)
        return shared_expert_output

    def _submodule_dispatch_forward(self, permutated_local_input_tokens, probs):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        dispatched_input, probs = self.mlp.dispatch(
            permutated_local_input_tokens, probs
        )
        return dispatched_input, probs

    def _submodule_dense_forward(self, hidden_states):
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_moe_forward(self, dispatched_input, probs, pre_mlp_layernorm_output):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        shared_expert_output = self._submodule_shared_expert_forward(pre_mlp_layernorm_output)
        expert_output, mlp_bias = self._submodule_routed_experts_forward(dispatched_input, probs)

        return expert_output, shared_expert_output, mlp_bias

    def _submodule_routed_experts_forward(self, dispatched_input, probs):
        """
        Performs a forward pass for the MLP submodule, including only routed-expert computations.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.mlp.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
        )
        expert_output, mlp_bias = self.mlp.experts(dispatched_input, tokens_per_expert, permuted_probs)
        expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        return expert_output, mlp_bias

    def _submodule_combine_forward(self, expert_output):
        return self.mlp.token_dispatcher.token_combine(expert_output)

    def _submodule_post_combine_forward(
        self, expert_output, shared_expert_output, mlp_bias, residual
    ):
        """
        Re-combines the expert outputs (and optional shared_expert_output) into the same order
        as the original input tokens, applying any required bias.
        """
        output = self.mlp.token_dispatcher.combine_postprocess(expert_output)
        if shared_expert_output is not None:
            output += shared_expert_output
        mlp_output_with_bias = (output, mlp_bias)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_attention_dw(self):
        self.self_attention.backward_dw()

    def _submodule_attention_router_compound_dw(self):
        self._submodule_attention_dw()

    def _submodule_mlp_dw(self):
        self.mlp.backward_dw()

    def _submodule_attention_qkv_dw(self):
        self.self_attention._backward_qkv_proj()

    def _submodule_attention_proj_dw(self):
        self.self_attention._backward_output_proj()

    def _submodule_attention_proj_router_shared_expert_compound_dw(self):
        self.self_attention._backward_output_proj()
        self.mlp.backward_shared_expert_dw()

    def _submodule_routed_experts_dw(self):
        self.mlp.backward_routed_expert_dw()

    def backward_dw(self):
        self._submodule_attention_dw()
        self._submodule_mlp_dw()
