from functools import partial
from typing import Any, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
)
from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronCoreTransformerLayer

from dcu_megatron.core.transformer.utils import SubmoduleCallables, TransformerLayerSubmoduleCallables


class TransformerLayer(MegatronCoreTransformerLayer):
    def forward(
        self,
        hidden_states: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
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
        (
            hidden_states,
            pre_mlp_layernorm_output,
            tokens_per_expert,
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

        (tokens_per_expert, global_input_tokens) = self._submodule_dispatch_forward(
            tokens_per_expert,
            permutated_local_input_tokens,
        )

        (expert_output, shared_expert_output, mlp_bias) = self._submodule_moe_forward(
            tokens_per_expert,
            global_input_tokens,
            pre_mlp_layernorm_output
        )

        expert_output = self._submodule_combine_forward(expert_output)[0]

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

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states

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

        probs, routing_map = self.mlp.router(pre_mlp_layernorm_output)
        tokens_per_expert = self.mlp.token_dispatcher.meta_prepare(
            pre_mlp_layernorm_output, probs, routing_map
        )
        tokens_per_expert, permutated_local_input_tokens = self.mlp.token_dispatcher.dispatch_preprocess(
            pre_mlp_layernorm_output, routing_map, tokens_per_expert
        )

        outputs = [
            hidden_states,
            pre_mlp_layernorm_output,
            tokens_per_expert,
            permutated_local_input_tokens,
            probs,
        ]
        return tuple(outputs)

    def _submodule_dispatch_forward(self, tokens_per_expert, permutated_local_input_tokens):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        tokens_per_expert, global_input_tokens = self.mlp.token_dispatcher.dispatch_all_to_all(
            tokens_per_expert, permutated_local_input_tokens
        )
        return [tokens_per_expert, global_input_tokens]

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

    def _submodule_moe_forward(self, tokens_per_expert, global_input_tokens, pre_mlp_layernorm_output):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        shared_expert_output = None
        (dispatched_input, tokens_per_expert) = (
            self.mlp.token_dispatcher.dispatch_postprocess(tokens_per_expert, global_input_tokens)
        )
        expert_output, mlp_bias = self.mlp.experts(dispatched_input, tokens_per_expert)
        expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            shared_expert_output = self.mlp.shared_experts(pre_mlp_layernorm_output)
        return expert_output, shared_expert_output, mlp_bias

    def _submodule_combine_forward(self, hidden_states):
        return [self.mlp.token_dispatcher.combine_all_to_all(hidden_states)]

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

    def _submodule_attention_router_compound_dw(self):
        self.self_attention.backward_dw()
        # raise NotImplementedError("Not implemented")

    def _submodule_mlp_dw(self):
        self.mlp.backward_dw()
        # raise NotImplementedError("Not implemented")
