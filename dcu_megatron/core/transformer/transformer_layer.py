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
    def _callable_wrapper(
        self, is_forward, func, stream, event, *args, skip_detach=False, **kwargs
    ):
        """
        Wraps a function call so that it waits for a given CUDA event before
        proceeding and then runs the function on a specified CUDA stream.
        """
        torch.cuda.nvtx.range_push(func.__name__)
        event.wait(stream)
        with torch.cuda.stream(stream):
            outputs = func(*args, **kwargs)
        event.record(stream)
        if skip_detach:
            torch.cuda.nvtx.range_pop()
            return outputs
        detached_output_tensors = []
        if not is_forward:
            torch.cuda.nvtx.range_pop()
            return outputs, detached_output_tensors
        for tensor in outputs:
            if tensor is None:
                detached_output_tensors.append(None)
            elif tensor.dtype.is_floating_point:
                detached_output_tensors.append(tensor.detach().requires_grad_(True))
            else:
                detached_output_tensors.append(tensor.detach())
        torch.cuda.nvtx.range_pop()
        return outputs, detached_output_tensors

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
        tokens_per_expert, permutated_local_input_tokens, permuted_probs = self.mlp.token_dispatcher.dispatch_preprocess(
            pre_mlp_layernorm_output, routing_map, tokens_per_expert
        )

        outputs = [
            hidden_states,
            pre_mlp_layernorm_output,
            tokens_per_expert,
            permutated_local_input_tokens,
            permuted_probs,
        ]
        return tuple(outputs)

    def _submodule_dispatch_forward(self, tokens_per_expert, permutated_local_input_tokens, permuted_probs):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        tokens_per_expert, global_input_tokens, global_probs = self.mlp.token_dispatcher.dispatch_all_to_all(
            tokens_per_expert, permutated_local_input_tokens, permuted_probs
        )
        return [tokens_per_expert, global_input_tokens, global_probs]

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

    def _submodule_moe_forward(self, tokens_per_expert, global_input_tokens, global_prob, hidden_states):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        shared_expert_output = None
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.mlp.token_dispatcher.dispatch_postprocess(tokens_per_expert, global_input_tokens, global_prob)
        )
        expert_output, mlp_bias = self.mlp.experts(dispatched_input, tokens_per_expert, permuted_probs)
        expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            shared_expert_output = self.mlp.shared_experts(hidden_states)
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
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_attention_backward(
        self, hidden_states, pre_mlp_layernorm_output, detached_inputs
    ):
        pre_mlp_layernorm_output.backward(detached_inputs[1].grad)
        hidden_states.backward(detached_inputs[0].grad)

    def _submodule_attention_router_compound_backward(
        self,
        hidden_states,
        pre_mlp_layernorm_output,
        tokens_per_expert,
        permutated_local_input_tokens,
        probs,
        detached_inputs,
    ):
        permutated_local_input_tokens.backward(detached_inputs[3].grad)
        probs.backward(detached_inputs[4].grad)
        # tokens_per_expert.backward(detached_inputs[2].grad)
        pre_mlp_layernorm_output.backward(detached_inputs[1].grad)
        hidden_states.backward(detached_inputs[0].grad)

    def _submodule_dispatch_backward(self, global_input_tokens, detached_inputs):
        global_input_tokens.backward(detached_inputs[0].grad)

    def _submodule_dense_backward(self, output, detached_inputs):
        output.backward(detached_inputs[0].grad)

    def _submodule_moe_backward(
        self, expert_output, shared_expert_output, mlp_bias, detached_inputs
    ):
        expert_output.backward(detached_inputs[0].grad)
        shared_expert_output.backward(detached_inputs[1].grad)
        if mlp_bias is not None:
            mlp_bias.backward(detached_inputs[2].grad)

    def _submodule_combine_backward(self, hidden_states, detached_inputs):
        hidden_states.backward(detached_inputs[0].grad)

    def _submodule_post_combine_backward(self, output, output_grad):
        output.backward(output_grad)

    def _submodule_attention_router_compound_dgrad(self):
        raise NotImplementedError("Not implemented")

    def _submodule_attention_router_compound_dw(self):
        self.self_attention.backward_dw()
        # raise NotImplementedError("Not implemented")

    def _submodule_dispatch_dgrad(self):
        raise NotImplementedError("Not implemented")

    def _submodule_mlp_dgrad(self):
        raise NotImplementedError("Not implemented")

    def _submodule_mlp_dw(self):
        self.mlp.backward_dw()
        # raise NotImplementedError("Not implemented")

    def _submodule_combine_dgrad(self):
        raise NotImplementedError("Not implemented")

    def _submodule_identity_forward(self, *args):
        return args

    def _submodule_identity_backward(self, *args):
        pass

    def get_submodule_callables(self):
        """
        Returns a dictionary of submodule callables for the transformer layer.
        """
        from megatron.core.transformer.moe.moe_layer import MoELayer

        def get_func_with_default(func, default_func):
            if isinstance(self.mlp, MoELayer):
                return func
            return default_func

        attention_func = get_func_with_default(
            self._submodule_attention_router_compound_forward, self._submodule_attention_forward
        )
        attention_backward_func = get_func_with_default(
            self._submodule_attention_router_compound_backward, self._submodule_attention_backward
        )
        dispatch_func = get_func_with_default(
            self._submodule_dispatch_forward, self._submodule_identity_forward
        )
        dispatch_backward_func = get_func_with_default(
            self._submodule_dispatch_backward, self._submodule_identity_backward
        )
        mlp_func = get_func_with_default(self._submodule_moe_forward, self._submodule_dense_forward)
        mlp_backward_func = get_func_with_default(
            self._submodule_moe_backward, self._submodule_dense_backward
        )
        combine_func = get_func_with_default(
            self._submodule_combine_forward, self._submodule_identity_forward
        )
        combine_backward_func = get_func_with_default(
            self._submodule_combine_backward, self._submodule_identity_backward
        )
        post_combine_func = get_func_with_default(
            self._submodule_post_combine_forward, self._submodule_identity_forward
        )
        post_combine_backward_func = get_func_with_default(
            self._submodule_post_combine_backward, self._submodule_identity_backward
        )

        callables = TransformerLayerSubmoduleCallables(
            attention=SubmoduleCallables(
                forward=partial(self._callable_wrapper, True, attention_func, skip_detach=True),
                backward=partial(self._callable_wrapper, False, attention_backward_func),
                # dgrad=partial(self._callable_wrapper, False,self._submodule_attention_router_compound_dgrad),
                dw=partial(
                    self._callable_wrapper, False, self._submodule_attention_router_compound_dw
                ),
            ),
            dispatch=SubmoduleCallables(
                forward=partial(self._callable_wrapper, True, dispatch_func),
                backward=partial(self._callable_wrapper, False, dispatch_backward_func),
                # dgrad=partial(self._callable_wrapper, False, self._submodule_dispatch_dgrad),
            ),
            mlp=SubmoduleCallables(
                forward=partial(self._callable_wrapper, True, mlp_func),
                backward=partial(self._callable_wrapper, False, mlp_backward_func),
                # dgrad=partial(self._callable_wrapper, False, self._submodule_mlp_dgrad),
                dw=partial(self._callable_wrapper, False, self._submodule_mlp_dw),
            ),
            combine=SubmoduleCallables(
                forward=partial(self._callable_wrapper, True, combine_func),
                backward=partial(self._callable_wrapper, False, combine_backward_func),
                # dgrad=partial(self._callable_wrapper, False, self._submodule_combine_dgrad),
            ),
            post_combine=SubmoduleCallables(
                forward=partial(self._callable_wrapper, True, post_combine_func),
                backward=partial(self._callable_wrapper, False, post_combine_backward_func),
            ),
        )
        return callables
