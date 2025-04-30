from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronCoreTransformerLayer


class TransformerLayer(MegatronCoreTransformerLayer):
    def _submodule_attn_router_forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        state=None,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states, _ = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        probs, routing_map = self.mlp.router(pre_mlp_layernorm_output)
        local_tokens, probs, tokens_per_expert = self.mlp.token_dispatcher.dispatch_preprocess(
            pre_mlp_layernorm_output, routing_map, probs
        )

        return (local_tokens, probs, hidden_states, pre_mlp_layernorm_output, tokens_per_expert)

    def _submodule_dispatch_forward(self, local_tokens, probs, state=None):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        token_dispatcher = self.mlp.token_dispatcher
        if self.is_deepep:
            token_dispatcher._comm_manager.token_probs = probs

        return token_dispatcher.dispatch_all_to_all(local_tokens, probs)

    def _submodule_moe_forward(self, dispatched_tokens, probs=None, state=None):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        shared_expert_output = None
        token_dispatcher = self.mlp.token_dispatcher
        if self.is_deepep:
            token_dispatcher._comm_manager.dispatched_probs = state.dispatched_probs
            dispatched_tokens, tokens_per_expert, permuted_probs = (
                token_dispatcher.dispatch_postprocess(dispatched_tokens)
            )
        else:
            dispatched_tokens, permuted_probs = token_dispatcher.dispatch_postprocess(
                dispatched_tokens, probs
            )
            tokens_per_expert = state.tokens_per_expert
        expert_output, mlp_bias = self.mlp.experts(
            dispatched_tokens, tokens_per_expert, permuted_probs
        )
        assert mlp_bias is None, f"Bias is not supported in {token_dispatcher.__class__.__name__}"
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            shared_expert_output = self.mlp.shared_experts(state.pre_mlp_layernorm_output)
        expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        return expert_output, shared_expert_output, mlp_bias

    def _submodule_combine_forward(self, output, shared_expert_output=None, state=None):
        residual = state.residual
        token_dispatcher = self.mlp.token_dispatcher
        output = token_dispatcher.combine_all_to_all(output)
        output = token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        mlp_output_with_bias = (output, None)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_attn_router_dw(self):
        self.self_attention.backward_dw()

    def _submodule_mlp_dw(self):
        self.mlp.backward_dw()

    def _submodule_attn_router_postprocess(
        self, node, local_tokens, probs, residual, pre_mlp_layernorm_output, tokens_per_expert
    ):
        node.common_state.residual = node.detach(residual)
        if self.mlp.use_shared_expert:
            node.common_state.pre_mlp_layernorm_output = node.detach(pre_mlp_layernorm_output)

        if not self.is_deepep:
            node.common_state.tokens_per_expert = tokens_per_expert
        return local_tokens, probs

    def _submodule_dispatch_postprocess(self, node, dispatched_tokens, probs):
        if self.is_deepep:
            node.common_state.dispatched_probs = node.detach(probs)
            return dispatched_tokens
        else:
            return dispatched_tokens, probs

    def _submodule_mlp_postprocess(self, node, expert_output, shared_expert_output, mlp_bias):
        assert mlp_bias is None
        node.common_state.pre_mlp_layernorm_output = None
        if shared_expert_output is None:
            return expert_output
        return expert_output, shared_expert_output

    def _submodule_combine_postprocess(self, node, output):
        cur_stream = torch.cuda.current_stream()
        node.common_state.residual.record_stream(cur_stream)
        node.common_state.residual = None
        return output

    def _submodule_attn_postprocess(self, node, hidden_states, context):
        return hidden_states

    def _submodule_dense_postprocess(self, node, hidden_states):
        return hidden_states

    def _submodule_not_implemented(self, *args):
        raise NotImplementedError("This callable is not implemented.")

    def get_submodule_callables(self, chunk_state):
        """
        The forward callables take 2 parts of inputs:
        1. The ScheduleNode object.
        2. The input tensors.
        """
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

        self.is_moe = isinstance(self.mlp, MoELayer)
        self.is_deepep = False
        if self.is_moe:
            self.is_deepep = isinstance(self.mlp.token_dispatcher, MoEFlexTokenDispatcher)

        def get_func_with_default(func, default_func):
            if self.is_moe:
                return func
            return default_func

        def callable_wrapper(forward_func, postprocess_func, node, *args):
            state = getattr(node, 'common_state', None)
            callable_outputs = forward_func(*args, state=state)
            if isinstance(callable_outputs, tuple):
                outputs = postprocess_func(node, *callable_outputs)
            else:
                outputs = postprocess_func(node, callable_outputs)
            return outputs

        attn_func = get_func_with_default(
            self._submodule_attn_router_forward, self._forward_attention
        )

        def attn_wrapper(hidden_states, state=None):
            return attn_func(
                hidden_states=hidden_states,
                attention_mask=chunk_state.attention_mask,
                attention_bias=chunk_state.attention_bias,
                inference_params=chunk_state.inference_params,
                packed_seq_params=chunk_state.packed_seq_params,
                sequence_len_offset=chunk_state.sequence_len_offset,
                rotary_pos_emb=chunk_state.rotary_pos_emb,
                rotary_pos_cos=chunk_state.rotary_pos_cos,
                rotary_pos_sin=chunk_state.rotary_pos_sin,
                state=state,
            )

        attn_postprocess_func = get_func_with_default(
            self._submodule_attn_router_postprocess, self._submodule_attn_postprocess
        )

        dispatch_func = get_func_with_default(
            self._submodule_dispatch_forward, self._submodule_not_implemented
        )
        dispatch_postprocess_func = get_func_with_default(
            self._submodule_dispatch_postprocess, self._submodule_not_implemented
        )

        mlp_func = get_func_with_default(self._submodule_moe_forward, self._forward_mlp)
        mlp_postprocess_func = get_func_with_default(
            self._submodule_mlp_postprocess, self._submodule_dense_postprocess
        )

        combine_func = get_func_with_default(
            self._submodule_combine_forward, self._submodule_not_implemented
        )
        combine_postprocess_func = get_func_with_default(
            self._submodule_combine_postprocess, self._submodule_not_implemented
        )

        attn_forward = partial(callable_wrapper, attn_wrapper, attn_postprocess_func)
        dispatch_forward = partial(callable_wrapper, dispatch_func, dispatch_postprocess_func)
        mlp_forward = partial(callable_wrapper, mlp_func, mlp_postprocess_func)
        combine_forward = partial(callable_wrapper, combine_func, combine_postprocess_func)

        callables = TransformerLayerSubmoduleCallables(
            attention=SubmoduleCallables(forward=attn_forward, dw=self._submodule_attn_router_dw),
            dispatch=SubmoduleCallables(forward=dispatch_forward),
            mlp=SubmoduleCallables(forward=mlp_forward, dw=self._submodule_mlp_dw),
            combine=SubmoduleCallables(forward=combine_forward),
            is_moe=self.is_moe,
            is_deepep=self.is_deepep,
        )
        return callables
        