from typing import Optional
from functools import wraps

import torch

from megatron.core import tensor_parallel


def moe_layer_init_wrapper(moe_layer_init_func):
    @wraps(moe_layer_init_func)
    def wrapper(self, *args, **kwargs):

        moe_layer_init_func(self, *args, **kwargs)

        config = args[0] if len(args) > 1 else kwargs['config']

        self.experts_recompute = (
            config.recompute_granularity == 'selective' and "experts" in config.recompute_modules
        )

        self.router_recompute = (
            config.recompute_granularity == 'selective' and "router" in config.recompute_modules
        )

    return wrapper


def moe_layer_forward_wrapper(moe_layer_foward_func):
    @wraps(moe_layer_foward_func)
    def wrapper(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward_experts(dispatched_input, tokens_per_expert, permuted_probs):
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
            return expert_output, mlp_bias
        
        def custom_forward_router(hidden_states, padding_mask):
            probs, routing_map = self.route(hidden_states, padding_mask)
            return probs, routing_map

        if self.experts_recompute or self.router_recompute:
            assert intermediate_tensors is None, "intermediate_tensors should be None when recomputing experts or router"

            shared_expert_output = self.shared_experts_compute(hidden_states)

            residual = hidden_states
            if self.router_recompute:
                probs, routing_map = tensor_parallel.checkpoint(custom_forward_router, False, hidden_states, padding_mask)
            else:
                probs, routing_map = custom_forward_router(hidden_states, padding_mask)

            hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = (
                self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
            )
            if self.experts_recompute:
                expert_output, mlp_bias = tensor_parallel.checkpoint(custom_forward_experts, False, dispatched_input, tokens_per_expert, permuted_probs)
            else:
                expert_output, mlp_bias = custom_forward_experts(dispatched_input, tokens_per_expert, permuted_probs)

            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
            output = self.token_dispatcher.combine_preprocess(expert_output)

            output = self.combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        return moe_layer_foward_func(
            self,
            hidden_states=hidden_states,
            intermediate_tensors=intermediate_tensors,
            padding_mask=padding_mask,
        )

    return wrapper
