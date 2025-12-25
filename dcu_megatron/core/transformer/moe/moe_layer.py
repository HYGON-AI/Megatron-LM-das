from functools import wraps

import torch

from megatron.core import tensor_parallel


def moe_layer_init_wrapper(moe_layer_init_func):
    @wraps(moe_layer_init_func)
    def wrapper(self, *args, **kwargs ):

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
    def wrapper(self, hidden_states: torch.Tensor
    ):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward_experts(dispatched_input, tokens_per_expert, permuted_probs):
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
            return expert_output, mlp_bias
        
        def custom_forward_router(hidden_states):
            probs, routing_map = self.router(hidden_states)
            return probs, routing_map

        if self.experts_recompute or self.router_recompute:
            shared_expert_output = self.shared_experts_compute(hidden_states)

            residual = hidden_states
            if self.router_recompute:
                probs, routing_map = tensor_parallel.checkpoint(custom_forward_router, False, hidden_states)
            else:
                probs, routing_map = custom_forward_router(hidden_states)

            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs
            )

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

            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        return moe_layer_foward_func(self, hidden_states=hidden_states)

    return wrapper


class MoELayer():
    def backward_dw(self):
        self.backward_routed_expert_dw()
        self.backward_shared_expert_dw()

    def backward_shared_expert_dw(self):
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def backward_routed_expert_dw(self):
        self.experts.backward_dw()
