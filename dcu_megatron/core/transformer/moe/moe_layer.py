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

        def custom_forward_experts(dispatched_input, tokens_per_expert):
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            return expert_output, mlp_bias
        
        def custom_forward_router(hidden_states):
            probs, routing_map = self.router(hidden_states)
            return probs, routing_map

        if self.experts_recompute or self.router_recompute:
            probs, routing_map = tensor_parallel.checkpoint(custom_forward_router, False, hidden_states) \
                                 if self.router_recompute else self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = tensor_parallel.checkpoint(custom_forward_experts, False, dispatched_input, tokens_per_expert) \
                                      if self.experts_recompute else self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)

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
