from functools import wraps
from megatron.training import get_args


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
        fn(self, *args, **kwargs)
        if self.mlp.__class__ is MoELayer:
            if self.mlp.experts.__class__ is GroupedMLP:
                self.mlp.experts.layer_number = self.layer_number
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
            global_args = get_args()
            if global_args.n_shared_experts:
                self.mlp.shared_experts.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper
