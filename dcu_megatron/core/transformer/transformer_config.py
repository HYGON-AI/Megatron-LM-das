import warnings

from functools import wraps
from dataclasses import make_dataclass, field

from megatron.training import get_args


def transformer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def wrapper(self):
        # remover experts from recompute_modules. Otherwise _post_init_ will raise error
        if self.recompute_modules is None:
            self.recompute_modules = set()
        self.recompute_modules = set(self.recompute_modules)
        recompute_experts = "experts" in self.recompute_modules
        recompute_router  = "router"  in self.recompute_modules
        self.recompute_modules.discard("experts")
        self.recompute_modules.discard("router")
        post_init_func(self)
        if recompute_experts:
            self.recompute_modules.add("experts")
        if recompute_router:
            self.recompute_modules.add("router")
        self.recompute_modules = list(self.recompute_modules)

        args = get_args()
        fields = []
        for key, value in vars(args).items():
            field_name = str(key)
            field_type = type(value)
            if not hasattr(self, key):
                field_def = (field_name, field_type, field(init=False))
                fields.append(field_def)
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))

        for key, value in vars(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)

        if self.recompute_granularity == 'selective':
            if len(self.recompute_modules) > 0:
                modules_set = set(self.recompute_modules)
                assert not ('moe' in modules_set and ('experts' in modules_set or 'router' in modules_set)), (
                    "'moe' cannot be used together with 'experts' or 'router' in recompute_modules. "
                    "Please choose either 'moe' or a combination of 'experts' and/or 'router'."
                )

        # offload activations
        if self.fine_grained_activation_offloading:
            assert (
                not self.cpu_offloading
            ), "offload_activation can not be used with cpu_offloading"

        if self.offload_modules is None:
            self.offload_modules = ["core_attn"]

        if len(self.offload_modules) > 0:
            allowed_modules = {
                "attn_norm", "qkv_linear", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "expert_fc2",
                "shared_fc1", "shared_fc2", "moe_act"
            }
            invalid_modules = set(self.offload_modules) - allowed_modules
            assert not invalid_modules, (
                f'Invalid choices for offload_modules: {invalid_modules}. '
                f'Allowed modules are: {allowed_modules}'
            )

            if "attn_proj" in self.offload_modules and "core_attn" not in self.offload_modules:
                raise ValueError(
                    "attn_proj cannot be set to offload_modules alone without core_attn "
                    "because the input of attn_proj is the output of core_attn, "
                    "which is needed in core_attn.backward()."
                )

    return wrapper
