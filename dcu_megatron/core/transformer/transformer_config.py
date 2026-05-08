from functools import wraps
from dataclasses import make_dataclass, dataclass, field
from typing import Literal

from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig as MegatronCoreTransformerConfig


@dataclass
class TransformerConfig(MegatronCoreTransformerConfig):

    normalization: Literal['LayerNorm', 'RMSNorm', 'LightopRMSNorm'] = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm`, `RMSNorm` and `LightopRMSNorm`. """


def transformer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def wrapper(self):
        args = get_args()

        # remover experts from recompute_modules. Otherwise _post_init_ will raise error
        if self.recompute_modules is None:
            self.recompute_modules = set()
        self.recompute_modules = set(self.recompute_modules)
        recompute_experts = "experts" in self.recompute_modules
        recompute_router  = "router"  in self.recompute_modules
        self.recompute_modules.discard("experts")
        self.recompute_modules.discard("router")
        self.recompute_modules = list(self.recompute_modules)

        # set delay_wgrad_compute to avoid AssertionError(overlap_moe_expert_parallel_comm must be enabled when enabling delay_wgrad_compute)
        # set overlap_moe_expert_parallel_comm to avoid AssertionError
        if args.schedule_method == "dualpipev":
            origin_delay_wgrad_compute = self.delay_wgrad_compute
            self.delay_wgrad_compute = False

            origin_overlap_moe_expert_parallel_comm = self.overlap_moe_expert_parallel_comm
            self.overlap_moe_expert_parallel_comm = False

        post_init_func(self)
        if recompute_experts:
            self.recompute_modules.append("experts")
        if recompute_router:
            self.recompute_modules.append("router")

        if args.schedule_method == "dualpipev":
            self.delay_wgrad_compute = origin_delay_wgrad_compute
            self.overlap_moe_expert_parallel_comm = origin_overlap_moe_expert_parallel_comm

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
                if 'experts' in modules_set or 'router' in modules_set:
                    assert 'moe' not in modules_set, (
                        "'moe' cannot be used together with 'experts' or 'router' in recompute_modules. "
                        "Please choose either 'moe' or a combination of 'experts' and/or 'router'."
                    )

    return wrapper
