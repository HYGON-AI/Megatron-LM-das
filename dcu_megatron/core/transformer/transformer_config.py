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
        # TE will get an unexpected keyword argument 'delay_wgrad_compute' if delay_wgrad_compute = True
        exclude_keys = {
            "delay_wgrad_compute"
        }
        for key, value in vars(args).items():
            if key in exclude_keys:
                continue

            field_name = str(key)
            field_type = type(value)
            if not hasattr(self, key):
                field_def = (field_name, field_type, field(init=False))
                fields.append(field_def)
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))

        for key, value in vars(args).items():
            if key in exclude_keys:
                continue

            if not hasattr(self, key):
                setattr(self, key, value)

        if self.recompute_granularity == 'selective':
            if len(self.recompute_modules) > 0:
                modules_set = set(self.recompute_modules)
                assert not ('moe' in modules_set and ('experts' in modules_set or 'router' in modules_set)), (
                    "'moe' cannot be used together with 'experts' or 'router' in recompute_modules. "
                    "Please choose either 'moe' or a combination of 'experts' and/or 'router'."
                )

    return wrapper
