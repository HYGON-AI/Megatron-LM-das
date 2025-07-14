from functools import wraps
from dataclasses import make_dataclass, field

from megatron.training import get_args


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        fn(self)
        args = get_args()
        fields = []
        # TE will get an unexpected keyword argument 'delay_wgrad_compute' if split_bw = True
        exclude_keys = {
            "split_bw"
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

        if hasattr(self, "moe_pad_expert_input_to_capacity"):
            self.moe_pad_expert_input_to_capacity = True
        if hasattr(self, "moe_expert_capacity_factor"):
            self.moe_expert_capacity_factor = 1.0

    return wrapper
