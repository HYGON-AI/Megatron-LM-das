from functools import wraps


def transformer_block_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # mtp require seperate layernorms for main model and mtp modules, thus move finalnorm out of block
        config = args[0] if len(args) > 1 else kwargs['config']
        if (
            hasattr(config, "mtp_num_layers")
            and config.mtp_num_layers is not None
            and config.mtp_num_layers > 0
        ):
            self.main_final_layernorm = self.final_layernorm
            self.final_layernorm = None

    return wrapper


class TransformerBlock():
    def backward_dw(self):
        for layer in self.layers:
            layer.backward_dw()
