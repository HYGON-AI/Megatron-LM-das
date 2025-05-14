from functools import wraps

from megatron.core.transformer.transformer_block import TransformerBlock as MegatronCoreTransformerBlock

def transformer_block_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # mtp require seperate layernorms for main model and mtp modules, thus move finalnorm out of block
        config = args[0] if len(args) > 1 else kwargs['config']
        if hasattr(config, "mtp_num_layers") and config.mtp_num_layers is not None:
            self.main_final_layernorm = self.final_layernorm
            self.final_layernorm = None

    return wrapper


class TransformerBlock(MegatronCoreTransformerBlock):

    def get_layer_callables(self, layer_number: int):
        """
        Get the callables for the layer at the given layer number.
        """
        return self.layers[layer_number].get_submodule_callables()
