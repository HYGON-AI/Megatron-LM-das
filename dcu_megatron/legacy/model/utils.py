from megatron.training import get_args
from megatron.legacy.model import LayerNorm
from .rms_norm import LightopRMSNorm


def get_norm(config):
    args = get_args()
    if args.normalization == "LayerNorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)
    elif args.normalization == "RMSNorm":
        if args.apply_layernorm_1p:
            raise NotImplementedError('RMSNorm does not currently support the layernorm_1p formulation.')

        return LightopRMSNorm(dim=config.hidden_size,
                              eps=config.layernorm_epsilon)
    else:
        raise Exception(f"unsupported norm type '{args.normalization}'.")
