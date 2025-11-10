from typing import Optional
from functools import wraps

from megatron.training import get_args
from megatron.core import mpu
from megatron.core.transformer.transformer_config import TransformerConfig


def tie_word_embeddings_state_dict_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedule_method == "dualpipev":
            return

        fn(*args, **kwargs)

    return wrapper


def get_mtp_num_layers_to_build(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the number of MTP layers to build."""

    args = get_args()
    if args.schedule_method == "dualpipev":
        if mpu.is_pipeline_first_stage(ignore_virtual=True) and not args.dualpipev_first_chunk:
            return config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            return 0

    if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0


class MultiTokenPredictionLayer:
    def backward_dw(self):
        self.eh_proj.backward_dw()
        self.transformer_layer.backward_dw()


class MultiTokenPredictionBlock:
    def backward_dw(self):
        for layer in self.layers:
            layer.backward_dw()
