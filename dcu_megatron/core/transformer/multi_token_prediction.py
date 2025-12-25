from typing import Optional
from functools import wraps

from megatron.training import get_args
from megatron.core import mpu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.utils import is_vp_last_stage


def tie_word_embeddings_state_dict_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedule_method == "dualpipev":
            return

        fn(*args, **kwargs)

    return wrapper


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""

    args = get_args()
    if args.schedule_method == "dualpipev":
        if mpu.is_pipeline_first_stage(ignore_virtual=True) and not args.dualpipev_first_chunk:
            return config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            return 0

    vp_size = config.virtual_pipeline_model_parallel_size
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1
    if is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_last_pp_stage:
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
