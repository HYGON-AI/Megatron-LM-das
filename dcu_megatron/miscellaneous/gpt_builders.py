from functools import wraps

from dcu_megatron.training.utils import print_rank_message
from dcu_megatron.core.parallel_state import get_parallel_group_ranks


def gpt_builder_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        print_rank_message(model, rank_id=get_parallel_group_ranks()["PIPELINE_MODEL_PARALLEL_GROUP"][0])

        return model

    return wrapper
