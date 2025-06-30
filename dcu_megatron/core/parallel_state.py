from functools import wraps
from collections import defaultdict

import torch

from megatron.training import print_rank_0
from megatron.core.utils import is_torch_min_version

PARALLEL_GROUP_RANKS_MAP = defaultdict(list)

def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    global PARALLEL_GROUP_RANKS_MAP
    if group_desc is not None:
        PARALLEL_GROUP_RANKS_MAP[group_desc].append(ranks)

    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'use_local_synchronization': use_local_synchronization,
        'group_desc': group_desc,
    }
    if not is_torch_min_version('2.4.0'):
        kwargs.pop('group_desc')
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop('timeout')
    return torch.distributed.new_group(**kwargs)


def initialize_model_parallel_wrapper(fn):

    group_dict = {
        'tp-group' : 'TENSOR_MODEL_PARALLEL_GROUP',
        'pp-group' : 'PIPELINE_MODEL_PARALLEL_GROUP',
        'dp-group' : 'DATA_PARALLEL_GROUP',
        'ep-group' : 'EXPERT_MODEL_PARALLEL_GROUP',
        'etp-group': 'EXPERT_TENSOR_PARALLEL_GROUP',
        'edp-group': 'EXPERT_DATA_PARALLEL_GROUP',
        'cp-group' : 'CONTEXT_PARALLEL_GROUP',
    }
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)

        global PARALLEL_GROUP_RANKS_MAP
        for group_key, group_value in group_dict.items():
            print_rank_0(f"{group_key}: {PARALLEL_GROUP_RANKS_MAP[group_value]}")

    return wrapper