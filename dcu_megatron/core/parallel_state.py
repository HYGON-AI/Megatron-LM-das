import time
import inspect
import warnings
from functools import wraps
from collections import defaultdict

import torch
import torch.distributed

from megatron.training import get_args, print_rank_0
from megatron.core.utils import is_torch_min_version
from megatron.core import parallel_state

PARALLEL_GROUP_RANKS_MAP = defaultdict(list)
_GROUP_MAP = {}
_COMM_LOGS = []
_GROUP_NAME_DICT = { 
    'tp_group' : 'TENSOR_MODEL_PARALLEL_GROUP',
    'pp_group' : 'PIPELINE_MODEL_PARALLEL_GROUP',
    'dp_group' : 'DATA_PARALLEL_GROUP',
    'ep_group' : 'EXPERT_MODEL_PARALLEL_GROUP',
    'etp_group': 'EXPERT_TENSOR_PARALLEL_GROUP',
    'edp_group': 'EXPERT_DATA_PARALLEL_GROUP',
    'cp_group' : 'CONTEXT_PARALLEL_GROUP',
    'tp-cp_group' : "TENSOR_AND_CONTEXT_PARALLEL_GROUP",
    'embd-pp_group': 'EMBEDDING_GROUP',
    'pos_embd-pp_group': 'POSITION_EMBEDDING_GROUP',
    'tp-ep_group': 'EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
    'tp-dp-cp_group': 'TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP',
    'tp-pp_group': 'MODEL_PARALLEL_GROUP',
    'dp-cp_group':'DATA_PARALLEL_GROUP_WITH_CP',
}

_FUNC_NAME_DICT = {
    'broadcast': 'broadcast',
    'all_reduce': 'all_reduce',
    'all_gather': 'all_gather',
    'all_gather_into_tensor': 'all_gather',
    'reduce_scatter': 'reduce_scatter',
    'reduce_scatter_tensor': 'reduce_scatter',
    'all_to_all_single': 'all_to_all',
    'isend' : 'send_recv_pp', 
    'irecv' : 'send_recv_pp', 
}


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
    group = torch.distributed.new_group(**kwargs)
    # global _global_process_group_list
    if parallel_state._global_process_group_list is None:
        # None stands for the default process group
        parallel_state._global_process_group_list = [None]
    if torch.distributed.get_rank() in ranks:
        parallel_state._global_process_group_list.append(group)

    global _GROUP_MAP
    _GROUP_MAP[group] = group_desc

    return group


def get_parallel_group_ranks():
    global PARALLEL_GROUP_RANKS_MAP
    return PARALLEL_GROUP_RANKS_MAP


def initialize_model_parallel_wrapper(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)

        global PARALLEL_GROUP_RANKS_MAP

        for group_key, group_value in _GROUP_NAME_DICT.items():
            print_rank_0(f"{group_key}: {PARALLEL_GROUP_RANKS_MAP[group_value]}")

    return wrapper


_DUALPIPE_CHUNK = None

def set_dualpipe_chunk(chunk_id):
    """set_dualpipe_chunk for fp16forward patch"""
    global _DUALPIPE_CHUNK
    _DUALPIPE_CHUNK = chunk_id


def get_dualpipe_chunk():
    global _DUALPIPE_CHUNK
    if _DUALPIPE_CHUNK is not None:
        return _DUALPIPE_CHUNK
    else:
        raise AssertionError("_DUALPIPE_CHUNK is None")


__TRAIN_ITER = None

def set_train_iter(train_iter):
    """set train iter for timer"""
    global __TRAIN_ITER
    __TRAIN_ITER = train_iter


def get_train_iter():
    global __TRAIN_ITER
    return __TRAIN_ITER


def log_timing_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        megatron_args = get_args()
        log_time = (
            megatron_args.comm_time_log_iter is not None
            and get_train_iter() is not None
            and get_train_iter() == megatron_args.comm_time_log_iter
        )

        if log_time:
            sig = inspect.signature(fn)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            arguments = bound_args.arguments

            reversed_dict = {v: k for k, v in _GROUP_NAME_DICT.items()}
            start_time = time.time()

        result = fn(*args, **kwargs)

        if log_time:
            elapsed_time = time.time() - start_time

            global _COMM_LOGS

            group = arguments.get('group', None)
            comm_group = reversed_dict[_GROUP_MAP[group]] if group is not None else "all"
            _COMM_LOGS.append({
                "type": _FUNC_NAME_DICT[fn.__name__],
                "group": comm_group,
                "time": elapsed_time
            })

        return result
    
    return wrapper
