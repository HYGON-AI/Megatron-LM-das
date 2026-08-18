# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import time
import inspect
from functools import wraps
from collections import defaultdict
from datetime import timedelta
from typing import Callable, List, Optional

import torch
import torch.distributed

from megatron.training import print_rank_0
from megatron.core.utils import is_torch_min_version
from megatron.core import parallel_state
from megatron.core.parallel_state import (
    RankGenerator,
    overwrite_nccl_comm_cfgs,
)

from hcu_megatron.training.arguments import get_adaptor_args


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


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations
    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    nccl_options = None
    if get_adaptor_args().enable_vocab_parallel:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
    if pg_name in nccl_comm_cfgs:
        # When fields in nccl_options.config are not specified, NCCL applies default settings.
        # The default values for Hopper GPUs are as follows:
        # cga_cluster_size = 4, max_ctas = 32, min_ctas = 1
        # Default values may differ between GPU generations and NCCL versions.
        if nccl_options is None:
            nccl_options = torch.distributed.ProcessGroupNCCL.Options(
                is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
            )
        if "cga_cluster_size" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name]["cga_cluster_size"]
        if "max_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name]["max_ctas"]
        if "min_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name]["min_ctas"]
        if "net_name" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.net_name = nccl_comm_cfgs[pg_name]["net_name"]
            # verify net_name value
            if nccl_options.config.net_name.lower() not in ["ib", "socket"]:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported."
                    f"Accepted values: 'IB' or 'socket'."
                )

    if get_adaptor_args().enable_vocab_parallel and pg_name == "pp":
        nccl_options.is_high_priority_stream = True
    return nccl_options


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
        adaptor_args = get_adaptor_args()
        log_time = (
            adaptor_args.comm_time_log_iter is not None
            and get_train_iter() is not None
            and get_train_iter() == adaptor_args.comm_time_log_iter
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


# Vocabulary parallelism
_LM_HEAD_MODEL_PARALLEL_GROUP = None
_VIRTUAL_VOCAB_PARALLEL_CHUNK = None

def get_lm_head_model_parallel_group():
    """Get the language model head result reduce group the caller rank belongs to."""
    assert (
        _LM_HEAD_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _LM_HEAD_MODEL_PARALLEL_GROUP


def get_virtual_vocab_parallel_chunk():
    """Get the current chunk for vocab pipeline-parallel."""
    global _VIRTUAL_VOCAB_PARALLEL_CHUNK
    return _VIRTUAL_VOCAB_PARALLEL_CHUNK


def set_virtual_vocab_parallel_chunk(chunk):
    """Set the current chunk for vocab pipeline-parallel."""
    global _VIRTUAL_VOCAB_PARALLEL_CHUNK
    _VIRTUAL_VOCAB_PARALLEL_CHUNK = chunk


def initialize_model_parallel_wrapper(fn):
    @wraps(fn)
    def wrapper(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_comm_backend: Optional[str] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[List[int]] = None,
        hybrid_context_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        create_gloo_process_groups: bool = True,
        high_priority_stream_groups: Optional[List[str]] = None,
        sharp_enabled_group: Optional[str] = None,
        rank_offset: int = 0,
        local_world_size: Optional[int] = None,
    ) -> None:
        fn(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_comm_backend=pipeline_model_parallel_comm_backend,
            use_sharp=use_sharp,
            context_parallel_size=context_parallel_size,
            hierarchical_context_parallel_sizes=hierarchical_context_parallel_sizes,
            hybrid_context_parallel=hybrid_context_parallel,
            expert_model_parallel_size=expert_model_parallel_size,
            num_distributed_optimizer_instances=num_distributed_optimizer_instances,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            nccl_communicator_config_path=nccl_communicator_config_path,
            distributed_timeout_minutes=distributed_timeout_minutes,
            order=order,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            create_gloo_process_groups=create_gloo_process_groups,
            high_priority_stream_groups=high_priority_stream_groups,
            sharp_enabled_group=sharp_enabled_group,
            rank_offset=rank_offset,
            local_world_size=local_world_size,
        )

        if get_adaptor_args().enable_vocab_parallel:
            global _LM_HEAD_MODEL_PARALLEL_GROUP
            assert _LM_HEAD_MODEL_PARALLEL_GROUP is None, 'lm head model parallel group is already initialized'

            world_size: int = (
                local_world_size if local_world_size is not None else torch.distributed.get_world_size()
            )

            model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

            if world_size % model_size != 0:
                raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

            data_parallel_size: int = world_size // model_size

            rank = torch.distributed.get_rank()

            decoder_rank_generator = RankGenerator(
                tp=tensor_model_parallel_size,
                ep=1,
                dp=data_parallel_size,
                pp=pipeline_model_parallel_size,
                cp=context_parallel_size,
                order=order,
                rank_offset=rank_offset,
            )

            nccl_comm_cfgs = {}
            if nccl_communicator_config_path is not None:
                try:
                    import yaml
                except ImportError:
                    raise RuntimeError(
                        "Cannot import `yaml`. Setting custom nccl communicator configs "
                        "requires the yaml package."
                    )

                with open(nccl_communicator_config_path, "r") as stream:
                    nccl_comm_cfgs = yaml.safe_load(stream)

            # Set is_high_priority_stream flag to the nccl_comm_cfgs if it is in high_priority_stream_groups
            high_priority_stream_groups = high_priority_stream_groups or []
            for pg_name in high_priority_stream_groups:
                overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, ("is_high_priority_stream", True))

            for ranks in decoder_rank_generator.get_ranks('pp'):
                group = create_group(
                    ranks,
                    timeout=timedelta(minutes=distributed_timeout_minutes),
                    pg_options=get_nccl_options("pp-lmhead", nccl_comm_cfgs),
                    group_desc="LM_HEAD_MODEL_PARALLEL_GROUP",
                )
                if rank in ranks:
                    _LM_HEAD_MODEL_PARALLEL_GROUP = group

        # output parallel group info
        global PARALLEL_GROUP_RANKS_MAP

        for group_key, group_value in _GROUP_NAME_DICT.items():
            print_rank_0(f"{group_key}: {PARALLEL_GROUP_RANKS_MAP[group_value]}")

    return wrapper


def destroy_model_parallel_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        global _LM_HEAD_MODEL_PARALLEL_GROUP
        _LM_HEAD_MODEL_PARALLEL_GROUP = None

    return wrapper
