""" Storage writer for PyT Distributed format allowing asynchronous save. """

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch import multiprocessing as mp
from hyckpt_torch import _write_items

from megatron.core.dist_checkpointing.strategies.async_utils import _disable_gc
from megatron.core.dist_checkpointing.strategies.filesystem_async import _process_memory

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file


@staticmethod
@_disable_gc()
def write_preloaded_data(
    transform_list,
    local_proc_idx: int,
    write_bucket: WriteBucket,
    results_queue: mp.SimpleQueue,
    count_queue: mp.JoinableQueue,
    use_fsync: bool,
) -> None:
    """
    Performs actual data saving to storage.

    Args:
        local_proc_idx (int): index of a local process that performs writing
        write_bucket (WriteBucket): data to write to storage
        results_queue (mp.Queue): queue to return the write results
            to the proxy checkpoint process.
        count_queue (mp.JoinableQueue): queue to marks worker task as completed
        use_fsync (bool): if True, calls os.fsync at the end of saving

    Returns: None, the write result are put into the `queue`
    """
    logger = logging.getLogger(__name__)
    logger.debug(f'{local_proc_idx} started')
    mem_before = _process_memory()
    rank = torch.distributed.get_rank()

    local_results = []
    try:
        local_results = _write_items(write_bucket)
        '''
        for result in local_results:
            if hasattr(result.index, 'index'):
                from dataclasses import replace
                new_index = replace(result.index, index=rank)
                new_result = replace(result, index=new_index)
        '''
        local_output = (local_proc_idx, local_results)
    except Exception as e:
        logger.debug(f'{local_proc_idx} failed')
        local_output = (local_proc_idx, e)

    results_queue.put(local_output)
    # Signal this process is done.
    count_queue.get()
    count_queue.task_done()

    mem_after = _process_memory()
    logger.debug(
        f"{local_proc_idx} consumed: {mem_after - mem_before},"
        f" before: {mem_before}, after: {mem_after}"
    )


@staticmethod
def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
    """Preload tensors in state_dict to host memory through CPU memory
    Args:
        write_buckets(List): List of `WriteBucket`,
                                which includes what to be saved in a checkpoint
        non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
    """
    result = []
    for bucket in write_buckets:
        file_name, storage_key, (bytes_data, tensor_data) = bucket
        tensor_data = [
            (item, tensor.to("cpu", non_blocking=False)) for item, tensor in tensor_data
        ]
        result.append((file_name, storage_key, (bytes_data, tensor_data)))
    if non_blocking:
        torch.cuda.synchronize()
    return result
