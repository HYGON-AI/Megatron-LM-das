# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

""" Storage writer for PyT Distributed format allowing asynchronous save. """

import logging
import queue
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from hyckpt_torch import _write_items

from megatron.core.dist_checkpointing.strategies.async_utils import _disable_gc
from megatron.core.dist_checkpointing.strategies.filesystem_async import _process_memory

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file


class FileSystemWriterAsync:
    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        transform_list,
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: Optional[queue.Queue],
        count_queue: Optional[queue.Queue],
        use_fsync: bool,
    ) -> Union[Tuple[int, Exception], None]:
        """
        Performs actual data saving to storage (used by worker threads).

        Args:
            local_proc_idx (int): index of the worker that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (queue.Queue): queue to return the write results
            count_queue (queue.Queue): queue to signal worker task completion (get + task_done)
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None when running in a worker (results put in queue); result tuple when main worker
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

        if results_queue is not None:
            results_queue.put(local_output)
        if count_queue is not None:
            # Signal this process is done.
            count_queue.get()
            count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )
        return local_output


    @staticmethod
    def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
        """
        Preloads tensors in `state_dict` to host memory via CPU memory.

        Args:
            write_buckets (List): List of `WriteBucket` objects that define what to
                save in a checkpoint.
            non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
        """
        result = []

        for bucket in write_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            tensor_list = []
            for item, tensor in tensor_data:
                # we belive these tensors are detached from the model trainers
                tensor_list.append((item, tensor.to("cpu", non_blocking=False)))
                # This is required for `PersistentAsyncCaller` to remove reference
                del tensor
            result.append((file_name, storage_key, (bytes_data, tensor_list)))
        if non_blocking:
            torch.cuda.synchronize()
        return result
