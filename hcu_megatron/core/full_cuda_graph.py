# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch

from megatron.core.full_cuda_graph import (
    get_shared_capture_stream,
    get_graph_pool,
    logger,
    FullCudaGraphWrapper as MegatronCoreFullCudaGraphWrapper
)
from megatron.core.tensor_parallel.random import get_all_rng_states
from megatron.training import get_args

try:
    from primus_turbo.pytorch.hygon import prepare_ck_grouped_gemm_workspace
    HAVE_TURBO = True
except (ImportError, ModuleNotFoundError):
    HAVE_TURBO = False


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        # labels and loss_mask of the first stage are set to None by get_batch_on_this_tp_rank
        # tokens and position_ids of the last stage are set to None by get_batch_on_this_tp_rank
        if tgt is None:
            src = None
        else:
            tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


class FullCudaGraphWrapper:
    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else 'validation'
        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str] is None
            MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()

            # CK descriptors are stream-scoped and must be prepared before capture.
            if get_args().use_primus_grouped_gemm:
                assert HAVE_TURBO, "primus_turbo.pytorch is NOT installed"
                workspace_info = prepare_ck_grouped_gemm_workspace(stream=capture_stream)
                logger.info(f"Prepared CK grouped-GEMM workspace on capture stream: {workspace_info}")

            with torch.cuda.graph(
                MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str],
                stream=capture_stream,
                pool=get_graph_pool(self.use_single_mempool),
                capture_error_mode="thread_local",
            ):
                MegatronCoreFullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str] is None:
            MegatronCoreFullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            MegatronCoreFullCudaGraphWrapper.cuda_graph[training_str].replay()
        self.next_iter(training_str)
        return MegatronCoreFullCudaGraphWrapper.result[training_str]
