# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List

import torch

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _unshard_if_dtensor, _reshard_if_dtensor
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync.
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and torch.distributed.get_world_size(parallel_state.get_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))

        if hasattr(model_module,
                   "share_mtp_embedding_and_output_weight") and model_module.share_mtp_embedding_and_output_weight:
            weight = model_module.shared_embedding_or_mtp_embedding_weight()
            grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))
