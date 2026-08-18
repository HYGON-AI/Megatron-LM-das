# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch
from megatron.core import mpu
from torch.utils.checkpoint import detach_variable


def should_recompute(config, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = config.virtual_pipeline_model_parallel_size
    pp_size = config.pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = config.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = config.num_layers // pp_size
    else:
        layer_per_chunk = config.num_layers

    if vpp_rank is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_rank = 0
    if vpp_size is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_size = 1

    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = config.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True

        return False

    if num_recompute is None:
        return True

    return recompute_priority < num_recompute
