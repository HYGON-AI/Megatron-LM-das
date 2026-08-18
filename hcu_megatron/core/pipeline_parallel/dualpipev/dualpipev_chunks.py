# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from functools import wraps
from typing import Optional
from megatron.core import mpu
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import fp32_to_float16, float16_to_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state

from hcu_megatron.core.parallel_state import get_dualpipe_chunk
from hcu_megatron.training.arguments import get_adaptor_args


def dualpipev_fp16forward(self, *inputs, fp32_output=True, **kwargs):
    dualpipe_first_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 0
    if dualpipe_first_stage:
        inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    dualpipe_last_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    if dualpipe_last_stage and fp32_output is True:
        outputs = float16_to_fp32(outputs)
    return outputs


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """

    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    args = get_adaptor_args()

    if config.pipeline_model_parallel_layout is not None:
        if getattr(args, "schedule_method", None) == "dualpipev" and vp_stage is None:
            vp_stage = 1 - int(getattr(args, 'dualpipev_first_chunk', True))
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if args.num_layers_to_build is not None:
        if isinstance(args.num_layers_to_build, int):
            return args.num_layers_to_build

        if getattr(args, 'dualpipev_first_chunk', True):
            return args.num_layers_to_build[pp_rank]
        else:
            return args.num_layers_to_build[-1-pp_rank]

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size
        if getattr(args, "schedule_method", None) == "dualpipev":
            pipeline_stages_left *= 2

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        assert (
            layers_to_distribute % pipeline_stages_left == 0
        ), "With uneven pipelineing the left over layers must be divisible by left over stages"
        num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.
        if (
            is_first_pp_stage
            and getattr(args, 'dualpipev_first_chunk', True)
            and config.num_layers_in_first_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if (
            is_first_pp_stage
            and not getattr(args, 'dualpipev_first_chunk', True)
            and config.num_layers_in_last_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), "num_layers should be divisible by pipeline_model_parallel_size"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size
        if getattr(args, "schedule_method", None) == "dualpipev":
            assert (
                num_layers_per_pipeline_rank % 2 == 0
            ), "num_layers should be divisible by pipeline_model_parallel_size * 2"
            num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

    # Non-interleaved pipeline parallelism:
    # Each stage gets a contiguous set of layers.
    num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if getattr(args, "schedule_method", None) == "dualpipev":
        if is_first_pp_stage:
            if  args.dualpipev_first_chunk and config.account_for_embedding_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"
            elif  not args.dualpipev_first_chunk and config.account_for_loss_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

        return num_layers_to_build

    if is_first_pp_stage and config.account_for_embedding_in_pipeline_split:
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if is_last_pp_stage and config.account_for_loss_in_pipeline_split:
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def _allreduce_embedding_grads_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = get_adaptor_args()
        if args.schedule_method == 'dualpipev':
            # dualpipev no need to do embedding allreduce
            # embedding and lm head are on save rank.
            if not args.untie_embeddings_and_output_weights:
                raise NotImplementedError
            else:
                return
        else:
            return fn(*args, **kwargs)

    return wrapper
