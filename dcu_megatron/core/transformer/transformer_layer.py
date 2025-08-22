from typing import Optional

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig


def get_transformer_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    args = get_args()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    actual_rank = pipeline_rank if getattr(args, 'dualpipev_first_chunk', True) else 2 * pipeline_size - 1 - pipeline_rank
    if args.num_layers_to_build is not None:
        if isinstance(args.num_layers_to_build, int):
            return args.num_layers_to_build * actual_rank
        else:
            return sum(args.num_layers_to_build[:actual_rank])

    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                middle_pipeline_stages *= 2

            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if middle_pipeline_stages > 0:
                num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
            else:
                num_layers_per_pipeline_rank = 0

            middle_pipeline_rank = (
                pipeline_rank
                if config.num_layers_in_first_pipeline_stage is None
                else pipeline_rank - 1
            )

            if not getattr(args, 'dualpipev_first_chunk', True):
                middle_pipeline_rank = (
                    config.pipeline_model_parallel_size
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.pipeline_model_parallel_size - 1
                ) + (config.pipeline_model_parallel_size - (pipeline_rank + 1))

            if getattr(args, 'dualpipev_first_chunk', True) and pipeline_rank == 0:
                    offset = 0
            else:
                offset = (
                    middle_pipeline_rank * num_layers_per_pipeline_rank
                ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

            if getattr(args, 'dualpipev_first_chunk', True):
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = num_layers - (pipeline_rank + 1) * num_layers_per_pipeline_rank

            # Reduce the offset of embedding layer from the total layer number
            if config.account_for_embedding_in_pipeline_split:
                if not parallel_state.is_pipeline_first_stage():
                    offset -= 1
                elif not getattr(args, 'dualpipev_first_chunk', True):
                    offset -= 1
    else:
        offset = 0
    return offset


class TransformerLayer():
    def backward_dw(self):
        self.self_attention.backward_dw()
        self.mlp.backward_dw()
