import os
import re

from argparse import ArgumentParser
from megatron.core import parallel_state
from megatron.core.utils import is_te_min_version, is_torch_min_version

from ..feature import AbstractFeature


def _eval_pattern(pattern):
    """ Validate and evaluate a string containing a Python list expression """
    assert isinstance(pattern, str)

    # validate input, only allow comma, digits, [, ], (, ), +, and *
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*]').search(pattern)):
        raise ValueError(f"Invalid pattern: {pattern}")

    return eval(pattern)


def num_layers_build_type(x):
    """number of layers to build.

    Accepts either:
    - An integer N: meaning n layers for each model block
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[2]*1)*3" evaluates to [1,1,1,2,1,1,1,2,1,1,1,2]
      The pattern length must match the total number of transformer blocks.
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)


class PipelineFeature(AbstractFeature):

    def __init__(self):
        super().__init__('schedule-method')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--schedule-method', type=str,
                           default='vanilla',
                           choices=['vanilla', 'dualpipev'],
                           help='Use pipeline provided by megatron if schedule-method is set to vanilla')
        # MoE communication overlap arguments
        group.add_argument('--overlap-moe-expert-parallel-comm-impl', type=str,
                           default='dcu_megatron',
                           choices=['megatron', 'dcu_megatron'],
                           help='What TransformerLayerSchedulePlan implementation to use..'
                           ' megatron: use the schedule plan implemented by megatron'
                           ' dcu_megatron: use the schedule plan implemented by us')
        group.add_argument('--num-layers-to-build',
                           type=num_layers_build_type,
                           default=None,
                           help='number of layers to build: '
                                '- An integer N: meaning n layers for each model block '
                                '- A string containing a Python list expression that defines a custom pattern')

    def pre_validate_args(self, args):
        if args.schedule_method != "dualpipev":
            return args

        pp_size = args.pipeline_model_parallel_size * 2
        if args.num_layers is None and args.num_layers_to_build is not None:
            pp_size = args.pipeline_model_parallel_size
            if isinstance(args.num_layers_to_build, int):
                args.num_layers = args.num_layers_to_build * pp_size * 2
            else:
                assert len(args.num_layers_to_build) == pp_size * 2, "The pattern length must match the total number of transformer blocks"
                args.num_layers = sum(args.num_layers_to_build)

        return args

    def validate_args(self, args):
        if args.schedule_method == "dualpipev":
            if args.delay_wgrad_compute and args.overlap_grad_reduce:
                assert bool(int(os.getenv("NVTE_OVERLAP_GRAD_REDUCE", "0"))), \
                    "NVTE_OVERLAP_GRAD_REDUCE should be set to 1 when --delay-wgrad-compute and --overlap-grad-reduce are set"

        if args.overlap_moe_expert_parallel_comm:
            assert is_torch_min_version("2.6.0"), "A2A Overlap encounters hang issue with torch version < 2.6.0"
            # Expert model parallelism requirements
            assert (
                args.expert_model_parallel_size > 1
            ), 'overlap_moe_expert_parallel_comm is only supported with expert model parallelism'
            assert args.moe_token_dispatcher_type in [
                'alltoall',
                'flex',
            ], 'overlap_moe_expert_parallel_comm is supported with alltoall/flex token dispatcher'

            assert (
                args.recompute_granularity != 'full'
            ), 'disable full recomputation when enabling overlap_moe_expert_parallel_comm'
            assert (
                args.recompute_method is None
            ), 'disable recomputation method when enabling overlap_moe_expert_parallel_comm'
            assert (
                args.recompute_num_layers is None
            ), 'recompute_num_layers must be None when enabling overlap_moe_expert_parallel_comm'

        if args.schedule_method == "dualpipev":
            if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
                raise AssertionError("The dualpipev and virtual_pipeline are incompatible.")

            layers_to_distribute = args.num_layers
            pipeline_stages_left = args.pipeline_model_parallel_size * 2
            if args.num_layers_to_build is not None:
                assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
                    "--decoder-first-pipeline-num-layers and --decoder-last-pipeline-num-layers should NOT be set when using --num-layers-to-build"

                if isinstance(args.num_layers_to_build, int):
                    assert args.num_layers_to_build * pipeline_stages_left == layers_to_distribute, "num-layers-to-build mismatch with num-layers"
                else:
                    assert len(args.num_layers_to_build) == pipeline_stages_left, "The pattern length must match the total number of transformer blocks"
                    assert sum(args.num_layers_to_build) == args.num_layers

            if args.decoder_first_pipeline_num_layers is not None and args.decoder_last_pipeline_num_layers is not None:
                if args.decoder_first_pipeline_num_layers is not None:
                    layers_to_distribute -= args.decoder_first_pipeline_num_layers
                    pipeline_stages_left -= 1
                if args.decoder_last_pipeline_num_layers is not None:
                    layers_to_distribute -= args.decoder_last_pipeline_num_layers
                    pipeline_stages_left -= 1
                if layers_to_distribute < pipeline_stages_left:
                    raise AssertionError(
                        'number of layers must be at least 2*pipeline_model_parallel_size in dualpipe')

            num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
            if num_micro_batch < args.pipeline_model_parallel_size:
                raise AssertionError(
                    "num_micro_batch should NOT be smaller than pipeline_model_parallel_size")

            if not args.delay_wgrad_compute:
                raise AssertionError("delay-wgrad-compute should be True")

            if not is_te_min_version("2.4.0"):
                raise AssertionError("Must have at least transformer-engine version of 2.4.0")

        if args.overlap_moe_expert_parallel_comm:
            assert args.transformer_impl == "transformer_engine", \
                "moe a2a overlap is only supported with transformer_engine implementation"
            assert args.schedule_method == "dualpipev" or args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None, \
                'moe a2a overlap is only supported with vpp or dualpipev'

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.pipeline_parallel.schedules import get_forward_backward_func_wrapper

        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                    get_forward_backward_func_wrapper,
                                    apply_wrapper=True)

        if args.schedule_method == "vanilla":
            return

        if args.schedule_method == "dualpipev":
            from megatron.training.utils import print_rank_0

            from dcu_megatron.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
                get_model,
                dualpipev_fp16forward,
                get_num_layers_to_build,
                _allreduce_embedding_grads_wrapper
            )
            from dcu_megatron.training.training import evaluate
            from dcu_megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            from dcu_megatron.training.utils import get_batch_on_this_tp_rank
            from dcu_megatron.training.training import pretrain
            from dcu_megatron.core.models.gpt.gpt_model import GPTModel
            from dcu_megatron.training.global_vars import _set_tensorboard_writer, _set_wandb_writer, _set_one_logger
            from dcu_megatron.core.models.common.language_module.language_module import LanguageModule
            from dcu_megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build
            from dcu_megatron.core.tensor_parallel.layers import VocabParallelEmbedding
            from dcu_megatron.core.transformer.multi_token_prediction import tie_word_embeddings_state_dict_wrapper

            patch_manager.register_patch('megatron.training.training.get_model', get_model)
            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads._allreduce_embedding_grads', _allreduce_embedding_grads_wrapper)

            # use first rank
            patch_manager.register_patch('megatron.training.training.evaluate', evaluate)

            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.get_transformer_layer_offset', get_transformer_layer_offset)

            # support dualpipev, two data iterators
            patch_manager.register_patch('megatron.training.training.pretrain', pretrain)

            # support dualpipev, broadcast loss_mask and labels
            patch_manager.register_patch(
                'megatron.training.utils.get_batch_on_this_tp_rank',
                get_batch_on_this_tp_rank)

            # (1) introduce an attribute dualpipev_first_chunk. (2) remove embedding when using dualpipev
            patch_manager.register_patch(
                'megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                GPTModel.__init__)
            patch_manager.register_patch(
                'megatron.core.models.gpt.gpt_model.GPTModel.shared_embedding_or_output_weight',
                GPTModel.shared_embedding_or_output_weight)

            # set _GLOBAL_TENSORBOARD_WRITER, _GLOBAL_WANDB_WRITER, _GLOBAL_ONE_LOGGER
            patch_manager.register_patch('megatron.training.global_vars._set_tensorboard_writer', _set_tensorboard_writer)
            patch_manager.register_patch('megatron.training.global_vars._set_wandb_writer', _set_wandb_writer)
            patch_manager.register_patch('megatron.training.global_vars._set_one_logger', _set_one_logger)

            # support mtp
            patch_manager.register_patch('megatron.core.models.common.language_module.language_module.LanguageModule.setup_embeddings_and_output_layer',
                                         LanguageModule.setup_embeddings_and_output_layer)
            patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.get_mtp_num_layers_to_build',
                                         get_mtp_num_layers_to_build)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                         VocabParallelEmbedding.__init__)
            patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.tie_word_embeddings_state_dict',
                                         tie_word_embeddings_state_dict_wrapper,
                                         apply_wrapper=True)

        from dcu_megatron.core.transformer.transformer_layer import TransformerLayer
        from dcu_megatron.core.transformer.transformer_block import TransformerBlock
        from dcu_megatron.core.models.gpt.gpt_model import GPTModel
        from dcu_megatron.core.transformer.multi_latent_attention import MLASelfAttention
        from dcu_megatron.core.transformer.attention import Attention
        from dcu_megatron.core.transformer.moe.moe_layer import MoELayer
        from dcu_megatron.core.distributed.data_parallel_base import _BaseDataParallel
        from dcu_megatron.core.transformer.module import Float16Module
        from dcu_megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer, MultiTokenPredictionBlock

        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.build_schedule_plan',
                                    GPTModel.build_schedule_plan,
                                    create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.backward_dw',
                                    TransformerLayer.backward_dw,
                                    create_dummy=True)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.backward_dw',
                                    GPTModel.backward_dw,
                                    create_dummy=True)
        patch_manager.register_patch('megatron.core.distributed.data_parallel_base._BaseDataParallel.backward_dw',
                                    _BaseDataParallel.backward_dw,
                                    create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.module.Float16Module.backward_dw',
                                    Float16Module.backward_dw,
                                    create_dummy=True)

        patch_manager.register_cls_funcs('megatron.core.transformer.multi_latent_attention.MLASelfAttention',
                                         [MLASelfAttention.compute_qkv,
                                          MLASelfAttention.compute_attn,
                                          MLASelfAttention.compute_proj,],
                                         create_dummy=True)
        patch_manager.register_cls_funcs('megatron.core.transformer.attention.Attention',
                                         [Attention.compute_qkv,
                                          Attention.compute_attn,
                                          Attention.compute_proj,],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.backward_dw',
                                    TransformerBlock.backward_dw,
                                    create_dummy=True)
        patch_manager.register_cls_funcs('megatron.core.transformer.moe.moe_layer.MoELayer',
                                         [MoELayer.backward_dw,
                                          MoELayer.backward_shared_expert_dw,
                                          MoELayer.backward_routed_expert_dw,],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.backward_dw',
                                    MultiTokenPredictionLayer.backward_dw,
                                    create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.backward_dw',
                                    MultiTokenPredictionBlock.backward_dw,
                                    create_dummy=True)
