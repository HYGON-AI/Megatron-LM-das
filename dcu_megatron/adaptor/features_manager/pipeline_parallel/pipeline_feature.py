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
                           choices=['vanilla', 'dualpipev', 'seq1f1b', 'interleaved_seq1f1b', 'ripipe'],
                           help='Use pipeline provided by megatron if schedule-method is set to vanilla')
        # MoE communication overlap arguments
        group.add_argument('--overlap-ep-comm-with-split-attn', action="store_true",
                           default=False,
                           help='whether to split attention')
        group.add_argument('--num-layers-to-build',
                           type=num_layers_build_type,
                           default=None,
                           help='number of layers to build: '
                                '- An integer N: meaning n layers for each model block '
                                '- A string containing a Python list expression that defines a custom pattern')
        # Vocabulary parallelism.
        group.add_argument('--enable-vocab-parallel', action='store_true',
                           help='Enables vocabulary parallelism at the vocabulary layers. '
                           'Must be enabled together with pipeline model parallelism.')
        group.add_argument('--disable-backward-fusion', action='store_true',
                           help='disables the forward-backward fusion for the output '
                           'layer. requires two communication barriers instead of one')
        group.add_argument('--schedule-timer-start', type=int, default=10,
                           help='Start iteration of the vocabulary parallelism schedule timer')
        group.add_argument('--schedule-timer-end', type=int, default=20,
                           help='End iteration of the vocabulary parallelism schedule timer')

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
            assert args.pipeline_model_parallel_layout is None, "pipeline-model-parallel-layout shoule be None when using dualpipev"

            if args.delay_wgrad_compute and args.overlap_grad_reduce:
                assert bool(int(os.getenv("NVTE_OVERLAP_GRAD_REDUCE", "0"))), \
                    "NVTE_OVERLAP_GRAD_REDUCE should be set to 1 when --delay-wgrad-compute and --overlap-grad-reduce are set"

        if args.schedule_method == "dualpipev":
            if args.virtual_pipeline_model_parallel_size is not None:
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
            assert args.schedule_method == "dualpipev" or args.virtual_pipeline_model_parallel_size is not None, \
                'moe a2a overlap is only supported with vpp or dualpipev'

        # Vocabulary parallelism.
        if args.enable_vocab_parallel:
            assert args.pipeline_model_parallel_size > 1, 'pipeline parallel size '\
                'must be > 1 when vocab parallel is enabled'
            assert args.virtual_pipeline_model_parallel_size is None, 'vocab parallel'\
                'with interleaved schedule is not supported yet'
            assert (
                args.make_vocab_size_divisible_by %
                (args.tensor_model_parallel_size * args.pipeline_model_parallel_size) == 0
            ), f'vocab size must be divisible by model parallel size ({args.tensor_model_parallel_size * args.pipeline_model_parallel_size}) for vocab parallel'
            assert args.untie_embeddings_and_output_weights, '--enable-vocab-parallel requires' \
                'untie embeddings and output weights'
        else:
            args.disable_backward_fusion = False

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.pipeline_parallel.schedules import get_forward_backward_func_wrapper
        from dcu_megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build

        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                    get_forward_backward_func_wrapper,
                                    apply_wrapper=True)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.get_mtp_num_layers_to_build',
                                    get_mtp_num_layers_to_build)

        if args.schedule_method == "dualpipev":
            from megatron.training.utils import print_rank_0

            from dcu_megatron.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
                dualpipev_fp16forward,
                get_num_layers_to_build,
                _allreduce_embedding_grads_wrapper
            )
            from dcu_megatron.training.training import evaluate
            from dcu_megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            from dcu_megatron.training.training import build_train_valid_test_data_iterators_wrapper
            from dcu_megatron.core.models.gpt.gpt_model import GPTModel
            from dcu_megatron.core.models.gpt.fine_grained_callables import build_layer_callables_without_split_attn
            from dcu_megatron.training.global_vars import _set_tensorboard_writer, _set_wandb_writer, _set_one_logger
            from dcu_megatron.core.models.common.language_module.language_module import LanguageModule
            from dcu_megatron.core.tensor_parallel.layers import VocabParallelEmbedding
            from dcu_megatron.core.transformer.multi_token_prediction import tie_word_embeddings_state_dict_wrapper
            from dcu_megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
            from dcu_megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
            from dcu_megatron.core.pipeline_parallel.fine_grained_activation_offload_dualpipev import FineGrainedActivationOffloadingInterface

            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad', _allreduce_embedding_grads_wrapper)

            # use first rank
            patch_manager.register_patch('megatron.training.training.evaluate', evaluate)

            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.get_transformer_layer_offset', get_transformer_layer_offset)

            # support dualpipev, two data iterators
            patch_manager.register_patch('megatron.training.training.build_train_valid_test_data_iterators',
                                         build_train_valid_test_data_iterators_wrapper,
                                         apply_wrapper=True)

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
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                         VocabParallelEmbedding.__init__)
            patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.tie_word_embeddings_state_dict',
                                         tie_word_embeddings_state_dict_wrapper,
                                         apply_wrapper=True)

            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_step_calc_loss',
                                         forward_step_calc_loss)

            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                                         DistributedDataParallel._make_backward_post_hook)

            patch_manager.register_patch('megatron.core.models.gpt.fine_grained_callables.build_layer_callables',
                                        build_layer_callables_without_split_attn)

            patch_manager.register_patch('megatron.core.pipeline_parallel.fine_grained_activation_offload.FineGrainedActivationOffloadingInterface.init_chunk_handler',
                                        FineGrainedActivationOffloadingInterface.init_chunk_handler)

        if args.enable_vocab_parallel:
            from dcu_megatron.core.parallel_state import destroy_model_parallel_wrapper
            from dcu_megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
            from dcu_megatron.core.transformer.module import Float16Module

            patch_manager.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                                        destroy_model_parallel_wrapper,
                                        create_dummy=True)
            patch_manager.register_cls_funcs('megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator',
                                             [P2PCommunicator._communicate,
                                              P2PCommunicator.recv_forward,
                                              P2PCommunicator.send_backward_recv_forward])
            # embedding/output layer
            patch_manager.register_cls_funcs('megatron.core.transformer.module.Float16Module',
                                             [Float16Module.__init__,
                                              Float16Module.forward])

        from dcu_megatron.core.transformer.transformer_layer import TransformerLayer
        from dcu_megatron.core.transformer.transformer_block import TransformerBlock
        from dcu_megatron.core.models.gpt.gpt_model import GPTModel
        from dcu_megatron.core.transformer.multi_latent_attention import MLASelfAttention
        from dcu_megatron.core.transformer.attention import Attention
        from dcu_megatron.core.transformer.moe.moe_layer import MoELayer
        from dcu_megatron.core.distributed.data_parallel_base import _BaseDataParallel
        from dcu_megatron.core.transformer.module import Float16Module
        from dcu_megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer, MultiTokenPredictionBlock
        from dcu_megatron.core.pipeline_parallel.utils import NoopScheduleNode, ScheduleNode

        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.backward_dw',
                                    TransformerLayer.backward_dw,
                                    create_dummy=True)
        if args.schedule_method == "dualpipev" or args.overlap_ep_comm_with_split_attn:
            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.build_schedule_plan',
                                        GPTModel.build_schedule_plan)
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
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.forward',
                                     MultiTokenPredictionBlock.forward)
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.backward_dw',
                                    MultiTokenPredictionBlock.backward_dw,
                                    create_dummy=True)

        patch_manager.register_cls_funcs('megatron.core.pipeline_parallel.utils.NoopScheduleNode',
                                         [NoopScheduleNode.forward,
                                          NoopScheduleNode.backward,])
        patch_manager.register_cls_funcs('megatron.core.pipeline_parallel.utils.ScheduleNode',
                                         [ScheduleNode.forward,
                                          ScheduleNode._forward,
                                          ScheduleNode.backward,
                                          ScheduleNode._backward,])
