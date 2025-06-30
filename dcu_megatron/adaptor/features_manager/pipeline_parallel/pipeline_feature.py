from argparse import ArgumentParser
from megatron.core.utils import is_te_min_version

from ..feature import AbstractFeature


class PipelineFeature(AbstractFeature):

    def __init__(self):
        super().__init__('schedule-method')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--schedule-method', type=str,
                           default='interleaved_1f1b',
                           choices=['dualpipev',
                                    'interleaved_1f1b'])
        group.add_argument('--combined-1f1b', action='store_true',
                           help='Batch-level overlapping in 1f1b stage.')
        group.add_argument('--combined-1f1b-recipe', type=str,
                           choices=['ep_a2a', 'golden'],
                           default='golden',
                           help='Options are "ep_a2a" and "golden".')
        group.add_argument('--split-bw', action='store_true',
                           help='Split dgrad and wgrad for batch-level overlapping')

    def validate_args(self, args):
        if args.schedule_method == "dualpipev":
            if args.num_layers_per_virtual_pipeline_stage is not None:
                raise AssertionError(
                    "The dualpipev and virtual_pipeline are incompatible.")
            if args.num_layers < args.pipeline_model_parallel_size * 2:
                raise AssertionError(
                    'number of layers must be at least 2*pipeline_model_parallel_size in dualpipe')
            num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
            if num_micro_batch < args.pipeline_model_parallel_size * 2 - 1:
                raise AssertionError(
                    "num_micro_batch should be greater than pipeline_model_parallel_size * 2 - 1")

        if args.combined_1f1b and args.transformer_impl != "transformer_engine":
            raise AssertionError("moe a2a overlap is only supported with transformer_engine implementation")

    def register_patches(self, patch_manager, args):
        if args.schedule_method == "interleaved_1f1b" and not args.combined_1f1b:
            args.schedule_method = None

        if args.schedule_method == "dualpipev":
            from megatron.training.utils import print_rank_0
            from dcu_megatron.core.pipeline_parallel.dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
            from dcu_megatron.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
                get_model,
                dualpipev_fp16forward,
                get_num_layers_to_build,
                train_step,
                _allreduce_embedding_grads_wrapper
            )
            from dcu_megatron.training.training import evaluate
            from dcu_megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            from dcu_megatron.training.utils import get_batch_on_this_tp_rank
            from dcu_megatron.training.training import build_train_valid_test_data_iterators_wrapper

            patch_manager.register_patch(
                'megatron.training.training.get_model', get_model)
            patch_manager.register_patch(
                'megatron.training.training.train_step', train_step)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                         forward_backward_pipelining_with_cutinhalf)
            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads._allreduce_embedding_grads', _allreduce_embedding_grads_wrapper)

            # use first rank
            patch_manager.register_patch(
                'megatron.training.training.evaluate', evaluate)

            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.get_transformer_layer_offset', get_transformer_layer_offset)

            # support dualpipev, two data iterators
            patch_manager.register_patch(
                'megatron.training.training.build_train_valid_test_data_iterators',
                build_train_valid_test_data_iterators_wrapper,
                apply_wrapper=True)

            # support dualpipev, broadcast loss_mask and labels
            patch_manager.register_patch(
                'megatron.training.utils.get_batch_on_this_tp_rank',
                get_batch_on_this_tp_rank)

        if args.schedule_method == "interleaved_1f1b":
            from dcu_megatron.core.pipeline_parallel.schedules import get_pp_rank_microbatches, forward_backward_pipelining_with_interleaving
            # num_warmup_microbatches + 1
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_pp_rank_microbatches',
                                        get_pp_rank_microbatches)

            # a2a_overlap
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                        forward_backward_pipelining_with_interleaving)

        if args.combined_1f1b:
            from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear

            from dcu_megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
            from dcu_megatron.core.transformer.transformer_layer import TransformerLayer
            from dcu_megatron.core.models.gpt.gpt_model import GPTModel
            from dcu_megatron.core.extensions.transformer_engine import (
                _get_extra_te_kwargs_wrapper,
                TELinear,
                TELayerNormColumnParallelLinear,
            )
            from dcu_megatron.core.transformer.multi_latent_attention import MLASelfAttention
            from dcu_megatron.core.transformer.attention import SelfAttention
            from dcu_megatron.core.transformer.mlp import MLP
            from dcu_megatron.core.transformer.moe.experts import TEGroupedMLP
            from dcu_megatron.core.transformer.moe.moe_layer import MoELayer

            patch_manager.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher',
                                        MoEAlltoAllTokenDispatcher)

            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer',
                                        TransformerLayer)

            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.build_schedule_plan',
                                        GPTModel.build_schedule_plan,
                                        create_dummy=True)

            # backward_dw
            patch_manager.register_patch('megatron.core.extensions.transformer_engine._get_extra_te_kwargs',
                                        _get_extra_te_kwargs_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('megatron.core.extensions.transformer_engine.TELinear',
                                        TELinear)
            patch_manager.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear',
                                        TELayerNormColumnParallelLinear)
            TEColumnParallelLinear.__bases__ = (TELinear,)
            TERowParallelLinear.__bases__ = (TELinear,)

            if is_te_min_version("1.9.0.dev0"):
                from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
                from dcu_megatron.core.extensions.transformer_engine import TEGroupedLinear

                patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear',
                                             TEGroupedLinear)
                TEColumnParallelGroupedLinear.__bases__ = (TEGroupedLinear,)
                TERowParallelGroupedLinear.__bases__ = (TEGroupedLinear,)

            patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MLASelfAttention.backward_dw',
                                        MLASelfAttention.backward_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.backward_dw',
                                        SelfAttention.backward_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.backward_qkv_dw',
                                        SelfAttention.backward_qkv_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.backward_proj_dw',
                                        SelfAttention.backward_proj_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.backward_dw',
                                        MLP.backward_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.backward_dw',
                                        TEGroupedMLP.backward_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.backward_dw',
                                        MoELayer.backward_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.backward_shared_expert_dw',
                                        MoELayer.backward_shared_expert_dw,
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.backward_routed_expert_dw',
                                        MoELayer.backward_routed_expert_dw,
                                        create_dummy=True)
