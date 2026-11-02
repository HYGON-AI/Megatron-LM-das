# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from functools import wraps

from ..feature import AbstractFeature


def _get_param_groups_ultraep_wrapper(original_func):
    """Exclude temporary UltraEP replica parameters from optimizer groups."""

    @wraps(original_func)
    def wrapper(model_chunks, config, config_overrides):
        replica_params = []
        for model_chunk in model_chunks:
            for param in model_chunk.parameters():
                if getattr(param, 'is_eplb_replica', False) and param.requires_grad:
                    param.requires_grad_(False)
                    replica_params.append(param)
        try:
            return original_func(model_chunks, config, config_overrides)
        finally:
            for param in replica_params:
                param.requires_grad_(True)

    return wrapper


class UltraEPFeature(AbstractFeature):
    def __init__(self):
        super().__init__('ultraep', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='UltraEP')
        group.add_argument(
            '--moe-enable-ultraep',
            action='store_true',
            default=False,
            dest='moe_enable_ultraep',
            help='Enable UltraEP expert-parallelism load balancing.',
        )
        group.add_argument(
            '--moe-num-redundant-experts-per-rank',
            type=int,
            default=0,
            dest='moe_num_redundant_experts_per_rank',
            help='Number of redundant (replica) experts per EP rank for UltraEP.',
        )
        group.add_argument(
            '--moe-ultraep-autotune',
            action='store_true',
            default=False,
            dest='moe_ultraep_autotune',
            help=(
                'One-shot tuning of unlocked UltraEP Weight Sync launch parameters '
                'and Grad Reduce SMS using real training iterations.'
            ),
        )
        group.add_argument(
            '--moe-ultraep-autotune-start-iteration',
            type=int,
            default=3,
            dest='moe_ultraep_autotune_start_iteration',
            help='Process-local training iteration at which UltraEP autotuning starts.',
        )
        group.add_argument(
            '--moe-ultraep-autotune-grad-reduce-max-sms',
            type=int,
            default=None,
            dest='moe_ultraep_autotune_grad_reduce_max_sms',
            help=(
                'Optional even upper bound for Grad Reduce SMS candidates; the '
                'autotuner still reserves its configured device headroom.'
            ),
        )
        group.add_argument('--moe-ultraep-adaptive', action='store_true',
                           help=('Compare complete training steps with UltraEP on/off; '
                                 'ignored when UltraEP is disabled.'))
        group.add_argument('--moe-ultraep-adaptive-samples', type=int, default=3,
                           help='Odd number of valid timed steps per mode (>= 3).')
        group.add_argument('--moe-ultraep-adaptive-max-samples', type=int, default=5,
                           help='Maximum odd sample count per mode when paired comparisons disagree.')
        group.add_argument('--moe-ultraep-adaptive-min-speedup', type=float, default=0.01,
                           help='Minimum fractional step-time improvement needed to enable UltraEP.')
        group.add_argument('--moe-ultraep-adaptive-max-calibration-steps', type=int, default=48,
                           help='Total local step budget, including warmup, tuning and invalid steps.')

    def validate_args(self, args):
        ultraep_enabled = getattr(args, 'moe_enable_ultraep', False)
        autotune_enabled = getattr(args, 'moe_ultraep_autotune', False)
        adaptive_enabled = getattr(args, 'moe_ultraep_adaptive', False)
        if adaptive_enabled and not ultraep_enabled:
            args.moe_ultraep_adaptive = False
            adaptive_enabled = False
        if ultraep_enabled:
            assert getattr(args, 'moe_num_redundant_experts_per_rank', 0) > 0, (
                '--moe-num-redundant-experts-per-rank must be > 0 when '
                '--moe-enable-ultraep is set.'
            )
        if autotune_enabled:
            assert ultraep_enabled, (
                '--moe-ultraep-autotune requires --moe-enable-ultraep.'
            )
            assert getattr(args, 'moe_ultraep_autotune_start_iteration', 3) > 0, (
                '--moe-ultraep-autotune-start-iteration must be positive.'
            )
            max_sms = getattr(
                args, 'moe_ultraep_autotune_grad_reduce_max_sms', None
            )
            assert max_sms is None or (max_sms > 0 and max_sms % 2 == 0), (
                '--moe-ultraep-autotune-grad-reduce-max-sms must be a positive '
                'even number.'
            )
            cuda_graph_impl = getattr(args, 'cuda_graph_impl', None)
            assert cuda_graph_impl in (None, 'none'), (
                '--moe-ultraep-autotune currently requires CUDA graph capture '
                'to be disabled because launch parameters change during calibration.'
            )
            assert not getattr(args, 'optimizer_cuda_graph', False), (
                '--moe-ultraep-autotune currently requires '
                '--optimizer-cuda-graph to be disabled.'
            )
        if adaptive_enabled:
            from hcu_megatron.training.ultraep_adaptive import AdaptiveConfig

            AdaptiveConfig(
                samples=getattr(args, 'moe_ultraep_adaptive_samples', 3),
                max_samples=getattr(args, 'moe_ultraep_adaptive_max_samples', 5),
                min_speedup=getattr(args, 'moe_ultraep_adaptive_min_speedup', 0.01),
                max_calibration_steps=getattr(args, 'moe_ultraep_adaptive_max_calibration_steps', 48),
            )
            assert not getattr(args, 'overlap_moe_expert_parallel_comm', False), (
                'UltraEP adaptive currently requires EP overlap to be disabled.'
            )
            assert all(getattr(args, name, 1) == 1 for name in (
                'context_parallel_size', 'expert_tensor_parallel_size',
            )), 'UltraEP adaptive currently requires CP1 / ETP1.'
            assert getattr(args, 'moe_token_dispatcher_type', 'alltoall') == 'alltoall', (
                'UltraEP adaptive currently requires the alltoall dispatcher.'
            )
            assert getattr(args, 'transformer_impl', 'transformer_engine') == 'transformer_engine', (
                'UltraEP adaptive currently requires Transformer Engine experts.'
            )
            assert getattr(args, 'moe_grouped_gemm', True) and getattr(args, 'swiglu', True), (
                'UltraEP adaptive currently requires SwiGLU grouped experts.'
            )
            assert not getattr(args, 'use_transformer_engine_op_fuser', False), (
                'UltraEP adaptive does not yet support the TE op fuser.'
            )
            assert getattr(args, 'bf16', True) and not getattr(args, 'add_bias_linear', False), (
                'UltraEP adaptive currently requires bias-free BF16 experts.'
            )
            assert not getattr(args, 'fp8', None) and not getattr(args, 'fp4', None), (
                'UltraEP adaptive currently requires BF16 experts.'
            )
            assert getattr(args, 'gradient_accumulation_fusion', True), (
                'UltraEP adaptive requires fused accumulation into replica main_grad buffers.'
            )
            assert getattr(args, 'cuda_graph_impl', 'none') in (None, 'none') and not getattr(
                args, 'optimizer_cuda_graph', False
            ), 'UltraEP adaptive requires CUDA graph capture to be disabled.'
            assert not any(getattr(args, name, None) for name in (
                'recompute_granularity', 'delay_wgrad_compute', 'moe_shared_expert_overlap',
                'moe_latent_size', 'mtp_num_layers',
            )), (
                'UltraEP adaptive does not yet support recompute, delayed wgrad, '
                'shared-expert overlap, latent MoE or MTP.'
            )
        return args

    def register_patches(self, patch_manager, args):
        if not getattr(args, 'moe_enable_ultraep', False):
            # Adaptor arguments are parsed separately from training arguments.
            args.moe_ultraep_adaptive = False
            return

        from hcu_megatron.core.distributed.param_and_grad_buffer import (
            _param_and_grad_buffer_init_wrapper,
            _distributed_data_parallel_init_wrapper,
            _compute_full_param_layout_ultraep_wrapper,
        )
        from hcu_megatron.core.transformer.moe.experts import (
            te_grouped_linear_sharded_state_dict_wrapper,
            te_grouped_mlp_sharded_state_dict_wrapper,
        )
        from hcu_megatron.training.checkpointing import generate_state_dict_ultraep_wrapper
        from hcu_megatron.core.distributed.distributed_data_parallel import ultraep_backward_post_hook_wrapper

        patch_manager.register_patch(
            'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
            _param_and_grad_buffer_init_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
            _distributed_data_parallel_init_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
            ultraep_backward_post_hook_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.compute_full_param_layout',
            _compute_full_param_layout_ultraep_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.optimizer._get_param_groups',
            _get_param_groups_ultraep_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.extensions.transformer_engine.TEGroupedLinear._sharded_state_dict_grouped',
            te_grouped_linear_sharded_state_dict_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.experts.TEGroupedMLP.sharded_state_dict',
            te_grouped_mlp_sharded_state_dict_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.training.checkpointing.generate_state_dict',
            generate_state_dict_ultraep_wrapper,
            apply_wrapper=True,
        )
        if getattr(args, 'moe_ultraep_autotune', False) or getattr(args, 'moe_ultraep_adaptive', False):
            from hcu_megatron.training.ultraep_autotune import (
                ultraep_autotune_train_step_wrapper,
            )

            patch_manager.register_patch(
                'hcu_megatron.training.training.train_step',
                ultraep_autotune_train_step_wrapper,
                apply_wrapper=True,
            )
