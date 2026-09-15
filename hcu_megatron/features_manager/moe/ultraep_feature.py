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

    def validate_args(self, args):
        if getattr(args, 'moe_enable_ultraep', False):
            assert getattr(args, 'moe_num_redundant_experts_per_rank', 0) > 0, (
                '--moe-num-redundant-experts-per-rank must be > 0 when '
                '--moe-enable-ultraep is set.'
            )
        return args

    def register_patches(self, patch_manager, args):
        if not getattr(args, 'moe_enable_ultraep', False):
            return

        from hcu_megatron.core.transformer.moe.moe_layer_ultraep import (
            moe_layer_ultraep_init_wrapper,
            moe_layer_ultraep_forward_wrapper,
        )
        from hcu_megatron.core.distributed.param_and_grad_buffer import (
            _param_and_grad_buffer_init_wrapper,
            _distributed_data_parallel_init_wrapper,
            _compute_full_param_layout_ultraep_wrapper,
        )
        from hcu_megatron.core.transformer.moe.experts import (
            te_grouped_linear_sharded_state_dict_ultraep_wrapper,
            teg_grouped_mlp_sharded_state_dict_ultraep_wrapper,
        )
        from hcu_megatron.training.checkpointing import generate_state_dict_ultraep_wrapper

        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
            moe_layer_ultraep_init_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.forward',
            moe_layer_ultraep_forward_wrapper,
            apply_wrapper=True,
        )
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
            te_grouped_linear_sharded_state_dict_ultraep_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.experts.TEGroupedMLP.sharded_state_dict',
            teg_grouped_mlp_sharded_state_dict_ultraep_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.training.checkpointing.generate_state_dict',
            generate_state_dict_ultraep_wrapper,
            apply_wrapper=True,
        )
