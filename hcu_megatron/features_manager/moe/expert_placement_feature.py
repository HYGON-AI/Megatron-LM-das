# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import warnings
from argparse import ArgumentParser

from ..feature import AbstractFeature


class ExpertPlacementFeature(AbstractFeature):
    """Correlated expert placement from a placement.json (see expert_placement.py)."""

    def __init__(self):
        super().__init__('expert-placement', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='expert placement')
        group.add_argument(
            '--moe-expert-placement-path',
            type=str,
            default=None,
            dest='moe_expert_placement_path',
            help='placement.json from fit_placement.py: per-layer logical expert -> EP rank assignment. '
                 'Checkpoints keep logical expert order, so placements can change between save and load.',
        )
        group.add_argument(
            '--moe-dedup-alltoall',
            action='store_true',
            default=False,
            dest='moe_dedup_alltoall',
            help='With --moe-token-dispatcher-type alltoall, send each token once per destination EP rank '
                 'instead of once per selected expert (dedup_token_dispatcher.py).',
        )

    def validate_args(self, args):
        if getattr(args, 'moe_dedup_alltoall', False):
            assert args.moe_token_dispatcher_type == 'alltoall', (
                '--moe-dedup-alltoall requires --moe-token-dispatcher-type alltoall'
            )
            assert not getattr(args, 'moe_pad_expert_input_to_capacity', False), (
                '--moe-dedup-alltoall does not support --moe-pad-expert-input-to-capacity'
            )
            assert getattr(args, 'cuda_graph_impl', 'none') in (None, 'none'), (
                '--moe-dedup-alltoall does not support CUDA graphs'
            )
        if not getattr(args, 'moe_expert_placement_path', None):
            return args
        assert getattr(args, 'num_experts', None), '--moe-expert-placement-path requires an MoE model'
        for flag, name in (('moe_enable_ultraep', '--moe-enable-ultraep'), ('mega_moe', 'MegaMoE')):
            assert not getattr(args, flag, False), f'--moe-expert-placement-path is incompatible with {name}'
        if args.save or args.load:
            assert args.ckpt_format == 'torch_dist', (
                '--moe-expert-placement-path requires --ckpt-format torch_dist '
                '(other formats store experts by physical slot)'
            )
            if args.use_distributed_optimizer:
                # dp_reshardable stores optimizer buckets in physical order and would silently
                # mismatch weights after a placement change.
                assert args.dist_ckpt_optim_fully_reshardable, (
                    '--moe-expert-placement-path requires --dist-ckpt-optim-fully-reshardable'
                )
        if args.moe_token_dispatcher_type != 'flex' and not getattr(args, 'moe_dedup_alltoall', False):
            warnings.warn(
                f'--moe-token-dispatcher-type {args.moe_token_dispatcher_type} sends one row per '
                '(token, expert); expert placement only reduces traffic with a deduplicating '
                'dispatcher such as flex/DeepEP or --moe-dedup-alltoall'
            )
        # Group-limited routing (moe_router_num_groups > 1) is checked per layer when the placement
        # is loaded: every logical group must stay on the EP ranks it occupies without a placement.
        return args

    def register_patches(self, patch_manager, args):
        if getattr(args, 'moe_dedup_alltoall', False):
            from hcu_megatron.core.transformer.moe.dedup_token_dispatcher import (
                MoEDedupAlltoAllTokenDispatcher,
            )

            # Replaces the class in every module that imported it (moe_layer, hcu moe_layer).
            patch_manager.register_patch(
                'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher',
                MoEDedupAlltoAllTokenDispatcher,
            )
        if not getattr(args, 'moe_expert_placement_path', None):
            return

        from hcu_megatron.core.transformer.moe.expert_placement import (
            moe_layer_route_placement_wrapper,
            moe_layer_set_layer_number_placement_wrapper,
            moe_layer_sharded_state_dict_placement_wrapper,
        )

        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.set_layer_number',
            moe_layer_set_layer_number_placement_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.route',
            moe_layer_route_placement_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.sharded_state_dict',
            moe_layer_sharded_state_dict_placement_wrapper,
            apply_wrapper=True,
        )
        if getattr(args, 'use_bridge', False) and getattr(args, 'load_weights', False):
            from hcu_megatron.core.transformer.moe.expert_placement import (
                bridge_to_megatron_provider_placement_wrapper,
            )

            # Bridge imports HF expert i into local slot i; move experts to their placement slots
            # right after the import, before DDP and the optimizer copy the parameters.
            patch_manager.register_patch(
                'megatron.bridge.models.conversion.auto_bridge.AutoBridge.to_megatron_provider',
                bridge_to_megatron_provider_placement_wrapper,
                apply_wrapper=True,
            )
