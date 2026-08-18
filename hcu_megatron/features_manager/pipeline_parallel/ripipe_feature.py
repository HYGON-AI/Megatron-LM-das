# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature


class RiPipeFeature(AbstractFeature):

    def __init__(self):
        super().__init__('schedule-method')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='ripipe')
        group.add_argument('--recompute-in-bubble', action='store_true',
                           help='use bubble to do recompute to reduce memory')
        group.add_argument('--recompute-in-advance', action='store_true',
                           help='recompute early to reduce bubble and improve training.')

    def pre_validate_args(self, args):
        return args

    def validate_args(self, args):
        # Ensure recompute_in_bubble and recompute_in_advance are not both enabled.
        if getattr(args, 'recompute_in_bubble', False) and getattr(args, 'recompute_in_advance', False):
            raise AssertionError('--recompute-in-bubble and --recompute-in-advance cannot be enabled at the same time')

        if getattr(args, 'recompute_in_bubble', False) or getattr(args, 'recompute_in_advance', False):
            assert args.schedule_method == "ripipe", "--recompute-in-bubble or --recompute-in-advance requires --schedule-method=ripipe"
        
        # Feature-specific validation and argument adjustments.
        if getattr(args, 'recompute_in_advance', False):
            # Check incompatible features.
            incompatible_features_for_advance = []
            if getattr(args, 'recompute_method', None) == "uniform":
                incompatible_features_for_advance.append("uniform recompute")
            if getattr(args, 'selective_recompute', False):
                incompatible_features_for_advance.append("selective recompute")
            if getattr(args, 'adaptive_selective_recompute', False):
                incompatible_features_for_advance.append("adaptive selective recompute")
            if getattr(args, 'no_align_grad_reduce', False):
                incompatible_features_for_advance.append("no-align-grad-reduce")
            if getattr(args, 'no_overlap_p2p_communication', False):
                incompatible_features_for_advance.append("no-overlap-p2p-communication")
            
            if incompatible_features_for_advance:
                incompatible_str = ", ".join(incompatible_features_for_advance)
                raise AssertionError(f'--recompute-in-advance is not compatible with {incompatible_str}')
            
            # Feature-specific validation.
            if getattr(args, 'recompute_method', None) == "uniform":
                raise AssertionError('recompute_in_advance does not support uniform recompute_method')
            if not getattr(args, 'recompute_num_layers', None):
                raise AssertionError('recompute_num_layers can not be None or 0 when using recompute_in_advance')
            if getattr(args, 'pipeline_model_parallel_size', 1) <= 1 or getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is None:
                raise AssertionError('recompute_in_advance only support pipelining with interleaving')
            if getattr(args, 'num_layers_per_virtual_pipeline_stage', 1) != 1:
                args.recompute_in_advance = False
            # Ensure recompute_granularity is set to "full" for riPipe to work properly
            args.recompute_granularity = "full"
            args.recompute_method = "block"
            # Ensure recompute_modules includes 'mlp' for riPipe to work properly
            if getattr(args, 'recompute_modules', None) is None:
                args.recompute_modules = ['mlp']
            elif 'mlp' not in args.recompute_modules:
                args.recompute_modules.append('mlp')
            
        if getattr(args, 'recompute_in_bubble', False):
            # Check incompatible features.
            incompatible_features_for_bubble = []
            if getattr(args, 'recompute_method', None) == "uniform":
                incompatible_features_for_bubble.append("uniform recompute")
            if getattr(args, 'recompute_method', None) == "block":
                incompatible_features_for_bubble.append("block recompute")
            if getattr(args, 'selective_recompute', False):
                incompatible_features_for_bubble.append("selective recompute")
            if getattr(args, 'adaptive_selective_recompute', False):
                incompatible_features_for_bubble.append("adaptive selective recompute")
            if getattr(args, 'no_align_grad_reduce', False):
                incompatible_features_for_bubble.append("no-align-grad-reduce")
            if getattr(args, 'no_overlap_p2p_communication', False):
                incompatible_features_for_bubble.append("no-overlap-p2p-communication")
            if getattr(args, 'swap_attention', False):
                incompatible_features_for_bubble.append("swap-attention")
            # Check MoE-related features.
            if hasattr(args, 'recompute_modules') and args.recompute_modules:
                if 'moe_act' in args.recompute_modules:
                    incompatible_features_for_bubble.append("--moe-adaptive-recompute-activation")
                if 'experts' in args.recompute_modules or 'moe' in args.recompute_modules:
                    incompatible_features_for_bubble.append("--moe-layer-recompute")
            
            if incompatible_features_for_bubble:
                incompatible_str = ", ".join(incompatible_features_for_bubble)
                raise AssertionError(f'--recompute-in-bubble is not compatible with {incompatible_str}')
            
            # Feature-specific validation.
            if getattr(args, 'recompute_num_layers', None):
                raise AssertionError('recompute_num_layers must be None or 0 when using recompute_in_bubble')
            if getattr(args, 'pipeline_model_parallel_size', 1) <= 1 or getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is None:
                raise AssertionError('recompute_in_bubble only support pipelining with interleaving')
            if not getattr(args, 'prefetch', False):
                # Following is a trick to realize bubble recomputation. We first enable all recomputation,
                # and then disable recomputation for all layers except the ones chosen for bubble recomputation.
                args.recompute_granularity = "full"
                args.recompute_method = "block"
            if getattr(args, 'enable_recompute_layers_per_pp_rank', False):
                args.recompute_num_layers = args.num_layers // args.pipeline_model_parallel_size
            else:
                args.recompute_num_layers = args.num_layers_per_virtual_pipeline_stage
            # Ensure recompute_modules includes 'mlp' for riPipe to work properly
            if getattr(args, 'recompute_modules', None) is None:
                args.recompute_modules = ['mlp']
            elif 'mlp' not in args.recompute_modules:
                args.recompute_modules.append('mlp')

        return args

    def register_patches(self, patch_manager, args):
        # Handle ripipe-specific patches
        if args.schedule_method == "ripipe":
            # Only register the patch if either recompute_in_advance or recompute_in_bubble is enabled
            if getattr(args, 'recompute_in_advance', False) or getattr(args, 'recompute_in_bubble', False):
                from hcu_megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
                patch_manager.register_patch(
                    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                    DistributedDataParallel._make_backward_post_hook)
