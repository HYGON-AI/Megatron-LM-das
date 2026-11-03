# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import warnings

from argparse import ArgumentParser

from ..feature import AbstractFeature


class EchoExpertFeature(AbstractFeature):
    def __init__(self):
        super().__init__('moe-enable-echo')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-enable-echo', action='store_true',
                           default=False, dest='moe_enable_echo',
                           help='Enable echo-expert offloading planner. ')
        group.add_argument('--moe-num-echo-experts', type=int, default=None,
                           dest='moe_num_echo_experts',
                           help='Number of spare/echo experts (must be divisible by EP size).')
        group.add_argument('--moe-echo-expert-dispatch-type', type=str,
                           default='alltoall', choices=['alltoall', 'deepep'],
                           help='Communication backend for echo expert weights (not tokens). '
                                'and BF16 expert weights.')
        group.add_argument('--moe-echo-algorithm', type=str, default='sinkhorn',
                           choices=['sinkhorn', 'greedy', 'auction'],
                           dest='moe_echo_algorithm',
                           help='Assignment algorithm for offloading planner.')
        group.add_argument('--moe-echo-ranks-per-node', type=int, default=8,
                           dest='moe_echo_ranks_per_node',
                           help='Ranks per physical node (topology hint for '
                                'sinkhorn/auction).')
        group.add_argument('--moe-echo-dump-dir', type=str, default=None,
                           dest='moe_echo_dump_dir',
                           help='If set, rank-local per-layer offloading plans are '
                                'dumped here (every 100 steps).')
        group.add_argument('--moe-echo-expert-dispatch-overlap', action='store_true',
                           default=False, dest='moe_echo_expert_dispatch_overlap',
                           help='and home expert GEMM using a dedicated CUDA stream.')

    def validate_args(self, args):
        dispatch_type = getattr(args, 'moe_echo_expert_dispatch_type', 'alltoall')
        if dispatch_type not in ('alltoall', 'deepep'):
            raise ValueError(f'Unsupported echo expert dispatch type: {dispatch_type}')
        if not getattr(args, 'moe_enable_echo', False):
            echo_defaults = {
                'moe_num_echo_experts': None,
                'moe_echo_expert_dispatch_type': 'alltoall',
                'moe_echo_algorithm': 'sinkhorn',
                'moe_echo_ranks_per_node': 8,
                'moe_echo_dump_dir': None,
                'moe_echo_expert_dispatch_overlap': False,
            }
            changed = [
                name for name, default in echo_defaults.items()
                if getattr(args, name, default) != default
            ]
            if changed:
                flags = ', '.join(
                    '--' + name.replace('_', '-') for name in changed
                )
                warnings.warn(
                    f'{flags} have no effect without --moe-enable-echo'
                )
            return args

        if args.moe_num_echo_experts is None:
            raise ValueError('--moe-num-echo-experts must be set when --moe-enable-echo is on')
        if (getattr(args, 'gradient_accumulation_fusion', False)
                and getattr(args, 'delay_wgrad_compute', False)):
            raise ValueError(
                'Echo with gradient accumulation fusion requires immediate wgrad '
                'computation; disable --delay-wgrad-compute.'
            )
        ep_size = int(getattr(args, 'expert_model_parallel_size', 1) or 1)
        num_echo = int(args.moe_num_echo_experts)
        if num_echo <= 0 or num_echo % ep_size != 0:
            raise ValueError(
                f'--moe-num-echo-experts ({num_echo}) must be divisible '
                f'by expert_model_parallel_size ({ep_size})'
            )
        args.moe_num_echo_experts = num_echo
        if dispatch_type == 'deepep' and ep_size <= 1:
            raise ValueError('--moe-echo-expert-dispatch-type deepep requires expert_model_parallel_size > 1')

        # Phase 2a-lite: warn about --moe-router-force-load-balancing.
        # It replaces router logits with uniform random noise → no hot
        # experts → auction/sinkhorn/greedy will emit trivial (near-empty)
        # offloading plans and echo benchmarks become meaningless. We do NOT
        # auto-strip the flag (user chose "warn only").
        if getattr(args, 'moe_router_force_load_balancing', False):
            warnings.warn(
                '--moe-router-force-load-balancing is on together with --moe-enable-echo. '
                'The router will emit uniform-random routing, so the offloading planner '
                'will not see any hot experts and echo becomes a no-op. Consider dropping '
                '--moe-router-force-load-balancing to see auction/sinkhorn/greedy effects.'
            )
        return args

    def register_patches(self, patch_manager, args):
        if not getattr(args, 'moe_enable_echo', False):
            return

        from hcu_megatron.core.transformer.moe.moe_layer import (
            echo_forward_wrapper,
            moe_layer_init_echo_wrapper,
        )

        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
            moe_layer_init_echo_wrapper,
            apply_wrapper=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_layer.MoELayer.forward',
            echo_forward_wrapper,
            apply_wrapper=True,
        )
