# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import warnings

from argparse import ArgumentParser

from ..feature import AbstractFeature


class SyncFreeMoeFeature(AbstractFeature):
    def __init__(self):
        super().__init__('sync-free-moe')
        self.all_sync_free_moe_params = None

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-sync-free-moe', action='store_true',
                           default=False,
                           dest='sync_free_moe',
                           help='use sync free moe')
        group.add_argument('--sync-free-moe-backend',
                           type=str, default='primus', choices=['deepep', 'primus'],
                           help='The backend to use for sync free moe')
        group.add_argument('--turbo-sync-free-moe-stage',
                           type=int, default=None, choices=[1, 2, 3],
                           help='Sync-Free MoE optimization levels provided by primus')
        group.add_argument('--use-primus-topk-router', action='store_true', default=False,
                           help='Replace TopKRouter with PrimusTopKRouter')
        group.add_argument('--use-primus-moe-permute-fusion', action='store_true', default=False,
                           help='Patch TE and Megatron MoE with fused permutation implementations')
        group.add_argument('--use-primus-deepep', action='store_true', default=False,
                           help='Replace MoE token dispatcher with PrimusTurbo DeepEP implementation')
        group.add_argument('--use-primus-grouped-gemm', action='store_true', default=False,
                           help='use PrimusTurboGroupedMLP')
        group.add_argument('--use-primus-fused-act-with-probs', action='store_true', default=False,
                           help='use fused act with probs provided by primus turbo')
        group.add_argument('--turbo-deepep-num-cu', type=int, default=32,
                           help='the number of CUs to use for Primus-Turbo DeepEP')
        group.add_argument('--turbo-deepep-use-comm-stream', action='store_true', default=False,
                           help='Primus-Turbo DeepEP will use an internal stream to dispatch/combine when enabled, '
                                'default used current_stream. Both set`sync_free_moe=True` and '
                                '`use_primus_deepep=True` first')

    def _get_sync_free_moe_options(self, args) -> dict:
        sync_free_moe_options = {
            1: {
                "use_primus_topk_router": True,
                "use_primus_moe_permute_fusion": True,
            },
            2: {
                "use_primus_topk_router": True,
                "use_primus_deepep": True,
                "use_primus_moe_permute_fusion": True,
                "use_primus_grouped_gemm": True,
            },
            3: {
                "use_primus_topk_router": True,
                "use_primus_deepep": True,
                "use_primus_moe_permute_fusion": True,
                "use_primus_grouped_gemm": True,
                "use_primus_fused_act_with_probs": True,
            },
        }
        self.all_sync_free_moe_params = list(sync_free_moe_options[3].keys())

        stage = args.turbo_sync_free_moe_stage

        if stage > 3 or stage < 0:
            raise ValueError("turbo_sync_free_moe_stage only support [0-3]")

        return sync_free_moe_options[stage]

    def validate_args(self, args):
        if not args.sync_free_moe:
            if (
                args.use_primus_topk_router
                or args.use_primus_moe_permute_fusion
                or args.use_primus_deepep
                or args.use_primus_grouped_gemm
                or args.use_primus_fused_act_with_probs
                or args.turbo_sync_free_moe_stage
            ):
                warnings.warn(f"parameters specific to sync free moe does not take effect when enable-sync-free-moe is not set.")

            return args

        if args.use_primus_fused_act_with_probs:
            if not args.use_primus_grouped_gemm:
                warnings.warn(f"use-primus-fused-act-with-probs does not take effect when use_primus_grouped_gemm is not set")

        # prioritize the use of turbo_sync_free_moe_stage
        if args.turbo_sync_free_moe_stage:
            options = self._get_sync_free_moe_options(args)
            for param in self.all_sync_free_moe_params:
                if param in options:
                    setattr(args, param, options[param])

                elif getattr(args, param, False):
                    warnings.warn(f"{param} is set to False when turbo_sync_free_moe_stage is {args.turbo_sync_free_moe_stage}")
                    setattr(args, param, False)

            return args

        if args.use_primus_fused_act_with_probs:
            args.turbo_sync_free_moe_stage = 3
        elif args.use_primus_deepep or args.use_primus_grouped_gemm:
            args.turbo_sync_free_moe_stage = 2
        elif args.use_primus_topk_router or args.use_primus_moe_permute_fusion:
            args.turbo_sync_free_moe_stage = 1

        if args.sync_free_moe_backend == "deepep":
            assert args.moe_token_dispatcher_type == "flex", "DeepEP backend is only supported with flex token dispatcher."
            assert args.moe_flex_dispatcher_backend == "deepep"
            assert args.use_primus_grouped_gemm, "--use-primus-grouped-gemm should be set when enabling sync free moe with deepep."
            assert not args.use_primus_deepep, "--use-primus-deepep should NOT be set when enabling sync free moe with deepep."

        if args.use_primus_deepep:
            assert args.moe_token_dispatcher_type == "flex", "Primus DeepEP backend is only supported with flex token dispatcher."

        return args

    def register_patches(self, patch_manager, args):
        args = self.validate_args(args)
        if args.sync_free_moe:
            if args.use_primus_topk_router:
                from hcu_megatron.core.transformer.moe.router import PrimusTopKRouter

                patch_manager.register_patch("megatron.core.transformer.moe.router.TopKRouter.routing",
                                             PrimusTopKRouter.routing)

            if args.use_primus_moe_permute_fusion:
                from hcu_megatron.core.extensions.transformer_engine import (
                    moe_permute,
                    moe_permute_with_probs,
                    moe_sort_chunks_by_index,
                    moe_sort_chunks_by_index_with_probs,
                    moe_unpermute,
                )

                patch_manager.register_patch("megatron.core.extensions.transformer_engine.fused_permute",
                                             moe_permute)
                patch_manager.register_patch("megatron.core.extensions.transformer_engine.fused_permute_with_probs",
                                             moe_permute_with_probs)
                patch_manager.register_patch("megatron.core.extensions.transformer_engine.fused_sort_chunks_by_index",
                                             moe_sort_chunks_by_index)
                patch_manager.register_patch("megatron.core.extensions.transformer_engine.fused_sort_chunks_by_index_with_probs",
                                             moe_sort_chunks_by_index_with_probs)
                patch_manager.register_patch("megatron.core.extensions.transformer_engine.fused_unpermute",
                                             moe_unpermute)

            if args.use_primus_deepep:
                from hcu_megatron.core.transformer.moe.token_dispatcher import PrimusTurboDeepEPTokenDispatcher

                patch_manager.register_patch("megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher",
                                             PrimusTurboDeepEPTokenDispatcher)

            if args.use_primus_grouped_gemm:
                from hcu_megatron.core.extensions.transformer_engine_spec_provider import te_spec_provider_grouped_mlp_modules_wrapper

                patch_manager.register_patch("megatron.core.extensions.transformer_engine_spec_provider.TESpecProvider.grouped_mlp_modules",
                                             te_spec_provider_grouped_mlp_modules_wrapper,
                                             apply_wrapper=True)

            if args.sync_free_moe_backend == "deepep":
                from hcu_megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

                patch_manager.register_patch("megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher",
                                            MoEFlexTokenDispatcher)
