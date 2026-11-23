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
                           type=str, default=None, choices=['deepep', 'turbo'],
                           help='The backend to use for sync free moe')
        group.add_argument('--turbo-sync-free-moe-stage',
                           type=int, default=2, choices=[1, 2, 3],
                           help='Sync-Free MoE optimization levels provided by primus')
        group.add_argument('--use-turbo-topk-router', action='store_true', default=False,
                           help='Replace TopKRouter with PrimusTopKRouter')
        group.add_argument('--use-turbo-moe-permute-fusion', action='store_true', default=False,
                           help='Patch TE and Megatron MoE with fused permutation implementations')
        group.add_argument('--use-turbo-grouped-gemm', action='store_true', default=False,
                           help='use PrimusGroupedMLP')
        group.add_argument('--use-turbo-fused-act-with-probs', action='store_true', default=False,
                           help='use fused act with probs provided by primus turbo')
        group.add_argument('--turbo-deepep-use-comm-stream', action='store_true', default=False,
                           help='Primus-Turbo DeepEP will use an internal stream to dispatch/combine when enabled, '
                                'default used current_stream. Both set`sync_free_moe=True` and '
                                '`sync_free_moe_backend=turbo` first')
        group.add_argument('--use-turbo-op-fuser', action='store_true', default=False,
                           help='use operation fuser API provided by turbo to enable advanced fusions')

    def _get_sync_free_moe_options(self, args) -> dict:
        sync_free_moe_options = {
            1: {
                "use_turbo_topk_router": True,
                "use_turbo_moe_permute_fusion": True,
            },
            2: {
                "use_turbo_topk_router": True,
                "use_turbo_moe_permute_fusion": True,
                "use_turbo_grouped_gemm": True,
            },
            3: {
                "use_turbo_topk_router": True,
                "use_turbo_moe_permute_fusion": True,
                "use_turbo_grouped_gemm": True,
                "use_turbo_fused_act_with_probs": True,
            },
        }
        self.all_sync_free_moe_params = list(sync_free_moe_options[3].keys())

        stage = args.turbo_sync_free_moe_stage

        if stage > 3 or stage < 0:
            raise ValueError("turbo_sync_free_moe_stage only support [0-3]")

        return sync_free_moe_options[stage]

    def pre_validate_args(self, args):
        if not args.sync_free_moe:
            return args

        if (
            args.use_turbo_topk_router
            or args.use_turbo_moe_permute_fusion
            or args.use_turbo_grouped_gemm
            or args.use_turbo_fused_act_with_probs
        ):
            warnings.warn("values of turbo-related parameters change with turbo_sync_free_moe_stage; "
                          "directly setting turbo-related parameters may be ineffective.")

        # prioritize the use of turbo_sync_free_moe_stage
        options = self._get_sync_free_moe_options(args)
        if args.turbo_sync_free_moe_stage > 1:
            assert args.sync_free_moe_backend is not None, "sync_free_moe_backend should be specified when turbo_sync_free_moe_stage > 1"

        for param in self.all_sync_free_moe_params:
            if param in options:
                setattr(args, param, options[param])

            elif getattr(args, param, False):
                warnings.warn(f"{param} is set to False when turbo_sync_free_moe_stage is {args.turbo_sync_free_moe_stage}")
                setattr(args, param, False)

        return args

    def validate_args(self, args):
        if args.sync_free_moe:
            if args.fp8 or args.fp4:
                assert args.moe_router_padding_for_quantization, "moe_router_padding_for_quantization should be True if using fp8/fp4."

        return args

    def register_patches(self, patch_manager, args):
        args = self.pre_validate_args(args)
        if args.sync_free_moe:
            if args.use_turbo_topk_router:
                from hcu_megatron.core.transformer.moe.router import PrimusTopKRouter

                patch_manager.register_patch("megatron.core.transformer.moe.router.TopKRouter.routing",
                                             PrimusTopKRouter.routing)

            if args.use_turbo_moe_permute_fusion:
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

            if args.turbo_sync_free_moe_stage > 1 and args.sync_free_moe_backend == "turbo":
                from hcu_megatron.core.transformer.moe.token_dispatcher import PrimusTurboDeepEPTokenDispatcher

                patch_manager.register_patch("megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher",
                                             PrimusTurboDeepEPTokenDispatcher)

            if args.use_turbo_grouped_gemm:
                from hcu_megatron.core.extensions.transformer_engine_spec_provider import te_spec_provider_grouped_mlp_modules_wrapper
                from hcu_megatron.core.full_cuda_graph import FullCudaGraphWrapper

                patch_manager.register_patch("megatron.core.extensions.transformer_engine_spec_provider.TESpecProvider.grouped_mlp_modules",
                                             te_spec_provider_grouped_mlp_modules_wrapper,
                                             apply_wrapper=True)
                patch_manager.register_patch('megatron.core.full_cuda_graph.FullCudaGraphWrapper.__call__',
                                            FullCudaGraphWrapper.__call__)

            if args.turbo_sync_free_moe_stage > 1 and args.sync_free_moe_backend == "deepep":
                from hcu_megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

                patch_manager.register_patch("megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher",
                                            MoEFlexTokenDispatcher)
