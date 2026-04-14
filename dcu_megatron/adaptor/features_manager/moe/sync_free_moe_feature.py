import warnings

from argparse import ArgumentParser

from ..feature import AbstractFeature


class SyncFreeMoeFeature(AbstractFeature):
    def __init__(self):
        super().__init__('sync-free-moe')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-sync-free-moe', action='store_true',
                           default=False,
                           dest='sync_free_moe',
                           help='use sync free moe')
        group.add_argument('--use-primus-topk-router', action='store_true', default=False,
                           help='Replace TopKRouter with PrimusTopKRouter')
        group.add_argument('--use-primus-moe-permute-fusion', action='store_true', default=False,
                           help='Patch TE and Megatron MoE with fused permutation implementations')
        group.add_argument('--use-primus-deepep', action='store_true', default=False,
                           help='Replace MoE token dispatcher with PrimusTurbo DeepEP implementation')
        group.add_argument('--use-primus-grouped-mlp', action='store_true', default=False,
                           help='use PrimusTurboGroupedMLP')
        group.add_argument('--use-primus-fused-act-with-probs', action='store_true', default=False,
                           help='use fused act with probs provided by primus turbo')

    def validate_args(self, args):
        if args.sync_free_moe and args.use_primus_deepep:
            assert args.tensor_model_parallel_size == 1, "tensor_model_parallel_size should be 1 when using primus deepep"

        if not args.sync_free_moe:
            if args.use_primus_topk_router:
                warnings.warn(f"use-primus-topk-router does not take effect when enable-sync-free-moe is not set")

            if args.use_primus_moe_permute_fusion:
                warnings.warn(f"use-primus-moe-permute-fusion does not take effect when enable-sync-free-moe is not set")

            if args.use_primus_deepep:
                warnings.warn(f"use-primus-deepep does not take effect when enable-sync-free-moe is not set")

            if args.use_primus_grouped_mlp:
                warnings.warn(f"uuse-primus-grouped-mlp does not take effect when enable-sync-free-moe is not set")

    def register_patches(self, patch_manager, args):
        if args.sync_free_moe:
            if args.use_primus_topk_router:
                from dcu_megatron.core.transformer.moe.router import PrimusTopKRouter

                patch_manager.register_patch("megatron.core.transformer.moe.router.TopKRouter.routing",
                                             PrimusTopKRouter.routing)

            if args.use_primus_moe_permute_fusion:
                from dcu_megatron.core.extensions.transformer_engine import (
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
                from dcu_megatron.core.transformer.moe.token_dispatcher import PrimusTurboDeepEPTokenDispatcher

                patch_manager.register_patch("megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher",
                                             PrimusTurboDeepEPTokenDispatcher)

            if args.use_primus_grouped_mlp:
                from dcu_megatron.core.extensions.transformer_engine_spec_provider import te_spec_provider_grouped_mlp_modules_wrapper

                patch_manager.register_patch("megatron.core.extensions.transformer_engine_spec_provider.TESpecProvider.grouped_mlp_modules",
                                             te_spec_provider_grouped_mlp_modules_wrapper,
                                             apply_wrapper=True)
