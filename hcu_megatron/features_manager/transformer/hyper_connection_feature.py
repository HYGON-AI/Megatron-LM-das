# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature


class HyperConnectionFeature(AbstractFeature):

    def __init__(self):
        super().__init__('enable-hyper-connections')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-hyper-connections', action='store_true', default=False,
                           help='use gathered input of AGKernel for wgrad computation')
        group.add_argument('--mhc-use-tilekernels', action='store_true', default=False,
                           help='Whether to transpose weight when using flux kernel')
        group.add_argument('--num-residual-streams', type=int, default=4,
                           help='Number of residual streams (n in paper).')
        group.add_argument('--mhc-sinkhorn-iterations', type=int, default=1,
                           help='Number of Sinkhorn-Knopp iterations for doubly stochastic projection..')
        group.add_argument('--mhc-recompute-layer-num', type=int, default=1,
                           help='Number of layers per MHC recompute block.'
                                'When set, every `mhc_recompute_layer_num` layers form a recompute block. The last layer'
                                'in each recompute block (i.e., layer_number % mhc_recompute_layer_num == 0 or the final'
                                'layer in the transformer block) will:'
                                '- NOT checkpoint its final MLP BDA'
                                '- Register the unified recompute hook on its MLP BDA output'
                                '- A new CheckpointManager is created for subsequent layers'
                                'If None, all layers in the transformer block share a single recompute block.'
                                'Must be a positive integer when set.')
        group.add_argument('--mhc-init-gating-factor', type=float, default=0.01,
                           help='Initial value of Gating Factor (alpha in paper).')
        group.add_argument('--mhc-expand-emb', action='store_true', default=False,
                           help='Whether to expand the embedding dimension for mHC.')
        group.add_argument('--mhc-tau', type=float, default=0.05,
                           help='Number of residual streams (n in paper).')
        group.add_argument('--mhc-lite', action='store_true', default=False,
                           help='Number of residual streams (n in paper).')
        group.add_argument('--use-vwn', action='store_true', default=False,
                           help='If true, use vwn.')
        group.add_argument('--mhc-hres-vwnstyle', action='store_true', default=False,
                           help='If true, use mhc-hres-vwnstyle.')
        group.add_argument('--use-mhc-svd', action='store_true', default=False,
                           help='If true, use use-mhc-svd.')
        group.add_argument('--mhc-fuse-h-post-compute', action='store_true', default=False,
                           help='If true, use mhc-fuse-h-post-compute.')
        group.add_argument('--mhc-log-amax-per-step', type=int, default=1,
                           help='mhc-log-amax-per-step.')
        group.add_argument('--mhc-fix-muons', action='store_true', default=False,
                           help='mhc-fix-muons.')
        group.add_argument('--mhc-fuse-aggregate-compute', action='store_true', default=False,
                           help='mhc-fuse-aggregate-compute.')
        group.add_argument('--mhc-init-hpre-use-module-layer', action='store_true', default=False,
                           help='If true, use module-level layer index (2*layer + is_mlp) for h_pre initialization.')

    def register_patches(self, patch_manager, args):
        # flux
        from hcu_megatron.core.tensor_parallel.layers import (
            FluxColumnParallelLinear,
            FluxRowParallelLinear,
        )
        from hcu_megatron.core.transformer.transformer_block import TransformerBlock
        from hcu_megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
        from hcu_megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        from hcu_megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layer_specs,
            get_gpt_layer_with_flux_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )
        from hcu_megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving

        if args.enable_hyper_connections:
            patch_manager.register_patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_layer_specs",
                                        get_gpt_decoder_layer_specs)
            if args.parallel_linear_impl == 'flux':
                patch_manager.register_patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec",
                                            get_gpt_layer_with_flux_spec)
            else:
                patch_manager.register_patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec",
                                            get_gpt_layer_with_transformer_engine_spec)
            patch_manager.register_patch("megatron.core.transformer.transformer_layer.TransformerLayer.__call__",
                                        TransformerLayer.__call__,
                                        create_dummy=True)
            patch_manager.register_patch("megatron.core.transformer.transformer_layer.TransformerLayerSubmodules",
                                        TransformerLayerSubmodules)
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                        TransformerBlock.forward)
            patch_manager.register_cls_funcs('megatron.core.transformer.transformer_block.TransformerBlock',
                                        [TransformerBlock._build_mhc_recompute_layer_plan,
                                         TransformerBlock._finalize_mhc_recompute_layer,],
                                        create_dummy=True)
            patch_manager.register_patch('megatron.core.tensor_parallel.random.CheckpointWithoutOutput.__init__',
                                        CheckpointWithoutOutput.__init__)
            patch_manager.register_patch('megatron.core.tensor_parallel.random.CheckpointWithoutOutput.checkpoint',
                                        CheckpointWithoutOutput.checkpoint)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                        forward_backward_pipelining_without_interleaving)
