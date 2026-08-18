# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature


class ParallelLinearFeature(AbstractFeature):
    def __init__(self):
        super().__init__('parallel-linear-impl')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--parallel-linear-impl', type=str,
                           default=None,
                           choices=['flux'],
                           help='Specify the method to replace ColumnParallelLinear/RowParallelLinear')
        group.add_argument('--save-flux-gather-input', action='store_true', default=False,
                           help='use gathered input of AGKernel for wgrad computation')
        group.add_argument('--flux-transpose-weight', action='store_true', default=False,
                           help='Whether to transpose weight when using flux kernel')
        group.add_argument('--disable-bw-flux-gemmrs-op', action='store_false', default=True, dest='enable_bw_flux_gemmrs_op',
                           help='Do not use flux.GemmRS in backward pass')

    def validate_args(self, args):
        if args.parallel_linear_impl == "flux" and args.transformer_impl != 'transformer_engine':
            raise AssertionError('flux is only supported with transformer_engine implementation')
        return args

    def register_patches(self, patch_manager, args):
        # flux
        from hcu_megatron.core.tensor_parallel.layers import (
            FluxColumnParallelLinear,
            FluxRowParallelLinear
        )
        from hcu_megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_flux_spec

        if args.parallel_linear_impl == 'flux':
            patch_manager.register_patch("megatron.core.extensions.transformer_engine.TEColumnParallelLinear",
                                         FluxColumnParallelLinear)
            patch_manager.register_patch("megatron.core.extensions.transformer_engine.TERowParallelLinear",
                                         FluxRowParallelLinear)
            patch_manager.register_patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec",
                                         get_gpt_layer_with_flux_spec)
