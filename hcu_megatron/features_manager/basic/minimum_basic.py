# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature


class MinimumBasicFeature(AbstractFeature):

    def __init__(self):
        super().__init__('minimum-basic', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
        group.add_argument('--world-size', type=int, default=8,
                        help='number of nodes for distributed training')
        group.add_argument('--dist-url', type=str, default="env://",
                        help='Which master node url for distributed training.')

    def register_patches(self, patch_manager, args):
        from hcu_megatron.training.arguments import parse_args
        from hcu_megatron.training.initialize import _initialize_distributed

        patch_manager.register_patch('megatron.training.arguments.parse_args', parse_args)
        # specify init_method
        patch_manager.register_patch('megatron.training.initialize._initialize_distributed',
                                    _initialize_distributed)
