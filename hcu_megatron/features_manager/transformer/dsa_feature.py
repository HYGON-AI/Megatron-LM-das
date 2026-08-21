# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from hcu_megatron.features_manager.feature import AbstractFeature


class DSAFeature(AbstractFeature):

    def __init__(self):
        super().__init__('dsa', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        pass

    def register_patches(self, patch_manager, args):
        from hcu_megatron.core.transformer.experimental_attention_variant.dsa import (
            rotate_activation,
            get_hadamard_matrix,
            hadamard_transform_optimized,
            _unpack_thd_to_sbh,
            _pack_sbh_to_thd,
            DSAIndexer,
            DSAttention,
        )

        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.get_hadamard_matrix',
            get_hadamard_matrix,
            create_dummy=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform_optimized',
            hadamard_transform_optimized,
            create_dummy=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.rotate_activation',
            rotate_activation,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa._unpack_thd_to_sbh',
            _unpack_thd_to_sbh,
            create_dummy=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa._pack_sbh_to_thd',
            _pack_sbh_to_thd,
            create_dummy=True,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer.forward_before_topk',
            DSAIndexer.forward_before_topk,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer.forward_with_scores',
            DSAIndexer.forward_with_scores,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.DSAttention.forward',
            DSAttention.forward,
        )