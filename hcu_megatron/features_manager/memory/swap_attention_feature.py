# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature
from megatron.core.utils import is_te_min_version


class SwapAttentionFeature(AbstractFeature):

    def __init__(self):
        super().__init__('swap-attention', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--swap-attention', action='store_true', default=False,
                           help='switch to open swap-attention feature.'
                                'The default is False.')
        # input_layernorm,self_attention,post_attention_norm
        group.add_argument('--swap-modules', type=str, default="self_attention",
                           help='Swap modules for model. Can be used together with "--swap-attention."')
        group.add_argument('--specify-layers', type=str, default=None,
                           help='Specify the swap layer. Can be used together with "--swap-attention."eg"0, 2, 4, 6"')
        group.add_argument('--reduce-recompute-for-last-chunk', action='store_true', default=False,
                           help='with recompute, now default False')


    def validate_args(self, args):
        adaptive_recompute_device_size = getattr(args, 'adaptive-recompute-device-size', -1)
        adaptive_recompute_device_swap = getattr(args, 'adaptive-recompute-device-swap', False)
        if (adaptive_recompute_device_size > 0 or adaptive_recompute_device_swap) and args.swap_attention:
            raise AssertionError('adaptive selective recompute is not compatible with swap_attention feature')

        self.incompatible_check(args, 'adaptive_memory_optimization')
        is_enable_lora = hasattr(args, "lora_target_modules") and args.lora_target_modules
        if is_enable_lora:
            raise AssertionError('swap attention is not compatible with LoRA')

        return args

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            if hasattr(args, "use_mcore_models") and args.use_mcore_models:
                if not is_te_min_version("2.5.0") and hasattr(args, "overlap_grad_reduce") and args.overlap_grad_reduce:
                    raise AssertionError("With overlap_grad_reduce must have at least transformer-engine version of 2.5.0")
            from hcu_megatron.core.memory.swap_attention.adaptor_swap_atten import allowed_recomputing_swap_module_wrapper
            from megatron.legacy.model.transformer import ParallelTransformerLayer
            from megatron.core.transformer.transformer_layer import TransformerLayer

            if hasattr(args, "use_legacy_models") and not args.use_legacy_models:
                allowed_recomputing_swap_module_wrapper(TransformerLayer)
            else:
                allowed_recomputing_swap_module_wrapper(ParallelTransformerLayer)
            from hcu_megatron.core.memory.swap_attention.adaptor_swap_atten import setup_model_and_optimizer_wrapper
            patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)

            from hcu_megatron.core.memory.common import linear_forward_main_grad_wrapper, linear_backward_main_grad_wrapper
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                                linear_forward_main_grad_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                linear_backward_main_grad_wrapper)
