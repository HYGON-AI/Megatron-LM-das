from argparse import ArgumentParser

from ..feature import AbstractFeature
from megatron.core.utils import is_te_min_version


class CPUOffloadFeature(AbstractFeature):

    def __init__(self):
        super().__init__('offload-activation', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--offload-activation', action='store_true',
                           help='Offload the activation to CPU')
        group.add_argument('--offload-modules', nargs='*', type=str, default=None,
                           help='The submodules to offload. '
                           'choices: "self_attn", "qkv_linear", "core_attn", "attn_linear", "router_fc1", "router_fc2", '
                           '         "shared_fc1", "shared_fc2".'
                           'default: ["core_attn"].'
                           '"self_attn": offload the self_attn part of the transformer layer. '
                           '"qkv_linear": offload the qkv_linear part of the transformer layer. '
                           '"core_attn": offload the core attention part of the transformer layer. '
                           '"attn_linear": offload the attn linear projection part of the transformer layer. '
                           '"router_fc1": offload the moe router_fc1 part of the transformer layer. '
                           '"router_fc2": offload the moe router_fc2 part of the transformer layer. '
                           '"shared_fc1": offload the shared_fc1 part of the transformer layer. '
                           '"shared_fc2": offload the shared_fc2 part of the transformer layer.')


    def validate_args(self, args):
        pass

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.models.gpt.gpt_model import gpt_model_forward_wrapper
        from dcu_megatron.core.transformer.attention import Attention
        from dcu_megatron.core.transformer.moe.experts import TEGroupedMLP
        from dcu_megatron.core.transformer.mlp import MLP
        from dcu_megatron.core.transformer.transformer_layer import TransformerLayer, transformer_layer_forward_wrapper

        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                     gpt_model_forward_wrapper,
                                     apply_wrapper=True)

        patch_manager.register_cls_funcs('megatron.core.transformer.attention.Attention',
                                         [Attention._offload_qkv_linear_forward,
                                          Attention._offload_core_attention_forward,
                                          Attention._offload_attn_linear_forward,],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward',
                                     Attention.forward)

        patch_manager.register_cls_funcs('megatron.core.transformer.moe.experts.TEGroupedMLP',
                                         [TEGroupedMLP._offload_router_fc1_forward,
                                          TEGroupedMLP._offload_router_fc2_forward],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
                                     TEGroupedMLP.forward)

        patch_manager.register_cls_funcs('megatron.core.transformer.mlp.MLP',
                                         [MLP._offload_shared_fc1_forward,
                                          MLP._offload_shared_fc2_forward],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                                     MLP.forward)

        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
                                     TransformerLayer._forward_attention)
        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.forward',
                                     transformer_layer_forward_wrapper,
                                     apply_wrapper=True)
        