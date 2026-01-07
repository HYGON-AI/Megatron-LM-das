import os
from argparse import ArgumentParser

from ..feature import AbstractFeature


class CPUOffloadFeature(AbstractFeature):

    def __init__(self):
        super().__init__('fine-grained-activation-offloading', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--fine-grained-activation-offloading', action='store_true',
                           help='Offload the activation to CPU')
        group.add_argument('--offload-modules', nargs='*', type=str, default=None,
                           help='The submodules to offload. '
                           'choices: "attn_norm", "qkv_linear", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "expert_fc2", '
                           '         "shared_fc1", "shared_fc2", "moe_act".'
                           'default: ["core_attn"].'
                           '"attn_norm": offload the input of the normalization in the attention part. '
                           '"qkv_linear": offload the qkv_linear part of the transformer layer. '
                           '"core_attn": offload the core attention part of the transformer layer. '
                           '"attn_proj": offload the input of the attn linear projection part. '
                           '"mlp_norm": offload the input of the normalization in the mlp part. '
                           '"expert_fc1": offload the input of the expert fc1 part. '
                           '"expert_fc2": offload the input of the expert fc2 part. '
                           '"shared_fc1": offload the shared_fc1 part of the transformer layer. '
                           '"shared_fc2": offload the shared_fc2 part of the transformer layer. '
                           '"moe_act": offload the activation function part of the moe layer.')
        group.add_argument('--min-offloaded-tensor-size', type=int, default=1024*1024,
                            help='The minimum size of the tensor to be offloaded.')

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.models.gpt.gpt_model import GPTModel
        from dcu_megatron.core.transformer.attention import Attention
        from dcu_megatron.core.transformer.multi_latent_attention import MultiLatentAttention
        from dcu_megatron.core.transformer.moe.experts import TEGroupedMLP
        from dcu_megatron.core.transformer.mlp import MLP
        from dcu_megatron.core.transformer.transformer_layer import TransformerLayer
        from dcu_megatron.core.transformer.transformer_block import TransformerBlock
        from dcu_megatron.core.extensions.transformer_engine import te_module_init_wrapper
        from dcu_megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_wrapper
        from dcu_megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
        from dcu_megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        from dcu_megatron.core.models.gpt.fine_grained_callables import build_layer_callables_without_split_attn

        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.preprocess_for_fine_grained_offloading',
                                     GPTModel.preprocess_for_fine_grained_offloading,
                                     create_dummy=True)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                                     GPTModel.__init__)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.build_schedule_plan',
                                     GPTModel.build_schedule_plan)

        patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward',
                                     Attention.forward)
        patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MultiLatentAttention.forward',
                                     MultiLatentAttention.forward)

        patch_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
                                     TEGroupedMLP.forward)

        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                     forward_backward_pipelining_wrapper,
                                     apply_wrapper=True)

        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                                     MLP.forward)
        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                     forward_backward_pipelining_wrapper,
                                     apply_wrapper=True)

        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
                                     TransformerLayer._forward_attention)

        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                     TransformerBlock.forward)

        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.forward',
                                     MultiTokenPredictionBlock.forward)

        patch_manager.register_cls_funcs('megatron.core.tensor_parallel.random.CheckpointWithoutOutput',
                                         [CheckpointWithoutOutput.checkpoint,
                                          CheckpointWithoutOutput._recompute],
                                         create_dummy=True)

        patch_manager.register_patch('megatron.core.models.gpt.fine_grained_callables.build_layer_callables',
                                     build_layer_callables_without_split_attn)
