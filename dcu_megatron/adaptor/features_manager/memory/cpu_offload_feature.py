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

    def validate_args(self, args):
        if args.fine_grained_activation_offloading:
            if args.schedule_method == "dualpipev":
                assert os.environ.get("USE_DUALPIPEV_SCHEDULE", 0), "USE_DUALPIPEV_SCHEDULE should be set to 1"
            elif args.schedule_method == "interleaved_1f1b":
                assert not os.environ.get("USE_DUALPIPEV_SCHEDULE", 0), "USE_DUALPIPEV_SCHEDULE should be set to 0"
            else:
                raise ValueError(f"schedule_method should be dualpipev or interleaved_1f1b")

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.models.gpt.gpt_model import GPTModel
        from dcu_megatron.core.transformer.attention import Attention
        from dcu_megatron.core.transformer.multi_latent_attention import MultiLatentAttention
        from dcu_megatron.core.transformer.moe.experts import TEGroupedMLP
        from dcu_megatron.core.transformer.mlp import MLP
        from dcu_megatron.core.transformer.transformer_layer import TransformerLayer
        from dcu_megatron.core.transformer.transformer_block import TransformerBlock
        from dcu_megatron.core.extensions.transformer_engine import te_module_init_wrapper
        from dcu_megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving_wrapper

        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.initialize_model_chunk_offload_handler',
                                     GPTModel.initialize_model_chunk_offload_handler,
                                     create_dummy=True)

        patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward',
                                     Attention.forward)
        patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MultiLatentAttention.forward',
                                     MultiLatentAttention.forward)

        patch_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
                                     TEGroupedMLP.forward)

        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                     forward_backward_pipelining_with_interleaving_wrapper,
                                     apply_wrapper=True)

        patch_manager.register_cls_funcs('megatron.core.transformer.mlp.MLP',
                                         [MLP._offload_shared_fc1_forward,
                                          MLP._offload_shared_fc2_forward],
                                         create_dummy=True)
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                                     MLP.forward)

        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
                                     TransformerLayer._forward_attention)

        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                     TransformerBlock.forward)

        # update fine_grained_activation_offloading param
        patch_manager.register_patch('transformer_engine.pytorch.module.linear.Linear.__init__',
                                     te_module_init_wrapper,
                                     apply_wrapper=True)
        patch_manager.register_patch('transformer_engine.pytorch.module.layernorm_linear.LayerNormLinear.__init__',
                                     te_module_init_wrapper,
                                     apply_wrapper=True)
        patch_manager.register_patch('transformer_engine.pytorch.module.grouped_linear.GroupedLinear.__init__',
                                     te_module_init_wrapper,
                                     apply_wrapper=True)
        patch_manager.register_patch('transformer_engine.pytorch.module.batched_linear.BatchedLinear.__init__',
                                     te_module_init_wrapper,
                                     apply_wrapper=True)
