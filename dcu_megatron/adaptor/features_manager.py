from megatron.core.utils import is_te_min_version


def a2a_overlap_adaptation(patches_manager):
    """
        patches_manager: MegatronPatchesManager
    """

    from ..core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from ..core.transformer.transformer_block import TransformerBlock
    from ..core.transformer.transformer_layer import TransformerLayer
    from ..core.models.gpt.gpt_model import GPTModel
    from ..core.pipeline_parallel.schedules import get_pp_rank_microbatches, forward_backward_pipelining_with_interleaving
    from ..core.extensions.transformer_engine import _get_extra_te_kwargs_wrapper, TELinear, TELayerNormColumnParallelLinear
    from ..core.transformer.multi_latent_attention import MLASelfAttention
    from ..core.transformer.mlp import MLP
    from ..core.transformer.moe.experts import TEGroupedMLP
    from ..core.transformer.moe.moe_layer import MoELayer

    # num_warmup_microbatches + 1
    patches_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_pp_rank_microbatches',
                                   get_pp_rank_microbatches)

    # a2a_overlap
    patches_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                   forward_backward_pipelining_with_interleaving)

    patches_manager.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher',
                                   MoEAlltoAllTokenDispatcher)

    patches_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock',
                                   TransformerBlock)

    patches_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer',
                                   TransformerLayer)

    patches_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel',
                                   GPTModel)

    # backward_dw
    patches_manager.register_patch('megatron.core.extensions.transformer_engine._get_extra_te_kwargs',
                                   _get_extra_te_kwargs_wrapper,
                                   apply_wrapper=True)
    patches_manager.register_patch('megatron.core.extensions.transformer_engine.TELinear',
                                   TELinear)
    patches_manager.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear',
                                   TELayerNormColumnParallelLinear)
    if is_te_min_version("1.9.0.dev0"):
        from ..core.extensions.transformer_engine import TEGroupedLinear
        patches_manager.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear',
                                       TEGroupedLinear)

    patches_manager.register_patch('megatron.core.transformer.multi_latent_attention.MLASelfAttention',
                                   MLASelfAttention)
    patches_manager.register_patch('megatron.core.transformer.mlp.MLP',
                                   MLP)
    patches_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP',
                                   TEGroupedMLP)
    patches_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer',
                                   MoELayer)
