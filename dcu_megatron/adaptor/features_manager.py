from megatron.core.utils import is_te_min_version


def a2a_overlap_adaptation(patches_manager):
    """
        patches_manager: MegatronPatchesManager
    """
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    from ..core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from ..core.transformer.transformer_layer import TransformerLayer
    from ..core.models.gpt.gpt_model import GPTModel
    from ..core.pipeline_parallel.schedules import get_pp_rank_microbatches, forward_backward_pipelining_with_interleaving
    from ..core.extensions.transformer_engine import (
        _get_extra_te_kwargs_wrapper,
        TELinear,
        TELayerNormColumnParallelLinear,
    )
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

    patches_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer',
                                   TransformerLayer)

    patches_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.build_schedule_plan',
                                   GPTModel.build_schedule_plan,
                                   create_dummy=True)

    # backward_dw
    if is_te_min_version("2.4.0.dev0"):
        patches_manager.register_patch('megatron.core.extensions.transformer_engine._get_extra_te_kwargs',
                                    _get_extra_te_kwargs_wrapper,
                                    apply_wrapper=True)
    patches_manager.register_patch('megatron.core.extensions.transformer_engine.TELinear',
                                   TELinear)
    patches_manager.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear',
                                   TELayerNormColumnParallelLinear)
    TEColumnParallelLinear.__bases__ = (TELinear,)
    TERowParallelLinear.__bases__ = (TELinear,)

    if is_te_min_version("1.9.0.dev0"):
        from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
        from ..core.extensions.transformer_engine import TEGroupedLinear

        patches_manager.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear',
                                       TEGroupedLinear)
        TEColumnParallelGroupedLinear.__bases__ = (TEGroupedLinear,)
        TERowParallelGroupedLinear.__bases__ = (TEGroupedLinear,)

    patches_manager.register_patch('megatron.core.transformer.multi_latent_attention.MLASelfAttention.backward_dw',
                                   MLASelfAttention.backward_dw,
                                   create_dummy=True)
    patches_manager.register_patch('megatron.core.transformer.mlp.MLP.backward_dw',
                                   MLP.backward_dw,
                                   create_dummy=True)
    patches_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.backward_dw',
                                   TEGroupedMLP.backward_dw,
                                   create_dummy=True)
    patches_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.backward_dw',
                                   MoELayer.backward_dw,
                                   create_dummy=True)
