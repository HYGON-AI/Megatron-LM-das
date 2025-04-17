import os
import abc
import sys
import types
import argparse
import torch


class MegatronAdaptation:
    """
        A module manager supports adaptation registration, application and execution.
    """
    _patch_info_collection = {}
    _args = None

    @classmethod
    def execute(cls):
        """
        Execute adaptations.
        """
        for adaptation in [CoreAdaptation(), LegacyAdaptation()]:
            adaptation.execute()
        MegatronAdaptation.apply()
        # MegatronAdaptation.post_execute()

    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=False, create_dummy=False, apply_wrapper=False, remove_origin_wrappers=False):
        """
        Register adaptations into collection.
        """
        if orig_func_name not in cls._patch_info_collection:
            from .patch_utils import Patch
            cls._patch_info_collection[orig_func_name] = Patch(
                orig_func_name,
                new_func,
                create_dummy,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers
            )
        else:
            cls._patch_info_collection.get(orig_func_name).set_patch_func(
                new_func,
                force_patch,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers
            )

    @classmethod
    def apply(cls):
        """
        Apply adaptations.
        """
        for patch in cls._patch_info_collection.values():
            patch.apply_patch()

    @classmethod
    def post_execute(cls):
        """
        Execute after other adaptations.
        """
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.transformer_block import TransformerBlock


class MegatronAdaptationABC:
    """
    Abstract class for adaptation.
    """
    @abc.abstractmethod
    def execute(self):
        """
        Do Adaptation
        """


class CoreAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    def execute(self):
        self.patch_core_distributed()
        self.patch_core_models()
        self.patch_core_transformers()
        self.patch_core_extentions()
        self.patch_tensor_parallel()
        self.patch_training()
        self.patch_miscellaneous()

    def patch_core_distributed(self):
        # Mtp share embedding
        from ..core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
        MegatronAdaptation.register('megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                                    _allreduce_word_embedding_grads)

    def patch_core_models(self):
        from ..core.models.common.embeddings.language_model_embedding import (
            language_model_embedding_forward,
            language_model_embedding_init_func
        )
        from ..core.models.gpt.gpt_model import (
            gpt_model_forward,
            gpt_model_init_wrapper,
            shared_embedding_or_mtp_embedding_weight
        )
        from ..training.utils import get_batch_on_this_tp_rank

        # Embedding
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
            language_model_embedding_init_func)
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
            language_model_embedding_forward)

        MegatronAdaptation.register('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

        # GPT Model
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_model_forward)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                                    gpt_model_init_wrapper,
                                    apply_wrapper=True)

        from megatron.core.models.gpt.gpt_model import GPTModel
        setattr(GPTModel, 'shared_embedding_or_mtp_embedding_weight', shared_embedding_or_mtp_embedding_weight)

    def patch_core_transformers(self):
        from ..core import transformer_block_init_wrapper, transformer_block_forward
        from ..core.transformer.transformer_config import TransformerConfigPatch, MLATransformerConfigPatch
        
        # Transformer block
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    transformer_block_forward)

        # Transformer config
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig',
                                    TransformerConfigPatch)
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.MLATransformerConfig',
                                    MLATransformerConfigPatch)

        # Moe
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity',
                                    torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False}),
                                    apply_wrapper=True)
        # MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.switch_load_balancing_loss_func',
        #                             torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False, "triton.cudagraph_support_input_mutation":True}),
        #                             apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.permute',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)

    def patch_core_extentions(self):
        import transformer_engine as te

        from ..core.extensions.transformer_engine import TEDotProductAttentionPatch
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        MegatronAdaptation.register('megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__',
                                    TEDotProductAttentionPatch.__init__)

        if int(os.getenv("GROUPED_GEMM_BatchLinear", '0')):
            TEGroupedLinear.__bases__ = (te.pytorch.BatchLinear,)

    def patch_tensor_parallel(self):
        from ..core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
        from ..core.tensor_parallel import vocab_parallel_embedding_forward, vocab_parallel_embedding_init

        # VocabParallelEmbedding
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    vocab_parallel_embedding_forward)
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                    vocab_parallel_embedding_init)

        # VocabParallelCrossEntropy
        MegatronAdaptation.register('megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
                                    VocabParallelCrossEntropy.calculate_predicted_logits)
        # _VocabParallelCrossEntropy
        MegatronAdaptation.register('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    remove_origin_wrappers=True)        
        MegatronAdaptation.register('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    staticmethod,
                                    apply_wrapper=True)

        # flux
        if os.getenv("USE_FLUX_OVERLAP", 0):
            from ..core.tensor_parallel import (
                FluxColumnParallelLinear,
                FluxRowParallelLinear
            )
            from ..core.models.gpt.gpt_layer_specs import get_gpt_layer_with_flux_spec

            MegatronAdaptation.register("megatron.core.extensions.transformer_engine.TEColumnParallelLinear",
                                        FluxColumnParallelLinear)
            MegatronAdaptation.register("megatron.core.extensions.transformer_engine.TERowParallelLinear",
                                        FluxRowParallelLinear)
            MegatronAdaptation.register("megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec",
                                        get_gpt_layer_with_flux_spec)

    def patch_training(self):
        from ..training.tokenizer import build_tokenizer
        from ..training.initialize import _initialize_distributed
        from ..training.initialize import _compile_dependencies
        from ..training.training import train

        MegatronAdaptation.register('megatron.training.tokenizer.tokenizer.build_tokenizer',
                                    build_tokenizer)
        MegatronAdaptation.register('megatron.training.initialize._initialize_distributed',
                                    _initialize_distributed)
        MegatronAdaptation.register('megatron.training.initialize._compile_dependencies',
                                    _compile_dependencies)

        # traing.train
        MegatronAdaptation.register('megatron.training.training.train',
                                    train)

    def patch_miscellaneous(self):
        from ..training.arguments import parse_args

        MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args)


class LegacyAdaptation(MegatronAdaptationABC):
    """
        Adaptations for models in legacy structure.
    """

    def execute(self):
        self.patch_legacy_models()

    def patch_legacy_models(self):
        from ..legacy.model.transformer import ParallelMLPPatch, ParallelAttentionPatch
        from ..legacy.model.utils import get_norm

        # ParallecMLP
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelMLP.__init__',
                                    ParallelMLPPatch.__init__)

        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelAttention.forward',
                                    ParallelAttentionPatch.forward)

        # rms_norm.RMSNorm
        MegatronAdaptation.register('megatron.legacy.model.rms_norm.RMSNorm.forward',
                                    torch.compile(mode="max-autotune-no-cudagraphs"),
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.legacy.model.utils.get_norm',
                                    get_norm)


MegatronAdaptation.execute()
