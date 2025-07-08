import os
import abc
import argparse
import torch

from megatron.core.utils import is_te_min_version

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from dcu_megatron.training.arguments import process_adaptor_args


_ARGS = None


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def get_adaptor_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='Adaptor Arguments', allow_abbrev=False)
        _ARGS, unknown = process_adaptor_args(parser).parse_known_args()
        parser_unknown_args(_ARGS, unknown)
    return _ARGS


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

        # apply features
        feature_adaptation()

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
        pass


def feature_adaptation():
    adaptor_args = get_adaptor_args()

    # Advanced acceleration algorithm
    adaptation_l2(MegatronPatchesManager, adaptor_args)

    MegatronPatchesManager.apply_patches()


def adaptation_l2(patches_manager, adaptor_args):
    """
    Advanced acceleration algorithm
    """
    for feature in ADAPTOR_FEATURES:
        if getattr(adaptor_args, feature.feature_name, None) and feature.optimization_level == 2:
            feature.register_patches(patches_manager, adaptor_args)


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
        self.path_core_parallel_state()

    def patch_core_distributed(self):
        pass

    def patch_core_models(self):
        from ..core.models.gpt.gpt_model import gpt_model_init_wrapper, gpt_model_forward

        # GPT Model
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                                    gpt_model_init_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                    gpt_model_forward)

    def patch_core_transformers(self):
        from ..core import transformer_block_init_wrapper
        from ..core.transformer.transformer_config import transformer_config_post_init_wrapper
        
        # Transformer block. If mtp_num_layers > 0, move final_layernorm outside
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)

        # Transformer config, add new params
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                    transformer_config_post_init_wrapper)

        # Moe
        # MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity',
        #                             torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False}),
        #                             apply_wrapper=True)
        # MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.switch_load_balancing_loss_func',
        #                             torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False, "triton.cudagraph_support_input_mutation":True}),
        #                             apply_wrapper=True)
        # MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.permute',
        #                             torch.compile(mode='max-autotune-no-cudagraphs'),
        #                             apply_wrapper=True)
        # MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute',
        #                             torch.compile(mode='max-autotune-no-cudagraphs'),
        #                             apply_wrapper=True)

    def patch_core_extentions(self):
        import transformer_engine as te

        from ..core.extensions.transformer_engine import TEDotProductAttentionPatch
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        if not is_te_min_version("1.10.0"):
            # kv channels, te_min_version 1.10.0 -> 1.9.0
            MegatronAdaptation.register('megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__',
                                        TEDotProductAttentionPatch.__init__)

        if int(os.getenv("GROUPED_GEMM_BatchLinear", '0')):
            TEGroupedLinear.__bases__ = (te.pytorch.BatchedLinear if is_te_min_version("2.3.0.dev0") else te.pytorch.BatchLinear,)

    def patch_tensor_parallel(self):
        from functools import partial
        from dcu_megatron.core.tensor_parallel.mappings import all_to_all
        from ..core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy

        # VocabParallelEmbedding
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)

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

        # reduce_scatter_to_sequence_parallel_region
        MegatronAdaptation.register('megatron.core.tensor_parallel.mappings.reduce_scatter_to_sequence_parallel_region',
                                    torch._dynamo.disable,
                                    apply_wrapper=True)
        # reduce_from_tensor_model_parallel_region
        MegatronAdaptation.register('megatron.core.tensor_parallel.mappings.reduce_from_tensor_model_parallel_region',
                                    torch._dynamo.disable,
                                    apply_wrapper=True)

        adaptor_args = get_adaptor_args()
        if adaptor_args.use_quantize_comm:
            MegatronAdaptation.register('megatron.core.tensor_parallel.mappings.all_to_all',
                                        partial(all_to_all, use_quantize_comm=True))

    def patch_training(self):
        from ..training.tokenizer import build_tokenizer
        from ..training.initialize import _initialize_distributed
        from ..training.initialize import _compile_dependencies
        from ..training.training import train
        from ..training.initialize import _set_random_seed

        MegatronAdaptation.register('megatron.training.tokenizer.tokenizer.build_tokenizer',
                                    build_tokenizer)
        # specify init_method
        MegatronAdaptation.register('megatron.training.initialize._initialize_distributed',
                                    _initialize_distributed)
        # remove fused_kernels
        MegatronAdaptation.register('megatron.training.initialize._compile_dependencies',
                                    _compile_dependencies)

        # 添加固定seed
        MegatronAdaptation.register('megatron.training.initialize._set_random_seed',
                                    _set_random_seed)

        # add trace_handler
        MegatronAdaptation.register('megatron.training.training.train',
                                    train)

    def patch_miscellaneous(self):
        from ..training.arguments import parse_args

        MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args)

    def path_core_parallel_state(self):
        from ..core.parallel_state import initialize_model_parallel_wrapper, create_group

        MegatronAdaptation.register('megatron.core.parallel_state.create_group', 
                                    create_group)
        MegatronAdaptation.register('megatron.core.parallel_state.initialize_model_parallel',
                                    initialize_model_parallel_wrapper,
                                    apply_wrapper=True)

class LegacyAdaptation(MegatronAdaptationABC):
    """
        Adaptations for models in legacy structure.
    """

    def execute(self):
        self.patch_legacy_models()

    def patch_legacy_models(self):
        from ..legacy.model.transformer import (
            parallel_mlp_init_wrapper,
            ParallelAttentionPatch,
            parallel_attention_init_wrapper
        )
        from ..legacy.model.utils import get_norm

        # ParallecMLP
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelMLP.__init__',
                                    parallel_mlp_init_wrapper,
                                    apply_wrapper=True)

        # ParallelAttention
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelAttention.__init__',
                                    parallel_attention_init_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelAttention.forward',
                                    ParallelAttentionPatch.forward)

        # rms_norm.RMSNorm
        MegatronAdaptation.register('megatron.legacy.model.rms_norm.RMSNorm.forward',
                                    torch.compile(mode="max-autotune-no-cudagraphs"),
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.legacy.model.utils.get_norm',
                                    get_norm)


MegatronAdaptation.execute()
