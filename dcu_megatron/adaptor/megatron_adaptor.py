import os
import abc
import argparse
import torch

from megatron.core.utils import is_te_min_version

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from dcu_megatron.training.arguments import process_adaptor_args
from megatron.training import get_args

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

    @staticmethod
    def register_cls_funcs(orig_class, new_funcs: list = None, create_dummy=False):
        if not orig_class.endswith("."):
            orig_class += "."

        for new_func in new_funcs:
            assert hasattr(new_func, '__name__') and not new_func.__name__.endswith(('wrapper', 'decorator'))

            orig_func_name = orig_class + new_func.__name__
            MegatronAdaptation.register(orig_func_name, new_func=new_func, create_dummy=create_dummy)

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
        self.patch_core_tokenizers()
        self.patch_core_extentions()
        self.patch_tensor_parallel()
        self.patch_training()
        self.patch_miscellaneous()
        self.patch_core_dist_checkpointing()

    def patch_core_dist_checkpointing(self):
        adaptor_args = get_adaptor_args()
        if adaptor_args.use_ckpt_memory_cache:
            from ..core.dist_checkpoint.strategies.filesystem_async import write_preloaded_data, preload_tensors
            from ..core.dist_checkpoint.strategies.cached_metadata_filesystem_reader import CachedMetadataFileSystemReader
            from ..core.dist_checkpoint.strategies.torch import get_reformulation_metadata
            from ..core.dist_checkpoint.strategies.torch import TorchDistLoadShardedStrategy

            from ..core.dist_checkpoint.validation import _compute_shards_access, _validate_sharding_for_key_flattened
            from ..core.dist_checkpoint.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
            from ..core.dist_checkpoint.exchange_utils import determine_main_replica_uniform_distribution

            # ckpt-memory-cache
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.write_preloaded_data',
                                        write_preloaded_data)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors',
                                        preload_tensors)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader.CachedMetadataFileSystemReader',
                                        CachedMetadataFileSystemReader)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.torch.get_reformulation_metadata',
                                        get_reformulation_metadata)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.torch.TorchDistLoadShardedStrategy',
                                        TorchDistLoadShardedStrategy)
            #ckpt-memory-cache load norm
            MegatronAdaptation.register('megatron.core.dist_checkpointing.validation._compute_shards_access',
                                        _compute_shards_access)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.validation._validate_sharding_for_key_flattened',
                                        _validate_sharding_for_key_flattened)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelLoadStrategyWrapper.apply_loading_parallelization',
                                        FullyParallelLoadStrategyWrapper.apply_loading_parallelization)
            MegatronAdaptation.register('megatron.core.dist_checkpointing.exchange_utils.determine_main_replica_uniform_distribution',
                                        determine_main_replica_uniform_distribution)

    def patch_core_distributed(self):
        from ..core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
        MegatronAdaptation.register(
            'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync',
            _ParamAndGradBucketGroup.finish_grad_sync)

    def patch_core_models(self):
        from ..core.models.gpt.gpt_model import gpt_model_postprocess, GPTModel
        from ..core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

        # GPT Model
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                                    GPTModel.__init__)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                    GPTModel.forward)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel._preprocess',
                                    GPTModel._preprocess)

        # Transformer block. If mtp_num_layers > 0, move final_layernorm outside
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel._postprocess',
                                    gpt_model_postprocess)

        # Vocabulary parallelism
        MegatronAdaptation.register('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
                                    LanguageModelEmbedding.__init__)
        MegatronAdaptation.register('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                                    LanguageModelEmbedding.forward)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.shared_embedding_or_output_weight',
                                    GPTModel.shared_embedding_or_output_weight)

    def patch_core_transformers(self):
        from ..core.transformer.transformer_config import transformer_config_post_init_wrapper
        from ..core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward_wrapper
        from ..core.transformer.attention import attention_init_wrapper, SelfAttention
        from ..core.transformer.moe.experts import te_grouped_mlp_init_wrapper
        from ..core.transformer.transformer_layer import transformer_layer_init_wrapper
        from ..core.transformer.mlp import mlp_init_wrapper
        from ..core.transformer.moe.experts import TEGroupedMLP

        # Transformer config, add new params
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                    transformer_config_post_init_wrapper)
        # support experts_recompute
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                    moe_layer_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.forward',
                                    moe_layer_forward_wrapper)
        # use_qk_norm
        MegatronAdaptation.register('megatron.core.transformer.attention.SelfAttention.get_query_key_value_tensors',
                                    SelfAttention.get_query_key_value_tensors)
        # fused gelu and mul
        MegatronAdaptation.register('megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
                                    TEGroupedMLP.forward)
        # cpu offload.
        MegatronAdaptation.register('megatron.core.transformer.attention.Attention.__init__',
                                    attention_init_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.transformer.moe.experts.TEGroupedMLP.__init__',
                                    te_grouped_mlp_init_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                                    transformer_layer_init_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.core.transformer.mlp.MLP.__init__',
                                    mlp_init_wrapper,
                                    apply_wrapper=True)

        from ..core.transformer.attention import AttentionPatch
        from ..core.transformer.transformer_block import TransformerBlockPatch
        from ..core.transformer.transformer_layer import TransformerLayerPatch

        MegatronAdaptation.register(
            'megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
            TransformerBlockPatch._checkpointed_forward)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    TransformerBlockPatch.forward)
        MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
                                    TransformerLayerPatch._forward_attention)

        MegatronAdaptation.register('megatron.core.transformer.attention.Attention.forward',
                                    AttentionPatch.forward)

    def patch_core_tokenizers(self):
        from ..core.tokenizers.text.utils.build_tokenizer import build_tokenizer_wrapper

        MegatronAdaptation.register('megatron.core.tokenizers.text.utils.build_tokenizer.build_tokenizer',
                                    build_tokenizer_wrapper,
                                    apply_wrapper=True)

    def patch_core_extentions(self):
        import transformer_engine as te

        from ..core.extensions.transformer_engine import TEDotProductAttentionPatch
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        if not is_te_min_version("1.10.0"):
            # kv channels, te_min_version 1.10.0 -> 1.9.0
            MegatronAdaptation.register('megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__',
                                        TEDotProductAttentionPatch.__init__)

        if int(os.getenv("GROUPED_GEMM_BatchLinear", '0')):
            TEGroupedLinear.__bases__ = (te.pytorch.BatchedLinear,)

    def patch_tensor_parallel(self):
        from ..core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
        from ..core.parallel_state import log_timing_wrapper

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
        
        # NCCL time log
        adaptor_args = get_adaptor_args()
        if adaptor_args.comm_time_log_iter is not None:
            MegatronAdaptation.register('megatron.core.distributed.param_and_grad_buffer.dist_all_gather_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('megatron.core.distributed.param_and_grad_buffer.dist_reduce_scatter_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('megatron.core.tensor_parallel.mappings.dist_all_gather_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('megatron.core.tensor_parallel.mappings.dist_reduce_scatter_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)

            MegatronAdaptation.register('torch.distributed.broadcast',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.all_reduce',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.all_gather',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.all_gather_into_tensor',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.reduce_scatter',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.reduce_scatter_tensor',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.all_to_all_single',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.isend',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            MegatronAdaptation.register('torch.distributed.irecv',
                                        log_timing_wrapper,
                                        apply_wrapper=True)

    def patch_training(self):
        from ..training.tokenizer import build_tokenizer_wrapper, pad
        from ..training.initialize import _initialize_distributed
        from ..training.initialize import _compile_dependencies
        from ..training.training import train
        from ..training.initialize import _set_random_seed
        from ..training.training import train_step
        from ..training.training import setup_model_and_optimizer
        from ..training.datasets import _get_ltor_masks_and_position_ids
        from ..training.utils import get_batch_on_this_tp_rank

        MegatronAdaptation.register('megatron.training.tokenizer.tokenizer.build_tokenizer',
                                    build_tokenizer_wrapper,
                                    apply_wrapper=True)
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
        # support dualpipev, edgc
        MegatronAdaptation.register('megatron.training.training.train_step',
                                    train_step)
        # (1) edgc, (2) ckpt add save/load iter info to ckpt
        MegatronAdaptation.register('megatron.training.training.setup_model_and_optimizer',
                                    setup_model_and_optimizer)

        # support sft
        MegatronAdaptation.register('megatron.training.tokenizer.sft_tokenizer.SFTTokenizer.pad',
                                    pad)
        MegatronAdaptation.register('megatron.training.datasets.sft_dataset.SFTDataset._get_ltor_masks_and_position_ids',
                                    _get_ltor_masks_and_position_ids)

        # (1) dualpipev, (2) vocabulary parallelism
        MegatronAdaptation.register('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

    def patch_miscellaneous(self):
        from ..training.arguments import parse_args, validate_args_func_decorator, _print_args_wrapper
        from ..core.parallel_state import create_group, initialize_model_parallel_wrapper
        from ..miscellaneous.gpt_builders import gpt_builder_wrapper

        MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args)
        MegatronAdaptation.register('megatron.training.arguments.validate_args',
                                    validate_args_func_decorator,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.training.yaml_arguments.validate_yaml',
                                    validate_args_func_decorator,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.training.arguments._print_args',
                                    _print_args_wrapper,
                                    apply_wrapper=True)
        MegatronAdaptation.register('megatron.training.yaml_arguments._print_args',
                                    _print_args_wrapper,
                                    apply_wrapper=True)

        # output parallel groups
        MegatronAdaptation.register('megatron.core.parallel_state.create_group', 
                                    create_group)
        MegatronAdaptation.register('megatron.core.parallel_state.initialize_model_parallel',
                                    initialize_model_parallel_wrapper,
                                    apply_wrapper=True)

        # output model info
        MegatronAdaptation.register('gpt_builders.gpt_builder',
                                    gpt_builder_wrapper,
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
            parallel_attention_init_wrapper,
            ParallelTransformerLayerPatch,
            ParallelTransformerPatch
        )
        from ..legacy.model.utils import get_norm
        from ..legacy.model.language_model import TransformerLanguageModelPatch
        from ..legacy.model.gpt_model import GPTModelPatch

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

        MegatronAdaptation.register('megatron.legacy.model.language_model.TransformerLanguageModel.forward',
                                    TransformerLanguageModelPatch.forward)
        MegatronAdaptation.register('megatron.legacy.model.gpt_model.GPTModel.forward',
                                    GPTModelPatch.forward)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformerLayer.forward',
                                    ParallelTransformerLayerPatch.forward)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformer.forward',
                                    ParallelTransformerPatch.forward)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward',
                                    ParallelTransformerPatch._checkpointed_forward)


MegatronAdaptation.execute()
