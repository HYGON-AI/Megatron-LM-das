# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
import torch

from argparse import ArgumentParser

from ..feature import AbstractFeature
from hcu_megatron.patch_utils import get_inner_func


class MegatronBasicFeature(AbstractFeature):

    def __init__(self):
        super().__init__('megatron-basic', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--use-ckpt-memory-cache', action='store_true', default=False,
                    help='Whether to enable memory caching for checkpoints (to speed up access by keeping checkpoints in memory)')
        group.add_argument('--comm-time-log-iter', type=int, default=None,
                        help='iter to log communication time')

    def register_patches(self, patch_manager, args):
        self.register_core_distributed_patches(patch_manager, args)
        self.register_core_models_patches(patch_manager, args)
        self.register_core_transformers_patches(patch_manager, args)
        self.register_core_ssm_patches(patch_manager, args)
        self.register_core_tokenizers_patches(patch_manager, args)
        self.register_core_extentions_patches(patch_manager, args)
        self.register_tensor_parallel_patches(patch_manager, args)
        self.register_pipeline_parallel_patches(patch_manager, args)
        self.register_training_patches(patch_manager, args)
        self.register_miscellaneous_patches(patch_manager, args)
        self.register_core_dist_checkpointing_patches(patch_manager, args)

    def register_core_dist_checkpointing_patches(self, patch_manager, args):
        if args.use_ckpt_memory_cache:
            from hcu_megatron.core.dist_checkpoint.strategies.filesystem_async import FileSystemWriterAsync
            from hcu_megatron.core.dist_checkpoint.strategies.torch import TorchDistLoadShardedStrategy

            from hcu_megatron.core.dist_checkpoint.validation import _compute_shards_access
            from hcu_megatron.core.dist_checkpoint.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
            from hcu_megatron.core.dist_checkpoint.exchange_utils import determine_main_replica_uniform_distribution

            # ckpt-memory-cache
            patch_manager.register_patch('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.write_preloaded_data',
                                        FileSystemWriterAsync.write_preloaded_data)
            patch_manager.register_patch('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors',
                                        FileSystemWriterAsync.preload_tensors)
            patch_manager.register_cls_funcs('megatron.core.dist_checkpointing.strategies.torch.TorchDistLoadShardedStrategy',
                                                  [TorchDistLoadShardedStrategy.load,
                                                   TorchDistLoadShardedStrategy.load_tensors_metadata,
                                                   TorchDistLoadShardedStrategy.load_sharded_metadata,])
            #ckpt-memory-cache load norm
            patch_manager.register_patch('megatron.core.dist_checkpointing.validation._compute_shards_access',
                                        _compute_shards_access)
            patch_manager.register_patch('megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelLoadStrategyWrapper.apply_loading_parallelization',
                                        FullyParallelLoadStrategyWrapper.apply_loading_parallelization)
            patch_manager.register_patch('megatron.core.dist_checkpointing.exchange_utils.determine_main_replica_uniform_distribution',
                                        determine_main_replica_uniform_distribution)

    def register_core_distributed_patches(self, patch_manager, args):
        from hcu_megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

        patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync',
            _ParamAndGradBucketGroup.finish_grad_sync)
        patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync',
            _ParamAndGradBucketGroup.start_grad_sync)

    def register_core_models_patches(self, patch_manager, args):
        from hcu_megatron.core.models.gpt.gpt_model import gpt_model_postprocess, GPTModel
        from hcu_megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

        # GPT Model
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                                    GPTModel.__init__)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                    GPTModel.forward)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel._preprocess',
                                    GPTModel._preprocess)

        # support vocab parallel
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel._postprocess',
                                    gpt_model_postprocess)

        # Vocabulary parallelism
        patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
                                    LanguageModelEmbedding.__init__)
        patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                                    LanguageModelEmbedding.forward)
        patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.shared_embedding_or_output_weight',
                                    GPTModel.shared_embedding_or_output_weight,
                                    create_dummy=True)

    def register_core_transformers_patches(self, patch_manager, args):
        from hcu_megatron.core.transformer.transformer_config import transformer_config_post_init_wrapper
        from hcu_megatron.core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward_wrapper
        from hcu_megatron.core.transformer.attention import attention_init_wrapper
        from hcu_megatron.core.transformer.moe.experts import TEGroupedMLP

        # Transformer config, add new params
        patch_manager.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                    transformer_config_post_init_wrapper)
        # support experts_recompute
        patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                    moe_layer_init_wrapper)
        patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.forward',
                                    moe_layer_forward_wrapper)
        # fused gelu and mul
        patch_manager.register_patch('megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
                                    get_inner_func(TEGroupedMLP.forward, "bias_act_func"),
                                    patch_inner_func=True,
                                    inner_func_name="bias_act_func",)
        # (1) cpu offload. (2) seq1f1b
        patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__',
                                    attention_init_wrapper,
                                    apply_wrapper=True)

        from hcu_megatron.core.recompute import checkpointed_forward
        from hcu_megatron.core.transformer.attention import Attention
        from hcu_megatron.core.transformer.transformer_block import TransformerBlock
        from hcu_megatron.core.transformer.transformer_layer import TransformerLayer
        from hcu_megatron.core.transformer.multi_latent_attention import multi_latent_attention_forward_wrapper
        from hcu_megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        patch_manager.register_patch('megatron.core.recompute.checkpointed_forward',
                                    checkpointed_forward)
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    TransformerBlock.forward)
        patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
                                    TransformerLayer._forward_attention)
        # support recompute_mtp_layer_ids
        patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer._checkpointed_forward',
                                    MultiTokenPredictionLayer._checkpointed_forward)

        patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward',
                                    Attention.forward)
        patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MultiLatentAttention.forward',
                                    multi_latent_attention_forward_wrapper,
                                    apply_wrapper=True)

    def register_core_ssm_patches(self, patch_manager, args):
        from hcu_megatron.core.ssm.gated_delta_net import GatedDeltaNet

        patch_manager.register_patch(
            'megatron.core.ssm.gated_delta_net.GatedDeltaNet._apply_gated_norm',
            GatedDeltaNet._apply_gated_norm)
        patch_manager.register_patch(
            'megatron.core.ssm.gated_delta_net.GatedDeltaNet._apply_gated_norm_fallback',
            GatedDeltaNet._apply_gated_norm_fallback,
            create_dummy=True)
        patch_manager.register_patch(
            'megatron.core.ssm.gated_delta_net.GatedDeltaNet._can_use_fused_gated_rmsnorm',
            GatedDeltaNet._can_use_fused_gated_rmsnorm,
            create_dummy=True)

    def register_core_tokenizers_patches(self, patch_manager, args):
        from hcu_megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer_wrapper

        patch_manager.register_patch('megatron.core.tokenizers.utils.build_tokenizer.build_tokenizer',
                                    build_tokenizer_wrapper,
                                    apply_wrapper=True)

    def register_core_extentions_patches(self, patch_manager, args):
        import transformer_engine as te

        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        if int(os.getenv("GROUPED_GEMM_BatchLinear", '0')):
            TEGroupedLinear.__bases__ = (te.pytorch.BatchedLinear,)

    def register_tensor_parallel_patches(self, patch_manager, args):
        from hcu_megatron.core.parallel_state import log_timing_wrapper
        from hcu_megatron.core.tensor_parallel.random import checkpoint_wrapper

        # VocabParallelEmbedding
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)
        # _VocabParallelCrossEntropy
        patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    remove_origin_wrappers=True)        
        patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    torch.compile(mode='max-autotune-no-cudagraphs'),
                                    apply_wrapper=True)
        patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                                    staticmethod,
                                    apply_wrapper=True)
        
        # reduce_scatter_to_sequence_parallel_region
        patch_manager.register_patch('megatron.core.tensor_parallel.mappings.reduce_scatter_to_sequence_parallel_region',
                                    torch._dynamo.disable,
                                    apply_wrapper=True)
        # reduce_from_tensor_model_parallel_region
        patch_manager.register_patch('megatron.core.tensor_parallel.mappings.reduce_from_tensor_model_parallel_region',
                                    torch._dynamo.disable,
                                    apply_wrapper=True)
        
        # checkpoint_wrapper for RiPipe
        patch_manager.register_patch('megatron.core.tensor_parallel.random.checkpoint',
                                    checkpoint_wrapper,
                                    apply_wrapper=True)
        
        # NCCL time log
        if args.comm_time_log_iter is not None:
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer.dist_all_gather_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer.dist_reduce_scatter_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.dist_all_gather_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.dist_reduce_scatter_func',
                                        log_timing_wrapper,
                                        apply_wrapper=True)

            patch_manager.register_patch('torch.distributed.broadcast',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.all_reduce',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.all_gather',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.all_gather_into_tensor',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.reduce_scatter',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.reduce_scatter_tensor',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.all_to_all_single',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.isend',
                                        log_timing_wrapper,
                                        apply_wrapper=True)
            patch_manager.register_patch('torch.distributed.irecv',
                                        log_timing_wrapper,
                                        apply_wrapper=True)

    def register_pipeline_parallel_patches(self, patch_manager, args):
        from hcu_megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        from hcu_megatron.core.pipeline_parallel.utils import set_ideal_affinity_for_current_gpu

        # replace set_ideal_affinity_for_current_gpu with a no-op func
        patch_manager.register_patch("megatron.core.pipeline_parallel.utils.set_ideal_affinity_for_current_gpu",
                                    set_ideal_affinity_for_current_gpu)
        # support full-iteration CUDA Graphs
        patch_manager.register_patch("megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator._communicate",
                                    P2PCommunicator._communicate)

    def register_training_patches(self, patch_manager, args):
        from hcu_megatron.training.training import train
        from hcu_megatron.training.initialize import _set_random_seed
        from hcu_megatron.training.training import train_step
        from hcu_megatron.training.training import setup_model_and_optimizer
        from hcu_megatron.training.argument_utils import core_transformer_config_from_args_wrapper
        from hcu_megatron.training.training import save_checkpoint_and_time_wrapper

        # Add a fixed seed.
        patch_manager.register_patch('megatron.training.initialize._set_random_seed',
                                    _set_random_seed)

        # add trace_handler
        patch_manager.register_patch('megatron.training.training.train',
                                    train)
        # support dualpipev, edgc
        patch_manager.register_patch('megatron.training.training.train_step',
                                    train_step)
        # (1) edgc, (2) ckpt add save/load iter info to ckpt
        patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer',
                                    setup_model_and_optimizer)
        # prevent re-initialization of config
        patch_manager.register_patch('megatron.training.argument_utils.core_transformer_config_from_args',
                                    core_transformer_config_from_args_wrapper)
        # support edgc, ckpt add save/load iter info to ckpt
        patch_manager.register_patch('megatron.training.training.save_checkpoint_and_time',
                                    save_checkpoint_and_time_wrapper,
                                    apply_wrapper=True)

        # 保存ckpt时同步导出一份HF格式权重(bridge)。
        # save_checkpoint_and_time_wrapper 内部对 edgc 和 HF 导出各自判断开关,
        # 不开 save_hf_weights 时其行为与原函数一致。
        # edgc 开启时该 wrapper 已由 gradient_compress_feature 注册, 重复注册
        # 会触发 patch_utils 的 "wrapper has already been applied", 故此处排除。
        if getattr(args, 'save_hf_weights', False) and not args.enable_dynamic_grad_comp:
            from hcu_megatron.training.training import save_checkpoint_and_time_wrapper

            patch_manager.register_patch('megatron.training.training.save_checkpoint_and_time',
                                        save_checkpoint_and_time_wrapper,
                                        apply_wrapper=True)

    def register_miscellaneous_patches(self, patch_manager, args):
        from hcu_megatron.core.full_cuda_graph import clone_tensors_in_struct
        from hcu_megatron.core.parallel_state import create_group, initialize_model_parallel_wrapper
        from hcu_megatron.miscellaneous.gpt_builders import gpt_builder_wrapper
        from hcu_megatron.training.arguments import validate_args_func_decorator, _print_args

        patch_manager.register_patch('megatron.training.arguments.validate_args',
                                    validate_args_func_decorator,
                                    apply_wrapper=True)
        patch_manager.register_patch('megatron.training.yaml_arguments.validate_yaml',
                                    validate_args_func_decorator,
                                    apply_wrapper=True)
        patch_manager.register_patch('megatron.training.arguments._print_args',
                                    _print_args,)
        patch_manager.register_patch('megatron.training.yaml_arguments._print_args',
                                    _print_args,)
        # output parallel groups
        patch_manager.register_patch('megatron.core.parallel_state.create_group', 
                                    create_group)
        patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                    initialize_model_parallel_wrapper,
                                    apply_wrapper=True)
        # output model info
        patch_manager.register_patch('gpt_builders.gpt_builder',
                                    gpt_builder_wrapper,
                                    apply_wrapper=True)
        # support full-iteration CUDA Graphs
        patch_manager.register_patch('megatron.core.full_cuda_graph.clone_tensors_in_struct',
                                    clone_tensors_in_struct)
        
