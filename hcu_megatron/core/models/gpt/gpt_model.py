# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.extensions.transformer_engine import TELMHeadColumnParallelLinear
from megatron.core.fp8_utils import is_mxfp8_output_proj_active
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
    process_mtp_loss,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MegatronCoreGPTModel
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    is_using_quantization_scales,
)

from hcu_megatron.core.transformer.transformer_block import TransformerBlock
from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
from hcu_megatron.core.tensor_parallel import VocabParallelOutput
from hcu_megatron.training.arguments import get_adaptor_args


def gpt_model_postprocess(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    padding_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
    output_processor=None,
    output_processor_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """
    in_inference_mode = InferenceMode.is_active()
    if in_inference_mode:
        assert runtime_gather_output, "Inference must always gather TP logits"

    # Check if speculative decoding is active. When it is, MTP must be
    # computed *after* verification so that it is conditioned on verified
    # tokens rather than stale speculative tokens from the previous step.
    is_spec_decode = (
        in_inference_mode
        and inference_context is not None
        and inference_context.is_dynamic_batching()
        and inference_context.num_speculative_tokens > 0
    )

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess and not (in_inference_mode or is_spec_decode):
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=None,  # MTP layers don't use KV cache,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            embedding=self.embedding,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    if self.config.mtp_num_layers:
        assert self.config.mtp_num_layers > 0
        if in_inference_mode or is_spec_decode:
            # Cache decoder hidden states for serial MTP computation
            # after speculative token verification.
            self._decoder_hidden_states_cache = hidden_states
        else:
            # In training/eval, use the utility function for processing MTP loss/scaling.
            hidden_states = process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                is_training=self.training,
                compute_language_model_loss=self.compute_language_model_loss,
                config=self.config,
                cp_group=self.pg_collection.cp,
                packed_seq_params=packed_seq_params,
                scale_logits_fn=self._scale_logits if self.config.use_mup else None,
            )
    sequence_parallel_override = False

    if output_processor is not None:
        return output_processor(
            hidden_states=hidden_states,
            output_layer=self.output_layer,
            output_weight=output_weight,
            labels=labels,
            loss_mask=loss_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            context=output_processor_context,
            compute_language_model_loss=self.compute_language_model_loss,
            scale_logits=self._scale_logits,
            config=self.config,
        )

    if (
        in_inference_mode
        and inference_context is not None
        and inference_context.config.materialize_only_last_token_logits
    ):
        if inference_context.is_static_batching():
            hidden_states = hidden_states[-1:, :, :]
        else:
            if self.output_layer.sequence_parallel:
                # Perform the sequence parallel gather here instead of after the output layer
                # because we need to slice the last token logits from the full view of the
                # packed logits across all requests.
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=self.pg_collection.tp
                )
                self.output_layer.sequence_parallel = False
                sequence_parallel_override = True

            # Reshape [S, B, H] (with B=1) to [1, S, H] for logit extraction,
            # then back to [S’, B, H] for the output layer.
            reshaped = hidden_states.squeeze(1).unsqueeze(0)
            hidden_states = inference_context.last_token_logits(reshaped).unsqueeze(1)

    if get_adaptor_args().enable_vocab_parallel:
        assert labels is not None, "not supported yet"
        labels = labels.transpose(0, 1).contiguous()
        loss, _ = self.output_layer(hidden_states, weight=output_weight, labels=labels)
        loss = loss.transpose(0, 1).contiguous()
        return loss

    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )

    # Apply MuP output scaling to logits
    logits = self._scale_logits(logits)

    # Restore sequence parallel execution to the output layer if necessary.
    if sequence_parallel_override:
        assert (
            in_inference_mode
            and inference_context.is_dynamic_batching()
            and inference_context.config.materialize_only_last_token_logits
        )
        self.output_layer.sequence_parallel = True

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            }
        )
        log_config_to_disk(self.config, payload, prefix='input_and_logits')

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    loss = self.compute_language_model_loss(labels, logits)

    return loss


class GPTModel:
    """
    patch megatron GPTModel
    """
    # (1) introduce an attribute dualpipev_first_chunk. (2) support flux. (3) remove embedding when using dualpipev. (4) activation offload
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        split_vocab_embedding: bool = False,
        noop_block: bool = False,
        include_layer_norm: bool = False,
    ) -> None:
        super(MegatronCoreGPTModel, self).__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage
        self.disable_param_offloading = True

        # Vocabulary parallelism
        self.split_vocab_embedding = split_vocab_embedding
        self.noop_block = noop_block
        self.include_layer_norm = include_layer_norm
        self.has_vocab_embedding = (
            (self.pre_process and (not self.split_vocab_embedding))
            or ((not self.pre_process) and self.split_vocab_embedding)
        )

        args = get_adaptor_args()
        self.dualpipev_first_chunk = getattr(args, 'dualpipev_first_chunk', True)

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None and mtp_on_this_rank(
            layout=self.config.pipeline_model_parallel_layout,
            mtp_num_layers=self.config.mtp_num_layers,
            ignore_virtual=False,
            vp_stage=vp_stage,
        )

        if self.pre_process or self.mtp_process or self.split_vocab_embedding:
            from .utils import SkipEmbeddingAllocationContextManager
            with SkipEmbeddingAllocationContextManager(self.mtp_process and args.schedule_method == 'dualpipev'):
                self.embedding = LanguageModelEmbedding(
                    config=self.config,
                    vocab_size=self.vocab_size,
                    max_sequence_length=self.max_sequence_length,
                    position_embedding_type=position_embedding_type,
                    scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                    tp_group=self.pg_collection.tp,
                    split_vocab_embedding=self.split_vocab_embedding,
                    vocab_embedding_only=(args.enable_vocab_parallel and not self.pre_process),
                )

        # dualpipev use shared embedding weight
        skip_embedding_allocation = self.mtp_process and args.schedule_method == 'dualpipev'
        if skip_embedding_allocation:
            def remove_shared_embedding_check(self, incompatible_keys):
                """
                Remove embedding weight from unexpected keys.
                """
                keys = deepcopy(incompatible_keys.unexpected_keys)
                for key in keys:
                    if 'embedding.word_embeddings.weight' in key:
                        incompatible_keys.unexpected_keys.remove(key)

            self.register_load_state_dict_post_hook(remove_shared_embedding_check)

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=getattr(self.config, "yarn_rotary_scaling_factor"),
                original_max_position_embeddings=getattr(
                    self.config, "yarn_original_max_position_embeddings"
                ),
                beta_fast=getattr(self.config, "yarn_beta_fast"),
                beta_slow=getattr(self.config, "yarn_beta_slow"),
                mscale=getattr(self.config, "yarn_mscale"),
                mscale_all_dim=getattr(self.config, "yarn_mscale_all_dim"),
                correction_range_round_to_int=getattr(
                    self.config, "yarn_correction_range_round_to_int"
                ),
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = MultimodalRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
            self.mrope_section = self.config.mrope_section
            assert (
                self.mrope_section is not None
            ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
            noop_block=self.noop_block,
            force_layer_norm=self.include_layer_norm,
            post_layer_norm=not get_adaptor_args().enable_vocab_parallel,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config,
                spec=self.mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=self.pg_collection,
            )

            self._setup_mtp_cuda_graphs()

        # Output
        if self.post_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            if args.enable_vocab_parallel:
                self.output_layer = VocabParallelOutput(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=(
                        config.embedding_init_method
                        if config.use_mup and not self.share_embeddings_and_output_weights
                        else config.init_method
                    ),
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    fuse_forward_input_grad=not args.disable_backward_fusion,
                    sync_allreduce=False,
                )
            elif args.parallel_linear_impl == "flux":
                from hcu_megatron.core.tensor_parallel.layers import FluxColumnParallelLinear

                self.output_layer = FluxColumnParallelLinear(
                    self.config.hidden_size,
                    self.vocab_size,
                    config=self.config,
                    init_method=(
                        config.embedding_init_method
                        if config.use_mup and not self.share_embeddings_and_output_weights
                        else config.init_method
                    ),
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                    tp_group=self.pg_collection.tp,
                )
            else:
                output_layer_cls = (
                    TELMHeadColumnParallelLinear
                    if is_mxfp8_output_proj_active(config)
                    else tensor_parallel.ColumnParallelLinear
                )
                self.output_layer = output_layer_cls(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=(
                        config.embedding_init_method
                        if config.use_mup and not self.share_embeddings_and_output_weights
                        else config.init_method
                    ),
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                    tp_group=self.pg_collection.tp,
                )

        if self.has_vocab_embedding or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )
        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def preprocess_for_fine_grained_offloading(self):
        """Preprocess for fine-grained activation offloading."""
        if get_adaptor_args().schedule_method == "dualpipev":
            off_interface.init_chunk_handler(
                getattr(self, 'dualpipev_first_chunk', True),
                min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
            )
        else:
            off_interface.init_chunk_handler(
                vp_size=self.config.virtual_pipeline_model_parallel_size,
                vp_stage=self.vp_stage,
                min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
                max_inflight_offloads=self.config.fine_grained_offloading_max_inflight_offloads,
            )
        if self.disable_param_offloading:
            for param in self.decoder.parameters():
                off_interface.mark_not_offloadable(param)
            if self.mtp_process:
                for param in self.mtp.parameters():
                    off_interface.mark_not_offloadable(param)
            if self.post_process:
                for param in self.output_layer.parameters():
                    off_interface.mark_not_offloadable(param)
            self.disable_param_offloading = False

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: When dualpipe is enabled, return the weights from dual_chunk, otherwise follow the original logic.
        """
        if not self.pre_process and self.post_process and get_adaptor_args().schedule_method == 'dualpipev':
            return get_shared_embedding_from_dual_chunk()

        if self.has_vocab_embedding or getattr(self, 'mtp_process', False):
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        *,
        output_processor: Optional[Callable[..., Tensor]] = None,
        output_processor_context: Optional[Any] = None,
    ):
        """Builds a computation schedule plan for the model.

        This function creates a schedule plan for a model chunk, including
        preprocessing, transformer layers, and postprocessing.
        The schedule plan is used to optimize computation and memory usage
        in distributed environments.

        Args:
            input_ids (Tensor): Input token IDs.
            position_ids (Tensor): Position IDs.
            attention_mask (Tensor): Attention mask.
            decoder_input (Tensor, optional): Decoder input tensor. Defaults to None.
            labels (Tensor, optional): Labels for loss computation. Defaults to None.
            inference_context (BaseInferenceContext, optional):
                Inference context. Defaults to None.
            packed_seq_params (PackedSeqParams, optional):
                Parameters for packed sequences. Defaults to None.
            extra_block_kwargs (dict, optional):
                Additional keyword arguments for blocks. Defaults to None.
            runtime_gather_output (Optional[bool], optional):
                Whether to gather output at runtime. Defaults to None.
            inference_params (InferenceParams, optional):
                Parameters for inference. Defaults to None.
            loss_mask (Optional[Tensor], optional): Loss mask. Defaults to None.
            padding_mask (Optional[Tensor], optional): Padding mask. Defaults to None.
            output_processor (Callable, optional): Custom postprocess hook to run in the
                schedule-plan postprocess node instead of the default logits/loss path.
            output_processor_context (Any, optional): User-defined context object forwarded to
                `output_processor`.

        Returns:
            TransformerModelChunkSchedulePlan: The model chunk schedule plan.
        """

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()
        if self.config.moe_paged_stash:
            self.preprocess_for_paged_stash()

        from hcu_megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

        return TransformerModelChunkSchedulePlan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
            padding_mask,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
        )

    def backward_dw(self):
        self.decoder.backward_dw()

        if self.mtp_process:
            self.mtp.backward_dw()

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        padding_mask: Optional[Tensor] = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        args = get_adaptor_args()
        in_inference_mode = InferenceMode.is_active()

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            if padding_mask is not None:
                assert padding_mask.shape == input_ids.shape, (
                    f"padding_mask shape {padding_mask.shape} does not match "
                    f"input_ids shape {input_ids.shape}"
                )
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if padding_mask is not None and self.config.sequence_parallel:
                padding_mask = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        padding_mask.transpose(0, 1).contiguous()
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        # this is used to store combined cos/sin embeddings, exclusively for flash infer rope
        rotary_pos_cos_sin = None

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            use_flash_infer_fused_rope = (
                hasattr(inference_context, 'use_flashinfer_fused_rope')
                and inference_context.use_flashinfer_fused_rope
            )
            if (
                in_inference_mode
                and inference_context is not None
                and (self.config.flash_decode or use_flash_infer_fused_rope)
            ):
                assert (
                    not self.config.flash_decode
                ) or inference_context.is_static_batching(), (
                    "Flash decode is only applicable to static batching."
                )
                # Flash decoding uses precomputed cos and sin for RoPE
                if self.config.flash_decode:
                    rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                        inference_context.max_sequence_length,
                        self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                    )
                elif use_flash_infer_fused_rope:
                    assert not self.mtp_process, "MTP not tested with flashinfer_fused_rope"
                    rotary_pos_cos_sin = self.rotary_pos_emb_cache.setdefault(
                        inference_context.max_sequence_length,
                        torch.cat(
                            self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                            -1,
                        ),
                    )
            elif args.pipe_sp_splits != 1:
                rotary_pos_emb = self.rotary_pos_emb(
                    self.max_sequence_length,
                    packed_seq=packed_seq_params is not None
                               and packed_seq_params.qkv_format == 'thd',
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
        elif self.position_embedding_type == 'yarn':
            if not InferenceMode.is_active() or not self.config.flash_decode:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb, _ = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
            else:
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "YarnRotaryEmbedding yet."
                )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if not InferenceMode.is_active() or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(
                    position_ids,
                    self.mrope_section,
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            in_inference_mode
            and inference_context is not None
            and (self.config.cuda_graph_impl == "local" or self.config.flash_decode)
            and inference_context.is_static_batching()
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        else:
            sequence_len_offset = None

        if in_inference_mode:
            # Clear the outputs for padding tokens when using dynamic batching with
            # quantization scales to avoid corrupting amax calculations
            if (
                inference_context is not None
                and inference_context.is_dynamic_batching()
                and is_using_quantization_scales(self.config)
            ):
                decoder_input[inference_context.padding_slice] = 0.0

            # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
            # reference held by this caller function, enabling early garbage collection for
            # inference. Skip wrapping if decoder_input is logged after decoder completion.
            if not has_config_logger_enabled(self.config):
                decoder_input = WrappedTensor(decoder_input)

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        )
        if rotary_pos_cos_sin is not None:
            # only in the case of flashinfer fused rope will we
            # return this extra tensor
            # this is for backwards compatibility with
            # legacy unit tests, which break if you
            # return a 7 tuple instead of 6.
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        output_processor: Optional[Callable[..., Tensor]] = None,
        output_processor_context: Optional[Any] = None,
        micro_sp_idx=None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            padding_mask (Tensor, optional): Padding mask for MoE routing.
                Shape [bsz, seq_length]. True = padding (exclude), False = valid (include).
                Only used for MoE layers to exclude padding tokens from routing computations.
            output_processor (Callable, optional): Custom postprocess hook that receives
                decoder hidden states and output-layer helpers, then returns the model output.
            output_processor_context (Any, optional): User-defined context object forwarded to
                `output_processor`.
        """

        if self.has_vocab_embedding and (not self.pre_process):
            embedding_output = self.embedding(input_ids=input_ids, position_ids=position_ids)
            return embedding_output

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        if self.config.moe_paged_stash:
            self.preprocess_for_paged_stash()

        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preproc_output[:6]

        rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            micro_sp_idx=micro_sp_idx,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
        )
