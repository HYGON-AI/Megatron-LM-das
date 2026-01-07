from copy import deepcopy
from collections import OrderedDict
from typing import Literal, Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
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
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MegatronCoreGPTModel

from dcu_megatron.core.transformer.transformer_block import GPTBlockWithMTPContextManager
from dcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk


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
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """
    in_inference_mode = inference_context is not None and not self.training
    if in_inference_mode:
        assert runtime_gather_output, "Inference must always gather TP logits"

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    if self.mtp_process:
        mtp_labels = labels.clone()
        hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(mtp_labels)
        for mtp_layer_number in range(self.config.mtp_num_layers):
            # output
            mtp_logits, _ = self.output_layer(
                hidden_states_list[mtp_layer_number + 1],
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
            )
            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
            loss_mask, num_tokens = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
            )
            mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
            mtp_loss = loss_mask * mtp_loss
            if self.training:
                # TODO(shifangx): remove the use of parallel_state here
                # after moving loss logging to loss_func in pretrain_gpt.py
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    mtp_layer_number,
                    self.config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    ),
                )
            mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
            if self.config.calculate_per_token_loss:
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states, mtp_loss_scale * mtp_loss
                )
            else:
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                )

    if (
        self.mtp_process is not None
        and getattr(self.decoder, "main_final_layernorm", None) is not None
    ):
        # move block main model final norms here
        hidden_states = self.decoder.main_final_layernorm(hidden_states)

    sequence_parallel_override = False
    if in_inference_mode and inference_context.materialize_only_last_token_logits:
        if inference_context.is_static_batching():
            hidden_states = hidden_states[-1:, :, :]
        else:
            if self.output_layer.sequence_parallel:
                # Perform the sequence parallel gather here instead of after the output layer
                # because we need to slice the last token logits from the full view of the
                # packed logits across all requests.
                # TODO(ksanthanam): Make the equivalent change in the `MambaModel` code after
                # merging in !3722.
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=self.pg_collection.tp
                )
                self.output_layer.sequence_parallel = False
                sequence_parallel_override = True

            # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
            # state ([B, H]) → unsqueeze back to [1, B, H]
            # (so that the output layer, which expects S×B×H, receives only the final token)
            hidden_states = inference_context.last_token_logits(
                hidden_states.squeeze(1).unsqueeze(0)
            ).unsqueeze(1)
    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )

    # Restore sequence parallel execution to the output layer if necessary.
    if sequence_parallel_override:
        assert (
            in_inference_mode
            and inference_context.is_dynamic_batching()
            and inference_context.materialize_only_last_token_logits
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


def gpt_model_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        if (
            self.config.schedule_method == "dualpipev"
            and not self.dualpipev_first_chunk
            and self.mtp_process
        ):
            self.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        return fn(self, *args, **kwargs)

    return wrapper


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

        args = get_args()
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
        self.mtp_process = mtp_block_spec is not None

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
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
        with GPTBlockWithMTPContextManager(self.mtp_process):
            self.decoder = TransformerBlock(
                config=self.config,
                spec=transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
                pg_collection=self.pg_collection,
                vp_stage=vp_stage,
            )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage
            )

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

            if args.parallel_linear_impl == "flux":
                from dcu_megatron.core.tensor_parallel.layers import FluxColumnParallelLinear

                self.output_layer = FluxColumnParallelLinear(
                    self.config.hidden_size,
                    self.vocab_size,
                    config=self.config,
                    init_method=self.config.init_method,
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
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                    tp_group=self.pg_collection.tp,
                )

        if self.pre_process or self.post_process:
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

        args = get_args()

        num_layers = self.decoder.num_layers_per_pipeline_rank
        if self.mtp_process:
            num_layers = num_layers + self.config.mtp_num_layers

        if args.schedule_method == "dualpipev":
            from dcu_megatron.core.pipeline_parallel.fine_grained_activation_offload_dualpipev import (
                fine_grained_offloading_init_chunk_handler,
            )
            fine_grained_offloading_init_chunk_handler(
                getattr(self, 'dualpipev_first_chunk', True),
                min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
            )
        else:
            from dcu_megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                fine_grained_offloading_init_chunk_handler,
            )
            # last_stage_is_loss = (pp_rank == pp_size - 1) and self.config.last_vp_stage_is_loss
            # TODO: will be an issue when dense layer is placed  across different pipeline stages
            fine_grained_offloading_init_chunk_handler(
                vp_size=self.config.virtual_pipeline_model_parallel_size,
                vp_stage=self.vp_stage,
                min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
            )

        if self.disable_param_offloading:
            for param in self.decoder.parameters():
                param.offloading_activation = False
            if self.mtp_process:
                for param in self.mtp.parameters():
                    param.offloading_activation = False
            if self.post_process:
                for param in self.output_layer.parameters():
                    param.offloading_activation = False
            self.disable_param_offloading = False

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: When dualpipe is enabled, return the weights from dual_chunk, otherwise follow the original logic.
        """
        if not self.pre_process and self.post_process and get_args().schedules_method == 'dualpipev':
            return get_shared_embedding_from_dual_chunk()

        if self.pre_process or self.mtp_process:
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

        Returns:
            TransformerModelChunkSchedulePlan: The model chunk schedule plan.
        """

        from ..common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

        if get_args().fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

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
        )

    def backward_dw(self):
        self.decoder.backward_dw()

        if self.mtp_process:
            self.mtp.backward_dw()
