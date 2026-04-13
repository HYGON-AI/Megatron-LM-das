from typing import Any, Optional
from functools import wraps

from torch import Tensor

from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import (
    deprecate_inference_params,
    nvtx_range_pop,
    nvtx_range_push,
)


def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    args = get_args()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_first_pp_stage = pp_rank == 0

    actual_rank = pp_rank if getattr(args, 'dualpipev_first_chunk', True) else 2 * pipeline_size - 1 - pp_rank
    if args.num_layers_to_build is not None:
        if isinstance(args.num_layers_to_build, int):
            return args.num_layers_to_build * actual_rank
        else:
            return sum(args.num_layers_to_build[:actual_rank])

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                middle_pipeline_stages *= 2

            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if middle_pipeline_stages > 0:
                num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
            else:
                num_layers_per_pipeline_rank = 0

            middle_pipeline_rank = (
                pp_rank
                if config.num_layers_in_first_pipeline_stage is None
                else pp_rank - 1
            )

            if not getattr(args, 'dualpipev_first_chunk', True):
                middle_pipeline_rank = (
                    config.pipeline_model_parallel_size
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.pipeline_model_parallel_size - 1
                ) + (config.pipeline_model_parallel_size - (pp_rank + 1))

            if getattr(args, 'dualpipev_first_chunk', True) and pp_rank == 0:
                    offset = 0
            else:
                offset = (
                    middle_pipeline_rank * num_layers_per_pipeline_rank
                ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size
            if args.schedule_method == 'dualpipev':
                num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

            if getattr(args, 'dualpipev_first_chunk', True):
                offset = pp_rank * num_layers_per_pipeline_rank
            else:
                offset = num_layers - (pp_rank + 1) * num_layers_per_pipeline_rank

            # Reduce the offset of embedding layer from the total layer number
            if config.account_for_embedding_in_pipeline_split:
                if not is_first_pp_stage:
                    offset -= 1
                elif not getattr(args, 'dualpipev_first_chunk', True):
                    offset -= 1
    else:
        offset = 0
    return offset


def transformer_layer_init_wrapper(transformer_layer_init_func):
    @wraps(transformer_layer_init_func)
    def wrapper(
        self,
        config,
        submodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        transformer_layer_init_func(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP

        if self.mlp.__class__ is MoELayer:
            if self.mlp.experts.__class__ is GroupedMLP:
                self.mlp.experts.layer_number = self.layer_number
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
            global_args = get_args()
            if global_args.n_shared_experts:
                self.mlp.shared_experts.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper


class TransformerLayer():

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
        micro_sp_idx=None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    self.input_layernorm, hidden_states
                )
        else:
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = self.input_layernorm(hidden_states)

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if using_fused_tp_inference_kernel:
            # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
            # operation in attention's out_proj (linear_proj)
            self._set_proj_residual(residual)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            micro_sp_idx=micro_sp_idx,
        )
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        if using_fused_tp_inference_kernel:
            # In inference optimized transformer layer, there is no bias and dropout
            # The remaining residual add is already handled inside the
            # self attention module.
            hidden_states = attention_output_with_bias[0]
        else:
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_output_with_bias, residual, self.hidden_dropout
                )
        nvtx_range_pop(suffix="self_attn_bda")

        # Delay the offload of the attention norm until after the self_attn_bda has been computed
        # because the residual is needed in the self_attn_bda.
        if self.offload_attn_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[residual]
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def backward_dw(self):
        self.self_attention.backward_dw()
        self.mlp.backward_dw()
