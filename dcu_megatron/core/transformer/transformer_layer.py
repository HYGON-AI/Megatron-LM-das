from typing import Any, Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import (
    deprecate_inference_params,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.transformer_layer import make_viewless_tensor

from dcu_megatron.core.pipeline_parallel import (
    PipelineOffloadManager,
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from .utils import DelayReleaseQKVLinearTensorContextManager


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

        self.offload_attn_norm = (
            self.config.fine_grained_activation_offloading
            and "attn_norm" in self.config.offload_modules
            and not isinstance(self.input_layernorm, IdentityOp)
        )
        self.offload_mlp_norm = (
            self.config.fine_grained_activation_offloading
            and "mlp_norm" in self.config.offload_modules
            and not isinstance(self.pre_mlp_layernorm, IdentityOp)
        )
        self.offload_qkv_linear = (
            self.config.fine_grained_activation_offloading
            and "qkv_linear" in self.config.offload_modules
        )

        # delay release input_layernorm_output
        self.delay_release_qkv_linear_tensor = self.offload_qkv_linear and isinstance(self.input_layernorm, IdentityOp)

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
        *,
        inference_params: Optional[Any] = None,
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

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        if self.offload_attn_norm:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="attn_norm")

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_attn_norm):
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    self.input_layernorm, hidden_states
                )
        else:
            with get_fine_grained_offloading_context(self.offload_attn_norm):
                input_layernorm_output = self.input_layernorm(hidden_states)

        if self.offload_attn_norm:
            input_layernorm_output, = fine_grained_offloading_group_commit(
                input_layernorm_output,
                name="attn_norm",
                forced_released_tensors=[hidden_states],
                delay_release_module="attn_norm",
            )

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        with DelayReleaseQKVLinearTensorContextManager(self.delay_release_qkv_linear_tensor):
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
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        if self.offload_attn_norm:
            cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
            release_func = cur_forward_chunk.module_release_func_map.pop("attn_norm", None)
            if release_func is not None:
                release_func()
                del release_func

        if self.offload_qkv_linear:
            cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
            release_func = cur_forward_chunk.module_release_func_map.pop("qkv_linear", None)
            if release_func is not None:
                release_func()
                del release_func

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

    def _forward_mlp(self, hidden_states, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        if self.offload_mlp_norm:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="mlp_norm")

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    self.pre_mlp_layernorm, hidden_states
                )
        else:
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        if self.offload_mlp_norm:
            pre_mlp_layernorm_output, = fine_grained_offloading_group_commit(
                pre_mlp_layernorm_output,
                name="mlp_norm",
                forced_released_tensors=[hidden_states],
                delay_release_module="mlp_norm",
            )

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )

        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)

        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="mlp_bda")

        if self.offload_mlp_norm:
            cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
            release_func = cur_forward_chunk.module_release_func_map.pop("mlp_norm", None)
            if release_func is not None:
                release_func()
                del release_func

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def backward_dw(self):
        self.self_attention.backward_dw()
        self.mlp.backward_dw()
