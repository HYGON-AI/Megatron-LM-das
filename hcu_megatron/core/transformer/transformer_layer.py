# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Any, Optional, TYPE_CHECKING, Dict, Union
from functools import wraps
from dataclasses import dataclass, field

import torch
import functools
from torch import Tensor

from megatron.training import get_args
from megatron.core.transformer.enums import CudaGraphScope, LayerType
from megatron.core import parallel_state, tensor_parallel
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronCoreTransformerLayer
from megatron.core.typed_torch import apply_module

from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)
from hcu_megatron.core.tensor_parallel.random import CheckpointManager
from hcu_megatron.training.arguments import get_adaptor_args


@functools.lru_cache(maxsize=None)
def _get_offloading_interface():
    """Get the offloading interface for fine-grained activation offloading."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    args = get_adaptor_args()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()

    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_first_pp_stage = pp_rank == 0

    if args.schedule_method == 'dualpipev' and vp_stage is None:
        vp_stage = 1 - int(getattr(args, 'dualpipev_first_chunk', True))

    actual_rank = pp_rank if vp_stage == 0 else 2 * pipeline_size - 1 - pp_rank
    if args.num_layers_to_build is not None:
        if isinstance(args.num_layers_to_build, int):
            return args.num_layers_to_build * actual_rank
        else:
            return sum(args.num_layers_to_build[:actual_rank])

    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )

        elif (
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

            if vp_stage == 0:
                middle_pipeline_rank = (
                    pp_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pp_rank - 1
                )
            else:
                middle_pipeline_rank = (
                    config.pipeline_model_parallel_size
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.pipeline_model_parallel_size - 1
                ) + (config.pipeline_model_parallel_size - (pp_rank + 1))

            if vp_stage == 0 and pp_rank == 0:
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

            if vp_stage == 0:
                offset = pp_rank * num_layers_per_pipeline_rank
            else:
                offset = num_layers - (pp_rank + 1) * num_layers_per_pipeline_rank

            # Reduce the offset of embedding layer from the total layer number
            if config.account_for_embedding_in_pipeline_split:
                if not is_first_pp_stage:
                    offset -= 1
                elif vp_stage == 1:
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
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        transformer_layer_init_func(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.experts import SequentialMLP

        if self.mlp.__class__ is MoELayer:
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
            global_args = get_args()
            if global_args.n_shared_experts:
                self.mlp.shared_experts.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm: Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm: Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm: Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: LayerNormBuilder = IdentityOp
    self_attention_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: LayerNormBuilder = IdentityOp
    cross_attention_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


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
        inference_context=None,
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

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    apply_module(self.input_layernorm), hidden_states
                )
        else:
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = apply_module(self.input_layernorm)(hidden_states)

        if isinstance(input_layernorm_output, tuple):
            if len(input_layernorm_output) != 2:
                raise ValueError(
                    f"When the output of input_layernorm is a tuple, it is "
                    f"expected to have 2 elements (output, residual), but "
                    f"got {len(input_layernorm_output)}"
                )
            input_layernorm_output, residual = input_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        using_fused_tp_inference_kernel = (
            InferenceMode.is_active() and self.config.inference_fuse_tp_communication
        )

        if using_fused_tp_inference_kernel:
            # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
            # operation in attention's out_proj (linear_proj)
            self._set_proj_residual(residual)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        if micro_sp_idx is not None:
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
        else:
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

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = apply_module(self.pre_cross_attn_layernorm)(hidden_states)

        if isinstance(pre_cross_attn_layernorm_output, tuple):
            if len(pre_cross_attn_layernorm_output) != 2:
                raise ValueError(
                    f"When the output of pre_cross_attn_layernorm_output "
                    f"is a tuple, it is expected to have 2 elements "
                    f"(output, residual), but "
                    f"got {len(pre_cross_attn_layernorm_output)}"
                )
            pre_cross_attn_layernorm_output, residual = pre_cross_attn_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()
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
        if self.is_moe_layer:
            self.mlp.backward_dw(routed_experts=True, shared_experts=True)
            return
        self.mlp.backward_dw()

    def __call__(self, *args, **kwargs):
        # Extract mhc_recompute_manager before CUDA graph manager processes kwargs,
        # since CheckpointManager is not a CUDA-graph-supported type.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        kwargs.pop("is_last_layer_in_recompute_block", None)
        return super().__call__(*args, **kwargs)


class HyperConnectionTransformerLayer(MegatronCoreTransformerLayer):
    """A transformer layer with Manifold-Constrained Hyper-Connections (mHC).

    Extends TransformerLayer by adding hyper connection modules around self-attention
    and MLP. The n-stream hidden states are aggregated before each sub-layer and
    expanded back afterwards using learned mappings (H_pre, H_post, H_res).

    Cross-attention hyper connection is not supported.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
    ):
        self.submodules_config = submodules
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
        )

        if self.submodules_config.cross_attention_hyper_connection is not IdentityOp:
            raise ValueError(
                "HyperConnectionTransformerLayer does not support cross-attention "
                "hyper connections. Use IdentityOp for cross_attention_hyper_connection."
            )

        assert self.submodules_config.self_attention_hyper_connection is not IdentityOp, (
            "HyperConnectionTransformerLayer requires self_attention_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )
        assert self.submodules_config.mlp_hyper_connection is not IdentityOp, (
            "HyperConnectionTransformerLayer requires mlp_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )

        self.self_attention_hyper_connection = build_module(
            self.submodules_config.self_attention_hyper_connection,
            config=self.config,
            layer_number=self.layer_number,
            hc_type='attn',
        )

        self.mlp_hyper_connection = build_module(
            self.submodules_config.mlp_hyper_connection, config=self.config, layer_number=self.layer_number, hc_type='mlp',
        )

        # When mHC recompute is active, skip checkpointing if the layernorm
        # is IdentityOp (fused into TE linear) — there is nothing to recompute.
        self.mhc_checkpoint_input_layernorm = not isinstance(self.input_layernorm, IdentityOp)
        self.mhc_checkpoint_pre_mlp_layernorm = not isinstance(self.pre_mlp_layernorm, IdentityOp)
        self.off_interface = _get_offloading_interface()

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. The base class returns [s, b, C], but mHC layers operate on
        n-stream hidden states of shape [s, b, n*C].
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        hs = static_inputs["hidden_states"]
        n = self.config.num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )

        # Add input_ids for hash-based MoE routing under CUDA graphs.
        # Only add for layers that actually use hash routing,
        # since other layers (e.g. on later PP stages) receive input_ids=None.
        if (
            self.is_moe_layer
            and self.config.moe_n_hash_layers > 0
            and getattr(self.mlp.router, 'is_hash_layer', False)
        ):
            static_inputs["input_ids"] = torch.zeros(
                (micro_batch_size, seq_length), dtype=torch.long, device=torch.cuda.current_device()
            )

        return static_inputs

    def _get_submodules_under_cudagraphs(self):
        """Override to include hyper connection modules.

        The base TransformerLayer._get_submodules_under_cudagraphs does not include
        self_attention_hyper_connection / mlp_hyper_connection. Their learnable
        parameters (mapping_proj, alpha_*, bias) need manual pre-forward hooks
        during CUDA graph replay so that parameter all-gathers are triggered.
        """
        submodules = super()._get_submodules_under_cudagraphs()

        if not self.config.cuda_graph_scope:
            return submodules

        if CudaGraphScope.attn in self.config.cuda_graph_scope:
            submodules.append(self.self_attention_hyper_connection)
        if (not self.is_moe_layer and CudaGraphScope.mlp in self.config.cuda_graph_scope) or (
            self.is_moe_layer
            and (
                CudaGraphScope.moe in self.config.cuda_graph_scope
                or CudaGraphScope.moe_router in self.config.cuda_graph_scope
            )
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules

    def forward(self, *args, **kwargs):
        """Forward pass with MHC recompute manager support."""
        kwargs.pop("dynamic_inference_decode_only", None)

        mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)

        hidden_states, context = self._forward_attention(
            *args, mhc_recompute_manager=mhc_recompute_manager, **kwargs
        )

        output = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
            input_ids=kwargs.get("input_ids", None),
            mhc_recompute_manager=mhc_recompute_manager,
        )
        return output, context

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
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager: Optional['CheckpointManager'] = None,
        *,
        inference_params: Optional[Any] = None,
        micro_sp_idx=None,
    ):
        """Forward attention with hyper connection pre/post processing on self-attention."""
        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = hidden_states

        nvtx_range_push(suffix="self_attention_hyper_connection")
        hidden_states, self_attn_h_res, self_attn_hc_h_post = self.self_attention_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        # Optional Input Layer norm

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        checkpoint_input_layernorm = self.recompute_input_layernorm or (
            mhc_recompute_manager is not None and self.mhc_checkpoint_input_layernorm
        )
        attn_norm_manager = self.off_interface(self.offload_attn_norm, hidden_states, "attn_norm")
        if checkpoint_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            with attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    self.input_layernorm, hidden_states
                )
        else:
            with attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm(hidden_states)

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
        )
        nvtx_range_pop(suffix="self_attention")

        if checkpoint_input_layernorm:
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        nvtx_range_push(suffix="self_attention_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                self_attn_h_res,
                residual,
                self_attn_hc_h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_recompute_manager,
            )
        nvtx_range_pop(suffix="self_attention_fused_h_res_h_post_bda")

        if self.offload_attn_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[residual]
            )

        # Cross-attention (no hyper connection support).
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states,
        inference_context=None,
        padding_mask=None,
        input_ids=None,
        mhc_recompute_manager: Optional['CheckpointManager'] = None,
    ):
        """Forward MLP with hyper connection pre/post processing."""
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not
            None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_mlp_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager

        residual = hidden_states

        nvtx_range_push(suffix="mlp_hyper_connection")
        hidden_states, mlp_h_res, mlp_hc_h_post = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        nvtx_range_pop(suffix="mlp_hyper_connection")

        # Optional Layer norm post the cross-attention.
        checkpoint_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm or (
            mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        )
        self.mlp_norm_manager = self.off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm")
        if checkpoint_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            with self.mlp_norm_manager as hidden_states:
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    self.pre_mlp_layernorm, hidden_states
                )
        else:
            with self.mlp_norm_manager as hidden_states:
                pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        with self.mlp_norm_manager as hidden_states:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        nvtx_range_push(suffix="mlp")
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
            and not self.config.transformer_impl == "inference_optimized"
        )

        moe_kwargs = {}
        if self.is_moe_layer and input_ids is not None:
            moe_kwargs['input_ids'] = input_ids

        if self.recompute_mlp:
            if self.config.fp8 or self.config.fp4:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                    padding_mask=padding_mask,
                    **moe_kwargs,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    functools.partial(self.mlp, padding_mask=padding_mask, **moe_kwargs),
                    False,
                    pre_mlp_layernorm_output,
                )
        elif should_chunk_mlp_for_prefill:
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)
            outputs = [self.mlp(chunk) for chunk in chunks]
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)
        else:
            mlp_output_with_bias = self.mlp(
                pre_mlp_layernorm_output, padding_mask=padding_mask, **moe_kwargs
            )

        nvtx_range_pop(suffix="mlp")

        # During TE CUDA graph partial MoE capture, skip HC post-processing and return
        # intermediate outputs + HC state. The post-processing will be done during replay.
        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and CudaGraphScope.moe_router in self.config.cuda_graph_scope
        ):
            if self.recompute_pre_mlp_layernorm or (
                mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
            ):
                for tensor in mlp_output_with_bias:
                    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(tensor)
            # Append HC state (mlp_hc_h_post, mlp_h_res, residual) for replay.
            return list(mlp_output_with_bias) + [mlp_hc_h_post, mlp_h_res, residual]

        return self._forward_post_mlp_with_fused_hyper_connection(
            mlp_output_with_bias, mlp_h_res, residual, mlp_hc_h_post, mhc_mlp_bda_manager
        )

    def _forward_post_mlp_with_fused_hyper_connection(
        self,
        mlp_output_with_bias,
        mlp_h_res,
        residual,
        mlp_hc_h_post,
        mhc_mlp_bda_recompute_manager: Optional['CheckpointManager'] = None,
    ):
        """
        Perform operations after the MLP computation with fused hyper connection kernel.

        This method uses the fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        Args:
            mlp_output_with_bias (Tensor): Output tensor of the MLP layer with bias.
            mlp_h_res (Tensor): [s, b, n, n] - residual mixing matrix from hyper connection.
            residual (Tensor): [s, b, n*C] - original residual (n-stream hidden states).
            mlp_hc_h_post (Tensor): [s, b, n] - expansion weights from hyper connection.
            mhc_recompute_manager: Optional CheckpointManager for checkpoint management.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )
        if self.recompute_pre_mlp_layernorm or (
            mhc_mlp_bda_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        ):
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        nvtx_range_push(suffix="mlp_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_hyper_connection.fused_h_res_h_post_bda(
                mlp_h_res,
                residual,
                mlp_hc_h_post,
                mlp_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_mlp_bda_recompute_manager,
            )
        nvtx_range_pop(suffix="mlp_fused_h_res_h_post_bda")

        if self.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output

    def _te_cuda_graph_replay_impl(self, args, kwargs, context):
        """Implementation of _te_cuda_graph_replay with hyper connection support.

        Overrides the parent's _te_cuda_graph_replay_impl so that the
        delay_offload_until_cuda_graph lifecycle (enter_replay/exit_replay) in
        the parent's _te_cuda_graph_replay is preserved.

        During MoE partial CUDA graph capture, the graph outputs include HC state
        (mlp_hc_h_post, mlp_h_res) in addition to the base class outputs. This method
        extracts the HC state and uses it for post-processing after resuming the MoE forward.
        """
        cuda_graph_output = list(
            GraphableMegatronModule._te_cuda_graph_replay(self, *args, **kwargs)
        )

        # Flush delayed offload groups from previous layers after graph replay.
        if self.config.delay_offload_until_cuda_graph:
            self.off_interface.flush_delayed_groups()

        if kwargs.get('context') is not None:
            context = cuda_graph_output.pop()

        if (
            not self.config.cuda_graph_scope
            or (not self.is_moe_layer and CudaGraphScope.mlp in self.config.cuda_graph_scope)
            or (self.is_moe_layer and CudaGraphScope.moe in self.config.cuda_graph_scope)
        ):
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
            assert (
                not self.config.overlap_moe_expert_parallel_comm
            ), "EP overlap must be \
                disabled when CUDA graph captures the whole MLP/MoE part."
        elif self.is_moe_layer and CudaGraphScope.moe_router in self.config.cuda_graph_scope:
            # Pop HC state (appended during capture in _forward_mlp).
            residual = cuda_graph_output.pop()
            mlp_h_res = cuda_graph_output.pop()
            mlp_hc_h_post = cuda_graph_output.pop()

            shared_expert_output, routing_map = None, None
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                shared_expert_output = cuda_graph_output.pop()

            if CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope:
                (hidden_states, probs), attr_outputs = (
                    cuda_graph_output[:2],
                    cuda_graph_output[2:],
                )
                valid_cudagraph_attrs = self.mlp.token_dispatcher.valid_cudagraph_attrs
                assert len(attr_outputs) == len(
                    valid_cudagraph_attrs
                ), f"attr_outputs: {len(attr_outputs)} != {len(valid_cudagraph_attrs)}"
                for i, attr_name in enumerate(valid_cudagraph_attrs):
                    self.mlp.token_dispatcher.set_cudagraph_attr(attr_name, attr_outputs[i])
            else:
                assert len(cuda_graph_output) == 3, (
                    "CUDA graph output should be [hidden_states, probs, routing_map], "
                    f"but got {len(cuda_graph_output)} elements"
                )
                hidden_states, probs, routing_map = cuda_graph_output

            # Resume the MoELayer forward pass from the end of the CUDA graph scope.
            nvtx_range_push(suffix="mlp")
            self.mlp.cudagraph_tensor_store.set(
                hidden_states=hidden_states,
                probs=probs,
                routing_map=routing_map,
                shared_expert_output=shared_expert_output,
            )
            # If EP overlap is enabled, remaining of mlp will be called as fine_grained_callables
            # and should be skipped here.
            if self.config.overlap_moe_expert_parallel_comm:
                probs, routing_map = self.mlp.route(hidden_states)
                hidden_states, probs = self.mlp.preprocess(hidden_states, probs, routing_map)
                nvtx_range_pop(suffix="mlp")
                return residual, hidden_states, probs, shared_expert_output
            mlp_output_with_bias = self.mlp(hidden_states)
            self.mlp.cudagraph_tensor_store.clear()
            nvtx_range_pop(suffix="mlp")

            # HC post-processing with fused h_res, h_post and BDA.
            recompute_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm
            self.recompute_pre_mlp_layernorm = False
            output = self._forward_post_mlp_with_fused_hyper_connection(
                mlp_output_with_bias, mlp_h_res, residual, mlp_hc_h_post
            )
            self.recompute_pre_mlp_layernorm = recompute_pre_mlp_layernorm
        else:
            output = self._forward_mlp(*cuda_graph_output, input_ids=kwargs.get("input_ids", None))
        return output, context
