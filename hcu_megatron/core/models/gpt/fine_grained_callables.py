# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from contextlib import nullcontext
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode, make_viewless
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.typed_torch import apply_module, copy_signature
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode as MegatronCoreTransformerLayerNode

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from hcu_megatron.training.arguments import get_adaptor_args


class TransformerLayerNode(MegatronCoreTransformerLayerNode):
    """Base class for transformer layer computation nodes.

    This class provides common functionality for different types of
    transformer layer nodes (attention, MLP, etc.)
    """

    def forward(self, inputs=(), stream_wait_event=None, stream_record_event=None, is_recompute=False):
        """Schedule node forward"""
        self.is_recompute = is_recompute
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        output = self._forward(
                *inputs,
                stream_wait_event=stream_wait_event,
                stream_record_event=stream_record_event,
                is_recompute=is_recompute,
            )
        if self.is_layer_last_node:
            self._post_forward_hook()
        return output

    def _forward(self, *inputs, stream_wait_event=None, stream_record_event=None, is_recompute=False):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} forward"):
            if stream_wait_event is not None:
                stream_wait_event.wait(self.stream)

            self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
            for i, input in enumerate(self.inputs):
                if input is not None:
                    input.requires_grad = inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data, is_recompute=is_recompute)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple([make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data])

            if not get_adaptor_args().integrate_recompute_to_ep_comm_overlap or is_recompute:
                self.outputs = data

            if stream_record_event is not None:
                stream_record_event.record(self.stream)

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            for input in inputs:
                if input is not None:
                    input.record_stream(self.stream)
                    input.untyped_storage().resize_(0)

        return data

    def forward_impl(self, *args, is_recompute=False):
        """Calls the submodule as the forward pass."""
        return self.submodule(self, *args, is_recompute=is_recompute)

    def backward(self, *output_grad, stream_wait_event=None, stream_record_event=None):
        """Execute backward pass and corresponding hooks."""
        grads = super(MegatronCoreTransformerLayerNode, self).backward(
            *output_grad,
            stream_wait_event=stream_wait_event,
            stream_record_event=stream_record_event,
        )
        if not self.delay_wgrad_compute and self.is_layer_first_node:
            self._post_backward_hook()
        return grads

    def detach(self, t):
        """Detaches a tensor and stores it for backward computation."""
        if get_adaptor_args().integrate_recompute_to_ep_comm_overlap and not self.is_recompute:
            return t
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        if not get_adaptor_args().integrate_recompute_to_ep_comm_overlap or self.is_recompute:
            self.before_detached = self.before_detached + (t,)
            self.detached = self.detached + (detached,)
        return detached


def build_transformer_layer_callables(layer: TransformerLayer):
    """Create callables for transformer layer nodes.
    Divides the transformer layer's operations into a sequence of smaller, independent
    functions. This decomposition separates computation-heavy tasks (e.g., self-attention,
    MLP) from communication-heavy tasks (e.g., MoE's All-to-All).

    The five callables are:
    1. Attention (computation)
    2. Post-Attention (computation)
    3. MoE Dispatch (communication)
    4. MLP / MoE Experts (computation)
    5. MoE Combine (communication)

    By assigning these functions to different CUDA streams (e.g., a compute stream
    and a communication stream), the scheduler can overlap their execution, preventing
    tasks from competing for resources and hiding communication latency by running them
    in parallel with functions from other micro-batches.

    Args:
        layer: The transformer layer to build callables for.

    Returns:
        A tuple containing:
        - forward_funcs: List of callable functions for the layer
        - backward_dw: Dict of weight gradient functions for the layer
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "deepep"
    )
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )

    def submodule_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor, is_recompute=False):
        """
        Performs same attnention forward logic as GPT Model and forward pass for
        computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """

        if (
            isinstance(layer, GraphableMegatronModule)
            and hasattr(layer, 'cuda_graphs')
            and layer.cuda_graphs
        ):
            layer.set_te_cuda_graph_backward_dw_wrapper()
            forward_func = layer._te_cuda_graph_replay
        else:
            # wrapper function that keeps consistent api with cuda graph replay
            def forward_func(
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                rotary_pos_emb: Optional[Tensor] = None,
                rotary_pos_cos: Optional[Tensor] = None,
                rotary_pos_sin: Optional[Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                sequence_len_offset: Optional[Tensor] = None,
            ):
                hidden_states, _ = layer._forward_attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
                if not isinstance(layer.mlp, MoELayer):
                    return hidden_states, None, None, None
                if layer.recompute_pre_mlp_layernorm:
                    layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                    with off_interface(
                        layer.offload_mlp_norm, hidden_states, "mlp_norm"
                    ) as hidden_states:
                        pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                            apply_module(layer.pre_mlp_layernorm), hidden_states
                        )
                else:
                    with off_interface(
                        layer.offload_mlp_norm, hidden_states, "mlp_norm"
                    ) as hidden_states:
                        pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(
                            hidden_states
                        )

                # When using fused residual norm (e.g. TEFusedResidualRMSNorm),
                # the layernorm returns (normalized_output, residual). Unpack
                # and use the fused residual for the downstream BDA connection.
                if isinstance(pre_mlp_layernorm_output, tuple):
                    if len(pre_mlp_layernorm_output) != 2:
                        raise ValueError(
                            f"When the output of pre_mlp_layernorm is a tuple, it is "
                            f"expected to have 2 elements (output, residual), but "
                            f"got {len(pre_mlp_layernorm_output)}"
                        )
                    pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

                shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
                local_tokens, probs = layer.mlp.preprocess(
                    pre_mlp_layernorm_output, probs, routing_map, is_recompute=is_recompute
                )
                return hidden_states, local_tokens, probs, shared_expert_output

        hidden_states, local_tokens, probs, shared_expert_output = forward_func(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )
        if not isinstance(layer.mlp, MoELayer):
            return hidden_states

        # Detach here for mlp_bda residual connection
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            # Detach here for shared expert connection in moe_combine
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor, is_recompute=False,
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs, is_recompute=is_recompute,)

        # `dispatched_probs` is needed by backward pass of swiglu, therefore it's
        # passed to moe_forward within `layer_state` to avoid the free_input process
        # of the input tensors.
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: torch.Tensor, is_recompute=False,):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs, is_recompute=is_recompute,)

        # For HybridEP, tokens_per_expert is generated on comm stream, as the input to
        # `routed_experts_compute`, a ref is needed to prevent it from being freed.
        if enable_hybridep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def submodule_combine_forward(node: ScheduleNode, output: torch.Tensor, is_recompute=False,):
        """
        # Triggers token combine and the remaining computation in the transformer layer.
        # The `mlp_bda` computation is placed after `mlp.combine` due to data dependency.
        # This ordering is also critical for pipeline performance. Starting the `mlp.combine`
        # communication at first allows it to be overlapped with computation from another
        # microbatch. If `mlp_bda` were to run first, it would compete for SM resources
        # with another microbatch's computation and expose the communication.
        """
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        output = layer.mlp.combine(output, is_recompute=is_recompute,)
        output = layer.mlp.postprocess(output, shared_expert_output, is_recompute=is_recompute,)

        mlp_output_with_bias = (output, None)
        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        if layer.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record tensors created on comp stream to comm stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # final layer norm from decoder
        final_layernorm = node.chunk_state.model.decoder.final_layernorm
        if not node.is_mtp and final_layernorm and node.is_last_layer:
            output = final_layernorm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    @copy_signature(layer._forward_mlp, handle_first_dst_param='preserve')
    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
        kwargs.pop("is_recompute", None)
        return layer._forward_mlp(*args, **kwargs)

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    attn_func = submodule_attn_forward
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    layer.init_backward_dw_wrapper()

    forward_funcs = [attn_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"attn": layer.backward_dw_wrapper, "mlp": layer.mlp}
    return forward_funcs, backward_dw


def build_mtp_layer_callables_without_split_attn(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables(layer.mtp_model_layer)
    attn_forward, dispatch_forward, mlp_forward, combine_forward, _ = forward_funcs
    is_moe = isinstance(layer.mtp_model_layer.mlp, MoELayer)
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_attn_forward(node, hidden_states, is_recompute=False,):
        # MTP Block Preprocess
        if node.is_first_layer:
            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]
            if (
                get_adaptor_args().schedule_method == "dualpipev"
                and node.chunk_state.model.embedding.word_embeddings.weight is None
            ):
                from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
                node.chunk_state.model.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        input_ids, position_ids, padding_mask, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
            packed_seq_params=node.chunk_state.packed_seq_params,
            padding_mask=node.chunk_state.padding_mask,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids
        node.chunk_state.padding_mask = padding_mask

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        assert (
            node.chunk_state.packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return attn_forward(node, hidden_states, is_recompute=is_recompute,)

    def submodule_mtp_postprocess_forward(node, hidden_states, is_recompute=False,):
        hidden_states = layer._postprocess(hidden_states)
        node.chunk_state.mtp_hidden_states.append(hidden_states)
        if node.is_last_layer:
            hidden_states = torch.cat(node.chunk_state.mtp_hidden_states, dim=0)
            node.chunk_state.mtp_hidden_states = None
        return hidden_states

    def rng_context_wrapper(func, *args, **kwargs):
        """
        Wrapper to add rng context to submodule callables
        """
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            return func(*args, **kwargs)

    # Build forward and backward callable functions
    # attn_forward already has rng context, no need to wrap
    attn_func = submodule_mtp_attn_forward
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [attn_func, dispatch_func, mlp_func, combine_func, mtp_post_process_func]
    if isinstance(backward_dw["attn"], list):
        backward_dw["attn"].append(layer.eh_proj)
    else:
        backward_dw["attn"] = [backward_dw["attn"], layer.eh_proj]

    return forward_funcs, backward_dw


def build_layer_callables_without_split_attn(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer and MultiTokenPredictionLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables_without_split_attn(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")


def build_transformer_layer_callables_with_split_attn(layer: TransformerLayer):
    """Create callables for transformer layer nodes.
    Divides the transformer layer's operations into a sequence of smaller, independent
    functions. This decomposition separates computation-heavy tasks (e.g., self-attention,
    MLP) from communication-heavy tasks (e.g., MoE's All-to-All).

    The five callables are:
    1. Attention (computation)
    2. Post-Attention (computation)
    3. MoE Dispatch (communication)
    4. MLP / MoE Experts (computation)
    5. MoE Combine (communication)

    By assigning these functions to different CUDA streams (e.g., a compute stream
    and a communication stream), the scheduler can overlap their execution, preventing
    tasks from competing for resources and hiding communication latency by running them
    in parallel with functions from other micro-batches.

    Args:
        layer: The transformer layer to build callables for.

    Returns:
        A tuple containing:
        - forward_funcs: List of callable functions for the layer
        - backward_dw: Dict of weight gradient functions for the layer
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "deepep"
    )
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )

    def submodule_attention_qkv_forward(
        node: ScheduleNode,
        hidden_states: torch.Tensor,
        is_recompute=False,
    ):
        # Optional Input Layer norm
        if layer.recompute_input_layernorm:
            layer.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(layer.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = layer.input_layernorm_checkpoint.checkpoint(
                    apply_module(layer.input_layernorm), hidden_states
                )
        else:
            with off_interface(layer.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = apply_module(layer.input_layernorm)(hidden_states)

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

        if layer.config.fp32_residual_connection:
            residual = residual.float()

        using_fused_tp_inference_kernel = (
            InferenceMode.is_active() and layer.config.inference_fuse_tp_communication
        )

        if using_fused_tp_inference_kernel:
            # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
            # operation in attention's out_proj (linear_proj)
            layer._set_proj_residual(residual)

        # Self attention.
        qkv_output = layer.self_attention.compute_qkv(
            input_layernorm_output,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
        )

        # Detach here for residual residual connection
        node.layer_state.attn_residual = node.detach(residual)

        return qkv_output

    def submodule_attention_core_attn_forward(
        node: ScheduleNode,
        *qkv_output,
        is_recompute=False,
    ):
        core_attn_out = layer.self_attention.compute_attn(
            qkv_output,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )

        return core_attn_out

    def submodule_attention_proj_forward(
        node: ScheduleNode,
        core_attn_out,
        is_recompute=False,
    ):

        attn_residual = node.layer_state.attn_residual

        attention_output_with_bias = layer.self_attention.compute_proj(
            core_attn_out
        )

        if layer.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            layer.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.self_attn_bda(layer.training, layer.config.bias_dropout_fusion)(
                attention_output_with_bias, attn_residual, layer.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # Delay the offload of the attention norm until after the self_attn_bda has been computed
        # because the residual is needed in the self_attn_bda.
        if layer.offload_attn_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[attn_residual]
            )

        node.layer_state.attn_residual = None

        return hidden_states

    def submodule_attention_proj_router_shared_expert_compound_forward(
        node: ScheduleNode,
        core_attn_out,
        is_recompute=False,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """

        hidden_states = submodule_attention_proj_forward(node, core_attn_out,)

        # Optional Layer norm post the cross-attention.
        if layer.recompute_pre_mlp_layernorm:
            layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(layer.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
                pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                    apply_module(layer.pre_mlp_layernorm), hidden_states
                )
        else:
            with off_interface(layer.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
                pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(hidden_states)

        # When using fused residual norm (e.g. TEFusedResidualRMSNorm),
        # the layernorm returns (normalized_output, residual). Unpack
        # and use the fused residual for the downstream BDA connection.
        if isinstance(pre_mlp_layernorm_output, tuple):
            if len(pre_mlp_layernorm_output) != 2:
                raise ValueError(
                    f"When the output of pre_mlp_layernorm is a tuple, it is "
                    f"expected to have 2 elements (output, residual), but "
                    f"got {len(pre_mlp_layernorm_output)}"
                )
            pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

        shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
        probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
        local_tokens, probs = layer.mlp.preprocess(
            pre_mlp_layernorm_output, probs, routing_map, is_recompute=is_recompute
        )

        # Detach here for mlp_bda residual connection
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            # Detach here for shared expert connection in moe_combine
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor, is_recompute=False,
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs, is_recompute=is_recompute,)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
   
        return dispatched_tokens

    def submodule_routed_experts_forward(node: ScheduleNode, dispatched_input, is_recompute=False):
        """
        Performs a forward pass for the MLP submodule, including only routed-expert computations.
        """
        def custom_forward(
            dispatched_input, permuted_probs
        ):
            expert_output, mlp_bias = layer.mlp.routed_experts_compute(dispatched_input, permuted_probs, is_recompute=is_recompute)
            assert mlp_bias is None, f"mlp_bias is not supported for {type(layer.mlp.token_dispatcher)}"
            return expert_output

        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        args = [
            dispatched_input,
            dispatched_probs,
        ]
        if layer.mlp.moe_layer_recompute:
            if layer.config.fp8:
                expert_output = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *args,
                )
            else:
                expert_output = tensor_parallel.checkpoint(
                    custom_forward, False, *args
                )
        else:
            expert_output = custom_forward(*args)
        del args

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        node.layer_state.dispatched_probs = None
        return expert_output

    def submodule_combine_forward(
        node: ScheduleNode,
        output: torch.Tensor,
        is_recompute=False,
    ):
        """
        # Triggers token combine and the remaining computation in the transformer layer.
        # The `mlp_bda` computation is placed after `mlp.combine` due to data dependency.
        # This ordering is also critical for pipeline performance. Starting the `mlp.combine`
        # communication at first allows it to be overlapped with computation from another
        # microbatch. If `mlp_bda` were to run first, it would compete for SM resources
        # with another microbatch's computation and expose the communication.
        """
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        output = layer.mlp.combine(output, is_recompute=is_recompute,)
        output = layer.mlp.postprocess(output, shared_expert_output, is_recompute=is_recompute,)
        mlp_output_with_bias = (output, None)

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )

        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        if layer.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record residual to comm stream, since it's created on comp stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            node.layer_state.shared_expert_output.record_stream(torch.cuda.current_stream())
        # release tensor reference after use
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # final layer norm from decoder
        final_layernorm = node.chunk_state.model.decoder.final_layernorm
        if not node.is_mtp and final_layernorm and node.is_last_layer:
            output = final_layernorm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
        kwargs.pop("is_recompute", None)
        output = layer._forward_mlp(*args, **kwargs)
        return output

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    attn_qkv_func = submodule_attention_qkv_forward
    core_attn_func = submodule_attention_core_attn_forward
    attn_proj_func = submodule_attention_proj_router_shared_expert_compound_forward if is_moe else submodule_attention_proj_forward
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_routed_experts_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    forward_funcs = [attn_qkv_func, core_attn_func, attn_proj_func, dispatch_func, mlp_func, combine_func, None]
    attn_proj_dw_funcs = [layer.self_attention.linear_proj]
    if is_moe and layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
        attn_proj_dw_funcs.append(layer.mlp.shared_experts)

    if isinstance(layer.self_attention, MLASelfAttention):
        attn_qkv_dw_funcs = [
            layer.self_attention.linear_kv_up_proj,
            layer.self_attention.linear_kv_down_proj,
        ]
        if layer.config.q_lora_rank is None:
            attn_qkv_dw_funcs.append(
                layer.self_attention.linear_q_proj
            )
        else:
            attn_qkv_dw_funcs.extend([
                layer.self_attention.linear_q_down_proj,
                layer.self_attention.linear_q_up_proj
            ])
    else:
        attn_qkv_dw_funcs = layer.self_attention.linear_qkv

    backward_dw = {
        "attn_qkv": attn_qkv_dw_funcs,
        "attn_proj": attn_proj_dw_funcs,
        "mlp": layer.mlp.experts if is_moe else layer.mlp
    }

    return forward_funcs, backward_dw


def build_mtp_layer_callables_with_split_attn(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables_with_split_attn(layer.mtp_model_layer)
    attn_qkv_forward, core_attn_forward, attn_proj_forward, dispatch_forward, mlp_forward, combine_forward, _ = (
        forward_funcs
    )
    is_moe = isinstance(layer.mtp_model_layer.mlp, MoELayer)
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_attn_qkv_forward(node, hidden_states, is_recompute=False):
        # MTP Block Preprocess
        if node.is_first_layer:
            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]
            if (
                get_adaptor_args().schedule_method == "dualpipev"
                and node.chunk_state.model.embedding.word_embeddings.weight is None
            ):
                from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
                node.chunk_state.model.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        input_ids, position_ids, padding_mask, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
            packed_seq_params=node.chunk_state.packed_seq_params,
            padding_mask=node.chunk_state.padding_mask,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids
        node.chunk_state.padding_mask = padding_mask

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        assert (
            node.chunk_state.packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return attn_qkv_forward(node, hidden_states, is_recompute=is_recompute)

    def submodule_mtp_postprocess_forward(node, hidden_states, is_recompute=False):
        hidden_states = layer._postprocess(hidden_states)
        node.chunk_state.mtp_hidden_states.append(hidden_states)
        if node.is_last_layer:
            hidden_states = torch.cat(node.chunk_state.mtp_hidden_states, dim=0)
            node.chunk_state.mtp_hidden_states = None
        return hidden_states

    def rng_context_wrapper(func, *args, **kwargs):
        """
        Wrapper to add rng context to submodule callables
        """
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            return func(*args, **kwargs)

    # Build forward and backward callable functions
    # attn_forward already has rng context, no need to wrap
    attn_qkv_func = submodule_mtp_attn_qkv_forward
    core_attn_func = partial(rng_context_wrapper, core_attn_forward)
    attn_proj_func = partial(rng_context_wrapper, attn_proj_forward)
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [
        attn_qkv_func,
        core_attn_func,
        attn_proj_func,
        dispatch_func,
        mlp_func,
        combine_func,
        mtp_post_process_func,
    ]

    attn_proj_dw_funcs = [layer.mtp_model_layer.self_attention.linear_proj]
    if is_moe and layer.mtp_model_layer.mlp.use_shared_expert and not layer.mtp_model_layer.mlp.shared_expert_overlap:
        attn_proj_dw_funcs.append(layer.mtp_model_layer.mlp.shared_experts)

    if isinstance(layer.mtp_model_layer.self_attention, MLASelfAttention):
        attn_qkv_dw_funcs = [
            layer.mtp_model_layer.self_attention.linear_kv_up_proj,
            layer.mtp_model_layer.self_attention.linear_kv_down_proj,
            layer.eh_proj,
        ]
        if layer.config.q_lora_rank is None:
            attn_qkv_dw_funcs.append(
                layer.mtp_model_layer.self_attention.linear_q_proj
            )
        else:
            attn_qkv_dw_funcs.extend([
                layer.mtp_model_layer.self_attention.linear_q_down_proj,
                layer.mtp_model_layer.self_attention.linear_q_up_proj
            ])
    else:
        attn_qkv_dw_funcs = [layer.mtp_model_layer.self_attention.linear_qkv, layer.eh_proj]

    backward_dw = {
        "attn_qkv": attn_qkv_dw_funcs,
        "attn_proj": attn_proj_dw_funcs,
        "mlp": layer.mtp_model_layer.mlp.experts if is_moe else None
    }
    return forward_funcs, backward_dw


def build_layer_callables_with_split_attn(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer and MultiTokenPredictionLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables_with_split_attn(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables_with_split_attn(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
