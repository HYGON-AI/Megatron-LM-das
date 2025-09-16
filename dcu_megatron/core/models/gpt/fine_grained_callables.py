# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import weakref
from typing import Optional

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel.utils import make_viewless
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from dcu_megatron.core.pipeline_parallel.utils import ScheduleNode


def weak_method(method):
    """Creates a weak reference to a method to prevent circular references.

    This function creates a weak reference to a method and returns a wrapper function
    that calls the method when invoked. This helps prevent memory leaks from circular
    references.
    """
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


def should_free_input(name, is_moe, is_deepep):
    """Determine if the node should free its input memory.

    Args:
        name: Node name
        is_moe: Whether it's a MoE model
        is_deepep: Whether it's a DeepEP model

    Returns:
        bool: Whether to free input memory
    """
    # For dense layers [attn, fake, mlp, fake], the input is needed during backward pass
    if not is_moe:
        return False
    # Define which nodes should free input memory
    # Since we split the computing graph into multiple nodes, we can manually control
    # when and how to free the input memory.
    # The input and output of A2A are not needed anymore after the forward pass,
    # so we can free the input memory after the forward pass.
    free_input_nodes = {
        "mlp": True,
        "moe_combine": True,
        # For non-deepep mode, the input is the un-dispatched tokens and probs before dispatch A2A
        # and it's not needed anymore after the forward pass
        # For deepep mode, they are both needed in backward pass, so they cannot be freed.
        "moe_dispatch": not is_deepep,
    }

    return free_input_nodes.get(name, False)


class TransformerLayerState:
    """State shared within a transformer layer.

    This class holds state that is shared between different nodes
    within a transformer layer.
    """

    pass


class PreProcessNode(ScheduleNode):
    """Node responsible for preprocessing operations in the model.

    This node handles embedding and rotary positional embedding computations
    before the main transformer layers.
    """

    def __init__(self, gpt_model, chunk_state, event, stream):
        """Initializes a preprocessing node.

        Args:
            gpt_model: The GPT model instance.
            chunk_state (TransformerChunkState): State shared within a chunk
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="pre_process")
        self.gpt_model = gpt_model
        self.chunk_state = chunk_state

    def forward_impl(self):
        """Implements the forward pass for preprocessing.

        This method handles:
        1. Decoder embedding computation
        2. Rotary positional embedding computation
        3. Sequence length offset computation for flash decoding

        Returns:
            The processed decoder input tensor.
        """
        # Get decoder input
        if not self.gpt_model.pre_process:
            self.chunk_state.decoder_input = self.gpt_model.decoder.input_tensor
        # Run GPTModle._preprocess
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self.gpt_model._preprocess(
                input_ids=self.chunk_state.input_ids,
                position_ids=self.chunk_state.position_ids,
                decoder_input=self.chunk_state.decoder_input,
                packed_seq_params=self.chunk_state.packed_seq_params,
            )
        )

        # Saved for later use
        self.chunk_state.decoder_input = decoder_input
        self.chunk_state.rotary_pos_emb = rotary_pos_emb
        self.chunk_state.rotary_pos_cos = rotary_pos_cos
        self.chunk_state.rotary_pos_sin = rotary_pos_sin
        self.chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):
    """Node responsible for postprocessing operations in the model.

    This node handles final layer normalization and output layer computation
    after the main transformer layers.
    """

    def __init__(self, gpt_model, chunk_state, event, stream):
        """Initializes a postprocessing node.

        Args:
            gpt_model: The GPT model instance.
            chunk_state (TransformerChunkState): State shared within a chunk
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="post_process")
        self.gpt_model = gpt_model
        self.chunk_state = chunk_state

    def forward_impl(self, hidden_states):
        """Implements the forward pass for postprocessing.

        This method handles:
        1. Final layer normalization
        2. Output layer computation
        3. Loss computation if labels are provided

        Args:
            hidden_states: The hidden states from the transformer layers.

        Returns:
            The logits or loss depending on whether labels are provided.
        """
        labels = self.chunk_state.labels
        # Final layer norm from Decoder
        if self.gpt_model.decoder.final_layernorm is not None:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # Run GPTModel._postprocess
        loss = self.gpt_model._postprocess(
            hidden_states=hidden_states,
            input_ids=self.chunk_state.input_ids,
            position_ids=self.chunk_state.position_ids,
            labels=labels,
            decoder_input=self.chunk_state.decoder_input,
            rotary_pos_emb=self.chunk_state.rotary_pos_emb,
            rotary_pos_cos=self.chunk_state.rotary_pos_cos,
            rotary_pos_sin=self.chunk_state.rotary_pos_sin,
            mtp_in_postprocess=False,
            loss_mask=self.chunk_state.loss_mask,
            attention_mask=self.chunk_state.attention_mask,
            packed_seq_params=self.chunk_state.packed_seq_params,
            sequence_len_offset=self.chunk_state.sequence_len_offset,
            runtime_gather_output=self.chunk_state.runtime_gather_output,
            extra_block_kwargs=self.chunk_state.extra_block_kwargs,
        )

        # For now, 1f1b only supports fp16 module
        return float16_to_fp32(loss)


class TransformerLayerNode(ScheduleNode):
    """Base class for transformer layer computation nodes.

    This class provides common functionality for different types of
    transformer layer nodes (attention, MLP, etc.)
    """

    def __init__(
        self,
        stream,
        event,
        layer_state,
        chunk_state,
        submodule,
        name="default",
        bwd_dw_callables=None,
        extra_args={},
    ):
        """Initialize a transformer layer node.

        Args:
            stream (torch.cuda.Stream): CUDA stream for execution
            event (torch.cuda.Event): Synchronization event
            layer_state (TransformerLayerState): State shared within a layer
            chunk_state (TransformerChunkState): State shared within a chunk
            submodule (function): The submodule contain forward and dw function
            it's the per_batch_state_context, o.w. nullcontext
            name (str): Node name, also used to determine memory strategy
            bwd_dw_callables (list): List of weight gradient functions for the layer.
            extra_args (dict): Extra arguments for the node: is_moe, enable_deepep.
        """
        # determine whether to free input memory
        is_moe = extra_args.get("is_moe", False)
        enable_deepep = extra_args.get("enable_deepep", False)
        free_input = should_free_input(name, is_moe, enable_deepep)
        self.delay_wgrad_compute = extra_args.get("delay_wgrad_compute", False)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_input=free_input,
            name=name,
        )
        self.layer_state = layer_state
        self.common_state = layer_state
        self.chunk_state = chunk_state
        self.submodule = submodule
        self.detached = tuple()
        self.before_detached = tuple()

        # Initialize list to store registered dw callables
        self.bwd_dw_callables = []
        if bwd_dw_callables is not None:
            self.bwd_dw_callables = (
                bwd_dw_callables if isinstance(bwd_dw_callables, list) else [bwd_dw_callables]
            )

    def detach(self, t):
        """Detaches a tensor and stores it for backward computation."""
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def forward_impl(self, *args):
        """Calls the submodule as the forward pass."""
        return self.submodule(self, *args)

    def backward_impl(self, outputs, output_grad):
        """Implements the backward pass for the transformer layer node."""
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self._release_state()
        # return grads for record stream
        return grads

    def backward_dw(self):
        """Computes the weight gradients for the transformer layer node."""
        if not self.delay_wgrad_compute:
            return
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            for module in self.bwd_dw_callables:
                module.backward_dw()
        self.bwd_dw_callables = None

    def _release_state(self):
        # Release reference as early as possible, this helps avoid memory leak.
        self.before_detached = None
        self.detached = None
        self.layer_state = None
        self.common_state = None
        self.chunk_state = None
        self.submodule = None


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
    enable_deepep = layer.config.moe_enable_deepep


    def submodule_attention_qkv_forward(
        node: ScheduleNode,
        hidden_states: torch.Tensor,
    ):
        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if layer.recompute_input_layernorm:
            layer.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = layer.input_layernorm_checkpoint.checkpoint(
                layer.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = layer.input_layernorm(hidden_states)

        # Self attention.
        query, key, value = layer.self_attention.compute_qkv(
            input_layernorm_output,
            packed_seq_params=node.chunk_state.packed_seq_params,
        )

        # Detach here for residual residual connection
        node.layer_state.attn_residual = node.detach(hidden_states)

        return query, key, value

    def submodule_attention_core_attn_forward(
        node: ScheduleNode,
        query,
        key,
        value,
    ):
        core_attn_out = layer.self_attention.compute_attn(
            query,
            key,
            value,
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

        node.layer_state.attn_residual = None

        return hidden_states

    def _submodule_attention_postprocess_router_compound_forward(
        node: ScheduleNode,
        core_attn_out,
    ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """

        hidden_states = submodule_attention_proj_forward(
            node,
            core_attn_out,
        )

        # Optional Layer norm post the cross-attention.
        if layer.recompute_pre_mlp_layernorm:
            layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                layer.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = layer.pre_mlp_layernorm(hidden_states)

        permutated_local_input_tokens, probs, pre_mlp_layernorm_output = layer.mlp.router_and_preprocess(pre_mlp_layernorm_output)
        outputs = [
            hidden_states,
            permutated_local_input_tokens,
            probs,
            pre_mlp_layernorm_output,
        ]
        return tuple(outputs)

    def _submodule_shared_expert_forward(node: ScheduleNode, pre_mlp_layernorm_output):
        """
        Performs a forward pass for shared experts.
        """
        def custom_forward(pre_mlp_layernorm_output):
            shared_expert_output = None
            if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
                shared_expert_output = layer.mlp.shared_experts(pre_mlp_layernorm_output)
            return shared_expert_output

        args = [
            pre_mlp_layernorm_output,
        ]
        if layer.mlp.moe_layer_recompute:
            if layer.config.fp8:
                shared_expert_output = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *args,
                )
            else:
                shared_expert_output = tensor_parallel.checkpoint(
                    custom_forward, False, *args
                )
        else:
            shared_expert_output = custom_forward(*args)
        del args

        return shared_expert_output

    def submodule_attention_proj_router_shared_expert_compound_forward(
        node: ScheduleNode,
        core_attn_out,
    ):
        """
        Performs a combined forward pass that includes self-attention, MLP routing and shared-experts logic.
        """

        (
            hidden_states,
            local_tokens,
            probs,
            pre_mlp_layernorm_output,
        ) = _submodule_attention_postprocess_router_compound_forward(
            node,
            core_attn_out,
        )

        shared_expert_output = _submodule_shared_expert_forward(node, pre_mlp_layernorm_output)

        # detached here
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)
   
        return dispatched_tokens, dispatched_probs

    def submodule_routed_experts_forward(node: ScheduleNode, dispatched_input, dispatched_probs):
        """
        Performs a forward pass for the MLP submodule, including only routed-expert computations.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            layer.mlp.token_dispatcher.dispatch_postprocess(dispatched_input, dispatched_probs)
        )

        def custom_forward(
            dispatched_input, tokens_per_expert, permuted_probs
        ):
            expert_output, mlp_bias = layer.mlp.experts(dispatched_input, tokens_per_expert, permuted_probs)
            assert mlp_bias is None
            return expert_output

        args = [
            dispatched_input,
            tokens_per_expert,
            permuted_probs,
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

        expert_output = layer.mlp.token_dispatcher.combine_preprocess(expert_output)
        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def submodule_combine_forward(
        node: ScheduleNode,
        output: torch.Tensor,
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

        shared_expert_output = None
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            shared_expert_output = node.layer_state.shared_expert_output

        output = layer.mlp.combine(output, shared_expert_output)
        mlp_output_with_bias = (output, None)

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
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
        return output

    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
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
        attn_proj_dw_funcs.append(layer.mlp.experts.shared_experts)
    backward_dw = {
        "attn_qkv": layer.self_attention.linear_qkv,
        "attn_proj": attn_proj_dw_funcs,
        "mlp": layer.mlp.experts if is_moe else None
    }

    return forward_funcs, backward_dw


def build_layer_callables(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
