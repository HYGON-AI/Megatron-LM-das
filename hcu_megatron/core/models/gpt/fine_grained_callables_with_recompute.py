from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch

from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import _get_all_rng_states
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from hcu_megatron.core.pipeline_parallel.utils import ScheduleNode
from hcu_megatron.core.pipeline_parallel import (
    PipelineOffloadManager,
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
)


def build_transformer_layer_callables_with_recompute(layer: TransformerLayer):
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

    def submodule_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Performs same attnention forward logic as GPT Model.
        """
        node.layer_state.saved_tensors = (
            hidden_states,
            node.chunk_state.attention_mask,
            node.chunk_state.rotary_pos_emb,
            node.chunk_state.rotary_pos_cos,
            node.chunk_state.rotary_pos_sin,
            node.chunk_state.packed_seq_params,
            node.chunk_state.sequence_len_offset,
        )
        node.layer_state.rng_states = _get_all_rng_states()

        with torch.no_grad():
            hidden_states, _ = layer._forward_attention(
                hidden_states=hidden_states,
                attention_mask=node.chunk_state.attention_mask,
                rotary_pos_emb=node.chunk_state.rotary_pos_emb,
                rotary_pos_cos=node.chunk_state.rotary_pos_cos,
                rotary_pos_sin=node.chunk_state.rotary_pos_sin,
                packed_seq_params=node.chunk_state.packed_seq_params,
                sequence_len_offset=node.chunk_state.sequence_len_offset,
            )

        return hidden_states

    def submodule_post_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Run forward pass for computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """
        with torch.no_grad():
            if layer.offload_mlp_norm:
                hidden_states = fine_grained_offloading_group_start(hidden_states, name="mlp_norm")
            if layer.recompute_pre_mlp_layernorm:
                layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                with get_fine_grained_offloading_context(layer.offload_mlp_norm):
                    pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                        layer.pre_mlp_layernorm, hidden_states
                    )
            else:
                with get_fine_grained_offloading_context(layer.offload_mlp_norm):
                    pre_mlp_layernorm_output = layer.pre_mlp_layernorm(hidden_states)

            local_tokens, probs, _ = layer.mlp.router_and_preprocess(pre_mlp_layernorm_output)

            # Detach here for mlp_bda residual connection
            node.layer_state.residual = node.detach(hidden_states)
            if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
                # Detach here for shared expert connection
                node.layer_state.pre_mlp_layernorm_output = node.detach(pre_mlp_layernorm_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        with torch.no_grad():
            token_dispatcher = layer.mlp.token_dispatcher
            if enable_deepep:
                # update token_probs to be the detached version, prevents
                # backward graph from connecting to attn submodule
                token_dispatcher._comm_manager.token_probs = probs

            dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)
            node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: torch.Tensor):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        shared_expert_output = None
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        pre_mlp_layernorm_output = getattr(node.layer_state, 'pre_mlp_layernorm_output', None)
        shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
        expert_output, mlp_bias = layer.mlp.routed_experts_compute(
            dispatched_tokens, dispatched_probs, pre_mlp_layernorm_output
        )

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)
        # release tensor reference after use
        node.layer_state.dispatched_probs = None
        node.layer_state.pre_mlp_layernorm_output = None
        if shared_expert_output is None:
            # Return only expert_output, since shared_expert_output causes backward on None
            return expert_output
        return expert_output, shared_expert_output

    def submodule_combine_forward(
        node: ScheduleNode,
        output: torch.Tensor,
        shared_expert_output: Optional[torch.Tensor] = None,
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

        output = layer.mlp.combine(output, shared_expert_output)
        mlp_output_with_bias = (output, None)

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        if layer.offload_mlp_norm:
            (hidden_states,) = fine_grained_offloading_group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record residual to comm stream, since it's created on comp stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        return output

    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
        return layer._forward_mlp(*args, **kwargs)

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    attn_func = submodule_attn_forward
    post_attn_func = submodule_post_attn_forward if is_moe else raise_not_implemented
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    forward_funcs = [attn_func, post_attn_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"attn": layer.self_attention, "mlp": layer.mlp}
    return forward_funcs, backward_dw


def build_mtp_layer_callables_without_split_attn(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables_without_split_attn(layer.transformer_layer)
    attn_forward, post_attn_forward, dispatch_forward, mlp_forward, combine_forward, _ = (
        forward_funcs
    )
    is_moe = isinstance(layer.transformer_layer.mlp, MoELayer)
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_attn_forward(node, hidden_states):
        # MTP Block Preprocess
        if node.is_first_layer:
            # Final layer norm from Decoder
            final_layernorm = node.chunk_state.model.decoder.final_layernorm
            if final_layernorm:
                hidden_states = final_layernorm(hidden_states)
                hidden_states = make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )
                hidden_states = node.detach(hidden_states)
            offset = get_mtp_layer_offset(layer.config)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]
            if (
                get_args().schedule_method == "dualpipev"
                and node.chunk_state.model.embedding.word_embeddings.weight is None
            ):
                from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
                node.chunk_state.model.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        input_ids, position_ids, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids

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
            return attn_forward(node, hidden_states)

    def submodule_mtp_postprocess_forward(node, hidden_states):
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
    post_attn_func = partial(rng_context_wrapper, post_attn_forward)
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [
        attn_func,
        post_attn_func,
        dispatch_func,
        mlp_func,
        combine_func,
        mtp_post_process_func,
    ]
    backward_dw = {
        "attn": [layer.transformer_layer.self_attention, layer.eh_proj],
        "mlp": layer.transformer_layer.mlp,
    }
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
        return build_transformer_layer_callables_without_split_attn(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables_without_split_attn(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
