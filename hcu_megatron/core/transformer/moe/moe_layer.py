# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Optional
from functools import wraps

import torch

from megatron.core import tensor_parallel
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import maybe_skip_or_early_return_by_cudagraph
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.moe.moe_layer import (
    MoESubmodules,
    _RecordExpertDgradCompletion,
    _RegisterDelayedWgradForExperts,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import internal_api

from hcu_megatron.training.arguments import get_adaptor_args


def moe_layer_init_wrapper(moe_layer_init_func):
    @wraps(moe_layer_init_func)
    def wrapper(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        moe_layer_init_func(
            self,
            config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )

        self.experts_recompute = (
            config.recompute_granularity == 'selective' and "experts" in config.recompute_modules
        )

        self.router_recompute = (
            config.recompute_granularity == 'selective' and "router" in config.recompute_modules
        )

        # Initialize token dispatcher
        if get_adaptor_args().integrate_recompute_to_ep_comm_overlap:
            if config.moe_token_dispatcher_type == "allgather":
                self.recompute_token_dispatcher = MoEAllGatherTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )
            elif config.moe_token_dispatcher_type == "alltoall":
                self.recompute_token_dispatcher = MoEAlltoAllTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )
            elif config.moe_token_dispatcher_type == "flex":
                self.recompute_token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )

    return wrapper


def moe_layer_forward_wrapper(moe_layer_foward_func):
    @wraps(moe_layer_foward_func)
    def wrapper(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward_experts(dispatched_input, tokens_per_expert, permuted_probs):
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
            return expert_output, mlp_bias
        
        def custom_forward_router(hidden_states, padding_mask):
            probs, routing_map = self.route(hidden_states, padding_mask)
            return probs, routing_map

        if self.experts_recompute or self.router_recompute:
            assert intermediate_tensors is None, "intermediate_tensors should be None when recomputing experts or router"

            shared_expert_output = self.shared_experts_compute(hidden_states)

            residual = hidden_states
            if self.router_recompute:
                probs, routing_map = tensor_parallel.checkpoint(custom_forward_router, False, hidden_states, padding_mask)
            else:
                probs, routing_map = custom_forward_router(hidden_states, padding_mask)

            hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = (
                self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
            )
            if self.experts_recompute:
                expert_output, mlp_bias = tensor_parallel.checkpoint(custom_forward_experts, False, dispatched_input, tokens_per_expert, permuted_probs)
            else:
                expert_output, mlp_bias = custom_forward_experts(dispatched_input, tokens_per_expert, permuted_probs)

            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
            output = self.token_dispatcher.combine_preprocess(expert_output)

            output = self.combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        return moe_layer_foward_func(
            self,
            hidden_states=hidden_states,
            intermediate_tensors=intermediate_tensors,
            padding_mask=padding_mask,
        )

    return wrapper


class MoELayer():
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def get_token_dispatcher(self, is_recompute=False,):
        if get_adaptor_args().integrate_recompute_to_ep_comm_overlap and is_recompute:
            return self.recompute_token_dispatcher

        return self.token_dispatcher

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor, is_recompute=False,
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.
        """
        # Latent-MoE + NVLS-inference shared-expert overlap: launch the shared
        # expert on its side stream BEFORE fc1_latent_proj so it sees the full
        # hidden_states. The corresponding join+add runs in postprocess after
        # fc2_latent_proj. Skipped on the training / NCCL paths.
        token_dispatcher = self.get_token_dispatcher(is_recompute)
        if (
            self.config.moe_latent_size
            and self.shared_expert_overlap
            and isinstance(token_dispatcher, NVLSAllGatherVDispatcher)
        ):
            stream = SharedExpertMLP.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._latent_shared_expert_output = apply_module(self.shared_experts)(hidden_states)
        elif self.config.moe_latent_size:
            if self.shared_expert_overlap:
                if self.training:
                    raise AssertionError(
                        "Shared expert overlap with MoE latent projections is not supported "
                        "during training. Disable moe_shared_expert_overlap."
                    )
                raise AssertionError(
                    "Shared expert overlap with MoE latent projections requires the NVLS "
                    "inference dispatcher. Either disable moe_shared_expert_overlap or set "
                    "inference_moe_token_dispatcher_type='nvls'."
                )
        # Project the hidden_states from hidden dimension down to latent dimenion.
        if self.config.moe_latent_size:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor, is_recompute=False,):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RegisterDelayedWgradForExperts.apply(self, hidden_states)
        return self.get_token_dispatcher(is_recompute).token_dispatch(hidden_states, probs)

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor, is_recompute=False,):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        token_dispatcher = self.get_token_dispatcher(is_recompute)
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RecordExpertDgradCompletion.apply(
                self._delayed_wgrad_event, hidden_states
            )
        dispatched_input, tokens_per_expert, permuted_probs = (
            token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if hasattr(self, "_inference_token_dispatcher") and InferenceMode.is_active():
            routing_map = token_dispatcher.routing_map
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, routing_map=routing_map
            )
        else:
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(token_dispatcher)}"
        output = token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, is_recompute=False,):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.get_token_dispatcher(is_recompute).token_combine(output)
        return output

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor], is_recompute=False,):
        """Project the output back from latent dimension to hidden dimension after combine
        in latent dimension if needed. Combine expert output with shared_experts if needed.

        _latent_shared_expert_output is inference-only (latent-MoE + NVLS dispatcher with
        shared-expert overlap). It is populated in preprocess and joined here, after
        fc2_latent_proj, so the dimensions match the full hidden dim."""

        output = self.get_token_dispatcher(is_recompute).combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        elif (
            isinstance(self.token_dispatcher, NVLSAllGatherVDispatcher)
            and self._latent_shared_expert_output is not None
        ):
            # This codepath is for inference-only shared-expert overlap of latent MoEs.
            # Must happen post-fc2_latent_proj so dimensions match.
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            output = output + self._latent_shared_expert_output
            self._latent_shared_expert_output = None
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor, is_recompute=False,):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map, is_recompute)
        return hidden_states, probs, residual
