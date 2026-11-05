# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import Optional
from functools import wraps

import torch
import copy

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
    """Initialize recompute dispatchers and optional echo expert slots together."""
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
        echo_enabled = getattr(config, "moe_enable_echo", False)
        orig_num_moe_experts = config.num_moe_experts
        num_echo = config.moe_num_echo_experts if echo_enabled else 0
        try:
            if echo_enabled:
                # Both upstream initialization and recompute dispatchers need echo slots.
                object.__setattr__(config, "num_moe_experts", orig_num_moe_experts + num_echo)
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
        finally:
            if echo_enabled:
                # Restore home-only dimensions even if initialization fails.
                object.__setattr__(config, "num_moe_experts", orig_num_moe_experts)

        if not echo_enabled:
            return

        # Router rebuild: use home-only config (self.config already restored above).
        from megatron.core.transformer.moe.router import TopKRouter
        new_router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        if hasattr(new_router, "set_layer_number") and self.layer_number is not None:
            new_router.set_layer_number(self.layer_number)
        self.router = new_router

        ep_size = self.ep_group.size()
        num_local_home = orig_num_moe_experts // ep_size
        K = num_echo // ep_size
        self.home_expert_indices = list(range(num_local_home))
        self.echo_expert_indices = list(range(num_local_home, num_local_home + K))

        from hcu_megatron.core.transformer.moe.experts import (
            _grouped_mlp_free_expert_parameters,
        )
        _grouped_mlp_free_expert_parameters(self.experts, self.echo_expert_indices)

        for module_name in ("linear_fc1", "linear_fc2"):
            layer = getattr(self.experts, module_name)
            ref_w = getattr(layer, "weight0")
            for i in self.echo_expert_indices:
                dummy = torch.empty(ref_w.shape, dtype=ref_w.dtype, device=ref_w.device)
                # Give TE's GroupedLinear backward the attrs it expects
                # (grad-accumulation fusion path).
                dummy.main_grad = torch.zeros(ref_w.shape, dtype=ref_w.dtype, device=ref_w.device)
                dummy.grad_added_to_main_grad = True
                setattr(layer, f"weight{i}", dummy)

        dispatch_type = getattr(config, 'moe_echo_expert_dispatch_type', 'alltoall')
        if dispatch_type == 'alltoall':
            from hcu_megatron.core.transformer.moe.token_dispatcher import (
                MoEElasticExpertDispatcher,
            )
            self.expert_dispatcher = MoEElasticExpertDispatcher(config, pg_collection)
        elif dispatch_type == 'deepep':
            from hcu_megatron.core.transformer.moe.token_dispatcher import (
                MoEDeepEPExpertDispatcher,
            )
            self.expert_dispatcher = MoEDeepEPExpertDispatcher(config, pg_collection)
        else:
            raise ValueError(f'Unsupported echo expert dispatch type: {dispatch_type}')
        self._echo_pg_collection = pg_collection
        self._echo_num_home_experts = orig_num_moe_experts
        self._echo_num_echo_experts = num_echo

        # Create dedicated CUDA stream for overlap (class variable, shared across layers).
        if config.moe_echo_expert_dispatch_overlap:
            from megatron.core.transformer.moe.moe_layer import MoELayer as UpstreamMoELayer
            if not hasattr(UpstreamMoELayer, 'moe_a2a_stream'):
                UpstreamMoELayer.moe_a2a_stream = torch.cuda.Stream()

    return wrapper


def moe_layer_forward_wrapper(moe_layer_foward_func):
    """Run echo routing when enabled; otherwise preserve the standard forward path."""
    @wraps(moe_layer_foward_func)
    def wrapper(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if getattr(self.config, "moe_enable_echo", False):
            from hcu_megatron.core.transformer.moe.offloading_planner import gen_offloading_plan
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            ep_group = self.ep_group
            ep_size = ep_group.size()
            ep_rank = ep_group.rank()
            num_experts = self._echo_num_home_experts  # home-only (post-shrink)
            num_echo = self._echo_num_echo_experts
            K = num_echo // ep_size

            # Start shared experts before routing.
            shared_expert_output = None
            if hasattr(self, "shared_experts_compute"):
                shared_expert_output = self.shared_experts_compute(hidden_states)

            # 2. Route ONCE.
            probs, routing_map = self.route(hidden_states, padding_mask)
            # 3. Offloading planner.
            tokens_per_expert_current = routing_map.sum(dim=0).to(torch.int32)
            gathered = gather_from_sequence_parallel_region(
                tokens_per_expert_current, group=ep_group
            )
            tokens_per_expert_per_ep = gathered.reshape(ep_size, num_experts)

            algo = getattr(self.config, "moe_echo_algorithm", "sinkhorn")
            if algo == "greedy":
                assignment = "approx_bin_packing" if K == 1 else "one_shot_greedy"
            elif algo == "auction":
                assignment = "auction"
            else:
                assignment = "sinkhorn"

            ranks_per_node = getattr(self.config, "moe_echo_ranks_per_node", 8)

            rerouting_map, rerouted_probs, expert_offloading_map = gen_offloading_plan(
                routing_map,
                probs,
                tokens_per_expert_per_ep,
                ep_rank,
                num_ep_ranks=ep_size,
                num_spare_experts_per_ep_rank=K,
                assignment_algorithm=assignment,
                ranks_per_node=ranks_per_node,
                ep_group=ep_group,
            )
            dump_dir = getattr(self.config, "moe_echo_dump_dir", None)
            if dump_dir:
                _echo_dump_plan(
                    self,
                    dump_dir,
                    assignment,
                    rerouting_map,
                    expert_offloading_map,
                )

            # 4. Expert-weight A2A — install dispatched weights on echo slots.
            from hcu_megatron.core.transformer.moe.experts import (
                _grouped_mlp_get_expert_weights,
                _grouped_mlp_set_expert_weights,
                echo_dispatch_home_weights,
            )

            if self.config.moe_echo_expert_dispatch_overlap:
                # ================================================================
                # True overlap mode (Loong-style):
                #   Default stream: [home fc1 GEMM] ∥ moe_a2a stream: [fc1 A2A]
                #                          ↓
                #                    [wait event] → [echo fc1 GEMM]
                #   Default stream: [home fc2 GEMM] ∥ moe_a2a stream: [fc2 A2A]
                #                          ↓
                #                    [wait event] → [echo fc2 GEMM]
                # ================================================================
                from megatron.core.transformer.moe.moe_layer import MoELayer as UpstreamMoELayer
                from hcu_megatron.core.transformer.moe.experts_overlap import (
                    _grouped_mlp_forward_with_dispatch_overlap,
                )

                # Preprocess expert offloading metadata once (one D2H sync); fc2
                # shares it and only differs in buffer_idx.
                fc1_meta = self.expert_dispatcher.preprocess(expert_offloading_map)
                fc2_meta = copy.copy(fc1_meta)
                fc1_meta.buffer_idx = 0
                fc2_meta.buffer_idx = 1

                # Token dispatch on default stream.
                hidden_states, probs = self.preprocess(hidden_states, rerouted_probs, rerouting_map)
                dispatched_input, probs = self.dispatch(hidden_states, probs)
                dispatched_input, tokens_per_expert, permuted_probs = \
                    self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)

                # Create events for A2A synchronization.
                fc1_event = torch.cuda.Event()
                fc2_event = torch.cuda.Event()

                # Delegate to Loong-style overlap forward.
                output, mlp_bias = _grouped_mlp_forward_with_dispatch_overlap(
                    self.experts,
                    dispatched_input,
                    tokens_per_expert,
                    permuted_probs,
                    len(self.home_expert_indices),
                    self.home_expert_indices,
                    self.echo_expert_indices,
                    self.expert_dispatcher,
                    fc1_meta,
                    fc2_meta,
                    UpstreamMoELayer.moe_a2a_stream,
                    fc1_event,
                    fc2_event,
                )

                # Combine on default stream.
                output = self.token_dispatcher.combine_preprocess(output)
                output = self.combine(output)

            else:
                # Sync mode: expert A2A blocks token dispatch.
                # Metadata depends only on expert_offloading_map: build it once
                # (one D2H sync) and reuse for fc1 and fc2.
                meta = self.expert_dispatcher.preprocess(expert_offloading_map)
                for module_name in ("fc1", "fc2"):
                    home_weights = _grouped_mlp_get_expert_weights(
                        self.experts, module_name, self.home_expert_indices
                    )
                    home_weights = echo_dispatch_home_weights(
                        home_weights, self.config.gradient_accumulation_fusion,
                    )
                    dispatched = self.expert_dispatcher.expert_dispatch(meta, *home_weights)
                    _grouped_mlp_set_expert_weights(
                        self.experts, module_name, dispatched, self.echo_expert_indices
                    )

                # Standard forward path.
                hidden_states, probs = self.preprocess(hidden_states, rerouted_probs, rerouting_map)
                dispatched_input, probs = self.dispatch(hidden_states, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                output = self.combine(output)

            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

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
            # NCCL-EP zero-copy: experts write fc2 output and fc1 dgrad straight into the combine /
            # dispatch symm buffers. Passed only when set (non-TEGroupedMLP experts don't accept
            # these kwargs).
            output_buffer, grad_input_buffer = token_dispatcher.get_expert_zero_copy_buffers()
            expert_kwargs = {}
            if output_buffer is not None:
                expert_kwargs["output_buffer"] = output_buffer
            if grad_input_buffer is not None:
                expert_kwargs["grad_input_buffer"] = grad_input_buffer
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, **expert_kwargs
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


def _echo_dump_plan(
    moe_layer,
    dump_dir,
    assignment,
    rerouting_map,
    expert_offloading_map,
):
    """Rank-local dump of per-layer echo plan ."""
    import os
    import torch
    import torch.distributed as dist

    ep_group = moe_layer.ep_group

    global_rank = dist.get_rank()
    ep_rank = ep_group.rank()
    ep_size = ep_group.size()

    layer_number = getattr(moe_layer, "layer_number", -1)

    if not hasattr(moe_layer, "_echo_dump_step"):
        moe_layer._echo_dump_step = 0

    step = moe_layer._echo_dump_step
    moe_layer._echo_dump_step = step + 1

    # Dump only first step per layer per rank (avoid huge disk churn).
    if step != 0:
        return

    os.makedirs(dump_dir, exist_ok=True)

    path = os.path.join(
        dump_dir,
        f"echo_global{global_rank}_ep{ep_rank}_"
        f"layer{layer_number}_step{step}_{assignment}.pt",
    )

    torch.save(
        {
            "assignment": assignment,
            "layer_number": layer_number,

            # Rank information
            "global_rank": global_rank,
            "ep_rank": ep_rank,
            "ep_size": ep_size,

            "rerouting_map_sum_per_expert":
                rerouting_map.sum(dim=0).detach().to("cpu"),

            "expert_offloading_map":
                expert_offloading_map.detach().to("cpu"),
        },
        path,
    )

    print(
        f"[moe_layer_forward_wrapper] dumped"
        f"global_rank={global_rank} "
        f"ep_rank={ep_rank}/{ep_size} "
        f"layer={layer_number} "
        f"algo={assignment} -> {path}",
        flush=True,
    )
