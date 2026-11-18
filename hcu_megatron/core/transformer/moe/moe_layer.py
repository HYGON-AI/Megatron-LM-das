# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from functools import wraps
from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import (
    HAVE_TE,
    MoESubmodules,
    _RecordExpertDgradCompletion,
    _RegisterDelayedWgradForExperts,
    te_checkpoint,
)
from megatron.core.transformer.moe.moe_utils import maybe_skip_or_early_return_by_cudagraph
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
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
        eplb_enabled = getattr(config, 'moe_enable_ultraep', False)

        if eplb_enabled:
            from hcu_megatron.core.transformer.moe.eplb_manager import (
                HAVE_EPLB,
                get_or_create_eplb_manager,
            )

            if not HAVE_EPLB:
                raise ImportError(
                    "moe_enable_ultraep=True but ultra_ep could not be imported."
                )
            # Create EPLBManager before the original init so we know
            # num_local_physical_experts. Derive ep_group from parallel_state
            # because self.ep_group is not yet set at this point.
            from megatron.core import parallel_state
            ep_group = parallel_state.get_expert_model_parallel_group()
            eplb_manager = get_or_create_eplb_manager(config=config, ep_group=ep_group)

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

        self.eplb_enabled = eplb_enabled
        self.eplb_manager = None

        if self.eplb_enabled:
            self.eplb_manager = eplb_manager
            self.num_local_physical_experts = (
                self.eplb_manager.num_local_physical_experts
            )
            self.num_global_physical_experts = (
                self.eplb_manager.num_global_physical_experts
            )
            self.local_physical_expert_indices = (
                self.eplb_manager.local_physical_expert_indices
            )
            self._eplb_grad_reduce_event_handles = {}
            self._eplb_weight_sync_event_handle = None
            self._eplb_master_ptrs_registered = False

            # Inject instance methods.
            import types
            self._eplb_register_redundant_experts = types.MethodType(
                _eplb_register_redundant_experts, self
            )
            self._eplb_register_master_experts = types.MethodType(
                _eplb_register_master_experts, self
            )
            self._eplb_start_grad_reduce = types.MethodType(
                _eplb_start_grad_reduce, self
            )
            self._eplb_finish_grad_reduce = types.MethodType(
                _eplb_finish_grad_reduce, self
            )
            self._eplb_register_master_grad_ready = types.MethodType(
                _eplb_register_master_grad_ready, self
            )

            # Phase 1: create replica weight params backed by UltraEP buffers
            # and extend linear_fc1/fc2.num_gemms to num_local_physical_experts.
            adaptive = getattr(config, 'moe_ultraep_adaptive', False)
            baseline_state = _capture_expert_execution_state(self) if adaptive else None
            self._eplb_register_redundant_experts()
            if adaptive:
                self.eplb_manager.register_adaptive_layer(
                    self, baseline_state, _capture_expert_execution_state(self)
                )
            # Phase 2 (_eplb_register_master_experts) called lazily after DDP init.

        else:
            self.num_local_physical_experts = self.num_local_experts
            self.num_global_physical_experts = self.config.num_moe_experts
            self.local_physical_expert_indices = self.local_expert_indices

    return wrapper


def moe_layer_forward_wrapper(moe_layer_foward_func):
    @wraps(moe_layer_foward_func)
    def wrapper(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.eplb_enabled and getattr(self.eplb_manager, 'ultraep_active', True):
            return _ultraep_forward(self, hidden_states, padding_mask)

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

            if self.router_recompute:
                probs, routing_map = tensor_parallel.checkpoint(custom_forward_router, False, hidden_states, padding_mask)
            else:
                probs, routing_map = custom_forward_router(hidden_states, padding_mask)

            hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = (
                self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
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


# ---------------------------------------------------------------------------
# UltraEP replica execution
# ---------------------------------------------------------------------------

class _EPLBReplicaGradReduceStartFunction(torch.autograd.Function):
    """Wraps the MoE input so its backward fires replica grad-reduce after MoE backward."""

    @staticmethod
    def forward(ctx, hidden_states, moe_layer, virtual_layer_id):
        ctx.moe_layer = moe_layer
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        ctx.moe_layer._eplb_start_grad_reduce(virtual_layer_id=ctx.virtual_layer_id)
        return grad_output, None, None


class _EPLBReplicaGradReduceFinishFunction(torch.autograd.Function):
    """Finishes EPLB replica grad-reduce and registers master param grads as DDP-ready."""

    @staticmethod
    def forward(ctx, hidden_states, moe_layer, virtual_layer_id):
        ctx.moe_layer = moe_layer
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        ctx.moe_layer._eplb_finish_grad_reduce(ctx.virtual_layer_id)
        ctx.moe_layer._eplb_register_master_grad_ready()
        return grad_output, None, None


class _EPLBWeightSyncFunction(torch.autograd.Function):
    """Re-syncs replica weights with masters during backward (no-recompute path)."""

    @staticmethod
    def forward(ctx, hidden_states, moe_layer, virtual_layer_id):
        ctx.moe_layer = moe_layer
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.moe_layer.eplb_manager is not None:
            # Synchronize replica weights before the expert backward pass.
            ctx.moe_layer.eplb_manager.weight_sync(
                layer_id=ctx.virtual_layer_id,
                async_finish=False,
            )
        return grad_output, None, None


# ---------------------------------------------------------------------------
# Instance methods injected onto MoELayer
# ---------------------------------------------------------------------------

def _eplb_register_redundant_experts(self):
    """Phase 1: mark replica params and point them at UltraEP shared buffers.

    Called during __init__, BEFORE DDP/_ParamAndGradBuffer init.
    """
    num_local_master = self.eplb_manager.num_local_master_experts
    num_local_redundant = self.eplb_manager.num_local_redundant_experts
    num_local_physical = num_local_master + num_local_redundant
    expert_fc1_numel = self.eplb_manager.expert_fc1_numel
    expert_fc2_numel = self.eplb_manager.expert_fc2_numel

    local_replica_weight_buffers = [
        self.eplb_manager.local_replica_fc1_weight_buffer,
        self.eplb_manager.local_replica_fc2_weight_buffer,
    ]
    local_replica_grad_buffers = [
        self.eplb_manager.local_replica_fc1_grad_buffer,
        self.eplb_manager.local_replica_fc2_grad_buffer,
    ]

    from megatron.core.transformer.moe.experts import TEGroupedMLP
    assert isinstance(self.experts, TEGroupedMLP), (
        f"EPLB requires TEGroupedMLP experts, got {type(self.experts)}"
    )

    for module_idx, linear_module in enumerate(
        [self.experts.linear_fc1, self.experts.linear_fc2]
    ):
        expert_weight0 = getattr(linear_module, 'weight0', None)
        assert expert_weight0 is not None
        if module_idx == 0:
            assert expert_fc1_numel == expert_weight0.numel()
        else:
            assert expert_fc2_numel == expert_weight0.numel()
        module_shape = expert_weight0.shape

        local_replica_weight_buffer = local_replica_weight_buffers[module_idx]
        local_replica_grad_buffer = local_replica_grad_buffers[module_idx]

        # Mark master expert params for deferred DDP ready-registration.
        for expert_idx in range(num_local_master):
            master_weight = getattr(linear_module, f'weight{expert_idx}', None)
            assert master_weight is not None, (
                f"weight{expert_idx} not found in {linear_module}"
            )
            setattr(master_weight, 'is_eplb_master', True)
            master_weight._eplb_manager = self.eplb_manager

        # Point replica params at UltraEP's cross-layer shared buffers.
        # TEGroupedMLP allocates weight0..weight{num_local_master-1} during its
        # own __init__ (logical experts only).  Create replica slots here as
        # new Parameters backed directly by the UltraEP shared buffer rows.
        for expert_idx in range(num_local_master, num_local_physical):
            local_replica_offset = expert_idx - num_local_master
            replica_data = local_replica_weight_buffer[
                local_replica_offset
            ].view(module_shape)
            # Create a new Parameter whose storage IS the UltraEP buffer row.
            replica_weight = torch.nn.Parameter(replica_data, requires_grad=True)
            setattr(linear_module, f'weight{expert_idx}', replica_weight)
            setattr(replica_weight, 'is_eplb_replica', True)
            setattr(replica_weight, 'is_eplb_master', False)
            replica_weight.main_grad = local_replica_grad_buffer[
                local_replica_offset
            ].view(module_shape)
            # Ensure bias{expert_idx} exists so GroupedLinear.forward doesn't error.
            if not hasattr(linear_module, f'bias{expert_idx}'):
                dtype = expert_weight0.dtype
                device = expert_weight0.device
                setattr(
                    linear_module,
                    f'bias{expert_idx}',
                    torch.Tensor().to(dtype=dtype, device=device),
                )

        # Expand GroupedLinear's num_gemms so its forward loops over all physical
        # expert slots (masters + replicas).  This must happen after all weight{i}
        # and bias{i} attributes are set.
        linear_module.num_gemms = num_local_physical
        # Keep the logical count available to the checkpoint wrapper while the
        # runtime uses the physical count for forward execution.
        linear_module.num_local_master_experts = num_local_master

    self.experts.num_local_experts = num_local_physical
    self.experts.num_local_master_experts = num_local_master
    self._eplb_master_ptrs_registered = False

    # Update token_dispatcher(s) so they understand the physical expert space.
    # After reroute, routing_map last-dim = num_global_physical_experts and
    # num_local_tokens_per_expert has length num_global_physical, so every
    # dispatcher field that uses num_local_experts / num_experts /
    # local_expert_indices must be updated to the physical values.
    # sort_input_by_local_experts / restore_output_by_local_experts are index
    # tensors built at __init__ time from num_local_experts — they must also be
    # rebuilt with the physical count so that sort_chunks_by_idxs in
    # dispatch_postprocess/combine_preprocess gets a sorted_idxs of the right size.
    num_global_physical = self.eplb_manager.num_global_physical_experts
    physical_local_indices = self.eplb_manager.local_physical_expert_indices
    for dispatcher in filter(None, [
        getattr(self, 'token_dispatcher', None),
        getattr(self, 'recompute_token_dispatcher', None),
    ]):
        dispatcher.num_local_experts = num_local_physical
        dispatcher.num_experts = num_global_physical
        dispatcher.local_expert_indices = physical_local_indices

        # MoEFlexTokenDispatcher (deepep backend) uses DeepEP kernels for routing,
        # it does not have permute_idx_device or sort index tensors.
        if not isinstance(dispatcher, MoEFlexTokenDispatcher):
            dev = dispatcher.permute_idx_device
            input_chunk_idxs = torch.arange(
                num_global_physical * dispatcher.tp_size, device=dev
            )
            dispatcher.sort_input_by_local_experts = (
                input_chunk_idxs.reshape(-1, num_local_physical).T.ravel()
            )
            dispatcher.restore_output_by_local_experts = (
                input_chunk_idxs.reshape(num_local_physical, -1).T.ravel()
            )
        else:
            # Patch _DeepepManager so setup_metadata reshapes to physical expert dim.
            # After UltraEP reroute(), routing_map last-dim = num_global_physical,
            # but _DeepepManager was initialized with num_experts = num_global_logical.
            comm = getattr(dispatcher, '_comm_manager', None)
            if comm is not None:
                comm.num_experts = num_global_physical
                comm.num_local_experts = num_local_physical


def _eplb_register_master_experts(self):
    """Phase 2: pass master weight/grad pointers to the UltraEP runtime.

    MUST be called after DDP init (when main_grad has been assigned by
    _ParamAndGradBuffer). Safe to call multiple times — no-op after first call.
    """
    if self._eplb_master_ptrs_registered:
        return

    num_local_master = self.eplb_manager.num_local_master_experts
    master_fc1_weights, master_fc2_weights = [], []
    master_fc1_grads, master_fc2_grads = [], []

    for module_idx, linear_module in enumerate(
        [self.experts.linear_fc1, self.experts.linear_fc2]
    ):
        for expert_idx in range(num_local_master):
            master_weight = getattr(linear_module, f'weight{expert_idx}', None)
            assert master_weight is not None
            assert hasattr(master_weight, 'main_grad'), (
                f"weight{expert_idx}.main_grad missing — call after DDP init."
            )
            if module_idx == 0:
                master_fc1_weights.append(master_weight.data)
                master_fc1_grads.append(master_weight.main_grad)
            else:
                master_fc2_weights.append(master_weight.data)
                master_fc2_grads.append(master_weight.main_grad)

    # Register master parameter addresses with the native UltraEP runtime.
    self.eplb_manager.runtime.construct_local_master_ptr_pool(
        layer_id=self.layer_number,
        fc1_weights=master_fc1_weights,
        fc2_weights=master_fc2_weights,
        fc1_grads=master_fc1_grads,
        fc2_grads=master_fc2_grads,
    )
    self._eplb_master_ptrs_registered = True


def _eplb_start_grad_reduce(self, virtual_layer_id: int, async_finish: bool = True):
    # Reduce replica gradients for the current virtual layer.
    assert virtual_layer_id not in self._eplb_grad_reduce_event_handles, (
        "UltraEP Grad Reduce was launched twice for one virtual layer"
    )
    self._eplb_grad_reduce_event_handles[virtual_layer_id] = (
        self.eplb_manager.grad_reduce(
            layer_id=virtual_layer_id,
            async_finish=async_finish,
        )
    )


def _eplb_finish_grad_reduce(self, virtual_layer_id: int):
    event_handle = self._eplb_grad_reduce_event_handles.pop(
        virtual_layer_id, None
    )
    if event_handle is not None:
        wait_grad_reduce = getattr(
            self.eplb_manager.runtime, "wait_grad_reduce", None
        )
        if wait_grad_reduce is None:
            event_handle.current_stream_wait()
        else:
            wait_grad_reduce(event_handle, layer_id=virtual_layer_id)


def _eplb_register_master_grad_ready(self):
    """Register EPLB master expert grads as DDP-ready after replica grad-reduce."""
    for param in self.experts.parameters():
        if not getattr(param, 'is_eplb_master', False):
            continue
        bucket_group = getattr(param, '_ddp_bucket_group', None)
        if bucket_group is None:
            continue
        if not bucket_group.ddp_config.overlap_grad_reduce:
            continue
        bucket_group.register_grad_ready(param)


def _capture_expert_execution_state(layer):
    """Capture only execution metadata; never copy parameters or gradients."""
    objects = [
        (layer.experts, ('num_local_experts',)),
        (layer.experts.linear_fc1, ('num_gemms',)),
        (layer.experts.linear_fc2, ('num_gemms',)),
    ]
    for dispatcher in filter(None, [layer.token_dispatcher, getattr(layer, 'recompute_token_dispatcher', None)]):
        objects.append((dispatcher, (
            'num_local_experts', 'num_experts', 'local_expert_indices',
            'sort_input_by_local_experts', 'restore_output_by_local_experts',
        )))
    return [(obj, {key: getattr(obj, key) for key in keys if hasattr(obj, key)}) for obj, keys in objects]


def _ultraep_forward(
    self,
    hidden_states: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
):
    """Execute replica routing and communication while UltraEP is active."""
    # Lazily finalize master pointer registration after DDP init.
    if not self._eplb_master_ptrs_registered:
        self._eplb_register_master_experts()

    # Weight Sync reads master weights before TE's own forward pre-hook.
    # Finish lazy optimizer all-gathers before peers can read those weights.
    for linear in (self.experts.linear_fc1, self.experts.linear_fc2):
        hook = getattr(linear.weight0, '_eplb_forward_pre_hook', None)
        if hook is not None:
            hook(linear)

    # Allocate virtual layer ID OUTSIDE custom_forward so activation
    # recompute re-uses the same slot.
    virtual_layer_id = self.eplb_manager.allocate_microbatch_slot(
        self.layer_number
    )
    # Expose the most-recently-allocated virtual ID so the monitor hook can
    # read quota at the correct slot (quota is indexed by virtual_layer_id,
    # not by the 1-indexed real layer_number).
    self._eplb_last_virtual_layer_id = virtual_layer_id

    def custom_forward(hidden_states):
        # FinishFunction wraps input so its backward fires AFTER
        # StartFunction.backward (reverse order), ensuring grad_reduce
        # completes before DDP sees master grads as ready.
        hidden_states = _EPLBReplicaGradReduceFinishFunction.apply(
            hidden_states, self, virtual_layer_id
        )

        shared_expert_output = self.shared_experts_compute(hidden_states)
        probs, routing_map = self.route(hidden_states, padding_mask)

        # Update replica placement and kick off async weight sync.
        self.eplb_manager.update_placement(virtual_layer_id, routing_map)
        # Start asynchronous replica weight synchronization.
        self._eplb_weight_sync_event_handle = (
            self.eplb_manager.weight_sync(
                layer_id=virtual_layer_id, async_finish=True
            )
        )
        # Reroute from logical to physical expert space.
        # ultra_ep HIP kernel requires float32 probs (ultra_ep_hip.cpp:913).
        probs_fp32 = probs.float()
        probs_fp32, routing_map = self.eplb_manager.reroute(
            virtual_layer_id, probs_fp32, routing_map
        )
        probs = probs_fp32.to(probs.dtype)

        # HCU preprocess returns 2-tuple (no residual).
        # NOTE: weight_sync wait is deferred to AFTER preprocess.
        # dtoh_stream.wait_stream(main_stream) fires inside dispatch_preprocess;
        # if weight_sync wait is on main_stream at that point, dtoh also blocks
        # on weight_sync (~2ms × 384 calls = ~750ms extra). By waiting AFTER
        # preprocess, dtoh_stream only waits for the reroute kernel, not weight_sync.
        hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

        # Mark the autograd boundary before token dispatch.
        hidden_states = _EPLBReplicaGradReduceStartFunction.apply(
            hidden_states, self, virtual_layer_id
        )

        # dispatch 只读取 token、路由和概率，不读取 expert 权重。
        # 此时可以让它与 UltraEP 的异步 weight sync 并行。
        dispatched_input, probs = self.dispatch(hidden_states, probs)

        # 专家 GEMM 才会读取（可能刚同步完成的）副本权重。
        if self._eplb_weight_sync_event_handle is not None:
            wait_weight_sync = getattr(
                self.eplb_manager.runtime, "wait_weight_sync", None
            )
            if wait_weight_sync is None:
                self._eplb_weight_sync_event_handle.current_stream_wait()
            else:
                wait_weight_sync(
                    self._eplb_weight_sync_event_handle,
                    layer_id=virtual_layer_id,
                )
            self._eplb_weight_sync_event_handle = None

        output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
        # HCU combine has no shared_expert_output arg.
        output = self.combine(output)
        output = self.postprocess(output, shared_expert_output)

        if not self.moe_layer_recompute:
            output = _EPLBWeightSyncFunction.apply(
                output, self, virtual_layer_id
            )
        return output, mlp_bias

    if self.moe_layer_recompute:
        if (self.config.fp8 or self.config.fp4) and HAVE_TE:
            import megatron.core.parallel_state as ps
            outputs = te_checkpoint(
                custom_forward,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                ps.get_tensor_model_parallel_group(),
                hidden_states,
            )
        else:
            outputs = tensor_parallel.checkpoint(
                custom_forward, False, hidden_states
            )
    else:
        outputs = custom_forward(hidden_states)

    return outputs
