# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/AMD-AGI/Primus

from typing import List, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    import primus_turbo.pytorch as primus_turbo_torch
except ImportError:
    primus_turbo_torch = None

from megatron.core.config import is_experimental_enabled
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
    permute,
)
from megatron.core.transformer.moe.token_dispatcher import (
    _HybridEPManager,
    logger,
    MoETokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager as MegatronCoreDeepepManager
from megatron.core.transformer.moe.token_dispatcher import _NCCLEPManager
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher as MegatronCoreMoEFlexTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

from hcu_megatron.core.transformer.moe.fused_a2a import fused_dispatch
from hcu_megatron.training import get_args


class PrimusTurboDeepEPTokenDispatcher(MoETokenDispatcher):
    """
    PrimusTurbo token dispatcher using DeepEP.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the DeepEP token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)

        if self.tp_size * self.ep_size <= 1:
            raise ValueError("DeepEP token dispatcher requires TPxEP > 1")
        if self.config.moe_token_dispatcher_type != "flex":
            raise ValueError("DeepEP backend is only supported with flex token dispatcher.")
        assert (
            self.config.moe_pad_expert_input_to_capacity is False
        ), "DeepEP token dispatcher does not support --moe-pad-expert-input-to-capacity"

        args = get_args()

        # enable sync-free moe to elimiate deepep cpu busy-wait
        num_worst_tokens, permute_max_token_num = 0, 0
        if args.turbo_sync_free_moe_stage > 1:
            if args.sequence_parallel:
                seq_length = args.seq_length // args.tensor_model_parallel_size
            else:
                seq_length = args.seq_length
            num_tokens = seq_length // args.context_parallel_size * args.micro_batch_size
            num_worst_tokens = num_tokens * self.tp_ep_group.size()
            if args.turbo_sync_free_moe_stage > 2:
                # fully sync-free moe
                permute_max_token_num = num_worst_tokens * config.moe_router_topk

        use_turbo_grouped_gemm = args.use_turbo_grouped_gemm
        assert primus_turbo_torch is not None, "Failed to import 'primus_turbo'. Please make sure it is installed."
        self.deepep_dispatcher = primus_turbo_torch.modules.DeepEPTokenDispatcher(
            num_experts=config.num_moe_experts,
            router_topk=config.moe_router_topk,
            ep_group=self.ep_group,
            tp_group=self.tp_group,
            tp_ep_group=self.tp_ep_group,
            expert_capacity_factor=config.moe_expert_capacity_factor,
            permute_fusion=config.moe_permute_fusion,
            permute_max_token_num=permute_max_token_num,
            deepep_async_finish=True,
            deepep_allocate_on_comm_stream=True,
            deepep_use_comm_stream=args.turbo_deepep_use_comm_stream,
            deepep_num_use_cu=args.turbo_deepep_num_cu,
            deepep_num_worst_tokens=num_worst_tokens,
            deepep_use_cuda_num_tokens_per_expert=use_turbo_grouped_gemm,
        )
        # This is just a place holder.
        # The communication manager class is not used in Primus Turbo's DeepEP dispatcher.
        # But it may get referenced in some Megatron code paths.
        self._comm_manager = self.deepep_dispatcher

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Initializes routing metadata and prepares tensors for fused dispatch.

        This method reshapes input tensors and processes routing information into a
        unified format, where the routing map is expanded to cover the TPxEP communication domain,
        enabling the token dispatch logic to be agnostic to parallelism strategies.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            A tuple of reshaped hidden states and token probabilities.
        """
        self.hidden_shape = hidden_states.shape
        # view as [num_tokens, hidden_size]
        hidden_states = hidden_states.view(-1, self.config.hidden_size)

        # when force_load_balancing, we use even token_indices to make sure each expert get same number of tokens
        token_indices = None
        hidden_states, probs = self.deepep_dispatcher._pre_dispatch(
            hidden_states, probs, routing_map, token_indices
        )
        return hidden_states, probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Execute fused permutation and AlltoAll communication.

        This method currently leverages DeepEP's fused dispatch kernel, which combines token
        permutation and AlltoAll communication into a single optimized operation.
        The fused approach reduces memory bandwidth requirements and enables better
        overlap between computation and communication operations.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched
            probs (torch.Tensor): Routing probabilities (unused in current implementation)
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        dispatched_tokens, dispatched_probs = self.deepep_dispatcher._exec_dispatch(hidden_states, probs)
        return dispatched_tokens, dispatched_probs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Converts dispatched tokens to a per-expert format for expert processing.

        This method transforms the output of the fused dispatch into the tensor
        organization required for the expert computation.

        Args:
            hidden_states (torch.Tensor): Hidden states after fused dispatch
            probs (torch.Tensor): Routing probabilities after fused dispatch

        Returns:
            A tuple of permuted tokens, token counts per expert, and permuted probabilities.
        """
        permuted_input, tokens_per_expert, permuted_probs = self.deepep_dispatcher._post_dispatch(
            hidden_states, probs
        )
        if self.config.moe_router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return permuted_input, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pre-processes hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self.deepep_dispatcher._pre_combine(hidden_states)
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This is the inverse of the `token_dispatch` operation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Combined tokens after fused un-permutation and communication.
        """
        combined_tokens = self.deepep_dispatcher._exec_combine(hidden_states)
        return combined_tokens

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Restores the original tensor shape and finalizes the MoE layer output.

        This method performs the final step of the MoE token processing pipeline
        by reshaping the combined tokens back to their original input dimensions.

        Args:
            hidden_states (torch.Tensor): Combined tokens.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        hidden_states = self.deepep_dispatcher._post_combine(hidden_states)
        return hidden_states.view(self.hidden_shape)


class _DeepepManager(MegatronCoreDeepepManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
        num_worst_tokens: int = 0,
    ):
        """
        Initialize the DeepEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
            num_worst_tokens (int): the worst number of tokens to receive.
        """
        super().__init__(
            group,
            num_local_experts,
            router_topk,
            num_experts,
            config,
        )
        self.num_worst_tokens = num_worst_tokens

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning(
                    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states,
                self.token_indices,
                self.token_probs,
                self.num_experts,
                self.group,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                num_worst_tokens=self.num_worst_tokens,
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_experimental_enabled() and self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.config.moe_router_padding_for_quantization:
            self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
                self.dispatched_routing_map, self.tokens_per_expert
            )

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        (
            hidden_states,
            permuted_probs,
            self.reversed_mapping_for_combine,
            self.pad_offsets,
            self.tokens_per_expert,
        ) = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.num_worst_tokens * min(self.router_topk, self.num_local_experts), # self.tokens_per_expert.sum().item()
            fused=self.permute_fusion,
            tokens_per_expert=self.tokens_per_expert,
            align_size=get_align_size_for_quantization(self.config),
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs


class MoEFlexTokenDispatcher(MegatronCoreMoEFlexTokenDispatcher):
    """A flexible token dispatcher that abstracts the underlying tensor and expert
    parallelism. It uses a single communication group over all TP and EP ranks,
    making the dispatch logic independent of the specific parallelism strategy.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super(MegatronCoreMoEFlexTokenDispatcher, self).__init__(config=config, pg_collection=pg_collection)

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        if self.config.moe_flex_dispatcher_backend == "deepep":
            assert self.tp_size * self.ep_size > 1, "DeepEP dispatcher requires TPxEP > 1"

            args = get_args()
            # enable sync-free moe to elimiate deepep cpu busy-wait
            num_worst_tokens = 0
            if args.sync_free_moe and args.sync_free_moe_backend == "deepep":
                if args.sequence_parallel:
                    seq_length = args.seq_length // args.tensor_model_parallel_size
                else:
                    seq_length = args.seq_length
                num_tokens = seq_length // args.context_parallel_size * args.micro_batch_size
                num_worst_tokens = num_tokens * self.tp_ep_group.size()

            self._comm_manager = _DeepepManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
                num_worst_tokens=num_worst_tokens,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        elif self.config.moe_flex_dispatcher_backend == "hybridep":
            self._comm_manager = _HybridEPManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.routing_map']
        elif self.config.moe_flex_dispatcher_backend == "ncclep":
            assert self.tp_size * self.ep_size > 1, "NCCL EP dispatcher requires TPxEP > 1"
            self._comm_manager = _NCCLEPManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        else:
            raise ValueError(
                f"Invalid backend: {self.config.moe_flex_dispatcher_backend}"
                "Please set --moe-flex-dispatcher-backend to deepep, hybridep, or ncclep"
            )


class MoEElasticExpertMetadata:
    """Metadata for the Elastic expert dispatcher.

    Not a dataclass — attributes are set by MoEElasticExpertDispatcher.preprocess
    and populated lazily by _materialize_dispatch_metadata.
    """
    handle = None
    buffer_idx = 0  # 0=FC1, 1=FC2


class MoEElasticExpertDispatcher:
    """Dispatches expert weights to remote EP ranks based on offloading map.

    Complementary to MoEAlltoAllTokenDispatcher: while the token dispatcher
    routes tokens to experts, this dispatcher routes expert weights to ranks
    that need them (i.e. copies home-expert weights into echo/spare slots on
    other ranks so they can process the extra tokens auction/sinkhorn/greedy
    assigned to them).
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if pg_collection is None:
            raise ValueError(
                "MoEElasticExpertDispatcher requires pg_collection "
                "with an expert-parallel process group."
            )
        self.config = config
        self.ep_group = pg_collection.ep
        self.ep_size = self.ep_group.size()
        self.ep_rank = self.ep_group.rank()

        self.num_local_experts = config.moe_num_echo_experts // self.ep_size
        assert getattr(self.config, "moe_enable_echo", False), (
            "Elastic expert dispatcher requires --moe-enable-echo"
        )

        self.permute_idx_device = "cpu"
        input_chunk_idxs = torch.arange(
            self.config.moe_num_echo_experts, device=self.permute_idx_device
        )
        # [num_local_experts, ep_size]. Sort input chunks by local experts.
        self.sort_input_by_local_experts = (
            input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel()
        )

    def preprocess(self, routing_map: torch.Tensor) -> MoEElasticExpertMetadata:
        """Preprocesses the (home→echo) offloading map for expert dispatch.

        Args:
            routing_map: expert_offloading_map, shape [num_home, num_echo] bool.

        Returns:
            MoEElasticExpertMetadata with lazy input_splits/output_splits.
        """
        num_home_experts, num_spare_experts = routing_map.shape
        num_local_home_experts = num_home_experts // self.ep_size
        num_local_spare_experts = num_spare_experts // self.ep_size
        metadata = MoEElasticExpertMetadata()
        metadata.global_routing_map = routing_map

        metadata.local_to_global_routing_map = routing_map[
            self.ep_rank * num_local_home_experts : (self.ep_rank + 1) * num_local_home_experts, :
        ].reshape(num_local_home_experts, self.ep_size, num_local_spare_experts)
        metadata.global_to_local_routing_map = routing_map[
            :, self.ep_rank * num_local_spare_experts : (self.ep_rank + 1) * num_local_spare_experts
        ].reshape(self.ep_size, num_local_home_experts, num_local_spare_experts)

        metadata.input_splits = None
        metadata.output_splits = None
        metadata.num_out_experts = None
        metadata.input_chunks_per_rank = None
        metadata.has_experts_per_slot = None

        return metadata

    @staticmethod
    def _send_order(local_to_global_map: torch.Tensor) -> List[int]:
        """Local home-expert indices to send, in the same order as the
        argsort-based permute() over the [num_local_home, num_echo] map:
        echo column major, ascending home index within a column."""
        flat = local_to_global_map.flatten(start_dim=1)
        return torch.nonzero(flat.T.cpu())[:, 1].tolist()

    def _materialize_dispatch_metadata(self, metadata: MoEElasticExpertMetadata):
        """Populate dispatch metadata if it was not filled by preprocess()."""
        if metadata.input_splits is None:
            metadata.input_splits = metadata.local_to_global_routing_map.sum(dim=[0, 2]).tolist()
            metadata.output_splits = metadata.global_to_local_routing_map.sum(dim=[1, 2]).tolist()
            metadata.num_out_experts = sum(metadata.input_splits)
            metadata.input_chunks_per_rank = (
                metadata.global_to_local_routing_map.sum(dim=1).int().ravel().cpu()
            )
            metadata.has_experts_per_slot = (
                metadata.global_to_local_routing_map.sum(dim=[0, 1]).tolist()
            )

    def expert_dispatch_preprocess(
        self, metadata: MoEElasticExpertMetadata, *expert_weights
    ) -> dict:
        """Pack only the home-expert weights that are sent, in permute() order.

        Equivalent to stack(all home weights) + permute(local_to_global map),
        but skips stacking/permuting unsent experts, which also removes the
        per-expert zero-grad accumulation in backward.
        """
        self._materialize_dispatch_metadata(metadata)
        send_order = getattr(metadata, 'send_order', None)
        if send_order is None:
            send_order = self._send_order(metadata.local_to_global_routing_map)
            metadata.send_order = send_order
        assert len(send_order) == metadata.num_out_experts
        assert metadata.local_to_global_routing_map.shape[0] == len(expert_weights)
        weight_shape = expert_weights[0].shape
        if send_order:
            permuted_expert_weights = torch.stack(
                [expert_weights[i].reshape(-1) for i in send_order], dim=0
            )
        else:
            # Nothing to send: keep a [0, numel] tensor attached to the graph so
            # this rank still joins the backward all_to_all.
            permuted_expert_weights = expert_weights[0].reshape(1, -1)[:0]
        return dict(permuted_weights=permuted_expert_weights, weight_shape=weight_shape)

    def expert_dispatch_communication(
        self, metadata: MoEElasticExpertMetadata, preprocessed: dict, *expert_weights
    ) -> List[torch.Tensor]:
        """AlltoAll the permuted weights + sort back into per-slot list."""
        from megatron.core.tensor_parallel import all_to_all
        from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

        weight_shape = preprocessed['weight_shape']
        permuted_expert_weights = preprocessed['permuted_weights']
        dispatched_expert_weights = all_to_all(
            self.ep_group,
            permuted_expert_weights,
            metadata.output_splits,
            metadata.input_splits,
        )
        dispatched_expert_weights, _ = sort_chunks_by_idxs(
            dispatched_expert_weights,
            metadata.input_chunks_per_rank,
            self.sort_input_by_local_experts,
            fused=False,
        )
        expert_weights = torch.split(
            dispatched_expert_weights, metadata.has_experts_per_slot, dim=0
        )
        weight_list = []
        for weight in expert_weights:
            if weight.numel() > 0:
                weight_list.append(weight.reshape(weight_shape))
            else:
                weight_list.append(weight)
        return weight_list

    def expert_dispatch(
        self, metadata: MoEElasticExpertMetadata, *expert_weights
    ) -> List[torch.Tensor]:
        """One-shot: preprocess + A2A + sort. Returns list of dispatched weights."""
        preprocessed = self.expert_dispatch_preprocess(metadata, *expert_weights)
        return self.expert_dispatch_communication(metadata, preprocessed, *expert_weights)


# ============================================================================
# DeepEP expert-weight dispatcher: BF16 chunks, synchronous and overlap paths.
# Each (home expert, echo slot) copy is routed as rows with top-k=1.
# ============================================================================
class _DeepEPWeightDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chunks, indices, num_experts, group, stream, ticket):
        from megatron.core.transformer.moe.fused_a2a import get_buffer, get_hidden_bytes

        # Retain this exact buffer until backward; FC1/FC2 or token dispatch may
        # otherwise replace Megatron's global buffer with a differently sized one.
        buffer = get_buffer(group, get_hidden_bytes(chunks))
        overlap = stream is not None
        if overlap:
            stream.wait_stream(torch.cuda.current_stream(chunks.device))
            chunks.record_stream(stream)
            indices.record_stream(stream)
        # apply() runs on the caller stream so backward joins the token/home
        # gradient graph there. Only forward transport runs on the side stream.
        with torch.cuda.stream(stream) if overlap else nullcontext():
            previous = None
            if overlap:
                from megatron.core.transformer.moe.fused_a2a import EventHandle, EventOverlap
                previous = EventOverlap(EventHandle())
            per_rank, per_rdma_rank, per_expert, in_rank, layout_event = buffer.get_dispatch_layout(
                indices, num_experts, previous_event=previous, async_finish=overlap,
            )
            received, received_indices, _, _, handle, done = buffer.dispatch(
                chunks,
                topk_idx=indices,
                # Unit routing weights are metadata, not matrix multipliers.
                topk_weights=torch.ones_like(indices, dtype=torch.float32),
                num_tokens_per_rank=per_rank,
                num_tokens_per_rdma_rank=per_rdma_rank,
                num_tokens_per_expert=per_expert,
                is_token_in_rank=in_rank,
                previous_event=layout_event,
                async_finish=overlap,
            )
            if overlap:
                # DeepEP owns a communication stream. Bridge its completion to
                # our side stream, never to the home-GEMM stream at launch time.
                done.current_stream_wait()
                ticket['done'] = torch.cuda.Event()
                ticket['done'].record(stream)
        if overlap:
            received.record_stream(torch.cuda.current_stream(chunks.device))
            received_indices.record_stream(torch.cuda.current_stream(chunks.device))
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.mark_non_differentiable(received_indices)
        return received, received_indices

    @staticmethod
    def backward(ctx, grad_received, grad_indices):
        # No probability factors: expert copies need the sum of replica wgrads.
        grad_chunks, _, _ = ctx.buffer.combine(
            grad_received.contiguous(), ctx.handle, async_finish=False,
        )
        return grad_chunks, None, None, None, None, None


class MoEDeepEPExpertDispatcher(MoEElasticExpertDispatcher):
    """Drop-in weight dispatcher using a DeepEP Buffer backend.

    Uses the standard high-throughput Buffer API. Overlap launches asynchronous
    dispatch on a side stream and waits only when echo GEMM needs the weights.
    """
    # Validated with simulated FC1 and FC2 weight matrices in DeepEP chunked communication benchmarks.
    # With stable timings, 8192 performed well and similarly to larger chunks, so it remains the default.
    weight_chunk_size = 8192

    def __init__(self, config, pg_collection):
        super().__init__(config, pg_collection)
        if self.ep_size <= 1:
            raise ValueError('DeepEP echo expert dispatch requires EP size > 1')
        from megatron.core.transformer.moe import fused_a2a

        if not fused_a2a.HAVE_DEEP_EP:
            raise RuntimeError(
                'deep_ep package exposing Buffer; install it in the training '
                'environment or select alltoall.'
            )

        # Megatron keeps ONE global DeepEP Buffer and rebuilds it whenever the
        # group object differs or the buffer is too small. Across nodes every
        # rebuild re-inits rocSHMEM and allocs/frees on the symmetric heap while
        # the old Buffer is still alive (autograd ctx), which corrupts the heap.
        # So share the token dispatcher's group object (tp_ep) and size the
        # buffer for both weight chunks and tokens up front: it is never rebuilt.
        self.comm_group = self.ep_group
        tp_ep_group = getattr(pg_collection, 'tp_ep', None)
        if tp_ep_group is not None and tp_ep_group.size() == self.ep_size:
            self.comm_group = tp_ep_group
        elif getattr(config, 'moe_flex_dispatcher_backend', None) == 'deepep':
            raise ValueError(
                'DeepEP echo expert dispatch together with the DeepEP token dispatcher '
                'requires expert_tensor_parallel_size == 1 so both share one Buffer.'
            )
        token_hidden_bytes = config.hidden_size * max(config.params_dtype.itemsize, 2)
        fused_a2a.get_buffer(
            self.comm_group, max(self.weight_chunk_size * 2, token_hidden_bytes),
        )

    def expert_dispatch_preprocess(self, metadata, *expert_weights):
        if not expert_weights:
            raise ValueError('DeepEP echo expert dispatch requires home expert weights')
        reference = expert_weights[0]
        if reference.dtype != torch.bfloat16 or not reference.is_cuda:
            raise ValueError('DeepEP echo expert dispatch requires BF16 weights on CUDA/ROCm/HCU')
        if any(w.shape != reference.shape or w.dtype != reference.dtype
               or w.device != reference.device for w in expert_weights):
            raise ValueError('Home expert weights must have the same shape, dtype and device')
        self._materialize_dispatch_metadata(metadata)
        if any(count > 1 for count in metadata.has_experts_per_slot):
            raise ValueError('Each echo slot can receive at most one home expert')

        # Each selected pair is a separate copy. Autograd's index_select backward
        # sums gradients when a home expert has replicas on several target ranks.
        local_map = metadata.local_to_global_routing_map.reshape(
            len(expert_weights), -1,
        )
        pairs = local_map.nonzero(as_tuple=False)
        stacked = torch.stack([w.reshape(-1) for w in expert_weights])
        numel = reference.numel()
        chunks_per_weight = (numel + self.weight_chunk_size - 1) // self.weight_chunk_size
        stacked = F.pad(stacked, (0, chunks_per_weight * self.weight_chunk_size - numel))
        chunks = stacked.index_select(0, pairs[:, 0]).reshape(-1, self.weight_chunk_size)
        indices = pairs[:, 1].repeat_interleave(chunks_per_weight).reshape(-1, 1).contiguous()
        return dict(chunks=chunks, indices=indices, weight_shape=reference.shape,
                    weight_numel=numel, chunks_per_weight=chunks_per_weight)

    def expert_dispatch_communication(self, metadata, preprocessed, *expert_weights):

        received, indices = _DeepEPWeightDispatch.apply(
            preprocessed['chunks'], preprocessed['indices'],
            self.config.moe_num_echo_experts, self.comm_group, None, None,
        )
        return self._unpack_weights(metadata, preprocessed, received, indices)

    def expert_dispatch_overlap_start(self, metadata, preprocessed, stream, earlier=None):
        from .experts_overlap import _AfterEarlierWeight

        chunks = preprocessed['chunks']
        if earlier is not None:
            # FC2 combine backward must finish before FC1 combine starts.
            chunks = _AfterEarlierWeight.apply(chunks, earlier)
        ticket = {}
        received, indices = _DeepEPWeightDispatch.apply(
            chunks, preprocessed['indices'], self.config.moe_num_echo_experts,
            self.comm_group, stream, ticket,
        )
        ticket['indices'] = indices
        return received, ticket

    def expert_dispatch_overlap_finish(self, metadata, preprocessed, received, ticket):
        if 'done' in ticket:
            torch.cuda.current_stream(received.device).wait_event(ticket['done'])
        return self._unpack_weights(metadata, preprocessed, received, ticket['indices'])

    @staticmethod
    def _unpack_weights(metadata, preprocessed, received, indices):
        # DeepEP preserves source row order. A slot has at most one source, so a
        # stable slot sort preserves chunk order within each reconstructed matrix.
        order = torch.argsort(indices[:, 0], stable=True)
        received = received.index_select(0, order)
        sizes = [count * preprocessed['chunks_per_weight']
                 for count in metadata.has_experts_per_slot]
        weights = []
        for chunks in torch.split(received, sizes, dim=0):
            flat = chunks.reshape(-1)[:preprocessed['weight_numel']]
            weights.append(flat.reshape(preprocessed['weight_shape']) if flat.numel() else flat)
        return weights
