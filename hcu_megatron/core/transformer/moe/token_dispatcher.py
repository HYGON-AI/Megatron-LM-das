from typing import List, Optional

import torch

import primus_turbo.pytorch as primus_turbo_torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args


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
        assert (
            self.config.moe_enable_deepep
        ), "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        assert (
            self.config.moe_pad_expert_input_to_capacity is False
        ), "DeepEP token dispatcher does not support --moe-pad-expert-input-to-capacity"

        args = get_args()

        # enable sync-free moe to elimiate deepep cpu busy-wait
        num_worst_tokens, permute_max_token_num = 0, 0
        if args.turbo_sync_free_moe_stage > 1:
            if args.sequence_parallel:
                seq_length = args.seq_length // self.tp_size
            else:
                seq_length = args.seq_length
            num_tokens = seq_length // args.context_parallel_size * args.micro_batch_size
            num_worst_tokens = num_tokens * self.tp_ep_group.size()
            if args.turbo_sync_free_moe_stage > 2:
                # fully sync-free moe
                permute_max_token_num = num_worst_tokens * config.moe_router_topk

        use_turbo_grouped_gemm = args.use_primus_grouped_gemm
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
