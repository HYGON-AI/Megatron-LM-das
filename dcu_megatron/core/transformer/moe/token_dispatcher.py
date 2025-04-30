from megatron.core.transformer.moe.token_dispatcher import _DeepepManager as MegatronCoreDeepepManager

class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted probs of each token produced by the router.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", tokens_per_expert
        )
        global_input_tokens = all_to_all(
            self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs, self.output_splits, self.input_splits
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

        return global_input_tokens, tokens_per_expert, global_probs    

class _DeepepManager(MegatronCoreDeepepManager):
    """ 
        patch megatron _DeepepManager. async
    """

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
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
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flex token dispatcher using DeepEP.
    """

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            Tuple containing:
            - torch.Tensor: Reshaped hidden states
            - torch.Tensor: Token probabilities from the communication manager
            - None: Placeholder for compatibility
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states, self._comm_manager.token_probs, None

    def dispatch_all_to_all(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to dispatch tokens across expert parallel ranks.
        """
        return (
            self._comm_manager.dispatch(hidden_states, async_finish, allocate_on_comm_stream),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the dispatched hidden states after all-to-all communication.

        This method retrieves the permuted hidden states by experts, calculates the number of tokens
        per expert, and returns the processed data ready for expert processing.
        """
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to the routing map and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states and routing information
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _, _ = self.dispatch_preprocess(hidden_states, routing_map, probs)
        hidden_states, _ = self.dispatch_all_to_all(
            hidden_states, async_finish=False, allocate_on_comm_stream=False
        )
        global_input_tokens, tokens_per_expert, permuted_probs = self.dispatch_postprocess(
            hidden_states
        )

        return global_input_tokens, tokens_per_expert, permuted_probs


    def combine_preprocess(self, hidden_states: torch.Tensor):
        """
        Pre-processes the hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def combine_all_to_all(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to combine tokens after expert processing.
        """
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the combined hidden states after all-to-all communication.

        This method reshapes the combined hidden states to match the original input shape.
        """
        return hidden_states.view(self.hidden_shape)

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverses the token permutation process to restore the original token order.

        This method implements the token unpermutation process in three steps:
        1. Pre-process the hidden states to restore their original ordering
        2. Perform all-to-all communication to combine tokens
        3. Post-process the combined tokens to match the original input shape
        """
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = self.combine_preprocess(hidden_states)
        hidden_states = self.combine_all_to_all(hidden_states, False, False)
        hidden_states = self.combine_postprocess(hidden_states)

        return hidden_states, None       
