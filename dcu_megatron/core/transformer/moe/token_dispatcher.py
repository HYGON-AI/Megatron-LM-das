from contextlib import contextmanager
from typing import Optional, Tuple

import torch

from megatron.training import get_args
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import (
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher as MegatronCoreMoEAlltoAllTokenDispatcher

from dcu_megatron.core.tensor_parallel import all_to_all


# decouple perbatch state from MoEAlltoAllTokenDispatcher
class MoEAlltoAllPerBatchState:
    def __init__(self, build_event=False):
        self.num_global_tokens_per_local_expert = None
        self.output_splits_tp = None
        self.output_splits = None
        self.input_splits = None
        self.num_out_tokens = None
        self.capacity = None
        self.hidden_shape = None
        self.probs = None
        self.routing_map = None
        self.reversed_local_input_permutation_mapping = None
        self.cuda_sync_point = "no_sync"
        self.hidden_shape_before_permute = None
        self.tokens_per_expert = None


class MoEAlltoAllTokenDispatcher(MegatronCoreMoEAlltoAllTokenDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # use_quantize_comm
        args = get_args()
        self.use_quantize_comm = args.use_quantize_comm

    def collect_per_batch_state(self, state: MoEAlltoAllPerBatchState):
        state.num_global_tokens_per_local_expert = getattr(
            self, "num_global_tokens_per_local_expert", None
        )
        state.output_splits_tp = getattr(self, "output_splits_tp", None)
        state.output_splits = getattr(self, "output_splits", None)
        state.input_splits = getattr(self, "input_splits", None)
        state.num_out_tokens = getattr(self, "num_out_tokens", None)
        state.capacity = getattr(self, "capacity", None)
        state.hidden_shape = getattr(self, "hidden_shape", None)
        state.probs = getattr(self, "probs", None)
        state.routing_map = getattr(self, "routing_map", None)
        state.reversed_local_input_permutation_mapping = getattr(
            self, "reversed_local_input_permutation_mapping", None
        )
        state.hidden_shape_before_permute = getattr(self, "hidden_shape_before_permute", None)
        state.cuda_sync_point = getattr(self, "cuda_sync_point", None)
        state.tokens_per_expert = getattr(self, "tokens_per_expert", None)

    def apply_per_batch_state(self, state: MoEAlltoAllPerBatchState):
        self.num_global_tokens_per_local_expert = state.num_global_tokens_per_local_expert
        self.output_splits_tp = state.output_splits_tp
        self.output_splits = state.output_splits
        self.input_splits = state.input_splits
        self.num_out_tokens = state.num_out_tokens
        self.capacity = state.capacity
        self.hidden_shape = state.hidden_shape
        self.probs = state.probs
        self.routing_map = state.routing_map
        self.reversed_local_input_permutation_mapping = (
            state.reversed_local_input_permutation_mapping
        )
        self.hidden_shape_before_permute = state.hidden_shape_before_permute
        self.cuda_sync_point = state.cuda_sync_point
        self.tokens_per_expert = state.tokens_per_expert

    @contextmanager
    def per_batch_state_context(self, state: MoEAlltoAllPerBatchState):
        origin_state = MoEAlltoAllPerBatchState()
        self.collect_per_batch_state(origin_state)
        try:
            self.apply_per_batch_state(state)
            yield
        finally:
            self.collect_per_batch_state(state)
            self.apply_per_batch_state(origin_state)
