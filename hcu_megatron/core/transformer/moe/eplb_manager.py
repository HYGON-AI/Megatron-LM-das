from typing import Dict, Tuple

import torch
from megatron.core import utils
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import ultra_ep
    HAVE_EPLB = True
except ImportError:
    HAVE_EPLB = False


class EPLBManager:
    """Wrapper around ultra_ep.Manager for expert-parallelism load balancing.

    When pipeline parallelism is enabled (pp_size > 1), allocates per-microbatch
    virtual layer IDs so each in-flight micro-batch keeps its own placement state.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ep_group: torch.distributed.ProcessGroup,
    ):
        self.group = ep_group
        self.rank = utils.get_pg_rank(ep_group)
        self.num_ranks = utils.get_pg_size(ep_group)

        self.num_local_master_experts = config.num_moe_experts // self.num_ranks
        self.num_local_redundant_experts = config.moe_num_redundant_experts_per_rank
        self.num_local_physical_experts = (
            self.num_local_master_experts + self.num_local_redundant_experts
        )
        self.num_global_logical_experts = config.num_moe_experts
        self.num_global_physical_experts = (
            self.num_global_logical_experts
            + self.num_ranks * self.num_local_redundant_experts
        )
        self.local_physical_expert_indices = [
            self.rank * self.num_local_physical_experts + i
            for i in range(self.num_local_physical_experts)
        ]

        self.expert_fc1_numel = 2 * config.hidden_size * config.moe_ffn_hidden_size
        self.expert_fc2_numel = config.hidden_size * config.moe_ffn_hidden_size
        self.expert_total_numel = self.expert_fc1_numel + self.expert_fc2_numel

        pp_size = config.pipeline_model_parallel_size
        vpp_size = config.virtual_pipeline_model_parallel_size
        if vpp_size is None or vpp_size <= 1:
            max_inflight_mbs = pp_size
        else:
            max_inflight_mbs = pp_size * (vpp_size + 1)

        # Slot budget for the per-microbatch virtual layer allocator.
        #
        # Each in-flight micro-batch on this PP stage needs one slot per real
        # MoE layer.  Full activation recompute (`moe_layer_recompute=True`)
        # replays the forward for a slot a second time within the same
        # micro-batch window, and selective recompute can, in principle, do the
        # same, so we multiply by 3 to leave headroom for one forward + one
        # recompute + one margin.
        #
        # This factor is empirical.  Configurations that stack multiple
        # independent recompute passes (e.g. custom pipeline schedules that
        # replay a forward more than twice while an earlier slot is still
        # live) may still exhaust the pool.  `allocate_microbatch_slot` below
        # asserts the runtime never returns a negative / out-of-range id so
        # over-subscription surfaces loudly instead of silently reusing a slot
        # whose placement state is still being read by a prior micro-batch.
        self._recompute_slot_multiplier = 3
        self.max_microbatches = max(1, max_inflight_mbs) * self._recompute_slot_multiplier

        # HCU BLOCKER: ultra_ep.Manager instantiates the compiled C++ extension
        # (ultra_ep._C). Only reachable when HAVE_EPLB=True.
        #
        self.runtime = ultra_ep.Manager(
            group=self.group,
            num_layers=config.num_layers,
            num_local_master_experts=self.num_local_master_experts,
            num_local_redundant_experts=self.num_local_redundant_experts,
            expert_fc1_numel=self.expert_fc1_numel,
            expert_fc2_numel=self.expert_fc2_numel,
            is_train=True,
            explicitly_destroy=False,
            max_microbatches=self.max_microbatches,
        )

        # Mirror replica GPU buffers from the runtime.
        self.local_replica_weight_buffer: torch.Tensor = (
            self.runtime.local_replica_weight_buffer
        )
        self.local_replica_fc1_weight_buffer: torch.Tensor = (
            self.runtime.local_replica_fc1_weight_buffer
        )
        self.local_replica_fc2_weight_buffer: torch.Tensor = (
            self.runtime.local_replica_fc2_weight_buffer
        )
        self.local_replica_grad_buffer: torch.Tensor = (
            self.runtime.local_replica_grad_buffer
        )
        self.local_replica_fc1_grad_buffer: torch.Tensor = (
            self.runtime.local_replica_fc1_grad_buffer
        )
        self.local_replica_fc2_grad_buffer: torch.Tensor = (
            self.runtime.local_replica_fc2_grad_buffer
        )

    @torch.no_grad()
    def update_placement(self, layer_id: int, routing_map: torch.Tensor):
        """Update replica placement via C++ greedy EPLB algorithm (no broadcast needed)."""
        self.runtime.update_placement(layer_id, routing_map)

    def reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        backend: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand routing from logical to physical expert space with round-robin dispatch."""
        return self.runtime.reroute(layer_id, probs, routing_map, backend)

    def allocate_microbatch_slot(self, real_layer_id: int) -> int:
        """Allocate a virtual layer ID for the next micro-batch on this layer.

        The returned id is an opaque, monotonically-increasing handle produced
        by the ultra_ep runtime; the runtime maps it to a physical slot
        internally (modulo the pool sized by ``max_microbatches``).  Do not
        interpret the returned integer as a slot index.
        """
        return self.runtime.allocate_microbatch_slot(real_layer_id)


_eplb_manager_registry: Dict[int, EPLBManager] = {}


def get_or_create_eplb_manager(
    config: TransformerConfig,
    ep_group: torch.distributed.ProcessGroup,
) -> EPLBManager:
    key = id(ep_group)
    global _eplb_manager_registry
    if key not in _eplb_manager_registry:
        _eplb_manager_registry[key] = EPLBManager(config, ep_group)
    return _eplb_manager_registry[key]


def clear_eplb_manager_registry():
    global _eplb_manager_registry
    _eplb_manager_registry.clear()
