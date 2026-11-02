from typing import Dict, Tuple

import torch
from megatron.core import utils
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import ultra_ep
    HAVE_EPLB = True
except ImportError:
    HAVE_EPLB = False


class _ReplicaWeightEvent:
    """Include peer completion in the runtime's existing timed wait API."""

    def __init__(self, native, peers_ready):
        self.native = native
        self.event = native.event
        self.peers_ready = peers_ready

    def current_stream_wait(self):
        self.native.current_stream_wait()
        torch.cuda.current_stream().wait_event(self.peers_ready)


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
        self.adaptive_config = None
        self.adaptive_controller = None
        self._adaptive_layers = []
        self._peer_group = None
        self._peer_fence = None
        self._replica_comm_stream = None
        if getattr(config, 'moe_ultraep_adaptive', False):
            from hcu_megatron.training.ultraep_adaptive import AdaptiveConfig

            self.adaptive_config = AdaptiveConfig(
                samples=getattr(config, 'moe_ultraep_adaptive_samples', 3),
                max_samples=getattr(config, 'moe_ultraep_adaptive_max_samples', 5),
                min_speedup=getattr(config, 'moe_ultraep_adaptive_min_speedup', 0.01),
                max_calibration_steps=getattr(config, 'moe_ultraep_adaptive_max_calibration_steps', 48),
            )
        self.ultraep_active = self.adaptive_config is None

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

        # Create the native UltraEP runtime manager (ultra_ep._C).
        # This path is entered only when HAVE_EPLB is true.

        autotune_enabled = bool(getattr(config, "moe_ultraep_autotune", False))
        manager_kwargs = dict(
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
        if autotune_enabled:
            required_runtime_methods = (
                "autotune_iteration_end",
                "autotune_needs_iteration_time",
                "wait_grad_reduce",
                "wait_weight_sync",
            )
            missing_apis = []
            if not hasattr(ultra_ep, "AutotuneConfig"):
                missing_apis.append("AutotuneConfig")
            missing_apis.extend(
                f"Manager.{name}"
                for name in required_runtime_methods
                if not hasattr(ultra_ep.Manager, name)
            )
            if missing_apis:
                raise RuntimeError(
                    "--moe-ultraep-autotune requires an autotune-enabled "
                    "UltraEP runtime; missing APIs: " + ", ".join(missing_apis)
                )
            manager_kwargs["autotune"] = ultra_ep.AutotuneConfig(
                enabled=True,
                start_iteration=getattr(
                    config, "moe_ultraep_autotune_start_iteration", 3
                ),
                grad_reduce_max_sms=getattr(
                    config, "moe_ultraep_autotune_grad_reduce_max_sms", None
                ),
            )

        self.runtime = ultra_ep.Manager(**manager_kwargs)
        # This counter is intentionally local to this Manager lifetime rather
        # than Megatron's checkpointed global iteration. A resumed job must
        # still receive its two conservative warm-up iterations.
        self._autotune_iteration = 0
        if self.rank == 0 and getattr(self.runtime, "autotune_enabled", False):
            print(
                "UltraEP Megatron autotune enabled: "
                f"start_iteration={self.runtime.autotune_start_iteration}, "
                "sequence=weight-sync -> grad-reduce, "
                f"grad_reduce_max_sms="
                f"{getattr(config, 'moe_ultraep_autotune_grad_reduce_max_sms', None)}",
                flush=True,
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
        # Finish communicator creation before PP can post receives. Creating
        # it during the first UltraEP forward can block another pipeline stage.
        self._prepare_peer_fence()

    @torch.no_grad()
    def update_placement(self, layer_id: int, routing_map: torch.Tensor):
        """Update replica placement via C++ greedy EPLB algorithm (no broadcast needed)."""
        self.runtime.update_placement(layer_id, routing_map)

    def register_adaptive_layer(self, layer, baseline_state, ultraep_state):
        self._adaptive_layers.append((layer, baseline_state, ultraep_state))
        for module, fields in (ultraep_state if self.ultraep_active else baseline_state):
            for name, value in fields.items():
                setattr(module, name, value)

    def set_ultraep_active(self, active: bool, *, synchronize=True):
        """Switch only after a complete step; shared parameter storage stays fixed."""
        if active == self.ultraep_active:
            return
        if self.adaptive_config is None:
            raise RuntimeError('Runtime UltraEP mode changes require adaptive mode')
        for layer, _, _ in self._adaptive_layers:
            if layer._eplb_grad_reduce_event_handles or layer._eplb_weight_sync_event_handle is not None:
                raise RuntimeError('Cannot switch UltraEP with pending replica communication')
        if synchronize:
            torch.cuda.synchronize()
        for _, baseline, ultraep in self._adaptive_layers:
            for module, fields in (ultraep if active else baseline):
                for name, value in fields.items():
                    setattr(module, name, value)
        self.ultraep_active = active

    def discard_autotune_measurements(self):
        # Invalid/skipped steps must not contaminate the next candidate's waits.
        for name in ('_autotune_ws_wait_events', '_autotune_grad_wait_events'):
            events = getattr(self.runtime, name, None)
            if events is not None:
                events.clear()

    def _prepare_peer_fence(self):
        if self._peer_group is None:
            if torch.distributed.group.WORLD.bound_device_id is not None:
                # new_group would split WORLD when it has a bound device.
                # Only this EP group's members may construct a MoE Manager
                # (a different PP stage can contain no MoE layers).
                self._peer_group = torch.distributed.split_group(
                    parent_pg=self.group, split_ranks=[list(range(self.num_ranks))],
                )
            else:
                self._peer_group = torch.distributed.new_group(
                    ranks=torch.distributed.get_process_group_ranks(self.group),
                    use_local_synchronization=True,
                )
            self._peer_fence = torch.zeros(1, dtype=torch.int32, device='cuda')
        if self._replica_comm_stream is None:
            self._replica_comm_stream = self.runtime.get_comm_stream()
        return self._replica_comm_stream

    def _fence_on_comm_stream(self, stream, *, record=False):
        with torch.cuda.stream(stream):
            torch.distributed.all_reduce(self._peer_fence, group=self._peer_group)
            if record:
                ready = torch.cuda.Event()
                ready.record()
                return ready
        return None

    def weight_sync(self, layer_id: int, async_finish=True):
        # Native WS completes this rank's outgoing writes. Peers must also
        # finish writing our shared slots before GEMM reads them. The fence
        # stays on the native stream and can overlap token dispatch.
        stream = self._prepare_peer_fence()
        native = self.runtime.weight_sync(layer_id=layer_id, async_finish=True)
        event = _ReplicaWeightEvent(native, self._fence_on_comm_stream(stream, record=True))
        if async_finish:
            return event
        event.current_stream_wait()
        return None

    def grad_reduce(self, layer_id: int, async_finish=True):
        stream = self._prepare_peer_fence()
        # A master may read any peer's replica gradients. Local Wgrad readiness
        # alone does not prove remote Wgrad has completed.
        stream.wait_stream(torch.cuda.current_stream())
        self._fence_on_comm_stream(stream)
        event = self.runtime.grad_reduce(layer_id=layer_id, async_finish=async_finish)
        # WS and GR reuse native task metadata. A cached WS plan cannot retain
        # that metadata after GR has overwritten it, including when the tuner
        # replays the last virtual layer without a new placement update.
        native = getattr(self.runtime, 'runtime', None)
        invalidate = getattr(native, 'invalidate_weight_sync_task_cache', None)
        if invalidate is not None:
            invalidate()
        # The next WS uses this same stream and ends with a peer fence. That
        # fence also proves all prior GR reads/zeroing have finished before
        # the next expert GEMM can reuse a replica gradient slot. WS only
        # writes weights, so it is safe to overlap it with remote GR tails;
        # a separate post-GR collective would repeat the same dependency.
        return event

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

    @property
    def autotune_collecting(self) -> bool:
        return bool(getattr(self.runtime, "autotune_collecting", False))

    @property
    def next_autotune_iteration(self) -> int:
        return self._autotune_iteration + 1

    @property
    def autotune_needs_iteration_time(self) -> bool:
        return self.runtime.autotune_needs_iteration_time(
            self.next_autotune_iteration
        )

    def autotune_iteration_end(self, iteration_time_ms=None) -> None:
        """Advance the one-shot tuner exactly once per complete train step."""
        if not self.autotune_collecting:
            return
        self._autotune_iteration += 1
        if self.rank == 0 and self._autotune_iteration == 1:
            print(
                "UltraEP Megatron autotune train-step callback active; "
                f"calibration starts at local iteration {self.runtime.autotune_start_iteration}.",
                flush=True,
            )
        self.runtime.autotune_iteration_end(
            self._autotune_iteration,
            iteration_time_ms=iteration_time_ms,
        )


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


def get_eplb_managers() -> Tuple[EPLBManager, ...]:
    """Return Managers in deterministic construction order on this rank."""
    return tuple(_eplb_manager_registry.values())


def get_collecting_eplb_managers() -> Tuple[EPLBManager, ...]:
    return tuple(
        manager for manager in get_eplb_managers() if manager.autotune_collecting
    )
