# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""UltraEP routing and gradient ownership for packed MegaMoE experts."""

import torch


class _UltraEPPackedExperts(torch.autograd.Function):
    """Snapshot physical weights and join both expert wgrads before reduction.

    Replica storage is shared by layers and microbatches. A separate packed
    tensor per forward keeps the fused operator's saved weights valid through
    backward, even after another layer reuses UltraEP's replica buffers.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, layer, virtual_layer_id):
        ctx.layer = layer
        ctx.virtual_layer_id = virtual_layer_id
        ctx.weights = (w1, w2)
        manager = layer.eplb_manager
        ctx.set_materialize_grads(False)
        return (
            x,
            torch.cat((w1, manager.local_replica_fc1_weight_buffer.view(-1, *w1.shape[1:]))),
            torch.cat((w2, manager.local_replica_fc2_weight_buffer.view(-1, *w2.shape[1:]))),
        )

    @staticmethod
    def backward(ctx, grad_x, grad_w1, grad_w2):
        layer = ctx.layer
        manager = layer.eplb_manager
        num_master = manager.num_local_master_experts
        for weight, grad, replica_grad in zip(
            ctx.weights,
            (grad_w1, grad_w2),
            (manager.local_replica_fc1_grad_buffer, manager.local_replica_fc2_grad_buffer),
        ):
            # MegaMoE returns wgrad through autograd; the native UltraEP DDP
            # hook intentionally skips masters, so accumulate it here instead.
            if grad is not None:
                weight.main_grad.add_(grad[:num_master])
                replica_grad.copy_(grad[num_master:].reshape_as(replica_grad))
            else:
                replica_grad.zero_()

        # All three outputs belong to this autograd node. Both stage wgrads
        # have arrived before we publish the shared replica gradients.
        event = manager.runtime.grad_reduce(
            layer_id=ctx.virtual_layer_id, async_finish=True
        )
        layer._ultraep_wait(event, "grad_reduce", ctx.virtual_layer_id)
        for weight in ctx.weights:
            bucket = getattr(weight, "_ddp_bucket_group", None)
            if bucket is not None and bucket.ddp_config.overlap_grad_reduce:
                bucket.register_grad_ready(weight)
        return grad_x, None, None, None, None


class MegaMoEUltraEPMixin:
    """Optional UltraEP path; the router and checkpoint keep logical experts."""

    def _init_ultraep(self):
        self.eplb_enabled = bool(getattr(self.config, "moe_enable_ultraep", False))
        self.eplb_manager = None
        if not self.eplb_enabled:
            return

        from hcu_megatron.core.transformer.moe.eplb_manager import (
            HAVE_EPLB,
            get_or_create_eplb_manager,
        )

        if not HAVE_EPLB:
            raise ImportError("MegaMoE with --moe-enable-ultraep requires ultra_ep.")
        self.eplb_manager = get_or_create_eplb_manager(self.config, self.ep_group)
        assert self.eplb_manager.num_local_redundant_experts > 0, (
            "MegaMoE UltraEP requires --moe-num-redundant-experts-per-rank > 0"
        )
        self.num_local_physical_experts = self.eplb_manager.num_local_physical_experts
        self.num_global_physical_experts = self.eplb_manager.num_global_physical_experts
        self.local_physical_expert_indices = self.eplb_manager.local_physical_expert_indices
        self._eplb_master_ptrs_registered = False
        for module in (self.experts.fc1_weight, self.experts.fc2_weight):
            module.weight.is_eplb_master = True

    def _ultraep_register_master_experts(self, w1, w2):
        if self._eplb_master_ptrs_registered:
            return
        assert self.layer_number is not None, "UltraEP requires a MoE layer number"
        for weight in (w1, w2):
            assert hasattr(weight, "main_grad"), (
                "MegaMoE UltraEP master main_grad missing; call after DDP initialization."
            )
        self.eplb_manager.runtime.construct_local_master_ptr_pool(
            layer_id=self.layer_number,
            fc1_weights=list(w1.detach().unbind()),
            fc2_weights=list(w2.detach().unbind()),
            fc1_grads=list(w1.main_grad.unbind()),
            fc2_grads=list(w2.main_grad.unbind()),
        )
        self._eplb_master_ptrs_registered = True

    def _ultraep_wait(self, event, operation, virtual_layer_id):
        if event is None:
            return
        wait = getattr(self.eplb_manager.runtime, f"wait_{operation}", None)
        if wait is None:
            event.current_stream_wait()
        else:
            wait(event, layer_id=virtual_layer_id)

    def _forward_ultraep(self, hidden_states, padding_mask):
        # Register the real layer before allocating a virtual slot, matching
        # the runtime's native MoE registration order. Module calls complete
        # any pending distributed optimizer all-gathers first.
        if not self._eplb_master_ptrs_registered:
            self._ultraep_register_master_experts(
                self.experts.fc1_weight(), self.experts.fc2_weight()
            )
        virtual_layer_id = self.eplb_manager.allocate_microbatch_slot(self.layer_number)
        self._eplb_last_virtual_layer_id = virtual_layer_id

        def custom_forward(hidden_states):
            return self._ultraep_forward(hidden_states, padding_mask, virtual_layer_id)

        if (
            self.config.recompute_granularity == "selective"
            and "moe" in self.config.recompute_modules
        ):
            from megatron.core import tensor_parallel

            return tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        return custom_forward(hidden_states)

    def _ultraep_forward(self, hidden_states, padding_mask, virtual_layer_id):
        # Calling the modules first also completes distributed optimizer weight
        # all-gathers before UltraEP reads master pointers for weight_sync.
        w1, w2 = self.experts.fc1_weight(), self.experts.fc2_weight()

        input_shape = hidden_states.shape
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = self.router(hidden_states, padding_mask)
        probs = probs.reshape(-1, self.config.num_moe_experts)
        routing_map = routing_map.reshape_as(probs)
        self.eplb_manager.update_placement(virtual_layer_id, routing_map)
        event = self.eplb_manager.runtime.weight_sync(
            layer_id=virtual_layer_id, async_finish=True
        )
        probs, routing_map = self.eplb_manager.reroute(
            virtual_layer_id, probs.float(), routing_map
        )
        # Select by the mask, including selected experts with probability zero.
        # A padded token may have fewer than topk entries; those get zero weight.
        _, topk_idx = routing_map.to(torch.int32).topk(self.router.topk, dim=-1)
        topk_weights = probs.gather(1, topk_idx) * routing_map.gather(1, topk_idx)

        self._ultraep_wait(event, "weight_sync", virtual_layer_id)
        x = hidden_states.reshape(-1, self.hidden_size).to(torch.bfloat16)
        x, packed_w1, packed_w2 = _UltraEPPackedExperts.apply(
            x, w1, w2, self, virtual_layer_id
        )
        y = self.experts(x, topk_idx, topk_weights.float(), weights=(packed_w1, packed_w2))
        y = y.reshape(input_shape).to(hidden_states.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y
