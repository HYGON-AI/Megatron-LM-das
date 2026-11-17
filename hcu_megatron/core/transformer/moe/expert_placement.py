# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Correlated expert placement (arXiv 2610.09372, Alg. 1) for MoELayer.

``placement.json`` (written by fit_placement.py) assigns every logical expert of
every MoE layer to an EP rank. It is applied without touching the dispatcher or
the experts:

* physical expert ``p`` = position in ``rank_to_experts[0] + rank_to_experts[1] + ...``;
  rank ``r`` keeps hosting physical experts ``[r*E/P, (r+1)*E/P)``, exactly as in
  stock Megatron, so ``local_expert_indices`` / dispatcher slicing stay valid.
* ``MoELayer.route`` reorders the router's ``probs`` / ``routing_map`` columns from
  logical to physical order (``phys2log``). The router itself, aux losses and
  expert bias stay in logical order; gradients flow back through ``index_select``.
* ``MoELayer.sharded_state_dict`` rewrites the expert axis of every expert
  ShardedTensor / ShardedObject / factory from physical to logical index, so a
  ``torch_dist`` checkpoint always stores experts by logical ID and can be loaded
  under any placement, including stock contiguous placement.

The dispatcher sees ordinary contiguous placement; with ``--moe-token-dispatcher-type
flex`` (DeepEP) each token is already sent once per destination rank, so co-located
experts directly reduce dispatch/combine traffic.
"""
import json
import logging
from dataclasses import replace
from functools import lru_cache, wraps
from typing import List, Optional, Sequence, Tuple

import torch
from megatron.core import utils
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_placement(path: str, num_experts: int, ep_size: int) -> Tuple[Tuple[int, ...], ...]:
    """Return one ``phys2log`` tuple per MoE layer, validated against the model config."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("num_experts") != num_experts:
        raise ValueError(f"{path}: num_experts={payload.get('num_experts')}, model has {num_experts}")
    if payload.get("num_ranks") != ep_size:
        raise ValueError(f"{path}: num_ranks={payload.get('num_ranks')}, expert-model-parallel-size is {ep_size}")
    per_rank = num_experts // ep_size
    layers = []
    for i, layer in enumerate(payload["layers"]):
        rank_to_experts = layer["rank_to_experts"]
        if len(rank_to_experts) != ep_size or any(len(e) != per_rank for e in rank_to_experts):
            raise ValueError(f"{path}: layer {i} must give every rank exactly {per_rank} experts")
        phys2log = tuple(int(e) for experts in rank_to_experts for e in experts)
        if sorted(phys2log) != list(range(num_experts)):
            raise ValueError(f"{path}: layer {i} is not a permutation of 0..{num_experts - 1}")
        layers.append(phys2log)
    return tuple(layers)


def _moe_layer_pattern(config) -> List[int]:
    freq = config.moe_layer_freq
    if isinstance(freq, int):
        return [1 if i % freq == 0 else 0 for i in range(config.num_layers)]
    return list(freq)


def moe_layer_ordinal(config, layer_number: int) -> int:
    """0-based index of decoder layer ``layer_number`` (1-based) among MoE layers."""
    pattern = _moe_layer_pattern(config)
    if not pattern[layer_number - 1]:
        raise ValueError(f"layer {layer_number} is not an MoE layer according to moe_layer_freq")
    return sum(pattern[: layer_number - 1])


def check_group_limited_routing(config, phys2log: Sequence[int], ep_size: int, layer_number: int) -> None:
    """Group-limited routing must keep each logical group on its own set of EP ranks.

    With ``moe_router_num_groups = G`` the router picks ``group_topk`` groups of E/G
    contiguous logical experts per token; G is chosen so that a group maps to a device or a
    node (device-/node-limited routing). A placement may permute experts only within the
    EP ranks their group occupies under contiguous placement, otherwise the routing limit
    no longer bounds the number of devices/nodes a token reaches.
    """
    groups = getattr(config, "moe_router_num_groups", None) or 1
    if groups <= 1:
        return
    E = len(phys2log)
    per_group, per_rank = E // groups, E // ep_size
    for p, e in enumerate(phys2log):
        rank, group = p // per_rank, e // per_group
        # Ranks occupied by `group` under contiguous placement.
        lo, hi = (group * per_group) // per_rank, ((group + 1) * per_group - 1) // per_rank
        if not lo <= rank <= hi:
            raise ValueError(
                f"layer {layer_number}: expert {e} (router group {group}) placed on EP rank {rank}, "
                f"outside ranks {lo}-{hi} of its group; fit the placement with --router-groups {groups}"
            )


def bridge_to_megatron_provider_placement_wrapper(to_megatron_provider_func):
    """Permute HF-imported experts into their placement slots before DDP wrapping.

    Bridge loads HF expert ``e`` into physical slot ``e`` (stock contiguous order). Physical
    slot ``p`` must hold logical expert ``phys2log[p]``; the weights of logical expert e live
    on rank ``e // per_rank`` after import, so they are exchanged across the EP group.
    """

    @wraps(to_megatron_provider_func)
    def wrapper(self, *args, **kwargs):
        provider = to_megatron_provider_func(self, *args, **kwargs)
        if kwargs.get("load_weights", args[0] if args else True):
            provider.register_pre_wrap_hook(permute_imported_experts)
        return provider

    return wrapper


@torch.no_grad()
def permute_imported_experts(model):
    """Pre-wrap hook: move every MoE layer's experts from logical to placement order."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            phys2log = getattr(module, "ec_phys2log", None)
            if phys2log is not None and getattr(module, "experts", None) is not None:
                _permute_layer_experts(module, phys2log.tolist())
    return model


def _local_expert_tensors(experts) -> List[List[torch.Tensor]]:
    """Per local slot, the list of its parameter tensors (SequentialMLP or TEGroupedMLP)."""
    import re

    slots = {}
    for name, p in experts.named_parameters():
        m = re.fullmatch(r"local_experts\.(\d+)\..+", name) or re.fullmatch(r".+\D(\d+)", name)
        if m is None:
            raise RuntimeError(f"cannot map expert parameter {name} to a local slot")
        slots.setdefault(int(m.group(1)), []).append(p)
    return [slots[i] for i in sorted(slots)]


def _permute_layer_experts(layer, phys2log: List[int]) -> None:
    group = layer.ep_group
    ep_size, ep_rank = utils.get_pg_size(group), utils.get_pg_rank(group)
    slots = _local_expert_tensors(layer.experts)
    per_rank = len(slots)
    flat = [torch.cat([t.reshape(-1) for t in params]) for params in slots]
    numel = flat[0].numel()
    # Slot i of this rank currently holds logical expert ep_rank*per_rank + i; destination is
    # the physical slot that maps to that logical expert.
    log2phys = [0] * len(phys2log)
    for p, e in enumerate(phys2log):
        log2phys[e] = p
    send = [[] for _ in range(ep_size)]
    for i in range(per_rank):
        dst = log2phys[ep_rank * per_rank + i]
        send[dst // per_rank].append((dst % per_rank, flat[i]))
    # Fixed-size exchange: every rank sends per_rank expert blobs, prefixed by target slot.
    send_buf = torch.empty(ep_size, per_rank, numel + 1, dtype=flat[0].dtype, device=flat[0].device)
    counts = [len(s) for s in send]
    for r, items in enumerate(send):
        for j, (slot, blob) in enumerate(items):
            send_buf[r, j, 0] = slot
            send_buf[r, j, 1:] = blob
    cnt = torch.tensor(counts, device=send_buf.device)
    recv_cnt = torch.empty_like(cnt)
    recv_buf = torch.empty_like(send_buf)
    if ep_size > 1:
        torch.distributed.all_to_all_single(recv_cnt, cnt, group=group)
        torch.distributed.all_to_all_single(recv_buf, send_buf, group=group)
    else:
        recv_cnt.copy_(cnt)
        recv_buf.copy_(send_buf)
    for r, n in enumerate(recv_cnt.tolist()):
        for j in range(n):
            slot = int(recv_buf[r, j, 0])
            offset = 0
            for t in slots[slot]:
                t.copy_(recv_buf[r, j, 1 + offset : 1 + offset + t.numel()].view_as(t))
                offset += t.numel()


def moe_layer_set_layer_number_placement_wrapper(set_layer_number_func):
    """Attach ``ec_phys2log`` once the global layer number is known.

    TransformerLayer builds MoELayer without ``layer_number`` and calls
    ``set_layer_number`` right after, so the placement is looked up here.
    """

    @wraps(set_layer_number_func)
    def wrapper(self, layer_number: int):
        set_layer_number_func(self, layer_number)
        from hcu_megatron.training.arguments import get_adaptor_args

        path = getattr(get_adaptor_args(), "moe_expert_placement_path", None)
        if not path:
            return
        if self.is_mtp_layer:
            logger.warning("expert placement is not applied to MTP layers; keeping contiguous placement")
            return
        config = self.config
        ep_size = utils.get_pg_size(self.ep_group)
        layers = load_placement(path, config.num_moe_experts, ep_size)
        num_moe_layers = sum(_moe_layer_pattern(config))
        if len(layers) != num_moe_layers:
            raise ValueError(f"{path}: {len(layers)} layers, model has {num_moe_layers} MoE layers")
        phys2log = layers[moe_layer_ordinal(config, layer_number)]
        check_group_limited_routing(config, phys2log, ep_size, layer_number)
        if moe_layer_ordinal(config, layer_number) == 0 and torch.distributed.get_rank() == 0:
            print(f"[expert placement] {path}: {len(layers)} MoE layers, EP={ep_size}, "
                  f"layer {layer_number} rank 0 hosts experts {list(phys2log[: config.num_moe_experts // ep_size])}",
                  flush=True)
        # Non-persistent integer buffer: follows .cuda(), is not saved, is not cast to bf16.
        self.register_buffer("ec_phys2log", torch.tensor(phys2log, dtype=torch.long), persistent=False)

    return wrapper


def moe_layer_route_placement_wrapper(route_func):
    @wraps(route_func)
    def wrapper(self, *args, **kwargs):
        probs, routing_map = route_func(self, *args, **kwargs)
        phys2log = getattr(self, "ec_phys2log", None)
        if phys2log is not None:
            if phys2log.device != probs.device:
                phys2log = self.ec_phys2log = phys2log.to(probs.device)
            # Column p of the output is logical expert phys2log[p].
            probs = probs.index_select(-1, phys2log)
            routing_map = routing_map.index_select(-1, phys2log)
        return probs, routing_map

    return wrapper


def _remap_expert_axis(obj, axis: int, phys2log: Sequence[int], num_experts: int):
    """Rewrite physical -> logical expert index on ``axis`` of a sharded checkpoint entry."""
    if isinstance(obj, dict):
        return {k: _remap_expert_axis(v, axis, phys2log, num_experts) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_remap_expert_axis(v, axis, phys2log, num_experts) for v in obj)
    if isinstance(obj, ShardedTensorFactory):
        build_fn = obj.build_fn

        def remapped_build_fn(*args, **kwargs):
            return _remap_expert_axis(build_fn(*args, **kwargs), axis, phys2log, num_experts)

        return replace(obj, build_fn=remapped_build_fn)
    if isinstance(obj, ShardedTensor):
        if not (axis < obj.prepend_axis_num and obj.axis_fragmentations[axis] == num_experts):
            raise RuntimeError(f"{obj.key}: expected expert axis {axis} with {num_experts} shards")
        offset = list(obj.global_offset)
        offset[axis] = phys2log[offset[axis]]
        return replace(obj, global_offset=tuple(offset))
    if isinstance(obj, ShardedObject):
        if obj.global_shape[axis] != num_experts:
            raise RuntimeError(f"{obj.key}: expected expert axis {axis} with {num_experts} shards")
        offset = list(obj.global_offset)
        offset[axis] = phys2log[offset[axis]]
        return replace(obj, global_offset=tuple(offset))
    return obj


def moe_layer_sharded_state_dict_placement_wrapper(sharded_state_dict_func):
    @wraps(sharded_state_dict_func)
    def wrapper(self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None):
        sharded_sd = sharded_state_dict_func(self, prefix, sharded_offsets, metadata)
        phys2log = getattr(self, "ec_phys2log", None)
        if phys2log is None:
            return sharded_sd
        if (metadata or {}).get("singleton_local_shards", False):
            raise NotImplementedError("expert placement does not support singleton_local_shards checkpoints")
        phys2log = phys2log.tolist()
        num_experts = self.config.num_moe_experts
        # Experts append their EP axis right after the offsets passed in here.
        axis = len(sharded_offsets)
        experts_prefix = f"{prefix}experts."
        for k, v in sharded_sd.items():
            if k.startswith(experts_prefix):
                sharded_sd[k] = _remap_expert_axis(v, axis, phys2log, num_experts)
        return sharded_sd

    return wrapper
