# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence checks for hcu_megatron/core/transformer/moe/expert_placement.py.

A MoELayer with a placement must compute exactly the same function as the
stock layer when physical slot p holds the weights of logical expert
phys2log[p], and its torch_dist checkpoint must store experts by logical ID.

Run inside the training container from the repository root, e.g.:
    PYTHONPATH=$PWD:$PWD/3rdparty/Megatron-LM torchrun --nproc-per-node 4 \
        tests/unit_tests/expert_placement/test_expert_placement.py [--grouped-gemm]

Checks (each rank prints FAIL lines; the script exits non-zero on any failure):
  1. load_placement / moe_layer_ordinal parse and validate placement.json;
  2. forward output, input grad, router grad and per-logical-expert grads match;
  3. checkpoint saved under placement A loads into stock placement and into
     placement B with every logical expert landing in the right slot.
"""
import argparse
import json
import os
import re
import sys
import tempfile

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

from hcu_megatron.core.transformer.moe import expert_placement as ec

NUM_EXPERTS, TOPK, HIDDEN, SEQ, MBS = 16, 4, 64, 32, 2
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"[rank {dist.get_rank()}] FAIL: {msg}", flush=True)


def close(a, b, msg, atol=1e-5, rtol=1e-4):
    a, b = a.detach(), b.detach()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    check(ok, f"{msg}: max abs diff {float((a - b).abs().max()):.3e}")


def make_config(ep_size, grouped):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped,
        gated_linear_unit=True,  # exercises the swiglu ShardedTensorFactory path
        add_bias_linear=False,
        expert_model_parallel_size=ep_size,
        params_dtype=torch.float32,
    )


def build(config, grouped, seed):
    if grouped:
        spec = get_gpt_layer_with_transformer_engine_submodules(num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
    else:
        spec = get_gpt_layer_local_submodules(num_experts=NUM_EXPERTS, moe_grouped_gemm=False)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    layer = MoELayer(config, get_submodules(spec.mlp)).cuda()
    layer.set_layer_number(1)
    return layer


def set_placement(layer, phys2log):
    layer.register_buffer("ec_phys2log", torch.tensor(phys2log, dtype=torch.long).cuda(), persistent=False)


def local_expert_params(layer):
    """{local slot: {param suffix: param}} for SequentialMLP and TEGroupedMLP."""
    out = {}
    for name, p in layer.experts.named_parameters():
        m = re.fullmatch(r"local_experts\.(\d+)\.(.+)", name)
        if m:
            out.setdefault(int(m[1]), {})[m[2]] = p
            continue
        m = re.fullmatch(r"(.+\D)(\d+)", name)  # TEGroupedMLP: linear_fc1.weight3
        if m:
            out.setdefault(int(m[2]), {})[m[1]] = p
            continue
        raise RuntimeError(f"unexpected expert parameter {name}")
    return out


def gather_slots(layer, grad=False):
    """All physical slots across the EP group: {physical id: {suffix: tensor}}."""
    group = ps.get_expert_model_parallel_group()
    size = group.size()
    local = local_expert_params(layer)
    per = len(local)
    full = {}
    for i in range(per):
        for suf, p in local[i].items():
            t = (p.grad if grad else p).detach().contiguous()
            buf = [torch.empty_like(t) for _ in range(size)]
            dist.all_gather(buf, t, group=group)
            for r in range(size):
                full.setdefault(r * per + i, {})[suf] = buf[r]
    return full


def copy_permuted(src_layer, dst_layer, phys2log):
    """dst slot p <- logical expert phys2log[p] of src (src uses stock placement)."""
    logical = gather_slots(src_layer)
    rank = ps.get_expert_model_parallel_rank()
    local = local_expert_params(dst_layer)
    per = len(local)
    with torch.no_grad():
        for i, params in local.items():
            for suf, p in params.items():
                p.copy_(logical[phys2log[rank * per + i]][suf])


def check_slots(layer, phys2log, reference, what):
    slots = gather_slots(layer)
    for p, params in slots.items():
        for suf, t in params.items():
            close(t, reference[phys2log[p]][suf], f"{what}: slot {p} {suf} != logical {phys2log[p]}", 0, 0)


def test_placement_file(ep_size, tmpdir):
    config = make_config(ep_size, False)
    perm = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(1)).tolist()
    per = NUM_EXPERTS // ep_size
    path = os.path.join(tmpdir, "placement.json")
    if dist.get_rank() == 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_ranks": ep_size, "num_experts": NUM_EXPERTS,
                       "layers": [{"rank_to_experts": [sorted(perm[r * per:(r + 1) * per]) for r in range(ep_size)]}]}, f)
    dist.barrier()
    (phys2log,) = ec.load_placement(path, NUM_EXPERTS, ep_size)
    check(sorted(phys2log) == list(range(NUM_EXPERTS)), "load_placement must return a permutation")
    check(ec.moe_layer_ordinal(config, 1) == 0, "moe_layer_ordinal(layer 1) should be 0")
    try:
        ec.load_placement(path, NUM_EXPERTS, ep_size * 2)
        check(False, "load_placement must reject a mismatched EP size")
    except ValueError:
        pass


def test_forward_backward(ep_size, grouped, phys2log):
    config = make_config(ep_size, grouped)
    ref = build(config, grouped, seed=123)
    test = build(config, grouped, seed=456)
    test.load_state_dict(ref.state_dict())
    copy_permuted(ref, test, phys2log)
    set_placement(test, phys2log)

    torch.manual_seed(1000 + dist.get_rank())
    x = torch.randn(SEQ, MBS, HIDDEN, device="cuda")
    g = torch.randn(SEQ, MBS, HIDDEN, device="cuda")
    outs, grads = [], []
    for layer in (ref, test):
        xi = x.clone().requires_grad_(True)
        out, _ = layer(xi)
        out.backward(g)
        outs.append(out.detach())
        grads.append(xi.grad)
    close(outs[0], outs[1], "forward output")
    close(grads[0], grads[1], "input grad")
    close(ref.router.weight.grad, test.router.weight.grad, "router weight grad")

    ref_grads = gather_slots(ref, grad=True)
    for p, params in gather_slots(test, grad=True).items():
        for suf, t in params.items():
            close(t, ref_grads[phys2log[p]][suf], f"expert grad slot {p} {suf} vs logical {phys2log[p]}")
    return ref, test


def test_checkpoint(ep_size, grouped, ref, test, phys2log, ckpt_dir):
    logical = gather_slots(ref)
    dist_checkpointing.save(test.sharded_state_dict(), ckpt_dir)

    config = make_config(ep_size, grouped)
    stock = build(config, grouped, seed=789)
    stock.load_state_dict(dist_checkpointing.load(stock.sharded_state_dict(), ckpt_dir))
    check_slots(stock, list(range(NUM_EXPERTS)), logical, "load into stock placement")
    close(stock.router.weight, ref.router.weight, "router weight after load", 0, 0)

    other = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(2)).tolist()
    moved = build(config, grouped, seed=790)
    set_placement(moved, other)
    moved.load_state_dict(dist_checkpointing.load(moved.sharded_state_dict(), ckpt_dir))
    check_slots(moved, other, logical, "load into a different placement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped-gemm", action="store_true", help="TEGroupedMLP instead of SequentialMLP")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=Utils.world_size)
    ep_size = ps.get_expert_model_parallel_world_size()
    assert NUM_EXPERTS % ep_size == 0, f"world size {ep_size} must divide {NUM_EXPERTS}"

    # Apply the same wrappers the ExpertPlacementFeature registers.
    MoELayer.route = ec.moe_layer_route_placement_wrapper(MoELayer.route)
    MoELayer.sharded_state_dict = ec.moe_layer_sharded_state_dict_placement_wrapper(MoELayer.sharded_state_dict)

    phys2log = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(0)).tolist()
    tmp = [tempfile.mkdtemp(prefix="ec_placement_test_") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(tmp, src=0)
    ckpt_dir = os.path.join(tmp[0], "ckpt")
    if dist.get_rank() == 0:
        os.makedirs(ckpt_dir)
    dist.barrier()

    test_placement_file(ep_size, tmp[0])
    ref, test = test_forward_backward(ep_size, args.grouped_gemm, phys2log)
    test_checkpoint(ep_size, args.grouped_gemm, ref, test, phys2log, ckpt_dir)

    failed = torch.tensor([len(FAILURES)], device="cuda")
    dist.all_reduce(failed)
    if dist.get_rank() == 0:
        print("expert placement checks:", "PASSED" if int(failed) == 0 else f"FAILED ({int(failed)} failures)", flush=True)
    Utils.destroy_model_parallel()
    sys.exit(0 if int(failed) == 0 else 1)


if __name__ == "__main__":
    main()
