# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence checks for hcu_megatron/core/transformer/moe/dedup_token_dispatcher.py.

The deduplicating dispatcher must compute the same MoE output and gradients as the stock
MoEAlltoAllTokenDispatcher (only the reduction order differs), with and without a correlated
expert placement, while sending fewer rows. Each scenario covers a feature used by a
mainstream model:

    base            dropless top-k                          (Qwen3, Qwen3.5)
    capacity        --moe-expert-capacity-factor 1          (Qwen3-235B, Mixtral)
    shared-overlap  shared experts overlapped with the A2A  (DeepSeek-V3, GLM)
    group-limited   moe_router_num_groups + placement       (DeepSeek-V3)
    etp2            expert tensor parallel size 2           (GPT-567B); needs >= 4 GPUs

Run inside the training container from the repository root, e.g.:
    PYTHONPATH=$PWD:$PWD/3rdparty/Megatron-LM torchrun --nproc-per-node 4 \
        tests/unit_tests/expert_placement/test_dedup_dispatcher.py [--grouped-gemm] [--permute-fusion]
"""
import argparse
import dataclasses
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import megatron.core.transformer.moe.moe_layer as moe_layer_mod  # noqa: E402
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.transformer.moe.moe_layer import MoELayer  # noqa: E402
from tests.unit_tests.test_utilities import Utils  # noqa: E402
from test_expert_placement import (  # noqa: E402
    FAILURES, HIDDEN, MBS, NUM_EXPERTS, SEQ, build, check, close, copy_permuted, gather_slots,
    make_config, set_placement,
)

from hcu_megatron.core.transformer.moe import expert_placement as ec  # noqa: E402
from hcu_megatron.core.transformer.moe.dedup_token_dispatcher import (  # noqa: E402
    MoEDedupAlltoAllTokenDispatcher,
)

STOCK = moe_layer_mod.MoEAlltoAllTokenDispatcher
RESULTS = []
COMMON = {}  # config overrides applied to every scenario (e.g. moe_permute_fusion)


def build_with(dispatcher_cls, config, grouped, seed):
    moe_layer_mod.MoEAlltoAllTokenDispatcher = dispatcher_cls
    try:
        return build(config, grouped, seed)
    finally:
        moe_layer_mod.MoEAlltoAllTokenDispatcher = STOCK


def run(layer, x, g):
    xi = x.clone().requires_grad_(True)
    out, _ = layer(xi)
    out.backward(g)
    return out.detach(), xi.grad


def compare(tag, grouped, phys2log=None, **overrides):
    ep_size = ps.get_expert_model_parallel_world_size()
    config = dataclasses.replace(make_config(ep_size, grouped), **COMMON, **overrides)
    ref = build_with(STOCK, config, grouped, seed=123)
    test = build_with(MoEDedupAlltoAllTokenDispatcher, config, grouped, seed=456)
    check(isinstance(test.token_dispatcher, MoEDedupAlltoAllTokenDispatcher), f"{tag}: dedup dispatcher not used")
    test.load_state_dict(ref.state_dict())
    if phys2log is not None:
        ec.check_group_limited_routing(config, phys2log, ep_size, 1)
        copy_permuted(ref, test, phys2log)
        set_placement(test, phys2log)

    torch.manual_seed(2000 + dist.get_rank())
    x = torch.randn(SEQ, MBS, HIDDEN, device="cuda")
    g = torch.randn(SEQ, MBS, HIDDEN, device="cuda")
    MoEDedupAlltoAllTokenDispatcher.stats_rows = 0
    MoEDedupAlltoAllTokenDispatcher.stats_rows_without_dedup = 0
    out_ref, gx_ref = run(ref, x, g)
    out_test, gx_test = run(test, x, g)
    close(out_ref, out_test, f"{tag}: forward output")
    close(gx_ref, gx_test, f"{tag}: input grad")
    close(ref.router.weight.grad, test.router.weight.grad, f"{tag}: router weight grad")
    if config.moe_shared_expert_intermediate_size:
        for (n, a), (_, b) in zip(ref.shared_experts.named_parameters(), test.shared_experts.named_parameters()):
            close(a.grad, b.grad, f"{tag}: shared expert grad {n}")

    perm = phys2log if phys2log is not None else list(range(NUM_EXPERTS))
    ref_grads = gather_slots(ref, grad=True)
    for p, params in gather_slots(test, grad=True).items():
        for suf, t in params.items():
            close(t, ref_grads[perm[p]][suf], f"{tag}: expert grad slot {p} {suf} vs logical {perm[p]}")

    rows = torch.tensor([MoEDedupAlltoAllTokenDispatcher.stats_rows,
                         MoEDedupAlltoAllTokenDispatcher.stats_rows_without_dedup], device="cuda")
    dist.all_reduce(rows)
    check(0 < int(rows[0]) <= int(rows[1]), f"{tag}: rows sent {int(rows[0])} vs stock {int(rows[1])}")
    RESULTS.append((tag, int(rows[0]), int(rows[1])))


def random_placement(seed):
    return torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)).tolist()


def group_respecting_placement(groups, seed):
    """Permute experts only within each router group's block of slots."""
    per = NUM_EXPERTS // groups
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(groups):
        out += (k * per + torch.randperm(per, generator=g)).tolist()
    return out


def check_group_validation(ep_size):
    """A placement that moves an expert out of its router group must be rejected."""
    config = dataclasses.replace(make_config(ep_size, False), moe_router_num_groups=2, moe_router_group_topk=1)
    bad = list(range(NUM_EXPERTS))
    half = NUM_EXPERTS // 2
    bad[0], bad[half] = bad[half], bad[0]
    try:
        ec.check_group_limited_routing(config, bad, ep_size, 1)
        check(ep_size == 1, "group-limited: placement crossing router groups was not rejected")
    except ValueError:
        pass


def check_hf_import_permutation(grouped):
    """permute_imported_experts must move contiguously loaded experts into placement slots."""
    ep_size = ps.get_expert_model_parallel_world_size()
    config = make_config(ep_size, grouped)
    ref = build(config, grouped, seed=321)
    test = build(config, grouped, seed=654)
    test.load_state_dict(ref.state_dict())  # as bridge would load HF weights: contiguous order
    phys2log = random_placement(5)
    set_placement(test, phys2log)
    ec.permute_imported_experts([test])
    logical = gather_slots(ref)
    for p, params in gather_slots(test).items():
        for suf, t in params.items():
            close(t, logical[phys2log[p]][suf], f"hf-import: slot {p} {suf} != logical {phys2log[p]}", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped-gemm", action="store_true", help="TEGroupedMLP instead of SequentialMLP")
    ap.add_argument("--permute-fusion", action="store_true",
                    help="--moe-permute-fusion: Transformer Engine permute/unpermute kernels in both dispatchers")
    args = ap.parse_args()
    gg = args.grouped_gemm
    if args.permute_fusion:
        COMMON["moe_permute_fusion"] = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    MoELayer.route = ec.moe_layer_route_placement_wrapper(MoELayer.route)

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=Utils.world_size)
    ep_size = ps.get_expert_model_parallel_world_size()
    compare("base/contiguous", gg)
    compare("base/placement", gg, random_placement(0))
    compare("capacity/placement", gg, random_placement(1), moe_expert_capacity_factor=1.0)
    compare("shared-overlap/placement", gg, random_placement(2),
            moe_shared_expert_intermediate_size=96, moe_shared_expert_overlap=True)
    if ep_size % 2 == 0:
        compare("group-limited/placement", gg, group_respecting_placement(2, 3),
                moe_router_num_groups=2, moe_router_group_topk=1)
    check_group_validation(ep_size)
    check_hf_import_permutation(gg)

    if Utils.world_size >= 4 and Utils.world_size % 2 == 0:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=Utils.world_size // 2,
                                        expert_tensor_parallel_size=2)
        compare("etp2/placement", gg, random_placement(4), expert_tensor_parallel_size=2,
                expert_model_parallel_size=Utils.world_size // 2)

    failed = torch.tensor([len(FAILURES)], device="cuda")
    dist.all_reduce(failed)
    if dist.get_rank() == 0:
        for tag, sent, base in RESULTS:
            print(f"[{tag}] rows sent {sent} vs stock {base} ({100.0 * (1 - sent / base):.1f}% removed)")
        print("dedup dispatcher checks:", "PASSED" if int(failed) == 0 else f"FAILED ({int(failed)} failures)",
              flush=True)
    Utils.destroy_model_parallel()
    sys.exit(0 if int(failed) == 0 else 1)


if __name__ == "__main__":
    main()
