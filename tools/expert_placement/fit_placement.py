"""Fit per-layer correlated expert placement on routing traces and report
paper Table I metrics on held-out traces.

Trace file (torch.save, loaded with weights_only=True), one of:
  - int tensor [layers, tokens, topk]   selected expert IDs
  - bool/0-1 tensor [layers, tokens, num_experts]   routing maps
  - a single 2-D tensor (one layer)
  - a list of per-layer 2-D tensors
Several files are concatenated along the token axis.

Example:
  python fit_placement.py --fit step1991.pt step1995.pt --eval step1983.pt \
      --num-experts 64 --ep 8 --nodes 2 --out placement.json
"""
import argparse
import sys
from typing import List
import torch
from expert_coupling import (
    ExpertCouplingStats,
    contiguous_placement,
    layerwise_placement,
    placement_metrics,
    save_layer_placements,
    to_routing_map,
)

METRICS = ("same_gpu_pct", "gpus_per_token", "dedup_pct", "recv_imbalance", "compute_imbalance")


def load_traces(paths: List[str], num_experts: int) -> List[torch.Tensor]:
    layers = None
    for path in paths:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 2:
                obj = [obj]
            elif obj.ndim == 3:
                obj = list(obj.unbind(0))
            else:
                raise ValueError(f"{path}: expected a 2-D or 3-D tensor, got {tuple(obj.shape)}")
        if not isinstance(obj, (list, tuple)) or not all(isinstance(t, torch.Tensor) for t in obj):
            raise ValueError(f"{path}: expected a tensor or a list of tensors")
        maps = [to_routing_map(t, num_experts) for t in obj]
        if layers is None:
            layers = [[m] for m in maps]
        elif len(maps) != len(layers):
            raise ValueError(f"{path}: {len(maps)} layers, expected {len(layers)}")
        else:
            for acc, m in zip(layers, maps):
                acc.append(m)
    return [torch.cat(acc, 0) for acc in layers]


def fmt(m):
    s = "  ".join(f"{k} {m[k]:7.2f}" for k in METRICS)
    if "nodes_per_token" in m:
        s += f"  nodes_per_token {m['nodes_per_token']:5.2f}"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True, help="trace files used to build co-selection stats")
    ap.add_argument("--eval", nargs="+", help="held-out trace files used only for metrics")
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--ep", type=int, required=True, help="expert-parallel size")
    ap.add_argument("--nodes", type=int, default=1, help="nodes spanned by the EP group (hierarchical placement)")
    ap.add_argument("--max-swaps", type=int, default=1000)
    ap.add_argument("--init", choices=("contiguous", "greedy"), default="contiguous")
    ap.add_argument("--max-load-ratio", type=float, default=None,
                    help="cap each node/rank at this multiple of the mean token load (e.g. 1.1); "
                         "default: no cap")
    ap.add_argument("--router-groups", type=int, default=1,
                    help="moe_router_num_groups of the model (group-limited routing); experts then "
                         "only move within the EP ranks of their router group")
    ap.add_argument("--out", default="placement.json")
    args = ap.parse_args(argv)

    fit = load_traces(args.fit, args.num_experts)
    if args.eval:
        held_out = load_traces(args.eval, args.num_experts)
        if len(held_out) != len(fit):
            sys.exit(f"--eval has {len(held_out)} layers, --fit has {len(fit)}")
    else:
        print("warning: no --eval traces; metrics below are in-sample and optimistic", file=sys.stderr)
        held_out = fit

    stats = []
    for routes in fit:
        s = ExpertCouplingStats(args.num_experts)
        s.update(routes)
        stats.append(s)
    placements = layerwise_placement(stats, args.ep, num_nodes=args.nodes,
                                      max_swaps=args.max_swaps, init=args.init,
                                      max_load_ratio=args.max_load_ratio,
                                      router_groups=args.router_groups)

    base = contiguous_placement(args.num_experts, args.ep)
    keys = METRICS + (("nodes_per_token",) if args.nodes > 1 else ())
    totals = {"contiguous": {k: 0.0 for k in keys}, "correlated": {k: 0.0 for k in keys}}
    for layer, (p, routes) in enumerate(zip(placements, held_out)):
        mb = placement_metrics(routes, base, args.ep, num_nodes=args.nodes)
        mc = placement_metrics(routes, p.expert_to_rank, args.ep, num_nodes=args.nodes)
        print(f"layer {layer:3d} ({p.num_swaps} swaps)")
        print(f"  contiguous  {fmt(mb)}")
        print(f"  correlated  {fmt(mc)}")
        for k in keys:
            totals["contiguous"][k] += mb[k]
            totals["correlated"][k] += mc[k]
    n = max(1, len(placements))
    print("mean over layers")
    for name, t in totals.items():
        print(f"  {name:<10}  " + "  ".join(f"{k} {v / n:7.2f}" for k, v in t.items()))

    save_layer_placements(args.out, placements, args.ep)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
