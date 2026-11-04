# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Collect per-layer MoE routing maps for offline expert-placement fitting.

Disabled unless EC_TRACE_ITERS is set. Environment:
    EC_TRACE_ITERS       comma-separated iterations to record, e.g. 1983,1987,1991
                         (same numbering as train_step's ``iteration`` argument)
    EC_TRACE_DIR         output directory (default: ec_traces)
    EC_TRACE_MAX_TOKENS  tokens kept per layer per iteration (default: 8192)

One rank per pipeline stage writes (data-parallel rank 0, tensor-parallel rank 0):
    routes_iter{it}_pp{pp}.pt   list of bool [tokens, num_experts], one per local MoE layer
    layers_pp{pp}.json          module names of those layers, same order
Merge stages into one file per iteration (run the file directly: ``python -m`` imports the
hcu_megatron package, which initializes the GPU and fails on a GPU-less shell):
    python hcu_megatron/training/route_collector.py merge --dir ec_traces --pp 1 --iters 1983 1987
"""
import argparse
import json
import os
from typing import List, Optional, Sequence

import torch


class RouteCollector:
    def __init__(self, model, iters: Sequence[int], out_dir: str = "ec_traces",
                 max_tokens: int = 8192, pp_rank: int = 0, writer: bool = True):
        chunks = model if isinstance(model, (list, tuple)) else [model]
        # Decoder MoE layers only: MTP routers are not part of the placement (and would shift
        # the layer count). Recorded with their global layer number so pipeline stages and
        # virtual-pipeline chunks can be merged in model order.
        self.routers = [(f"chunk{c}.{name}", mod)
                        for c, chunk in enumerate(chunks)
                        for name, mod in chunk.named_modules()
                        if type(mod).__name__ == "TopKRouter" and ".mtp." not in f".{name}."]
        if not self.routers:
            raise RuntimeError("no TopKRouter modules found; adapt RouteCollector to your router class")
        self.iters = set(iters)
        self.out_dir = out_dir
        self.max_tokens = max_tokens
        self.pp_rank = pp_rank
        self.writer = writer
        self.active = False
        self.iteration = None
        self.buf: List[List[torch.Tensor]] = [[] for _ in self.routers]
        self.count = [0] * len(self.routers)
        if writer:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"layers_pp{pp_rank}.json"), "w", encoding="utf-8") as f:
                json.dump([{"name": n, "layer_number": m.layer_number} for n, m in self.routers], f, indent=2)
        for i, (_, mod) in enumerate(self.routers):
            mod.register_forward_hook(self._hook(i))

    @classmethod
    def from_env(cls, model):
        iters = os.environ.get("EC_TRACE_ITERS", "")
        if not iters:
            return None
        from megatron.core import parallel_state as ps
        writer = ps.get_data_parallel_rank() == 0 and ps.get_tensor_model_parallel_rank() == 0
        return cls(model, [int(x) for x in iters.split(",") if x],
                   out_dir=os.environ.get("EC_TRACE_DIR", "ec_traces"),
                   max_tokens=int(os.environ.get("EC_TRACE_MAX_TOKENS", "8192")),
                   pp_rank=ps.get_pipeline_model_parallel_rank(), writer=writer)

    def _hook(self, i):
        def fn(module, inputs, output):
            # Grad-enabled forward only: skips the no-grad first pass of activation
            # recompute (and eval), so each microbatch is recorded once.
            if not (self.active and self.writer and torch.is_grad_enabled()):
                return
            need = self.max_tokens - self.count[i]
            if need <= 0:
                return
            outs = output if isinstance(output, (tuple, list)) else (output,)
            rm = next((t for t in outs if torch.is_tensor(t) and t.dtype == torch.bool and t.ndim == 2), None)
            if rm is None:
                raise RuntimeError("router output has no bool [tokens, experts] routing map; adapt RouteCollector._hook")
            rm = rm[:need].detach().cpu()  # syncs, but only on recorded iterations
            self.buf[i].append(rm)
            self.count[i] += rm.shape[0]
        return fn

    def begin_step(self, iteration: int):
        self.iteration = iteration
        self.active = iteration in self.iters

    def end_step(self):
        if self.active and self.writer:
            if any(c == 0 for c in self.count):
                raise RuntimeError(f"iteration {self.iteration}: some MoE layers recorded no tokens")
            path = os.path.join(self.out_dir, f"routes_iter{self.iteration}_pp{self.pp_rank}.pt")
            torch.save([torch.cat(b, 0) for b in self.buf], path)
            print(f"[RouteCollector] wrote {path} ({len(self.buf)} layers, {min(self.count)} tokens/layer)", flush=True)
        self.active = False
        self.buf = [[] for _ in self.routers]
        self.count = [0] * len(self.routers)


_ROUTE_COLLECTOR: Optional[RouteCollector] = None


def init_route_collector(model) -> Optional[RouteCollector]:
    global _ROUTE_COLLECTOR
    _ROUTE_COLLECTOR = RouteCollector.from_env(model)
    return _ROUTE_COLLECTOR


def get_route_collector() -> Optional[RouteCollector]:
    return _ROUTE_COLLECTOR


def merge(out_dir: str, pp: int, iters: Sequence[int]):
    """Concatenate per-stage traces into one file per iteration, in global layer order."""
    entries = []  # (layer_number, stage, index within the stage file, name)
    for s in range(pp):
        with open(os.path.join(out_dir, f"layers_pp{s}.json"), encoding="utf-8") as f:
            for i, item in enumerate(json.load(f)):
                if isinstance(item, str):  # traces written before layer numbers were recorded
                    item = {"name": item, "layer_number": None}
                entries.append((item["layer_number"], s, i, f"pp{s}.{item['name']}"))
    if all(e[0] is not None for e in entries):
        entries.sort(key=lambda e: e[0])
        numbers = [e[0] for e in entries]
        assert len(set(numbers)) == len(numbers), f"duplicate layer numbers: {numbers}"
    with open(os.path.join(out_dir, "layers.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": e[3], "layer_number": e[0]} for e in entries], f, indent=2)
    for it in iters:
        stages = [torch.load(os.path.join(out_dir, f"routes_iter{it}_pp{s}.pt"), map_location="cpu",
                             weights_only=True) for s in range(pp)]
        layers: List[torch.Tensor] = [stages[s][i] for _, s, i, _ in entries]
        path = os.path.join(out_dir, f"s{it}.pt")
        torch.save(layers, path)
        print(f"wrote {path}: {len(layers)} layers, tokens/layer {[t.shape[0] for t in layers][:4]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="merge per-pipeline-stage traces into one file per iteration")
    m.add_argument("--dir", default="ec_traces")
    m.add_argument("--pp", type=int, default=1, help="pipeline-parallel size used during collection")
    m.add_argument("--iters", type=int, nargs="+", required=True)
    a = ap.parse_args()
    merge(a.dir, a.pp, a.iters)
