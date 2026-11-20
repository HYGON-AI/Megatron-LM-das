# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Deduplicating all-to-all token dispatcher (arXiv 2610.09372, Sec. V-A a).

The stock ``MoEAlltoAllTokenDispatcher`` sends one row per (token, expert): with top-k
routing a token whose k experts share a rank is sent k times. This dispatcher sends one
row per unique (token, destination EP rank). Each row carries the routing probabilities
of the token's experts on that rank (-inf for experts it does not use), bit-cast into extra
columns of the same all-to-all buffer as the hidden states:

    dispatch_preprocess   [T, H] -> packed [S, W]: rows grouped by destination rank, then meta
    token_dispatch        one all-to-all over the EP group -> [R, W]
    dispatch_postprocess  (ETP > 1) all-gather rows/meta in the expert-TP group;
                          expand each row to its selected local experts, grouped by expert
                          -> [N, H], tokens_per_expert [L], probs [N]
    (experts)             TEGroupedMLP / SequentialMLP apply the probs in the activation
    combine_preprocess    sum the (already prob-weighted) expert outputs of each row (fp32);
                          (ETP > 1) reduce-scatter in the expert-TP group -> [R, H]
    token_combine         reverse all-to-all -> [S, H]
    combine_postprocess   sum the partial outputs of each token over its ranks -> [T, H]

The math is the same as the stock dispatcher (sum over the token's experts of
prob * expert(x)); only the reduction order changes.

Index maps. Each side builds one map with a 1-D cumsum over a flattened mask, with no host
sync: ``pos`` (1-based slot of every (token, rank) / (expert, row) pair, 0 if unused) and its
compaction ``idx`` (the pair of every slot). Every data movement, forward and backward, is a
gather by ``idx`` or a sum over a strided view of ``pos``; summing as a gather writes each
output row once, so no zero fill, atomics or separate dtype conversion are needed, and the
result is deterministic. Both run in one small kernel (csrc/dedup_gather_sum.cu, JIT-built
on first use; set HCU_DEDUP_NATIVE=0, or on build failure, to use equivalent PyTorch ops).
On ROCm the Transformer Engine permute/unpermute (Triton) cost ~0.5-1 ms of host time per
call, which dominated a host-bound step; ``--moe-permute-fusion`` does not affect this
dispatcher.

Expert placement is orthogonal: with ``--moe-expert-placement-path`` the routing map is
already in physical order, and co-located experts collapse into one row here.

Supported: dropless and capacity-factor (token-dropping) routing, expert tensor parallelism,
shared-expert overlap, router padding for FP8/FP4. Not supported: padding expert inputs to
capacity (``--moe-pad-expert-input-to-capacity``) and CUDA graphs, which need static shapes.
One device-to-host sync per layer (split sizes), as in the stock dispatcher.
"""
import os
import warnings
from typing import List

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import (
    get_align_size_for_quantization,
    pad_routing_map,
)
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

# meta value of an expert the row does not use; router probs are finite, so "> -inf" is the mask.
_UNUSED = float("-inf")

# ----------------------------------------------------------------------------- index kernels
_NATIVE = None  # False: unavailable


def _native():
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = False
        if os.environ.get("HCU_DEDUP_NATIVE", "1") != "0" and torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load

                src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "dedup_gather_sum.cu")
                _NATIVE = load(name="hcu_dedup_gather_sum", sources=[src], extra_cuda_cflags=["-O3"],
                               verbose=False)
            except Exception as e:  # noqa: BLE001 - any build problem falls back to PyTorch ops
                warnings.warn(f"dedup dispatcher: native index kernels unavailable ({e}); using PyTorch ops")
    return _NATIVE or None


def _gather_sum(src, table, base, mod, out):
    """out[m] = sum over k with v = table[m, k] - base >= 0 of src[v % mod] (mod 0: src[v]).

    table is [M1, M2, K] (any strides), out [M1 * M2, H]; accumulates in fp32 (fp64 for fp64).
    """
    if out.numel() == 0:
        return out
    lib = _native()
    if lib is not None and src.is_cuda and (out.dtype == src.dtype or out.dtype == torch.float32):
        lib.gather_sum(src, table, base, mod, out)
        return out
    M, K = out.shape[0], table.shape[-1]
    v = table.reshape(M, K) - base
    valid = v >= 0
    v = v.clamp(min=0)
    if mod > 0:
        v = v % mod
    acc = torch.float64 if src.dtype == torch.float64 else torch.float32
    rows = src.index_select(0, v.reshape(-1)).view(M, K, -1).to(acc)
    out.copy_((rows * valid.unsqueeze(-1)).sum(1))
    return out


def _gather_elem(src, n0, table, base, out):
    """out[a, b] = src[v] (src[v % n0, v // n0] if n0 > 0) for v = table[a, b] - base >= 0, else 0."""
    if out.numel() == 0:
        return out
    lib = _native()
    if lib is not None and src.is_cuda:
        lib.gather_elem(src, n0, table, base, out)
        return out
    v = table - base
    valid = v >= 0
    v = v.clamp(min=0)
    vals = src[v % n0, v // n0] if n0 > 0 else src[v]
    out.copy_(torch.where(valid, vals, torch.zeros((), dtype=vals.dtype, device=vals.device)))
    return out


def _compact(flat_mask, values):
    """(pos, idx): pos = 1-based slot of each True entry of flat_mask (0 for False), and
    idx[slot - 1] = values[entry]. idx is sized for the upper bound; slots past the True
    count hold 0 and are sliced off once the count is known. No host sync."""
    pos = torch.cumsum(flat_mask, 0, dtype=torch.long)
    pos.mul_(flat_mask)
    idx = torch.zeros(flat_mask.numel() + 1, dtype=torch.long, device=flat_mask.device)
    idx.scatter_(0, pos, values)
    return pos, idx[1:]


class _Layout:
    """Packed all-to-all row: [hidden (H) | pad | meta (L values of meta_dtype, bit-cast)]."""

    def __init__(self, H, L, dtype, meta_dtype):
        es, mes = torch.finfo(dtype).bits // 8, torch.finfo(meta_dtype).bits // 8
        assert mes % es == 0, f"router probs ({meta_dtype}) narrower than activations ({dtype})"
        r = mes // es
        self.H, self.L, self.dtype, self.meta_dtype = H, L, dtype, meta_dtype
        self.meta_start = -(-H // r) * r
        self.width = self.meta_start + L * r

    def new(self, rows, device):
        buf = torch.empty((rows, self.width), dtype=self.dtype, device=device)
        if self.meta_start > self.H:
            buf[:, self.H:self.meta_start].zero_()
        return buf

    def hidden(self, buf):
        return buf[:, :self.H]

    def meta(self, buf):
        return buf[:, self.meta_start:].view(self.meta_dtype)


class _DispatchGather(torch.autograd.Function):
    """packed[s] = [hidden[idx[s] % T] | meta_src[idx[s]]], idx[s] = p*T + t (rank-major).

    Backward: grad_hidden[t] = sum_p g[pos[p, t] - 1]; grad_probs[t, p*L:(p+1)*L] = g_meta[pos[p, t] - 1].
    """

    @staticmethod
    def forward(ctx, hidden, probs, meta_src, idx, pos, P, layout):
        T = hidden.shape[0]
        packed = layout.new(idx.numel(), hidden.device)
        table = idx.view(-1, 1, 1)
        _gather_sum(hidden, table, 0, T, layout.hidden(packed))
        _gather_sum(meta_src, table, 0, 0, layout.meta(packed))
        ctx.save_for_backward(pos)
        ctx.layout, ctx.T, ctx.P, ctx.probs_dtype = layout, T, P, probs.dtype
        return packed

    @staticmethod
    def backward(ctx, g):
        (pos,) = ctx.saved_tensors
        layout, T, P = ctx.layout, ctx.T, ctx.P
        g = g.contiguous()
        pos_pt = pos.view(P, T)
        grad_hidden = grad_probs = None
        if ctx.needs_input_grad[0]:
            grad_hidden = torch.empty((T, layout.H), dtype=g.dtype, device=g.device)
            _gather_sum(layout.hidden(g), pos_pt.t().unsqueeze(1), 1, 0, grad_hidden)
        if ctx.needs_input_grad[1]:
            grad_probs = torch.empty((T, P * layout.L), dtype=ctx.probs_dtype, device=g.device)
            _gather_sum(layout.meta(g), pos_pt.unsqueeze(-1).transpose(0, 1), 1, 0,
                        grad_probs.view(T * P, layout.L))
        return grad_hidden, grad_probs, None, None, None, None, None


class _Expand(torch.autograd.Function):
    """(row, local expert) pairs, expert-major: idx[n] = l*R + r.

    x[n] = recv_hidden[r], probs[n] = recv_meta[r, l]. Backward sums over the experts of each
    row through pos[l, r] and writes the probs gradient into the meta columns.
    """

    @staticmethod
    def forward(ctx, recv, idx, pos, layout):
        R, N = recv.shape[0], idx.numel()
        x = torch.empty((N, layout.H), dtype=recv.dtype, device=recv.device)
        _gather_sum(layout.hidden(recv), idx.view(-1, 1, 1), 0, R, x)
        probs = torch.empty((N, 1), dtype=layout.meta_dtype, device=recv.device)
        _gather_elem(layout.meta(recv), R, idx.view(-1, 1), 0, probs)
        ctx.save_for_backward(pos)
        ctx.layout, ctx.R = layout, R
        return x, probs.view(N)

    @staticmethod
    def backward(ctx, grad_x, grad_probs):
        (pos,) = ctx.saved_tensors
        layout, R = ctx.layout, ctx.R
        pos_rl = pos.view(layout.L, R).t()
        g = layout.new(R, pos.device)
        if grad_x is None:
            layout.hidden(g).zero_()
        else:
            _gather_sum(grad_x.contiguous(), pos_rl.unsqueeze(1), 1, 0, layout.hidden(g))
        if grad_probs is None:
            layout.meta(g).zero_()
        else:
            _gather_elem(grad_probs.to(layout.meta_dtype).contiguous(), 0, pos_rl, 1, layout.meta(g))
        return g, None, None, None


class _SumByPos(torch.autograd.Function):
    """out[j] = sum_k x[table[j, k] - 1] (fp32 accumulate); backward: grad_x[i] = g[idx[i] % mod]."""

    @staticmethod
    def forward(ctx, x, table, idx, mod, num_out, out_dtype):
        out = torch.empty((num_out, x.shape[-1]), dtype=out_dtype, device=x.device)
        _gather_sum(x.contiguous(), table, 1, 0, out)
        ctx.save_for_backward(idx)
        ctx.mod, ctx.in_dtype = mod, x.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        g = g.contiguous()
        grad = torch.empty((idx.numel(), g.shape[-1]), dtype=ctx.in_dtype, device=g.device)
        if g.dtype == grad.dtype or grad.dtype == torch.float32:
            _gather_sum(g, idx.view(-1, 1, 1), 0, ctx.mod, grad)
        else:  # e.g. fp32 reduce-scatter gradient -> bf16 expert output (ETP > 1)
            grad.copy_(_gather_sum(g, idx.view(-1, 1, 1), 0, ctx.mod, torch.empty_like(grad, dtype=g.dtype)))
        return grad, None, None, None, None, None


class _Split(torch.autograd.Function):
    """packed -> (hidden, meta) dense copies; backward re-packs. Only for ETP > 1, where the
    expert-TP gather/reduce-scatter must see the meta in its own dtype."""

    @staticmethod
    def forward(ctx, packed, layout):
        ctx.layout = layout
        return layout.hidden(packed).contiguous(), layout.meta(packed).contiguous()

    @staticmethod
    def backward(ctx, gh, gm):
        layout = ctx.layout
        ref = gh if gh is not None else gm
        g = layout.new(ref.shape[0], ref.device)
        if gh is None:
            layout.hidden(g).zero_()
        else:
            layout.hidden(g).copy_(gh)
        if gm is None:
            layout.meta(g).zero_()
        else:
            layout.meta(g).copy_(gm)
        return g, None


class _Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, meta, layout):
        ctx.layout = layout
        packed = layout.new(hidden.shape[0], hidden.device)
        layout.hidden(packed).copy_(hidden)
        layout.meta(packed).copy_(meta)
        return packed

    @staticmethod
    def backward(ctx, g):
        layout = ctx.layout
        return layout.hidden(g).contiguous(), layout.meta(g).contiguous(), None


class MoEDedupAlltoAllTokenDispatcher(MoETokenDispatcher):
    """Drop-in replacement for MoEAlltoAllTokenDispatcher that deduplicates rows per rank."""

    # Rows actually sent vs. rows the stock dispatcher would send, summed over all layers and
    # calls on this rank since the last reset. Read by tests and the one-time log below.
    stats_rows = 0
    stats_rows_without_dedup = 0
    _logged = False

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        self.local_expert_indices = local_expert_indices

        # Per-layer scratch buffers and constants, reused to save host-side allocations. Only
        # tensors that are consumed within the same call live here: anything autograd saves for
        # backward (index maps) is freshly allocated, so several microbatches in flight
        # (pipeline parallelism) never share it.
        self._scratch = {}
        assert self.num_experts == self.ep_size * num_local_experts, "experts must divide evenly over EP"
        assert not config.moe_pad_expert_input_to_capacity, (
            "deduplicating dispatcher does not support --moe-pad-expert-input-to-capacity"
        )
        assert config.cuda_graph_impl == "none", "deduplicating dispatcher does not support CUDA graphs"
        _native()  # build/load the index kernels at model construction, not inside the first step

    # ------------------------------------------------------------------ helpers
    def _scratch_tensor(self, name, shape, *, device, dtype, pin_memory=False) -> torch.Tensor:
        """Return a reusable scratch tensor for a fixed shape/device/dtype."""
        key = (name, tuple(shape), str(device), dtype, bool(pin_memory))
        tensor = self._scratch.get(key)
        if tensor is None:
            tensor = torch.empty(shape, device=device, dtype=dtype, pin_memory=pin_memory)
            self._scratch[key] = tensor
        return tensor

    def _neg_inf(self, dtype, device) -> torch.Tensor:
        key = ("neg_inf", str(device), dtype)
        t = self._scratch.get(key)
        if t is None:
            t = torch.full((), _UNUSED, dtype=dtype, device=device)
            self._scratch[key] = t
        return t

    _arange_cache = {}

    @classmethod
    def _arange(cls, n, device) -> torch.Tensor:
        """A view of arange(n) from a cached buffer that grows on demand."""
        buf = cls._arange_cache.get(str(device))
        if buf is None or buf.numel() < n:
            buf = torch.arange(max(n, 2 * (0 if buf is None else buf.numel())), device=device)
            cls._arange_cache[str(device)] = buf
        return buf[:n]

    # ------------------------------------------------------------------ counts
    def _exchange_counts(self, counts: torch.Tensor):
        """Exchange counts[0] = per-destination [P, 1+L] counts; returns (host buffer, event).

        Column 0 is the number of (deduplicated) rows, columns 1.. the number of
        (token, local expert) assignments. The received counts land in counts[1:] (expert-TP
        peers in order): one small EP all-to-all and, with ETP > 1, one small all-gather. The
        device-to-host copy runs on a side stream so the caller can queue GPU work before
        waiting on it (as the stock dispatcher does).
        """
        send = counts[0]
        if self.tp_size > 1:
            recv = self._scratch_tensor("recv_counts", send.shape, device=send.device, dtype=send.dtype)
        else:
            recv = counts[1]
        if self.ep_size > 1:
            dist.all_to_all_single(recv, send, group=self.ep_group)
        else:
            recv.copy_(send)
        if self.tp_size > 1:
            dist.all_gather_into_tensor(counts[1:], recv, group=self.tp_group)

        stream = type(self)._dtoh_stream()
        host = self._scratch_tensor(
            "host_counts", counts.shape, device=torch.device("cpu"), dtype=counts.dtype, pin_memory=True
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            host.copy_(counts, non_blocking=True)
            event = stream.record_event()
        return host, event

    def _read_counts(self, host: torch.Tensor, event) -> None:
        event.synchronize()
        counts = host.numpy()  # numpy: cheaper than torch ops on small CPU tensors
        sent, recv = counts[0], counts[1:]
        self.input_splits = sent[:, 0].tolist()
        self.output_splits = recv[self.tp_rank, :, 0].tolist()
        self.output_splits_tp = recv[:, :, 0].sum(1).tolist()  # rows each expert-TP peer holds
        # [L] CPU tensor, as the experts expect; a new array, so the host buffer can be reused.
        self.tokens_per_expert = torch.from_numpy(recv[:, :, 1:].sum((0, 1)))
        self.num_assignments_sent = int(sent[:, 1:].sum())

    _stream = None

    @classmethod
    def _dtoh_stream(cls):
        if cls._stream is None:
            cls._stream = torch.cuda.Stream()
        return cls._stream

    # ------------------------------------------------------------------ dispatch
    def dispatch_preprocess(self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor):
        """Returns (packed rows, None): the probs travel inside the packed buffer."""
        assert routing_map.dtype == torch.bool and routing_map.dim() == 2 and probs.dim() == 2
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        if self.config.moe_router_padding_for_quantization:
            # Padded entries have probability 0: they only round expert token counts up.
            routing_map = pad_routing_map(routing_map, get_align_size_for_quantization(self.config))
        self.routing_map = routing_map
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        T, P, L = routing_map.shape[0], self.ep_size, self.num_local_experts
        device = routing_map.device
        self.layout = _Layout(hidden_states.shape[-1], L, hidden_states.dtype, probs.dtype)
        rm = routing_map.view(T, P, L)
        # [P, T]: token t needs rank p, rank-major so that flattening needs no copy.
        rank_map = self._scratch_tensor("rank_map", (P, T), device=device, dtype=torch.bool)
        torch.any(rm.transpose(0, 1), dim=-1, out=rank_map)
        counts = self._scratch_tensor("counts", (1 + self.tp_size, P, 1 + L), device=device, dtype=torch.long)
        torch.sum(rank_map, dim=1, out=counts[0, :, 0])
        torch.sum(rm, dim=0, out=counts[0, :, 1:])
        host_counts, counts_ready = self._exchange_counts(counts)

        # Queued before waiting for the counts, so the GPU works during the device-to-host copy.
        # Rows go out rank-major, tokens ascending within a rank: idx[s] = p*T + t.
        self.dispatch_pos, idx = _compact(rank_map.view(-1), self._arange(T * P, device))
        meta_src = self._scratch_tensor("meta_src", (P, T, L), device=device, dtype=probs.dtype)
        with torch.no_grad():  # differentiated by _DispatchGather
            torch.where(rm.transpose(0, 1), probs.view(T, P, L).transpose(0, 1),
                        self._neg_inf(probs.dtype, device), out=meta_src)

        self._read_counts(host_counts, counts_ready)
        num_rows = sum(self.input_splits)
        self.dispatch_idx = idx[:num_rows]

        cls = type(self)
        cls.stats_rows += num_rows
        cls.stats_rows_without_dedup += self.num_assignments_sent
        if not cls._logged and (not dist.is_initialized() or dist.get_rank() == 0):
            cls._logged = True
            base = self.num_assignments_sent
            print(f"[dedup alltoall] first layer on rank 0: {num_rows} rows sent instead of {base} "
                  f"({100.0 * (1 - num_rows / max(base, 1)):.1f}% removed)", flush=True)

        packed = _DispatchGather.apply(hidden_states, probs, meta_src.view(P * T, L), self.dispatch_idx,
                                       self.dispatch_pos, P, self.layout)
        self.num_tokens = T
        return packed, None

    def token_dispatch(self, packed: torch.Tensor, _probs=None):
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        recv = all_to_all(
            self.ep_group, packed, self.output_splits, self.input_splits, use_nccl_stream=self.use_nccl_stream
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(recv)
        return recv, None

    def dispatch_postprocess(self, recv: torch.Tensor, _probs=None):
        layout = self.layout
        if self.tp_size > 1:
            # Every expert-TP shard computes all rows with its slice of the expert weights. Gather
            # hidden and meta separately: the backward reduce-scatter must sum meta in its dtype.
            hidden, meta = _Split.apply(recv, layout)
            hidden = gather_from_sequence_parallel_region(
                hidden, group=self.tp_group, output_split_sizes=self.output_splits_tp
            )
            meta = gather_from_sequence_parallel_region(
                meta, group=self.tp_group, output_split_sizes=self.output_splits_tp
            )
            recv = _Join.apply(hidden, meta, layout)
        L, R = self.num_local_experts, recv.shape[0]
        device = recv.device
        num_out = int(self.tokens_per_expert.sum())  # CPU tensor: no sync
        # (row, local expert) pairs, expert-major with rows ascending (the layout grouped GEMM
        # expects): compact the flattened [L, R] mask, idx[n] = l*R + r.
        mask = torch.empty((L, R), device=device, dtype=torch.bool)  # R varies: not a scratch buffer
        torch.gt(layout.meta(recv).t(), _UNUSED, out=mask)
        self.expand_pos, idx = _compact(mask.view(-1), self._arange(L * R, device))
        self.expand_idx = idx[:num_out]
        self.num_recv_rows = R
        expert_input, expert_probs = _Expand.apply(recv, self.expand_idx, self.expand_pos, layout)
        tokens_per_expert, self.tokens_per_expert = self.tokens_per_expert, None
        return expert_input, tokens_per_expert, expert_probs

    # ------------------------------------------------------------------- combine
    def combine_preprocess(self, expert_output: torch.Tensor):
        # Experts already scaled by the routing probs; one partial sum per received row (fp32
        # accumulate), out[r] = sum_l y[pos[l, r] - 1].
        R, L = self.num_recv_rows, self.num_local_experts
        table = self.expand_pos.view(L, R).t().unsqueeze(1)
        if self.tp_size > 1:
            # Reduce the expert-TP partial sums in fp32, then return to the activation dtype.
            out = _SumByPos.apply(expert_output, table, self.expand_idx, R, R, torch.float32)
            out = reduce_scatter_to_sequence_parallel_region(
                out, group=self.tp_group, input_split_sizes=self.output_splits_tp
            )
            return out.to(expert_output.dtype)
        return _SumByPos.apply(expert_output, table, self.expand_idx, R, R, expert_output.dtype)

    def token_combine(self, hidden_states: torch.Tensor, async_finish: bool = True, allocate_on_comm_stream: bool = True):
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        rows = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits, use_nccl_stream=self.use_nccl_stream
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(rows)
            self.shared_experts.post_forward_comm()
        return rows

    def combine_postprocess(self, rows: torch.Tensor):
        # Sum each token's partial outputs over its destination ranks: out[t] = sum_p rows[pos[p, t] - 1].
        P, T = self.ep_size, self.num_tokens
        table = self.dispatch_pos.view(P, T).t().unsqueeze(1)
        out = _SumByPos.apply(rows, table, self.dispatch_idx, T, T, rows.dtype)
        out = out.view(self.hidden_shape)
        if self.shared_experts is not None:
            out = out + self.shared_experts.get_output()
        return out
