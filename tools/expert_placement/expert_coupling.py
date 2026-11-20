"""Core algorithmic demo for expert coupling; independent of Megatron internals."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import torch
import torch.nn.functional as F


def to_routing_map(routes: torch.Tensor, num_experts: Optional[int] = None) -> torch.Tensor:
    """Return a [tokens, experts] bool routing map.

    Accepts a bool/0-1 map [tokens, experts] or integer expert IDs
    [tokens, topk]. Pass ``num_experts`` whenever it is known: a non-bool input
    whose width differs from ``num_experts`` is always read as IDs, so batches
    that miss the highest expert or contain only IDs 0/1 are handled correctly.
    Without ``num_experts``, a 0/1-valued input is read as a map.
    """
    if routes.ndim != 2:
        raise ValueError(f"routes must be rank-2, got {tuple(routes.shape)}")
    if routes.dtype == torch.bool:
        out = routes
    elif (num_experts is None or num_experts == routes.shape[1]) and bool(torch.all((routes == 0) | (routes == 1))):
        out = routes.bool()
    else:
        if routes.is_floating_point():
            raise ValueError("expert IDs must be an integer tensor")
        ids = routes.long()
        if ids.numel() and int(ids.min()) < 0:
            raise ValueError("expert IDs must be non-negative")
        if num_experts is None:
            num_experts = int(ids.max()) + 1 if ids.numel() else 0
        if ids.numel() and int(ids.max()) >= num_experts:
            raise ValueError(f"expert ID {int(ids.max())} out of range for {num_experts} experts")
        out = torch.zeros((ids.shape[0], num_experts), dtype=torch.bool, device=ids.device)
        out.scatter_(1, ids, True)
    if num_experts is not None and out.shape[1] != num_experts:
        raise ValueError(f"expected {num_experts} experts, got {out.shape[1]}")
    return out


@dataclass
class PlacementResult:
    """Expert-to-EP-rank assignment for one MoE layer.

    ``objective_*`` is the co-selection weight kept on the same rank
    (sum over expert pairs sharing a rank); ``before`` is contiguous placement.
    """
    expert_to_rank: List[int]
    objective_before: float
    objective_after: float
    loads: List[int]
    num_nodes: int = 1
    num_swaps: int = 0

    def rank_to_experts(self) -> List[List[int]]:
        """Global expert IDs hosted by each rank, ascending (= local slot order)."""
        out: List[List[int]] = [[] for _ in self.loads]
        for e, r in enumerate(self.expert_to_rank):
            out[r].append(e)
        return out

    def to_dict(self) -> Dict:
        return {
            "expert_to_rank": self.expert_to_rank,
            "rank_to_experts": self.rank_to_experts(),
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "loads": self.loads,
            "num_nodes": self.num_nodes,
            "num_swaps": self.num_swaps,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ExpertCouplingStats:
    """Accumulate within-layer expert co-selection counts (one instance per layer)."""
    def __init__(self, num_experts: int, device: str = "cpu"):
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.co_select = torch.zeros((num_experts, num_experts), dtype=torch.float64, device=device)
        self.load = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.num_tokens = 0

    @torch.no_grad()
    def update(self, routes: torch.Tensor):
        routes = to_routing_map(routes, self.num_experts).to(self.co_select.device)
        # 0/1 products are exact in fp32 per batch; accumulate across batches in fp64.
        x = routes.to(torch.float32)
        self.co_select += (x.T @ x).double()
        self.load += x.sum(0).double()
        self.num_tokens += x.shape[0]

    def normalized(self, eps: float = 1e-12) -> torch.Tensor:
        # P(i and j selected) normalized by marginal probabilities. Diagonal
        # is retained for diagnostics but placement ignores it.
        denom = self.load[:, None] * self.load[None, :]
        return self.co_select / denom.clamp_min(eps) * (self.num_tokens ** 2)

    def save(self, path: str):
        torch.save({
            "num_experts": self.num_experts,
            "co_select": self.co_select.cpu(),
            "load": self.load.cpu(),
            "num_tokens": self.num_tokens,
        }, path)

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(d["num_experts"])
        obj.co_select = d["co_select"].double()
        obj.load = d["load"].double()
        obj.num_tokens = int(d["num_tokens"])
        return obj


def _balanced_caps(n: int, num_groups: int) -> List[int]:
    return [n // num_groups + (1 if g < n % num_groups else 0) for g in range(num_groups)]


def contiguous_placement(num_experts: int, num_ranks: int) -> List[int]:
    """Megatron default: rank r hosts experts [r*E/P, (r+1)*E/P)."""
    if not (1 <= num_ranks <= num_experts):
        raise ValueError("num_ranks must be in [1, num_experts]")
    out: List[int] = []
    for r, cap in enumerate(_balanced_caps(num_experts, num_ranks)):
        out += [r] * cap
    return out


def _prepare_affinity(co_select: torch.Tensor) -> torch.Tensor:
    w = co_select.detach().to("cpu", dtype=torch.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("co_select must be square")
    w = (w + w.T) / 2
    w.fill_diagonal_(0)
    return w


def placement_objective(co_select: torch.Tensor, expert_to_rank: Sequence[int]) -> float:
    """Co-selection weight of expert pairs that share a rank (each pair once)."""
    w = _prepare_affinity(co_select)
    a = torch.as_tensor(list(expert_to_rank), dtype=torch.long)
    if a.numel() != w.shape[0]:
        raise ValueError("expert count does not match placement")
    same = a[:, None] == a[None, :]
    return float((w * same).sum() / 2)


def _greedy_init(w: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Balanced greedy seed: experts in decreasing total affinity join the
    non-full group they are most attached to (ties: lower load, lower id)."""
    n = w.shape[0]
    caps = _balanced_caps(n, num_groups)
    loads = [0] * num_groups
    aff = torch.zeros((n, num_groups), dtype=w.dtype)
    assign = torch.full((n,), -1, dtype=torch.long)
    for e in sorted(range(n), key=lambda i: (-float(w[i].sum()), i)):
        g = max((g for g in range(num_groups) if loads[g] < caps[g]),
                key=lambda g: (float(aff[e, g]), -loads[g], -g))
        assign[e] = g
        loads[g] += 1
        aff[:, g] += w[:, e]
    return assign


def _swap_refine(
    w: torch.Tensor,
    assign: torch.Tensor,
    num_groups: int,
    max_swaps: int,
    load: Optional[torch.Tensor] = None,
    max_load: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Alg. 1 refinement: apply the best improving pairwise swap until none
    improves or ``max_swaps`` is reached. Swaps keep group sizes fixed.

    m[i, g] = affinity of expert i to group g. Swapping a (group A) with
    b (group B) changes the objective by
        m[a,B] + m[b,A] - m[a,A] - m[b,B] - 2 w[a,b],
    evaluated for all pairs at once in O(E^2) per swap.

    With ``load`` (tokens routed to each expert) and ``max_load``, a swap is only
    allowed if both groups end at most at ``max_load`` tokens, or if it does not raise
    the larger of the two. Co-locating correlated experts otherwise tends to stack hot
    experts on one rank, and dispatch, expert compute and combine wait for the busiest rank.
    """
    assign = assign.clone()
    m = w @ F.one_hot(assign, num_groups).to(w.dtype)
    eps = 1e-9 * max(1.0, float(w.abs().max()) if w.numel() else 1.0)
    group_load = None
    if load is not None and max_load is not None:
        load = load.to(w.dtype)
        group_load = torch.zeros(num_groups, dtype=w.dtype).index_add_(0, assign, load)
    swaps = 0
    while swaps < max_swaps:
        own = m.gather(1, assign[:, None]).squeeze(1)
        to_other = m[:, assign]  # to_other[a, b] = m[a, group(b)]
        gain = to_other + to_other.T - own[:, None] - own[None, :] - 2 * w
        gain.masked_fill_(assign[:, None] == assign[None, :], float("-inf"))
        if group_load is not None:
            la = group_load[assign]                          # load of each expert's group
            new_a = la[:, None] - load[:, None] + load[None, :]
            new_b = la[None, :] - load[None, :] + load[:, None]
            worst_new = torch.maximum(new_a, new_b)
            worst_old = torch.maximum(la[:, None], la[None, :])
            gain.masked_fill_((worst_new > max_load) & (worst_new > worst_old), float("-inf"))
        idx = int(torch.argmax(gain))
        a, b = divmod(idx, w.shape[0])
        if not float(gain[a, b]) > eps:
            break
        ga, gb = int(assign[a]), int(assign[b])
        delta = w[:, a] - w[:, b]
        m[:, ga] -= delta
        m[:, gb] += delta
        if group_load is not None:
            group_load[ga] += load[b] - load[a]
            group_load[gb] += load[a] - load[b]
        assign[a], assign[b] = gb, ga
        swaps += 1
    return assign, swaps


def _partition(
    w: torch.Tensor,
    num_groups: int,
    max_swaps: int,
    init: str,
    load: Optional[torch.Tensor] = None,
    max_load_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    if init == "contiguous":
        assign = torch.tensor(contiguous_placement(w.shape[0], num_groups), dtype=torch.long)
    elif init == "greedy":
        assign = _greedy_init(w, num_groups)
    else:
        raise ValueError("init must be 'contiguous' or 'greedy'")
    if num_groups == 1:
        return assign, 0
    max_load = None
    if load is not None and max_load_ratio is not None:
        max_load = float(max_load_ratio) * float(load.sum()) / num_groups
    return _swap_refine(w, assign, num_groups, max_swaps, load, max_load)


def correlated_placement(
    co_select: torch.Tensor,
    num_ranks: int,
    *,
    num_nodes: int = 1,
    max_swaps: int = 1000,
    init: str = "contiguous",
    expert_load: Optional[torch.Tensor] = None,
    max_load_ratio: Optional[float] = None,
    router_groups: int = 1,
) -> PlacementResult:
    """Correlated expert placement (paper Alg. 1) for one layer.

    Starts from contiguous placement (or a greedy seed) and applies the best
    improving pairwise swap until none remains, so every rank keeps exactly
    E/P experts. With ``num_nodes > 1`` the partition is hierarchical: experts
    are first split across nodes (this step sets inter-node traffic, which is what
    two-level dispatchers such as DeepEP pay for), then across the ranks of each node.
    EP ranks [n*P/N, (n+1)*P/N) are assumed to live on node n. ``max_swaps`` applies
    to each partition step separately.

    ``expert_load`` (tokens per expert) with ``max_load_ratio`` caps every node and
    rank at ``max_load_ratio`` x the mean token load (see ``_swap_refine``).

    ``router_groups > 1`` handles group-limited routing (DeepSeek-V3 style device-/
    node-limited routing): experts are only moved within the EP ranks their router
    group occupies under contiguous placement, so the routing limit keeps bounding the
    devices/nodes a token reaches. If a group is smaller than a rank, no move is
    possible and the contiguous placement is returned.

    Note: single-swap search can stop at a zero-gain local optimum; try
    ``init="greedy"`` and keep whichever result has the larger objective.
    """
    w = _prepare_affinity(co_select)
    n = w.shape[0]
    if not (1 <= num_ranks <= n):
        raise ValueError("num_ranks must be in [1, num_experts]")
    if num_nodes < 1 or num_ranks % num_nodes:
        raise ValueError("num_ranks must be a multiple of num_nodes")
    if num_nodes > 1 and n % num_ranks:
        raise ValueError("hierarchical placement requires num_experts % num_ranks == 0")
    load = None if expert_load is None else torch.as_tensor(expert_load, dtype=torch.float64).cpu()

    def place(idx: torch.Tensor, ranks: int, nodes: int) -> Tuple[torch.Tensor, int]:
        """Partition experts ``idx`` over ``ranks`` local ranks (``nodes`` local nodes)."""
        sub_w = w[idx][:, idx]
        sub_load = None if load is None else load[idx]
        if nodes <= 1:
            return _partition(sub_w, ranks, max_swaps, init, sub_load, max_load_ratio)
        node_of, swaps = _partition(sub_w, nodes, max_swaps, init, sub_load, max_load_ratio)
        per_node = ranks // nodes
        out = torch.empty(idx.numel(), dtype=torch.long)
        for node in range(nodes):
            sel = (node_of == node).nonzero().flatten()
            local, s = _partition(sub_w[sel][:, sel], per_node, max_swaps, init,
                                  None if sub_load is None else sub_load[sel], max_load_ratio)
            out[sel] = node * per_node + local
            swaps += s
        return out, swaps

    if router_groups <= 1:
        e2r, swaps = place(torch.arange(n), num_ranks, num_nodes)
    else:
        if n % router_groups:
            raise ValueError("num_experts must be a multiple of router_groups")
        per_group, per_rank = n // router_groups, n // num_ranks
        e2r = torch.tensor(contiguous_placement(n, num_ranks), dtype=torch.long)
        swaps = 0
        if per_group > per_rank:
            if per_group % per_rank:
                raise ValueError("router groups must cover whole EP ranks")
            ranks_per_group = per_group // per_rank
            ranks_per_node = num_ranks // num_nodes
            # A group spanning several nodes keeps the node split; within a node it is flat.
            nodes_per_group = max(1, ranks_per_group // ranks_per_node)
            for g in range(router_groups):
                idx = torch.arange(g * per_group, (g + 1) * per_group)
                local, s = place(idx, ranks_per_group, nodes_per_group)
                e2r[idx] = g * ranks_per_group + local
                swaps += s

    expert_to_rank = e2r.tolist()
    return PlacementResult(
        expert_to_rank=expert_to_rank,
        objective_before=placement_objective(w, contiguous_placement(n, num_ranks)),
        objective_after=placement_objective(w, expert_to_rank),
        loads=torch.bincount(e2r, minlength=num_ranks).tolist(),
        num_nodes=num_nodes,
        num_swaps=swaps,
    )


def layerwise_placement(
    stats: Sequence[Union[ExpertCouplingStats, torch.Tensor]],
    num_ranks: int,
    **kwargs,
) -> List[PlacementResult]:
    """Independent placement per MoE layer (each layer has its own permutation)."""
    out = []
    for s in stats:
        kw = dict(kwargs)
        if isinstance(s, ExpertCouplingStats):
            if kw.get("max_load_ratio") is not None:
                kw.setdefault("expert_load", s.load)
            s = s.co_select
        out.append(correlated_placement(s, num_ranks, **kw))
    return out


def save_layer_placements(path: str, placements: Sequence[PlacementResult], num_ranks: int):
    payload = {
        "num_ranks": num_ranks,
        "num_nodes": placements[0].num_nodes if placements else 1,
        "num_experts": len(placements[0].expert_to_rank) if placements else 0,
        "layers": [p.to_dict() for p in placements],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def placement_metrics(
    routes: torch.Tensor,
    expert_to_rank: Sequence[int],
    num_ranks: int,
    *,
    num_nodes: int = 1,
    home_rank: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Table I metrics for one layer under a given placement.

    same_gpu_pct      tokens whose selected experts all sit on one rank
    gpus_per_token    distinct destination ranks per token
    dedup_pct         dispatched rows removed by a deduplicating dispatcher
                      (one row per (token, rank) instead of per (token, expert))
    nodes_per_token   distinct destination nodes per token (num_nodes > 1)
    With ``home_rank`` [tokens] (the rank currently holding each token):
    local_slot_pct    share of (token, expert) pairs served on the home rank
    wire_rows(_dedup) rows leaving the home rank without / with dedup
    """
    e2r = torch.as_tensor(list(expert_to_rank), dtype=torch.long)
    x = to_routing_map(routes, e2r.numel()).cpu()
    if e2r.numel() and (int(e2r.min()) < 0 or int(e2r.max()) >= num_ranks):
        raise ValueError("placement refers to a rank outside [0, num_ranks)")
    if num_nodes < 1 or num_ranks % num_nodes:
        raise ValueError("num_ranks must be a multiple of num_nodes")
    slots = x.float() @ F.one_hot(e2r, num_ranks).float()  # [tokens, ranks]: experts used per rank
    hit = slots > 0
    n_tokens = max(1, x.shape[0])
    rows = float(slots.sum())
    rows_dedup = float(hit.sum())
    out = {
        "same_gpu_pct": 100.0 * float((hit.sum(1) == 1).sum()) / n_tokens,
        "gpus_per_token": rows_dedup / n_tokens,
        "dedup_pct": 100.0 * (1.0 - rows_dedup / rows) if rows else 0.0,
        "rows": rows,
        "rows_dedup": rows_dedup,
        # Busiest rank / mean: rows received (deduplicated) and expert-compute rows.
        "recv_imbalance": float(hit.sum(0).max()) / max(rows_dedup / num_ranks, 1e-9),
        "compute_imbalance": float(slots.sum(0).max()) / max(rows / num_ranks, 1e-9),
    }
    if num_nodes > 1:
        node_of_rank = torch.arange(num_ranks) // (num_ranks // num_nodes)
        node_hit = (slots @ F.one_hot(node_of_rank, num_nodes).float()) > 0
        out["nodes_per_token"] = float(node_hit.sum()) / n_tokens
    if home_rank is not None:
        home = torch.as_tensor(home_rank, dtype=torch.long).cpu()
        if home.shape != (x.shape[0],):
            raise ValueError("home_rank must have shape [tokens]")
        local = float(slots.gather(1, home[:, None]).sum())
        local_dedup = float(hit.gather(1, home[:, None]).sum())
        out["local_slot_pct"] = 100.0 * local / rows if rows else 0.0
        out["wire_rows"] = rows - local
        out["wire_rows_dedup"] = rows_dedup - local_dedup
    return out


class NextRankPredictor:
    """Empirical cross-layer predictor: previous-layer expert IDs -> next rank.

    Fit on rows containing previous-layer top-k expert IDs and the actual
    next-layer desired rank. At inference, predicts the most frequent rank for
    the observed expert tuple. Falls back to a smoothed single-expert lookup,
    then to the caller-provided default rank.
    """
    def __init__(self):
        self.combo_counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self.single_counts: Dict[int, Dict[int, int]] = {}

    @staticmethod
    def _topk_tuple(expert_ids: Iterable[int]) -> Tuple[int, ...]:
        return tuple(sorted(int(x) for x in expert_ids))

    def update(self, prev_expert_ids: Iterable[int], next_rank: int):
        key = self._topk_tuple(prev_expert_ids)
        self.combo_counts.setdefault(key, {}).setdefault(int(next_rank), 0)
        self.combo_counts[key][int(next_rank)] += 1
        for e in key:
            self.single_counts.setdefault(e, {}).setdefault(int(next_rank), 0)
            self.single_counts[e][int(next_rank)] += 1

    @staticmethod
    def _best(counts: Optional[Dict[int, int]]):
        if not counts:
            return None, 0.0
        rank, n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        total = sum(counts.values())
        return rank, n / total

    def predict(self, prev_expert_ids: Iterable[int], default_rank: int = 0):
        key = self._topk_tuple(prev_expert_ids)
        rank, conf = self._best(self.combo_counts.get(key))
        if rank is not None:
            return rank, conf, "combo"
        candidates = [self._best(self.single_counts.get(e)) for e in key]
        candidates = [(r, c) for r, c in candidates if r is not None]
        if candidates:
            return max(candidates, key=lambda rc: rc[1])[0], max(candidates, key=lambda rc: rc[1])[1], "single"
        return int(default_rank), 0.0, "fallback"

    def save(self, path: str):
        payload = {
            "combo_counts": [{"key": list(k), "counts": v} for k, v in self.combo_counts.items()],
            "single_counts": {str(k): v for k, v in self.single_counts.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        obj = cls()
        for item in p["combo_counts"]:
            obj.combo_counts[tuple(item["key"])] = {int(k): int(v) for k, v in item["counts"].items()}
        obj.single_counts = {int(k): {int(r): int(v) for r, v in counts.items()}
                             for k, counts in p["single_counts"].items()}
        return obj


def assign_by_confidence(predictions: Sequence[Tuple[int, float]], num_ranks: int, capacity: int):
    """Greedy token placement under uniform rank capacity.

    predictions: (preferred_rank, confidence) for each token. Returns rank IDs.
    If total capacity is insufficient, raises instead of silently dropping tokens.
    """
    if len(predictions) > num_ranks * capacity:
        raise ValueError("capacity too small for number of tokens")
    result = [-1] * len(predictions)
    loads = [0] * num_ranks
    for idx in sorted(range(len(predictions)), key=lambda i: predictions[i][1], reverse=True):
        pref, _ = predictions[idx]
        if not 0 <= pref < num_ranks:
            pref = 0
        candidates = [r for r in range(num_ranks) if loads[r] < capacity]
        rank = pref if pref in candidates else min(candidates, key=lambda r: (loads[r], r))
        result[idx] = rank
        loads[rank] += 1
    return result
