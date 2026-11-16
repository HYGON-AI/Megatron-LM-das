# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

from __future__ import annotations
import math
import random
import json
from dataclasses import dataclass
from typing import Any, Mapping
from collections import defaultdict, deque
from pathlib import Path


@dataclass(frozen=True)
class OpDurationSpec:
    # Normal-distribution duration config for one operation type.

    mean: float
    variance: float = 0.0
    min_value: float = 0.0

    def sample(self, rng: random.Random) -> float:
        if self.variance < 0:
            raise ValueError(f"variance must be non-negative, got {self.variance}")
        if self.mean < 0:
            raise ValueError(f"mean must be non-negative, got {self.mean}")

        if self.variance == 0:
            return max(self.min_value, float(self.mean))

        value = rng.gauss(float(self.mean), math.sqrt(float(self.variance)))
        return max(float(self.min_value), value)


def op_type_value(op_type: Any) -> str:
    # Return a stable string value for pp_scheduler.OpType or a plain string.

    value = getattr(op_type, "value", op_type)
    return str(value)


class DurationSampler:
    def __init__(
        self,
        specs: Mapping[Any, OpDurationSpec | Mapping[str, float]],
        *,
        default_spec: OpDurationSpec | None = None,
        seed: int | None = None,
    ):
        self.rng = random.Random(seed)
        self.default_spec = default_spec
        self.specs = {
            op_type_value(op_type): self._coerce_spec(spec)
            for op_type, spec in specs.items()
        }

    def sample(self, op_type: Any, fallback_duration: float | None = None) -> float:
        op_value = op_type_value(op_type)
        spec = self.specs.get(op_value, self.default_spec)
        if spec is not None:
            return spec.sample(self.rng)
        if fallback_duration is not None:
            return max(0.0, float(fallback_duration))
        raise KeyError(f"No duration spec for op type {op_value!r}")

    @staticmethod
    def _coerce_spec(spec: OpDurationSpec | Mapping[str, float]) -> OpDurationSpec:
        if isinstance(spec, OpDurationSpec):
            return spec
        return OpDurationSpec(
            mean=float(spec["mean"]),
            variance=float(spec.get("variance", 0.0)),
            min_value=float(spec.get("min_value", 0.0)),
        )


COMPUTE_OP_TYPES = frozenset({"F", "B", "VF", "VB", "GF", "GB"})


@dataclass(frozen=True)
class PlanOp:
    id: str
    rank: int
    index: int
    original_index: int
    op_type: str
    microbatch_id: int | None
    chunk_id: int | None
    base_duration: float
    label: str

    @property
    def is_forward(self) -> bool:
        return self.op_type == "F"

    @property
    def is_backward(self) -> bool:
        return self.op_type == "B"


@dataclass(frozen=True)
class Plan:
    ops_by_rank: dict[int, list[PlanOp]]
    pp_size: int
    num_microbatches: int
    scheduler_name: str
    vpp_size: int = 1
    pipeline_layout: dict[str, int] | None = None

    @property
    def ops(self) -> list[PlanOp]:
        return [
            op
            for rank in sorted(self.ops_by_rank)
            for op in self.ops_by_rank[rank]
        ]

    @classmethod
    def from_scheduler(cls, scheduler: Any, *, include_op_types=None) -> "Plan":
        if not getattr(scheduler, "schedules", None):
            scheduler.generate_schedule()

        allowed = frozenset(include_op_types or COMPUTE_OP_TYPES)
        ops_by_rank: dict[int, list[PlanOp]] = {}
        pp_size = int(getattr(scheduler, "pp_size"))

        for rank in range(pp_size):
            rank_ops: list[PlanOp] = []
            for original_index, raw_op in enumerate(scheduler.schedules.get(rank, [])):
                op_value = op_type_value(getattr(raw_op, "op_type", ""))
                if op_value not in allowed:
                    continue

                rank_index = len(rank_ops)
                microbatch_id = getattr(raw_op, "microbatch_id", None)
                chunk_id = getattr(raw_op, "chunk_id", None)
                op_id = _make_op_id(rank, rank_index, op_value, microbatch_id, chunk_id)
                rank_ops.append(
                    PlanOp(
                        id=op_id,
                        rank=rank,
                        index=rank_index,
                        original_index=original_index,
                        op_type=op_value,
                        microbatch_id=microbatch_id,
                        chunk_id=chunk_id,
                        base_duration=float(getattr(raw_op, "duration", 0.0)),
                        label=str(raw_op),
                    )
                )
            ops_by_rank[rank] = rank_ops

        return cls(
            ops_by_rank=ops_by_rank,
            pp_size=pp_size,
            num_microbatches=int(getattr(scheduler, "num_microbatches")),
            scheduler_name=scheduler.__class__.__name__,
            vpp_size=int(getattr(scheduler, "vpp_size", 1)),
            pipeline_layout=getattr(scheduler, "pipeline_layout", None),
        )


def _make_op_id(
    rank: int,
    index: int,
    op_type: str,
    microbatch_id: int | None,
    chunk_id: int | None,
) -> str:
    mb = "none" if microbatch_id is None else str(microbatch_id)
    chunk = "none" if chunk_id is None else str(chunk_id)
    return f"r{rank}:i{index}:{op_type}:mb{mb}:c{chunk}"


class DependencyGraph:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.dependency_reasons = self._build_dependency_reasons()
        self.dependencies = {
            op_id: set(reasons_by_dep)
            for op_id, reasons_by_dep in self.dependency_reasons.items()
        }
        self.topological_order = self._topological_sort()

    def _build_dependency_reasons(self) -> dict[str, dict[str, set[str]]]:
        deps: dict[str, dict[str, set[str]]] = {op.id: {} for op in self.plan.ops}
        by_identity = {
            self._identity(op.rank, op.microbatch_id, op.chunk_id, op.op_type): op
            for op in self.plan.ops
        }
        by_type_and_batch: dict[tuple[str, int], list[PlanOp]] = defaultdict(list)

        for op in self.plan.ops:
            if op.microbatch_id is not None:
                by_type_and_batch[(op.op_type, self._batch_id(op.microbatch_id))].append(op)

        # Local rank order is always a hard execution constraint.
        for rank_ops in self.plan.ops_by_rank.values():
            for previous, current in zip(rank_ops, rank_ops[1:]):
                self._add_dep(current, previous, deps, "rank_order")

        for op in self.plan.ops:
            if op.microbatch_id is None:
                continue

            if op.op_type == "F":
                self._add_llm_forward_deps(op, deps, by_identity, by_type_and_batch)
            elif op.op_type == "B":
                self._add_llm_backward_deps(op, deps, by_identity, by_type_and_batch)
            elif op.op_type == "GF":
                if self._has_disaggregated_layout():
                    self._add_identity_dep(
                        op,
                        deps,
                        by_identity,
                        self._last_llm_rank(),
                        op.microbatch_id,
                        self.plan.vpp_size - 1,
                        "F",
                        "llm_output_ready",
                    )
                else:
                    self._add_identity_dep(
                        op,
                        deps,
                        by_identity,
                        self._last_llm_rank(),
                        op.microbatch_id,
                        self.plan.vpp_size - 1,
                        "F",
                        "llm_output_ready",
                    )
            elif op.op_type == "GB":
                self._add_identity_dep(
                    op,
                    deps,
                    by_identity,
                    op.rank,
                    op.microbatch_id,
                    self.plan.vpp_size - 1,
                    "GF",
                    "generator_backward_consumes_forward",
                )
            elif op.op_type == "VB":
                if self._has_disaggregated_layout():
                    self._add_identity_dep(
                        op,
                        deps,
                        by_identity,
                        self._first_llm_rank(),
                        op.microbatch_id,
                        0,
                        "B",
                        "llm_input_gradient_ready",
                    )
                else:
                    self._add_identity_dep(
                        op,
                        deps,
                        by_identity,
                        self._first_llm_rank(),
                        op.microbatch_id,
                        0,
                        "B",
                        "llm_input_gradient_ready",
                    )

        return deps

    def _add_llm_forward_deps(
        self,
        op: PlanOp,
        deps: dict[str, dict[str, set[str]]],
        by_identity: dict[tuple[int, int | None, int | None, str], PlanOp],
        by_type_and_batch: dict[tuple[str, int], list[PlanOp]],
    ) -> None:
        chunk = self._chunk(op)
        first_llm_rank = self._first_llm_rank()
        last_llm_rank = self._last_llm_rank()

        if op.rank > first_llm_rank:
            self._add_identity_dep(
                op,
                deps,
                by_identity,
                op.rank - 1,
                op.microbatch_id,
                chunk,
                "F",
                "llm_forward_from_previous_rank",
            )
        elif chunk > 0:
            self._add_identity_dep(
                op,
                deps,
                by_identity,
                last_llm_rank,
                op.microbatch_id,
                chunk - 1,
                "F",
                "vpp_forward_from_previous_chunk",
            )

        if op.rank == first_llm_rank and chunk == 0:
            if self._has_disaggregated_layout():
                self._add_identity_dep(
                    op,
                    deps,
                    by_identity,
                    self._encoder_stage(),
                    op.microbatch_id,
                    0,
                    "VF",
                    "encoder_forward_ready",
                )
            else:
                self._add_identity_dep(
                    op,
                    deps,
                    by_identity,
                    self._ve_rank(op.microbatch_id),
                    op.microbatch_id,
                    0,
                    "VF",
                    "ve_forward_ready",
                )

    def _add_llm_backward_deps(
        self,
        op: PlanOp,
        deps: dict[str, dict[str, set[str]]],
        by_identity: dict[tuple[int, int | None, int | None, str], PlanOp],
        by_type_and_batch: dict[tuple[str, int], list[PlanOp]],
    ) -> None:
        chunk = self._chunk(op)
        first_llm_rank = self._first_llm_rank()
        last_llm_rank = self._last_llm_rank()
        self._add_identity_dep(
            op,
            deps,
            by_identity,
            op.rank,
            op.microbatch_id,
            chunk,
            "F",
            "backward_consumes_forward_activation",
        )

        if op.rank < last_llm_rank:
            self._add_identity_dep(
                op,
                deps,
                by_identity,
                op.rank + 1,
                op.microbatch_id,
                chunk,
                "B",
                "llm_backward_from_next_rank",
            )
        elif chunk + 1 < self.plan.vpp_size:
            self._add_identity_dep(
                op,
                deps,
                by_identity,
                first_llm_rank,
                op.microbatch_id,
                chunk + 1,
                "B",
                "vpp_backward_from_next_chunk",
            )

        if op.rank == last_llm_rank and chunk == self.plan.vpp_size - 1:
            if self._has_disaggregated_layout() and self._has_generator():
                self._add_identity_dep(
                    op,
                    deps,
                    by_identity,
                    self._generator_stage(),
                    op.microbatch_id,
                    0,
                    "GB",
                    "generator_backward_ready",
                )
            else:
                self._add_identity_dep(
                    op,
                    deps,
                    by_identity,
                    self._ve_rank(op.microbatch_id),
                    op.microbatch_id,
                    0,
                    "GB",
                    "generator_backward_ready",
                )

    def _add_batch_deps(
        self,
        op: PlanOp,
        deps: dict[str, dict[str, set[str]]],
        by_type_and_batch: dict[tuple[str, int], list[PlanOp]],
        dep_type: str,
        *,
        rank: int | None = None,
        chunk: int | None = None,
        final_forward: bool = False,
        reason: str,
    ) -> None:
        batch_id = self._batch_id(op.microbatch_id)
        for candidate in by_type_and_batch.get((dep_type, batch_id), []):
            if rank is not None and candidate.rank != rank:
                continue
            if chunk is not None and self._chunk(candidate) != chunk:
                continue
            if final_forward and not (
                candidate.rank == self.plan.pp_size - 1
                and self._chunk(candidate) == self.plan.vpp_size - 1
            ):
                continue
            if candidate.id != op.id:
                self._add_dep(op, candidate, deps, reason)

    @staticmethod
    def _add_identity_dep(
        op: PlanOp,
        deps: dict[str, dict[str, set[str]]],
        by_identity: dict[tuple[int, int | None, int | None, str], PlanOp],
        rank: int,
        microbatch_id: int | None,
        chunk_id: int | None,
        op_type: str,
        reason: str,
    ) -> None:
        dependency = by_identity.get((rank, microbatch_id, chunk_id, op_type))
        if dependency is not None and dependency.id != op.id:
            DependencyGraph._add_dep(op, dependency, deps, reason)

    @staticmethod
    def _add_dep(
        op: PlanOp,
        dependency: PlanOp,
        deps: dict[str, dict[str, set[str]]],
        reason: str,
    ) -> None:
        deps[op.id].setdefault(dependency.id, set()).add(reason)

    def _topological_sort(self) -> list[str]:
        reverse_edges: dict[str, set[str]] = defaultdict(set)
        indegree = {op.id: len(self.dependencies[op.id]) for op in self.plan.ops}

        for op_id, dep_ids in self.dependencies.items():
            for dep_id in dep_ids:
                reverse_edges[dep_id].add(op_id)

        queue = deque(op.id for op in self.plan.ops if indegree[op.id] == 0)
        order: list[str] = []

        while queue:
            op_id = queue.popleft()
            order.append(op_id)
            for child_id in sorted(reverse_edges.get(op_id, ())):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    queue.append(child_id)

        if len(order) != len(indegree):
            cyclic_ids = sorted(op_id for op_id, degree in indegree.items() if degree > 0)
            sample = ", ".join(cyclic_ids[:8])
            raise ValueError(f"Dependency graph contains a cycle. Sample blocked ops: {sample}")

        return order

    def _batch_id(self, microbatch_id: int | None) -> int:
        if microbatch_id is None:
            return -1
        return int(microbatch_id) // max(1, self._llm_pp_size())

    @staticmethod
    def _chunk(op: PlanOp) -> int:
        return 0 if op.chunk_id is None else int(op.chunk_id)

    @staticmethod
    def _identity(
        rank: int,
        microbatch_id: int | None,
        chunk_id: int | None,
        op_type: str,
    ) -> tuple[int, int | None, int | None, str]:
        return (rank, microbatch_id, 0 if chunk_id is None else chunk_id, op_type)

    def _first_llm_rank(self) -> int:
        layout = self.plan.pipeline_layout or {}
        return int(layout.get("llm_stage_offset", 0))

    def _llm_pp_size(self) -> int:
        layout = self.plan.pipeline_layout or {}
        return int(layout.get("llm_pp_size", self.plan.pp_size))

    def _last_llm_rank(self) -> int:
        return self._first_llm_rank() + self._llm_pp_size() - 1

    def _has_disaggregated_layout(self) -> bool:
        return bool(self.plan.pipeline_layout)

    def _has_generator(self) -> bool:
        layout = self.plan.pipeline_layout
        if not layout:
            return True
        return bool(layout.get("has_generator", "generator_stage" in layout))

    def _ve_rank(self, microbatch_id: int | None) -> int:
        if microbatch_id is None:
            return 0
        return int(microbatch_id) % self.plan.pp_size

    def _encoder_stage(self) -> int:
        layout = self.plan.pipeline_layout or {}
        return int(layout.get("encoder_stage", 0))

    def _generator_stage(self) -> int:
        layout = self.plan.pipeline_layout or {}
        return int(layout.get("generator_stage", self.plan.pp_size - 1))


CHROME_TRACE_CHUNK_COLORS = [
    "thread_state_running",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "good",
    "bad",
    "terrible",
]


@dataclass(frozen=True)
class SimulatedOp:
    id: str
    rank: int
    index: int
    original_index: int
    op_type: str
    microbatch_id: int | None
    chunk_id: int | None
    label: str
    duration: float
    start_time: float
    end_time: float
    wait_time: float
    deps: list[str]
    dep_reasons: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "index": self.index,
            "original_index": self.original_index,
            "op_type": self.op_type,
            "microbatch_id": self.microbatch_id,
            "chunk_id": self.chunk_id,
            "label": self.label,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "wait_time": self.wait_time,
            "deps": self.deps,
            "dep_reasons": self.dep_reasons,
        }


@dataclass(frozen=True)
class SimulationResult:
    metadata: dict[str, Any]
    ops: list[SimulatedOp]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "ops": [op.to_dict() for op in self.ops],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    def to_chrome_trace_dict(
        self,
        *,
        time_unit_scale: float = 1000.0,
        perfetto_compat: bool = False,
    ) -> dict[str, Any]:
        # Return a Chrome Trace JSON payload.
        #
        # Chrome Trace timestamps are conventionally microseconds. Simulator time
        # units are abstract, so the default maps one simulator unit to 1000 us.

        events: list[dict[str, Any]] = []
        ranks = sorted({op.rank for op in self.ops})
        pid = 1
        tid_by_rank = {rank: rank + 1 for rank in ranks}

        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": self.metadata.get("scheduler", "PP Simulator")},
            }
        )

        for rank in ranks:
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid_by_rank[rank],
                    "args": {"name": f"PP Rank {rank}"},
                }
            )
            events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid_by_rank[rank],
                    "args": {"sort_index": rank},
                }
            )

        slice_events = []
        for op in self.ops:
            pp_size = int(self.metadata.get("pp_size", 1))
            trace_category = _trace_category(op)
            slice_events.append(
                {
                    "name": _trace_event_name(op, perfetto_compat=perfetto_compat),
                    "cat": trace_category,
                    "cname": _trace_color_name(op),
                    "ph": "X",
                    "pid": pid,
                    "tid": tid_by_rank[op.rank],
                    "ts": op.start_time * time_unit_scale,
                    "dur": op.duration * time_unit_scale,
                    "args": {
                        "id": op.id,
                        "label": op.label,
                        "op_type": op.op_type,
                        "microbatch_id": op.microbatch_id,
                        "batch_id": (
                            op.microbatch_id // pp_size
                            if op.microbatch_id is not None and pp_size > 0
                            else None
                        ),
                        "chunk_id": op.chunk_id,
                        "rank": op.rank,
                        "pp_rank": op.rank,
                        "index": op.index,
                        "original_index": op.original_index,
                        "start_time": op.start_time,
                        "end_time": op.end_time,
                        "duration": op.duration,
                        "wait_time": op.wait_time,
                        "visual_category": trace_category,
                        "deps": op.deps,
                        "dep_reasons": op.dep_reasons,
                    },
                }
            )
        events.extend(sorted(slice_events, key=lambda event: (event["ts"], event["tid"], event["name"])))

        return {
            "displayTimeUnit": "ms",
            "metadata": self.metadata,
            "summary": self.summary,
            "traceEvents": events,
        }

    def to_chrome_trace(
        self,
        path: str | Path | None = None,
        *,
        indent: int = 2,
        time_unit_scale: float = 1000.0,
        perfetto_compat: bool = False,
    ) -> str:
        payload = json.dumps(
            self.to_chrome_trace_dict(
                time_unit_scale=time_unit_scale,
                perfetto_compat=perfetto_compat,
            ),
            indent=indent,
            sort_keys=True,
        )
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload


@dataclass(frozen=True)
class MonteCarloReport:
    metadata: dict[str, Any]
    statistics: dict[str, Any]
    trial_summaries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "statistics": self.statistics,
            "trial_summaries": self.trial_summaries,
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload


def _trace_category(op: SimulatedOp) -> str:
    if op.chunk_id is None:
        return f"{op.op_type}/chunk_none"
    return f"{op.op_type}/chunk_{_chunk_label(op.chunk_id)}"


def _trace_event_name(op: SimulatedOp, *, perfetto_compat: bool) -> str:
    if not perfetto_compat:
        return op.label

    chunk = "none" if op.chunk_id is None else _chunk_label(op.chunk_id)
    mb = "none" if op.microbatch_id is None else str(op.microbatch_id)
    return f"rank{op.rank}/{op.op_type}/chunk_{chunk}/mb{mb}"


def _trace_color_name(op: SimulatedOp) -> str:
    if op.chunk_id is None:
        return CHROME_TRACE_CHUNK_COLORS[0]
    return CHROME_TRACE_CHUNK_COLORS[int(op.chunk_id) % len(CHROME_TRACE_CHUNK_COLORS)]


def _chunk_label(chunk_id: int) -> str:
    chunk_id = int(chunk_id)
    if chunk_id < 0:
        return str(chunk_id)

    letters = []
    value = chunk_id
    while True:
        letters.append(chr(ord("a") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


class PipelineSimulator:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.graph = DependencyGraph(plan)
        self._ops_by_id = {op.id: op for op in plan.ops}

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> "PipelineSimulator":
        return cls(Plan.from_scheduler(scheduler))

    def simulate(
        self,
        duration_specs: Mapping[Any, OpDurationSpec | Mapping[str, float]],
        *,
        seed: int | None = None,
        default_spec: OpDurationSpec | None = None,
        duration_overrides: Mapping[str, float] | None = None,
    ) -> SimulationResult:
        if duration_overrides is None:
            sampler = DurationSampler(duration_specs, default_spec=default_spec, seed=seed)
            sampled_durations = {
                op.id: sampler.sample(op.op_type, fallback_duration=op.base_duration)
                for op in self.plan.ops
            }
        else:
            sampled_durations = {
                op.id: max(0.0, float(duration_overrides[op.id]))
                for op in self.plan.ops
            }

        rank_free_time = {rank: 0.0 for rank in self.plan.ops_by_rank}
        simulated_by_id: dict[str, SimulatedOp] = {}

        for op_id in self.graph.topological_order:
            op = self._ops_by_id[op_id]
            dep_ids = sorted(self.graph.dependencies[op.id])
            dependency_ready_time = max(
                (simulated_by_id[dep_id].end_time for dep_id in dep_ids),
                default=0.0,
            )
            previous_rank_time = rank_free_time[op.rank]
            start_time = max(previous_rank_time, dependency_ready_time)
            duration = sampled_durations[op.id]
            end_time = start_time + duration

            simulated_by_id[op.id] = SimulatedOp(
                id=op.id,
                rank=op.rank,
                index=op.index,
                original_index=op.original_index,
                op_type=op.op_type,
                microbatch_id=op.microbatch_id,
                chunk_id=op.chunk_id,
                label=op.label,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                wait_time=max(0.0, start_time - previous_rank_time),
                deps=dep_ids,
                dep_reasons={
                    dep_id: sorted(self.graph.dependency_reasons[op.id][dep_id])
                    for dep_id in dep_ids
                },
            )
            rank_free_time[op.rank] = end_time

        ops = [
            simulated_by_id[op.id]
            for rank in sorted(self.plan.ops_by_rank)
            for op in self.plan.ops_by_rank[rank]
        ]
        summary = self._build_summary(ops)

        return SimulationResult(
            metadata={
                "scheduler": self.plan.scheduler_name,
                "pp_size": self.plan.pp_size,
                "vpp_size": self.plan.vpp_size,
                "num_microbatches": self.plan.num_microbatches,
                "seed": seed,
                "op_count": len(ops),
                "pipeline_layout": self.plan.pipeline_layout,
            },
            ops=ops,
            summary=summary,
        )

    def monte_carlo(
        self,
        duration_specs: Mapping[Any, OpDurationSpec | Mapping[str, float]],
        *,
        num_trials: int,
        seed: int | None = None,
        default_spec: OpDurationSpec | None = None,
        validate: bool = False,
    ) -> MonteCarloReport:
        if num_trials <= 0:
            raise ValueError(f"num_trials must be positive, got {num_trials}")

        trial_summaries: list[dict[str, Any]] = []
        for trial_index in range(num_trials):
            trial_seed = None if seed is None else seed + trial_index
            result = self.simulate(
                duration_specs,
                seed=trial_seed,
                default_spec=default_spec,
            )
            if validate:
                self.validate_result(result)
            trial_summaries.append(
                {
                    "trial_index": trial_index,
                    "seed": trial_seed,
                    "summary": result.summary,
                }
            )

        return MonteCarloReport(
            metadata={
                "scheduler": self.plan.scheduler_name,
                "pp_size": self.plan.pp_size,
                "vpp_size": self.plan.vpp_size,
                "num_microbatches": self.plan.num_microbatches,
                "op_count": len(self.plan.ops),
                "num_trials": num_trials,
                "seed": seed,
                "pipeline_layout": self.plan.pipeline_layout,
            },
            statistics=_build_monte_carlo_statistics(trial_summaries),
            trial_summaries=trial_summaries,
        )

    def validate_result(self, result: SimulationResult, *, tolerance: float = 1e-9) -> None:
        by_id = {op.id: op for op in result.ops}
        for op in result.ops:
            for dep_id in op.deps:
                if op.start_time + tolerance < by_id[dep_id].end_time:
                    raise AssertionError(f"{op.id} starts before dependency {dep_id} ends")

        for rank in sorted(self.plan.ops_by_rank):
            rank_ops = [op for op in result.ops if op.rank == rank]
            for previous, current in zip(rank_ops, rank_ops[1:]):
                if current.start_time + tolerance < previous.end_time:
                    raise AssertionError(f"{current.id} overlaps previous rank op {previous.id}")

    def _build_summary(self, ops: list[SimulatedOp]) -> dict[str, Any]:
        makespan = max((op.end_time for op in ops), default=0.0)
        rank_compute_time: dict[int, float] = {rank: 0.0 for rank in self.plan.ops_by_rank}
        rank_wait_time: dict[int, float] = {rank: 0.0 for rank in self.plan.ops_by_rank}
        op_type_time: dict[str, float] = {}

        for op in ops:
            rank_compute_time[op.rank] += op.duration
            rank_wait_time[op.rank] += op.wait_time
            op_type_time[op.op_type] = op_type_time.get(op.op_type, 0.0) + op.duration

        return {
            "makespan": makespan,
            "rank_compute_time": {str(rank): value for rank, value in rank_compute_time.items()},
            "rank_wait_time": {str(rank): value for rank, value in rank_wait_time.items()},
            "rank_utilization": {
                str(rank): (value / makespan if makespan > 0 else 0.0)
                for rank, value in rank_compute_time.items()
            },
            "op_type_time": dict(sorted(op_type_time.items())),
            "total_wait_time": sum(rank_wait_time.values()),
        }


def _build_monte_carlo_statistics(trial_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [trial["summary"] for trial in trial_summaries]
    return {
        "makespan": _summarize_values(summary["makespan"] for summary in summaries),
        "total_wait_time": _summarize_values(summary["total_wait_time"] for summary in summaries),
        "rank_compute_time": _summarize_nested_metric(summaries, "rank_compute_time"),
        "rank_wait_time": _summarize_nested_metric(summaries, "rank_wait_time"),
        "rank_utilization": _summarize_nested_metric(summaries, "rank_utilization"),
        "op_type_time": _summarize_nested_metric(summaries, "op_type_time"),
    }


def _summarize_nested_metric(summaries: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    keys = sorted({key for summary in summaries for key in summary[metric_name]})
    return {
        key: _summarize_values(summary[metric_name].get(key, 0.0) for summary in summaries)
        for key in keys
    }


def _summarize_values(values: Any) -> dict[str, float]:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = sum((value - mean) ** 2 for value in sorted_values) / count
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p50": _percentile(sorted_values, 50),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
