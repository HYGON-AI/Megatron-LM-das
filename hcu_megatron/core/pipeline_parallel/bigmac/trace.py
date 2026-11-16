# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import json
from pathlib import Path
from typing import Iterable, Optional


LLM_CHUNKED_OP_TYPE_VALUES = {
    "F",
    "B",
    "SF",
    "RF",
    "SB",
    "RB",
    "SFRF",
    "SFRB",
    "SBRF",
    "SBRB",
    "LIOF",
    "LION",
}


def _is_fused_op(op) -> bool:
    return hasattr(op, "op1") and hasattr(op, "op2")


def iter_trace_ops(op):
    # Yield the visible trace op and any inner ops from a fused schedule item.
    if _is_fused_op(op):
        yield op
        yield op.op1
        yield op.op2
        return
    yield op


def mark_op_trace(op, start_us: int, end_us: int):
    for trace_op in iter_trace_ops(op):
        trace_op.trace_start_us = start_us
        trace_op.trace_end_us = end_us
        trace_op.trace_base_event = None
        trace_op.trace_start_event = None
        trace_op.trace_end_event = None
        trace_op.trace_origin_us = None


def mark_op_trace_events(op, base_event, start_event, end_event, origin_us: int):
    for trace_op in iter_trace_ops(op):
        trace_op.trace_start_us = None
        trace_op.trace_end_us = None
        trace_op.trace_base_event = base_event
        trace_op.trace_start_event = start_event
        trace_op.trace_end_event = end_event
        trace_op.trace_origin_us = origin_us


def _resolve_event_timestamps(op) -> bool:
    base_event = getattr(op, "trace_base_event", None)
    start_event = getattr(op, "trace_start_event", None)
    end_event = getattr(op, "trace_end_event", None)
    origin_us = getattr(op, "trace_origin_us", None)
    if (
        base_event is None
        or start_event is None
        or end_event is None
        or origin_us is None
    ):
        return False

    end_event.synchronize()
    start_us = int(origin_us + base_event.elapsed_time(start_event) * 1000)
    end_us = int(origin_us + base_event.elapsed_time(end_event) * 1000)
    for trace_op in iter_trace_ops(op):
        trace_op.trace_start_us = start_us
        trace_op.trace_end_us = end_us
    return True


def _op_type_value(op) -> str:
    op_type = getattr(op, "op_type", None)
    return getattr(op_type, "value", str(op_type))


def _should_show_chunk_in_name(op) -> bool:
    return _op_type_value(op) in LLM_CHUNKED_OP_TYPE_VALUES


def _single_event_name(op) -> str:
    microbatch_id = getattr(op, "microbatch_id", None)
    name = f"{_op_type_value(op)}{'' if microbatch_id is None else microbatch_id}"
    chunk_id = getattr(op, "chunk_id", None)
    if chunk_id is not None and _should_show_chunk_in_name(op):
        name = f"{name}-C{chunk_id}"
    return name


def _event_name(op) -> str:
    if _is_fused_op(op):
        return f"{_op_type_value(op)}[{_single_event_name(op.op1)}, {_single_event_name(op.op2)}]"
    return _single_event_name(op)


def _event_chunk_ids(op) -> list:
    chunk_ids = []
    for trace_op in iter_trace_ops(op):
        chunk_id = getattr(trace_op, "chunk_id", None)
        if chunk_id is not None and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return chunk_ids


def _event_tid(op) -> str:
    return "executor"


def _event_args(op) -> dict:
    chunk_ids = _event_chunk_ids(op)
    args = {
        "op": str(op),
        "op_type": _op_type_value(op),
        "microbatch_id": getattr(op, "microbatch_id", None),
        "chunk_id": getattr(op, "chunk_id", None),
    }
    if chunk_ids:
        args["chunk_ids"] = chunk_ids
    if _is_fused_op(op):
        args["op1"] = str(op.op1)
        args["op2"] = str(op.op2)
        args["op1_chunk_id"] = getattr(op.op1, "chunk_id", None)
        args["op2_chunk_id"] = getattr(op.op2, "chunk_id", None)
    for attr in ("src_rank", "dst_rank"):
        if hasattr(op, attr):
            args[attr] = getattr(op, attr)
    return args


def _format_rank(rank: int, rank_width: int = None) -> str:
    if rank_width is None:
        return str(rank)
    return f"{rank:0{rank_width}d}"


def build_chrome_trace_events(schedule: Iterable, pp_rank: int, pp_rank_width: int = None) -> list:
    events = []
    for op in schedule:
        start_us = getattr(op, "trace_start_us", None)
        end_us = getattr(op, "trace_end_us", None)
        if start_us is None or end_us is None:
            if _resolve_event_timestamps(op):
                start_us = getattr(op, "trace_start_us", None)
                end_us = getattr(op, "trace_end_us", None)
        if start_us is None or end_us is None:
            continue
        events.append(
            {
                "name": _event_name(op),
                "cat": "pp_executor",
                "ph": "X",
                "ts": int(start_us),
                "dur": max(0, int(end_us) - int(start_us)),
                "pid": f"pp_rank_{_format_rank(pp_rank, pp_rank_width)}",
                "tid": _event_tid(op),
                "args": _event_args(op),
            }
        )
    return events


def export_chrome_trace_events(events: Iterable, trace_dir, filename: str) -> str:
    output_dir = Path(trace_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = {
        "displayTimeUnit": "ms",
        "traceEvents": list(events),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def export_chrome_trace(
    schedule: Iterable,
    trace_dir,
    pp_rank: int,
    filename: Optional[str] = None,
    pp_rank_width: int = None,
) -> str:
    if filename is None:
        filename = f"pp_rank_{_format_rank(pp_rank, pp_rank_width)}_trace.json"
    return export_chrome_trace_events(
        build_chrome_trace_events(schedule, pp_rank, pp_rank_width=pp_rank_width),
        trace_dir,
        filename,
    )
