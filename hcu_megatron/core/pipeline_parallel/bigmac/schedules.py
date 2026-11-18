# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import os
import warnings
from typing import Dict, List, Optional, Tuple
from enum import Enum


def _resolve_ve_forward_limit(ve_forward_limit: int) -> int:
    env_value = os.getenv("VE_FORWARD_LIMIT")
    if env_value is None:
        return ve_forward_limit

    try:
        return int(env_value)
    except ValueError:
        warnings.warn(
            f"Invalid VE_FORWARD_LIMIT value: {env_value!r}. Falling back to "
            f"function argument ve_forward_limit={ve_forward_limit}.",
            stacklevel=2,
        )
        return ve_forward_limit


class OpType(Enum):
    # Operation type: forward or backward propagation
    FORWARD = "F"
    BACKWARD = "B"
    IDLE = "IDLE"
    VEFORWARD = "VF"
    VEBACKWARD = "VB"
    VESENDFORWARD = "VSF"
    VESENDBACKWARD = "VSB"
    VERECVFORWARD = "VRF"
    VERECVBACKWARD = "VRB"
    SENDFORWARD = "SF"
    RECVFORWARD = "RF"
    SENDBACKWARD = "SB"
    RECVBACKWARD = "RB"
    NULLCOMMUNICATION = "NC"
    GENFORWARD = "GF"
    GENBACKWARD = "GB"
    GENSENDFORWARD = "GSF"
    GENRECVFORWARD = "GRF"
    GENRECVBACKWARD = "GRB"
    GENSENDBACKWARD = "GSB"
    SENDFORWARDRECVFORWARD = "SFRF"
    SENDFORWARDRECVBACKWARD = "SFRB"
    SENDBACKWARDRECVFORWARD = "SBRF"
    SENDBACKWARDRECVBACKWARD = "SBRB"


LLM_SEND_RECV_OP_TYPES = [OpType.SENDFORWARD, OpType.RECVFORWARD, OpType.SENDBACKWARD, OpType.RECVBACKWARD]
VE_SEND_RECV_OP_TYPES = [OpType.VESENDFORWARD, OpType.VERECVFORWARD, OpType.VESENDBACKWARD, OpType.VERECVBACKWARD]
LLM_SEND_OP_TYPES = [OpType.SENDFORWARD, OpType.SENDBACKWARD]
LLM_RECV_OP_TYPES = [OpType.RECVFORWARD, OpType.RECVBACKWARD]
LLM_COMPUTE_OP_TYPES = [OpType.FORWARD, OpType.BACKWARD]
NULL_OP_TYPES = [OpType.IDLE, OpType.NULLCOMMUNICATION]
LLM_FUSION_OP_TYPES = [OpType.SENDFORWARDRECVFORWARD, OpType.SENDFORWARDRECVBACKWARD, OpType.SENDBACKWARDRECVFORWARD, OpType.SENDBACKWARDRECVBACKWARD]

SEND_TO_RECV_OP_TYPES = {
    OpType.SENDFORWARD: OpType.RECVFORWARD,
    OpType.SENDBACKWARD: OpType.RECVBACKWARD,
    OpType.VESENDFORWARD: OpType.VERECVFORWARD,
    OpType.VESENDBACKWARD: OpType.VERECVBACKWARD,
    OpType.GENSENDFORWARD: OpType.GENRECVFORWARD,
    OpType.GENSENDBACKWARD: OpType.GENRECVBACKWARD,
}
RECV_TO_SEND_OP_TYPES = {recv: send for send, recv in SEND_TO_RECV_OP_TYPES.items()}
SEND_OP_TYPES = set(SEND_TO_RECV_OP_TYPES.keys())
RECV_OP_TYPES = set(RECV_TO_SEND_OP_TYPES.keys())
SEND_RECV_OP_TYPES = SEND_OP_TYPES | RECV_OP_TYPES
DEADLOCK_DETECTION_OP_TYPES = set(LLM_SEND_RECV_OP_TYPES)


def _default_duration(op_type: OpType) -> float:
    # Return the default duration used by schedule visualizers and simulators.
    if op_type == OpType.BACKWARD:
        return 2
    return 1


class ScheduleOp:
    # Schedule operation representing what should be executed at a given step

    def __init__(
        self,
        microbatch_id: Optional[int],
        op_type: OpType,
        duration: Optional[float] = None,
    ):
        self.microbatch_id = microbatch_id
        self.op_type = op_type
        self.duration = duration if duration is not None else _default_duration(op_type)
    
    def __repr__(self):
        if self.op_type == OpType.IDLE:
            return "IDLE"
        return f"{self.op_type.value}{self.microbatch_id}"
    
    def __str__(self):
        return self.__repr__()


class CommunicationOp(ScheduleOp):
    def __init__(
        self,
        comp_op: ScheduleOp,
        op_type: OpType,
        src_rank: int,
        dst_rank: int,
        duration: Optional[float] = None,
    ):
        microbatch_id = comp_op.microbatch_id
        super().__init__(microbatch_id, op_type, duration)
        self.op_type = op_type
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.pinned = False
        if hasattr(comp_op, 'chunk_id'):
            self.chunk_id = comp_op.chunk_id
        else:
            self.chunk_id = None
    
    def __repr__(self):
        if self.op_type == OpType.IDLE:
            return "IDLE"
        if self.op_type == OpType.NULLCOMMUNICATION:
            return "NC"
        if self.chunk_id is not None:
            return f"{self.op_type.value}{self.microbatch_id}-C{self.chunk_id}-{self.src_rank}->{self.dst_rank}"
        else:
            return f"{self.op_type.value}{self.microbatch_id}-{self.src_rank}->{self.dst_rank}"
    
    def __str__(self):
        return self.__repr__()


def get_fused_op_type(op1: OpType, op2: OpType) -> OpType:
    if op1 == OpType.SENDFORWARD and op2 == OpType.RECVFORWARD:
        return OpType.SENDFORWARDRECVFORWARD
    elif op1 == OpType.SENDFORWARD and op2 == OpType.RECVBACKWARD:
        return OpType.SENDFORWARDRECVBACKWARD
    elif op1 == OpType.SENDBACKWARD and op2 == OpType.RECVFORWARD:
        return OpType.SENDBACKWARDRECVFORWARD
    elif op1 == OpType.SENDBACKWARD and op2 == OpType.RECVBACKWARD:
        return OpType.SENDBACKWARDRECVBACKWARD
    else:
        raise ValueError(f"Invalid operation type: {op1} and {op2} to fuse")

class FusedScheduleOp:
    def __init__(self, op1: ScheduleOp, op2: ScheduleOp):
        self.op1 = op1
        self.op2 = op2
        self.op_type = get_fused_op_type(op1.op_type, op2.op_type)
    
    def __repr__(self):
        return f"[{self.op1}, {self.op2}]"
    
    def __str__(self):
        return self.__repr__()


def _normalized_chunk_id_for_match(op: ScheduleOp, pp_size: int) -> Optional[int]:
    chunk_id = getattr(op, "chunk_id", None)
    if chunk_id is None or pp_size <= 1:
        return chunk_id

    # VPP wraps forward tensors from the last physical rank to rank 0's next
    # virtual chunk, and backward tensors from rank 0 to the last rank's
    # previous virtual chunk. Normalize those receiver-side chunk ids so both
    # sides describe the same communication edge.
    if (
        op.op_type is OpType.RECVFORWARD
        and op.src_rank == pp_size - 1
        and op.dst_rank == 0
    ):
        return chunk_id - 1
    if (
        op.op_type is OpType.RECVBACKWARD
        and op.src_rank == 0
        and op.dst_rank == pp_size - 1
    ):
        return chunk_id + 1
    return chunk_id


def _communication_key(
    op: ScheduleOp,
    pp_size: int,
) -> Optional[Tuple[OpType, Optional[int], Optional[int], int, int]]:
    if getattr(op, "op_type", None) not in SEND_RECV_OP_TYPES:
        return None
    return (
        op.op_type,
        op.microbatch_id,
        _normalized_chunk_id_for_match(op, pp_size),
        op.src_rank,
        op.dst_rank,
    )


def _matching_communication_key(
    op: ScheduleOp,
    pp_size: int,
) -> Optional[Tuple[OpType, Optional[int], Optional[int], int, int]]:
    op_type = getattr(op, "op_type", None)
    if op_type in SEND_TO_RECV_OP_TYPES:
        matching_op_type = SEND_TO_RECV_OP_TYPES[op_type]
    elif op_type in RECV_TO_SEND_OP_TYPES:
        matching_op_type = RECV_TO_SEND_OP_TYPES[op_type]
    else:
        return None
    return (
        matching_op_type,
        op.microbatch_id,
        _normalized_chunk_id_for_match(op, pp_size),
        op.src_rank,
        op.dst_rank,
    )


def _communication_peer(op: ScheduleOp) -> Optional[int]:
    op_type = getattr(op, "op_type", None)
    if op_type in SEND_OP_TYPES:
        return op.dst_rank
    if op_type in RECV_OP_TYPES:
        return op.src_rank
    return None


def _iter_schedule_ops(schedule: List[ScheduleOp]):
    for op in schedule:
        if isinstance(op, FusedScheduleOp):
            yield op.op1
            yield op.op2
        else:
            yield op


def _infer_vpp_size(schedules_with_comm: Dict[int, List[ScheduleOp]]) -> int:
    chunk_ids = [
        op.chunk_id
        for schedule in schedules_with_comm.values()
        for op in _iter_schedule_ops(schedule)
        if getattr(op, "chunk_id", None) is not None
    ]
    return max(chunk_ids) + 1 if chunk_ids else 1


def _is_pp_boundary_noop(op: ScheduleOp, pp_size: int, vpp_size: int) -> bool:
    if pp_size <= 0:
        return False

    chunk_id = getattr(op, "chunk_id", None)
    first_chunk = chunk_id is None or chunk_id == 0
    last_chunk = chunk_id is None or chunk_id == vpp_size - 1
    first_rank = 0
    last_rank = pp_size - 1

    if (
        op.op_type is OpType.RECVFORWARD
        and op.src_rank == last_rank
        and op.dst_rank == first_rank
        and first_chunk
    ):
        return True

    if (
        op.op_type is OpType.SENDFORWARD
        and op.src_rank == last_rank
        and op.dst_rank == first_rank
        and last_chunk
    ):
        return True

    if (
        op.op_type is OpType.RECVBACKWARD
        and op.src_rank == first_rank
        and op.dst_rank == last_rank
        and last_chunk
    ):
        return True

    if (
        op.op_type is OpType.SENDBACKWARD
        and op.src_rank == first_rank
        and op.dst_rank == last_rank
        and first_chunk
    ):
        return True

    return False


def _expand_deadlock_item(op, fused_op_mode: str) -> List[Dict]:
    if isinstance(op, FusedScheduleOp):
        if fused_op_mode == "sequential":
            return [
                {"display": str(op.op1), "ops": [op.op1]},
                {"display": str(op.op2), "ops": [op.op2]},
            ]
        if fused_op_mode == "nonblocking":
            return [{"display": str(op), "ops": [op.op1, op.op2]}]
        raise ValueError("fused_op_mode must be 'nonblocking' or 'sequential'")
    return [{"display": str(op), "ops": [op]}]


def _communication_ops(
    item: Dict,
    pp_size: int,
    vpp_size: int,
    ignore_pp_boundary_noops: bool,
) -> List[ScheduleOp]:
    return [
        op
        for op in item["ops"]
        if (
            getattr(op, "op_type", None) in DEADLOCK_DETECTION_OP_TYPES
            and not (
                ignore_pp_boundary_noops
                and _is_pp_boundary_noop(op, pp_size, vpp_size)
            )
        )
    ]


def _find_rank_cycles(wait_edges: List[Tuple[int, int, str]]) -> List[List[int]]:
    graph = {}
    for src, dst, _ in wait_edges:
        graph.setdefault(src, set()).add(dst)

    cycles = []
    seen = set()

    def normalize(cycle: List[int]) -> Tuple[int, ...]:
        body = cycle[:-1]
        rotations = [tuple(body[i:] + body[:i]) for i in range(len(body))]
        return min(rotations)

    def dfs(start: int, node: int, path: List[int], visiting: set):
        for neighbor in graph.get(node, ()):
            if neighbor == start:
                cycle = path + [start]
                key = normalize(cycle)
                if key not in seen:
                    seen.add(key)
                    cycles.append(cycle)
                continue
            if neighbor in visiting:
                continue
            visiting.add(neighbor)
            dfs(start, neighbor, path + [neighbor], visiting)
            visiting.remove(neighbor)

    for rank in graph:
        dfs(rank, rank, [rank], {rank})

    return cycles


def detect_communication_deadlock(
    schedules_with_comm: Dict[int, List[ScheduleOp]],
    fused_op_mode: str = "nonblocking",
    ignore_pp_boundary_noops: bool = True,
    max_steps: int = 100000,
) -> Dict:
    # Detect blocking LLM PP P2P communication cycles in a materialized pipeline schedule.
    #
    # Args:
    #     schedules_with_comm: Rank -> schedule list after communication ops are inserted.
    #         VE and generator/DiT communication ops are ignored by this detector.
    #     fused_op_mode: "nonblocking" treats FusedScheduleOp like Megatron-style
    #         isend/irecv posted together. "sequential" expands it into op1 then op2,
    #         which models an executor that naively runs blocking send/recv in order.
    #     ignore_pp_boundary_noops: Ignore PP/VPP boundary ops that do not
    #         correspond to real P2P communication, such as rank 0/chunk 0
    #         RECVFORWARD from the last rank and last-rank/last-chunk
    #         SENDFORWARD to rank 0.
    #     max_steps: Guard against malformed schedules.
    #
    # Returns:
    #     A dict with:
    #         has_deadlock: True if the static execution model reaches a blocked state.
    #         has_cycle: True if the blocked state contains a rank wait cycle.
    #         cycles: Rank cycles such as [0, 1, 0].
    #         wait_edges: Edges (rank -> peer) explaining who waits for whom.
    #         blocked_ops: Current blocked op for each blocked rank.
    #         reason: "communication_cycle", "blocked_communication", or None.
    programs = {
        rank: [
            item
            for op in schedule
            for item in _expand_deadlock_item(op, fused_op_mode)
        ]
        for rank, schedule in schedules_with_comm.items()
    }
    pcs = {rank: 0 for rank in programs}
    pp_size = max(programs.keys()) + 1 if programs else 0
    vpp_size = _infer_vpp_size(schedules_with_comm)

    for _ in range(max_steps):
        advanced_compute = True
        while advanced_compute:
            advanced_compute = False
            for rank, program in programs.items():
                if pcs[rank] >= len(program):
                    continue
                if not _communication_ops(
                    program[pcs[rank]],
                    pp_size,
                    vpp_size,
                    ignore_pp_boundary_noops,
                ):
                    pcs[rank] += 1
                    advanced_compute = True

        if all(pcs[rank] >= len(program) for rank, program in programs.items()):
            return {
                "has_deadlock": False,
                "has_cycle": False,
                "cycles": [],
                "wait_edges": [],
                "blocked_ops": {},
                "reason": None,
            }

        posted_keys = set()
        current_comm_items = {}
        for rank, program in programs.items():
            if pcs[rank] >= len(program):
                continue
            item = program[pcs[rank]]
            comm_ops = _communication_ops(
                item,
                pp_size,
                vpp_size,
                ignore_pp_boundary_noops,
            )
            if not comm_ops:
                continue
            current_comm_items[rank] = (item, comm_ops)
            for op in comm_ops:
                key = _communication_key(op, pp_size)
                if key is not None:
                    posted_keys.add(key)

        completable_ranks = []
        for rank, (_, comm_ops) in current_comm_items.items():
            if all(_matching_communication_key(op, pp_size) in posted_keys for op in comm_ops):
                completable_ranks.append(rank)

        if completable_ranks:
            for rank in completable_ranks:
                pcs[rank] += 1
            continue

        wait_edges = []
        for rank, (item, comm_ops) in current_comm_items.items():
            for op in comm_ops:
                matching_key = _matching_communication_key(op, pp_size)
                if matching_key in posted_keys:
                    continue
                peer = _communication_peer(op)
                if peer is None:
                    continue
                wait_edges.append(
                    (
                        rank,
                        peer,
                        f"{item['display']} waits for {matching_key[0].value}{matching_key[1]} "
                        f"{matching_key[3]}->{matching_key[4]}",
                    )
                )

        cycles = _find_rank_cycles(wait_edges)
        return {
            "has_deadlock": True,
            "has_cycle": bool(cycles),
            "cycles": cycles,
            "wait_edges": wait_edges,
            "blocked_ops": {
                rank: item["display"] for rank, (item, _) in current_comm_items.items()
            },
            "reason": "communication_cycle" if cycles else "blocked_communication",
        }

    raise RuntimeError("Communication deadlock detection exceeded maximum steps")


class PPSchedule:
    # Base class for pipeline parallelism scheduling
    
    def __init__(self, pp_size: int, num_microbatches: int):
        # Initialize pipeline scheduler
        #
        # Args:
        #     pp_size: Pipeline parallel size (number of stages)
        #     num_microbatches: Total number of microbatches
        self.pp_size = pp_size
        self.num_microbatches = num_microbatches
        self.schedules = {}  # key: rank, value: List[ScheduleOp]
        
    def generate_schedule(self):
        # Generate schedules for all ranks, subclass must implement this method
        raise NotImplementedError("Subclass must implement generate_schedule()")
    
    def get_schedule(self, rank: int) -> List[ScheduleOp]:
        # Get the schedule sequence for a specific rank
        #
        # Args:
        #     rank: Pipeline stage rank (0 to pp_size-1)
        #     
        # Returns:
        #     List of scheduled operations for this rank
        if rank not in self.schedules:
            self.generate_schedule()
        return self.schedules.get(rank, [])

    def detect_communication_deadlock(
        self,
        fused_op_mode: str = "nonblocking",
        ignore_pp_boundary_noops: bool = True,
        max_steps: int = 100000,
    ) -> Dict:
        if not hasattr(self, "schedules_with_comm"):
            self.add_communication_op()
        return detect_communication_deadlock(
            self.schedules_with_comm,
            fused_op_mode=fused_op_mode,
            ignore_pp_boundary_noops=ignore_pp_boundary_noops,
            max_steps=max_steps,
        )
    
    def print_schedule(self, rank: Optional[int] = None):
        # Print schedule information
        #
        # Args:
        #     rank: If specified, only print schedule for this rank; otherwise print all ranks
        if not self.schedules:
            self.generate_schedule()
        
        if rank is not None:
            print(f"\nRank {rank} Schedule:")
            print(f"{'Step':<6} {'Operation':<10}")
            print("-" * 20)
            for step, op in enumerate(self.schedules[rank]):
                print(f"{step:<6} {op}")
        else:
            for r in range(self.pp_size):
                self.print_schedule(r)
    
    def visualize_timeline(self):
        # Visualize timeline (schedules for all ranks) with operation durations
        if not self.schedules:
            self.generate_schedule()
        
        # Calculate total time for each rank
        rank_times = {}
        for rank in range(self.pp_size):
            total_time = 0
            for op in self.schedules[rank]:
                total_time += op.duration
            rank_times[rank] = total_time
        
        max_time = max(rank_times.values())
        
        print(f"\n{'='*80}")
        print(f"Pipeline Schedule Timeline ({self.__class__.__name__})")
        print(f"PP Size: {self.pp_size}, Microbatches: {self.num_microbatches}")
        print(f"Total Time: {max_time} (F=1, B=2)")
        print(f"{'='*80}\n")
        
        # Print header with time units
        header = f"{'Rank':<6}"
        for t in range(max_time):
            header += f" {t:<7}"
        print(header)
        print("-" * len(header))
        
        # Print schedule for each rank with duration
        for rank in range(self.pp_size):
            schedule = self.schedules[rank]
            
            # Build a timeline showing what operation is executing at each time unit
            timeline = []
            current_time = 0
            
            for op in schedule:
                # Repeat the operation for its duration
                for _ in range(op.duration):
                    timeline.append(str(op))
                current_time += op.duration
            
            # Print the rank's timeline
            row = f"{rank:<6}"
            for t in range(max_time):
                if t < len(timeline):
                    row += f" {timeline[t]:<7}"
                else:
                    row += f" {'':7}"
            print(row)
        print()

    def null_comm_op(self, op: ScheduleOp, current_rank: int) -> CommunicationOp:
        return CommunicationOp(op, OpType.NULLCOMMUNICATION, src_rank=current_rank, dst_rank=current_rank, duration=0.1)
    
    def should_recv_emb_from_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        # Text-only schedules have no vision encoder. VE schedules opt in.
        return False
    
    def should_send_grad_to_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        return False
    
    def should_send_forward(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.FORWARD and current_rank < self.pp_size - 1
    
    def should_recv_backward(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.BACKWARD and current_rank < self.pp_size - 1
    
    def pre_comm_op(self, op: ScheduleOp, current_rank: int) -> CommunicationOp:
        pre_comm_ops = []
        if op.op_type == OpType.IDLE:
            pre_comm_ops.append(self.null_comm_op(op, current_rank))
        elif op.op_type == OpType.FORWARD: # llm forward
            # if self.should_recv_emb_from_ve(op, current_rank): # recv from ve forward
            #     send_rank = op.microbatch_id % self.pp_size
            #     pre_comm_ops.append(CommunicationOp(op, OpType.VERECVFORWARD, src_rank=send_rank, dst_rank=current_rank, duration=0.1))
            # executor will handle if recv forward is really needed
            send_rank = (current_rank - 1) % self.pp_size
            pre_comm_ops.append(CommunicationOp(op, OpType.RECVFORWARD, src_rank=send_rank, dst_rank=current_rank, duration=0.1))
        elif op.op_type == OpType.BACKWARD: # llm backward
            send_rank = (current_rank + 1) % self.pp_size
            pre_comm_ops.append(CommunicationOp(op, OpType.RECVBACKWARD, src_rank=send_rank, dst_rank=current_rank, duration=0.1))
        elif op.op_type == OpType.VEFORWARD: # ve forward
            pre_comm_ops.append(self.null_comm_op(op, current_rank))
        elif op.op_type == OpType.VEBACKWARD: # ve backward
            pre_comm_ops.append(CommunicationOp(op, OpType.VERECVBACKWARD, src_rank=0, dst_rank=current_rank, duration=0.1))
        elif op.op_type == OpType.GENFORWARD: # gen forward
            pre_comm_ops.append(CommunicationOp(op, OpType.GENRECVFORWARD, src_rank=0, dst_rank=current_rank, duration=0.1))
        elif op.op_type == OpType.GENBACKWARD: # gen backward
            pre_comm_ops.append(self.null_comm_op(op, current_rank))
        else:
            raise ValueError(f"Invalid operation type: {op.op_type}")
        return pre_comm_ops
    
    def post_comm_op(self, op: ScheduleOp, current_rank: int) -> CommunicationOp:
        post_comm_ops = []
        if op.op_type == OpType.IDLE:
            post_comm_ops.append(self.null_comm_op(op, current_rank))
        elif op.op_type == OpType.FORWARD: # llm forward
            if self.should_send_forward(op, current_rank): # send forward to next rank
                recv_rank = (current_rank + 1) % self.pp_size
                post_comm_ops.append(CommunicationOp(op, OpType.SENDFORWARD, src_rank=current_rank, dst_rank=recv_rank, duration=0.1))
        elif op.op_type == OpType.BACKWARD: # llm backward
            if self.should_send_grad_to_ve(op, current_rank): # send grad to ve backward
                recv_rank = op.microbatch_id % self.pp_size
                post_comm_ops.append(CommunicationOp(op, OpType.VESENDBACKWARD, src_rank=current_rank, dst_rank=recv_rank, duration=0.1))
            else: # send grad to previous rank
                recv_rank = (current_rank - 1) % self.pp_size
                post_comm_ops.append(CommunicationOp(op, OpType.SENDBACKWARD, src_rank=current_rank, dst_rank=recv_rank, duration=0.1))
        elif op.op_type == OpType.VEFORWARD: # ve forward
            post_comm_ops.append(CommunicationOp(op, OpType.VESENDFORWARD, src_rank=current_rank, dst_rank=0, duration=0.1))
        elif op.op_type == OpType.VEBACKWARD: # ve backward
            post_comm_ops.append(self.null_comm_op(op, current_rank))
        elif op.op_type == OpType.GENFORWARD: # gen forward
            post_comm_ops.append(self.null_comm_op(op, current_rank))
        elif op.op_type == OpType.GENBACKWARD: # gen backward
            post_comm_ops.append(CommunicationOp(op, OpType.GENSENDBACKWARD, src_rank=current_rank, dst_rank=0, duration=0.1))
        else:
            raise ValueError(f"Invalid operation type: {op.op_type}")
        return post_comm_ops
    
    def add_communication_op(self):
        assert not hasattr(self, 'schedules_with_comm'), "schedules_with_comm already exists"
        self.schedules_with_comm = {}
        for rank in range(self.pp_size):
            self.schedules_with_comm[rank] = []
            for op in self.schedules[rank]:
                pre_comm_ops = self.pre_comm_op(op, rank)
                post_comm_ops = self.post_comm_op(op, rank)
                self.schedules_with_comm[rank].extend(pre_comm_ops)
                self.schedules_with_comm[rank].append(op)
                self.schedules_with_comm[rank].extend(post_comm_ops)
    
    def optimize_communication_ops(self):
        self.remove_null_ops()
        self.reorder_llm_pp_send_recv()
    
    
    def remove_null_ops(self):
        # remove idle, null communication, operations.
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            for op in schedules:
                if op.op_type not in NULL_OP_TYPES:
                    optimized_schedules.append(op)
            self.schedules_with_comm[i] = optimized_schedules
    
    
    def fuse_comm_ops(self):
        # fuse comm ops to achieve p2p communication overlap.
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type is OpType.SENDFORWARD and j+1 < len(schedules) and schedules[j+1].op_type is OpType.RECVBACKWARD:
                    optimized_schedules.append(FusedScheduleOp(schedules[j], schedules[j+1]))
                    j += 2
                elif schedules[j].op_type is OpType.SENDBACKWARD and j+1 < len(schedules) and schedules[j+1].op_type is OpType.RECVFORWARD:
                    optimized_schedules.append(FusedScheduleOp(schedules[j], schedules[j+1]))
                    j += 2
                else:
                    optimized_schedules.append(schedules[j])
                    j += 1
            self.schedules_with_comm[i] = optimized_schedules
    
    
    def reorder_llm_pp_send_recv(self):
        # Reorder llm pp send and recv operations to avoid communication deadlock.
        return # this is not needed for now, we fuse comm ops to achieve p2p communication overlap and avoid communication deadlock
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in LLM_SEND_RECV_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                if j == len(schedules) - 1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                if getattr(schedules[j], 'pinned', False) or getattr(schedules[j+1], 'pinned', False):
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                if i % 2 == 0: # send first, recv then
                    if schedules[j].op_type in LLM_RECV_OP_TYPES and schedules[j+1].op_type in LLM_SEND_OP_TYPES:
                        optimized_schedules.extend([schedules[j+1], schedules[j]])
                        j += 2
                        continue
                else: # recv first, send then
                    if schedules[j].op_type in LLM_SEND_OP_TYPES and schedules[j+1].op_type in LLM_RECV_OP_TYPES:
                        optimized_schedules.extend([schedules[j+1], schedules[j]])
                        j += 2
                        continue
                optimized_schedules.append(schedules[j])
                j += 1
            self.schedules_with_comm[i] = optimized_schedules

        
class GPipe(PPSchedule):
    # GPipe pipeline scheduling strategy
    #
    # GPipe characteristics:
    # 1. First execute forward passes for all microbatches (fill phase)
    # 2. Then execute backward passes for all microbatches (drain phase)
    # 3. Each stage processes microbatches in order
    
    def _get_priority(self, op: ScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # GPipe prioritizes: Forward first (by microbatch ID ascending), then Backward (by microbatch ID descending)
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.FORWARD:
            # Forward passes: prioritize by ascending microbatch ID
            return (0, op.microbatch_id)
        else:  # BACKWARD
            # Backward passes: prioritize by descending microbatch ID (use negative for descending)
            return (1, -op.microbatch_id)
    
    def _unlock_next_ops(self, rank: int, op: ScheduleOp, available_ops: dict):
        # Unlock next available operations based on dependencies after executing an operation.
        #
        # Args:
        #     rank: The rank that just executed the operation
        #     op: The operation that was just executed
        #     available_ops: Dictionary of available operations for each rank
        mb = op.microbatch_id
        
        if op.op_type == OpType.FORWARD:
            # After rank r executes Fj, unlock rank r+1's Fj
            if rank < self.pp_size - 1:
                next_op = ScheduleOp(mb, OpType.FORWARD)
                available_ops[rank + 1].add((mb, OpType.FORWARD))
            
            # Special case: if this is the last rank, unlock its backward for this microbatch
            if rank == self.pp_size - 1:
                available_ops[rank].add((mb, OpType.BACKWARD))
        
        elif op.op_type == OpType.BACKWARD:
            # After rank r executes Bj, unlock rank r-1's Bj
            if rank > 0:
                available_ops[rank - 1].add((mb, OpType.BACKWARD))
    
    def generate_schedule(self):
        # Generate GPipe schedule using simulation with available operations and time tracking
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Initialize available operations for each rank
        # Use sets of (microbatch_id, op_type) tuples
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free (finish current operation)
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track the currently executing operation and its completion time for each rank
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Rank 0 starts with all forward passes available
        for mb in range(self.num_microbatches):
            available_ops[0].add((mb, OpType.FORWARD))
        
        # Current time in the simulation
        current_time = 0
        
        # Simulation: keep scheduling until all ranks have no available operations
        max_time = 100000  # Safety limit
        
        while current_time < max_time:
            # Step 1: Check which operations just completed at current_time and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    # This operation just completed, unlock next operations
                    completed_op = executing_ops[rank]
                    self._unlock_next_ops(rank, completed_op, available_ops)
                    executing_ops[rank] = None
            
            # Step 2: Check if we're done
            has_work = any(len(ops) > 0 for ops in available_ops.values())
            if not has_work:
                # Check if any rank is still busy
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
            
            # Step 3: Schedule new operations for ranks that are free at current time
            for rank in range(self.pp_size):
                # Only schedule if this rank is free
                if rank_free_time[rank] <= current_time:
                    if len(available_ops[rank]) > 0:
                        # Convert available ops to ScheduleOp objects and sort by priority
                        ops_list = [ScheduleOp(mb, op_type) for mb, op_type in available_ops[rank]]
                        ops_list.sort(key=self._get_priority)
                        
                        # Pick the highest priority operation
                        selected_op = ops_list[0]
                        
                        # Add to this rank's schedule
                        self.schedules[rank].append(selected_op)
                        
                        # Remove from available set
                        available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type))
                        
                        # Update rank free time and track executing operation
                        rank_free_time[rank] = current_time + selected_op.duration
                        executing_ops[rank] = selected_op
                    else:
                        # No available work, add IDLE
                        idle_op = ScheduleOp(None, OpType.IDLE)
                        self.schedules[rank].append(idle_op)
                        rank_free_time[rank] = current_time + idle_op.duration
                        # IDLE operations don't unlock anything
                        executing_ops[rank] = None
            
            # Step 4: Advance time to the next event (next rank becoming free)
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1  # Avoid infinite loop
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")


class DistTrain1F1B(PPSchedule):
    # DistTrain-style disaggregated 1F1B baseline.
    #
    # The constructor's pp_size is the LLM pipeline size. The materialized schedule
    # has pp_size + 2 stages by default:
    #   0: encoder
    #   1..pp_size: LLM pipeline stages
    #   pp_size + 1: generator

    def __init__(self, pp_size: int, num_microbatches: int, include_generator: bool = True):
        self.llm_pp_size = pp_size
        self.encoder_stage = 0
        self.llm_stage_offset = 1
        self.include_generator = include_generator
        self.generator_stage = pp_size + 1 if include_generator else None
        self.pipeline_layout = {
            "encoder_stage": self.encoder_stage,
            "llm_stage_offset": self.llm_stage_offset,
            "llm_pp_size": self.llm_pp_size,
            "has_generator": self.include_generator,
        }
        if self.generator_stage is not None:
            self.pipeline_layout["generator_stage"] = self.generator_stage
        self.forward_duration = 1
        self.backward_duration = 2
        materialized_pp_size = pp_size + 2 if include_generator else pp_size + 1
        super().__init__(pp_size=materialized_pp_size, num_microbatches=num_microbatches)
        self.memory_limits = {stage: self.pp_size - stage for stage in range(self.pp_size)}

    def _make_op(self, stage: int, mb: int, op_type: OpType) -> ScheduleOp:
        forward_ops = {OpType.VEFORWARD, OpType.FORWARD, OpType.GENFORWARD}
        backward_ops = {OpType.VEBACKWARD, OpType.BACKWARD, OpType.GENBACKWARD}
        duration = None
        if op_type in forward_ops:
            duration = self.forward_duration
        elif op_type in backward_ops:
            duration = self.backward_duration
        op = ScheduleOp(mb, op_type, duration=duration)
        op.disttrain_stage = stage
        return op

    def _priority(self, op_key: Tuple[int, OpType]) -> Tuple[int, int]:
        mb, op_type = op_key
        priorities = {
            OpType.VEBACKWARD: 0,
            OpType.BACKWARD: 0,
            OpType.GENBACKWARD: 0,
            OpType.GENFORWARD: 1,
            OpType.FORWARD: 1,
            OpType.VEFORWARD: 1,
        }
        return (priorities.get(op_type, 2), mb)

    def _successors(self, stage: int, op: ScheduleOp) -> List[Tuple[int, Tuple[int, OpType]]]:
        mb = op.microbatch_id
        if op.op_type is OpType.VEFORWARD:
            return [(self.llm_stage_offset, (mb, OpType.FORWARD))]
        if op.op_type is OpType.FORWARD:
            if stage < self.llm_stage_offset + self.llm_pp_size - 1:
                return [(stage + 1, (mb, OpType.FORWARD))]
            if not self.include_generator:
                return [(stage, (mb, OpType.BACKWARD))]
            return [(self.generator_stage, (mb, OpType.GENFORWARD))]
        if op.op_type is OpType.GENFORWARD:
            return [(self.generator_stage, (mb, OpType.GENBACKWARD))]
        if op.op_type is OpType.GENBACKWARD:
            return [(self.llm_stage_offset + self.llm_pp_size - 1, (mb, OpType.BACKWARD))]
        if op.op_type is OpType.BACKWARD:
            if stage > self.llm_stage_offset:
                return [(stage - 1, (mb, OpType.BACKWARD))]
            return [(self.encoder_stage, (mb, OpType.VEBACKWARD))]
        return []

    def generate_schedule(self):
        self.schedules = {stage: [] for stage in range(self.pp_size)}
        available_ops = {stage: set() for stage in range(self.pp_size)}
        rank_free_time = {stage: 0 for stage in range(self.pp_size)}
        executing_ops = {stage: None for stage in range(self.pp_size)}
        rank_memory = {stage: 0 for stage in range(self.pp_size)}
        forward_ops = {OpType.VEFORWARD, OpType.FORWARD, OpType.GENFORWARD}
        backward_ops = {OpType.VEBACKWARD, OpType.BACKWARD, OpType.GENBACKWARD}

        for mb in range(self.num_microbatches):
            available_ops[self.encoder_stage].add((mb, OpType.VEFORWARD))

        current_time = 0
        max_time = 100000

        while current_time < max_time:
            for stage in range(self.pp_size):
                if executing_ops[stage] is not None and rank_free_time[stage] == current_time:
                    completed_op = executing_ops[stage]
                    for next_stage, next_op_key in self._successors(stage, completed_op):
                        available_ops[next_stage].add(next_op_key)
                    executing_ops[stage] = None

            has_work = any(ops for ops in available_ops.values())
            all_free = all(rank_free_time[stage] <= current_time for stage in range(self.pp_size))
            if not has_work and all_free:
                break

            for stage in range(self.pp_size):
                if rank_free_time[stage] > current_time:
                    continue
                if available_ops[stage]:
                    candidate_keys = list(available_ops[stage])
                    if rank_memory[stage] >= self.memory_limits[stage]:
                        candidate_keys = [
                            op_key for op_key in candidate_keys
                            if op_key[1] not in forward_ops
                        ]
                    if not candidate_keys:
                        idle_op = ScheduleOp(None, OpType.IDLE)
                        self.schedules[stage].append(idle_op)
                        rank_free_time[stage] = current_time + idle_op.duration
                        executing_ops[stage] = None
                        continue

                    selected_key = sorted(candidate_keys, key=self._priority)[0]
                    available_ops[stage].remove(selected_key)
                    mb, op_type = selected_key
                    selected_op = self._make_op(stage, mb, op_type)
                    self.schedules[stage].append(selected_op)
                    if op_type in forward_ops:
                        rank_memory[stage] += 1
                    elif op_type in backward_ops:
                        rank_memory[stage] -= 1
                    rank_free_time[stage] = current_time + selected_op.duration
                    executing_ops[stage] = selected_op
                else:
                    idle_op = ScheduleOp(None, OpType.IDLE)
                    self.schedules[stage].append(idle_op)
                    rank_free_time[stage] = current_time + idle_op.duration
                    executing_ops[stage] = None

            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1

        if current_time >= max_time:
            raise RuntimeError("DistTrain1F1B schedule generation exceeded maximum time")


class DistTrainEncoder1F1B(DistTrain1F1B):
    # DistTrain-style 1F1B baseline with encoder and LLM stages only.

    def __init__(self, pp_size: int, num_microbatches: int):
        super().__init__(
            pp_size=pp_size,
            num_microbatches=num_microbatches,
            include_generator=False,
        )


class Schedule1F1B(PPSchedule):
    # 1F1B (One-Forward-One-Backward) pipeline scheduling strategy
    #
    # 1F1B characteristics:
    # 1. Prioritizes backward passes over forward passes when available
    # 2. Alternates between forward and backward to reduce memory usage
    # 3. Same total time as GPipe but with lower peak memory
    
    def _get_priority(self, op: ScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # 1F1B prioritizes: Backward first (by microbatch ID descending), then Forward (by microbatch ID ascending)
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.BACKWARD:
            return (1, op.microbatch_id)
        else:  # FORWARD
            return (0, op.microbatch_id)
    
    def _unlock_next_ops(self, rank: int, op: ScheduleOp, available_ops: dict):
        # Unlock next available operations based on dependencies after executing an operation.
        #
        # Args:
        #     rank: The rank that just executed the operation
        #     op: The operation that was just executed
        #     available_ops: Dictionary of available operations for each rank
        mb = op.microbatch_id
        
        if op.op_type == OpType.FORWARD:
            # After rank r executes Fj, unlock rank r+1's Fj
            if rank < self.pp_size - 1:
                available_ops[rank + 1].add((mb, OpType.FORWARD))
            
            # Special case: if this is the last rank, unlock its backward for this microbatch
            if rank == self.pp_size - 1:
                available_ops[rank].add((mb, OpType.BACKWARD))
        
        elif op.op_type == OpType.BACKWARD:
            # After rank r executes Bj, unlock rank r-1's Bj
            if rank > 0:
                available_ops[rank - 1].add((mb, OpType.BACKWARD))
    
    def generate_schedule(self):
        # Generate 1F1B schedule using simulation with available operations and time tracking
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Initialize available operations for each rank
        # Use sets of (microbatch_id, op_type) tuples
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free (finish current operation)
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track the currently executing operation and its completion time for each rank
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track memory consumption for each rank
        # Forward: +1 memory, Backward: -1 memory
        # Rank i has memory limit of (pp_size - i)
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        # Rank 0 starts with all forward passes available
        for mb in range(self.num_microbatches):
            available_ops[0].add((mb, OpType.FORWARD))
        
        # Current time in the simulation
        current_time = 0
        
        # Simulation: keep scheduling until all ranks have no available operations
        max_time = 100000  # Safety limit
        
        while current_time < max_time:
            # Step 1: Check which operations just completed at current_time and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    # This operation just completed, unlock next operations
                    completed_op = executing_ops[rank]
                    
                    # Update memory: forward +1, backward -1
                    if completed_op.op_type == OpType.FORWARD:
                        rank_memory[rank] += 1
                    elif completed_op.op_type == OpType.BACKWARD:
                        rank_memory[rank] -= 1
                    
                    self._unlock_next_ops(rank, completed_op, available_ops)
                    executing_ops[rank] = None
            
            # Step 2: Check if we're done
            has_work = any(len(ops) > 0 for ops in available_ops.values())
            if not has_work:
                # Check if any rank is still busy
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
            
            # Step 3: Schedule new operations for ranks that are free at current time
            for rank in range(self.pp_size):
                # Only schedule if this rank is free
                if rank_free_time[rank] <= current_time:
                    if len(available_ops[rank]) > 0:
                        # Convert available ops to ScheduleOp objects and sort by priority
                        ops_list = [ScheduleOp(mb, op_type) for mb, op_type in available_ops[rank]]
                        ops_list.sort(key=self._get_priority)
                        
                        # Memory limit: rank i has memory limit of (pp_size - i)
                        memory_limit = self.pp_size - rank
                        
                        # Filter operations based on memory constraint
                        candidate_ops = list(available_ops[rank])
                        if rank_memory[rank] >= memory_limit:
                            # Memory full, filter out Forward operations
                            candidate_ops = [(mb, op_type) for mb, op_type in candidate_ops 
                                           if op_type != OpType.FORWARD]
                        
                        if candidate_ops:
                            # Convert to ScheduleOp objects and sort by priority
                            ops_list = [ScheduleOp(mb, op_type) for mb, op_type in candidate_ops]
                            ops_list.sort(key=self._get_priority)
                            
                            # Execute the highest priority operation
                            selected_op = ops_list[0]
                            self.schedules[rank].append(selected_op)
                            rank_free_time[rank] = current_time + selected_op.duration
                            executing_ops[rank] = selected_op
                            available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type))
                        else:
                            # No executable operations, add IDLE
                            idle_op = ScheduleOp(None, OpType.IDLE)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
                            executing_ops[rank] = None
                    else:
                        # No available work, add IDLE
                        idle_op = ScheduleOp(None, OpType.IDLE)
                        self.schedules[rank].append(idle_op)
                        rank_free_time[rank] = current_time + idle_op.duration
                        executing_ops[rank] = None
            
            # Step 4: Advance time to the next event (next rank becoming free)
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1  # Avoid infinite loop
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")


class Sandwich(PPSchedule):
    # Sandwich pipeline scheduling strategy with Vision Encoder (VE)
    #
    # Sandwich characteristics:
    # 1. VE forward/backward operations execute on ALL ranks simultaneously (as a column)
    #    - Rank i processes microbatch (batch_idx * pp_size + i) for VE operations
    # 2. LLM operations maintain normal pipeline dependencies (rank 0 -> rank n-1 for forward)
    # 3. VE Forward for mb must complete before mb's LLM forward can start on rank 0
    # 4. VE Backward for mb can only start after mb's LLM backward completes on rank 0
    # 5. Requires num_microbatches to be a multiple of pp_size
    # 6. Uses 1F1B memory limiting logic for LLM operations
    
    def __init__(self, pp_size: int, num_microbatches: int):
        if num_microbatches % pp_size != 0:
            raise ValueError(f"num_microbatches ({num_microbatches}) must be a multiple of pp_size ({pp_size})")
        super().__init__(pp_size, num_microbatches)

    def should_recv_emb_from_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.FORWARD

    def should_send_grad_to_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.BACKWARD and current_rank == 0
    
    def _get_priority(self, op: ScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # Priority: VE Forward > Backward > Forward > VE Backward
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.VEFORWARD:
            return (0, op.microbatch_id)
        elif op.op_type == OpType.BACKWARD:
            return (1, op.microbatch_id)
        elif op.op_type == OpType.FORWARD:
            return (2, op.microbatch_id)
        else:  # VEBACKWARD
            return (3, op.microbatch_id)
    
    def _unlock_next_ops(self, rank: int, op: ScheduleOp, available_ops: dict,
                         ve_forward_count: dict, rank0_backward_done: set,
                         ve_backward_batch_available: set):
        # Unlock next available operations when an operation completes.
        #
        # For VE operations: track completion count for column execution
        # For LLM operations: standard pipeline dependencies (like 1F1B)
        mb = op.microbatch_id
        
        if op.op_type == OpType.VEFORWARD:
            # VE forward completed on this rank
            batch_idx = mb // self.pp_size
            ve_forward_count[batch_idx] = ve_forward_count.get(batch_idx, 0) + 1
            
            # When all ranks complete VE forward for this batch, unlock LLM forward on rank 0
            if ve_forward_count[batch_idx] == self.pp_size:
                # Unlock all microbatches in this batch for LLM forward on rank 0
                for r in range(self.pp_size):
                    batch_mb = batch_idx * self.pp_size + r
                    available_ops[0].add((batch_mb, OpType.FORWARD))
        
        elif op.op_type == OpType.FORWARD:
            # After rank r executes Fj, unlock rank r+1's Fj (pipeline dependency)
            if rank < self.pp_size - 1:
                available_ops[rank + 1].add((mb, OpType.FORWARD))
            
            # If this is the last rank, unlock its backward for this microbatch
            if rank == self.pp_size - 1:
                available_ops[rank].add((mb, OpType.BACKWARD))
        
        elif op.op_type == OpType.BACKWARD:
            # After rank r executes Bj, unlock rank r-1's Bj (pipeline dependency)
            if rank > 0:
                available_ops[rank - 1].add((mb, OpType.BACKWARD))
            else:
                # Rank 0: check if all microbatches in batch are done for VE backward
                rank0_backward_done.add(mb)
                batch_idx = mb // self.pp_size
                
                # Check if entire batch completed rank 0 backward
                batch_complete = True
                for r in range(self.pp_size):
                    batch_mb = batch_idx * self.pp_size + r
                    if batch_mb not in rank0_backward_done:
                        batch_complete = False
                        break
                
                if batch_complete:
                    ve_backward_batch_available.add(batch_idx)
    
    def generate_schedule(self):
        # Generate Sandwich schedule with Vision Encoder support and 1F1B memory limiting
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Track available operations for each rank
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track currently executing operations
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track VE forward completion count per batch
        ve_forward_count = {}
        
        # Track which VE forward batches have been started
        ve_forward_batch_started = set()
        
        # Track rank 0 backward completion
        rank0_backward_done = set()
        
        # Track which VE backward batches are available and started
        ve_backward_batch_available = set()
        ve_backward_batch_started = set()
        
        # Track memory consumption for each rank (1F1B logic)
        # Forward: +1 memory, Backward: -1 memory
        # Rank i has memory limit of (pp_size - i)
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        current_time = 0
        max_time = 100000
        num_ve_batches = self.num_microbatches // self.pp_size
        
        while current_time < max_time:
            # Step 1: Check which operations completed and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    completed_op = executing_ops[rank]
                    self._unlock_next_ops(rank, completed_op, available_ops,
                                         ve_forward_count, rank0_backward_done,
                                         ve_backward_batch_available)
                    executing_ops[rank] = None
            
            # Step 2: Schedule new operations
            all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
            scheduled = False
            ve_backward_scheduled = False
            pending_idle_ranks = []
            
            # Priority 1: VE Forward column execution
            if all_ranks_free:
                for batch_idx in range(num_ve_batches):
                    if batch_idx not in ve_forward_batch_started:
                        # Execute VE forward for this batch on all ranks
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            ve_op = ScheduleOp(mb, OpType.VEFORWARD)
                            self.schedules[rank].append(ve_op)
                            rank_free_time[rank] = current_time + ve_op.duration
                            executing_ops[rank] = ve_op
                        
                        ve_forward_batch_started.add(batch_idx)
                        scheduled = True
                        break
            
            # Priority 2: VE Backward column execution
            # Only execute when all ranks are free AND no LLM operations are pending
            if not scheduled and all_ranks_free and len(ve_backward_batch_available) > 0:
                # Check if all ranks have no pending LLM operations
                has_llm_ops = any(
                    any(op_type in [OpType.FORWARD, OpType.BACKWARD] 
                        for mb, op_type in available_ops[r])
                    for r in range(self.pp_size)
                )
                
                if not has_llm_ops:
                    for batch_idx in sorted(ve_backward_batch_available):
                        if batch_idx not in ve_backward_batch_started:
                            ve_backward_batch_started.add(batch_idx)
                            ve_backward_batch_available.remove(batch_idx)
                            
                            # Execute VE backward for this batch on all ranks
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = ScheduleOp(mb, OpType.VEBACKWARD)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            scheduled = True
                            break
            
            # Priority 3: Individual LLM operations (with 1F1B memory limiting logic)
            if not scheduled:
                for rank in range(self.pp_size):
                    if rank_free_time[rank] <= current_time:
                        if len(available_ops[rank]) > 0:
                            # Memory limit: rank i has memory limit of (pp_size - i)
                            memory_limit = self.pp_size - rank
                            
                            # Filter operations based on memory constraint
                            candidate_ops = list(available_ops[rank])
                            if rank_memory[rank] >= memory_limit:
                                # Memory full, filter out Forward operations
                                candidate_ops = [(mb, op_type) for mb, op_type in candidate_ops 
                                               if op_type != OpType.FORWARD]
                            
                            if candidate_ops:
                                # Convert to ScheduleOp objects and sort by priority
                                ops_list = [ScheduleOp(mb, op_type) for mb, op_type in candidate_ops]
                                ops_list.sort(key=self._get_priority)
                                
                                # Execute the highest priority operation
                                selected_op = ops_list[0]
                                self.schedules[rank].append(selected_op)
                                rank_free_time[rank] = current_time + selected_op.duration
                                executing_ops[rank] = selected_op
                                available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type))
                                
                                # Update memory
                                if selected_op.op_type == OpType.FORWARD:
                                    rank_memory[rank] += 1
                                elif selected_op.op_type == OpType.BACKWARD:
                                    rank_memory[rank] -= 1
                                
                                scheduled = True
                            else:
                                # No executable operations, add IDLE
                                idle_op = ScheduleOp(None, OpType.IDLE)
                                self.schedules[rank].append(idle_op)
                                rank_free_time[rank] = current_time + idle_op.duration
                        else:
                            # No available operations, add IDLE
                            idle_op = ScheduleOp(None, OpType.IDLE)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
            
            # Advance time
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1
            
            # Check if done
            if (len(ve_forward_batch_started) >= num_ve_batches and
                len(ve_backward_batch_available) == 0 and
                all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")


class BigMac(Sandwich):
    # Memory-efficient Sandwich with VE forward limit and VE backward prioritization
    #
    # Additional features:
    # 1. VE forward column limit: Limits how many VE forward batches can be executing concurrently
    # 2. VE backward prioritization: When VE backward is available, forces all ranks to sync
    #    by inserting IDLE, then immediately executes VE backward
    
    def __init__(self,
                 pp_size: int,
                 num_microbatches: int,
                 ve_forward_limit: int = 3,
                 check_ve_forward_limit: bool = True):
        super().__init__(pp_size, num_microbatches)
        ve_forward_limit = _resolve_ve_forward_limit(ve_forward_limit)
        if self.pp_size == 1:
            ve_forward_limit = 1  # No point in having VE forward limit with only 1 rank
        self.ve_forward_limit = ve_forward_limit

        if check_ve_forward_limit and self.ve_forward_limit * self.pp_size < self.num_microbatches and self.pp_size > 1:
            warnings.warn(
                f"ve_forward_limit ({self.ve_forward_limit}) is smaller than "
                f"ceil(num_microbatches / pp_size) for num_microbatches="
                f"{self.num_microbatches}, pp_size={self.pp_size}; schedule "
                f"generation will preserve this configuration and expose PP bubbles.",
                stacklevel=2,
            )

    def _get_priority(self, op: ScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # Priority: VE Forward > Backward > Forward > VE Backward
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.VEFORWARD:
            return (0, op.microbatch_id)
        elif op.op_type == OpType.BACKWARD:
            return (2, op.microbatch_id)
        elif op.op_type == OpType.FORWARD:
            return (1, op.microbatch_id)
        else:  # VEBACKWARD
            return (3, op.microbatch_id)
    
    def optimize_communication_ops(self):
        self.align_p2p_with_ve_phases()
        self.make_llm_pp_p2p_adjacent()
        self.remove_null_ops()
        self.reorder_llm_pp_send_recv()

    def align_p2p_with_ve_phases(self):
        # Defer blocking sends whose receiver runs after a VE collective.
        #
        # The compute plan can finish B9 on rank 1 before VB0, while rank 0
        # schedules B9 after VB0 and the next VF wave. Sending B9 immediately
        # would block rank 1 before it can join rank 0's VE scatter. Place the
        # send in its receiver's phase, retaining the gradient in the executor
        # until then. VF/VB order and the bounded encoder activation window are
        # unchanged. This operates on the unfused, non-interleaved PP plan.
        phase_ends = {OpType.VESENDFORWARD, OpType.VEBACKWARD}
        recv_phases = {}
        reference_phases = None
        for schedule in self.schedules_with_comm.values():
            phase = 0
            phases = []
            for op in schedule:
                if op.op_type in phase_ends:
                    phases.append((op.op_type, op.microbatch_id // self.pp_size))
                    phase += 1
                if op.op_type in LLM_RECV_OP_TYPES and not _is_pp_boundary_noop(op, self.pp_size, 1):
                    recv_phases[_communication_key(op, self.pp_size)] = phase
            if reference_phases is not None and phases != reference_phases:
                raise RuntimeError("BigMac PP ranks disagree on VE collective order")
            reference_phases = phases

        for rank, schedule in self.schedules_with_comm.items():
            phase = 0
            deferred = {}
            remaining = []
            for op in schedule:
                if op.op_type in phase_ends:
                    phase += 1
                if op.op_type in LLM_SEND_OP_TYPES and not _is_pp_boundary_noop(op, self.pp_size, 1):
                    target = recv_phases.get(_matching_communication_key(op, self.pp_size))
                    if target is None or target < phase:
                        raise RuntimeError(f"BigMac PP cannot place {op} in its receiver's VE phase")
                    if target > phase:
                        deferred.setdefault(target, []).append(op)
                        continue
                remaining.append(op)

            phase = 0
            aligned = []
            for op in remaining:
                aligned.append(op)
                if op.op_type in phase_ends:
                    phase += 1
                    aligned.extend(deferred.pop(phase, []))
            if deferred:
                raise RuntimeError("BigMac PP has sends beyond the final VE phase")
            self.schedules_with_comm[rank] = aligned
    
    def make_llm_pp_p2p_adjacent(self):
        # make llm pp send and recv operations adjacent to each other.
        self.make_send_next_to_recv()
        self.make_recv_next_to_send()
    
    def make_send_next_to_recv(self):
        # make p2p send operation next to p2p recv operation.
        def find_next_recv_ops(schedules, start):
            for idx in range(start+1, len(schedules)):
                # Moving a receive across VF/VB (or their collectives) can
                # wait for a peer that cannot send until that phase completes.
                if schedules[idx].op_type not in NULL_OP_TYPES + LLM_SEND_RECV_OP_TYPES:
                    return -1
                if schedules[idx].op_type in LLM_RECV_OP_TYPES:
                    return idx
            return -1

        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in LLM_SEND_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_idx = find_next_recv_ops(schedules, j)
                if next_recv_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv = schedules[next_recv_idx]
                # move recv forward ahead
                optimized_schedules.append(schedules[j])
                optimized_schedules.append(next_recv)
                optimized_schedules.extend(schedules[j+1:next_recv_idx])
                j = next_recv_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules
    
    def make_recv_next_to_send(self):
        # make p2p recv operation next to p2p send operation.
        def find_next_send_ops(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type not in NULL_OP_TYPES + LLM_SEND_RECV_OP_TYPES:
                    return -1
                if schedules[idx].op_type in LLM_SEND_OP_TYPES:
                    return idx
            return -1
        
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in LLM_RECV_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_send_idx = find_next_send_ops(schedules, j)
                if next_send_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_send = schedules[next_send_idx]
                # move recv forward ahead
                optimized_schedules.append(schedules[j])
                optimized_schedules.append(next_send)
                optimized_schedules.extend(schedules[j+1:next_send_idx])
                j = next_send_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules

    def generate_schedule(self):
        if self.pp_size > 1:
            # Generate schedule with VE forward limit and VE backward prioritization
            self.schedules = {rank: [] for rank in range(self.pp_size)}
            
            # Track available operations for each rank
            available_ops = {rank: set() for rank in range(self.pp_size)}
            
            # Track when each rank will be free
            rank_free_time = {rank: 0 for rank in range(self.pp_size)}
            
            # Track currently executing operations
            executing_ops = {rank: None for rank in range(self.pp_size)}
            
            # Track VE forward completion count per batch
            ve_forward_count = {}
            
            # Track which VE forward batches have been started
            ve_forward_batch_started = set()
            
            # Track rank 0 backward completion
            rank0_backward_done = set()
            
            # Track which VE backward batches are available and started
            ve_backward_batch_available = set()
            ve_backward_batch_started = set()
            
            # Track how many VE backward batches have been executed
            ve_backward_executed = 0
            
            # Track memory consumption for each rank (1F1B logic)
            rank_memory = {rank: 0 for rank in range(self.pp_size)}
            
            current_time = 0
            max_time = 100000
            num_ve_batches = self.num_microbatches // self.pp_size
            
            while current_time < max_time:
                # Step 1: Check which operations completed and unlock next ops
                for rank in range(self.pp_size):
                    if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                        completed_op = executing_ops[rank]
                        self._unlock_next_ops(rank, completed_op, available_ops,
                                            ve_forward_count, rank0_backward_done,
                                            ve_backward_batch_available)
                        executing_ops[rank] = None
                        
                # Step 2: Schedule new operations
                all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                scheduled = False
                ve_backward_scheduled = False
                pending_idle_ranks = []
                
                # Calculate in-flight VE forward batches
                ve_forward_in_flight = len(ve_forward_batch_started) - len(ve_backward_batch_started)

                def should_prioritize_ve_backward() -> bool:
                    if len(ve_backward_batch_available) == 0:
                        return False

                    has_pending_ops = any(len(available_ops[r]) > 0 for r in range(self.pp_size))
                    has_unstarted_ve_forward = len(ve_forward_batch_started) < num_ve_batches
                    if not has_unstarted_ve_forward:
                        return not has_pending_ops

                    if ve_forward_in_flight < self.ve_forward_limit:
                        return False

                    oldest_batch = min(ve_backward_batch_available)
                    release_mb = (oldest_batch + self.ve_forward_limit - 1) * self.pp_size
                    return release_mb in rank0_backward_done
                
                # Priority 1: VE Forward column execution (with limit)
                if not scheduled and all_ranks_free and ve_forward_in_flight < self.ve_forward_limit:
                    for batch_idx in range(num_ve_batches):
                        if batch_idx not in ve_forward_batch_started:
                            # Execute VE forward for this batch on all ranks
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = ScheduleOp(mb, OpType.VEFORWARD)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            ve_forward_batch_started.add(batch_idx)
                            scheduled = True
                            break
                
                # Priority 2: VE Backward prioritization when the VE window is full.
                if not scheduled and should_prioritize_ve_backward():
                    if not all_ranks_free:
                        max_free_time = max(rank_free_time.values())
                        remaining_time = max_free_time - current_time

                        for rank in range(self.pp_size):
                            if rank_free_time[rank] <= current_time:
                                scheduled_short_op = False

                                if len(available_ops[rank]) > 0:
                                    for mb, op_type in available_ops[rank]:
                                        temp_op = ScheduleOp(mb, op_type)
                                        if temp_op.duration <= remaining_time:
                                            can_execute = True
                                            if temp_op.op_type == OpType.FORWARD:
                                                memory_limit = self.pp_size - rank
                                                if rank_memory[rank] >= memory_limit:
                                                    can_execute = False

                                            if can_execute:
                                                self.schedules[rank].append(temp_op)
                                                rank_free_time[rank] = current_time + temp_op.duration
                                                executing_ops[rank] = temp_op
                                                available_ops[rank].remove((mb, op_type))

                                                if temp_op.op_type == OpType.FORWARD:
                                                    rank_memory[rank] += 1
                                                elif temp_op.op_type == OpType.BACKWARD:
                                                    rank_memory[rank] -= 1

                                                scheduled_short_op = True
                                                break

                                if not scheduled_short_op:
                                    idle_op = ScheduleOp(None, OpType.IDLE)
                                    idle_op.duration = 1
                                    self.schedules[rank].append(idle_op)
                                    rank_free_time[rank] = current_time + idle_op.duration

                        scheduled = True
                        ve_backward_scheduled = True
                    else:
                        for batch_idx in sorted(ve_backward_batch_available):
                            if batch_idx not in ve_backward_batch_started:
                                ve_backward_batch_started.add(batch_idx)
                                ve_backward_batch_available.remove(batch_idx)
                                ve_backward_executed += 1

                                for rank in range(self.pp_size):
                                    mb = batch_idx * self.pp_size + rank
                                    ve_op = ScheduleOp(mb, OpType.VEBACKWARD)
                                    self.schedules[rank].append(ve_op)
                                    rank_free_time[rank] = current_time + ve_op.duration
                                    executing_ops[rank] = ve_op

                                scheduled = True
                                ve_backward_scheduled = True
                                break

                # Priority 3: Individual LLM operations (with 1F1B memory limiting logic)
                if not scheduled:
                    for rank in range(self.pp_size):
                        if rank_free_time[rank] <= current_time:
                            if len(available_ops[rank]) > 0:
                                # Memory limit: rank i has memory limit of (pp_size - i)
                                memory_limit = self.pp_size - rank
                                
                                # Filter operations based on memory constraint
                                candidate_ops = list(available_ops[rank])
                                if rank_memory[rank] >= memory_limit:
                                    # Memory full, filter out Forward operations
                                    candidate_ops = [(mb, op_type) for mb, op_type in candidate_ops 
                                                if op_type != OpType.FORWARD]
                                
                                if candidate_ops:
                                    # Convert to ScheduleOp objects and sort by priority
                                    ops_list = [ScheduleOp(mb, op_type) for mb, op_type in candidate_ops]
                                    ops_list.sort(key=self._get_priority)
                                    
                                    # Execute the highest priority operation
                                    selected_op = ops_list[0]
                                    self.schedules[rank].append(selected_op)
                                    rank_free_time[rank] = current_time + selected_op.duration
                                    executing_ops[rank] = selected_op
                                    available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type))
                                    
                                    # Update memory
                                    if selected_op.op_type == OpType.FORWARD:
                                        rank_memory[rank] += 1
                                    elif selected_op.op_type == OpType.BACKWARD:
                                        rank_memory[rank] -= 1
                                    
                                    scheduled = True
                                else:
                                    pending_idle_ranks.append(rank)
                            else:
                                pending_idle_ranks.append(rank)

                if pending_idle_ranks and not ve_backward_scheduled:
                    for rank in pending_idle_ranks:
                        idle_op = ScheduleOp(None, OpType.IDLE)
                        self.schedules[rank].append(idle_op)
                        rank_free_time[rank] = current_time + idle_op.duration
                
                # Advance time
                next_event_time = min(rank_free_time.values())
                if next_event_time > current_time:
                    current_time = next_event_time
                else:
                    current_time += 1
                
                # Check if done
                if (len(ve_forward_batch_started) >= num_ve_batches and
                    len(ve_backward_batch_started) >= num_ve_batches and
                    all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                    all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                    if all_free:
                        break
            
            if current_time >= max_time:
                raise RuntimeError("Schedule generation exceeded maximum time")
        elif self.pp_size == 1:
            # Generate schedule with VE forward limit and VE backward prioritization
            self.schedules = {rank: [] for rank in range(self.pp_size)}
            
            # Track available operations for each rank
            available_ops = {rank: set() for rank in range(self.pp_size)}
            
            # Track when each rank will be free
            rank_free_time = {rank: 0 for rank in range(self.pp_size)}
            
            # Track currently executing operations
            executing_ops = {rank: None for rank in range(self.pp_size)}
            
            # Track VE forward completion count per batch
            ve_forward_count = {}
            
            # Track which VE forward batches have been started
            ve_forward_batch_started = set()
            
            # Track which VE forward batches have completed (all LLM ops done)
            ve_forward_batch_completed = set()
            
            # Track rank 0 backward completion
            rank0_backward_done = set()
            
            # Track which VE backward batches are available and started
            ve_backward_batch_available = set()
            ve_backward_batch_started = set()
            
            # Track how many VE backward batches have been executed
            ve_backward_executed = 0
            
            # Track memory consumption for each rank (1F1B logic)
            rank_memory = {rank: 0 for rank in range(self.pp_size)}
            
            current_time = 0
            max_time = 100000
            num_ve_batches = self.num_microbatches // self.pp_size
            
            while current_time < max_time:
                # Step 1: Check which operations completed and unlock next ops
                for rank in range(self.pp_size):
                    if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                        completed_op = executing_ops[rank]
                        self._unlock_next_ops(rank, completed_op, available_ops,
                                            ve_forward_count, rank0_backward_done,
                                            ve_backward_batch_available)
                        executing_ops[rank] = None
                        
                        # Check if a VE forward batch is now complete (all LLM ops done)
                        if completed_op.op_type == OpType.BACKWARD and rank == 0:
                            mb = completed_op.microbatch_id
                            batch_idx = mb // self.pp_size
                            # Check if all microbatches in this batch have completed backward on rank 0
                            if all(batch_idx * self.pp_size + r in rank0_backward_done for r in range(self.pp_size)):
                                if batch_idx not in ve_forward_batch_completed:
                                    ve_forward_batch_completed.add(batch_idx)
                
                # Step 2: Schedule new operations
                all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                scheduled = False
                ve_backward_scheduled = False
                pending_idle_ranks = []
                
                # Calculate in-flight VE forward batches
                ve_forward_in_flight = len(ve_forward_batch_started) - len(ve_forward_batch_completed)
                
                # Priority 1: VE Backward prioritization
                if not scheduled and len(ve_backward_batch_available) > 0:
                    has_pending_ops = any(len(available_ops[r]) > 0 for r in range(self.pp_size))
                    if ve_backward_executed < num_ve_batches - 2 or not has_pending_ops:
                        if not all_ranks_free:
                            max_free_time = max(rank_free_time.values())
                            remaining_time = max_free_time - current_time
                            
                            for rank in range(self.pp_size):
                                if rank_free_time[rank] <= current_time:
                                    scheduled_short_op = False
                                    
                                    if len(available_ops[rank]) > 0:
                                        for mb, op_type in available_ops[rank]:
                                            temp_op = ScheduleOp(mb, op_type)
                                            if temp_op.duration <= remaining_time:
                                                can_execute = True
                                                if temp_op.op_type == OpType.FORWARD:
                                                    memory_limit = self.pp_size - rank
                                                    if rank_memory[rank] >= memory_limit:
                                                        can_execute = False
                                                
                                                if can_execute:
                                                    self.schedules[rank].append(temp_op)
                                                    rank_free_time[rank] = current_time + temp_op.duration
                                                    executing_ops[rank] = temp_op
                                                    available_ops[rank].remove((mb, op_type))
                                                    
                                                    if temp_op.op_type == OpType.FORWARD:
                                                        rank_memory[rank] += 1
                                                    elif temp_op.op_type == OpType.BACKWARD:
                                                        rank_memory[rank] -= 1
                                                    
                                                    scheduled_short_op = True
                                                    break
                                    
                                    if not scheduled_short_op:
                                        idle_op = ScheduleOp(None, OpType.IDLE)
                                        idle_op.duration = 1
                                        self.schedules[rank].append(idle_op)
                                        rank_free_time[rank] = current_time + idle_op.duration
                            
                            scheduled = True
                            ve_backward_scheduled = True
                        else:
                            for batch_idx in sorted(ve_backward_batch_available):
                                if batch_idx not in ve_backward_batch_started:
                                    ve_backward_batch_started.add(batch_idx)
                                    ve_backward_batch_available.remove(batch_idx)
                                    ve_backward_executed += 1
                                    
                                    for rank in range(self.pp_size):
                                        mb = batch_idx * self.pp_size + rank
                                        ve_op = ScheduleOp(mb, OpType.VEBACKWARD)
                                        self.schedules[rank].append(ve_op)
                                        rank_free_time[rank] = current_time + ve_op.duration
                                        executing_ops[rank] = ve_op
                                    
                                    scheduled = True
                                    ve_backward_scheduled = True
                                    break

                # Priority 2: VE Forward column execution (with limit)
                if not scheduled and all_ranks_free and ve_forward_in_flight < self.ve_forward_limit:
                    for batch_idx in range(num_ve_batches):
                        if batch_idx not in ve_forward_batch_started:
                            # Execute VE forward for this batch on all ranks
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = ScheduleOp(mb, OpType.VEFORWARD)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            ve_forward_batch_started.add(batch_idx)
                            scheduled = True
                            break
                
                # Priority 3: Individual LLM operations (with 1F1B memory limiting logic)
                if not scheduled:
                    for rank in range(self.pp_size):
                        if rank_free_time[rank] <= current_time:
                            if len(available_ops[rank]) > 0:
                                # Memory limit: rank i has memory limit of (pp_size - i)
                                memory_limit = self.pp_size - rank
                                
                                # Filter operations based on memory constraint
                                candidate_ops = list(available_ops[rank])
                                if rank_memory[rank] >= memory_limit:
                                    # Memory full, filter out Forward operations
                                    candidate_ops = [(mb, op_type) for mb, op_type in candidate_ops 
                                                if op_type != OpType.FORWARD]
                                
                                if candidate_ops:
                                    # Convert to ScheduleOp objects and sort by priority
                                    ops_list = [ScheduleOp(mb, op_type) for mb, op_type in candidate_ops]
                                    ops_list.sort(key=self._get_priority)
                                    
                                    # Execute the highest priority operation
                                    selected_op = ops_list[0]
                                    self.schedules[rank].append(selected_op)
                                    rank_free_time[rank] = current_time + selected_op.duration
                                    executing_ops[rank] = selected_op
                                    available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type))
                                    
                                    # Update memory
                                    if selected_op.op_type == OpType.FORWARD:
                                        rank_memory[rank] += 1
                                    elif selected_op.op_type == OpType.BACKWARD:
                                        rank_memory[rank] -= 1
                                    
                                    scheduled = True
                                else:
                                    pending_idle_ranks.append(rank)
                            else:
                                pending_idle_ranks.append(rank)

                if pending_idle_ranks and not ve_backward_scheduled:
                    for rank in pending_idle_ranks:
                        idle_op = ScheduleOp(None, OpType.IDLE)
                        self.schedules[rank].append(idle_op)
                        rank_free_time[rank] = current_time + idle_op.duration
                
                # Advance time
                next_event_time = min(rank_free_time.values())
                if next_event_time > current_time:
                    current_time = next_event_time
                else:
                    current_time += 1
                
                # Check if done
                if (len(ve_forward_batch_started) >= num_ve_batches and
                    len(ve_backward_batch_started) >= num_ve_batches and
                    all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                    all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                    if all_free:
                        break
            
            if current_time >= max_time:
                raise RuntimeError("Schedule generation exceeded maximum time")
        else:
            raise ValueError("pp_size must be at least 1")


# Example usage


class VppScheduleOp(ScheduleOp):
    # Schedule operation representing what should be executed at a given step

    def __init__(
        self,
        microbatch_id: Optional[int],
        op_type: OpType,
        chunk_id: int,
        duration: Optional[float] = None,
    ):
        self.microbatch_id = microbatch_id
        self.op_type = op_type
        self.chunk_id = chunk_id
        self.duration = duration if duration is not None else _default_duration(op_type)
    
    def __repr__(self):
        if self.op_type == OpType.IDLE:
            return "IDLE"
        return f"{self.op_type.value}{self.microbatch_id}-C{self.chunk_id}"
    
    def __str__(self):
        return self.__repr__()

class VPPSchedule(PPSchedule):
    # Base class for pipeline parallelism scheduling
    
    def __init__(self, pp_size: int, vpp_size: int, num_microbatches: int):
        # Initialize pipeline scheduler
        #
        # Args:
        #     pp_size: Pipeline parallel size (number of stages)
        #     vpp_size: Virtual pipeline parallel size (number of model chunks)
        #     num_microbatches: Total number of microbatches
        super().__init__(pp_size, num_microbatches)
        self.vpp_size = vpp_size
        assert self.num_microbatches % self.pp_size == 0, (
            "num_microbatches must be a multiple of pp_size, got "
            f"num_microbatches={self.num_microbatches}, pp_size={self.pp_size}"
        )
        
    def should_send_forward(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.FORWARD and not (current_rank == self.pp_size - 1 and op.chunk_id == self.vpp_size - 1)
    
    def should_recv_backward(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.BACKWARD and not (current_rank == self.pp_size - 1 and op.chunk_id == self.vpp_size - 1)
    
    def optimize_communication_ops(self):
        self.remove_null_ops()
        self.pin_comm_ops()
        self.vpp_p2p_overlap()
        self.reorder_llm_pp_send_recv()
    
    def fuse_comm_ops(self):
        # fuse comm ops to achieve p2p communication overlap.
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type in LLM_SEND_OP_TYPES and j+1 < len(schedules) and schedules[j+1].op_type in LLM_RECV_OP_TYPES:
                    optimized_schedules.append(FusedScheduleOp(schedules[j], schedules[j+1]))
                    j += 2
                else:
                    optimized_schedules.append(schedules[j])
                    j += 1
            self.schedules_with_comm[i] = optimized_schedules
    
    def pin_comm_ops(self):
        # some comm ops should never be moved, pin them.

        def find_first_backward(schedules, start):
            for idx in range(start, len(schedules)):
                if schedules[idx].op_type is OpType.BACKWARD:
                    return idx
            return -1
        
        def find_last_forward(schedules, end):
            for idx in range(end-1, -1, -1):
                if schedules[idx].op_type is OpType.FORWARD:
                    return idx
            return -1
        
        def count_forward_num(schedules, start, end):
            cnt = 0
            for idx in range(start, end):
                if schedules[idx].op_type is OpType.FORWARD:
                    cnt += 1
            return cnt

        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            memory_limit = self.memory_limits[i]
            first_backward_idx = find_first_backward(schedules, 0)
            assert first_backward_idx != -1, "no backward op found"
            last_forward_idx = find_last_forward(schedules, first_backward_idx)
            assert last_forward_idx != -1, "no forward op found"
            forward_num = count_forward_num(schedules, 0, last_forward_idx + 1)
            if forward_num < memory_limit:
                for j in range(last_forward_idx + 1, first_backward_idx):
                    if schedules[j].op_type is OpType.RECVBACKWARD:
                        schedules[j].pinned = True
                        break
            
                
    
    def move_ahead_recv_forward(self):
        # backward i, send backward i, recv forward j, forward j => recv forward j, backward i, send backward i, forward j

        def find_next_recv_forward(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type is OpType.RECVFORWARD:
                    return idx
            return -1

        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type is not OpType.BACKWARD:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_forward_idx = find_next_recv_forward(schedules, j)
                if next_recv_forward_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_forward = schedules[next_recv_forward_idx]
                # move recv forward ahead
                optimized_schedules.append(next_recv_forward)
                optimized_schedules.extend(schedules[j:next_recv_forward_idx])
                j = next_recv_forward_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules
    
    def move_ahead_recv_backward(self):
        # forward i, send forward i, recv backward j, backward j => recv backward j, forward i, send forward i, backward j
        def find_next_recv_backward(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type is OpType.FORWARD:
                    return -1
                if schedules[idx].op_type is OpType.RECVBACKWARD:
                    return idx
            return -1

        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type is not OpType.FORWARD:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_backward_idx = find_next_recv_backward(schedules, j)
                if next_recv_backward_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_backward = schedules[next_recv_backward_idx]
                if getattr(next_recv_backward, 'pinned', False):
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                # move recv backward ahead
                optimized_schedules.append(next_recv_backward)
                optimized_schedules.extend(schedules[j:next_recv_backward_idx])
                j = next_recv_backward_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules

    
    def vpp_p2p_overlap(self):
        # Overlap p2p communication and computation for vpp.
        self.move_ahead_recv_forward()
        self.move_ahead_recv_backward()


class VPP1F1BSchedule(VPPSchedule):
    # VPP 1F1B (One-Forward-One-Backward) pipeline scheduling strategy with Virtual Pipeline Parallelism
    #
    # VPP 1F1B characteristics:
    # 1. Uses VppScheduleOp with chunk_id for virtual pipeline parallelism
    # 2. Starts with chunk_id=0 forwards for microbatches 0 to pp_size on rank 0
    # 3. Forward on rank i unlocks same chunk_id forward on rank i+1
    # 4. Forward on rank pp_size-1 unlocks next chunk_id forward on rank 0, or backward if last chunk
    # 5. Backward follows reverse pattern
    # 6. Memory limit per rank: pp_size + 1 + (pp_size - rank - 1) * vpp_size
    # 7. Prioritizes operations with higher chunk_id
    
    def _get_priority(self, op: VppScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # VPP 1F1B prioritizes: Higher chunk_id first, then Backward over Forward
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority

        # Prioritize higher chunk_id (negate for descending order)
        # Then backward over forward
        if op.op_type == OpType.BACKWARD:
            return (1, -op.chunk_id, op.microbatch_id)
        else:  # FORWARD
            return (0, -op.chunk_id, op.microbatch_id)
    
    def _unlock_next_ops(self, rank: int, op: VppScheduleOp, available_ops: dict):
        # Unlock next available operations based on dependencies after executing an operation.
        #
        # Args:
        #     rank: The rank that just executed the operation
        #     op: The operation that was just executed
        #     available_ops: Dictionary of available operations for each rank
        mb = op.microbatch_id
        chunk_id = op.chunk_id
        
        if op.op_type == OpType.FORWARD:
            # After rank r executes F(mb, chunk), unlock rank r+1's F(mb, chunk)
            if rank < self.pp_size - 1:
                available_ops[rank + 1].add((mb, OpType.FORWARD, chunk_id))
            else:
                # rank == pp_size - 1
                # Check if we should unlock next chunk or backward
                if chunk_id + 1 < self.vpp_size:
                    # Unlock rank 0's next chunk forward
                    available_ops[0].add((mb, OpType.FORWARD, chunk_id + 1))
                else:
                    # Last chunk, unlock backward on rank pp_size-1
                    available_ops[rank].add((mb, OpType.BACKWARD, chunk_id))
        
        elif op.op_type == OpType.BACKWARD:
            # After rank r executes B(mb, chunk)
            if rank > 0:
                if chunk_id == self.vpp_size - 1 and mb == 0:
                    self.memory_limits[rank-1] += 1
                # Unlock rank r-1's B(mb, chunk)
                available_ops[rank - 1].add((mb, OpType.BACKWARD, chunk_id))
            else:
                # rank == 0
                # Check if we should unlock previous chunk backward
                if chunk_id > 0:
                    # Unlock rank pp_size-1's previous chunk backward
                    available_ops[self.pp_size - 1].add((mb, OpType.BACKWARD, chunk_id - 1))
    
    def generate_schedule(self):
        # Generate VPP 1F1B schedule using simulation with available operations and time tracking
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Initialize available operations for each rank
        # Use sets of (microbatch_id, op_type, chunk_id) tuples
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free (finish current operation)
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track the currently executing operation and its completion time for each rank
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track memory consumption for each rank
        # Forward on chunk: +1 memory, Backward on chunk: -1 memory
        # Rank i has memory limit of: pp_size + 1 + (pp_size - i - 1) * vpp_size
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        # Pre-compute memory limits for each rank
        self.memory_limits = {rank: (self.pp_size) * (self.vpp_size - 1) + (self.pp_size - rank - 1) * 2  + 1
                        for rank in range(self.pp_size)}
        for rank in self.memory_limits:
            if rank != self.pp_size - 1:
                self.memory_limits[rank] -= 1
        
        # Rank 0 starts with chunk_id=0 forwards for microbatches 0 to num_microbatches-1
        for mb in range(self.num_microbatches):
            available_ops[0].add((mb, OpType.FORWARD, 0))
        
        # Current time in the simulation
        current_time = 0
        
        # Simulation: keep scheduling until all ranks have no available operations
        max_time = 100000  # Safety limit
        
        while current_time < max_time:
            # Step 1: Check which operations just completed at current_time and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    # This operation just completed, unlock next operations
                    completed_op = executing_ops[rank]
                    
                    # Update memory: forward +1, backward -1
                    if completed_op.op_type == OpType.FORWARD:
                        rank_memory[rank] += 1
                    elif completed_op.op_type == OpType.BACKWARD:
                        rank_memory[rank] -= 1
                    
                    self._unlock_next_ops(rank, completed_op, available_ops)
                    executing_ops[rank] = None
            
            # Step 2: Check if we're done
            has_work = any(len(ops) > 0 for ops in available_ops.values())
            if not has_work:
                # Check if any rank is still busy
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
            
            # Step 3: Schedule new operations for ranks that are free at current time
            for rank in range(self.pp_size):
                # Only schedule if this rank is free
                if rank_free_time[rank] <= current_time:
                    if len(available_ops[rank]) > 0:
                        # Get pre-computed memory limit
                        memory_limit = self.memory_limits[rank]
                        
                        # Filter operations based on memory constraint
                        candidate_ops = list(available_ops[rank])
                        if rank_memory[rank] >= memory_limit:
                            # Memory full, filter out Forward operations
                            candidate_ops = [(mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops 
                                           if op_type != OpType.FORWARD]
                        
                        if candidate_ops:
                            # Convert to VppScheduleOp objects and sort by priority
                            ops_list = [VppScheduleOp(mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops]
                            ops_list.sort(key=self._get_priority)
                            
                            # Execute the highest priority operation
                            selected_op = ops_list[0]
                            self.schedules[rank].append(selected_op)
                            rank_free_time[rank] = current_time + selected_op.duration
                            executing_ops[rank] = selected_op
                            available_ops[rank].remove((selected_op.microbatch_id, selected_op.op_type, selected_op.chunk_id))
                        else:
                            # No executable operations, add IDLE
                            idle_op = VppScheduleOp(None, OpType.IDLE, 0)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
                            executing_ops[rank] = None
                    else:
                        # No available work, add IDLE
                        idle_op = VppScheduleOp(None, OpType.IDLE, 0)
                        self.schedules[rank].append(idle_op)
                        rank_free_time[rank] = current_time + idle_op.duration
                        executing_ops[rank] = None
            
            # Step 4: Advance time to the next event (next rank becoming free)
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1  # Avoid infinite loop
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")

class SandwichVPP(VPPSchedule):
    # Sandwich VPP pipeline scheduling strategy with Vision Encoder (VE) and Virtual Pipeline Parallelism
    #
    # SandwichVPP characteristics:
    # 1. VE forward/backward operations execute on ALL ranks simultaneously (as a column)
    #    - Rank i processes microbatch (batch_idx * pp_size + i) for VE operations
    # 2. LLM operations use VPP chunking with chunk_id
    # 3. VE Forward for batch must complete before batch's LLM chunk 0 forward can start on rank 0
    # 4. VE Backward for batch can only start after batch's LLM final chunk backward completes on rank 0
    # 5. Requires num_microbatches to be a multiple of pp_size
    # 6. Combines VPP 1F1B memory limiting logic for LLM operations
    
    def __init__(self, pp_size, vpp_size, num_microbatches):
        if num_microbatches % pp_size != 0:
            raise ValueError(f"num_microbatches ({num_microbatches}) must be a multiple of pp_size ({pp_size})")
        super().__init__(pp_size, vpp_size, num_microbatches)

    def should_recv_emb_from_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.FORWARD and current_rank == 0 and op.chunk_id == 0

    def should_send_grad_to_ve(self, op: ScheduleOp, current_rank: int) -> bool:
        return op.op_type == OpType.BACKWARD and current_rank == 0 and op.chunk_id == 0
    
    def _get_priority(self, op: VppScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # Priority: VE Forward > Backward (higher chunk) > Forward (higher chunk) > VE Backward
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.VEFORWARD:
            return (0, op.microbatch_id)
        elif op.op_type == OpType.BACKWARD:
            # Prioritize higher chunk_id for backward
            return (3, -op.chunk_id, op.microbatch_id)
        elif op.op_type == OpType.FORWARD:
            # Prioritize higher chunk_id for forward (but lower than backward)
            return (2, op.microbatch_id, -op.chunk_id)
        else:  # VEBACKWARD
            return (1, -op.microbatch_id)
    
    def _unlock_next_ops(self, rank: int, op: VppScheduleOp, available_ops: dict,
                         ve_forward_count: dict, rank0_backward_done: dict,
                         ve_backward_batch_available: set):
        # Unlock next available operations when an operation completes.
        #
        # For VE operations: track completion count for column execution
        # For LLM operations: VPP pipeline dependencies with chunk_id
        mb = op.microbatch_id
        
        if op.op_type == OpType.VEFORWARD:
            # VE forward completed on this rank
            batch_idx = mb // self.pp_size
            ve_forward_count[batch_idx] = ve_forward_count.get(batch_idx, 0) + 1
            
            # When all ranks complete VE forward for this batch, unlock LLM chunk 0 forward on rank 0
            if ve_forward_count[batch_idx] == self.pp_size:
                # Unlock all microbatches in this batch for LLM chunk 0 forward on rank 0
                for r in range(self.pp_size):
                    batch_mb = batch_idx * self.pp_size + r
                    available_ops[0].add((batch_mb, OpType.FORWARD, 0))
        
        elif op.op_type == OpType.FORWARD:
            chunk_id = op.chunk_id
            # After rank r executes F(mb, chunk), unlock rank r+1's F(mb, chunk)
            if rank < self.pp_size - 1:
                available_ops[rank + 1].add((mb, OpType.FORWARD, chunk_id))
            else:
                # rank == pp_size - 1
                # Check if we should unlock next chunk or backward
                if chunk_id + 1 < self.vpp_size:
                    # Unlock rank 0's next chunk forward
                    available_ops[0].add((mb, OpType.FORWARD, chunk_id + 1))
                else:
                    # Last chunk, unlock backward on rank pp_size-1
                    available_ops[rank].add((mb, OpType.BACKWARD, chunk_id))
        
        elif op.op_type == OpType.BACKWARD:
            chunk_id = op.chunk_id
            # After rank r executes B(mb, chunk)
            if rank > 0:
                if chunk_id == self.vpp_size - 1 and mb == 0:
                    self.memory_limits[rank-1] += 1
                # Unlock rank r-1's B(mb, chunk)
                available_ops[rank - 1].add((mb, OpType.BACKWARD, chunk_id))
            else:
                # rank == 0
                # Check if we should unlock previous chunk backward
                if chunk_id > 0:
                    # Unlock rank pp_size-1's previous chunk backward
                    available_ops[self.pp_size - 1].add((mb, OpType.BACKWARD, chunk_id - 1))
                else:
                    # chunk_id == 0, LLM backward complete for this mb
                    # Track completion for VE backward
                    batch_idx = mb // self.pp_size
                    if batch_idx not in rank0_backward_done:
                        rank0_backward_done[batch_idx] = set()
                    rank0_backward_done[batch_idx].add(mb)
                    
                    # Check if entire batch completed rank 0 backward (chunk 0)
                    if len(rank0_backward_done[batch_idx]) == self.pp_size:
                        ve_backward_batch_available.add(batch_idx)
    
    def generate_schedule(self):
        # Generate SandwichVPP schedule with Vision Encoder support and VPP 1F1B logic
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Track available operations for each rank
        # Use sets of (microbatch_id, op_type, chunk_id) tuples for LLM ops
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track currently executing operations
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track VE forward completion count per batch
        ve_forward_count = {}
        
        # Track which VE forward batches have been started
        ve_forward_batch_started = set()
        
        # Track rank 0 backward completion per batch (dict of sets)
        rank0_backward_done = {}
        
        # Track which VE backward batches are available and started
        ve_backward_batch_available = set()
        ve_backward_batch_started = set()
        
        # Track memory consumption for each rank (VPP 1F1B logic)
        # Forward on chunk: +1 memory, Backward on chunk: -1 memory
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        # Pre-compute memory limits for each rank (VPP formula)
        self.memory_limits = {
            rank: (self.pp_size) * (self.vpp_size - 1) + (self.pp_size - rank - 1) * 2 + 1
            for rank in range(self.pp_size)
        }
        for rank in self.memory_limits:
            if rank != self.pp_size - 1:
                self.memory_limits[rank] -= 1
        
        current_time = 0
        max_time = 100000
        num_ve_batches = self.num_microbatches // self.pp_size
        
        while current_time < max_time:
            # Step 1: Check which operations completed and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    completed_op = executing_ops[rank]
                    
                    # Update memory for LLM operations
                    if completed_op.op_type == OpType.FORWARD:
                        rank_memory[rank] += 1
                    elif completed_op.op_type == OpType.BACKWARD:
                        rank_memory[rank] -= 1
                    
                    self._unlock_next_ops(rank, completed_op, available_ops,
                                         ve_forward_count, rank0_backward_done,
                                         ve_backward_batch_available)
                    executing_ops[rank] = None
            
            # Step 2: Schedule new operations
            all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
            scheduled = False
            
            # Priority 1: VE Forward column execution
            if all_ranks_free:
                for batch_idx in range(num_ve_batches):
                    if batch_idx not in ve_forward_batch_started:
                        # Execute VE forward for this batch on all ranks
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            ve_op = VppScheduleOp(mb, OpType.VEFORWARD, chunk_id=0)
                            self.schedules[rank].append(ve_op)
                            rank_free_time[rank] = current_time + ve_op.duration
                            executing_ops[rank] = ve_op
                        
                        ve_forward_batch_started.add(batch_idx)
                        scheduled = True
                        break
            
            # Priority 2: VE Backward column execution
            # Only execute when all ranks are free AND no LLM operations are pending
            if not scheduled and all_ranks_free and len(ve_backward_batch_available) > 0:
                # Check if all ranks have no pending LLM operations
                has_llm_ops = any(
                    any(op_type in [OpType.FORWARD, OpType.BACKWARD] 
                        for mb, op_type, chunk_id in available_ops[r])
                    for r in range(self.pp_size)
                )
                
                if not has_llm_ops:
                    for batch_idx in sorted(ve_backward_batch_available):
                        if batch_idx not in ve_backward_batch_started:
                            ve_backward_batch_started.add(batch_idx)
                            ve_backward_batch_available.remove(batch_idx)
                            
                            # Execute VE backward for this batch on all ranks
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = VppScheduleOp(mb, OpType.VEBACKWARD, chunk_id=0)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            scheduled = True
                            break
            
            # Priority 3: Individual LLM operations (with VPP 1F1B memory limiting logic)
            if not scheduled:
                for rank in range(self.pp_size):
                    if rank_free_time[rank] <= current_time:
                        if len(available_ops[rank]) > 0:
                            # Memory limit from VPP formula
                            memory_limit = self.memory_limits[rank]
                            
                            # Filter operations based on memory constraint
                            candidate_ops = list(available_ops[rank])
                            if rank_memory[rank] >= memory_limit:
                                # Memory full, filter out Forward operations
                                candidate_ops = [(mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops 
                                               if op_type != OpType.FORWARD]
                            
                            if candidate_ops:
                                # Convert to VppScheduleOp objects and sort by priority
                                ops_list = [VppScheduleOp(mb, op_type, chunk_id) 
                                           for mb, op_type, chunk_id in candidate_ops]
                                ops_list.sort(key=self._get_priority)
                                
                                # Execute the highest priority operation
                                selected_op = ops_list[0]
                                self.schedules[rank].append(selected_op)
                                rank_free_time[rank] = current_time + selected_op.duration
                                executing_ops[rank] = selected_op
                                available_ops[rank].remove((selected_op.microbatch_id, 
                                                           selected_op.op_type, 
                                                           selected_op.chunk_id))
                                scheduled = True
                            else:
                                # No executable operations, add IDLE
                                idle_op = VppScheduleOp(None, OpType.IDLE, chunk_id=0)
                                self.schedules[rank].append(idle_op)
                                rank_free_time[rank] = current_time + idle_op.duration
                        else:
                            # No available operations, add IDLE
                            idle_op = VppScheduleOp(None, OpType.IDLE, chunk_id=0)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
            
            # Advance time
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1
            
            # Check if done
            if (len(ve_forward_batch_started) >= num_ve_batches and
                len(ve_backward_batch_started) >= num_ve_batches and
                all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")

class BigMacVPP(SandwichVPP):
    # Memory-efficient SandwichVPP with VE forward limit and VE backward prioritization
    #
    # Additional features over SandwichVPP:
    # 1. VE forward column limit: Limits how many VE forward batches can be in-flight concurrently
    # 2. VE backward prioritization: When VE backward is available, forces all ranks to sync
    #    by inserting IDLE, then immediately executes VE backward

    # For bigmac pipeline, ve_forward_limit is at least 3
    def __init__(self, pp_size: int, vpp_size: int, num_microbatches: int, ve_forward_limit: int = 3):
        super().__init__(pp_size, vpp_size, num_microbatches)
        self.ve_forward_limit = ve_forward_limit
        self.ve_backward_window_size = 1
    
    def optimize_communication_ops(self):
        self.make_llm_pp_p2p_adjacent()
        self.remove_null_ops()
        self.pin_comm_ops()
        self.vpp_p2p_overlap()
        self.reorder_llm_pp_send_recv()

    def make_llm_pp_p2p_adjacent(self):
        # make llm pp send and recv operations adjacent to each other.
        self.make_send_next_to_recv()
        self.make_recv_next_to_send()
    
    def make_send_next_to_recv(self):
        # make p2p send operation next to p2p recv operation.
        def find_next_recv_ops(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type in LLM_COMPUTE_OP_TYPES:
                    return -1
                if schedules[idx].op_type in LLM_RECV_OP_TYPES:
                    return idx
            return -1

        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in LLM_SEND_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv_idx = find_next_recv_ops(schedules, j)
                if next_recv_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_recv = schedules[next_recv_idx]
                # move recv forward ahead
                optimized_schedules.extend(schedules[j+1:next_recv_idx]) # ve related ops
                optimized_schedules.append(schedules[j])
                optimized_schedules.append(next_recv)
                j = next_recv_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules
    
    def make_recv_next_to_send(self):
        # make p2p recv operation next to p2p send operation.
        def find_next_send_ops(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type in LLM_COMPUTE_OP_TYPES:
                    return -1
                if schedules[idx].op_type in LLM_SEND_OP_TYPES:
                    return idx
            return -1
        
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in LLM_RECV_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_send_idx = find_next_send_ops(schedules, j)
                if next_send_idx == -1:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_send = schedules[next_send_idx]
                # move recv forward ahead
                optimized_schedules.extend(schedules[j+1:next_send_idx]) # ve related ops
                optimized_schedules.append(schedules[j])
                optimized_schedules.append(next_send)
                j = next_send_idx + 1
                
            self.schedules_with_comm[i] = optimized_schedules
    
    def generate_schedule(self):
        # Generate schedule with VE forward limit and VE backward prioritization
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Track available operations for each rank
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track currently executing operations
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track VE forward completion count per batch
        ve_forward_count = {}
        
        # Track which VE forward batches have been started
        ve_forward_batch_started = set()
        
        # Track which VE forward batches have completed (all LLM ops done)
        ve_forward_batch_completed = set()
        
        # Track rank 0 backward completion per batch (dict of sets)
        rank0_backward_done = {}
        
        # Track which VE backward batches are available and started
        ve_backward_batch_available = set()
        ve_backward_batch_started = set()
        
        # Track how many VE backward batches have been executed
        ve_backward_executed = 0
        
        # Track memory consumption for each rank (VPP 1F1B logic)
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        # Pre-compute memory limits for each rank (VPP formula)
        self.memory_limits = {
            rank: (self.pp_size) * (self.vpp_size - 1) + (self.pp_size - rank - 1) * 2 + 1
            for rank in range(self.pp_size)
        }
        for rank in self.memory_limits:
            if rank != self.pp_size - 1:
                self.memory_limits[rank] -= 1
        
        current_time = 0
        max_time = 100000
        num_ve_batches = self.num_microbatches // self.pp_size
        
        while current_time < max_time:
            # Step 1: Check which operations completed and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    completed_op = executing_ops[rank]
                    
                    # Update memory for LLM operations
                    if completed_op.op_type == OpType.FORWARD:
                        rank_memory[rank] += 1
                    elif completed_op.op_type == OpType.BACKWARD:
                        rank_memory[rank] -= 1
                    
                    self._unlock_next_ops(rank, completed_op, available_ops,
                                         ve_forward_count, rank0_backward_done,
                                         ve_backward_batch_available)
                    executing_ops[rank] = None
                    
                    # Check if a VE forward batch is now complete (all LLM ops done)
                    if completed_op.op_type == OpType.BACKWARD and rank == 0 and completed_op.chunk_id == 0:
                        mb = completed_op.microbatch_id
                        current_batch_idx = mb // self.pp_size
                        for batch_idx in range(current_batch_idx - self.ve_backward_window_size + 1, current_batch_idx + 1):
                            # Check if all microbatches in this batch have completed backward on rank 0
                            if batch_idx in rank0_backward_done and len(rank0_backward_done[batch_idx]) == self.pp_size:
                                if batch_idx not in ve_forward_batch_completed:
                                    ve_forward_batch_completed.add(batch_idx)
            
            # Step 2: Schedule new operations
            all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
            scheduled = False
            
            # Calculate in-flight VE forward batches
            ve_forward_in_flight = len(ve_forward_batch_started) - len(ve_forward_batch_completed)
            remaining_ve_backward = num_ve_batches - ve_backward_executed
            llm_completed = len(ve_forward_batch_completed) >= num_ve_batches
            
            # Priority 1: VE Backward prioritization (highest priority)
            if (len(ve_backward_batch_available) > 0 and
                (remaining_ve_backward > self.ve_forward_limit or llm_completed)):
                # has_pending_ops = any(len(available_ops[r]) > 0 for r in range(self.pp_size))
                if not all_ranks_free:
                    if rank_free_time[self.pp_size - 1] > current_time:
                        # Force sync: allow short operations that can finish before slowest rank
                        max_free_time = max(rank_free_time.values())
                        remaining_time = max_free_time - current_time
                        
                        for rank in range(self.pp_size):
                            if rank_free_time[rank] <= current_time:
                                scheduled_short_op = False
                                
                                if len(available_ops[rank]) > 0:
                                    # Look for operations that can complete before all ranks are free
                                    for mb, op_type, chunk_id in list(available_ops[rank]):
                                        temp_op = VppScheduleOp(mb, op_type, chunk_id)
                                        if temp_op.duration <= remaining_time:
                                            # Check memory limit for forward operations
                                            can_execute = True
                                            if temp_op.op_type == OpType.FORWARD:
                                                memory_limit = self.memory_limits[rank]
                                                if rank_memory[rank] >= memory_limit:
                                                    can_execute = False
                                            
                                            if can_execute:
                                                # Execute this short operation
                                                self.schedules[rank].append(temp_op)
                                                rank_free_time[rank] = current_time + temp_op.duration
                                                executing_ops[rank] = temp_op
                                                available_ops[rank].remove((mb, op_type, chunk_id))
                                                
                                                # Update memory
                                                if temp_op.op_type == OpType.FORWARD:
                                                    rank_memory[rank] += 1
                                                elif temp_op.op_type == OpType.BACKWARD:
                                                    rank_memory[rank] -= 1
                                                
                                                scheduled_short_op = True
                                                break
                                
                                if not scheduled_short_op:
                                    # No short operation available, add minimal IDLE
                                    idle_op = VppScheduleOp(None, OpType.IDLE, chunk_id=0)
                                    idle_op.duration = 1
                                    self.schedules[rank].append(idle_op)
                                    rank_free_time[rank] = current_time + idle_op.duration
                        
                        scheduled = True
                else:
                    # All ranks free, execute VE backward immediately
                    for batch_idx in sorted(ve_backward_batch_available):
                        if batch_idx not in ve_backward_batch_started:
                            ve_backward_batch_started.add(batch_idx)
                            ve_backward_batch_available.remove(batch_idx)
                            ve_backward_executed += 1
                            
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = VppScheduleOp(mb, OpType.VEBACKWARD, chunk_id=0)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            scheduled = True
                            break
            
            # Priority 2: VE Forward column execution (with limit)
            if not scheduled and all_ranks_free and ve_forward_in_flight < self.ve_forward_limit:
                for batch_idx in range(num_ve_batches):
                    if batch_idx not in ve_forward_batch_started:
                        # Execute VE forward for this batch on all ranks
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            ve_op = VppScheduleOp(mb, OpType.VEFORWARD, chunk_id=0)
                            self.schedules[rank].append(ve_op)
                            rank_free_time[rank] = current_time + ve_op.duration
                            executing_ops[rank] = ve_op
                        
                        ve_forward_batch_started.add(batch_idx)
                        scheduled = True
                        break
            
            # Priority 3: Individual LLM operations (with VPP 1F1B memory limiting logic)
            if not scheduled:
                for rank in range(self.pp_size):
                    if rank_free_time[rank] <= current_time:
                        if len(available_ops[rank]) > 0:
                            # Memory limit from VPP formula
                            memory_limit = self.memory_limits[rank]
                            
                            # Filter operations based on memory constraint
                            candidate_ops = list(available_ops[rank])
                            if rank_memory[rank] >= memory_limit:
                                # Memory full, filter out Forward operations
                                candidate_ops = [(mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops 
                                               if op_type != OpType.FORWARD]
                            
                            if candidate_ops:
                                # Convert to VppScheduleOp objects and sort by priority
                                ops_list = [VppScheduleOp(mb, op_type, chunk_id) 
                                           for mb, op_type, chunk_id in candidate_ops]
                                ops_list.sort(key=self._get_priority)
                                
                                # Execute the highest priority operation
                                selected_op = ops_list[0]
                                self.schedules[rank].append(selected_op)
                                rank_free_time[rank] = current_time + selected_op.duration
                                executing_ops[rank] = selected_op
                                available_ops[rank].remove((selected_op.microbatch_id, 
                                                           selected_op.op_type, 
                                                           selected_op.chunk_id))
                                scheduled = True
                            else:
                                # No executable operations, add IDLE
                                idle_op = VppScheduleOp(None, OpType.IDLE, chunk_id=0)
                                self.schedules[rank].append(idle_op)
                                rank_free_time[rank] = current_time + idle_op.duration
                        else:
                            # No available operations, add IDLE
                            idle_op = VppScheduleOp(None, OpType.IDLE, chunk_id=0)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
            
            # Advance time
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1
            
            # Check if done
            if (len(ve_forward_batch_started) >= num_ve_batches and
                len(ve_backward_batch_started) >= num_ve_batches and
                all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")


# Example usage


GEN_OP_TYPES = [
    OpType.GENFORWARD,
    OpType.GENBACKWARD,
    OpType.GENSENDFORWARD,
    OpType.GENSENDBACKWARD,
    OpType.GENRECVFORWARD,
    OpType.GENRECVBACKWARD,
]


class UnifiedScheduleOp(ScheduleOp):
    # Schedule operation representing what should be executed at a given step

    def __init__(
        self,
        microbatch_id: Optional[int],
        op_type: OpType,
        chunk_id: int,
        duration: Optional[float] = None,
    ):
        self.microbatch_id = microbatch_id
        self.op_type = op_type
        self.chunk_id = chunk_id
        self.duration = duration if duration is not None else _default_duration(op_type)
    
    def __repr__(self):
        if self.op_type == OpType.IDLE:
            return "IDLE"
        return f"{self.op_type.value}{self.microbatch_id}-C{self.chunk_id}"
    
    def __str__(self):
        return self.__repr__()


class UnifiedBigMacVPP(BigMacVPP):
    def __init__(self, pp_size: int, vpp_size: int, num_microbatches: int, ve_forward_limit: int = 3):
        super().__init__(pp_size, vpp_size, num_microbatches, ve_forward_limit)
        self.rank0_forward_group = 0

    def _get_priority(self, op: UnifiedScheduleOp) -> tuple:
        # Define priority for selecting operations from available set.
        # Priority: GEN Forward > GEN Backward > VE Forward > Backward (higher chunk) > Forward (higher chunk) > VE Backward
        #
        # Returns:
        #     A tuple for sorting - lower values = higher priority
        if op.op_type == OpType.VEFORWARD:
            return (2, op.microbatch_id)
        elif op.op_type == OpType.BACKWARD:
            # Prioritize higher chunk_id for backward
            return (4, op.microbatch_id, -op.chunk_id)
        elif op.op_type == OpType.FORWARD:
            # Prioritize higher chunk_id for forward (but lower than backward)
            return (3, op.microbatch_id, -op.chunk_id)
        elif op.op_type == OpType.VEBACKWARD:
            return (5, op.microbatch_id)
        elif op.op_type == OpType.GENFORWARD:
            return (0, op.microbatch_id)
        elif op.op_type == OpType.GENBACKWARD:
            return (1, op.microbatch_id)

    def _unlock_next_ops(self, rank: int, op: UnifiedScheduleOp, available_ops: dict,
                         ve_forward_count: dict, gen_forward_count: dict, before_gen_forward_count: dict,
                         gen_backward_count: dict, rank0_backward_done: dict,
                         ve_backward_batch_available: set, gen_forward_batch_available: set, 
                         gen_backward_batch_available: set):
        # Unlock next available operations when an operation completes.
        #
        # For VE operations: track completion count for column execution
        # For LLM operations: VPP pipeline dependencies with chunk_id
        mb = op.microbatch_id
        
        if op.op_type == OpType.VEFORWARD:
            # VE forward completed on this rank
            batch_idx = mb // self.pp_size
            ve_forward_count[batch_idx] = ve_forward_count.get(batch_idx, 0) + 1
            
            # When all ranks complete VE forward for this batch, unlock LLM chunk 0 forward on rank 0
            if ve_forward_count[batch_idx] == self.pp_size:
                # Unlock all microbatches in this batch for LLM chunk 0 forward on rank 0
                for r in range(self.pp_size):
                    batch_mb = batch_idx * self.pp_size + r
                    available_ops[0].add((batch_mb, OpType.FORWARD, 0))

        elif op.op_type == OpType.GENFORWARD:
            batch_idx = mb // self.pp_size
            gen_forward_count[batch_idx] = gen_forward_count.get(batch_idx, 0) + 1
            if gen_forward_count[batch_idx] == self.pp_size:
                gen_backward_batch_available.add(batch_idx)

        elif op.op_type == OpType.GENBACKWARD:
            batch_idx = mb // self.pp_size
            gen_backward_count[batch_idx] = gen_backward_count.get(batch_idx, 0) + 1
            if gen_backward_count[batch_idx] == self.pp_size:
                for r in range(self.pp_size):
                    batch_mb = batch_idx * self.pp_size + r
                    available_ops[self.pp_size - 1].add((batch_mb, OpType.BACKWARD, self.vpp_size - 1))
        
        elif op.op_type == OpType.FORWARD:
            chunk_id = op.chunk_id
            # After rank r executes F(mb, chunk), unlock rank r+1's F(mb, chunk)
            if rank < self.pp_size - 1:
                available_ops[rank + 1].add((mb, OpType.FORWARD, chunk_id))
                if (rank == 0) and (chunk_id == self.vpp_size - 1) and (mb % self.pp_size == self.pp_size - 1):
                    self.rank0_forward_group += 1
            else:
                # rank == pp_size - 1
                # Check if we should unlock next chunk or backward
                if chunk_id + 1 < self.vpp_size:
                    # Unlock rank 0's next chunk forward
                    available_ops[0].add((mb, OpType.FORWARD, chunk_id + 1))
                else:
                    # Last chunk, unlock gen forward
                    batch_idx = mb // self.pp_size
                    before_gen_forward_count[batch_idx] = before_gen_forward_count.get(batch_idx, 0) + 1
                    if before_gen_forward_count[batch_idx] == self.pp_size:
                        gen_forward_batch_available.add(batch_idx)
        
        elif op.op_type == OpType.BACKWARD:
            chunk_id = op.chunk_id
            # After rank r executes B(mb, chunk)
            if rank > 0:
                if chunk_id == self.vpp_size - 1 and mb == 0:
                    self.memory_limits[rank-1] += 1
                # Unlock rank r-1's B(mb, chunk)
                available_ops[rank - 1].add((mb, OpType.BACKWARD, chunk_id))
            else:
                # rank == 0
                # Check if we should unlock previous chunk backward
                if chunk_id > 0:
                    # Unlock rank pp_size-1's previous chunk backward
                    available_ops[self.pp_size - 1].add((mb, OpType.BACKWARD, chunk_id - 1))
                else:
                    # chunk_id == 0, LLM backward complete for this mb
                    # Track completion for VE backward
                    batch_idx = mb // self.pp_size
                    if batch_idx not in rank0_backward_done:
                        rank0_backward_done[batch_idx] = set()
                    rank0_backward_done[batch_idx].add(mb)
                    # Check if entire batch completed rank 0 backward (chunk 0)
                    if len(rank0_backward_done[batch_idx]) == self.pp_size:
                        ve_backward_batch_available.add(batch_idx)

    def generate_schedule(self):
        # Generate schedule with VE forward limit and VE backward prioritization
        self.schedules = {rank: [] for rank in range(self.pp_size)}
        
        # Track available operations for each rank
        available_ops = {rank: set() for rank in range(self.pp_size)}
        
        # Track when each rank will be free
        rank_free_time = {rank: 0 for rank in range(self.pp_size)}
        
        # Track currently executing operations
        executing_ops = {rank: None for rank in range(self.pp_size)}
        
        # Track VE forward completion count per batch
        ve_forward_count = {}
        
        # Track which VE forward batches have been started
        ve_forward_batch_started = set()
        
        # Track which VE forward batches have completed (all LLM ops done)
        ve_forward_batch_completed = set()

        # Track generator forward completion count per batch
        gen_forward_count = {}

        # Track before generator forward completion count per batch
        before_gen_forward_count = {}

        # Track generator backward completion count per batch
        gen_backward_count = {}

        # Track generator forward batch available
        gen_forward_batch_available = set()

        # Track generator backward batch available
        gen_backward_batch_available = set()
        
        # Track rank 0 backward completion per batch (dict of sets)
        rank0_backward_done = {}
        
        # Track which VE backward batches are available and started
        ve_backward_batch_available = set()
        ve_backward_batch_started = set()
        
        # Track how many VE backward batches have been executed
        ve_backward_executed = 0
        
        # Track memory consumption for each rank (VPP 1F1B logic)
        rank_memory = {rank: 0 for rank in range(self.pp_size)}
        
        # Pre-compute memory limits for each rank (VPP formula)
        self.memory_limits = {
            rank: (self.pp_size) * (self.vpp_size) + (self.pp_size - rank - 1) * 2 + 1
            for rank in range(self.pp_size)
        }
        for rank in self.memory_limits:
            if rank != self.pp_size - 1:
                self.memory_limits[rank] -= 1
        
        current_time = 0
        max_time = 10000
        num_ve_batches = self.num_microbatches // self.pp_size
        
        while current_time < max_time:
            # Step 1: Check which operations completed and unlock next ops
            for rank in range(self.pp_size):
                if executing_ops[rank] is not None and rank_free_time[rank] == current_time:
                    completed_op = executing_ops[rank]
                    
                    # Update memory for LLM operations
                    if completed_op.op_type == OpType.FORWARD:
                        rank_memory[rank] += 1
                    elif completed_op.op_type == OpType.BACKWARD:
                        rank_memory[rank] -= 1
                    
                    self._unlock_next_ops(rank, completed_op, available_ops,
                                         ve_forward_count, gen_forward_count, before_gen_forward_count,
                                         gen_backward_count, rank0_backward_done,
                                         ve_backward_batch_available, gen_forward_batch_available, 
                                         gen_backward_batch_available)
                    executing_ops[rank] = None
                    
                    # Check if a VE forward batch is now complete (all LLM ops done)
                    if completed_op.op_type == OpType.BACKWARD and rank == 0 and completed_op.chunk_id == 0:
                        mb = completed_op.microbatch_id
                        batch_idx = mb // self.pp_size
                        # Check if all microbatches in this batch have completed backward on rank 0
                        if batch_idx in rank0_backward_done and len(rank0_backward_done[batch_idx]) == self.pp_size:
                            if batch_idx not in ve_forward_batch_completed:
                                ve_forward_batch_completed.add(batch_idx)
            
            # Step 2: Schedule new operations
            all_ranks_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
            scheduled = False
            
            # Calculate in-flight VE forward batches
            ve_forward_in_flight = len(ve_forward_batch_started) - len(ve_forward_batch_completed)
            
            # Priority 1: VE Backward prioritization (highest priority)
            if len(ve_backward_batch_available) > 0:
                # has_pending_ops = any(len(available_ops[r]) > 0 for r in range(self.pp_size))
                if not all_ranks_free:
                    if rank_free_time[self.pp_size - 1] > current_time:
                        # Force sync: allow short operations that can finish before slowest rank
                        max_free_time = max(rank_free_time.values())
                        remaining_time = max_free_time - current_time
                        
                        for rank in range(self.pp_size):
                            if rank_free_time[rank] <= current_time:
                                scheduled_short_op = False
                                
                                if len(available_ops[rank]) > 0:
                                    # Look for operations that can complete before all ranks are free
                                    for mb, op_type, chunk_id in list(available_ops[rank]):
                                        temp_op = UnifiedScheduleOp(mb, op_type, chunk_id)
                                        if temp_op.duration <= remaining_time:
                                            # Check memory limit for forward operations
                                            can_execute = True
                                            if temp_op.op_type == OpType.FORWARD:
                                                memory_limit = self.memory_limits[rank]
                                                if rank_memory[rank] >= memory_limit:
                                                    can_execute = False
                                            
                                            if can_execute:
                                                # Execute this short operation
                                                self.schedules[rank].append(temp_op)
                                                rank_free_time[rank] = current_time + temp_op.duration
                                                executing_ops[rank] = temp_op
                                                available_ops[rank].remove((mb, op_type, chunk_id))
                                                
                                                # Update memory
                                                if temp_op.op_type == OpType.FORWARD:
                                                    rank_memory[rank] += 1
                                                elif temp_op.op_type == OpType.BACKWARD:
                                                    rank_memory[rank] -= 1
                                                
                                                scheduled_short_op = True
                                                break
                                
                                if not scheduled_short_op:
                                    # No short operation available, add minimal IDLE
                                    idle_op = UnifiedScheduleOp(None, OpType.IDLE, chunk_id=0)
                                    idle_op.duration = 1
                                    self.schedules[rank].append(idle_op)
                                    rank_free_time[rank] = current_time + idle_op.duration
                        
                        scheduled = True
                else:
                    # All ranks free, execute VE backward immediately
                    for batch_idx in sorted(ve_backward_batch_available):
                        if batch_idx not in ve_backward_batch_started:
                            ve_backward_batch_started.add(batch_idx)
                            ve_backward_batch_available.remove(batch_idx)
                            ve_backward_executed += 1
                            
                            for rank in range(self.pp_size):
                                mb = batch_idx * self.pp_size + rank
                                ve_op = UnifiedScheduleOp(mb, OpType.VEBACKWARD, chunk_id=0)
                                self.schedules[rank].append(ve_op)
                                rank_free_time[rank] = current_time + ve_op.duration
                                executing_ops[rank] = ve_op
                            
                            scheduled = True
                            break
            
            # Priority 2: Generator backward column execution
            if not scheduled and len(gen_forward_batch_available) > 0:
                if not all_ranks_free:
                    if rank_free_time[self.pp_size - 1] > current_time:
                        # Force sync: allow short operations that can finish before slowest rank
                        max_free_time = max(rank_free_time.values())
                        remaining_time = max_free_time - current_time
                        
                        for rank in range(self.pp_size):
                            if rank_free_time[rank] <= current_time:
                                scheduled_short_op = False
                                
                                if len(available_ops[rank]) > 0:
                                    # Look for operations that can complete before all ranks are free
                                    for mb, op_type, chunk_id in list(available_ops[rank]):
                                        temp_op = UnifiedScheduleOp(mb, op_type, chunk_id)
                                        if temp_op.duration <= remaining_time:
                                            # Check memory limit for forward operations
                                            can_execute = True
                                            if temp_op.op_type == OpType.FORWARD:
                                                memory_limit = self.memory_limits[rank]
                                                if rank_memory[rank] >= memory_limit:
                                                    can_execute = False
                                            
                                            if can_execute:
                                                # Execute this short operation
                                                self.schedules[rank].append(temp_op)
                                                rank_free_time[rank] = current_time + temp_op.duration
                                                executing_ops[rank] = temp_op
                                                available_ops[rank].remove((mb, op_type, chunk_id))
                                                
                                                # Update memory
                                                if temp_op.op_type == OpType.FORWARD:
                                                    rank_memory[rank] += 1
                                                elif temp_op.op_type == OpType.BACKWARD:
                                                    rank_memory[rank] -= 1
                                                
                                                scheduled_short_op = True
                                                break
                                
                                if not scheduled_short_op:
                                    # No short operation available, add minimal IDLE
                                    idle_op = UnifiedScheduleOp(None, OpType.IDLE, chunk_id=0)
                                    idle_op.duration = 1
                                    self.schedules[rank].append(idle_op)
                                    rank_free_time[rank] = current_time + idle_op.duration
                        
                        scheduled = True
                else:
                    # All ranks free, execute generator forward immediately
                    for batch_idx in sorted(gen_forward_batch_available):
                        gen_forward_batch_available.remove(batch_idx)
                        
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            gen_op = UnifiedScheduleOp(mb, OpType.GENFORWARD, chunk_id=self.vpp_size - 1)
                            self.schedules[rank].append(gen_op)
                            rank_free_time[rank] = current_time + gen_op.duration
                            executing_ops[rank] = gen_op
                        
                        scheduled = True
                        break

            # Priority 3: Generator forward column execution
            if not scheduled and len(gen_backward_batch_available) > 0:
                if not all_ranks_free:
                    if rank_free_time[self.pp_size - 1] > current_time:
                        # Force sync: allow short operations that can finish before slowest rank
                        max_free_time = max(rank_free_time.values())
                        remaining_time = max_free_time - current_time
                        
                        for rank in range(self.pp_size):
                            if rank_free_time[rank] <= current_time:
                                scheduled_short_op = False
                                
                                if len(available_ops[rank]) > 0:
                                    # Look for operations that can complete before all ranks are free
                                    for mb, op_type, chunk_id in list(available_ops[rank]):
                                        temp_op = UnifiedScheduleOp(mb, op_type, chunk_id)
                                        if temp_op.duration <= remaining_time:
                                            # Check memory limit for forward operations
                                            can_execute = True
                                            if temp_op.op_type == OpType.FORWARD:
                                                memory_limit = self.memory_limits[rank]
                                                if rank_memory[rank] >= memory_limit:
                                                    can_execute = False
                                            
                                            if can_execute:
                                                # Execute this short operation
                                                self.schedules[rank].append(temp_op)
                                                rank_free_time[rank] = current_time + temp_op.duration
                                                executing_ops[rank] = temp_op
                                                available_ops[rank].remove((mb, op_type, chunk_id))
                                                
                                                # Update memory
                                                if temp_op.op_type == OpType.FORWARD:
                                                    rank_memory[rank] += 1
                                                elif temp_op.op_type == OpType.BACKWARD:
                                                    rank_memory[rank] -= 1
                                                
                                                scheduled_short_op = True
                                                break
                                
                                if not scheduled_short_op:
                                    # No short operation available, add minimal IDLE
                                    idle_op = UnifiedScheduleOp(None, OpType.IDLE, chunk_id=0)
                                    idle_op.duration = 1
                                    self.schedules[rank].append(idle_op)
                                    rank_free_time[rank] = current_time + idle_op.duration
                        
                        scheduled = True
                else:
                    # All ranks free, execute generator backward immediately
                    for batch_idx in sorted(gen_backward_batch_available):
                        gen_backward_batch_available.remove(batch_idx)
                        
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            gen_op = UnifiedScheduleOp(mb, OpType.GENBACKWARD, chunk_id=0)
                            self.schedules[rank].append(gen_op)
                            rank_free_time[rank] = current_time + gen_op.duration
                            executing_ops[rank] = gen_op
                        
                        scheduled = True
                        break
            
            # Priority 4: VE Forward column execution (with limit)
            if not scheduled and all_ranks_free and ve_forward_in_flight < self.ve_forward_limit:
                for batch_idx in range(num_ve_batches):
                    if batch_idx not in ve_forward_batch_started:
                        # Execute VE forward for this batch on all ranks
                        for rank in range(self.pp_size):
                            mb = batch_idx * self.pp_size + rank
                            ve_op = UnifiedScheduleOp(mb, OpType.VEFORWARD, chunk_id=0)
                            self.schedules[rank].append(ve_op)
                            rank_free_time[rank] = current_time + ve_op.duration
                            executing_ops[rank] = ve_op
                        
                        ve_forward_batch_started.add(batch_idx)
                        scheduled = True
                        break
            
            # Priority 5: Individual LLM operations (with VPP 1F1B memory limiting logic)
            if not scheduled:
                for rank in range(self.pp_size):
                    if rank_free_time[rank] <= current_time:
                        if len(available_ops[rank]) > 0:
                            # Memory limit from VPP formula
                            memory_limit = self.memory_limits[rank]
                            
                            # Filter operations based on memory constraint
                            candidate_ops = list(available_ops[rank])
                            if rank_memory[rank] >= memory_limit:
                                # Memory full, filter out Forward operations
                                candidate_ops = [(mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops 
                                               if op_type != OpType.FORWARD]

                            # Filter out later-group forwards if an earlier group still has pending forwards
                            if rank == 0:
                                candidate_ops = [
                                    (mb, op_type, chunk_id) for mb, op_type, chunk_id in candidate_ops
                                    if op_type != OpType.FORWARD or mb // self.pp_size == self.rank0_forward_group
                                ]


                            if candidate_ops:
                                # Convert to UnifiedScheduleOp objects and sort by priority
                                ops_list = [UnifiedScheduleOp(mb, op_type, chunk_id) 
                                           for mb, op_type, chunk_id in candidate_ops]
                                ops_list.sort(key=self._get_priority)
                                
                                # Execute the highest priority operation
                                selected_op = ops_list[0]
                                self.schedules[rank].append(selected_op)
                                rank_free_time[rank] = current_time + selected_op.duration
                                executing_ops[rank] = selected_op
                                available_ops[rank].remove((selected_op.microbatch_id, 
                                                           selected_op.op_type, 
                                                           selected_op.chunk_id))
                                scheduled = True
                            else:
                                # No executable operations, add IDLE
                                idle_op = UnifiedScheduleOp(None, OpType.IDLE, chunk_id=0)
                                self.schedules[rank].append(idle_op)
                                rank_free_time[rank] = current_time + idle_op.duration
                        else:
                            # No available operations, add IDLE
                            idle_op = UnifiedScheduleOp(None, OpType.IDLE, chunk_id=0)
                            self.schedules[rank].append(idle_op)
                            rank_free_time[rank] = current_time + idle_op.duration
            
            # Advance time
            next_event_time = min(rank_free_time.values())
            if next_event_time > current_time:
                current_time = next_event_time
            else:
                current_time += 1
            
            # Check if done
            if (len(ve_forward_batch_started) >= num_ve_batches and
                len(ve_backward_batch_started) >= num_ve_batches and
                all(len(available_ops[r]) == 0 for r in range(self.pp_size))):
                all_free = all(rank_free_time[r] <= current_time for r in range(self.pp_size))
                if all_free:
                    break
        
        if current_time >= max_time:
            raise RuntimeError("Schedule generation exceeded maximum time")
    
    def optimize_communication_ops(self):
        self.make_llm_pp_p2p_adjacent()
        self.remove_null_ops()
        self.pin_comm_ops()
        self.vpp_p2p_overlap()
        self.reorder_llm_pp_send_recv()
        if self.num_microbatches == self.pp_size:
            option = 'between'
        else:
            option = 'after'
        self.move_generator_and_send_recv(option)
        
    
    def move_generator_and_send_recv(self, option='after'):
        # GRF, GF, GB, GSB, SF, RB => SF, GRF, GF, GB, GSB, RB

        def find_next_send_or_recv(schedules, start):
            for idx in range(start+1, len(schedules)):
                if schedules[idx].op_type in GEN_OP_TYPES:
                    continue
                elif schedules[idx].op_type in LLM_SEND_OP_TYPES:
                    return idx
                elif schedules[idx].op_type in LLM_RECV_OP_TYPES:
                    return idx
                else:
                    return -1
            return -1
        
        for i in range(self.pp_size):
            schedules = self.schedules_with_comm[i]
            optimized_schedules = []
            j = 0
            while j < len(schedules):
                if schedules[j].op_type not in GEN_OP_TYPES:
                    optimized_schedules.append(schedules[j])
                    j += 1
                    continue
                next_send_or_recv_idx = find_next_send_or_recv(schedules, j)
                assert next_send_or_recv_idx != -1
                next_send_or_recv = schedules[next_send_or_recv_idx]
                next_recv_or_send = schedules[next_send_or_recv_idx + 1]
                # move gen op forward
                if option == 'after':
                    optimized_schedules.append(next_send_or_recv)
                    optimized_schedules.append(next_recv_or_send)
                    optimized_schedules.extend(schedules[j:next_send_or_recv_idx])
                elif option == 'between':
                    optimized_schedules.append(next_send_or_recv)
                    optimized_schedules.extend(schedules[j:next_send_or_recv_idx])
                    optimized_schedules.append(next_recv_or_send)
                else:
                    raise ValueError(f"Invalid option: {option}")
                j = next_send_or_recv_idx + 2
                
            self.schedules_with_comm[i] = optimized_schedules
