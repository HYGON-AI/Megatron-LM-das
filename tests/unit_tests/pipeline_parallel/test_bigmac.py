# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CPU algorithm checks; no model weights or distributed runtime are required.
# Run from the repository root:
# python -B tests/unit_tests/pipeline_parallel/test_bigmac.py -v

import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from hcu_megatron.core.pipeline_parallel.bigmac.schedules import (
    BigMac, BigMacVPP, UnifiedBigMacVPP, CommunicationOp, FusedScheduleOp,
    OpType, ScheduleOp,
)
from hcu_megatron.core.pipeline_parallel.bigmac.simulator import (
    OpDurationSpec, PipelineSimulator,
)


class TestBigMacAlgorithm(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('VE_FORWARD_LIMIT', None)

    def test_encoder_gradient_routing(self):
        for schedule_type in (BigMac, BigMacVPP, UnifiedBigMacVPP):
            with self.subTest(schedule=schedule_type.__name__):
                kwargs = dict(pp_size=8, num_microbatches=32, ve_forward_limit=4)
                if schedule_type is not BigMac:
                    kwargs['vpp_size'] = 2
                scheduler = schedule_type(**kwargs)
                scheduler.generate_schedule()
                scheduler.add_communication_op()
                for optimized in (False, True):
                    with self.subTest(optimized=optimized):
                        if optimized:
                            scheduler.optimize_communication_ops()
                            scheduler.fuse_comm_ops()
                        ops = list(_unfused_communication_ops(scheduler))
                        expected = Counter((mb, 0, mb % 8) for mb in range(32))
                        for kind in (OpType.VESENDBACKWARD, OpType.VERECVBACKWARD):
                            routed = [op for op in ops if op.op_type == kind]
                            self.assertEqual(Counter((op.microbatch_id, op.src_rank, op.dst_rank)
                                                     for op in routed), expected)
                            if schedule_type is not BigMac:
                                self.assertTrue(all(op.chunk_id == 0 for op in routed))

    def test_bigmac_llm_forward_waits_for_ve_unit(self):
        scheduler = BigMac(pp_size=2, num_microbatches=4, ve_forward_limit=3)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        target = _find_op(simulator, rank=0, op_type='F', microbatch_id=0, chunk_id=None)
        vf0 = _find_op(simulator, rank=0, op_type='VF', microbatch_id=0, chunk_id=None)
        vf1 = _find_op(simulator, rank=1, op_type='VF', microbatch_id=1, chunk_id=None)
        reasons = simulator.graph.dependency_reasons[target.id]
        assert 've_forward_ready' in reasons[vf0.id]
        assert vf1.id not in reasons

    def test_bigmac_llm_forward_waits_only_for_matching_ve_forward(self):
        scheduler = BigMac(pp_size=4, num_microbatches=16, ve_forward_limit=3, check_ve_forward_limit=False)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        f12 = _find_op(simulator, rank=0, op_type='F', microbatch_id=12, chunk_id=None)
        vf12 = _find_op(simulator, rank=0, op_type='VF', microbatch_id=12, chunk_id=None)
        peer_vfs = [_find_op(simulator, rank=rank, op_type='VF', microbatch_id=12 + rank, chunk_id=None) for rank in range(1, 4)]
        reasons = simulator.graph.dependency_reasons[f12.id]
        assert 've_forward_ready' in reasons[vf12.id]
        for peer_vf in peer_vfs:
            assert peer_vf.id not in reasons

    def test_bigmac_low_ve_forward_limit_exposes_bubbles(self):
        scheduler = BigMac(pp_size=4, num_microbatches=16, ve_forward_limit=2, check_ve_forward_limit=False)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        result = simulator.simulate(_default_specs(), seed=23)
        simulator.validate_result(result)
        idle_counts = {rank: sum((1 for op in ops if op.op_type == OpType.IDLE)) for rank, ops in scheduler.schedules.items()}
        assert sum(idle_counts.values()) > 0

    def test_bigmac_low_ve_forward_limit_interleaves_ve_backward_before_next_ve_forward(self):
        scheduler = BigMac(pp_size=4, num_microbatches=16, ve_forward_limit=3, check_ve_forward_limit=False)
        scheduler.generate_schedule()
        rank0_ops = [str(op) for op in scheduler.schedules[0] if op.op_type != OpType.IDLE]
        assert rank0_ops.index('B8') < rank0_ops.index('VB0')
        assert rank0_ops[rank0_ops.index('VB0') + 1] == 'VF12'
        assert rank0_ops.index('VF12') < rank0_ops.index('B9')

    def test_bigmac_vpp_chunk_wrap_dependencies(self):
        scheduler = BigMacVPP(pp_size=2, vpp_size=2, num_microbatches=4, ve_forward_limit=3)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        f_rank0_chunk1 = _find_op(simulator, rank=0, op_type='F', microbatch_id=0, chunk_id=1)
        f_rank1_chunk0 = _find_op(simulator, rank=1, op_type='F', microbatch_id=0, chunk_id=0)
        b_rank1_chunk0 = _find_op(simulator, rank=1, op_type='B', microbatch_id=0, chunk_id=0)
        b_rank0_chunk1 = _find_op(simulator, rank=0, op_type='B', microbatch_id=0, chunk_id=1)
        reasons = simulator.graph.dependency_reasons
        assert 'vpp_forward_from_previous_chunk' in reasons[f_rank0_chunk1.id][f_rank1_chunk0.id]
        assert 'vpp_backward_from_next_chunk' in reasons[b_rank1_chunk0.id][b_rank0_chunk1.id]

    def test_unified_generator_dependencies_use_matching_microbatch(self):
        scheduler = UnifiedBigMacVPP(pp_size=2, vpp_size=2, num_microbatches=4, ve_forward_limit=3)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        gf0 = _find_op(simulator, rank=0, op_type='GF', microbatch_id=0, chunk_id=1)
        gb0 = _find_op(simulator, rank=0, op_type='GB', microbatch_id=0, chunk_id=0)
        b_last = _find_op(simulator, rank=1, op_type='B', microbatch_id=0, chunk_id=1)
        final_f0 = _find_op(simulator, rank=1, op_type='F', microbatch_id=0, chunk_id=1)
        final_f1 = _find_op(simulator, rank=1, op_type='F', microbatch_id=1, chunk_id=1)
        gb1 = _find_op(simulator, rank=1, op_type='GB', microbatch_id=1, chunk_id=0)
        reasons = simulator.graph.dependency_reasons
        assert 'llm_output_ready' in reasons[gf0.id][final_f0.id]
        assert final_f1.id not in reasons[gf0.id]
        assert 'generator_backward_consumes_forward' in reasons[gb0.id][gf0.id]
        assert gb1.id not in reasons[gb0.id]
        assert 'generator_backward_ready' in reasons[b_last.id][gb0.id]
        assert gb1.id not in reasons[b_last.id]

    def test_unified_generator_backward_does_not_wait_for_peer_generator_forward(self):
        scheduler = UnifiedBigMacVPP(pp_size=4, vpp_size=2, num_microbatches=8, ve_forward_limit=3)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        gb4 = _find_op(simulator, rank=0, op_type='GB', microbatch_id=4, chunk_id=0)
        gf4 = _find_op(simulator, rank=0, op_type='GF', microbatch_id=4, chunk_id=1)
        peer_gfs = [_find_op(simulator, rank=rank, op_type='GF', microbatch_id=4 + rank, chunk_id=1) for rank in range(1, 4)]
        reasons = simulator.graph.dependency_reasons[gb4.id]
        assert 'generator_backward_consumes_forward' in reasons[gf4.id]
        for peer_gf in peer_gfs:
            assert peer_gf.id not in reasons

    def test_bigmac_ve_backward_waits_only_for_matching_llm_backward(self):
        scheduler = BigMac(pp_size=4, num_microbatches=32, ve_forward_limit=3, check_ve_forward_limit=False)
        scheduler.generate_schedule()
        simulator = PipelineSimulator.from_scheduler(scheduler)
        vb30 = _find_op(simulator, rank=2, op_type='VB', microbatch_id=30, chunk_id=None)
        b30 = _find_op(simulator, rank=0, op_type='B', microbatch_id=30, chunk_id=None)
        peer_bs = [_find_op(simulator, rank=0, op_type='B', microbatch_id=mb, chunk_id=None) for mb in (28, 29, 31)]
        reasons = simulator.graph.dependency_reasons[vb30.id]
        assert 'llm_input_gradient_ready' in reasons[b30.id]
        for peer_b in peer_bs:
            assert peer_b.id not in reasons


def _unfused_communication_ops(scheduler):
    for ops in scheduler.schedules_with_comm.values():
        for op in ops:
            if isinstance(op, FusedScheduleOp):
                yield op.op1
                yield op.op2
            else:
                yield op


def _find_op(simulator, *, rank, op_type, microbatch_id, chunk_id):
    for op in simulator.plan.ops_by_rank[rank]:
        if op.op_type == op_type and op.microbatch_id == microbatch_id and (op.chunk_id == chunk_id):
            return op
    raise AssertionError(f'missing op rank={rank} op_type={op_type} microbatch_id={microbatch_id} chunk_id={chunk_id}')


def _default_specs():
    return {'F': OpDurationSpec(mean=1.0), 'B': OpDurationSpec(mean=2.0), 'VF': OpDurationSpec(mean=0.8), 'VB': OpDurationSpec(mean=0.9), 'GF': OpDurationSpec(mean=0.6), 'GB': OpDurationSpec(mean=0.7)}


_SEND_TO_RECV = {OpType.SENDFORWARD: OpType.RECVFORWARD,
                 OpType.SENDBACKWARD: OpType.RECVBACKWARD}
_RECV_TO_SEND = {v: k for k, v in _SEND_TO_RECV.items()}
_P2P = set(_SEND_TO_RECV) | set(_RECV_TO_SEND)
_COLLECTIVES = {OpType.VESENDFORWARD, OpType.VERECVBACKWARD}
_COMPUTE = {OpType.FORWARD, OpType.BACKWARD, OpType.VEFORWARD, OpType.VEBACKWARD}


def _blocked_operations(plans):
    # Model all-rank gather/scatter waits and independent fused P2P legs.
    # Boundary receives are no-ops; VSB stores gradients and VRB scatters them.
    size = len(plans)
    pcs = {rank: 0 for rank in plans}
    completed = set()
    for _ in range(4 * sum(map(len, plans.values())) + 1):
        progress = False
        posted = {}
        collectives = {}
        for rank, plan in plans.items():
            if pcs[rank] == len(plan):
                continue
            op = plan[pcs[rank]]
            if op.op_type in _COLLECTIVES:
                key = (op.op_type, op.microbatch_id // size)
                collectives.setdefault(key, []).append(rank)
                continue
            children = (op.op1, op.op2) if isinstance(op, FusedScheduleOp) else (op,)
            pending = False
            for i, child in enumerate(children):
                boundary = ((child.op_type == OpType.RECVFORWARD and rank == 0)
                            or (child.op_type == OpType.RECVBACKWARD and rank == size - 1)
                            or (child.op_type == OpType.SENDBACKWARD and rank == 0))
                if child.op_type not in _P2P or boundary:
                    continue
                token = (rank, pcs[rank], i)
                if token in completed:
                    continue
                key = (child.op_type, child.microbatch_id, child.src_rank, child.dst_rank)
                assert key not in posted, f'duplicate P2P operation: {key}'
                posted[key] = token
                pending = True
            if not pending:
                pcs[rank] += 1
                progress = True
        for key, token in posted.items():
            opposite = _SEND_TO_RECV.get(key[0], _RECV_TO_SEND.get(key[0]))
            if (opposite, *key[1:]) in posted:
                completed.add(token)
                progress = True
        for ranks in collectives.values():
            if len(ranks) == size:
                for rank in ranks:
                    pcs[rank] += 1
                progress = True
        if all(pcs[r] == len(plans[r]) for r in plans):
            return {}
        if not progress:
            return {r: str(plans[r][pcs[r]]) for r in plans if pcs[r] < len(plans[r])}
    raise AssertionError('protocol model exceeded its finite step bound')


def _comm(kind, mb, src, dst):
    return CommunicationOp(ScheduleOp(mb, OpType.FORWARD), kind, src, dst)


class TestVisionCommunicationSchedule(unittest.TestCase):
    def test_checker_detects_collective_p2p_cycle(self):
        plans = {
            0: [_comm(OpType.VERECVBACKWARD, 0, 0, 0),
                _comm(OpType.RECVBACKWARD, 0, 1, 0)],
            1: [_comm(OpType.SENDBACKWARD, 0, 1, 0),
                _comm(OpType.VERECVBACKWARD, 1, 0, 1)],
        }
        self.assertEqual(set(_blocked_operations(plans)), {0, 1})
        plans[1].reverse()
        self.assertEqual(_blocked_operations(plans), {})

    def test_checker_completes_fused_legs_independently(self):
        plans = {
            0: [FusedScheduleOp(_comm(OpType.SENDFORWARD, 0, 0, 1),
                                _comm(OpType.RECVBACKWARD, 0, 1, 0))],
            1: [_comm(OpType.RECVFORWARD, 0, 0, 1), ScheduleOp(0, OpType.BACKWARD),
                _comm(OpType.SENDBACKWARD, 0, 1, 0)],
        }
        self.assertEqual(_blocked_operations(plans), {})

    def test_large_batch_protocol_completes_without_changing_compute_or_ve_limit(self):
        with patch.dict(os.environ):
            os.environ.pop('VE_FORWARD_LIMIT', None)
            for size in (2, 3, 4, 8):
                for waves in (1, 2, 3, 4, 8, 16):
                    for limit in (1, 2, 3, 4, 8):
                        with self.subTest(pp=size, microbatches=size * waves, limit=limit):
                            scheduler = BigMac(size, size * waves, limit, check_ve_forward_limit=False)
                            scheduler.generate_schedule()
                            scheduler.add_communication_op()
                            before = {rank: [op for op in plan if op.op_type in _COMPUTE]
                                      for rank, plan in scheduler.schedules_with_comm.items()}
                            scheduler.optimize_communication_ops()
                            scheduler.fuse_comm_ops()
                            self.assertEqual(_blocked_operations(scheduler.schedules_with_comm), {})
                            for rank, plan in scheduler.schedules_with_comm.items():
                                self.assertEqual([op for op in plan if op.op_type in _COMPUTE], before[rank])
                                live = 0
                                for op in before[rank]:
                                    live += (op.op_type == OpType.VEFORWARD) - (op.op_type == OpType.VEBACKWARD)
                                    self.assertTrue(0 <= live <= limit)
                                self.assertEqual(live, 0)
                                for kind in (OpType.FORWARD, OpType.BACKWARD):
                                    self.assertEqual(Counter(op.microbatch_id for op in before[rank] if op.op_type == kind),
                                                     Counter(range(size * waves)))

    def test_pp4_32_microbatches_defers_gradient_send_until_receiver_phase(self):
        with patch.dict(os.environ, {'VE_FORWARD_LIMIT': '3'}):
            scheduler = BigMac(4, 32, check_ve_forward_limit=False)
        scheduler.generate_schedule()
        scheduler.add_communication_op()
        scheduler.optimize_communication_ops()
        plan = scheduler.schedules_with_comm[1]
        index = lambda kind, mb: next(i for i, op in enumerate(plan) if op.op_type == kind and op.microbatch_id == mb)
        # Rank 1 computes B9 early but sends it only after VB1 and VF13/VSF13.
        self.assertLess(index(OpType.BACKWARD, 9), index(OpType.VERECVBACKWARD, 1))
        self.assertLess(index(OpType.VESENDFORWARD, 13), index(OpType.SENDBACKWARD, 9))
        self.assertLess(index(OpType.VESENDFORWARD, 13), index(OpType.RECVFORWARD, 12))
        scheduler.fuse_comm_ops()
        self.assertEqual(_blocked_operations(scheduler.schedules_with_comm), {})

    def test_small_batch_requires_no_send_deferral(self):
        with patch.dict(os.environ, {'VE_FORWARD_LIMIT': '3'}):
            for count in (4, 8, 12):
                with self.subTest(microbatches=count):
                    scheduler = BigMac(4, count)
                    scheduler.generate_schedule()
                    scheduler.add_communication_op()
                    before = {r: list(plan) for r, plan in scheduler.schedules_with_comm.items()}
                    scheduler.align_p2p_with_ve_phases()
                    self.assertEqual(scheduler.schedules_with_comm, before)


if __name__ == '__main__':
    unittest.main()
