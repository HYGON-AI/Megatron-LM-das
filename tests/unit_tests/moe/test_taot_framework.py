# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""TAOT framework unit tests and EP integration tests.

Framework tests use fake runtimes. Integration tests use real TE, Triton and
communication backends. Run on one eight-HCU node with::

    python -m torch.distributed.run --nproc-per-node=8 -m pytest -s -vv \
        tests/unit_tests/moe/test_taot_framework.py

Set TAOT_TEST_DEEPEP=1 to include real DeepEP communication tests.
"""

import ast
import importlib.util
import sys
import types
import argparse
import inspect
import os
import warnings
from datetime import timedelta
from functools import wraps
from pathlib import Path
from contextlib import contextmanager
from contextlib import nullcontext
from typing import List, Optional
from typing import Union

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Parameter

ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / 'hcu_megatron/core/transformer/moe'


def _definitions(path, names, **scope):
    """Execute production definitions without importing the training runtime.

    Decorators are removed only to bypass compile/JIT during CPU tests; function
    bodies are unchanged. Triton kernels are tested separately on real hardware.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if getattr(n, 'name', None) in names]
    assert {n.name for n in tree.body} == set(names), f'missing definitions in {path}'
    for node in tree.body:
        node.decorator_list = []
    module = types.ModuleType('taot_test_definitions')
    module.__dict__.update(torch=torch, Optional=Optional, Union=Union, **scope)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture
def planner():
    return _definitions(
        MOE / 'offloading_planner.py',
        {'gen_intermediate', '_get_sinkhorn_statics', '_sinkhorn_transport_flow',
         'gen_assignment_sinkhorn', '_synchronize_offloading_plan'},
        _sinkhorn_static_cache={},
    )


class TestTAOTFeature:
    @pytest.fixture
    def feature(self):
        class BaseFeature:
            def __init__(self, name):
                self.feature_name = name
        module = _definitions(
            ROOT / 'hcu_megatron/features_manager/moe/echo_expert_feature.py',
            {'EchoExpertFeature'}, AbstractFeature=BaseFeature,
            ArgumentParser=argparse.ArgumentParser, warnings=warnings,
        )
        return module.EchoExpertFeature()

    @staticmethod
    def args(**overrides):
        values = dict(moe_enable_echo=True, moe_num_echo_experts=8,
                      expert_model_parallel_size=8,
                      moe_echo_expert_dispatch_type='alltoall',
                      gradient_accumulation_fusion=True, delay_wgrad_compute=False)
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_parser_defaults(self, feature):
        parser = argparse.ArgumentParser()
        feature.register_args(parser)
        args = parser.parse_args([])
        assert not args.moe_enable_echo
        assert args.moe_echo_algorithm == 'sinkhorn'
        assert args.moe_echo_expert_dispatch_type == 'alltoall'
        assert not args.moe_echo_expert_dispatch_overlap

    @pytest.mark.parametrize('changes', [
        dict(moe_num_echo_experts=None), dict(moe_num_echo_experts=0),
        dict(moe_num_echo_experts=7), dict(moe_num_echo_experts=-8),
        dict(delay_wgrad_compute=True), dict(moe_echo_expert_dispatch_type='invalid'),
        dict(expert_model_parallel_size=1, moe_echo_expert_dispatch_type='deepep'),
    ])
    def test_reject_invalid_configuration(self, feature, changes):
        with pytest.raises(ValueError):
            feature.validate_args(self.args(**changes))

    @pytest.mark.parametrize('backend', ['alltoall', 'deepep'])
    def test_accept_fusion(self, feature, backend):
        args = self.args(moe_echo_expert_dispatch_type=backend)
        assert feature.validate_args(args) is args
        assert args.gradient_accumulation_fusion


class TestTAOTPlanner:
    @pytest.mark.parametrize('totals', [
        [0] * 8, [4] * 8, [32, 0, 0, 0, 0, 0, 0, 0],
        [20, 4, 8, 4, 4, 4, 2, 2],
    ])
    @pytest.mark.parametrize('slots', [1, 2])
    def test_sinkhorn_capacity_and_ownership(self, planner, totals, slots):
        counts = torch.zeros(4, 8, dtype=torch.int32)
        counts[0] = torch.tensor(totals)
        saved = counts.clone()
        plan = planner.gen_assignment_sinkhorn(
            counts, 0, 4, num_spare_experts_per_ep_rank=slots,
            ranks_per_node=2, _force_triton=False,
        )
        loads = torch.tensor(totals).reshape(4, 2).sum(1)
        avg = loads.float().mean()
        moved_by_destination = plan.sum(0).reshape(4, slots).sum(1)
        moved_by_source = plan.sum(1).reshape(4, 2).sum(1)
        assert plan.shape == (8, 4 * slots)
        assert plan.dtype == torch.int32
        assert (plan >= 0).all()
        assert (moved_by_destination <= (avg - loads).clamp(min=0)).all()
        assert (moved_by_source <= (loads - avg).clamp(min=0)).all()
        assert ((plan > 0).sum(0) <= 1).all()  # One owner per Echo slot.
        for expert in range(8):
            owner = expert // 2
            assert not plan[expert, owner * slots:(owner + 1) * slots].any()
        if len(set(totals)) == 1:
            assert not plan.any()
        else:
            assert plan.sum() > 0  # Nontrivial fixtures must actually offload.
        torch.testing.assert_close(counts, saved)

    def test_intermediate_spill_and_capacity(self, planner):
        counts = torch.tensor([[20, 4, 2, 2], [0, 0, 0, 0]], dtype=torch.int32)
        spill, spare, avg = planner.gen_intermediate(counts, 0, 2)
        torch.testing.assert_close(spill, torch.tensor([10, 0, 0, 0], dtype=spill.dtype))
        torch.testing.assert_close(spare, torch.tensor([0, 10], dtype=spare.dtype))
        assert avg == 14

    def test_complete_plan_broadcast_uses_ep_root(self, planner, monkeypatch):
        group = object()
        expert_map = torch.tensor([[True, False], [False, False]])
        home = torch.zeros(2, 2, dtype=torch.int64)
        spare = torch.zeros(2, 2, dtype=torch.int32)
        # Simulate a root plan different from this rank's local plan.
        expected = (torch.tensor([[False, False], [True, False]]),
                    torch.tensor([[0, 2], [0, 3]], dtype=torch.int64),
                    torch.tensor([[2, 0], [3, 0]], dtype=torch.int32))
        packed = torch.cat([t.reshape(-1).int() for t in expected])
        monkeypatch.setattr(dist, 'get_world_size', lambda g: 2)
        monkeypatch.setattr(dist, 'get_global_rank', lambda g, rank: 7)
        def broadcast(tensor, src, group):
            assert src == 7
            assert group is expected_group
            tensor.copy_(packed)
        expected_group = group
        monkeypatch.setattr(dist, 'broadcast', broadcast)
        actual = planner._synchronize_offloading_plan(expert_map, home, spare, group)
        for received, reference in zip(actual, expected):
            assert received.dtype == reference.dtype
            torch.testing.assert_close(received, reference)

    def test_single_rank_plan_needs_no_collective(self, planner, monkeypatch):
        monkeypatch.setattr(dist, 'get_world_size', lambda g: 1)
        def unexpected(*args, **kwargs):
            pytest.fail('single rank must not broadcast')
        monkeypatch.setattr(dist, 'broadcast', unexpected)
        values = (torch.ones(2, 1, dtype=torch.bool), torch.ones(1, 2), torch.ones(1, 1))
        actual = planner._synchronize_offloading_plan(*values, object())
        assert all(a is b for a, b in zip(actual, values))


def test_echo_forward_factory_returns_callable():
    # Regression for accidentally deleting the OUTER `return wrapper` when
    # removing diagnostics; this would install MoELayer.forward = None.
    module = _definitions(MOE / 'moe_layer.py', {'moe_layer_forward_wrapper'}, wraps=wraps)
    sentinel = object()
    original = lambda self, **kwargs: sentinel
    wrapped = module.moe_layer_forward_wrapper(original)
    assert callable(wrapped)
    layer = types.SimpleNamespace(config=types.SimpleNamespace(moe_enable_echo=False),
                                  training=False, experts_recompute=False, router_recompute=False)
    assert wrapped(layer, hidden_states=torch.ones(1)) is sentinel


@pytest.mark.parametrize('echo_enabled', [False, True])
def test_init_restores_home_expert_count_on_failure(echo_enabled):
    module = _definitions(
        MOE / 'moe_layer.py', {'moe_layer_init_wrapper'}, wraps=wraps,
        TransformerConfig=object, MoESubmodules=object, ProcessGroupCollection=object,
    )
    config = types.SimpleNamespace(moe_enable_echo=echo_enabled, num_moe_experts=16,
                                   moe_num_echo_experts=8)
    def failing_init(self, cfg, **kwargs):
        assert cfg.num_moe_experts == (24 if echo_enabled else 16)
        raise RuntimeError('upstream init failed')
    with pytest.raises(RuntimeError, match='upstream init failed'):
        module.moe_layer_init_wrapper(failing_init)(types.SimpleNamespace(), config)
    assert config.num_moe_experts == 16


def test_echo_wrapper_routes_once_and_preserves_sync_schedule(production, monkeypatch):
    helpers, _, _ = production
    events = []
    group = types.SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    routes = torch.tensor([[True, False, False, False], [False, True, False, False]])
    probs = routes.float()
    expert_map = torch.zeros(4, 4, dtype=torch.bool)
    planner_api = types.ModuleType('taot_fake_planner')
    def plan(routing, probabilities, counts, rank, **kwargs):
        events.append('planner')
        assert routing is routes and probabilities is probs
        assert counts.shape == (2, 4)
        assert kwargs['ep_group'] is group
        return routes, probs, expert_map
    planner_api.gen_offloading_plan = plan
    monkeypatch.setitem(sys.modules, 'hcu_megatron.core.transformer.moe.offloading_planner', planner_api)
    mappings = types.ModuleType('taot_fake_mappings')
    def gather(counts, group):
        events.append('gather')
        return counts.repeat(2)
    mappings.gather_from_sequence_parallel_region = gather
    monkeypatch.setitem(sys.modules, 'megatron.core.tensor_parallel.mappings', mappings)

    class WeightDispatcher:
        def preprocess(self, mapping):
            assert mapping is expert_map
            return object()

        def expert_dispatch(self, metadata, *weights):
            events.append('weight_dispatch')
            return [w.clone() for w in weights]

    config = types.SimpleNamespace(moe_enable_echo=True, moe_echo_algorithm='sinkhorn',
                                   moe_echo_ranks_per_node=8, moe_echo_dump_dir=None,
                                   moe_echo_expert_dispatch_overlap=False,
                                   gradient_accumulation_fusion=True)
    experts = make_experts(True)
    originals = [experts.linear_fc1.weight0, experts.linear_fc2.weight0]
    def route(x, mask):
        events.append('route')
        return probs, routes
    def shared(x):
        events.append('shared')
        return torch.ones_like(x)
    def preprocess(x, p, r):
        events.append('token_preprocess')
        return x, p
    layer = types.SimpleNamespace(
        config=config, ep_group=group, _echo_num_home_experts=4, _echo_num_echo_experts=4,
        home_expert_indices=[0, 1], echo_expert_indices=[2, 3], experts=experts,
        expert_dispatcher=WeightDispatcher(), shared_experts_compute=shared, route=route,
        preprocess=preprocess, dispatch=lambda x, p: (x, p),
        routed_experts_compute=lambda x, p: (x + 2, None),
        combine=lambda x: x * 3, postprocess=lambda x, shared_output: x + shared_output,
    )
    def unexpected(*args, **kwargs):
        pytest.fail('Echo must not invoke the original forward')
    module = _definitions(MOE / 'moe_layer.py', {'moe_layer_forward_wrapper'}, wraps=wraps)
    x = torch.ones(2, 3, requires_grad=True)
    output, bias = module.moe_layer_forward_wrapper(unexpected)(layer, x)
    torch.testing.assert_close(output, (x + 2) * 3 + 1)
    assert bias is None
    assert events == ['shared', 'route', 'gather', 'planner',
                      'weight_dispatch', 'weight_dispatch', 'token_preprocess']
    assert experts.linear_fc1.weight0 is originals[0]
    assert experts.linear_fc2.weight0 is originals[1]
    assert experts.linear_fc1.weight2.main_grad.dtype == originals[0].main_grad.dtype
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_weight_helpers():
    # Import the production helpers without loading the hardware-specific package.
    tree = ast.parse((MOE / 'experts.py').read_text(encoding='utf-8'))
    names = {'DummyFunction', '_grouped_mlp_get_expert_weights',
             '_grouped_mlp_set_expert_weights', '_EchoHomeWeightGrad',
             '_EchoReplicaWeightGrad', 'echo_dispatch_home_weights',
             'echo_replica_weight'}
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in names]
    module = types.ModuleType('echo_test_weight_helpers')
    module.__dict__.update(torch=torch, List=list)
    exec(compile(tree, str(MOE / 'experts.py'), 'exec'), module.__dict__)
    return module


@pytest.fixture
def production(monkeypatch):
    overlap = load_file('echo_test_overlap', MOE / 'experts_overlap.py')
    helpers = load_weight_helpers()
    monkeypatch.setitem(sys.modules, 'hcu_megatron.core.transformer.moe.experts', helpers)
    utils = types.ModuleType('echo_test_moe_utils')
    utils.sort_chunks_by_idxs = lambda tensor, *args, **kwargs: (tensor, None)
    monkeypatch.setitem(sys.modules, 'megatron.core.transformer.moe.moe_utils', utils)
    monkeypatch.setattr(overlap, '_exchange', lambda out, inp, *args: out.copy_(inp))
    monkeypatch.setattr(overlap, '_apply_activation',
                        lambda experts, x, bias, probs: F.silu(x) * probs)
    return helpers, overlap, helpers


class FusedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, splits, fused, *weights):
        ctx.splits = splits
        ctx.fused = fused
        ctx.weights = weights
        ctx.save_for_backward(x)
        return torch.cat([part @ weight.T
                          for part, weight in zip(x.split(splits), weights)])

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx, dw = [], []
        # Match TE releases which decide the whole group's wgrad from slot 0.
        group_wgrad = ctx.weights[0].requires_grad
        for part, grad, weight in zip(x.split(ctx.splits), dy.split(ctx.splits), ctx.weights):
            dx.append(grad @ weight)
            wgrad = grad.T @ part
            if ctx.fused:
                assert weight.main_grad.dtype == ctx.weights[0].main_grad.dtype
                if group_wgrad:
                    weight.main_grad.add_(wgrad)
                if weight.requires_grad and group_wgrad:
                    weight.grad_added_to_main_grad = True
                    # A real TE dummy may be uninitialized. Poison it so using
                    # it instead of main_grad deterministically fails the test.
                    dw.append(torch.full_like(weight, float('nan')))
                else:
                    dw.append(None)
            else:
                dw.append(wgrad if weight.requires_grad and group_wgrad else None)
        return torch.cat(dx), None, None, *dw


class GroupedLinear(torch.nn.Module):
    def __init__(self, weights, fused):
        super().__init__()
        self.fused = fused
        self.num_gemms = len(weights) + 2
        for i, value in enumerate(weights):
            weight = Parameter(value.clone())
            weight.main_grad = torch.zeros_like(weight, dtype=torch.float64)
            weight.grad_added_to_main_grad = False
            self.register_parameter(f'weight{i}', weight)

    def forward(self, x, splits):
        weights = [getattr(self, f'weight{i}') for i in range(self.num_gemms)]
        return FusedGemm.apply(x, splits, self.fused, *weights), None


def make_experts(fused):
    torch.manual_seed(7)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(gradient_accumulation_fusion=fused),
        linear_fc1=GroupedLinear([torch.randn(4, 3) for _ in range(2)], fused),
        linear_fc2=GroupedLinear([torch.randn(3, 4) for _ in range(2)], fused),
    )


class Dispatcher:
    ep_group = None
    sort_input_by_local_experts = None

    @staticmethod
    def expert_dispatch_preprocess(metadata, *weights):
        if metadata.send:
            packed = torch.stack([weights[i].reshape(-1) for i in metadata.send])
        else:
            packed = weights[0].reshape(1, -1)[:0]
        return dict(permuted_weights=packed, weight_shape=weights[0].shape)


@pytest.fixture
def deepep(production, monkeypatch):
    """Run the real DeepEP autograd/unpack code with a loopback Buffer."""
    _, overlap, _ = production
    prefix = 'hcu_megatron.core.transformer.moe'
    monkeypatch.setitem(sys.modules, prefix + '.experts_overlap', overlap)
    # Extract the weight dispatchers from the shared module without importing
    # its hardware-specific token dispatcher dependencies, as for weight helpers.
    path = MOE / 'token_dispatcher.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'MoEElasticExpertMetadata', 'MoEElasticExpertDispatcher',
             '_DeepEPWeightDispatch', 'MoEDeepEPExpertDispatcher'}
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in names]
    module = types.ModuleType(prefix + '.token_dispatcher')
    module.__package__ = prefix
    module.__dict__.update(torch=torch, F=F, nullcontext=nullcontext,
                           List=List, Optional=Optional,
                           TransformerConfig=object, ProcessGroupCollection=object)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    monkeypatch.setitem(sys.modules, prefix + '.token_dispatcher', module)
    calls = []
    buffers = []

    class Event:
        def current_stream_wait(self):
            calls.append('deepep_wait')

    class Buffer:
        def get_dispatch_layout(self, indices, num_experts, **kwargs):
            calls.append(('layout', kwargs['async_finish']))
            return None, None, None, None, Event()

        def dispatch(self, chunks, **kwargs):
            calls.append(('dispatch', kwargs['async_finish']))
            return chunks.clone(), kwargs['topk_idx'].clone(), None, None, object(), Event()

        def combine(self, grad, handle, **kwargs):
            calls.append('combine')
            calls.append(('combine_buffer', buffers.index(self)))
            return grad.clone(), None, Event()

    def get_buffer(*args):
        buffer = Buffer()
        buffers.append(buffer)
        return buffer

    api = types.ModuleType('echo_test_fused_a2a')
    api.get_buffer = get_buffer
    api.get_hidden_bytes = lambda x: x.shape[1] * x.element_size()
    api.EventHandle = object
    api.EventOverlap = lambda handle: Event()
    monkeypatch.setitem(sys.modules, 'megatron.core.transformer.moe.fused_a2a', api)
    return module, calls


@pytest.mark.parametrize('backend', ['alltoall', 'deepep'])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('send,splits', [
    ([0], [2, 1, 3, 0]),  # Unsent home expert and unused Echo slot.
    ([0], [0, 1, 3, 0]),  # First home expert has no local tokens.
    ([0, 0], [2, 1, 3, 2]),  # Multiple replicas of the same home expert.
    ([], [2, 1, 0, 0]),  # Rank still participates in an empty weight exchange.
])
def test_overlap_matches_unfused_reference(production, deepep, backend, fused, send, splits):
    _, overlap, helpers = production
    experts = make_experts(fused)
    owners = [0, 1] + send + [0] * (2 - len(send))
    metadata = types.SimpleNamespace(
        send=send, input_splits=[len(send)], output_splits=[len(send)],
        input_chunks_per_rank=None,
        has_experts_per_slot=[int(i < len(send)) for i in range(2)],
    )
    dispatcher = Dispatcher()
    if backend == 'deepep':
        module, _ = deepep

        class ChunkDispatcher(module.MoEDeepEPExpertDispatcher):
            # Use small FP32 matrices for strict reference comparisons. The
            # production BF16 packer is exercised separately below.
            def expert_dispatch_preprocess(self, meta, *weights):
                packed = Dispatcher.expert_dispatch_preprocess(meta, *weights)
                return dict(chunks=packed['permuted_weights'],
                            indices=torch.arange(len(meta.send)).reshape(-1, 1),
                            weight_shape=packed['weight_shape'],
                            weight_numel=weights[0].numel(), chunks_per_weight=1)

        dispatcher = ChunkDispatcher.__new__(ChunkDispatcher)
        dispatcher.comm_group = None
        dispatcher.config = types.SimpleNamespace(moe_num_echo_experts=2)
    params = [getattr(layer, f'weight{i}')
              for layer in (experts.linear_fc1, experts.linear_fc2) for i in range(2)]
    refs = [p.detach().clone().requires_grad_() for p in params]
    # Simulate the DDP hook contract, including the no-double-accumulation flag.
    handles = []
    if fused:
        def ddp_hook(param):
            if not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad)
            param.grad = None
        handles = [p.register_post_accumulate_grad_hook(ddp_hook) for p in params]

    runs = []
    # Keep two forwards in flight before backward to detect slot-buffer reuse.
    for _ in range(2):
        x = torch.randn(sum(splits), 3, requires_grad=True)
        probs = torch.rand(sum(splits), requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        ref_probs = probs.detach().clone().requires_grad_()
        out, _ = overlap._grouped_mlp_forward_with_dispatch_overlap(
            experts, x, torch.tensor(splits), probs, 2, [0, 1], [2, 3],
            dispatcher, metadata, metadata, None, None, None,
        )
        expected = []
        for xx, pp, owner in zip(ref_x.split(splits), ref_probs.split(splits), owners):
            expected.append((F.silu(xx @ refs[owner].T) * pp[:, None]) @ refs[2 + owner].T)
        ref_out = torch.cat(expected)
        torch.testing.assert_close(out, ref_out)
        runs.append((out, ref_out, x, ref_x, probs, ref_probs))
        assert all(getattr(layer, f'weight{i}') is params[offset + i]
                   for layer, offset in ((experts.linear_fc1, 0), (experts.linear_fc2, 2))
                   for i in range(2))
    for out, ref_out, x, ref_x, probs, ref_probs in reversed(runs):
        out.square().sum().backward()
        ref_out.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(probs.grad, ref_probs.grad)
    for param, ref in zip(params, refs):
        actual = param.main_grad if fused else param.grad
        torch.testing.assert_close(actual, ref.grad.to(actual.dtype), rtol=2e-5, atol=2e-5)
    for handle in handles:
        handle.remove()
    if backend == 'deepep':
        _, calls = deepep
        # Two outstanding forwards: reverse layer order, FC2 before FC1 in each.
        assert [call[1] for call in calls if isinstance(call, tuple)
                and call[0] == 'combine_buffer'] == [3, 2, 1, 0]


@pytest.mark.parametrize('send', [[], [0], [0, 0], [1, 0]])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_deepep_chunk_pack_transport_and_stream_wait(deepep, production, monkeypatch,
                                                   send, asynchronous):
    module, calls = deepep
    _, overlap, _ = production
    dispatcher = module.MoEDeepEPExpertDispatcher.__new__(module.MoEDeepEPExpertDispatcher)
    dispatcher.config = types.SimpleNamespace(moe_num_echo_experts=2)
    dispatcher.comm_group = None
    dispatcher.weight_chunk_size = 8  # Matrices need two chunks and padding.
    dispatcher._materialize_dispatch_metadata = lambda meta: None
    mapping = torch.zeros(2, 2, dtype=torch.bool)
    for slot, owner in enumerate(send):
        mapping[owner, slot] = True
    meta = types.SimpleNamespace(local_to_global_routing_map=mapping,
                                 has_experts_per_slot=[int(i < len(send)) for i in range(2)])
    weights = [torch.randn(3, 5, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    # Only the device guard is bypassed: real BF16 packing, padding, dispatch
    # autograd, slot sorting and unpacking all run on CPU.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    packed = dispatcher.expert_dispatch_preprocess(meta, *weights)
    assert 'permuted_weights' not in packed
    assert packed['chunks'].shape == (2 * len(send), 8)

    class Stream:
        def wait_stream(self, other):
            calls.append('side_wait_caller')

        def wait_event(self, event):
            calls.append('caller_wait_done')

    class CudaEvent:
        def record(self, stream):
            calls.append('record_done')

    caller, side = Stream(), Stream()
    current = [caller]

    @contextmanager
    def stream_context(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: current[0])
    monkeypatch.setattr(torch.cuda, 'stream', stream_context)
    monkeypatch.setattr(torch.cuda, 'Event', CudaEvent)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda *args: None)
    received, ticket = overlap._start_weight_dispatch(
        dispatcher, meta, packed, side if asynchronous else None,
    )
    assert current[0] is caller
    assert ('dispatch', asynchronous) in calls
    assert 'caller_wait_done' not in calls  # Home GEMM can be enqueued now.
    unpacked = overlap._finish_weight_dispatch(dispatcher, meta, packed, received, ticket)
    if asynchronous:
        assert calls.index('deepep_wait') < calls.index('record_done') < calls.index('caller_wait_done')
    for slot, owner in enumerate(send):
        torch.testing.assert_close(unpacked[slot], weights[owner])
    sum(w.float().sum() for w in unpacked).backward()
    for owner, weight in enumerate(weights):
        torch.testing.assert_close(weight.grad, torch.full_like(weight, send.count(owner)))
    assert calls.count('combine') == 1


def test_sync_replica_buffers_are_independent_and_transport_real_grad(production):
    grad, _, helpers = production
    experts = make_experts(True)
    home = experts.linear_fc1.weight0
    replicas = []
    for _ in range(2):
        (source,) = grad.echo_dispatch_home_weights([home], True)
        helpers._grouped_mlp_set_expert_weights(experts, 'fc1', [source.clone()], [2])
        replicas.append(experts.linear_fc1.weight2)
    assert replicas[0].main_grad.data_ptr() != replicas[1].main_grad.data_ptr()
    replicas[0].main_grad.fill_(2)
    replicas[1].main_grad.fill_(3)
    for replica in reversed(replicas):
        replica.backward(torch.full_like(replica, float('nan')))
    torch.testing.assert_close(home.main_grad, torch.full_like(home.main_grad, 5))
    assert home.grad_added_to_main_grad
    assert torch.count_nonzero(home.grad) == 0


@pytest.mark.parametrize('fused', [False, True])
def test_restoring_home_preserves_ddp_flag(production, fused):
    _, _, helpers = production
    experts = make_experts(fused)
    home = experts.linear_fc1.weight0
    helpers._grouped_mlp_set_expert_weights(experts, 'fc1', [home], [0])
    assert experts.linear_fc1.weight0 is home
    assert home.grad_added_to_main_grad is False


def test_bf16_replica_uses_fp32_buffer_and_ignores_dummy(production):
    grad, _, _ = production
    home = Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
    home.main_grad = torch.zeros_like(home, dtype=torch.float32)
    home.grad_added_to_main_grad = False
    (source,) = grad.echo_dispatch_home_weights([home], True)
    replica = grad.echo_replica_weight(source.clone(), home.main_grad.dtype)
    assert replica.dtype == torch.bfloat16
    assert replica.main_grad.dtype == torch.float32
    replica.main_grad.fill_(2)
    replica.backward(torch.full_like(replica, float('nan')))
    torch.testing.assert_close(home.main_grad, torch.full_like(home.main_grad, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/ROCm and TE')
def test_real_te_fused_replica_gradient(production):
    """Run the same bridge against the installed TE on a training node."""
    te = pytest.importorskip('transformer_engine.pytorch')
    grad, _, helpers = production
    layer = te.GroupedLinear(
        num_gemms=2, in_features=32, out_features=32, bias=False,
        fuse_wgrad_accumulation=True, params_dtype=torch.bfloat16, device='cuda',
    )
    home = layer.weight0
    home.main_grad = torch.zeros_like(home, dtype=torch.float32)
    home.grad_added_to_main_grad = False
    delattr(layer, 'weight1')
    experts = types.SimpleNamespace(
        config=types.SimpleNamespace(gradient_accumulation_fusion=True), linear_fc1=layer,
    )
    (source,) = grad.echo_dispatch_home_weights([home], True)
    helpers._grouped_mlp_set_expert_weights(experts, 'fc1', [source.clone()], [1])
    x = torch.randn(32, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_w = home.detach().clone().requires_grad_()
    output = layer(x, [16, 16])
    expected = ref_x @ ref_w.T
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    output.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(home.main_grad, ref_w.grad.float(), rtol=0.03, atol=0.1)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.03, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/ROCm and Triton')
class TestTAOTGPUPlanner:
    @pytest.mark.parametrize('algorithm', ['sinkhorn', 'approx_bin_packing', 'one_shot_greedy'])
    def test_rerouting_preserves_assignments_probabilities_and_expert_output(self, algorithm):
        pytest.importorskip('triton')
        planner_module = load_file('taot_gpu_planner', MOE / 'offloading_planner.py')
        plan_fn = inspect.unwrap(planner_module.gen_offloading_plan)
        # Four logical EP ranks on one GPU. This tests real planning/rerouting
        # kernels, not communication. Every token initially selects experts 0/1.
        routes = torch.zeros(64, 8, dtype=torch.bool, device='cuda')
        routes[:, :2] = True
        probs = torch.zeros(64, 8, device='cuda')
        probs[:, 0] = 0.25
        probs[:, 1] = 0.75
        counts = torch.zeros(4, 8, dtype=torch.int32, device='cuda')
        counts[0] = routes.sum(0).int()
        rerouted, new_probs, expert_map = plan_fn(
            routes, probs, counts, 0, 4, assignment_algorithm=algorithm,
            ranks_per_node=2, _force_triton=False,
        )
        assert rerouted.shape == new_probs.shape == (64, 12)
        torch.testing.assert_close(rerouted.sum(1), routes.sum(1))
        torch.testing.assert_close(new_probs.sum(1), probs.sum(1))
        torch.testing.assert_close(new_probs.masked_select(~rerouted),
                                   torch.zeros_like(new_probs.masked_select(~rerouted)))
        assert ((expert_map > 0).sum(0) <= 1).all()
        # Independent oracle for the interleaved [home0, home1, echo] layout.
        values = torch.arange(1, 9, dtype=torch.float32, device='cuda')
        physical = torch.zeros(12, device='cuda')
        for rank in range(4):
            physical[rank * 3:rank * 3 + 2] = values[rank * 2:rank * 2 + 2]
            owner = expert_map[:, rank].nonzero().flatten()
            if owner.numel():
                physical[rank * 3 + 2] = values[owner[0]]
        assert rerouted[:, [2, 5, 8, 11]].any(), 'fixture must exercise Echo slots'
        torch.testing.assert_close(new_probs @ physical, probs @ values)

    def test_triton_sinkhorn_respects_capacity(self):
        pytest.importorskip('triton')
        module = load_file('taot_gpu_sinkhorn', MOE / 'offloading_planner.py')
        counts = torch.zeros(4, 8, dtype=torch.int32, device='cuda')
        counts[0, 0] = 64
        reference = module.gen_assignment_sinkhorn(counts, 0, 4, _force_triton=False)
        actual = module.gen_assignment_sinkhorn(counts, 0, 4, _force_triton=True)
        # This fixture has one hot expert, so no ambiguous owner tie exists.
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual.sum() == 48


@pytest.fixture(scope='module')
def taot_distributed_runtime():
    if not all(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        pytest.skip('requires torchrun with EP8 or EP16')
    world = int(os.environ['WORLD_SIZE'])
    if world not in (8, 16):
        pytest.skip(f'requires 8 or 16 ranks, got {world}')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA/ROCm')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group('nccl', init_method='env://', timeout=timedelta(seconds=120))
    assert dist.get_world_size() == world
    yield dist.group.WORLD
    if owns_group:
        dist.destroy_process_group()


class TestTAOTDistributed:
    def test_plan_synchronization(self, taot_distributed_runtime, planner):
        group = taot_distributed_runtime
        rank = dist.get_rank(group)
        values = (torch.full((2, 2), rank == 0, dtype=torch.bool, device='cuda'),
                  torch.full((2, 2), rank + 3, dtype=torch.int64, device='cuda'),
                  torch.full((2, 2), rank + 5, dtype=torch.int32, device='cuda'))
        actual = planner._synchronize_offloading_plan(*values, group)
        for value, expected in zip(actual, (True, 3, 5)):
            assert (value == expected).all()

    @pytest.mark.parametrize('backend', ['alltoall', 'deepep'])
    @pytest.mark.parametrize('overlap', [False, True])
    def test_ring_weight_transport_and_fused_gradient(
        self, taot_distributed_runtime, monkeypatch, backend, overlap,
    ):
        """Real ring dispatch/combine, with rank-dependent gradient sent home.

        Exercises the production transport Functions and fused bridges. The
        GEMM is represented by an explicit main_grad write; real TE is covered
        separately above. Does not claim full-model EP/DDP integration coverage.
        """
        group = taot_distributed_runtime
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        helpers = load_weight_helpers()
        exchange_module = load_file('taot_real_exchange', MOE / 'experts_overlap.py')
        if backend == 'deepep':
            if os.environ.get('TAOT_TEST_DEEPEP') != '1':
                pytest.skip('set TAOT_TEST_DEEPEP=1 on every rank')
            deep_ep = pytest.importorskip('deep_ep')
            from deep_ep.utils import EventHandle, EventOverlap
            # Use the repository's actual buffer sizing/cache implementation.
            api = _definitions(
                ROOT / '3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py',
                {'get_buffer', 'get_hidden_bytes'}, Buffer=deep_ep.Buffer, _buffer=None,
            )
            api.EventHandle, api.EventOverlap = EventHandle, EventOverlap
            monkeypatch.setitem(sys.modules, 'megatron.core.transformer.moe.fused_a2a', api)
            transport = _definitions(MOE / 'token_dispatcher.py', {'_DeepEPWeightDispatch'},
                                     nullcontext=nullcontext)
        home = Parameter(torch.full((1, 8192), float(rank + 1),
                                    device='cuda', dtype=torch.bfloat16))
        home.main_grad = torch.full_like(home, 5, dtype=torch.float32)
        home.grad_added_to_main_grad = False
        source, = helpers.echo_dispatch_home_weights([home], True)
        stream = torch.cuda.Stream() if overlap else torch.cuda.current_stream()
        ticket = {}
        if backend == 'alltoall':
            send = [int(i == (rank + 1) % world) for i in range(world)]
            recv = [int(i == (rank - 1) % world) for i in range(world)]
            received = exchange_module._WeightExchange.apply(
                source, group, send, recv, stream, ticket,
            )
        else:
            indices = torch.full((1, 1), (rank + 1) % world,
                                 device='cuda', dtype=torch.int64)
            received, _ = transport._DeepEPWeightDispatch.apply(
                source, indices, world, group, stream if overlap else None, ticket,
            )
        if 'done' in ticket:
            torch.cuda.current_stream().wait_event(ticket['done'])
        local_error = None
        try:
            assert received.shape == home.shape
            torch.testing.assert_close(
                received, torch.full_like(received, float((rank - 1) % world + 1)),
                rtol=0, atol=0,
            )
        except Exception as error:
            local_error = f'rank {rank}: {error}'
        # Complete backward on every rank even if a value assertion failed.
        replica = helpers.echo_replica_weight(received, home.main_grad.dtype)
        replica.main_grad.fill_(rank + 1)
        replica.backward(torch.full_like(replica, float('nan')))
        try:
            expected = 5 + (rank + 1) % world + 1
            torch.testing.assert_close(home.main_grad, torch.full_like(home.main_grad, expected),
                                       rtol=0, atol=0)
            assert home.grad_added_to_main_grad
            assert torch.count_nonzero(home.grad) == 0
        except Exception as error:
            local_error = (local_error or '') + f' gradient rank {rank}: {error}'
        errors = [None] * world
        dist.all_gather_object(errors, local_error, group=group)
        assert not any(errors), '\n'.join(e for e in errors if e)
