# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Core UltraEP framework/adaptive tests and EP8 model parity.

CPU tests cover framework contracts and selection policy. EP8 tests exercise
real weight sync, rerouting and gradient reduction through GPT training. Run::

    python -m torch.distributed.run --nproc-per-node=8 -m pytest -s -vv \
        tests/unit_tests/moe/test_ultraep_framework.py
"""

import argparse
import os
import sys
import types
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from hcu_megatron.core.distributed.distributed_data_parallel import (
    DistributedDataParallel as HcuDistributedDataParallel,
)
from hcu_megatron.core.distributed.param_and_grad_buffer import (
    _compute_full_param_layout_ultraep_wrapper,
    _distributed_data_parallel_init_wrapper,
    _param_and_grad_buffer_init_wrapper,
)
from hcu_megatron.core.transformer.moe import eplb_manager as eplb_manager_module
from hcu_megatron.core.transformer.moe import moe_layer
from hcu_megatron.core.transformer.moe.experts import (
    te_grouped_linear_sharded_state_dict_wrapper,
    te_grouped_mlp_sharded_state_dict_wrapper,
)
from hcu_megatron.features_manager.moe.ultraep_feature import (
    UltraEPFeature,
    _get_param_groups_ultraep_wrapper,
)
from hcu_megatron.training.checkpointing import generate_state_dict_ultraep_wrapper
from hcu_megatron.training.ultraep_adaptive import AdaptiveConfig, AdaptiveController
from hcu_megatron.training.ultraep_autotune import (
    ultraep_autotune_train_step_wrapper,
)


class FakeEventHandle:
    def __init__(self):
        self.wait_count = 0

    def current_stream_wait(self):
        self.wait_count += 1


def _feature_args(**overrides):
    values = dict(
        moe_enable_ultraep=True,
        moe_num_redundant_experts_per_rank=2,
        moe_ultraep_autotune=True,
        moe_ultraep_autotune_start_iteration=3,
        moe_ultraep_autotune_grad_reduce_max_sms=None,
        cuda_graph_impl="none",
        optimizer_cuda_graph=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class PatchRecorder:
    def __init__(self):
        self.patches = []

    def register_patch(self, target, replacement, **kwargs):
        self.patches.append((target, replacement.__name__, kwargs))


class TestUltraEPFeature:
    def test_register_args_defaults(self):
        parser = argparse.ArgumentParser()
        UltraEPFeature().register_args(parser)
        args = parser.parse_args([])
        assert args.moe_enable_ultraep is False
        assert args.moe_num_redundant_experts_per_rank == 0
        assert args.moe_ultraep_autotune is False
        assert args.moe_ultraep_adaptive is False
        assert args.moe_ultraep_autotune_start_iteration == 3
        assert args.moe_ultraep_autotune_grad_reduce_max_sms is None

    @pytest.mark.parametrize('tp,pp', [(1, 1), (2, 1), (1, 2)])
    def test_validate_accepts_supported_configuration(self, tp, pp):
        args = _feature_args(moe_ultraep_adaptive=True, moe_ultraep_autotune_grad_reduce_max_sms=48,
                             tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
                             context_parallel_size=1)
        assert UltraEPFeature().validate_args(args) is args

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"moe_num_redundant_experts_per_rank": 0}, "must be > 0"),
            ({"moe_enable_ultraep": False}, "requires --moe-enable-ultraep"),
            ({"moe_ultraep_autotune_start_iteration": 0}, "must be positive"),
            ({"moe_ultraep_autotune_grad_reduce_max_sms": 47}, "positive even"),
            ({"cuda_graph_impl": "local"}, "CUDA graph capture"),
            ({"optimizer_cuda_graph": True}, "optimizer-cuda-graph"),
            ({"moe_ultraep_adaptive": True, "overlap_moe_expert_parallel_comm": True}, "EP overlap"),
            ({"moe_ultraep_adaptive": True, "expert_tensor_parallel_size": 2}, "CP1 / ETP1"),
            ({"moe_ultraep_adaptive": True, "context_parallel_size": 2}, "CP1 / ETP1"),
        ],
    )
    def test_validate_rejects_invalid_configuration(self, overrides, message):
        with pytest.raises(AssertionError, match=message):
            UltraEPFeature().validate_args(_feature_args(**overrides))

    def test_patch_registration_is_conditional(self):
        feature = UltraEPFeature()
        disabled = PatchRecorder()
        feature.register_patches(disabled, _feature_args(moe_enable_ultraep=False))
        assert disabled.patches == []

        base = PatchRecorder()
        feature.register_patches(base, _feature_args(moe_ultraep_autotune=False))
        tuned = PatchRecorder()
        feature.register_patches(tuned, _feature_args())
        adaptive = PatchRecorder()
        args = _feature_args(moe_ultraep_autotune=False, moe_ultraep_adaptive=True)
        feature.validate_args(args)
        feature.register_patches(adaptive, args)

        train_step = "hcu_megatron.training.training.train_step"
        assert train_step not in {item[0] for item in base.patches}
        assert train_step in {item[0] for item in tuned.patches}
        assert train_step in {item[0] for item in adaptive.patches}
        assert len(tuned.patches) == len(base.patches) + 1
        assert not any('moe_layer.MoELayer.' in item[0] for item in tuned.patches)
        names = {item[1] for item in tuned.patches}
        assert "te_grouped_linear_sharded_state_dict_wrapper" in names
        assert "te_grouped_mlp_sharded_state_dict_wrapper" in names

    def test_disabling_ultraep_clears_adaptive_in_both_argument_namespaces(self):
        feature = UltraEPFeature()
        args = _feature_args(
            moe_enable_ultraep=False, moe_ultraep_autotune=False,
            moe_ultraep_adaptive=True, moe_ultraep_adaptive_samples=2,
            overlap_moe_expert_parallel_comm=True,
        )
        patch_args = deepcopy(args)
        assert feature.validate_args(args) is args
        recorder = PatchRecorder()
        feature.register_patches(recorder, patch_args)
        assert not args.moe_ultraep_adaptive and not patch_args.moe_ultraep_adaptive
        assert recorder.patches == []

    @pytest.mark.parametrize("raises", [False, True])
    def test_optimizer_group_wrapper_hides_and_restores_replicas(self, raises):
        model = nn.Module()
        model.master = nn.Parameter(torch.ones(1))
        model.replica = nn.Parameter(torch.ones(1))
        model.replica.is_eplb_replica = True

        def original(model_chunks, config, overrides):
            assert model_chunks[0].master.requires_grad
            assert not model_chunks[0].replica.requires_grad
            if raises:
                raise RuntimeError("expected")
            return "groups"

        wrapped = _get_param_groups_ultraep_wrapper(original)
        if raises:
            with pytest.raises(RuntimeError, match="expected"):
                wrapped([model], None, None)
        else:
            assert wrapped([model], None, None) == "groups"
        assert model.replica.requires_grad


class TestEPLBManager:

    def test_grad_reduce_invalidates_reused_ws_task_metadata(self, monkeypatch):
        manager = eplb_manager_module.EPLBManager.__new__(eplb_manager_module.EPLBManager)
        native = SimpleNamespace(invalidate_weight_sync_task_cache=MagicMock())
        completion = object()
        manager.runtime = SimpleNamespace(runtime=native, grad_reduce=MagicMock(return_value=completion))
        stream = SimpleNamespace(wait_stream=MagicMock())
        manager._prepare_peer_fence = lambda: stream
        manager._fence_on_comm_stream = MagicMock()
        monkeypatch.setattr(torch.cuda, 'current_stream', lambda: 'compute')
        assert manager.grad_reduce(7) is completion
        stream.wait_stream.assert_called_once_with('compute')
        manager.runtime.grad_reduce.assert_called_once_with(layer_id=7, async_finish=True)
        native.invalidate_weight_sync_task_cache.assert_called_once()
        manager._fence_on_comm_stream.assert_called_once_with(stream)

    @pytest.mark.parametrize('bound_device', [None, 'cuda:0'])
    def test_peer_group_initialization_only_requires_ep_members(self, monkeypatch, bound_device):
        manager = eplb_manager_module.EPLBManager.__new__(eplb_manager_module.EPLBManager)
        manager.group, manager.num_ranks = object(), 2
        manager._peer_group = manager._replica_comm_stream = None
        stream, peer_group = object(), object()
        manager.runtime = SimpleNamespace(get_comm_stream=MagicMock(return_value=stream))
        monkeypatch.setattr(dist, 'group', SimpleNamespace(WORLD=SimpleNamespace(bound_device_id=bound_device)))
        split = MagicMock(return_value=peer_group)
        create = MagicMock(return_value=peer_group)
        monkeypatch.setattr(dist, 'split_group', split)
        monkeypatch.setattr(dist, 'new_group', create)
        monkeypatch.setattr(dist, 'get_process_group_ranks', lambda group: [2, 3])
        monkeypatch.setattr(torch, 'zeros', MagicMock())
        assert manager._prepare_peer_fence() is stream
        assert manager._prepare_peer_fence() is stream
        assert manager._peer_group is peer_group
        manager.runtime.get_comm_stream.assert_called_once()
        if bound_device is not None:
            split.assert_called_once_with(parent_pg=manager.group, split_ranks=[[0, 1]])
            create.assert_not_called()
        else:
            create.assert_called_once_with(ranks=[2, 3], use_local_synchronization=True)
            split.assert_not_called()


class TestDistributedParameterHandling:
    @staticmethod
    def _params():
        params = [nn.Parameter(torch.zeros(2)) for _ in range(4)]
        params[1].is_eplb_replica = True
        params[3].is_eplb_replica = True
        return params

    @pytest.mark.parametrize("use_keyword", [False, True])
    def test_param_buffer_filters_replica_parameters(self, use_keyword):
        captured = []

        def original(self, *args, **kwargs):
            captured.extend(kwargs["params_with_names"] if kwargs else args[3])

        wrapped = _param_and_grad_buffer_init_wrapper(original)
        params_with_names = [(param, f"p{index}") for index, param in enumerate(self._params())]
        if use_keyword:
            wrapped(object(), params_with_names=params_with_names)
        else:
            wrapped(object(), "ddp", "param", "grad", params_with_names)
        assert [name for _, name in captured] == ["p0", "p2"]

    @pytest.mark.parametrize("raises", [False, True])
    def test_ddp_init_temporarily_hides_and_restores_replicas(self, raises):
        module = nn.Module()
        module.master = nn.Parameter(torch.ones(1))
        module.replica = nn.Parameter(torch.ones(1))
        module.replica.is_eplb_replica = True

        def original(self, config, ddp_config, wrapped_module):
            assert wrapped_module.master.requires_grad
            assert not wrapped_module.replica.requires_grad
            if raises:
                raise RuntimeError("expected")

        wrapped = _distributed_data_parallel_init_wrapper(original)
        if raises:
            with pytest.raises(RuntimeError, match="expected"):
                wrapped(object(), None, None, module)
        else:
            wrapped(object(), None, None, module)
        assert module.replica.requires_grad

    def test_full_param_layout_filters_replicas(self):
        params = self._params()
        wrapped = _compute_full_param_layout_ultraep_wrapper(
            lambda all_params, marker=None: (all_params, marker)
        )
        filtered, marker = wrapped(params, marker="ok")
        assert filtered == [params[0], params[2]]
        assert marker == "ok"

    @pytest.mark.parametrize('use_hcu_hook', [False, True])
    def test_master_ready_wrapper_switches_between_ultraep_and_baseline(self, use_hcu_hook):
        from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
        from hcu_megatron.core.distributed.distributed_data_parallel import ultraep_backward_post_hook_wrapper

        param = nn.Parameter(torch.ones(2))
        param.is_eplb_master = True
        param._eplb_manager = SimpleNamespace(ultraep_active=True)
        param.main_grad = torch.zeros_like(param)
        param.grad_added_to_main_grad = False
        bucket = MagicMock()
        ddp = SimpleNamespace(
            param_to_bucket_group={param: bucket},
            ddp_config=SimpleNamespace(overlap_grad_reduce=True, overlap_param_gather=True),
            force_all_reduce=False,
            _make_forward_pre_hook=MagicMock(),
        )
        original = (HcuDistributedDataParallel if use_hcu_hook else DistributedDataParallel)._make_backward_post_hook
        hook = ultraep_backward_post_hook_wrapper(original)(ddp, param)
        assert param._ddp_bucket_group is bucket
        assert param._eplb_forward_pre_hook is ddp._make_forward_pre_hook.return_value
        args = SimpleNamespace(
            gradient_accumulation_fusion=True, delay_wgrad_compute=False,
            recompute_in_advance=False, recompute_in_bubble=False,
        )
        with patch('hcu_megatron.core.distributed.distributed_data_parallel.get_args', return_value=args), patch(
            'hcu_megatron.core.distributed.distributed_data_parallel.is_graph_capturing', return_value=False,
        ), patch(
            'megatron.core.distributed.distributed_data_parallel.is_graph_capturing', return_value=False,
        ):
            for index, active in enumerate((True, False, True, False), start=1):
                param._eplb_manager.ultraep_active = active
                param.grad = torch.ones_like(param)
                hook()
                torch.testing.assert_close(param.main_grad, torch.full_like(param, index))
                assert param.grad is None
                assert bucket.register_grad_ready.call_count == index // 2
        assert bucket.register_grad_ready.call_args_list == [call(param, False)] * 2


def _grouped_linear(num_experts, numel):
    module = nn.Module()
    module.num_gemms = num_experts
    for index in range(num_experts):
        setattr(module, f"weight{index}", nn.Parameter(torch.zeros(numel)))
    return module


def _fake_moe_layer():
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    experts = object.__new__(TEGroupedMLP)
    nn.Module.__init__(experts)
    experts.linear_fc1 = _grouped_linear(2, 6)
    experts.linear_fc2 = _grouped_linear(2, 3)
    runtime = MagicMock()
    manager = SimpleNamespace(
        num_local_master_experts=2,
        num_local_redundant_experts=1,
        num_local_physical_experts=3,
        num_global_physical_experts=6,
        local_physical_expert_indices=[3, 4, 5],
        expert_fc1_numel=6,
        expert_fc2_numel=3,
        local_replica_fc1_weight_buffer=torch.zeros(1, 6),
        local_replica_fc2_weight_buffer=torch.zeros(1, 3),
        local_replica_fc1_grad_buffer=torch.zeros(1, 6),
        local_replica_fc2_grad_buffer=torch.zeros(1, 3),
        runtime=runtime,
        weight_sync=runtime.weight_sync,
        grad_reduce=runtime.grad_reduce,
    )
    dispatcher = SimpleNamespace(
        num_local_experts=2,
        num_experts=4,
        local_expert_indices=[2, 3],
        permute_idx_device=torch.device("cpu"),
        tp_size=1,
    )
    layer = SimpleNamespace(
        eplb_manager=manager,
        experts=experts,
        token_dispatcher=dispatcher,
        recompute_token_dispatcher=None,
        layer_number=4,
        _eplb_master_ptrs_registered=False,
        _eplb_grad_reduce_event_handles={},
    )
    return layer


class TestMoELayerUltraEP:
    @pytest.mark.parametrize('enabled', [False, True])
    def test_init_combines_common_setup_and_optional_ultraep(self, monkeypatch, enabled):
        from megatron.core import parallel_state

        layer = _fake_moe_layer()
        manager = eplb_manager_module.EPLBManager.__new__(eplb_manager_module.EPLBManager)
        manager.__dict__.update(vars(layer.eplb_manager))
        manager.adaptive_config, manager.ultraep_active = AdaptiveConfig(), False
        manager._adaptive_layers, events = [], []
        config = SimpleNamespace(
            moe_enable_ultraep=enabled, moe_ultraep_adaptive=enabled,
            num_moe_experts=4, recompute_granularity='selective', recompute_modules=['router'],
        )
        group, pg_collection = object(), object()
        monkeypatch.setattr(eplb_manager_module, 'HAVE_EPLB', enabled)
        monkeypatch.setattr(parallel_state, 'get_expert_model_parallel_group', lambda: group)

        def create_manager(**kwargs):
            events.append('manager')
            return manager

        factory = MagicMock(side_effect=create_manager)
        monkeypatch.setattr(eplb_manager_module, 'get_or_create_eplb_manager', factory)
        monkeypatch.setattr(moe_layer, 'get_adaptor_args', lambda: SimpleNamespace(
            integrate_recompute_to_ep_comm_overlap=False,
        ))

        def init(self, config, **kwargs):
            events.append('original')
            self.config = config
            self.num_local_experts = 2
            self.local_expert_indices = [2, 3]
            self.experts.num_local_experts = 2

        original = MagicMock(side_effect=init)
        wrapped = moe_layer.moe_layer_init_wrapper(original)
        wrapped(layer, config, pg_collection=pg_collection, is_mtp_layer=True, name='moe')
        original.assert_called_once_with(
            layer, config, submodules=None, layer_number=None,
            pg_collection=pg_collection, is_mtp_layer=True, name='moe',
        )
        assert layer.router_recompute and not layer.experts_recompute
        assert layer.eplb_enabled == enabled and config.num_moe_experts == 4
        if enabled:
            assert events == ['manager', 'original']
            factory.assert_called_once_with(config=config, ep_group=group)
            assert layer.eplb_manager is manager and not manager.ultraep_active
            assert len(manager._adaptive_layers) == 1 and manager._adaptive_layers[0][0] is layer
            assert layer.experts.num_local_experts == layer.token_dispatcher.num_local_experts == 2
            assert layer.experts.linear_fc1.num_gemms == 2
        else:
            assert events == ['original'] and layer.eplb_manager is None
            factory.assert_not_called()
            assert layer.experts.linear_fc1.num_gemms == 2

    def test_registers_master_and_replica_parameters_and_dispatcher(self):
        layer = _fake_moe_layer()
        moe_layer._eplb_register_redundant_experts(layer)

        for linear, buffer, grad_buffer in (
            (
                layer.experts.linear_fc1,
                layer.eplb_manager.local_replica_fc1_weight_buffer,
                layer.eplb_manager.local_replica_fc1_grad_buffer,
            ),
            (
                layer.experts.linear_fc2,
                layer.eplb_manager.local_replica_fc2_weight_buffer,
                layer.eplb_manager.local_replica_fc2_grad_buffer,
            ),
        ):
            assert linear.weight0.is_eplb_master
            assert linear.weight1.is_eplb_master
            assert linear.weight2.is_eplb_replica
            assert linear.weight2.data.data_ptr() == buffer[0].data_ptr()
            assert linear.weight2.main_grad.data_ptr() == grad_buffer[0].data_ptr()
            assert linear.num_gemms == 3
            assert linear.num_local_master_experts == 2

        assert layer.experts.num_local_experts == 3
        assert layer.token_dispatcher.num_local_experts == 3
        assert layer.token_dispatcher.num_experts == 6
        assert layer.token_dispatcher.local_expert_indices == [3, 4, 5]
        assert torch.equal(
            layer.token_dispatcher.sort_input_by_local_experts,
            torch.tensor([0, 3, 1, 4, 2, 5]),
        )
        assert torch.equal(
            layer.token_dispatcher.restore_output_by_local_experts,
            torch.tensor([0, 2, 4, 1, 3, 5]),
        )

    def test_master_pointer_registration_is_idempotent(self):
        layer = _fake_moe_layer()
        moe_layer._eplb_register_redundant_experts(layer)
        for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
            for index in range(2):
                getattr(linear, f"weight{index}").main_grad = torch.zeros_like(
                    getattr(linear, f"weight{index}")
                )

        moe_layer._eplb_register_master_experts(layer)
        moe_layer._eplb_register_master_experts(layer)
        call_args = layer.eplb_manager.runtime.construct_local_master_ptr_pool.call_args
        layer.eplb_manager.runtime.construct_local_master_ptr_pool.assert_called_once()
        assert call_args.kwargs["layer_id"] == 4
        assert len(call_args.kwargs["fc1_weights"]) == 2
        assert len(call_args.kwargs["fc2_grads"]) == 2

    def test_grad_reduce_events_are_keyed_by_virtual_layer(self):
        layer = _fake_moe_layer()
        first, second = FakeEventHandle(), FakeEventHandle()
        layer.eplb_manager.runtime.grad_reduce.side_effect = [first, second]
        layer.eplb_manager.runtime.wait_grad_reduce.side_effect = (
            lambda event, layer_id=None: event.current_stream_wait()
        )

        moe_layer._eplb_start_grad_reduce(layer, 11)
        moe_layer._eplb_start_grad_reduce(layer, 12)
        with pytest.raises(AssertionError, match="launched twice"):
            moe_layer._eplb_start_grad_reduce(layer, 11)
        moe_layer._eplb_finish_grad_reduce(layer, 12)
        moe_layer._eplb_finish_grad_reduce(layer, 11)
        assert first.wait_count == second.wait_count == 1
        assert layer._eplb_grad_reduce_event_handles == {}
        assert layer.eplb_manager.runtime.wait_grad_reduce.call_args_list == [
            call(second, layer_id=12),
            call(first, layer_id=11),
        ]

    def test_master_grad_ready_only_for_overlap_buckets(self):
        layer = _fake_moe_layer()
        moe_layer._eplb_register_redundant_experts(layer)
        overlap_bucket = MagicMock()
        overlap_bucket.ddp_config.overlap_grad_reduce = True
        sync_bucket = MagicMock()
        sync_bucket.ddp_config.overlap_grad_reduce = False
        layer.experts.linear_fc1.weight0._ddp_bucket_group = overlap_bucket
        layer.experts.linear_fc1.weight1._ddp_bucket_group = sync_bucket
        moe_layer._eplb_register_master_grad_ready(layer)
        overlap_bucket.register_grad_ready.assert_called_once_with(
            layer.experts.linear_fc1.weight0
        )
        sync_bucket.register_grad_ready.assert_not_called()

    def test_forward_overlaps_dispatch_before_weight_sync_wait(self):
        layer = _fake_moe_layer()
        events, event = [], FakeEventHandle()
        manager = layer.eplb_manager
        manager.allocate_microbatch_slot = lambda layer_id: 23
        manager.update_placement = MagicMock()
        manager.reroute = lambda layer_id, probs, routing: (probs, routing)
        manager.weight_sync.return_value = event

        def wait(handle, layer_id):
            assert handle is event and layer_id == 23
            events.append('wait_weight_sync')
            handle.current_stream_wait()

        manager.runtime.wait_weight_sync.side_effect = wait
        layer.eplb_enabled = layer._eplb_master_ptrs_registered = True
        layer._eplb_weight_sync_event_handle = None
        layer.moe_layer_recompute = False
        layer.shared_experts_compute = lambda hidden: None
        layer.route = lambda hidden, mask: (torch.ones(2, 2), torch.eye(2, dtype=torch.bool))
        layer.preprocess = lambda hidden, probs, routing: (hidden, probs)
        layer.dispatch = lambda hidden, probs: (events.append('dispatch') or hidden, probs)
        layer.routed_experts_compute = lambda hidden, probs: (events.append('experts') or hidden, None)
        layer.combine = lambda output: output
        layer.postprocess = lambda output, shared: output
        original = MagicMock()
        output, bias = moe_layer.moe_layer_forward_wrapper(original)(layer, torch.ones(2, 2))
        original.assert_not_called()
        assert output.shape == (2, 2) and bias is None
        assert events == ['dispatch', 'wait_weight_sync', 'experts']
        assert event.wait_count == 1 and layer._eplb_weight_sync_event_handle is None

    @pytest.mark.parametrize('ultraep_enabled', [False, True])
    def test_disabled_forward_delegates_unchanged(self, ultraep_enabled):
        original = MagicMock(return_value=("output", "bias"))
        wrapped = moe_layer.moe_layer_forward_wrapper(original)
        layer = SimpleNamespace(
            eplb_enabled=ultraep_enabled, eplb_manager=SimpleNamespace(ultraep_active=False),
            training=False, experts_recompute=False, router_recompute=False,
        )
        hidden = torch.ones(1)
        assert wrapped(layer, hidden, intermediate_tensors="i", padding_mask="p") == (
            "output",
            "bias",
        )
        original.assert_called_once_with(
            layer,
            hidden_states=hidden,
            intermediate_tensors="i",
            padding_mask="p",
        )

    @pytest.mark.parametrize('recompute', ['router', 'experts'])
    def test_baseline_recompute_uses_dispatched_tokens(self, monkeypatch, recompute):
        hidden, dispatched = torch.ones(2, 2), torch.full((2, 2), 3.0)
        probs, routing = torch.ones(2, 2), torch.eye(2, dtype=torch.bool)
        dispatcher = SimpleNamespace(
            dispatch_postprocess=MagicMock(return_value=(dispatched, torch.tensor([2]), probs)),
            combine_preprocess=lambda output: output,
        )
        experts = MagicMock(return_value=(dispatched * 2, None))
        layer = SimpleNamespace(
            eplb_enabled=False, training=False, token_dispatcher=dispatcher, experts=experts,
            router_recompute=recompute == 'router', experts_recompute=recompute == 'experts',
            shared_experts_compute=lambda hidden: None,
            route=lambda hidden, mask: (probs, routing),
            preprocess=lambda hidden, probs, routing: (hidden, probs),
            dispatch=lambda hidden, probs: (dispatched, probs),
            combine=lambda output: output, postprocess=lambda output, shared: output,
        )
        checkpoint = MagicMock(side_effect=lambda function, distribute, *args: function(*args))
        monkeypatch.setattr(moe_layer.tensor_parallel, 'checkpoint', checkpoint)
        original = MagicMock()
        output, bias = moe_layer.moe_layer_forward_wrapper(original)(layer, hidden)
        assert dispatcher.dispatch_postprocess.call_args.args[0] is dispatched
        assert experts.call_args.args[0] is dispatched
        torch.testing.assert_close(output, dispatched * 2)
        assert bias is None and checkpoint.call_count == 1
        original.assert_not_called()

    def test_enabled_init_requires_runtime(self, monkeypatch):
        monkeypatch.setattr(eplb_manager_module, "HAVE_EPLB", False)
        wrapped = moe_layer.moe_layer_init_wrapper(MagicMock())
        with pytest.raises(ImportError, match="ultra_ep could not be imported"):
            wrapped(object(), SimpleNamespace(moe_enable_ultraep=True))

    def test_switch_restores_logical_dispatcher_and_preserves_parameters(self):
        layer = _fake_moe_layer()
        layer._eplb_weight_sync_event_handle = None
        layer.experts.num_local_experts = 2
        dispatcher = layer.token_dispatcher
        dispatcher.sort_input_by_local_experts = torch.tensor([0, 2, 1, 3])
        dispatcher.restore_output_by_local_experts = torch.tensor([0, 2, 1, 3])
        manager = eplb_manager_module.EPLBManager.__new__(eplb_manager_module.EPLBManager)
        manager.__dict__.update(vars(layer.eplb_manager))
        manager.adaptive_config = AdaptiveConfig()
        manager._adaptive_layers = []
        manager.ultraep_active = False
        layer.eplb_manager = manager
        baseline = moe_layer._capture_expert_execution_state(layer)
        moe_layer._eplb_register_redundant_experts(layer)
        manager.register_adaptive_layer(layer, baseline, moe_layer._capture_expert_execution_state(layer))
        params = tuple(layer.experts.parameters())
        pointers = [param.data_ptr() for param in params]
        for active in (True, False, True, False):
            manager.set_ultraep_active(active, synchronize=False)
            assert layer.experts.linear_fc1.num_gemms == (3 if active else 2)
            assert dispatcher.num_experts == (6 if active else 4)
            assert dispatcher.local_expert_indices == ([3, 4, 5] if active else [2, 3])
            assert [param.data_ptr() for param in layer.experts.parameters()] == pointers
        assert torch.equal(dispatcher.sort_input_by_local_experts, baseline[-1][1]['sort_input_by_local_experts'])
        layer._eplb_grad_reduce_event_handles[1] = object()
        with pytest.raises(RuntimeError, match='pending replica'):
            manager.set_ultraep_active(True, synchronize=False)


class TestCheckpointFiltering:
    def test_grouped_linear_limits_physical_experts_and_restores_count(self):
        module = SimpleNamespace(num_gemms=4, num_local_master_experts=2)

        def original(self, *args, **kwargs):
            assert self.num_gemms == 2
            return {"count": self.num_gemms}

        result = te_grouped_linear_sharded_state_dict_wrapper(original)(module, None)
        assert result == {"count": 2}
        assert module.num_gemms == 4

    def test_grouped_mlp_filters_replica_keys_and_restores_count(self):
        module = SimpleNamespace(num_local_experts=4, num_local_master_experts=2)

        def original(self, **kwargs):
            assert self.num_local_experts == 2
            return {
                "linear_fc1.weight0": 0,
                "linear_fc1.weight1": 1,
                "linear_fc1.weight2": 2,
                "linear_fc2.bias3": 3,
                "metadata": 4,
            }

        result = te_grouped_mlp_sharded_state_dict_wrapper(original)(module)
        assert result == {
            "linear_fc1.weight0": 0,
            "linear_fc1.weight1": 1,
            "metadata": 4,
        }
        assert module.num_local_experts == 4

    @pytest.mark.parametrize("ckpt_format, should_filter", [("torch", True), ("torch_dist", False)])
    def test_generate_state_dict_filters_only_native_checkpoints(
        self, ckpt_format, should_filter
    ):
        config = SimpleNamespace(num_moe_experts=4, expert_model_parallel_size=2)
        models = [SimpleNamespace(config=config), SimpleNamespace(config=config)]
        original_state = {
            "model0": {"linear_fc1.weight0": 0, "linear_fc1.weight2": 2},
            "model1": {"linear_fc2.weight1": 1, "linear_fc2.weight3": 3},
        }

        def original(*args, **kwargs):
            return {key: dict(value) for key, value in original_state.items()}

        wrapped = generate_state_dict_ultraep_wrapper(original)
        result = wrapped(SimpleNamespace(ckpt_format=ckpt_format), models, None, None, None)
        if should_filter:
            assert result == {
                "model0": {"linear_fc1.weight0": 0},
                "model1": {"linear_fc2.weight1": 1},
            }
        else:
            assert result == original_state


class TestAutotuneTrainStep:
    @staticmethod
    def _install_manager_query(monkeypatch, managers):
        monkeypatch.setattr(eplb_manager_module, 'get_eplb_managers', lambda: ())
        monkeypatch.setattr(eplb_manager_module, 'get_collecting_eplb_managers', lambda: tuple(managers))

    def test_fast_path_has_no_device_synchronization(self, monkeypatch):
        self._install_manager_query(monkeypatch, [])
        train_step = MagicMock(return_value="result")
        wrapped = ultraep_autotune_train_step_wrapper(train_step)
        with patch("torch.cuda.synchronize") as synchronize:
            assert wrapped(1, marker=True) == "result"
        synchronize.assert_not_called()
        train_step.assert_called_once_with(1, marker=True)

    @pytest.mark.parametrize('timed', [False, True])
    def test_autotune_reports_complete_steps_to_all_managers(self, monkeypatch, timed):
        first, second = MagicMock(), MagicMock()
        first.autotune_needs_iteration_time = False
        second.autotune_needs_iteration_time = timed
        self._install_manager_query(monkeypatch, [first, second])
        wrapped = ultraep_autotune_train_step_wrapper(lambda: 'result')
        with patch('torch.cuda.synchronize') as synchronize, patch(
            'hcu_megatron.training.ultraep_autotune.time.perf_counter', side_effect=[2.0, 2.125],
        ):
            assert wrapped() == 'result'
        assert synchronize.call_count == (2 if timed else 1)
        for manager in (first, second):
            manager.autotune_iteration_end.assert_called_once_with(125.0 if timed else None)

    @pytest.mark.parametrize('has_local_manager', [True, False])
    def test_adaptive_bridge_reuses_timer_and_stops_after_selection(self, monkeypatch, has_local_manager):
        from hcu_megatron.training import ultraep_autotune as bridge
        from hcu_megatron.core.transformer.moe import eplb_manager

        manager = SimpleNamespace(
            adaptive_config=AdaptiveConfig(), adaptive_controller=None, ultraep_active=False,
            autotune_collecting=False, autotune_iteration_end=lambda elapsed: None,
        )
        manager.set_ultraep_active = lambda active, **kw: setattr(manager, 'ultraep_active', active)
        monkeypatch.setattr(eplb_manager, 'get_eplb_managers', lambda: (manager,) if has_local_manager else ())
        monkeypatch.setattr(bridge, '_adaptive_workload', lambda: (4096, 1, 4))
        tensor = torch.tensor
        monkeypatch.setattr(bridge.torch, 'tensor', lambda data, **kw: tensor(data, dtype=kw['dtype']))
        all_reduce = MagicMock()
        monkeypatch.setattr(bridge.torch.distributed, 'all_reduce', all_reduce)
        monkeypatch.setattr(bridge.torch.distributed, 'get_rank', lambda: 1)
        clock = [0.0]
        monkeypatch.setattr(bridge.time, 'perf_counter', lambda: clock[0])

        def step(config=None):
            clock[0] += 0.09 if manager.ultraep_active else 0.1
            return {}, False, False, False

        wrapped = bridge.ultraep_autotune_train_step_wrapper(step)
        config = SimpleNamespace(moe_ultraep_adaptive=True)
        with patch.object(bridge.torch.cuda, 'synchronize') as synchronize:
            for _ in range(17):
                wrapped(config=config)
            if has_local_manager:
                assert manager.ultraep_active
                assert manager.adaptive_controller.result['baseline_ms'] == pytest.approx(100)
            else:
                assert all_reduce.call_count > 0
            count = synchronize.call_count
            collectives = all_reduce.call_count
            wrapped(config=config)
            assert synchronize.call_count == count
            assert all_reduce.call_count == collectives


class TestUltraEPAdaptivePolicy:
    @staticmethod
    def _run(baseline, ultraep, *, tune_end=0, budget=48, invalid_step=None):
        controller = AdaptiveController(AdaptiveConfig(max_calibration_steps=budget))
        for step in range(tune_end):
            controller.step_end(None, tuning=step < tune_end - 1)
        values = {False: iter(baseline), True: iter(ultraep)}
        while not controller.complete:
            valid = controller.steps + 1 != invalid_step
            elapsed = next(values[controller.active]) if valid and controller.needs_timing else None
            controller.step_end(elapsed, valid=valid)
        return controller

    @pytest.mark.parametrize('baseline,ultraep,enabled,pairs,steps', [
        ([1000, 1500, 2000], [800, 1200, 1600], True, 3, 11),
        ([100] * 3, [100] * 3, False, 3, 11),
        ([100] * 3, [105] * 3, False, 3, 11),
        ([100] * 3, [99.5] * 3, False, 3, 11),
        ([100] * 5, [90, 110, 95, 120, 99], True, 5, 17),
        ([100, 100, 300, 100, 100], [101] * 5, False, 5, 17),
    ], ids=['noisy-gain', 'tie', 'slower', 'below-threshold', 'supplemented-gain', 'baseline-outlier'])
    def test_median_selection(self, baseline, ultraep, enabled, pairs, steps):
        controller = self._run(baseline, ultraep)
        assert controller.active is enabled and controller.steps == steps
        assert controller.result['samples'] == {'baseline': pairs, 'ultraep': pairs}
        assert controller.result['paired_samples'] == pairs

    @pytest.mark.parametrize('ultraep,invalid_step,steps', [
        ([90] * 3, None, 41),
        ([90, 110, 95, 120, 99], None, 47),
        ([90, 110, 95, 120, 99], 32, 48),
    ])
    def test_full_tuning_leaves_room_for_selection(self, ultraep, invalid_step, steps):
        controller = self._run(
            [100] * len(ultraep), ultraep, tune_end=31, invalid_step=invalid_step,
        )
        assert controller.active and controller.steps == steps

    @pytest.mark.parametrize('ultraep,enabled,median_ultraep', [
        ([98, 105, 97, 1], True, 98),
        ([99.5, 98, 99.7, 1], False, 99.5),
    ])
    def test_budget_uses_only_completed_pairs(self, ultraep, enabled, median_ultraep):
        controller = self._run([100] * 3, ultraep, budget=12)
        assert controller.active is enabled
        assert controller.result['samples'] == {'baseline': 3, 'ultraep': 4}
        assert controller.result['paired_samples'] == 3
        assert controller.result['ultraep_ms'] == median_ultraep

    def test_tuning_and_settling_steps_are_not_scored(self):
        controller = AdaptiveController(AdaptiveConfig())
        controller.step_end(None, tuning=True)
        controller.step_end(None, tuning=True)
        assert controller.phase == 'tune' and controller.active
        controller.step_end(9999, tuning=True)
        controller.step_end(9999, tuning=False)
        assert controller.phase == 'compare' and not controller.active
        controller.step_end(9999)
        controller.step_end(100)
        assert controller.times == {False: [100], True: []}

    @pytest.mark.parametrize('failure', ['invalid-step', 'stalled-tuner', 'workload-change'])
    def test_calibration_guards(self, failure):
        controller = AdaptiveController(AdaptiveConfig(max_calibration_steps=12))
        if failure == 'workload-change':
            controller.step_end(None, workload=(4096, 1, 4))
            controller.step_end(None, workload=(4096, 1, 8))
        else:
            for _ in range(12):
                controller.step_end(None, valid=failure != 'invalid-step', tuning=True)
        assert controller.complete and not controller.active
        assert controller.result['reason'] == (
            'workload changed during calibration' if failure == 'workload-change'
            else 'calibration budget exhausted'
        )


# Real single-node EP8 model tests.


@pytest.fixture(scope="module")
def _ultraep_environment():
    """Apply the single-node IPC environment without leaking it to other tests."""
    runtime_environment = {
        "HSA_USE_SVM": "0",
        "MAX_NUM_NVL_PEERS": "8",
        "ROCSHMEM_BACKEND": "ipc",
        # Limit the IPC heap for the tiny EP8 models.
        "ROCSHMEM_HEAP_SIZE": str(256 * 1024**2),
    }
    with patch.dict(os.environ, runtime_environment, clear=False):
        os.environ.pop("ROCSHMEM_GDA_PROVIDER", None)
        yield


@pytest.fixture(scope="module")
def _ultraep_distributed(_ultraep_environment):
    # Skip unavailable backends before GPU setup and process-group creation.
    try:
        import ultra_ep
    except ImportError as error:
        pytest.skip(f"UltraEP backend is unavailable: {error}")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in required):
        pytest.skip("run with torchrun --nproc-per-node=8")

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        pytest.skip(f"UltraEP integration requires exactly 8 ranks, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip(f"UltraEP integration requires 8 HCUs, got {torch.cuda.device_count()}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
    try:
        yield ultra_ep
    finally:
        if initialized_here and dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()


def _hot_router(self, hidden_states, padding_mask=None):
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = torch.nn.functional.linear(flat.float(), self.weight.float())
    # Fixed hot expert IDs avoid discontinuous route changes from one-ULP BF16
    # weight differences; probabilities and router gradients remain learned.
    first = (torch.arange(flat.shape[0], device=flat.device) % 2) * 2
    indices = torch.stack((first, first + 1), dim=-1)
    probs = torch.zeros_like(logits).scatter(1, indices, torch.softmax(logits.gather(1, indices), dim=-1))
    routing = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
    return probs, routing


@pytest.fixture(scope='module')
def model_environment(_ultraep_distributed):
    from hcu_megatron.megatron_adaptor import repatch
    from megatron.core import tensor_parallel
    from tests.unit_tests.test_utilities import Utils
    from hcu_megatron.core.transformer.moe.eplb_manager import clear_eplb_manager_registry

    env = {
        'ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA': '1',
        'ULTRA_EP_GRAD_REDUCE_DETERMINISTIC': '1', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0',
    }
    with patch.dict(os.environ, env), patch.object(sys, 'argv', ['test_ultraep_framework.py']):
        os.environ.pop('ROCSHMEM_GDA_PROVIDER', None)
        repatch(patch_adaptor_args={
            'moe_enable_ultraep': True, 'moe_num_redundant_experts_per_rank': 2,
            'moe_ultraep_adaptive': True, 'overlap_moe_expert_parallel_comm': False,
        }, patch_megatron_args={
            'num_layers': 2, 'hidden_size': 128, 'num_attention_heads': 4,
            'micro_batch_size': 1, 'seq_length': 16, 'global_batch_size': 32,
            'gradient_accumulation_fusion': True, 'data_parallel_size': 8,
        })
        Utils.initialize_model_parallel(expert_model_parallel_size=8)
        tensor_parallel.model_parallel_cuda_manual_seed(123)
        yield
        torch.cuda.synchronize()
        clear_eplb_manager_registry()
        Utils.destroy_model_parallel()


def _make_model(adaptive, autotune=False):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=4, kv_channels=32,
        ffn_hidden_size=512, moe_ffn_hidden_size=64, num_moe_experts=16, moe_router_topk=2,
        moe_router_dtype='fp32', moe_token_dispatcher_type='alltoall', moe_grouped_gemm=True,
        expert_model_parallel_size=8, expert_tensor_parallel_size=1,
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        add_bias_linear=False, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0, attention_dropout=0.0, attention_backend='unfused',
        gradient_accumulation_fusion=True,
    )
    config.moe_enable_ultraep = adaptive
    config.moe_ultraep_adaptive = adaptive
    config.moe_num_redundant_experts_per_rank = 2
    config.moe_ultraep_autotune = autotune
    model = GPTModel(
        config=config, transformer_layer_spec=get_gpt_decoder_block_spec(config=config, use_transformer_engine=True),
        vocab_size=128, max_sequence_length=32, pre_process=True, post_process=True,
    ).cuda().bfloat16()
    for layer in model.decoder.layers:
        layer.mlp.router.forward = types.MethodType(_hot_router, layer.mlp.router)
    return model


def _optimizer_tensors(optimizer):
    return [param for item in optimizer.chained_optimizers
            for group in item.optimizer.param_groups for param in group['params']]


def _gradients(model, optimizer, distributed):
    if distributed:
        names = {param: name for name, param in model.named_parameters()}
        gradients = {}
        for item in optimizer.chained_optimizers:
            for param in item.model_param_gbuf_map:
                region = item._get_model_param_range_map(param)['param']
                gradients[f'{names[param]}[{region.start}:{region.end}]'] = (
                    param.main_grad.flatten()[region.start:region.end].detach().clone()
                )
        return gradients
    return {name: param.main_grad.detach().clone() for name, param in model.named_parameters()
            if not getattr(param, 'is_eplb_replica', False)}


class TestUltraEPAdaptiveModelEP8:
    @pytest.mark.parametrize('distributed_optimizer', [False, True])
    def test_real_mode_switch_gradients_and_updates(self, model_environment, distributed_optimizer):
        microbatches = 2
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        reference, candidate = _make_model(False), _make_model(True)
        candidate.load_state_dict(deepcopy(reference.state_dict()), strict=False)

        def wrap(model):
            return DistributedDataParallel(config=model.config, module=model, ddp_config=DistributedDataParallelConfig(
                bucket_size=4096 if distributed_optimizer else None, grad_reduce_in_fp32=True,
                overlap_grad_reduce=True, use_distributed_optimizer=distributed_optimizer,
                overlap_param_gather=distributed_optimizer,
            ))

        reference, candidate = wrap(reference), wrap(candidate)
        optimizer_config = OptimizerConfig(
            optimizer='adam' if distributed_optimizer else 'sgd', lr=0.0001,
            bf16=True, params_dtype=torch.bfloat16,
            use_distributed_optimizer=distributed_optimizer, overlap_param_gather=distributed_optimizer,
            clip_grad=0.0, weight_decay=0.0, sgd_momentum=0.0,
        )
        optimizers = [get_megatron_optimizer(optimizer_config, [model]) for model in (reference, candidate)]
        manager = candidate.module.decoder.layers[0].mlp.eplb_manager
        for mode_index, active in enumerate((False, True, False, True)):
            manager.set_ultraep_active(active)
            # The candidate keeps its complete training history across switches.
            # Compare each new step against a baseline oracle starting at that same
            # state, rather than comparing diverging BF16/Adam trajectories.
            if distributed_optimizer:
                for model in (reference, candidate):
                    model.start_param_sync(force_sync=True)
            candidate_params = dict(candidate.module.named_parameters())
            with torch.no_grad():
                for name, param in reference.module.named_parameters():
                    param.copy_(candidate_params[name])
                for ref_item, actual_item in zip(optimizers[0].chained_optimizers, optimizers[1].chained_optimizers):
                    for ref_group, actual_group in zip(
                        ref_item.optimizer.param_groups, actual_item.optimizer.param_groups
                    ):
                        for ref_param, actual_param in zip(ref_group['params'], actual_group['params']):
                            ref_param.copy_(actual_param)
                            if actual_param in actual_item.optimizer.state:
                                ref_item.optimizer.state[ref_param] = deepcopy(
                                    actual_item.optimizer.state[actual_param]
                                )
            data = [{
                'input_ids': torch.randint(0, 128, (1, 16), device='cuda'),
                'labels': torch.randint(0, 128, (1, 16), device='cuda'),
                'position_ids': torch.arange(16, device='cuda').unsqueeze(0),
                'attention_mask': torch.triu(torch.ones((1, 1, 16, 16), device='cuda', dtype=torch.bool), diagonal=1),
            } for _ in range(microbatches)]
            losses = []
            for model, optimizer in zip((reference, candidate), optimizers):
                model.module.set_is_first_microbatch()
                model.zero_grad_buffer()
                optimizer.zero_grad()
                current = []
                for index, item in enumerate(data):
                    with model.no_sync() if index < len(data) - 1 else nullcontext():
                        loss = model(**item).float().mean() / len(data)
                        current.append(loss.detach())
                        loss.backward()
                model.finish_grad_sync()
                losses.append(current)
            torch.cuda.synchronize()
            for expected, actual in zip(*losses):
                torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.0001)
            expected_gradients = _gradients(reference.module, optimizers[0], distributed_optimizer)
            actual_gradients = _gradients(candidate.module, optimizers[1], distributed_optimizer)
            assert expected_gradients.keys() == actual_gradients.keys()
            errors = []
            for name in expected_gradients:
                actual, expected = actual_gradients[name], expected_gradients[name]
                relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                if not torch.isfinite(actual).all() or (actual - expected).norm() > 0.02 * expected.norm() + 1e-7:
                    errors.append(
                        f'{name}: relative_L2={relative_error.item():.5f}, '
                        f'expected_norm={expected.norm().item():.5f}, '
                        f'actual_norm={actual.norm().item():.5f}'
                    )
            failed = torch.tensor([int(bool(errors))], device='cuda')
            torch.distributed.all_reduce(failed)
            assert not failed.item(), (
                f'transition={mode_index}, mode={active}, rank={torch.distributed.get_rank()}: {errors}'
            )
            assert not any(layer.mlp._eplb_grad_reduce_event_handles for layer in candidate.module.decoder.layers)
            if active:
                assert torch.count_nonzero(manager.local_replica_grad_buffer) == 0
                assert torch.any(manager.runtime.logical_replica_counts > 1)
            before = [[param.detach().clone() for param in _optimizer_tensors(opt)] for opt in optimizers]
            for optimizer in optimizers:
                assert optimizer.step()[0]
            # BF16 weights can differ by one ULP after equivalent FP32 updates.
            # Check actual update vectors rather than loosening parameter tolerance.
            error = torch.zeros([], device='cuda')
            update = torch.zeros_like(error)
            for expected, old_expected, actual, old_actual in zip(
                _optimizer_tensors(optimizers[0]), before[0], _optimizer_tensors(optimizers[1]), before[1],
            ):
                expected_delta = expected.detach() - old_expected
                actual_delta = actual.detach() - old_actual
                error += (expected_delta - actual_delta).square().sum()
                update += expected_delta.square().sum()
            torch.distributed.all_reduce(error)
            torch.distributed.all_reduce(update)
            assert update > 0 and error.sqrt() <= 0.02 * update.sqrt(), f'mode={active}'

    def test_real_adaptive_complete_steps(self, model_environment):
        from hcu_megatron.training.ultraep_autotune import ultraep_autotune_train_step_wrapper
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        from hcu_megatron.core.transformer.moe.eplb_manager import (
            clear_eplb_manager_registry, get_eplb_managers,
        )
        # Keep old symmetric allocations alive until this test has finished.
        previous_managers = get_eplb_managers()
        clear_eplb_manager_registry()
        module = _make_model(True, autotune=True)
        model = DistributedDataParallel(config=module.config, module=module, ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=True,
        ))
        optimizer = get_megatron_optimizer(OptimizerConfig(
            optimizer='sgd', lr=0.001, bf16=True, params_dtype=torch.bfloat16,
            clip_grad=0.0, weight_decay=0.0, sgd_momentum=0.0,
        ), [model])
        manager = module.decoder.layers[0].mlp.eplb_manager
        manager.adaptive_config = AdaptiveConfig()
        manager.adaptive_controller = None
        manager.set_ultraep_active(False)
        modes = []

        def step():
            modes.append(manager.ultraep_active)
            module.set_is_first_microbatch()
            model.zero_grad_buffer()
            optimizer.zero_grad()
            tokens = torch.randint(0, 128, (1, 16), device='cuda')
            loss = model(
                input_ids=tokens, labels=tokens,
                position_ids=torch.arange(16, device='cuda').unsqueeze(0),
                attention_mask=torch.triu(torch.ones((1, 1, 16, 16), device='cuda', dtype=torch.bool), diagonal=1),
            ).float().mean()
            loss.backward()
            model.finish_grad_sync()
            successful = optimizer.step()[0]
            return {'loss': loss.detach()}, not successful, False, False

        wrapped = ultraep_autotune_train_step_wrapper(step)
        for _ in range(manager.adaptive_config.max_calibration_steps):
            wrapped()
            if manager.adaptive_controller.complete:
                break
        result = manager.adaptive_controller.result
        assert set(modes) == {False, True} and result is not None
        assert result['baseline_ms'] > 0 and result['ultraep_ms'] > 0
        assert min(result['samples'].values()) >= 3
        assert manager.runtime.autotune_complete
        assert manager.ultraep_active == result['enabled']
        results = [None] * 8
        torch.distributed.all_gather_object(results, result)
        assert all(item == result for item in results)
        del previous_managers
