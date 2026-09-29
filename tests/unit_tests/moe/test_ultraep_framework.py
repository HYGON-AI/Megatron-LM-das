# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""UltraEP framework unit tests and single-node EP8 integration tests.

Pure framework logic uses a fake Manager so it stays deterministic. The EP8
tests at the bottom use the installed extension and real RCCL/rocSHMEM. Run all
tests on one eight-HCU node with::

    python -m torch.distributed.run --nproc-per-node=8 -m pytest -s -vv \
        tests/unit_tests/moe/test_ultraep_framework.py
"""

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace
import types
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
from hcu_megatron.core.transformer.moe import moe_layer_ultraep
from hcu_megatron.core.transformer.moe.experts import (
    te_grouped_linear_sharded_state_dict_wrapper,
    te_grouped_mlp_sharded_state_dict_wrapper,
)
from hcu_megatron.features_manager.moe.ultraep_feature import (
    UltraEPFeature,
    _get_param_groups_ultraep_wrapper,
)
from hcu_megatron.training.checkpointing import (
    _filter_ultraep_native_model_state_dict,
    generate_state_dict_ultraep_wrapper,
)
from hcu_megatron.training.ultraep_autotune import (
    ultraep_autotune_train_step_wrapper,
)


@dataclass
class FakeAutotuneConfig:
    enabled: bool = False
    start_iteration: int = 3
    grad_reduce_max_sms: int | None = None


class FakeEventHandle:
    def __init__(self):
        self.wait_count = 0

    def current_stream_wait(self):
        self.wait_count += 1


class FakeUltraEPManager:
    """CPU replacement for ``ultra_ep.Manager`` used by EPLBManager tests."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)
        redundant = kwargs["num_local_redundant_experts"]
        fc1_numel = kwargs["expert_fc1_numel"]
        fc2_numel = kwargs["expert_fc2_numel"]
        self.local_replica_fc1_weight_buffer = torch.zeros(redundant, fc1_numel)
        self.local_replica_fc2_weight_buffer = torch.zeros(redundant, fc2_numel)
        self.local_replica_fc1_grad_buffer = torch.zeros(redundant, fc1_numel)
        self.local_replica_fc2_grad_buffer = torch.zeros(redundant, fc2_numel)
        self.local_replica_weight_buffer = torch.zeros(redundant, fc1_numel + fc2_numel)
        self.local_replica_grad_buffer = torch.zeros(redundant, fc1_numel + fc2_numel)
        config = kwargs.get("autotune")
        self.autotune_enabled = bool(config and config.enabled)
        self.autotune_collecting = self.autotune_enabled
        self.autotune_start_iteration = config.start_iteration if config else 3
        self.iteration_end_calls = []
        self.placement_calls = []
        self._next_virtual_layer = 100

    def update_placement(self, layer_id, routing_map):
        self.placement_calls.append((layer_id, routing_map))

    def reroute(self, layer_id, probs, routing_map, backend="cuda"):
        return probs + 1, ~routing_map

    def allocate_microbatch_slot(self, real_layer_id):
        self._next_virtual_layer += 1
        return self._next_virtual_layer

    def autotune_needs_iteration_time(self, iteration):
        return iteration >= self.autotune_start_iteration

    def autotune_iteration_end(self, iteration, iteration_time_ms=None):
        self.iteration_end_calls.append((iteration, iteration_time_ms))

    def wait_grad_reduce(self, event, layer_id=None):
        event.current_stream_wait()

    def wait_weight_sync(self, event, layer_id):
        event.current_stream_wait()


def _fake_ultra_ep(manager_cls=FakeUltraEPManager):
    return SimpleNamespace(Manager=manager_cls, AutotuneConfig=FakeAutotuneConfig)


def _manager_config(**overrides):
    values = dict(
        num_layers=4,
        num_moe_experts=8,
        moe_num_redundant_experts_per_rank=2,
        hidden_size=4,
        moe_ffn_hidden_size=3,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        moe_ultraep_autotune=False,
        moe_ultraep_autotune_start_iteration=3,
        moe_ultraep_autotune_grad_reduce_max_sms=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


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
        assert args.moe_ultraep_autotune_start_iteration == 3
        assert args.moe_ultraep_autotune_grad_reduce_max_sms is None

    def test_validate_accepts_supported_configuration(self):
        args = _feature_args(moe_ultraep_autotune_grad_reduce_max_sms=48)
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

        train_step = "hcu_megatron.training.training.train_step"
        assert train_step not in {item[0] for item in base.patches}
        assert train_step in {item[0] for item in tuned.patches}
        assert len(tuned.patches) == len(base.patches) + 1
        names = {item[1] for item in tuned.patches}
        assert "te_grouped_linear_sharded_state_dict_wrapper" in names
        assert "te_grouped_mlp_sharded_state_dict_wrapper" in names

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
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        FakeUltraEPManager.instances.clear()
        eplb_manager_module.clear_eplb_manager_registry()
        yield
        eplb_manager_module.clear_eplb_manager_registry()

    def _construct(self, monkeypatch, config=None, rank=1, size=2, ultra_ep=None):
        monkeypatch.setattr(eplb_manager_module.utils, "get_pg_rank", lambda group: rank)
        monkeypatch.setattr(eplb_manager_module.utils, "get_pg_size", lambda group: size)
        monkeypatch.setattr(
            eplb_manager_module, "ultra_ep", ultra_ep or _fake_ultra_ep(), raising=False
        )
        return eplb_manager_module.EPLBManager(config or _manager_config(), object())

    def test_constructs_counts_slots_and_runtime_buffers(self, monkeypatch):
        manager = self._construct(monkeypatch)
        assert manager.num_local_master_experts == 4
        assert manager.num_local_physical_experts == 6
        assert manager.num_global_physical_experts == 12
        assert manager.local_physical_expert_indices == list(range(6, 12))
        assert manager.expert_fc1_numel == 24
        assert manager.expert_fc2_numel == 12
        assert manager.max_microbatches == 6
        assert manager.local_replica_fc1_weight_buffer.shape == (2, 24)
        assert "autotune" not in manager.runtime.kwargs

    def test_virtual_pipeline_increases_slot_budget(self, monkeypatch):
        config = _manager_config(virtual_pipeline_model_parallel_size=3)
        manager = self._construct(monkeypatch, config=config)
        assert manager.max_microbatches == 2 * (3 + 1) * 3

    def test_passes_autotune_config_and_advances_local_iteration(self, monkeypatch):
        config = _manager_config(
            moe_ultraep_autotune=True,
            moe_ultraep_autotune_start_iteration=5,
            moe_ultraep_autotune_grad_reduce_max_sms=48,
        )
        manager = self._construct(monkeypatch, config=config, rank=0)
        runtime_config = manager.runtime.kwargs["autotune"]
        assert runtime_config == FakeAutotuneConfig(True, 5, 48)
        assert manager.next_autotune_iteration == 1
        assert manager.autotune_needs_iteration_time is False
        manager.autotune_iteration_end(12.5)
        assert manager.next_autotune_iteration == 2
        assert manager.runtime.iteration_end_calls == [(1, 12.5)]
        manager.runtime.autotune_collecting = False
        manager.autotune_iteration_end(99.0)
        assert manager.runtime.iteration_end_calls == [(1, 12.5)]

    def test_reports_missing_autotune_runtime_apis(self, monkeypatch):
        class OldManager:
            pass

        old_runtime = SimpleNamespace(Manager=OldManager)
        with pytest.raises(RuntimeError, match="AutotuneConfig") as error:
            self._construct(
                monkeypatch,
                config=_manager_config(moe_ultraep_autotune=True),
                ultra_ep=old_runtime,
            )
        assert "Manager.wait_weight_sync" in str(error.value)

    def test_runtime_delegates_and_registry(self, monkeypatch):
        manager = self._construct(monkeypatch)
        routing = torch.tensor([[True, False]])
        manager.update_placement(7, routing)
        assert manager.runtime.placement_calls == [(7, routing)]
        probs = torch.tensor([[0.25, 0.75]])
        new_probs, new_routing = manager.reroute(7, probs, routing, backend="cpu")
        assert torch.equal(new_probs, probs + 1)
        assert torch.equal(new_routing, ~routing)
        assert manager.allocate_microbatch_slot(3) == 101

        group = object()
        monkeypatch.setattr(eplb_manager_module, "EPLBManager", lambda config, ep_group: manager)
        first = eplb_manager_module.get_or_create_eplb_manager(_manager_config(), group)
        second = eplb_manager_module.get_or_create_eplb_manager(_manager_config(), group)
        assert first is second is manager
        assert eplb_manager_module.get_eplb_managers() == (manager,)
        manager.runtime.autotune_collecting = True
        assert eplb_manager_module.get_collecting_eplb_managers() == (manager,)


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

    def test_master_backward_hook_defers_ready_registration(self):
        master = nn.Parameter(torch.zeros(2), requires_grad=False)
        master.is_eplb_master = True
        bucket = MagicMock()
        ddp = SimpleNamespace(
            param_to_bucket_group={master: bucket},
            ddp_config=SimpleNamespace(overlap_grad_reduce=True),
            force_all_reduce=False,
        )
        hook = types.MethodType(
            HcuDistributedDataParallel._make_backward_post_hook, ddp
        )(master)
        args = SimpleNamespace(
            gradient_accumulation_fusion=False,
            delay_wgrad_compute=False,
            recompute_in_advance=False,
            recompute_in_bubble=False,
        )
        with patch(
            "hcu_megatron.core.distributed.distributed_data_parallel.get_args",
            return_value=args,
        ), patch(
            "hcu_megatron.core.distributed.distributed_data_parallel.is_graph_capturing",
            return_value=False,
        ):
            hook()
        bucket.register_grad_ready.assert_not_called()

    def test_normal_backward_hook_accumulates_and_registers(self):
        param = nn.Parameter(torch.zeros(2))
        param.grad = torch.ones(2)
        param.main_grad = torch.zeros(2)
        param.grad_added_to_main_grad = False
        bucket = MagicMock()
        ddp = SimpleNamespace(
            param_to_bucket_group={param: bucket},
            ddp_config=SimpleNamespace(overlap_grad_reduce=True),
            force_all_reduce=True,
        )
        hook = types.MethodType(
            HcuDistributedDataParallel._make_backward_post_hook, ddp
        )(param)
        args = SimpleNamespace(
            gradient_accumulation_fusion=False,
            delay_wgrad_compute=False,
            recompute_in_advance=False,
            recompute_in_bubble=False,
        )
        with patch(
            "hcu_megatron.core.distributed.distributed_data_parallel.get_args",
            return_value=args,
        ), patch(
            "hcu_megatron.core.distributed.distributed_data_parallel.is_graph_capturing",
            return_value=False,
        ):
            hook()
        assert torch.equal(param.main_grad, torch.ones(2))
        assert param.grad is None
        bucket.register_grad_ready.assert_called_once_with(param, True)


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
    def test_registers_master_and_replica_parameters_and_dispatcher(self):
        layer = _fake_moe_layer()
        moe_layer_ultraep._eplb_register_redundant_experts(layer)

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
        moe_layer_ultraep._eplb_register_redundant_experts(layer)
        for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
            for index in range(2):
                getattr(linear, f"weight{index}").main_grad = torch.zeros_like(
                    getattr(linear, f"weight{index}")
                )

        moe_layer_ultraep._eplb_register_master_experts(layer)
        moe_layer_ultraep._eplb_register_master_experts(layer)
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

        moe_layer_ultraep._eplb_start_grad_reduce(layer, 11)
        moe_layer_ultraep._eplb_start_grad_reduce(layer, 12)
        with pytest.raises(AssertionError, match="launched twice"):
            moe_layer_ultraep._eplb_start_grad_reduce(layer, 11)
        moe_layer_ultraep._eplb_finish_grad_reduce(layer, 12)
        moe_layer_ultraep._eplb_finish_grad_reduce(layer, 11)
        assert first.wait_count == second.wait_count == 1
        assert layer._eplb_grad_reduce_event_handles == {}
        assert layer.eplb_manager.runtime.wait_grad_reduce.call_args_list == [
            call(second, layer_id=12),
            call(first, layer_id=11),
        ]

    def test_grad_reduce_wait_falls_back_for_legacy_runtime(self):
        event = FakeEventHandle()
        layer = SimpleNamespace(
            _eplb_grad_reduce_event_handles={7: event},
            eplb_manager=SimpleNamespace(runtime=SimpleNamespace()),
        )
        moe_layer_ultraep._eplb_finish_grad_reduce(layer, 7)
        assert event.wait_count == 1

    def test_master_grad_ready_only_for_overlap_buckets(self):
        layer = _fake_moe_layer()
        moe_layer_ultraep._eplb_register_redundant_experts(layer)
        overlap_bucket = MagicMock()
        overlap_bucket.ddp_config.overlap_grad_reduce = True
        sync_bucket = MagicMock()
        sync_bucket.ddp_config.overlap_grad_reduce = False
        layer.experts.linear_fc1.weight0._ddp_bucket_group = overlap_bucket
        layer.experts.linear_fc1.weight1._ddp_bucket_group = sync_bucket
        moe_layer_ultraep._eplb_register_master_grad_ready(layer)
        overlap_bucket.register_grad_ready.assert_called_once_with(
            layer.experts.linear_fc1.weight0
        )
        sync_bucket.register_grad_ready.assert_not_called()

    def test_autograd_functions_preserve_virtual_layer_order(self):
        events = []
        runtime = SimpleNamespace(
            weight_sync=lambda **kwargs: events.append(("weight", kwargs))
        )
        layer = SimpleNamespace(
            eplb_manager=SimpleNamespace(runtime=runtime),
            _eplb_start_grad_reduce=lambda virtual_layer_id: events.append(
                ("start", virtual_layer_id)
            ),
            _eplb_finish_grad_reduce=lambda virtual_layer_id: events.append(
                ("finish", virtual_layer_id)
            ),
            _eplb_register_master_grad_ready=lambda: events.append(("ready",)),
        )
        value = torch.ones(2, requires_grad=True)
        value = moe_layer_ultraep._EPLBReplicaGradReduceFinishFunction.apply(
            value, layer, 17
        )
        value = moe_layer_ultraep._EPLBReplicaGradReduceStartFunction.apply(
            value, layer, 17
        )
        value = moe_layer_ultraep._EPLBWeightSyncFunction.apply(value, layer, 17)
        value.sum().backward()
        assert events == [
            ("weight", {"layer_id": 17, "async_finish": False}),
            ("start", 17),
            ("finish", 17),
            ("ready",),
        ]

    def test_forward_overlaps_dispatch_before_weight_sync_wait(self):
        events = []
        event = FakeEventHandle()

        class Runtime:
            def weight_sync(self, **kwargs):
                events.append("weight_sync")
                return event

            def wait_weight_sync(self, handle, layer_id):
                assert handle is event and layer_id == 23
                events.append("wait_weight_sync")
                handle.current_stream_wait()

        manager = SimpleNamespace(
            runtime=Runtime(),
            allocate_microbatch_slot=lambda layer_id: 23,
            update_placement=lambda layer_id, routing: events.append("placement"),
            reroute=lambda layer_id, probs, routing: (
                events.append("reroute") or probs,
                routing,
            ),
        )
        layer = SimpleNamespace(
            eplb_enabled=True,
            eplb_manager=manager,
            layer_number=4,
            _eplb_master_ptrs_registered=True,
            _eplb_weight_sync_event_handle=None,
            moe_layer_recompute=False,
            shared_experts_compute=lambda hidden: None,
            route=lambda hidden, mask: (
                events.append("route") or torch.ones(2, 2, dtype=torch.float16),
                torch.tensor([[True, False], [False, True]]),
            ),
            preprocess=lambda hidden, probs, routing: (
                events.append("preprocess") or hidden,
                probs,
            ),
            dispatch=lambda hidden, probs: (events.append("dispatch") or hidden, probs),
            routed_experts_compute=lambda hidden, probs: (
                events.append("experts") or hidden,
                None,
            ),
            combine=lambda output: events.append("combine") or output,
            postprocess=lambda output, shared: events.append("postprocess") or output,
        )
        wrapped = moe_layer_ultraep.moe_layer_ultraep_forward_wrapper(
            lambda *args, **kwargs: pytest.fail("original forward should not run")
        )
        output, bias = wrapped(layer, torch.ones(2, 2, requires_grad=True))
        assert output.shape == (2, 2) and bias is None
        assert events == [
            "route",
            "placement",
            "weight_sync",
            "reroute",
            "preprocess",
            "dispatch",
            "wait_weight_sync",
            "experts",
            "combine",
            "postprocess",
        ]
        assert event.wait_count == 1
        assert layer._eplb_weight_sync_event_handle is None

    def test_disabled_forward_delegates_unchanged(self):
        original = MagicMock(return_value=("output", "bias"))
        wrapped = moe_layer_ultraep.moe_layer_ultraep_forward_wrapper(original)
        layer = SimpleNamespace(eplb_enabled=False)
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

    def test_enabled_init_requires_runtime(self, monkeypatch):
        monkeypatch.setattr(moe_layer_ultraep, "HAVE_EPLB", False)
        wrapped = moe_layer_ultraep.moe_layer_ultraep_init_wrapper(MagicMock())
        with pytest.raises(ImportError, match="ultra_ep could not be imported"):
            wrapped(object(), SimpleNamespace(moe_enable_ultraep=True))


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

    @pytest.mark.parametrize(
        "wrapper,module,args",
        [
            (
                te_grouped_linear_sharded_state_dict_wrapper,
                SimpleNamespace(num_gemms=4, num_local_master_experts=2),
                (None,),
            ),
            (
                te_grouped_mlp_sharded_state_dict_wrapper,
                SimpleNamespace(num_local_experts=4, num_local_master_experts=2),
                (),
            ),
        ],
    )
    def test_grouped_checkpoint_wrappers_restore_counts_on_error(self, wrapper, module, args):
        attribute = "num_gemms" if hasattr(module, "num_gemms") else "num_local_experts"
        with pytest.raises(RuntimeError, match="expected"):
            wrapper(lambda *unused, **kwargs: (_ for _ in ()).throw(RuntimeError("expected")))(
                module, *args
            )
        assert getattr(module, attribute) == 4

    def test_native_filter_handles_wrapped_model_and_missing_config(self):
        config = SimpleNamespace(num_moe_experts=8, expert_model_parallel_size=2)
        model = SimpleNamespace(module=SimpleNamespace(config=config))
        state = {
            "decoder.mlp.linear_fc1.weight3": 3,
            "decoder.mlp.linear_fc1.weight4": 4,
            "decoder.mlp.linear_fc2.bias7": 7,
            "decoder.dense.weight": 8,
        }
        filtered = _filter_ultraep_native_model_state_dict(state, [model])
        assert filtered == {
            "decoder.mlp.linear_fc1.weight3": 3,
            "decoder.dense.weight": 8,
        }
        assert _filter_ultraep_native_model_state_dict(state, []) is state
        assert _filter_ultraep_native_model_state_dict(
            state, [SimpleNamespace()]
        ) is state

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
        monkeypatch.setattr(
            eplb_manager_module,
            "get_collecting_eplb_managers",
            lambda: tuple(managers),
        )

    def test_fast_path_has_no_device_synchronization(self, monkeypatch):
        self._install_manager_query(monkeypatch, [])
        train_step = MagicMock(return_value="result")
        wrapped = ultraep_autotune_train_step_wrapper(train_step)
        with patch("torch.cuda.synchronize") as synchronize:
            assert wrapped(1, marker=True) == "result"
        synchronize.assert_not_called()
        train_step.assert_called_once_with(1, marker=True)

    def test_collecting_non_timed_iteration_reports_none(self, monkeypatch):
        manager = MagicMock()
        manager.autotune_needs_iteration_time = False
        self._install_manager_query(monkeypatch, [manager])
        wrapped = ultraep_autotune_train_step_wrapper(lambda: "result")
        with patch("torch.cuda.synchronize") as synchronize:
            assert wrapped() == "result"
        synchronize.assert_not_called()
        manager.autotune_iteration_end.assert_called_once_with(None)

    def test_timed_iteration_synchronizes_and_reports_all_managers(self, monkeypatch):
        first, second = MagicMock(), MagicMock()
        first.autotune_needs_iteration_time = False
        second.autotune_needs_iteration_time = True
        self._install_manager_query(monkeypatch, [first, second])
        wrapped = ultraep_autotune_train_step_wrapper(lambda: "result")
        with patch("torch.cuda.synchronize") as synchronize, patch(
            "hcu_megatron.training.ultraep_autotune.time.perf_counter",
            side_effect=[2.0, 2.125],
        ):
            assert wrapped() == "result"
        assert synchronize.call_count == 2
        first.autotune_iteration_end.assert_called_once_with(125.0)
        second.autotune_iteration_end.assert_called_once_with(125.0)

    def test_failed_train_step_does_not_advance_tuner(self, monkeypatch):
        manager = MagicMock()
        manager.autotune_needs_iteration_time = True
        self._install_manager_query(monkeypatch, [manager])

        def fail():
            raise RuntimeError("training failed")

        wrapped = ultraep_autotune_train_step_wrapper(fail)
        with patch("torch.cuda.synchronize"):
            with pytest.raises(RuntimeError, match="training failed"):
                wrapped()
        manager.autotune_iteration_end.assert_not_called()

# Real single-node EP8 integration tests.

NUM_EXPERTS = 16
NUM_LOCAL_REPLICAS = 2
TOKENS_PER_RANK = 1024
FC1_NUMEL = 1024
FC2_NUMEL = 512
LAYER_ID = 0


def _distributed_validate(label, validation):
    """Report validation failures from every rank before failing the test."""
    local_error = None
    try:
        validation()
    except Exception as error:  # noqa: BLE001 - aggregate rank-local assertions
        local_error = f"rank {dist.get_rank()}: {type(error).__name__}: {error}"

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error is not None]
    assert not failures, f"{label} failed:\n" + "\n".join(failures)


def _fill_master_tensors(tensors, base):
    rank = dist.get_rank()
    for local_index, tensor in enumerate(tensors):
        logical_id = rank * len(tensors) + local_index
        tensor.fill_(float(base + logical_id))


def _assert_filled(tensor, expected, label):
    assert bool((tensor == expected).all().item()), (
        f"{label}: expected every element to be {expected}"
    )


@pytest.fixture(scope="module")
def _ultraep_environment():
    """Apply the single-node IPC environment without leaking it to other tests."""
    runtime_environment = {
        "HSA_USE_SVM": "0",
        "MAX_NUM_NVL_PEERS": "8",
        "ROCSHMEM_BACKEND": "ipc",
        # A 64 MiB symmetric heap is ample for this test's few KiB of state
        # and avoids eight model-scale 2 GiB allocations in CI.
        "ROCSHMEM_HEAP_SIZE": str(64 * 1024**2),
    }
    with patch.dict(os.environ, runtime_environment, clear=False):
        os.environ.pop("ROCSHMEM_GDA_PROVIDER", None)
        yield


@pytest.fixture(scope="module")
def ultraep_runtime(_ultraep_environment):
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in required):
        pytest.skip("run with torchrun --nproc-per-node=8")

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        pytest.skip(f"UltraEP integration requires exactly 8 ranks, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip(f"UltraEP integration requires 8 HCUs, got {torch.cuda.device_count()}")
    ultra_ep = pytest.importorskip("ultra_ep")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )

    rank = dist.get_rank()
    num_local_master = NUM_EXPERTS // world_size
    num_local_physical = num_local_master + NUM_LOCAL_REPLICAS
    manager = None
    setup_complete = False
    try:
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=num_local_master,
            num_local_redundant_experts=NUM_LOCAL_REPLICAS,
            expert_fc1_numel=FC1_NUMEL,
            expert_fc2_numel=FC2_NUMEL,
            is_train=True,
            explicitly_destroy=True,
            max_microbatches=2,
            weight_data_dtype=torch.bfloat16,
            grad_dtype=torch.float32,
            autotune=ultra_ep.AutotuneConfig(
                enabled=True,
                start_iteration=3,
                grad_reduce_max_sms=48,
            ),
        )

        fc1_weights = [
            torch.empty(FC1_NUMEL, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fc2_weights = [
            torch.empty(FC2_NUMEL, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fc1_grads = [
            torch.empty(FC1_NUMEL, dtype=torch.float32, device="cuda")
            for _ in range(num_local_master)
        ]
        fc2_grads = [
            torch.empty(FC2_NUMEL, dtype=torch.float32, device="cuda")
            for _ in range(num_local_master)
        ]
        _fill_master_tensors(fc1_weights, 1)
        _fill_master_tensors(fc2_weights, 33)
        _fill_master_tensors(fc1_grads, 65)
        _fill_master_tensors(fc2_grads, 97)
        manager.construct_local_master_ptr_pool(
            LAYER_ID,
            fc1_weights,
            fc2_weights,
            fc1_grads,
            fc2_grads,
        )

        routing_map = torch.zeros(
            (TOKENS_PER_RANK, NUM_EXPERTS),
            dtype=torch.bool,
            device="cuda",
        )
        routing_map[:, 0] = True
        manager.update_placement(
            LAYER_ID,
            routing_map,
            verify_reduced_loads=True,
        )
        torch.cuda.synchronize()
        dist.barrier()
        setup_complete = True
        yield SimpleNamespace(
            manager=manager,
            rank=rank,
            world_size=world_size,
            num_local_master=num_local_master,
            num_local_physical=num_local_physical,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_grads=fc1_grads,
            fc2_grads=fc2_grads,
            routing_map=routing_map,
        )
    finally:
        if setup_complete:
            dist.barrier()
        if manager is not None:
            manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestUltraEPFeatureEP8:
    def test_runtime_autotune_contract(self, ultraep_runtime):
        manager = ultraep_runtime.manager

        def validate():
            assert manager.nvl_domain_size == 8
            assert manager.autotune_enabled
            assert manager.autotune_collecting
            assert manager.autotune_start_iteration == 3
            assert manager.autotune_needs_iteration_time(1) is False
            assert manager.grad_reduce_num_sms <= 48
            assert manager.num_local_master_experts == 2
            assert manager.num_local_redundant_experts == 2
            assert manager.num_global_logical_experts == NUM_EXPERTS
            assert manager.num_global_physical_experts == 32

        _distributed_validate("runtime/autotune contract", validate)

        # Exercise the same process-local warm-up callbacks used by the
        # Megatron train-step bridge without advancing into candidate search.
        manager.autotune_iteration_end(1)
        manager.autotune_iteration_end(2)
        dist.barrier()

    def test_placement_and_reroute(self, ultraep_runtime):
        manager = ultraep_runtime.manager
        routing_map = ultraep_runtime.routing_map
        probs = routing_map.to(torch.float32)
        expanded_probs, expanded_routing = manager.reroute(
            LAYER_ID,
            probs,
            routing_map,
        )

        def validate():
            assert expanded_probs.shape == (
                TOKENS_PER_RANK,
                manager.num_global_physical_experts,
            )
            assert expanded_routing.shape == expanded_probs.shape
            assert bool((expanded_routing.sum(dim=1) == 1).all().item())
            assert int(manager.logical_replica_counts[LAYER_ID, 0].item()) > 1
            observed = manager.runtime.get_global_logical_expert_loads_tensor()
            assert int(observed[0].item()) == TOKENS_PER_RANK * 8
            assert int(observed[1:].sum().item()) == 0

        _distributed_validate("placement/reroute", validate)
        dist.barrier()

    def test_async_weight_sync(self, ultraep_runtime):
        state = ultraep_runtime
        manager = state.manager
        _fill_master_tensors(state.fc1_weights, 1)
        _fill_master_tensors(state.fc2_weights, 33)
        manager.local_replica_weight_buffer.zero_()

        dist.barrier()
        event = manager.weight_sync(LAYER_ID, async_finish=True)
        manager.wait_weight_sync(event, layer_id=LAYER_ID)
        torch.cuda.synchronize()
        dist.barrier()

        local_replica_physical_ids = (
            state.rank * state.num_local_physical
            + state.num_local_master
            + torch.arange(NUM_LOCAL_REPLICAS, device="cuda")
        )

        logical_ids = manager.physical_to_logical_map[
            LAYER_ID, local_replica_physical_ids
        ].tolist()
        local_valid = sum(logical_id >= 0 for logical_id in logical_ids)
        global_valid = torch.tensor(local_valid, dtype=torch.int32, device="cuda")
        dist.all_reduce(global_valid)
        assert int(global_valid.item()) > 0

        def validate():
            for replica_index, logical_id in enumerate(logical_ids):
                if logical_id < 0:
                    continue
                _assert_filled(
                    manager.local_replica_fc1_weight_buffer[replica_index],
                    float(logical_id + 1),
                    f"replica {replica_index} FC1",
                )
                _assert_filled(
                    manager.local_replica_fc2_weight_buffer[replica_index],
                    float(logical_id + 33),
                    f"replica {replica_index} FC2",
                )

        _distributed_validate("asynchronous weight sync", validate)
        dist.barrier()

    def test_virtual_layer_grad_reduce(self, ultraep_runtime):
        state = ultraep_runtime
        manager = state.manager
        _fill_master_tensors(state.fc1_grads, 65)
        _fill_master_tensors(state.fc2_grads, 97)

        replica_fc1 = manager.local_replica_fc1_grad_buffer
        replica_fc2 = manager.local_replica_fc2_grad_buffer
        replica_base = (
            state.rank * state.num_local_physical + state.num_local_master
        )
        for replica_index in range(NUM_LOCAL_REPLICAS):
            physical_id = replica_base + replica_index
            replica_fc1[replica_index].fill_(float(physical_id + 129))
            replica_fc2[replica_index].fill_(float(physical_id + 193))

        expected_fc1 = []
        expected_fc2 = []
        logical_to_physical = manager.logical_to_physical_map[LAYER_ID]
        for local_index in range(state.num_local_master):
            logical_id = state.rank * state.num_local_master + local_index
            master_physical_id = state.rank * state.num_local_physical + local_index
            physical_ids = [
                int(value)
                for value in logical_to_physical[logical_id].tolist()
                if int(value) >= 0
            ]
            replicas = [
                physical_id
                for physical_id in physical_ids
                if physical_id != master_physical_id
            ]
            expected_fc1.append(
                float(logical_id + 65 + sum(value + 129 for value in replicas))
            )
            expected_fc2.append(
                float(logical_id + 97 + sum(value + 193 for value in replicas))
            )

        layer = SimpleNamespace(
            eplb_manager=SimpleNamespace(runtime=manager),
            _eplb_grad_reduce_event_handles={},
        )
        dist.barrier()
        moe_layer_ultraep._eplb_start_grad_reduce(layer, virtual_layer_id=LAYER_ID)
        assert LAYER_ID in layer._eplb_grad_reduce_event_handles
        moe_layer_ultraep._eplb_finish_grad_reduce(layer, virtual_layer_id=LAYER_ID)
        assert layer._eplb_grad_reduce_event_handles == {}
        torch.cuda.synchronize()
        dist.barrier()

        local_replica_physical_ids = (
            state.rank * state.num_local_physical
            + state.num_local_master
            + torch.arange(NUM_LOCAL_REPLICAS, device="cuda")
        )

        def validate():
            for local_index in range(state.num_local_master):
                _assert_filled(
                    state.fc1_grads[local_index],
                    expected_fc1[local_index],
                    f"master {local_index} FC1 grad",
                )
                _assert_filled(
                    state.fc2_grads[local_index],
                    expected_fc2[local_index],
                    f"master {local_index} FC2 grad",
                )

            valid_replicas = (
                manager.physical_to_logical_map[
                    LAYER_ID, local_replica_physical_ids
                ]
                >= 0
            )
            if bool(valid_replicas.any().item()):
                assert bool(
                    (manager.local_replica_grad_buffer[valid_replicas] == 0)
                    .all()
                    .item()
                )

        _distributed_validate("virtual-layer grad reduce", validate)
        dist.barrier()
