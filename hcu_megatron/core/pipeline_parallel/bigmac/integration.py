# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

from __future__ import annotations
import sys
import inspect
import os
from functools import wraps
from typing import Callable, Iterator, List, Optional, Union
from .config import encoder_enabled, trace_options


_ORIGINAL_GET_FORWARD_BACKWARD_FUNC = None
_TRUTHY_ENV_VALUES = ("1", "true", "yes", "on")
_ENV_DEFAULT = object()


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in _TRUTHY_ENV_VALUES


def _should_trace_iteration(trace_enabled: bool, trace_iter_id: Optional[int], iter_id: int) -> bool:
    if not trace_enabled:
        return False
    if trace_iter_id is None:
        return True
    return iter_id == trace_iter_id


def _should_trace_dp_rank(trace_dp_rank: Optional[int], data_parallel_rank: int) -> bool:
    return trace_dp_rank is None or data_parallel_rank == trace_dp_rank


def _cleanup_pp_executor(pp_executor):
    owner = getattr(getattr(pp_executor, "encoder_forward_func", None), "__self__", None)
    if owner is not None and hasattr(owner, "finish_pp_executor_encoder_iteration"):
        owner.finish_pp_executor_encoder_iteration(pp_executor)
    if hasattr(pp_executor, "action_map"):
        pp_executor.action_map.clear()
    if hasattr(pp_executor, "data_iterator"):
        pp_executor.data_iterator = None
    if hasattr(pp_executor, "schedule_plan"):
        pp_executor.schedule_plan = []
    if hasattr(pp_executor, "scheduler"):
        pp_executor.scheduler = None
    pp_executor.forward_data_store = []


def _wrap_forward_backward_with_pp_executor(
    forward_backward_func,
    vp_size,
    trace_enabled: bool = False,
    trace_dir: Optional[str] = None,
    trace_iter_id: Optional[int] = None,
    trace_dp_rank: Optional[int] = 0,
    trace_state: Optional[dict] = None,
    with_ve: Optional[bool] = None,
):
    from megatron.core import parallel_state, mpu
    from megatron.core.pipeline_parallel import schedules
    if trace_state is None:
        trace_state = {"iter_id": 0}

    def call_original_forward_backward(**kwargs):
        signature = inspect.signature(forward_backward_func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return forward_backward_func(**kwargs)
        accepted_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
        return forward_backward_func(**accepted_kwargs)

    def forward_backward_pp_executor_wrapper(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,
        adjust_tensor_shapes_fn: Optional[Callable] = None,
        p2p_communicator: Optional[object] = None,
        pg_collection: Optional[object] = None,
        force_all_reduce: Optional[bool] = False,
        **extra_kwargs,
    ):
        use_ve_executor = encoder_enabled() if with_ve is None else with_ve
        unsupported = (
            forward_only
            or first_val_step is not None
            or adjust_tensor_shapes_fn is not None
            or p2p_communicator is not None
            or pg_collection is not None
            or force_all_reduce
            or decoder_seq_length is not None
            or bool(extra_kwargs)
        )
        if unsupported:
            if use_ve_executor and not forward_only:
                raise NotImplementedError(
                    "BigMac encoder training does not support custom pipeline groups, "
                    "tensor-shape callbacks or decoder schedules."
                )
            return call_original_forward_backward(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=decoder_seq_length,
                forward_only=forward_only,
                collect_non_loss_data=collect_non_loss_data,
                first_val_step=first_val_step,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
                p2p_communicator=p2p_communicator,
                pg_collection=pg_collection,
                force_all_reduce=force_all_reduce,
                **extra_kwargs,
            )

        current_iter_id = trace_state["iter_id"]
        trace_state["iter_id"] += 1
        forward_step_owner = getattr(forward_step_func, "__self__", None)
        if use_ve_executor:
            assert forward_step_owner is not None, (
                "MEGATRON_PP_EXECUTOR_WITH_VE requires a bound forward_step object "
                "that provides BigMac encoder hooks"
            )
            required_hooks = (
                "get_encoder_models",
                "encoder_forward_func",
                "get_encoder_output_meta",
            )
            missing_hooks = [
                hook for hook in required_hooks if not hasattr(forward_step_owner, hook)
            ]
            assert not missing_hooks, (
                "MEGATRON_PP_EXECUTOR_WITH_VE missing forward_step owner hooks: "
                + ", ".join(missing_hooks)
            )
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            if num_microbatches % pp_size:
                raise ValueError("BigMac encoder scheduling requires num_microbatches divisible by PP size")

        if vp_size is None:
            if use_ve_executor:
                from .executor import LLMPPExecutorWithVE

                executor_cls = LLMPPExecutorWithVE
                executor_args = ()
            else:
                from .executor import LLMPPExecutor

                executor_cls = LLMPPExecutor
                executor_args = ()
        else:
            if use_ve_executor:
                raise NotImplementedError("BigMac VE executor is only wired for non-VPP PP")
            from .executor_vpp import LLMVPPExecutor

            executor_cls = LLMVPPExecutor
            executor_args = (vp_size,)

        executor_kwargs = {}
        if use_ve_executor:
            executor_kwargs.update(
                encoder_models=forward_step_owner.get_encoder_models(model),
                encoder_forward_func=forward_step_owner.encoder_forward_func,
                get_encoder_output_meta_func=forward_step_owner.get_encoder_output_meta,
            )

        should_trace = _should_trace_iteration(
            trace_enabled,
            trace_iter_id,
            current_iter_id,
        ) and _should_trace_dp_rank(
            trace_dp_rank,
            mpu.get_data_parallel_rank(),
        )

        pp_executor = executor_cls(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            forward_only,
            parallel_state.get_pipeline_model_parallel_world_size(),
            *executor_args,
            parallel_state.get_pipeline_model_parallel_group(),
            parallel_state.get_pipeline_model_parallel_rank(),
            collect_non_loss_data=collect_non_loss_data,
            trace_enabled=should_trace,
            trace_dir=trace_dir,
            trace_iter_id=current_iter_id if should_trace else None,
            **executor_kwargs,
        )
        get_vp_rank = getattr(parallel_state, "get_virtual_pipeline_model_parallel_rank", None)
        previous_vp_rank = get_vp_rank() if vp_size is not None and get_vp_rank else None
        try:
            if use_ve_executor and hasattr(forward_step_owner, "configure_pp_executor"):
                forward_step_owner.configure_pp_executor(pp_executor)
            pp_executor.execute()
            forward_data_store = pp_executor.forward_data_store
        finally:
            _cleanup_pp_executor(pp_executor)
            del pp_executor
            if vp_size is not None and get_vp_rank:
                parallel_state.set_virtual_pipeline_model_parallel_rank(previous_vp_rank)
        return forward_data_store

    return forward_backward_pp_executor_wrapper


def get_forward_backward_func_wrapper(original, *, with_ve=None, **options):
    # Compose with DAS's selector; never capture a mutable global original.
    from megatron.core import parallel_state
    if getattr(original, "_hcu_bigmac_owner", None) is original:
        original._hcu_bigmac_options.update(options)
        if with_ve is not None:
            original._hcu_bigmac_options["with_ve"] = with_ve
        return original
    # Legacy auto-install may have run before DAS added its own selector
    # wrapper. Update that inner installation as well so explicit DAS CLI
    # options take precedence over its environment-derived defaults.
    inner = getattr(original, "__wrapped__", None)
    seen = set()
    while inner is not None and id(inner) not in seen:
        seen.add(id(inner))
        if getattr(inner, "_hcu_bigmac_owner", None) is inner:
            inner._hcu_bigmac_options.update(options)
            if with_ve is not None:
                inner._hcu_bigmac_options["with_ve"] = with_ve
        inner = getattr(inner, "__wrapped__", None)
    settings = trace_options()
    settings.update(options)
    settings["with_ve"] = with_ve
    trace_state = {"iter_id": 0}
    @wraps(original)
    def get_forward_backward_func(pp_size=None, vp_size=None):
        if pp_size is None and vp_size is None:
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        original_signature = inspect.signature(original)
        if len(original_signature.parameters) == 0:
            forward_backward_func = original()
        else:
            forward_backward_func = original(pp_size=pp_size, vp_size=vp_size)
        # Explicit DAS alternatives (DualPipeV, RiPipe, vocab parallel, etc.)
        # own their execution semantics. Do not wrap them a second time.
        if pp_size > 1 and not forward_backward_func.__module__.startswith("hcu_megatron."):
            forward_backward_func = _wrap_forward_backward_with_pp_executor(
                forward_backward_func,
                vp_size,
                trace_state=trace_state,
                **settings,
            )
        return forward_backward_func

    get_forward_backward_func._pp_executor_patched = True
    get_forward_backward_func._hcu_bigmac_patched = True
    get_forward_backward_func._hcu_bigmac_owner = get_forward_backward_func
    get_forward_backward_func._pp_executor_original_get_forward_backward_func = original
    get_forward_backward_func._hcu_bigmac_options = settings
    return get_forward_backward_func


def install(trace_enabled=None, trace_dir=None, trace_iter_id=None, trace_dp_rank=_ENV_DEFAULT):
    # Legacy explicit installation; DAS normally registers through BigMacFeature.
    global _ORIGINAL_GET_FORWARD_BACKWARD_FUNC
    from megatron.core import pipeline_parallel
    from megatron.core.pipeline_parallel import schedules

    original = schedules.get_forward_backward_func
    options = trace_options()
    if trace_enabled is not None:
        options["trace_enabled"] = trace_enabled
    if trace_dir is not None:
        options["trace_dir"] = trace_dir
    if trace_iter_id is not None:
        options["trace_iter_id"] = trace_iter_id
    if trace_dp_rank is not _ENV_DEFAULT:
        options["trace_dp_rank"] = trace_dp_rank
    patched = get_forward_backward_func_wrapper(original, **options)
    _ORIGINAL_GET_FORWARD_BACKWARD_FUNC = patched._pp_executor_original_get_forward_backward_func
    # Also replace aliases imported before installation, including DAS training.
    for name, module in list(sys.modules.items()):
        if name.startswith(("megatron.", "hcu_megatron.")) and module is not None:
            if vars(module).get("get_forward_backward_func") is original:
                module.get_forward_backward_func = patched
    schedules.get_forward_backward_func = patched
    pipeline_parallel.get_forward_backward_func = patched
    return patched
