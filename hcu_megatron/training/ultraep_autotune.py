# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Training bridge for UltraEP tuning and optional baseline admission."""

import math
import time
from functools import wraps

import torch


def ultraep_autotune_train_step_wrapper(train_step_func):
    """Measure and report complete train steps only while tuning needs them.

    The wrapped function already includes Megatron forward/backward and the
    optimizer step. Device synchronization is deliberately limited to active
    calibration iterations; the disabled and completed paths call the
    existing train_step directly.
    """

    adaptive_state = {}

    @wraps(train_step_func)
    def wrapper(*args, **kwargs):
        # Import lazily so merely installing the patch does not initialize
        # UltraEP or rocSHMEM before the first MoE layer creates its Manager.
        from hcu_megatron.core.transformer.moe.eplb_manager import (
            get_collecting_eplb_managers,
            get_eplb_managers,
        )

        adaptive_managers = tuple(
            manager for manager in get_eplb_managers() if manager.adaptive_config is not None
        )
        # A pipeline stage with no MoE layers has no Manager, but must still
        # join the WORLD timing/status reductions and follow the same policy.
        config = kwargs.get('config', args[5] if len(args) > 5 else None)
        if adaptive_managers or getattr(config, 'moe_ultraep_adaptive', False):
            return _adaptive_train_step(
                train_step_func, adaptive_managers, args, kwargs, adaptive_state, config,
            )

        managers = get_collecting_eplb_managers()
        if not managers:
            return train_step_func(*args, **kwargs)

        needs_timing = any(
            manager.autotune_needs_iteration_time for manager in managers
        )
        started_at = None
        if needs_timing:
            torch.cuda.synchronize()
            started_at = time.perf_counter()

        result = train_step_func(*args, **kwargs)

        iteration_time_ms = None
        if needs_timing:
            torch.cuda.synchronize()
            iteration_time_ms = (time.perf_counter() - started_at) * 1000.0
        else:
            # Runtime scouts use the EP communicator. Drain outstanding peer
            # fences before entering its barrier/metadata collectives.
            torch.cuda.synchronize()

        # All ranks in an EP group execute these calls in the same Manager
        # construction order. The UltraEP Manager performs the slowest-rank
        # reduction internally before advancing either tuning phase.
        for manager in managers:
            manager.autotune_iteration_end(iteration_time_ms)

        return result

    return wrapper


def _adaptive_workload():
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from hcu_megatron.training import get_args

    args = get_args()
    return (args.seq_length, args.micro_batch_size, get_num_microbatches())


def _set_adaptive_mode(managers, active):
    if any(manager.ultraep_active != active for manager in managers):
        # Outside the timed region. Drain every local stream before changing
        # dispatcher metadata and the backward hook's readiness policy.
        torch.cuda.synchronize()
        for manager in managers:
            manager.set_ultraep_active(active, synchronize=False)


def _adaptive_train_step(train_step_func, managers, args, kwargs, state, config):
    from .ultraep_adaptive import AdaptiveConfig, AdaptiveController

    controller = state.get('controller')
    if controller is None:
        adaptive_config = managers[0].adaptive_config if managers else AdaptiveConfig(
            samples=getattr(config, 'moe_ultraep_adaptive_samples', 3),
            max_samples=getattr(config, 'moe_ultraep_adaptive_max_samples', 5),
            min_speedup=getattr(config, 'moe_ultraep_adaptive_min_speedup', 0.01),
            max_calibration_steps=getattr(config, 'moe_ultraep_adaptive_max_calibration_steps', 48),
        )
        controller = AdaptiveController(adaptive_config)
        state['controller'] = controller
        for manager in managers:
            if manager.adaptive_config != controller.config:
                raise RuntimeError('All UltraEP Managers must use the same adaptive configuration')
            manager.adaptive_controller = controller
    if controller.complete:
        return train_step_func(*args, **kwargs)

    needs_timing = controller.needs_timing or (
        controller.active and any(
            manager.autotune_collecting and manager.autotune_needs_iteration_time for manager in managers
        )
    )
    workload = _adaptive_workload()
    forward_backward_runs = 0
    step_args, step_kwargs = list(args), dict(kwargs)
    forward_backward = kwargs.get('forward_backward_func', args[6] if len(args) > 6 else None)
    if forward_backward is not None:
        @wraps(forward_backward)
        def counted_forward_backward(*fb_args, **fb_kwargs):
            nonlocal forward_backward_runs
            forward_backward_runs += 1
            return forward_backward(*fb_args, **fb_kwargs)

        if 'forward_backward_func' in step_kwargs:
            step_kwargs['forward_backward_func'] = counted_forward_backward
        else:
            step_args[6] = counted_forward_backward
    if needs_timing:
        torch.cuda.synchronize()
        started_at = time.perf_counter()
    result = train_step_func(*step_args, **step_kwargs)
    elapsed_ms = None
    if needs_timing:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    # train_step's stable return prefix is loss, skipped_iter, checkpoint,
    # should_exit. All workers must accept/reject the same observation, even
    # when just one data-parallel replica skipped its optimizer update.
    invalid = bool(result[1] or result[3] or forward_backward_runs > 1)
    invalid = invalid or (elapsed_ms is not None and (not math.isfinite(elapsed_ms) or elapsed_ms <= 0))
    stats = torch.tensor(
        [elapsed_ms if elapsed_ms is not None and math.isfinite(elapsed_ms) else 0.0, float(invalid)],
        dtype=torch.float64, device='cuda',
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    valid = not bool(stats[1].item())
    global_elapsed_ms = float(stats[0].item()) if elapsed_ms is not None else None
    if controller.active:
        if not needs_timing and any(manager.autotune_collecting for manager in managers):
            torch.cuda.synchronize()
        for manager in managers:
            if valid:
                manager.autotune_iteration_end(elapsed_ms)
            else:
                manager.discard_autotune_measurements()
    # A world-wide status makes the policy safe if independent EP groups
    # finish their runtime tuning on different iterations.
    collecting = torch.tensor(
        [int(any(manager.autotune_collecting for manager in managers))],
        dtype=torch.int32, device='cuda',
    )
    torch.distributed.all_reduce(collecting, op=torch.distributed.ReduceOp.MAX)
    previous_phase = controller.phase
    controller.step_end(
        global_elapsed_ms, valid=valid, tuning=bool(collecting.item()), workload=workload,
    )
    _set_adaptive_mode(managers, controller.active)
    if torch.distributed.get_rank() == 0 and previous_phase != controller.phase:
        print(f'UltraEP adaptive: phase={controller.phase}, local_step={controller.steps}', flush=True)
        if controller.complete:
            print(f'UltraEP adaptive result: {controller.result}', flush=True)
    return result
