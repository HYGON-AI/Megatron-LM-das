# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Training-loop bridge for UltraEP's one-shot runtime autotuner."""

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

    @wraps(train_step_func)
    def wrapper(*args, **kwargs):
        # Import lazily so merely installing the patch does not initialize
        # UltraEP or rocSHMEM before the first MoE layer creates its Manager.
        from hcu_megatron.core.transformer.moe.eplb_manager import (
            get_collecting_eplb_managers,
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

        # All ranks in an EP group execute these calls in the same Manager
        # construction order. The UltraEP Manager performs the slowest-rank
        # reduction internally before advancing either tuning phase.
        for manager in managers:
            manager.autotune_iteration_end(iteration_time_ms)

        return result

    return wrapper
