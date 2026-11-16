# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import os
from importlib import import_module


_MODULE_EXPORTS = {
    "schedules": (
        "OpType", "ScheduleOp", "CommunicationOp", "FusedScheduleOp",
        "detect_communication_deadlock", "PPSchedule", "GPipe", "DistTrain1F1B",
        "DistTrainEncoder1F1B", "Schedule1F1B", "Sandwich", "BigMac",
        "VPP1F1BSchedule", "SandwichVPP", "BigMacVPP", "UnifiedBigMacVPP",
    ),
    "simulator": (
        "OpDurationSpec", "MonteCarloReport", "PipelineSimulator", "Plan",
        "PlanOp", "SimulatedOp", "SimulationResult",
    ),
    "executor": ("LLMPPExecutor", "LLMPPExecutorWithVE"),
    "executor_vpp": ("LLMVPPExecutor", "LLMVPPExecutorWithVE", "LLMVPPExecutorWithDecoder"),
    "integration": ("install",),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
_VISUALIZERS = (
    "visualize_schedule", "visualize_schedule_list", "visualize_vppschedule",
    "visualize_vppschedule_list", "visualize_unified_schedule",
)
__all__ = list(_EXPORTS) + list(_VISUALIZERS)


def __getattr__(name):
    if name in _VISUALIZERS:
        def visualize(*args, **kwargs):
            return getattr(import_module(".visualize", __name__), name)(*args, **kwargs)
        value = visualize
    elif name in _EXPORTS:
        value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def _is_truthy_env(name):
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


if _is_truthy_env("MEGATRON_USE_PP_EXECUTOR") and _is_truthy_env(
    "MEGATRON_PP_EXECUTOR_AUTO_INSTALL_ON_IMPORT"
):
    __getattr__("install")()
