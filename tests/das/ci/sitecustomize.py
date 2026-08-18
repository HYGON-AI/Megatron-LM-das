# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Compatibility shims, loaded as sitecustomize at interpreter startup
# (prepare_workspace.sh adds this dir to PYTHONPATH). No upstream source is
# modified.
#
# 1) typing.override is a 3.12+ API used by the pinned Megatron-LM submodule;
#    the CI image runs Python 3.10, so a no-op decorator provides it.
# 2) fused_weight_gradient_mlp_cuda is a compiled op shipped in the prebuilt
#    hcu-megatron wheel (not in this repo). Unit tests only import it (call
#    sites are on the TE + gradient_accumulation_fusion path, unused by tests).
#    - wheel installed (e.g. via DAS_HCU_MEGATRON_WHEEL): real module wins;
#    - not installed: stub injected, calls raise NotImplementedError honestly.
import importlib
import sys

# --- 1) typing.override (Python 3.12+) ---
if sys.version_info < (3, 12):
    import typing

    if not hasattr(typing, "override"):

        def override(method):
            method.__override__ = True
            return method

        typing.override = override

# --- 2) fused_weight_gradient_mlp_cuda (provided by the wheel) ---
if "fused_weight_gradient_mlp_cuda" not in sys.modules:
    try:
        importlib.import_module("fused_weight_gradient_mlp_cuda")
    except ImportError:
        import types

        def _not_implemented(*args, **kwargs):
            raise NotImplementedError(
                "fused_weight_gradient_mlp_cuda is a compiled op provided by the "
                "hcu-megatron wheel (install it via the DAS_HCU_MEGATRON_WHEEL CI "
                "interface); this stub only satisfies the module-level import."
            )

        _stub = types.ModuleType("fused_weight_gradient_mlp_cuda")
        _stub.wgrad_gemm_accum_fp32 = _not_implemented
        _stub.wgrad_gemm_accum_fp16 = _not_implemented
        sys.modules["fused_weight_gradient_mlp_cuda"] = _stub
