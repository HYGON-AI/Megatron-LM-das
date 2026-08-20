# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Compatibility shims, loaded as sitecustomize at interpreter startup
# (prepare_workspace.sh adds this dir to PYTHONPATH). No upstream source is
# modified.
#
# 1) typing.override is a 3.12+ API used by the pinned Megatron-LM submodule;
#    the CI image runs Python 3.10, so a no-op decorator provides it.
# 2) fused_weight_gradient_mlp_cuda is a compiled op shipped in the CI image
#    (apex package). It is intentionally NOT probed here: probing at startup
#    runs before torch is loaded, the import fails on libc10.so, and a stub
#    would then shadow the real module for the whole process. Call sites in
#    upstream Megatron-LM guard the import with try/except and degrade
#    gracefully when the op is unavailable.
import sys

# --- 1) typing.override (Python 3.12+) ---
if sys.version_info < (3, 12):
    import typing

    if not hasattr(typing, "override"):

        def override(method):
            method.__override__ = True
            return method

        typing.override = override
