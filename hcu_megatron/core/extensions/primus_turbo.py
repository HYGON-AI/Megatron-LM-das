###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE for license information.
###############################################################################
import contextlib
import gc
import os
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional, Tuple, Union

import primus_turbo.pytorch as primus_turbo_torch
import torch

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    TEGroupedLinear,
    TEQuantizationParams,
    TEQuantizationRecipe,
    condition_init_method,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size
from megatron.training.global_vars import get_args

# QuantizedTensor / QuantizedTensorPair are only used in the FP8/FP4 weight
# quantization paths (added in PR #735).  Older primus_turbo 0.2.0 builds shipped
# in the rocm/primus v26.2 / v26.3 containers do not export them yet.  Keep the
# module importable so the BF16 turbo attention / linear paths still work, and only
# fail (with a clear AttributeError on None) if an FP8 quantization path is hit.
try:
    from primus_turbo.pytorch.core import QuantizedTensor as PrimusTurboQuantizedTensor
    from primus_turbo.pytorch.core import (
        QuantizedTensorPair as PrimusTurboQuantizedTensorPair,
    )
except (ImportError, ModuleNotFoundError):
    PrimusTurboQuantizedTensor = None
    PrimusTurboQuantizedTensorPair = None

# ScalingRecipe was renamed to MXScalingRecipe in primus_turbo 0.2.0; keep a fallback
# alias so the module imports against both old and new builds.
try:
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe
except (ImportError, ModuleNotFoundError):
    from primus_turbo.pytorch.core.low_precision import MXScalingRecipe as ScalingRecipe

try:
    from primus_turbo.pytorch.core.quantized_tensor import create_quantized_weight
except (ImportError, ModuleNotFoundError):
    create_quantized_weight = None

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingStrategy,
    check_fp8_support,
    check_mxfp4_support,
    check_mxfp8_support,
    float8_e4m3,
)

# Imported from .constants (not .fp8) for TransformerEngine >= 2.12 compat;
# the symbol moved out of transformer_engine.pytorch.fp8 in that release.
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, Recipe

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAVE_TRITON = False

_dummy_wgrads = {}


if _HAVE_TRITON:

    @triton.jit
    def _inplace_add_kernel(dst_ptr, src_ptr, n_elements, BLOCK: tl.constexpr):
        """In-place ``dst += src`` over a flat buffer, accumulating in fp32.

        int64 offsets so a single launch covers tensors with > 2**31 elements
        (e.g. the consolidated grouped-expert ``main_grad`` of [E, N, K]),
        instead of Torch's ``add_`` which tiles into multiple ~528M-element
        ``vectorized_elementwise_kernel`` launches.
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        d = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + offs, (d + s).to(dst_ptr.dtype.element_ty), mask=mask)


def _triton_inplace_add_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst.add_(src)`` via a single Triton launch (fp32 accumulate).

    Falls back to Torch's ``add_`` when Triton is unavailable or the layout is
    unsupported (non-contiguous / shape mismatch). The write is in-place on
    ``dst``'s storage, so ``dst`` must be contiguous.
    """
    if not _HAVE_TRITON or not dst.is_cuda or not dst.is_contiguous() or dst.numel() != src.numel():
        return dst.add_(src)

    dst_flat = dst.view(-1)
    # reshape (not view) so a non-contiguous grad is materialized contiguously.
    src_flat = src.reshape(-1)
    n_elements = dst_flat.numel()
    BLOCK = 8192
    grid = (triton.cdiv(n_elements, BLOCK),)
    _inplace_add_kernel[grid](dst_flat, src_flat, n_elements, BLOCK=BLOCK)
    return dst


def _get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False) -> torch.Tensor:
    """Returns a dummy tensor of given shape.

    Supports arbitrary rank (2D for plain Linear weights, 3D for stacked
    grouped-linear weights ``(num_gemms, out_features, in_features)``, etc.).
    Tensors are cached by ``(shape, dtype)`` so each distinct weight layout
    only allocates one persistent buffer that gets reused across steps.
    """
    global _dummy_wgrads
    key = (tuple(shape), dtype)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()


class _MainGradShim:
    """Per-expert handle for primus_turbo's ``fused_grouped_wgrad`` over a
    *consolidated* ``[E, N, K]`` grouped-expert weight (OPT-1).

    ``PrimusTurboGroupedLinear`` keeps one consolidated weight with a single
    ``main_grad`` block, but ``fused_grouped_wgrad`` / ``_expert_main_grad_view``
    expect a list of per-expert handles, each exposing a 2-D ``main_grad`` and a
    ``grad_added_to_main_grad`` flag. These shims point at the contiguous 2-D
    slices ``main_grad[i]`` of the consolidated block, so the grouped GEMM
    backward accumulates each expert's wgrad straight into the right slice.
    """

    __slots__ = ("main_grad", "grad_added_to_main_grad")

    def __init__(self, main_grad_slice: torch.Tensor) -> None:
        self.main_grad = main_grad_slice
        self.grad_added_to_main_grad = False


def _bridge_weight_grad(
    x: torch.Tensor, weight: torch.nn.Parameter, weight_buffer: PrimusTurboQuantizedTensorPair
):
    """Bridge quantized weight gradient to the original weight's ``main_grad``.

    Must be called **before** the gemm so that in the backward pass the gemm
    backward fires first (producing the real weight gradient) and then
    ``_WeightGradBridge.backward`` receives it, writes it into
    ``weight.main_grad``, and emits a dummy wgrad so that ``weight``'s
    AccumulateGrad / DDP ``register_grad_ready`` hook fires in the correct
    order.

    """

    class _WeightGradBridge(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight, quantized_weight, quantized_weight_trans):
            ctx.save_for_backward(weight)
            return x, quantized_weight, quantized_weight_trans

        @staticmethod
        def backward(ctx, grad_x, grad_quantized_weight, grad_quantized_weight_trans):
            (weight,) = ctx.saved_tensors
            assert hasattr(weight, "main_grad"), "weight.main_grad should be set before backward pass."
            assert hasattr(
                weight, "grad_added_to_main_grad"
            ), "weight.grad_added_to_main_grad don't have grad_added_to_main_grad attribute."

            # NOTE: Set weight.grad_added_to_main_grad to True to avoid adding the
            # quantized weight gradient to main_grad twice.
            if grad_quantized_weight is None:
                # OPT-1 fused path: the grouped GEMM backward already accumulated the
                # expert wgrad straight into main_grad (under fused_grouped_wgrad) and
                # returned grad_b=None, so there is nothing to add here -- just flag it.
                weight.grad_added_to_main_grad = True
            else:
                # `is_gfx1250` only exists in newer primus_turbo builds (it gates a
                # gfx1250-specific elementwise-add workaround). Older / feature-branch
                # primus_turbo that predate it (e.g. the flydsl sparse-MLA attention branch)
                # don't define it; treat a missing symbol as False so those builds still work
                # on non-gfx1250 archs (gfx942 / gfx950) instead of raising ImportError.
                try:
                    from primus_turbo.pytorch.core.utils import is_gfx1250

                    _use_triton_inplace_add = is_gfx1250()
                except ImportError:
                    _use_triton_inplace_add = False

                if _use_triton_inplace_add:
                    # NOTE: The bandwith of torch's elementwise add kernel has issue. Use triton to temporary workaround for gfx1250.
                    _triton_inplace_add_(weight.main_grad, grad_quantized_weight)
                else:
                    weight.main_grad.add_(grad_quantized_weight)

                weight.grad_added_to_main_grad = True

            return grad_x, _get_dummy_wgrad(list(weight.shape), weight.dtype), None, None

    assert isinstance(
        weight_buffer, PrimusTurboQuantizedTensorPair
    ), "weight_buffer must be a PrimusTurboQuantizedTensorPair"
    assert weight_buffer.data is not None, "weight_buffer.data must not be None"

    x, quantized_weight, quantized_weight_trans = _WeightGradBridge.apply(
        x, weight, weight_buffer.data, weight_buffer.data_t
    )

    # wrapper quantized_weight and quantized_weight_trans into PrimusTurboQuantizedTensorPair
    return x, PrimusTurboQuantizedTensorPair(data=quantized_weight, data_t=quantized_weight_trans)


def _maybe_create_quantized_weight_buffers(
    weight: torch.Tensor,
    dest_dtype: torch.dtype,
    quant_config: "PrimusTurboQuantConfig",
    disable_parameter_transpose_cache: bool,
):
    """Quantize ``weight`` into a rowwise buffer plus an optional transposed
    (colwise) buffer, returning ``(rowwise, colwise_or_None)``.

    Prefers primus_turbo's ``create_quantized_weight`` helper, which picks the
    scaling recipe from the quant config and handles per-granularity transpose.
    Falls back to a manual rowwise/colwise quantize on older primus_turbo builds
    that do not export ``create_quantized_weight`` yet.
    """
    quant_config_internal = quant_config.data()
    need_weight_transpose_cache = not disable_parameter_transpose_cache

    if create_quantized_weight is not None:
        return create_quantized_weight(
            weight,
            dest_dtype,
            quant_config_internal,
            need_weight_transpose_cache=need_weight_transpose_cache,
        )

    # TODO(ruibin): Remove this fallback path once create_quantized_weight is
    # always available in the shipped primus_turbo build.
    def _weight_scaling_recipe(quant_config: Union[Float4QuantConfig, Float8QuantConfig]) -> ScalingRecipe:
        if isinstance(quant_config, Float4QuantConfig):
            weight_scaling_recipe = ScalingRecipe(
                use_2d_block=True,
                shuffle_scale=quant_config.use_preshuffle,
                shuffle_out=quant_config.use_preshuffle,
            )

        if isinstance(quant_config, Float8QuantConfig):
            if quant_config.granularity in [ScalingGranularity.BLOCKWISE, ScalingGranularity.MX_BLOCKWISE]:
                weight_scaling_recipe = ScalingRecipe(use_2d_block=True)
            else:
                weight_scaling_recipe = ScalingRecipe()

        return weight_scaling_recipe

    quantized_weight_rowwise = PrimusTurboQuantizedTensor.quantize(
        weight,
        dest_dtype=dest_dtype,
        granularity=quant_config.granularity,
        block_size=quant_config.block_size,
        scaling_recipe=_weight_scaling_recipe(quant_config),
        axis=-1,
    )

    quantized_weight_colwise = None
    if need_weight_transpose_cache:
        granularity = quant_config.granularity
        if granularity == ScalingGranularity.TENSORWISE:
            quantized_weight_colwise = quantized_weight_rowwise.transpose(-2, -1)
        elif granularity == ScalingGranularity.ROWWISE:
            # NOTE: rowwise quantization not support transpose, so we need to quantize the transposed weight manually.
            quantized_weight_colwise = PrimusTurboQuantizedTensor.quantize(
                weight.transpose(-2, -1),
                dest_dtype=dest_dtype,
                granularity=quant_config.granularity,
                block_size=quant_config.block_size,
                scaling_recipe=_weight_scaling_recipe(quant_config),
                axis=-2,
            )
        elif granularity in [ScalingGranularity.BLOCKWISE, ScalingGranularity.MX_BLOCKWISE]:
            quantized_weight_colwise = PrimusTurboQuantizedTensor.quantize(
                weight,
                dest_dtype=dest_dtype,
                granularity=quant_config.granularity,
                block_size=quant_config.block_size,
                scaling_recipe=_weight_scaling_recipe(quant_config),
                # axis=-2 means quant weight along axis 2 which will get a transposed quantized weight.
                axis=-2,
            )
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

    return quantized_weight_rowwise, quantized_weight_colwise


def _call_fp8_autocast_enter(
    *,
    enabled: bool,
    calibrating: bool,
    fp8_recipe: Optional[Recipe],
    fp8_group: Optional[dist_group_type],
    _graph: bool,
) -> None:
    """Dispatch to whichever FP8 enter API the installed TE exposes."""
    enter_fn = getattr(FP8GlobalStateManager, "autocast_enter", None)
    if enter_fn is None:
        enter_fn = getattr(FP8GlobalStateManager, "fp8_autocast_enter", None)
    if enter_fn is None:
        raise AttributeError("FP8GlobalStateManager has no autocast enter API")
    enter_fn(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
    )


def _call_fp8_autocast_exit(enabled: bool, *, _graph: bool) -> None:
    """Dispatch to whichever FP8 exit API the installed TE exposes."""
    exit_fn = getattr(FP8GlobalStateManager, "autocast_exit", None)
    if exit_fn is None:
        exit_fn = getattr(FP8GlobalStateManager, "fp8_autocast_exit", None)
    if exit_fn is None:
        raise AttributeError("FP8GlobalStateManager has no autocast exit API")
    exit_fn(enabled, _graph=_graph)


class PrimusTurboQuantConfig:

    def __init__(
        self,
        format: Format = Format.E4M3,
        granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
        strategy: ScalingStrategy = ScalingStrategy.DYNAMIC,
        scale_dtype: ScaleDtype = ScaleDtype.FP32,
        block_size: int = None,
        use_gradient_sr: bool = True,
    ):
        self._is_fp4 = False
        self._is_fp8 = False
        if format == Format.E2M1_X2:
            # FP4
            self._quant_config = Float4QuantConfig(
                format=format,
                granularity=granularity,
                strategy=strategy,
                scale_dtype=scale_dtype,
                block_size=block_size,
                use_gradient_sr=use_gradient_sr,
            )
            self._is_fp4 = True
        else:
            # FP8
            self._quant_config = Float8QuantConfig(
                format=format,
                granularity=granularity,
                strategy=strategy,
                scale_dtype=scale_dtype,
                block_size=block_size,
            )
            self._is_fp8 = True

    def data(self):
        return self._quant_config

    def is_fp4(self):
        return self._is_fp4

    def is_fp8(self):
        return self._is_fp8

    def block_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
        )

    def current_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.TENSORWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
        )

    def mxfp8_scaling(self):
        # NOTE: The mxfp8 recipe only support e4m3 format in megatron-lm backend.
        return (
            self._quant_config.granularity == ScalingGranularity.MX_BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
            and self._quant_config.format == Format.E4M3
        )

    def mxfp4_scaling(self):
        return (
            self._quant_config.granularity == ScalingGranularity.MX_BLOCKWISE
            and self._quant_config.strategy == ScalingStrategy.DYNAMIC
            and self._quant_config.format == Format.E2M1_X2
            and self._quant_config.scale_dtype == ScaleDtype.E8M0
        )


class PrimusTurboLowPrecisionGlobalStateManager(FP8GlobalStateManager):
    PRIMUS_TURBO_QUANT_CONFIG: PrimusTurboQuantConfig = None
    PRIMUS_TURBO_FP8_ENABLED: bool = False
    PRIMUS_TURBO_FP4_ENABLED: bool = False

    @classmethod
    def is_turbo_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.PRIMUS_TURBO_FP8_ENABLED

    @classmethod
    def is_turbo_fp4_enabled(cls) -> bool:
        """Is FP4 enabled"""
        return cls.PRIMUS_TURBO_FP4_ENABLED

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        FP8GlobalStateManager.reset()

        cls.PRIMUS_TURBO_FP8_ENABLED = False
        cls.PRIMUS_TURBO_FP4_ENABLED = False
        cls.PRIMUS_TURBO_QUANT_CONFIG = None

    @classmethod
    def autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
        enabled_turbo: bool = False,
        turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
    ) -> None:
        _call_fp8_autocast_enter(
            enabled=enabled,
            calibrating=calibrating,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
            _graph=_graph,
        )

        # Default is fp8 tensorwise
        turbo_quant_config = PrimusTurboQuantConfig() if turbo_quant_config is None else turbo_quant_config

        cls.PRIMUS_TURBO_FP8_ENABLED = enabled_turbo and turbo_quant_config.is_fp8()
        cls.PRIMUS_TURBO_FP4_ENABLED = enabled_turbo and turbo_quant_config.is_fp4()
        cls.PRIMUS_TURBO_QUANT_CONFIG = turbo_quant_config

        if enabled_turbo:
            fp8_available, reason_for_no_fp8 = check_fp8_support()
            assert fp8_available, reason_for_no_fp8
            if turbo_quant_config.mxfp8_scaling():
                mxfp8_available, reason_for_no_mxfp8 = check_mxfp8_support()
                assert mxfp8_available, reason_for_no_mxfp8
            if turbo_quant_config.mxfp4_scaling():
                mxfp4_available, reason_for_no_mxfp4 = check_mxfp4_support()
                assert mxfp4_available, reason_for_no_mxfp4

    @classmethod
    def get_turbo_quant_config(cls) -> PrimusTurboQuantConfig:
        """Return the turbo's quant_config"""
        return cls.PRIMUS_TURBO_QUANT_CONFIG

    @classmethod
    def get_fp8_autocast_state(
        cls,
    ) -> Tuple[bool, bool, Recipe, dist_group_type, bool, bool, bool, bool, PrimusTurboQuantConfig]:
        """FP8 autocast state getter"""
        return (
            FP8GlobalStateManager.FP8_ENABLED,
            FP8GlobalStateManager.FP8_CALIBRATION,
            FP8GlobalStateManager.FP8_RECIPE,
            FP8GlobalStateManager.FP8_DISTRIBUTED_GROUP,
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE,
            FP8GlobalStateManager.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP4_ENABLED,
            cls.PRIMUS_TURBO_QUANT_CONFIG,
        )

    @classmethod
    def set_fp8_autocast_state(
        cls,
        fp8_state: Tuple[bool, bool, Recipe, dist_group_type, bool, bool, bool, bool, PrimusTurboQuantConfig],
    ) -> None:
        """FP8 autocast state setter"""
        (
            FP8GlobalStateManager.FP8_ENABLED,
            FP8GlobalStateManager.FP8_CALIBRATION,
            FP8GlobalStateManager.FP8_RECIPE,
            FP8GlobalStateManager.FP8_DISTRIBUTED_GROUP,
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE,
            FP8GlobalStateManager.FP8_GRAPH_CAPTURING,
            cls.PRIMUS_TURBO_FP8_ENABLED,
            cls.PRIMUS_TURBO_FP4_ENABLED,
            cls.PRIMUS_TURBO_QUANT_CONFIG,
        ) = fp8_state


@contextmanager
def primus_turbo_fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
    enabled_turbo: bool = False,
    turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
) -> None:  # type: ignore
    fp8_state = PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state()
    PrimusTurboLowPrecisionGlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
        enabled_turbo=enabled_turbo,
        turbo_quant_config=turbo_quant_config,
    )
    try:
        yield
    finally:
        PrimusTurboLowPrecisionGlobalStateManager.set_fp8_autocast_state(fp8_state)
        # Use the base TE state manager so depth accounting stays in sync
        # across both old and new TE autocast APIs.
        _call_fp8_autocast_exit(enabled, _graph=_graph)


@contextmanager
def primus_turbo_fp4_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp4_recipe: Optional[Recipe] = None,
    fp4_group: Optional[dist_group_type] = None,
    _graph: bool = False,
    enabled_turbo: bool = False,
    turbo_quant_config: Optional[PrimusTurboQuantConfig] = None,
) -> None:  # type: ignore
    # TE currently uses fp8_autocast for fp8 and fp4 quantization.
    fp8_state = PrimusTurboLowPrecisionGlobalStateManager.get_fp8_autocast_state()
    PrimusTurboLowPrecisionGlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp4_recipe,
        fp8_group=fp4_group,
        _graph=_graph,
        enabled_turbo=enabled_turbo,
        turbo_quant_config=turbo_quant_config,
    )
    try:
        yield
    finally:
        PrimusTurboLowPrecisionGlobalStateManager.set_fp8_autocast_state(fp8_state)
        _call_fp8_autocast_exit(enabled, _graph=_graph)


def _get_fp8_autocast_for_quant_recipe(qrecipe: TEQuantizationRecipe):
    if FP8GlobalStateManager.is_fp8_enabled():
        if not qrecipe.override_quantized_autocast:
            return nullcontext()
    else:
        if not qrecipe.override_nonquantized_autocast:
            return nullcontext()

    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return primus_turbo_fp8_autocast(enabled=False, enabled_turbo=False)
    else:
        if (
            qrecipe.fp8_quantization_recipe == Fp8Recipe.custom
            or qrecipe.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            assert qrecipe.custom_recipe_factory is not None
            assert False, "Custom recipe is not supported for Primus-Turbo"

        elif qrecipe.fp8_quantization_recipe is not None:
            from primus.backends.megatron.core.fp8_utils import (
                MXFP8_SCALING_BLOCK_SIZE,
                SCALING_BLOCK_SIZE,
            )

            if qrecipe.fp8_format == "e4m3":
                fp8_format = Format.E4M3
            elif qrecipe.fp8_format == "hybrid":
                fp8_format = Format.HYBRID
            else:
                raise ValueError(f"Unhandled fp8_format {qrecipe.fp8_format}")

            if qrecipe.fp8_quantization_recipe == Fp8Recipe.tensorwise:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.TENSORWISE, format=fp8_format
                )
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.blockwise:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.BLOCKWISE, format=fp8_format, block_size=SCALING_BLOCK_SIZE
                )
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.mxfp8:
                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.MX_BLOCKWISE,
                    format=fp8_format,
                    block_size=MXFP8_SCALING_BLOCK_SIZE,
                    scale_dtype=ScaleDtype.E8M0,
                )
            else:
                raise ValueError(f"Unhandled fp8 recipe: {qrecipe.fp8_quantization_recipe}")

            return primus_turbo_fp8_autocast(
                enabled=False, enabled_turbo=True, turbo_quant_config=quant_recipe
            )
        else:
            # Fp4 configured.
            if qrecipe.fp4_quantization_recipe == Fp4Recipe.nvfp4:
                assert False, "NVFP4 is not supported for Primus-Turbo"
            elif qrecipe.fp4_quantization_recipe == Fp4Recipe.mxfp4:
                from primus.backends.megatron.core.fp4_utils import (
                    MXFP4_SCALING_BLOCK_SIZE,
                )

                quant_recipe = PrimusTurboQuantConfig(
                    granularity=ScalingGranularity.MX_BLOCKWISE,
                    format=Format.E2M1_X2,
                    block_size=MXFP4_SCALING_BLOCK_SIZE,
                    scale_dtype=ScaleDtype.E8M0,
                )
            else:
                raise ValueError(f"Unhandled fp4 recipe: {qrecipe.fp4_quantization_recipe}")

            return primus_turbo_fp4_autocast(
                enabled=False, enabled_turbo=True, turbo_quant_config=quant_recipe
            )


def _get_fp8_autocast_for_quant_params(qparams: TEQuantizationParams | None, training: bool):
    if qparams is None:
        return nullcontext()
    elif not training and qparams.evaluation_recipe is not None:
        return _get_fp8_autocast_for_quant_recipe(qparams.evaluation_recipe)
    else:
        return _get_fp8_autocast_for_quant_recipe(qparams.training_recipe)


def fused_bias_act_with_probs(
    intermediate_parallel: torch.Tensor,
    bias_parallel: torch.Tensor,
    permuted_probs: torch.Tensor,
    tokens_per_experts: torch.Tensor,
    activation_func: str,
):
    assert intermediate_parallel.ndim == 2
    assert permuted_probs.ndim == 1
    assert tokens_per_experts.device == intermediate_parallel.device

    # TODO(ruibin): fuse bias addition with activation function
    if bias_parallel is not None:
        intermediate_parallel = intermediate_parallel + bias_parallel

    num_tokens = intermediate_parallel.shape[0]
    row_mask = primus_turbo_torch.ops.tokens_per_expert_to_mask(tokens_per_experts, num_tokens)

    # TODO(ruibin): support more activation functions
    if activation_func == "silu":
        fused_act_with_probs = primus_turbo_torch.ops.swiglu_with_probs
    elif activation_func == "gelu":
        fused_act_with_probs = primus_turbo_torch.ops.geglu_with_probs
    else:
        raise ValueError(f"Activation function {activation_func} is not supported.")

    return fused_act_with_probs(intermediate_parallel, permuted_probs, row_mask)


class PrimusTurboGroupedLinear(TEGroupedLinear):
    """
    Wrapper for the PrimusTurbo `grouped_gemm` ops.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        args = get_args()
        self.offload = False # args.offload and "column_parallel_gemm" in args.offload_ops
        assert not self.offload, "gemm offload still have some problems"

        super().__init__(
            num_gemms,
            input_size,
            output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
            name=name,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboGroupedLinear only supports tensor parallel size = 1"

        w0 = self.weight0
        buffer = torch.empty(
            self.num_gemms,
            self.out_features,
            self.in_features,
            device=w0.device,
            dtype=w0.dtype,
        )

        with torch.no_grad():
            for i in range(self.num_gemms):
                weight = getattr(self, f"weight{i}")
                buffer[i].copy_(weight)

        self.register_parameter("weights", torch.nn.Parameter(buffer))

        # Capture the per-expert weights' extra attributes BEFORE deleting them.
        saved_weight_attrs = [dict(getattr(self, f"weight{i}").__dict__) for i in range(self.num_gemms)]

        # All experts share the same routing/parallel markers, so weight0's are
        # representative for the consolidated parameter.
        for attr_name, attr_val in saved_weight_attrs[0].items():
            setattr(self.weights, attr_name, attr_val)

        # Free the per-expert weight{i} Parameters now that their data has been
        # consolidated into self.weights.
        for i in range(self.num_gemms):
            name = f"weight{i}"
            if name in self._parameters:
                del self._parameters[name]

        gc.collect()
        torch.cuda.empty_cache()

        # Defer weight{i} view registration until after DDP has remapped
        # self.weights into the distributed-optimizer param buffer. Registering
        # views here would pin the pre-remap storage and leave a duplicate copy
        # of the consolidated weights resident on GPU.
        self._saved_weight_attrs = saved_weight_attrs
        self._weight_views_registered = False
        self.register_forward_pre_hook(self._forward_pre_hook_ensure_weight_views)

        self.register_buffer("quantized_weight_buffer", None, persistent=False)
        self.register_buffer("quantized_weight_t_buffer", None, persistent=False)

    def _ensure_weight_views(self) -> None:
        """Register per-expert weight{i} views after DDP param-buffer remap."""
        if self._weight_views_registered:
            return

        for i in range(self.num_gemms):
            weight_i = torch.nn.Parameter(self.weights[i], requires_grad=False)
            for attr_name, attr_val in self._saved_weight_attrs[i].items():
                setattr(weight_i, attr_name, attr_val)
            self.register_parameter(f"weight{i}", weight_i)

        self._weight_views_registered = True

    @staticmethod
    def _forward_pre_hook_ensure_weight_views(module, _inputs):
        module._ensure_weight_views()

    def state_dict(self, *args, **kwargs):
        self._ensure_weight_views()
        return super().state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor, m_splits: torch.Tensor):
        _is_first_microbatch = self.is_first_microbatch
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = self.forward_internal(x, m_splits, _is_first_microbatch)

        self.is_first_microbatch = False

        return out

    def forward_internal(
        self,
        x: torch.Tensor,
        m_splits: torch.Tensor,
        is_first_microbatch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward step of the legacy PrimusTurbo grouped-gemm MLP."""
        weights = self.weights
        # NOTE: keep x and m_splits on the same device
        if m_splits.device != x.device:
            m_splits = m_splits.to(x.device)

        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            assert (
                quant_config.mxfp8_scaling() or quant_config.current_scaling() or quant_config.block_scaling()
            ), "Turbo FP8 is enabled but quant config is not mxfp8, current scaling, or block scaling."

            if is_first_microbatch:
                (
                    self.quantized_weight_buffer,
                    self.quantized_weight_t_buffer,
                ) = _maybe_create_quantized_weight_buffers(
                    weights,
                    float8_e4m3,
                    quant_config,
                    disable_parameter_transpose_cache=self.disable_parameter_transpose_cache,
                )

            x, quantized_weights = _bridge_weight_grad(
                x,
                weights,
                PrimusTurboQuantizedTensorPair(
                    data=self.quantized_weight_buffer, data_t=self.quantized_weight_t_buffer
                ),
            )

            # OPT-1 (opt-in, single-GPU): accumulate the expert wgrad straight into
            # main_grad in the grouped GEMM backward (beta=1 ACCUMULATE) instead of
            # GEMM-wgrad -> _WeightGradBridge.main_grad.add_(). Per-expert shims point
            # at the consolidated [E,N,K] main_grad slices; the grouped backward
            # accumulates into them and returns grad_b=None, then the bridge backward
            # just flags grad_added. ONLY safe with no gradient all-reduce /
            # reduce-scatter (TP=1 / DP=1 / EP=1), since grad_b=None skips the reduce.
            # Needs a turbo wheel carrying fused_grouped_wgrad (else falls back).
            _wgrad_ctx = contextlib.nullcontext()
            _fwg_dbg = (
                os.environ.get("PRIMUS_TURBO_FUSE_WGRAD_DEBUG") == "1"
                and getattr(type(self), "_fwg_logn", 0) < 10
            )
            _fwg_flag = os.environ.get("PRIMUS_TURBO_FUSE_GROUPED_WGRAD", "0") == "1"
            _mg = getattr(weights, "main_grad", None)
            if _fwg_flag and _mg is not None and _mg.dim() == 3:
                try:
                    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import (
                        _expert_main_grad_view,
                        fused_grouped_wgrad,
                    )

                    _shims = [_MainGradShim(_mg[i]) for i in range(_mg.shape[0])]
                    if _fwg_dbg:
                        import sys

                        _v = _expert_main_grad_view(_shims)
                        print(
                            f"[OPT-1] gate PASS shape={tuple(_mg.shape)} contig={_mg.is_contiguous()} "
                            f"stride={_mg.stride()} view={'OK' if _v is not None else 'REJECTED'}",
                            file=sys.stderr,
                            flush=True,
                        )
                        type(self)._fwg_logn = getattr(type(self), "_fwg_logn", 0) + 1
                    _wgrad_ctx = fused_grouped_wgrad(_shims)
                except ImportError as _e:
                    if _fwg_dbg:
                        import sys

                        print(f"[OPT-1] ImportError: {_e}", file=sys.stderr, flush=True)
                        type(self)._fwg_logn = getattr(type(self), "_fwg_logn", 0) + 1
            elif _fwg_dbg:
                import sys

                print(
                    f"[OPT-1] gate FAIL flag={_fwg_flag} main_grad={'None' if _mg is None else f'dim={_mg.dim()}'}",
                    file=sys.stderr,
                    flush=True,
                )
                type(self)._fwg_logn = getattr(type(self), "_fwg_logn", 0) + 1

            with _wgrad_ctx:
                out = primus_turbo_torch.ops.grouped_gemm_fp8(
                    x,
                    quantized_weights,
                    m_splits,
                    trans_b=True,
                    config=quant_config.data(),
                )
        elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
            assert False, "FP4 is not supported in PrimusTurboGroupedLinear"
        else:
            out = primus_turbo_torch.ops.grouped_gemm(x, weights, m_splits, trans_b=True)

        return out, None


class PrimusTurboColumnParallelGroupedLinear(PrimusTurboGroupedLinear):
    """
    Wrapper for the PrimusTurboGroupedLinear layer but specialized
    to column-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
            name=name,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboColumnParallelGroupedLinear only supports expert tensor parallel size = 1"


class PrimusTurboRowParallelGroupedLinear(PrimusTurboGroupedLinear):
    """
    Wrapper for the PrimusTurboGroupedLinear layer but specialized
    to row-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
            name=name,
        )

        tp_size = get_pg_size(self._tp_group)
        assert tp_size == 1, "PrimusTurboRowParallelGroupedLinear only supports expert tensor parallel size = 1"
