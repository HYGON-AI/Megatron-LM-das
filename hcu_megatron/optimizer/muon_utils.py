# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid Newton-Schulz adapter for Emerging-Optimizers and Transformer Engine.

Execution policy:

* Local / non-TP path: use Emerging-Optimizers directly.
* TP duplicated path: use Emerging-Optimizers directly.
* TP distributed path: use Transformer Engine / hipSolverMp.

The local paths intentionally do not create any private ProcessGroupNCCL.  This
keeps the same process-group creation behavior as Emerging-Optimizers and avoids
changing which NCCL process group becomes the first process group in a process.
"""

import atexit
from typing import Literal

import torch
import torch.distributed as dist

from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
    newton_schulz as eo_newton_schulz,
    newton_schulz_tp as eo_newton_schulz_tp,
)
from transformer_engine.pytorch.newton_schulz import (
    CusolverMpCtx,
    get_coefficients,
    newton_schulz as te_newton_schulz,
)


_EPS = 1e-15

# TE contexts are needed only for the TP-distributed path.  Cache one context
# per caller-provided ProcessGroup so the cuSolverMp grid is not recreated on
# every optimizer step.
_CUSOLVER_MP_CONTEXTS: dict[dist.ProcessGroup, CusolverMpCtx] = {}


def _require_distributed() -> None:
    """Require an initialized torch.distributed environment for TE distributed NS."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Transformer Engine distributed Newton-Schulz requires "
            "torch.distributed to be initialized"
        )


def _get_cusolver_mp_context(group: dist.ProcessGroup) -> CusolverMpCtx:
    """Return a cached TE cuSolverMp context for ``group``."""

    _require_distributed()

    context = _CUSOLVER_MP_CONTEXTS.get(group)
    if context is None:
        # ProcessGroupNCCL/RCCL creates its communicator lazily.  Materialize it
        # before TE asks the process group for the raw communicator pointer.
        dist.barrier(
            group=group,
            device_ids=[torch.cuda.current_device()],
        )
        context = CusolverMpCtx(group)
        _CUSOLVER_MP_CONTEXTS[group] = context

    return context


def _destroy_cached_resources() -> None:
    """Best-effort cleanup of cached TE contexts at process exit."""

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    for context in list(_CUSOLVER_MP_CONTEXTS.values()):
        try:
            context.destroy()
        except Exception:
            pass

    _CUSOLVER_MP_CONTEXTS.clear()


atexit.register(_destroy_cached_resources)


def _distributed_normalize_p2(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    eps: float = _EPS,
) -> torch.Tensor:
    """Frobenius-normalize a tensor whose elements are sharded over ``group``."""

    squared_norm = (x * x).sum()
    dist.all_reduce(
        squared_norm,
        op=dist.ReduceOp.SUM,
        group=group,
    )
    return x / torch.sqrt(squared_norm).clamp_min(eps)


def _run_te_distributed_newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run TE's in-place Newton-Schulz on a distributed matrix shard."""

    coefficients = get_coefficients(
        steps,
        coefficient_type,  # type: ignore[arg-type]
    )

    iteration_dtype = (
        torch.bfloat16
        if torch.get_float32_matmul_precision() == "medium"
        else torch.float32
    )

    # TE mutates its input in-place.  Use private contiguous storage so the
    # adapter preserves EO's functional input/output behavior.
    output = x.to(dtype=iteration_dtype).contiguous()

    te_newton_schulz(
        output,
        _get_cusolver_mp_context(group),
        num_iterations=steps,
        coefficients=coefficients,
    )

    return output.float()


def newton_schulz_tp(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group: dist.ProcessGroup | None,
    partition_dim: int | None = None,
    tp_mode: Literal["duplicated", "distributed"] = "duplicated",
) -> torch.Tensor:
    """Apply Newton-Schulz with EO local semantics and TE distributed execution.

    ``partition_dim is None``
        Delegate directly to Emerging-Optimizers.  This is the non-TP / local
        path and does not require torch.distributed to be initialized.

    ``tp_mode == "duplicated"``
        Delegate directly to Emerging-Optimizers.  EO gathers the full matrix,
        runs its local Newton-Schulz implementation independently on every TP
        rank, and returns the local shard.

    ``tp_mode == "distributed"``
        Keep the matrix sharded and run Transformer Engine's distributed
        Newton-Schulz on the caller-provided TP process group.  TE distributes
        matrix columns, so row-sharded inputs are transposed before the
        operation and transposed back afterwards.
    """

    if x.ndim != 2:
        raise ValueError(f"Input tensor x must be 2D, got {x.ndim}D")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")

    # ------------------------------------------------------------------
    # EO paths: local/non-TP and TP duplicated.
    # ------------------------------------------------------------------
    # Delegating the whole path to EO is intentional.  Besides matching EO's
    # numerical implementation exactly, it guarantees that local NS does not
    # create a private ProcessGroupNCCL before Megatron initializes WORLD.
    if partition_dim is None:
        return eo_newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,  # type: ignore[arg-type]
        )

    if tp_mode == "duplicated":
        if tp_group is None:
            raise ValueError("tp_group must be provided when partition_dim is set")
        return eo_newton_schulz_tp(
            x,
            steps=steps,
            coefficient_type=coefficient_type,  # type: ignore[arg-type]
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode="duplicated",
        )

    # ------------------------------------------------------------------
    # TE path: TP distributed only.
    # ------------------------------------------------------------------
    if tp_mode != "distributed":
        raise ValueError(f"Invalid tp_mode: {tp_mode}")

    if partition_dim not in (0, 1):
        raise ValueError(f"Invalid partition_dim: {partition_dim}")
    if tp_group is None:
        raise ValueError("tp_group must be provided when tp_mode='distributed'")

    _require_distributed()

    # TE's distributed Newton-Schulz expects matrix columns to be sharded.
    # EO/Megatron partition_dim=0 means rows are sharded, so transpose first.
    transpose = partition_dim == 0

    # Work on private storage because TE mutates the matrix in-place.
    local_x = x.detach().clone().contiguous()
    if transpose:
        local_x = local_x.mT.contiguous()

    normalized = _distributed_normalize_p2(
        local_x,
        tp_group,
    )

    output = _run_te_distributed_newton_schulz(
        normalized,
        steps,
        coefficient_type,
        tp_group,
    )

    return output.mT.contiguous() if transpose else output
