# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare the TE TP Newton-Schulz adapter with EO on the same matrix shards.

Run with::

    PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH \
        torchrun --standalone --nproc-per-node=2 -m pytest -xvs \
        tests/unit_tests/optimizer/test_newton_schulz_tp_te.py
"""

import os

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("emerging_optimizers")
pytest.importorskip("transformer_engine.pytorch.newton_schulz")

from hcu_megatron.optimizer.muon_utils import (  # noqa: E402
    _destroy_cached_resources,
    eo_newton_schulz_tp,
    newton_schulz_tp as te_newton_schulz_tp,
)


@pytest.fixture(scope="module")
def tp_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torchrun --nproc-per-node=2")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm GPUs")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl", init_method="env://")
    try:
        yield dist.group.WORLD
    finally:
        # TE caches a cuSolverMp context by process group. Release it before
        # the process group is destroyed, including when an assertion fails.
        _destroy_cached_resources()
        if owns_process_group:
            dist.destroy_process_group()


@pytest.mark.parametrize("shape", [(128, 64), (64, 128)])
@pytest.mark.parametrize("partition_dim", [0, 1])
@pytest.mark.parametrize("matmul_precision", ["highest", "medium"])
def test_te_distributed_matches_eo_modes(
    tp_group, shape, partition_dim, matmul_precision
):
    """Compare TE with EO's gathered and distributed implementations."""
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(matmul_precision)
    try:
        # Each rank constructs the same full input, then keeps its own shard.
        generator = torch.Generator().manual_seed(2026)
        full = torch.randn(shape, generator=generator, dtype=torch.float32).cuda()
        rank = dist.get_rank(group=tp_group)
        local = full.chunk(2, dim=partition_dim)[rank].contiguous()
        original = local.clone()
        kwargs = dict(
            steps=5,
            coefficient_type="quintic",
            tp_group=tp_group,
            partition_dim=partition_dim,
        )

        eo_duplicated = eo_newton_schulz_tp(local, tp_mode="duplicated", **kwargs)
        eo_distributed = eo_newton_schulz_tp(local, tp_mode="distributed", **kwargs)
        actual = te_newton_schulz_tp(local, tp_mode="distributed", **kwargs)

        assert torch.equal(local, original), "TE must not modify the input shard"
        assert torch.isfinite(actual).all()
        # The gathered EO result checks full-matrix semantics; the distributed
        # EO result checks the implementation TE replaces in Megatron.
        # Allow for distributed rounding and BF16 iteration at medium.
        rtol, atol = (5e-3, 5e-4) if matmul_precision == "highest" else (3e-2, 3e-3)
        torch.testing.assert_close(
            actual, eo_duplicated, rtol=rtol, atol=atol,
            msg="TE distributed differs from EO duplicated",
        )
        torch.testing.assert_close(
            actual, eo_distributed, rtol=rtol, atol=atol,
            msg="TE distributed differs from EO distributed",
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
