# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native index kernels of the dedup dispatcher (csrc/dedup_gather_sum.cu) vs. PyTorch ops.

Single GPU, from the repository root:
    PYTHONPATH=$PWD:$PWD/3rdparty/Megatron-LM python tests/unit_tests/expert_placement/test_dedup_gather_sum.py
"""
import sys

import torch

from hcu_megatron.core.transformer.moe import dedup_token_dispatcher as dd

FAIL = []


def ref_gather_sum(src, table, base, mod, out_dtype):
    M, K = table.shape[0] * table.shape[1], table.shape[2]
    v = table.reshape(M, K) - base
    valid = v >= 0
    v = v.clamp(min=0)
    if mod > 0:
        v = v % mod
    acc = torch.float64 if src.dtype == torch.float64 else torch.float32
    rows = src.index_select(0, v.reshape(-1)).view(M, K, -1).to(acc)
    return (rows * valid.unsqueeze(-1)).sum(1).to(out_dtype)


def check(name, got, want, atol):
    ok = got.dtype == want.dtype and got.shape == want.shape and torch.allclose(got.float(), want.float(), atol=atol, rtol=0)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max diff {(got.float() - want.float()).abs().max().item():.2e}")
    if not ok:
        FAIL.append(name)


def main():
    lib = dd._native()
    assert lib is not None, "native kernels did not build (see the warning above)"
    torch.manual_seed(0)
    dev = "cuda"
    for H in (2048, 2064, 7, 1):
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            # gather with modulo from a column slice (dispatch: packed hidden rows from hidden[idx % T])
            T, P = 300, 4
            src = torch.randn(T, H, dtype=dt, device=dev)
            idx = torch.randint(0, T * P, (517,), device=dev)
            buf = torch.empty(517, H + 16, dtype=dt, device=dev)
            out = buf[:, :H]
            lib.gather_sum(src, idx.view(-1, 1, 1), 0, T, out)
            check(f"gather%T H={H} {dt}", out, ref_gather_sum(src, idx.view(-1, 1, 1), 0, T, dt), 0)
            # sum through a transposed 1-based position map (combine: out[t] = sum_p rows[pos[p, t] - 1])
            pos = torch.randint(0, 200, (P, T), device=dev) * (torch.rand(P, T, device=dev) < 0.4)
            rows = torch.randn(200, H, dtype=dt, device=dev)
            table = pos.t().unsqueeze(1)
            for odt in {dt, torch.float32}:
                out = torch.empty(T, H, dtype=odt, device=dev)
                lib.gather_sum(rows, table, 1, 0, out)
                check(f"sum   H={H} {dt}->{odt}", out, ref_gather_sum(rows, table, 1, 0, odt),
                      1e-6 if odt == torch.float32 else 1e-2)
            # [M1, M2, 1] table (backward of the probs: rows t*P + p of a [T*P, L] output)
            L = 8
            meta = torch.randn(200, L, dtype=torch.float32, device=dev)
            table = pos.unsqueeze(-1).transpose(0, 1)
            out = torch.empty(T * P, L, dtype=torch.float32, device=dev)
            lib.gather_sum(meta, table, 1, 0, out)
            check(f"probs-grad {dt}", out, ref_gather_sum(meta, table, 1, 0, torch.float32), 0)
    # gather_elem: 2-D strided src addressed as src[v % R, v // R], and 1-D src into a strided out
    R, L = 333, 8
    meta = torch.randn(R, 2 * L, device=dev)[:, :L]
    idx = torch.randint(0, R * L, (1000,), device=dev)
    out = torch.empty(1000, 1, device=dev)
    lib.gather_elem(meta, R, idx.view(-1, 1), 0, out)
    check("gather_elem 2-D", out.view(-1), meta[idx % R, idx // R], 0)
    pos = torch.randint(0, 1001, (L, R), device=dev) * (torch.rand(L, R, device=dev) < 0.3)
    src = torch.randn(1000, device=dev)
    out = torch.empty(R, 2 * L, device=dev)[:, L:]
    lib.gather_elem(src, 0, pos.t(), 1, out)
    v = pos.t() - 1
    check("gather_elem 1-D", out, torch.where(v >= 0, src[v.clamp(min=0)], torch.zeros(())), 0)
    print("dedup gather_sum checks:", "PASSED" if not FAIL else f"FAILED {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
