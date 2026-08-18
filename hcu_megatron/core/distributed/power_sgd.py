# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# PowerSGD with external EFLayoutManager
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import torch
from .param_and_grad_buffer import _ParamAndGradBucket
from megatron.training.utils import print_rank_0


def orthogonalize_test(Q: torch.Tensor, eps: float = 1e-8):
    n, m = Q.shape
    for i in range(m):
        col = Q.narrow(1, i, 1)
        norm = torch.sqrt(torch.sum(col * col)) + eps
        if norm < 10 * eps:
            col.normal_()
            norm = torch.sqrt(torch.sum(col * col)) + eps
        col.div_(norm)
        if i + 1 < m:
            rest = Q.narrow(1, i + 1, m - i - 1)
            proj = torch.sum(col * rest, dim=0, keepdim=True)
            rest.sub_(col.matmul(proj))
    return Q


def orthogonalize_cholesky(A, eps=1e-8):
    # A: [n, r]
    G = A.T @ A
    G = G + eps * torch.eye(G.size(0), device=A.device, dtype=A.dtype)
    R = torch.linalg.cholesky(G)
    A[:] = A @ torch.linalg.inv(R)
    return A


def orthogonalize_linalg(matrix):
    q, r = torch.linalg.qr(matrix)
    matrix.copy_(q)
    return matrix


@torch.no_grad()
@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


# ------------------------ EFLayoutManager ------------------------
class EFLayoutManager:

    def __init__(self, ef_store_dtype: torch.dtype = torch.bfloat16):
        self.ef_store_dtype = ef_store_dtype
        self.ef_buffers: Dict[torch.dtype, torch.Tensor] = None
        self.ef_index: List[
            Tuple[torch.dtype, int, int, Tuple[int, ...]]] = []
        self.fingerprint_to_indices: Dict[Tuple[torch.dtype, int, Tuple[int, ...]], List[int]] = {}
        self.is_distributed_optimizer_mode: bool = False
        self.intra_distributed_optimizer_instance_size: int = 1
        self.intra_distributed_optimizer_instance_rank: int = 0

    def build_ef_layout(self, module_or_list, device: torch.device = torch.device("cuda")):
        self.is_distributed_optimizer_mode = False
        if self.ef_buffers is not None and len(self.ef_index) > 0:
            return self.ef_buffers, self.ef_index

        params = []

        def collect_params(m):
            if isinstance(m, torch.nn.Module):
                for p in m.parameters():
                    if p.requires_grad:
                        params.append(p)
            elif isinstance(m, (list, tuple)):
                for subm in m:
                    collect_params(subm)
            else:
                raise TypeError(f"Unsupported type: {type(m)}")

        collect_params(module_or_list)

        type_counts: Dict[torch.dtype, int] = {}
        for p in params:
            type_counts[p.dtype] = type_counts.get(p.dtype, 0) + p.numel()

        self.ef_buffers = {}
        for t, cnt in type_counts.items():
            self.ef_buffers[t] = torch.zeros(cnt, dtype=self.ef_store_dtype, device=device)

        self.ef_index = []
        self.fingerprint_to_indices = {}
        ef_cursors_local = {t: 0 for t in type_counts.keys()}

        for idx, p in enumerate(params):
            t = p.dtype
            n = p.numel()
            off = ef_cursors_local[t]
            shape = tuple(p.shape)
            ef_info = (t, off, n, shape)
            self.ef_index.append(ef_info)
            fp = (t, n, shape)
            self.fingerprint_to_indices.setdefault(fp, []).append(idx)
            ef_cursors_local[t] += n

        total_elems = sum(buf.numel() for buf in self.ef_buffers.values())
        total_mem = total_elems * torch.finfo(self.ef_store_dtype).bits // 8 / 1024 ** 2
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print_rank_0(f"[Rank {rank}] Params: {len(self.ef_index)}, Elements: {total_elems}, "
                     f"Memory: {total_mem:.2f} MB (dtype={self.ef_store_dtype})")

        return self.ef_buffers, self.ef_index

    def build_ef_layout_with_distributed_optimizer(self, module_or_list, device: torch.device = torch.device("cuda"),
                                                   intra_distributed_optimizer_instance_size: int = 1,
                                                   intra_distributed_optimizer_instance_rank: int = 0):
        self.is_distributed_optimizer_mode = True
        self.intra_distributed_optimizer_instance_size = intra_distributed_optimizer_instance_size
        self.intra_distributed_optimizer_instance_rank = intra_distributed_optimizer_instance_rank

        if self.ef_buffers is not None and len(self.ef_index) > 0:
            return self.ef_buffers, self.ef_index

        params = []

        def collect_params(m):
            if isinstance(m, torch.nn.Module):
                for p in m.parameters():
                    if p.requires_grad:
                        params.append(p)
            elif isinstance(m, (list, tuple)):
                for subm in m:
                    collect_params(subm)
            else:
                raise TypeError(f"Unsupported type: {type(m)}")

        collect_params(module_or_list)

        world_size = intra_distributed_optimizer_instance_size
        rank = intra_distributed_optimizer_instance_rank

        type_counts: Dict[torch.dtype, int] = {}
        for p in params:
            type_counts[p.dtype] = type_counts.get(p.dtype, 0) + p.numel()

        rank_element_ranges_per_dtype = {}
        for t, total_numel in type_counts.items():
            elements_per_rank_dtype = total_numel // world_size
            self.start_idx = rank * elements_per_rank_dtype
            self.end_idx = (rank + 1) * elements_per_rank_dtype
            if rank == world_size - 1:
                self.end_idx = total_numel
            rank_element_ranges_per_dtype[t] = (self.start_idx, self.end_idx)

        self.ef_buffers = {}
        for t, (start_idx, end_idx) in rank_element_ranges_per_dtype.items():
            local_size = end_idx - start_idx
            if local_size > 0:
                self.ef_buffers[t] = torch.zeros(local_size, dtype=self.ef_store_dtype, device=device)
            else:
                self.ef_buffers[t] = torch.empty(0, dtype=self.ef_store_dtype, device=device)

        self.ef_index = []
        self.local_ef_map = {}
        self.fingerprint_to_indices = {}

        global_cursors = {t: 0 for t in type_counts.keys()}
        local_cursors = {t: 0 for t in type_counts.keys()}

        for idx, p in enumerate(params):
            t = p.dtype
            n = p.numel()
            shape = tuple(p.shape)

            self.ef_index.append((t, global_cursors[t], n, shape))

            start_range, end_range = rank_element_ranges_per_dtype[t]
            param_global_start_offset = global_cursors[t]
            param_global_end_offset = global_cursors[t] + n

            intersection_start = max(param_global_start_offset, start_range)
            intersection_end = min(param_global_end_offset, end_range)
            intersection_size = intersection_end - intersection_start

            if intersection_size > 0:
                local_offset_in_ef_buffer = local_cursors[t]
                self.local_ef_map[idx] = (local_offset_in_ef_buffer, intersection_size)
                local_cursors[t] += intersection_size

            global_cursors[t] += n

            fp = (t, n, shape)
            self.fingerprint_to_indices.setdefault(fp, []).append(idx)

        total_elems = sum(buf.numel() for buf in self.ef_buffers.values())
        total_mem = total_elems * torch.finfo(self.ef_store_dtype).bits // 8 / 1024 ** 2
        print_rank_0(f"[Rank {torch.distributed.get_rank()}] Params: {len(self.local_ef_map)} (local shards), Elements: {total_elems} (local), "
                     f"Memory: {total_mem:.2f} MB (dtype={self.ef_store_dtype})")

        return self.ef_buffers, self.ef_index


class Compressor(ABC):
    """
    Abstract base class for gradient compressors.
    Defines the interface for compressing and decompressing gradients within buckets.
    """

    @abstractmethod
    def compress_bucket(self, bucket: _ParamAndGradBucket) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Compress the gradients in the given bucket.

        Args:
            bucket: The _ParamAndGradBucket to compress.

        Returns:
            A tuple (compressed_data_1, compressed_data_2, metadata).
            `metadata` contains information needed for decompression (e.g., shapes, offsets).
            The two tensors are the main data to be communicated (e.g., P and Q in PowerSGD).
        """
        pass

    @abstractmethod
    def decompress_bucket(self, bucket: _ParamAndGradBucket, compressed_data_1: torch.Tensor,
                          compressed_data_2: torch.Tensor, metadata: Any) -> None:
        """
        Decompress the received data and update the bucket's grad_data.

        Args:
            bucket: The _ParamAndGradBucket to update.
            compressed_data_1: First part of received compressed data.
            compressed_data_2: Second part of received compressed data.
            metadata: Metadata returned by compress_bucket.
        """
        pass


# ------------------------ PowerSGDCompressor ------------------------
class PowerSGDCompressor:

    def __init__(self,
                 ef_layout_manager: EFLayoutManager,
                 rank: int = 1,
                 compression_dtype: torch.dtype = torch.float32,
                 reuse_query: bool = True,
                 verbose: bool = False):
        self.rank = rank
        self.compression_dtype = compression_dtype
        self.reuse_query = reuse_query
        self.verbose = verbose

        # external layout manager (must call build_ef_layout before training)
        self.ef_manager = ef_layout_manager

        # cache: param object -> index (fast path)
        self.param_to_index: Dict[torch.nn.Parameter, int] = {}

        # Q cache (per param object)
        self._q_cache: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self._q_cache_step: Any = None
        self._init_printer()

    def _init_printer(self):
        print_rank_0("===== PowerSGD Compressor=====")
        print_rank_0(f" >> rank: {self.rank}")
        print_rank_0(f" >> compression_dtype: {self.compression_dtype}")
        print_rank_0(f" >> ef_store_dtype: {self.ef_manager.ef_store_dtype}")
        print_rank_0(" >> use_error_feedback: True")
        print_rank_0("============================")

    def begin_iteration(self, step: int):
        if self._q_cache_step != step:
            self._q_cache.clear()
            self._q_cache_step = step

    # ---------------- helper: index lookup (fast path + fingerprint fallback) ----------------
    def _get_param_index(self, param: torch.nn.Parameter) -> int:
        # fast path
        if param in self.param_to_index:
            return self.param_to_index[param]

        # fingerprint fallback using ef_manager.fingerprint_to_indices
        fp = (param.dtype, param.numel(), tuple(param.shape))
        candidates = self.ef_manager.fingerprint_to_indices.get(fp)
        if not candidates:
            raise KeyError(f"EF layout has no slot for param fingerprint {fp}. Did you call build_ef_layout correctly?")

        used = set(self.param_to_index.values())
        for idx in candidates:
            if idx not in used:
                self.param_to_index[param] = idx
                if self.verbose:
                    print_rank_0(f"[PowerSGD] fingerprint-mapped param -> index {idx}, fp={fp}")
                return idx

        raise KeyError(f"All candidate indices for {fp} are already assigned. Layout mismatch.")

    def _ef_view(self, param: torch.nn.Parameter) -> torch.Tensor:
        idx = self._get_param_index(param)
        t, off, n, shape = self.ef_manager.ef_index[idx]
        buf = self.ef_manager.ef_buffers[t]
        return buf[off: off + n]  # 1D view

    # ---------------- PowerSGD core ----------------
    def _compress_2d_tensor(self, grad_2d: torch.Tensor, param=None) -> Tuple[torch.Tensor, torch.Tensor]:
        m, n = grad_2d.shape
        if m == 0 or n == 0:
            return grad_2d, torch.empty(0, dtype=grad_2d.dtype, device=grad_2d.device)

        r = min(m, n, self.rank)
        if r <= 0:
            return grad_2d, torch.empty(0, dtype=grad_2d.dtype, device=grad_2d.device)

        if self.reuse_query and (param in self._q_cache):
            Q = self._q_cache[param]
            if Q.shape != (n, r) or Q.dtype != grad_2d.dtype or Q.device != grad_2d.device:
                Q = torch.empty(n, r, device=grad_2d.device, dtype=grad_2d.dtype)
                Q.normal_()
                # orthogonalize_cholesky(Q)
                orthogonalize(Q)
                self._q_cache[param] = Q
        else:
            Q = torch.empty(n, r, device=grad_2d.device, dtype=grad_2d.dtype)
            Q.normal_()
            # orthogonalize_cholesky(Q)
            orthogonalize(Q)
            if self.reuse_query and (param is not None):
                self._q_cache[param] = Q

        Qc = Q if Q.dtype == self.compression_dtype else Q.to(self.compression_dtype)
        Gc = grad_2d if grad_2d.dtype == self.compression_dtype else grad_2d.to(self.compression_dtype)

        P = torch.matmul(Gc, Qc)
        return P, Qc

    def _decompress_2d_tensor(self, P: torch.Tensor, Q: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        recon = torch.matmul(P, Q.transpose(-2, -1))
        if recon.dtype != out_dtype:
            recon = recon.to(out_dtype)
        return recon

    # ---------------- compress_bucket ----------------
    def compress_bucket(self, bucket: _ParamAndGradBucket) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        device = bucket.grad_data.device

        # ensure layout exists
        if not (self.ef_manager.ef_buffers and len(self.ef_manager.ef_index) > 0):
            raise RuntimeError(
                "EF layout not initialized. Call ef_layout_manager.build_ef_layout(model) before training.")

        # first pass: metadata and sizes
        p_total = 0
        q_total = 0
        meta_P_shapes: List[Tuple[int, int]] = []
        meta_Q_shapes: List[Tuple[int, int]] = []
        meta_param_offsets: List[Tuple[int, Tuple[int, ...]]] = []

        if self.ef_manager.is_distributed_optimizer_mode:
            # Calculate total gradient elements in the current bucket
            bucket_grad_numel = 0
            for (param, offset, shape) in bucket.components:
                if param.requires_grad:
                    bucket_grad_numel += param.numel()

            # Get the start and end global indices for the current rank's range
            start_global_element_idx = self.ef_manager.start_idx
            end_global_element_idx = self.ef_manager.end_idx

        for (param, offset, shape) in bucket.components:
            if not param.requires_grad:
                m = shape[0] if len(shape) >= 1 else 0
                n = shape[1] if len(shape) >= 2 else 1
                meta_P_shapes.append((m, n))
                meta_Q_shapes.append((0, 0))
                p_total += m * n
                meta_param_offsets.append((offset, shape))
                continue

            if len(shape) == 1:
                m = shape[0]
                n = 1
            else:
                m = shape[0]
                n = shape[1] if len(shape) >= 2 else 1

            r = min(m, n, self.rank)
            if r > 0 and m > 0 and n > 0:
                meta_P_shapes.append((m, r))
                meta_Q_shapes.append((n, r))
                p_total += m * r
                q_total += n * r
            else:
                meta_P_shapes.append((m, n))
                meta_Q_shapes.append((0, 0))
                p_total += m * n
            meta_param_offsets.append((offset, shape))

        # Allocate buffers for P and Q
        for_P = torch.empty(p_total, dtype=self.compression_dtype, device=device) if p_total > 0 else torch.empty(0,
                                                                                                                  dtype=self.compression_dtype,
                                                                                                                  device=device)
        for_Q = torch.empty(q_total, dtype=self.compression_dtype, device=device) if q_total > 0 else torch.empty(0,
                                                                                                                  dtype=self.compression_dtype,
                                                                                                                  device=device)

        p_off = 0
        q_off = 0

        # second pass: actual computation
        for (param, offset, shape), P_shape, Q_shape in zip(bucket.components, meta_P_shapes, meta_Q_shapes):
            if self.ef_manager.is_distributed_optimizer_mode:
                param_numel = param.numel() if param.requires_grad else 0

                # Check if the parameter overlaps with the current rank's responsibility
                global_param_idx = self._get_param_index(param)
                param_global_start_offset = 0
                for i in range(global_param_idx):
                    param_global_start_offset += self.ef_manager.ef_index[i][2]  # ef_index[i][2] is numel
                param_global_end_offset = param_global_start_offset + param_numel

                # Check if there is any overlap between the parameter's range and the current rank's range
                overlap_start = max(param_global_start_offset, start_global_element_idx)
                overlap_end = min(param_global_end_offset, end_global_element_idx)

                should_process_ef = False
                if overlap_start < overlap_end:  # There is an overlap, meaning this param is partially or fully in the current rank's range
                    should_process_ef = True
            else:
                should_process_ef = True  # In non-distributed mode, process all parameters
        for (param, offset, shape), P_shape, Q_shape in zip(bucket.components, meta_P_shapes, meta_Q_shapes):
            if not param.requires_grad:
                m = P_shape[0]
                n = P_shape[1]
                if m * n > 0:
                    chunk = bucket.grad_data[offset: offset + m * n]
                    for_P[p_off: p_off + m * n].copy_(chunk.to(self.compression_dtype))
                    p_off += m * n
                continue

            # Normalize to 2D
            if len(shape) == 1:
                m = shape[0]
                n = 1
                grad_flat = bucket.grad_data[offset: offset + m * n]
                grad_2d = grad_flat.view(m, n)
            else:
                m = shape[0]
                n = shape[1] if len(shape) >= 2 else 1
                grad_flat = bucket.grad_data[offset: offset + m * n]
                grad_2d = grad_flat.view(m, n)

            # EF flat view (1D)
            ef_flat = self._ef_view(param)
            # Update grad_2d with the error feedback, but ensure we only add the part of ef_flat that fits
            if self.ef_manager.is_distributed_optimizer_mode:
                if should_process_ef:
                    # If EF slot is smaller than needed, only add the part of the error that fits
                    num_elements_to_add = min(ef_flat.numel(), m * n)  # Only add the elements that fit
                    grad_2d_flat = grad_2d.view(-1)
                    grad_2d_flat[:num_elements_to_add].add_(ef_flat.view(-1)[:num_elements_to_add].to(grad_2d.dtype))
            else:
                grad_2d.add_(ef_flat.view(grad_2d.shape))

            # non-compressed path
            if Q_shape[0] == 0 and Q_shape[1] == 0:
                numel = m * n
                for_P[p_off: p_off + numel].copy_(grad_2d.reshape(-1).to(self.compression_dtype))
                p_off += numel
                # zero ef slot
                ef_flat.zero_()
                continue
            # Always compress and decompress, regardless of the rank's responsibility
            P, Q = self._compress_2d_tensor(grad_2d, param=param)
            grad_recon = self._decompress_2d_tensor(P, Q, grad_2d.dtype)

            # new EF = grad - recon (m x n), write back as flat
            ef_new = grad_2d - grad_recon  # dtype = grad_2d.dtype
            if self.ef_manager.is_distributed_optimizer_mode:
                if should_process_ef:
                    if (ef_flat.numel() < m * n):
                        # If EF slot is smaller than needed, store only part of the data
                        num_elements_to_store = ef_flat.numel()  # EF slot size is less than required
                        if ef_flat.dtype != grad_2d.dtype:
                            ef_flat[:num_elements_to_store].copy_(
                                grad_2d.reshape(-1)[:num_elements_to_store].to(ef_flat.dtype))
                        else:
                            ef_flat[:num_elements_to_store].copy_(grad_2d.reshape(-1)[:num_elements_to_store])
                    else:
                        # Only update EF for parameters in the current rank's range
                        if ef_flat.dtype != ef_new.dtype:
                            ef_flat.copy_(ef_new.reshape(-1).to(ef_flat.dtype))
                        else:
                            ef_flat.copy_(ef_new.reshape(-1))
            else:
                # Non-distributed mode: always update EF
                if ef_flat.dtype != ef_new.dtype:
                    ef_flat.copy_(ef_new.reshape(-1).to(ef_flat.dtype))
                else:
                    ef_flat.copy_(ef_new.reshape(-1))

            # Write P/Q to flat buffers
            mP, rP = P_shape
            nQ, rQ = Q_shape
            nP = mP * rP
            nQ_elems = nQ * rQ
            for_P[p_off: p_off + nP].copy_(P.view(-1))
            for_Q[q_off: q_off + nQ_elems].copy_(Q.view(-1))
            p_off += nP
            q_off += nQ_elems

            del grad_recon, P, Q

        metadata = {
            'P_shapes': [torch.Size(s) for s in meta_P_shapes],
            'Q_shapes': [torch.Size(s) for s in meta_Q_shapes],
            'param_offsets': meta_param_offsets,
            'original_dtype': bucket.grad_data.dtype,
        }
        return for_P, for_Q, metadata

    # ---------------- decompress_bucket ----------------
    def decompress_bucket(self, bucket: _ParamAndGradBucket, compressed_data_1: torch.Tensor,
                          compressed_data_2: torch.Tensor, metadata: Any) -> None:
        for_P = compressed_data_1
        for_Q = compressed_data_2
        P_shapes = metadata['P_shapes']
        Q_shapes = metadata['Q_shapes']
        param_offsets = metadata['param_offsets']
        out_dtype = metadata['original_dtype']

        idx_P = 0
        idx_Q = 0

        for (offset, shape), P_shape, Q_shape in zip(param_offsets, P_shapes, Q_shapes):
            if len(shape) == 1:
                m = shape[0]
                n = 1
            else:
                m = shape[0] if len(shape) >= 1 else 0
                n = shape[1] if len(shape) >= 2 else 1

            if Q_shape.numel() == 0 or Q_shape[1] == 0:
                numel_P = m * n
                if numel_P == 0:
                    continue
                chunk = for_P[idx_P: idx_P + numel_P].to(out_dtype)
                bucket.grad_data[offset: offset + numel_P].copy_(chunk)
                idx_P += numel_P
            else:
                mP, rP = P_shape
                nQ, rQ = Q_shape
                numel_P = int(mP * rP)
                numel_Q = int(nQ * rQ)
                P = for_P[idx_P: idx_P + numel_P].view(mP, rP)
                Q = for_Q[idx_Q: idx_Q + numel_Q].view(nQ, rQ)
                idx_P += numel_P
                idx_Q += numel_Q

                grad_recon_2d = self._decompress_2d_tensor(P, Q, out_dtype)
                bucket.grad_data[offset: offset + m * n].copy_(grad_recon_2d.view(-1))
