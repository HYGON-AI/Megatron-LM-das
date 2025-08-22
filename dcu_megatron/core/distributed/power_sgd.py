from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch

from megatron.training.utils import print_rank_0

from .param_and_grad_buffer import _ParamAndGradBucket


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


class PowerSGDCompressor(Compressor):
    """
    Implements PowerSGD compression with error feedback for _ParamAndGradBucket.
    """

    def __init__(self, rank: int = 1, compression_dtype: torch.dtype = torch.float32):
        self.rank = rank
        self.compression_dtype = compression_dtype
        # Error feedback memory: param -> accumulated error tensor
        self.memory_out: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self.reuse_query = True
        self._cached_modified_Q = {}
        self._init_printer()

    def _init_printer(self):
        print_rank_0('===== PowerSGD Compressor =====')
        print_rank_0(f' >> compression_dtype: {self.compression_dtype}')
        print_rank_0(' >> use_error_feedback: True')
        print_rank_0('============================')

    def _compress_2d_tensor(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a single 2D gradient tensor using PowerSGD."""
        m, n = grad.shape
        if m < self.rank or n < self.rank:
            return grad, torch.empty(0, dtype=grad.dtype, device=grad.device)
        assert m >= self.rank and n >= self.rank, f"Rank {self.rank} too large for tensor {grad.shape}"
        key = id(grad)
        should_reuse = self.reuse_query and (key in self._cached_modified_Q)
        if should_reuse:
            Q = self._cached_modified_Q[key]
        else:
            # Generate random orthogonal matrix Q (n x rank)
            Q = torch.randn(n, self.rank, device=grad.device, dtype=grad.dtype)

        orthogonalize(Q)
        if self.reuse_query:
            self._cached_modified_Q[key] = Q.clone()
        # Compute P = M @ Q (m x rank)
        P = torch.matmul(grad.to(self.compression_dtype), Q)
        return P, Q

    def _decompress_2d_tensor(self, P: torch.Tensor, Q: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct the 2D gradient tensor from P and Q."""
        return torch.matmul(P, Q.transpose(-2, -1)).to(original_dtype)

    def compress_bucket(self, bucket: _ParamAndGradBucket) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Compress the gradients in the bucket.
        Returns P and Q tensors (flattened and concatenated) and metadata.
        Updates the error feedback memory.
        """
        from datetime import datetime
        # --- Step 1: Prepare ---
        # Get the components list (assumed to be available on the bucket)
        components = bucket.components  # List of {param, offset_in_bucket, shape, numel}
        bucket_P_list = []
        bucket_Q_list = []
        # Metadata to store for decompression
        metadata = {
            'P_shapes': [],
            'Q_shapes': [],
            'param_offsets': [],  # (offset_in_bucket, original_shape)
            'original_device': bucket.grad_data.device,
            'original_dtype': bucket.grad_data.dtype
        }
        # --- Step 2: Process each component ---
        for comp in components:
            param = comp[0]
            offset_in_bucket = comp[1]
            shape = comp[2]
            numel = param.numel()

            # Skip non-2D parameters for simplicity
            if len(shape) != 2:
                continue

            # Extract and reshape gradient
            grad_flat = bucket.grad_data[offset_in_bucket: offset_in_bucket + numel]
            grad_2d = grad_flat.view(shape)

            # Apply error feedback
            if param in self.memory_out:
                grad_2d = grad_2d + self.memory_out[param]

            # Apply PowerSGD compression

            P, Q = self._compress_2d_tensor(grad_2d)
            if Q.numel() > 0:
                # Calculate reconstruction error
                grad_recon = self._decompress_2d_tensor(P, Q, grad_2d.dtype)
                error = grad_2d - grad_recon

                # Update error feedback memory
                self.memory_out[param] = error

                # Store compressed data and metadata
                bucket_P_list.append(P.flatten())
                bucket_Q_list.append(Q.flatten())
                metadata['P_shapes'].append(P.shape)
                metadata['Q_shapes'].append(Q.shape)
            else:
                # Store compressed data and metadata
                bucket_P_list.append(grad_2d.flatten())
                bucket_Q_list.append(torch.tensor([], dtype=self.compression_dtype, device=grad_2d.device))
                metadata['P_shapes'].append(grad_2d.shape)
                metadata['Q_shapes'].append(torch.Size([0, 0]))
            metadata['param_offsets'].append((offset_in_bucket, shape))

        # --- Step 3: Concatenate and return ---
        # Handle empty case
        if not bucket_P_list:
            # Return empty tensors if no 2D params were compressed
            for_P = torch.tensor([], dtype=self.compression_dtype, device=bucket.grad_data.device)
            for_Q = torch.tensor([], dtype=self.compression_dtype, device=bucket.grad_data.device)
        else:
            for_P = torch.cat(bucket_P_list)
            for_Q = torch.cat(bucket_Q_list)
        return for_P, for_Q, metadata

    def decompress_bucket(self, bucket: _ParamAndGradBucket, compressed_data_1: torch.Tensor,
                          compressed_data_2: torch.Tensor, metadata: Any) -> None:
        """
        Decompress the received P and Q data and update bucket.grad_data.
        """
        # --- Step 1: Extract data ---
        for_P = compressed_data_1
        for_Q = compressed_data_2
        P_shapes = metadata['P_shapes']
        Q_shapes = metadata['Q_shapes']
        param_offsets = metadata['param_offsets']
        original_dtype = metadata['original_dtype']

        # --- Step 2: Reconstruct each parameter's gradient ---
        idx_P = 0
        idx_Q = 0
        for (offset_in_bucket, shape), P_shape, Q_shape in zip(param_offsets, P_shapes, Q_shapes):
            # Extract P
            numel_P = P_shape[0] * P_shape[1]
            P_flat = for_P[idx_P: idx_P + numel_P]
            P = P_flat.view(P_shape)
            idx_P += numel_P

            # Extract Q
            if Q_shape[0] == 0 or Q_shape[1] == 0:
                grad_recon_2d = P.view(shape)
            else:
                numel_Q = Q_shape[0] * Q_shape[1]
                Q_flat = for_Q[idx_Q: idx_Q + numel_Q]
                Q = Q_flat.view(Q_shape)
                idx_Q += numel_Q

                # Reconstruct gradient
                grad_recon_2d = self._decompress_2d_tensor(P, Q, original_dtype)
            grad_recon_flat = grad_recon_2d.view(-1)

            # Write back to bucket.grad_data
            bucket.grad_data[offset_in_bucket: offset_in_bucket + grad_recon_flat.numel()].copy_(grad_recon_flat)

        # Note: Non-2D parameters remain unchanged in bucket.grad_data


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col