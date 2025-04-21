import torch

from typing import List, Optional, Union
from importlib.metadata import version
from packaging.version import Version as PkgVersion


_flux_version = None


def get_flux_version():
    """Get flux version from __version__; if not available use pip's. Use caching."""

    def get_flux_version_str():
        import flux

        if hasattr(flux, '__version__'):
            return str(flux.__version__)
        else:
            return version("flux")

    global _flux_version
    if _flux_version is None:
        _flux_version = PkgVersion(get_flux_version_str())
    return _flux_version


def is_flux_min_version(version, check_equality=True):
    """Check if minimum version of `flux` is installed."""
    if check_equality:
        return get_flux_version() >= PkgVersion(version)
    return get_flux_version() > PkgVersion(version)


def tensor_slide(
        tensor: Optional[torch.Tensor],
        num_slice: int,
        dims: Union[int, List[int]] = -1,
        step: int = 1,
        return_first=False,
) -> List[Union[torch.Tensor, None]]:
    """通用滑动窗口函数，支持任意维度"""
    if tensor is None:
        # return `List[None]` to avoid NoneType Error
        return [None] * (num_slice + 1)

    if num_slice == 0:
        return [tensor]

    window_size = tensor.shape[-1] - num_slice
    dims = [dims] if isinstance(dims, int) else sorted(dims, reverse=True)

    # 连续多维度滑动
    slices = []
    for i in range(0, tensor.size(dims[-1]) - window_size + 1, step):
        slice_obj = [slice(None)] * tensor.dim()
        for dim in dims:
            slice_obj[dim] = slice(i, i + window_size)
        slices.append(tensor[tuple(slice_obj)])
        if return_first:
            return slices
    return slices
