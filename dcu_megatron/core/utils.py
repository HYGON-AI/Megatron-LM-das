import torch

from typing import List, Optional, Union


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
